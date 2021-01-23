# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def wrap_encoder(self, use_checkpoint=False):
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def set_checkpoint(self, use_checkpoint):
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint
            mod.module.use_checkpoint = use_checkpoint

class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        bsz, tc = input_ids.shape
        plen = tc // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, plen)
        attention_mask = attention_mask.view(bsz*self.n_passages, plen)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*plen, -1), ) + outputs[1:]
        return outputs 

class FilterWrapper(torch.nn.Module):
    def __init__(self, mod, use_checkpoint=False):
        super().__init__()
        self.mod = mod
        self.use_heckpoint=use_checkpoint
    
    def forward(self, *args, **kwargs):
        output = self.mod(*args, **kwargs)
        if self.use_checkpoint and self.training:
            none_idx = [i for i in range(len(output)) if output[i] is None]
            output = tuple(x for x in output if x is not None)
            output = output + (torch.tensor(none_idx, dtype=torch.float, requires_grad=True, device=output[0].device),)
        return output

class CheckpointWrapper(torch.nn.Module):
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):

        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, **kwargs)
                return custom_forward

            out = torch.utils.checkpoint.checkpoint(create_custom_forward(self.module), hidden_states, attention_mask, position_bias)
            none_idx = out[-1]
            outputs = ()
            counter = 0
            for k in range(len(out)+len(none_idx)-1):
                if k in none_idx:
                    outputs = outputs + (None,)
                else:
                    outputs = outputs + (out[counter],)
                    counter += 1
        else:
            outputs = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return outputs
    
def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(FilterWrapper(mod, use_checkpoint), use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def overwrite_forward_attention(model):
    for mod in model.decoder.block:
        attn = mod.layer[1].EncDecAttention # = T5AttentionScoreRegistration(mod.layer[1].EncDecAttention)
        attn.forward = types.MethodType(newforward, attn)

def get_crossattention_scores(model, context_mask):
    scores = []
    n_passages = context_mask.size(1)
    for mod in model.decoder.block:
        scores.append(mod.layer[1].EncDecAttention.score_storage)
    scores = torch.cat(scores, dim=2)
    scores = scores.view(scores.size(0), scores.size(1), scores.size(2), n_passages, -1) #batch_size, n_head, n_layers, n_passages, text_maxlength
    scores = scores.masked_fill(~context_mask[:, None, None], 0.)
    scores = scores.sum(dim=[1, 2, 4])
    ntokens = context_mask.sum(dim=[2])
    scores = scores/ntokens
    return scores

def reset_score_storage(model):
    for mod in model.decoder.block:
        mod.layer[1].EncDecAttention.score_storage = None

def newforward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    Self-attention (if kv is None) or attention over source sentence (provided by kv).
    """
    # Input is (bs, qlen, dim)
    # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
    # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
    bs, qlen, dim = input.size()

    if past_key_value_state is not None:
        assert self.is_decoder is True, "Encoder cannot cache past key value states"
        assert (
            len(past_key_value_state) == 2
        ), "past_key_value_state should have 2 past states: keys and values. Got {} past states".format(
            len(past_key_value_state)
        )
        real_qlen = qlen + past_key_value_state[0].shape[2] if query_length is None else query_length
    else:
        real_qlen = qlen

    if kv is None:
        klen = real_qlen
    else:
        klen = kv.size(1)

    def shape(x):
        """  projection """
        return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

    def unshape(x):
        """  compute context """
        return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

    q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

    if kv is None:
        k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
    elif past_key_value_state is None:
        k = v = kv
        k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
        v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

    if past_key_value_state is not None:
        if kv is None:
            k_, v_ = past_key_value_state
            k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
            v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
        else:
            k, v = past_key_value_state

    if self.is_decoder and use_cache is True:
        present_key_value_state = ((k, v),)
    else:
        present_key_value_state = (None,)

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

    if position_bias is None:
        if not self.has_relative_attention_bias:
            raise ValueError("No position_bias provided and no weights to compute position_bias")
        position_bias = self.compute_bias(real_qlen, klen)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value_state is not None:
            position_bias = position_bias[:, :, -1:, :]

        if mask is not None:
            position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

    scores += position_bias
    if self.score_storage is None:
        self.score_storage = scores
    weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
    weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

    # Mask heads if we want to
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
    context = unshape(context)  # (bs, qlen, dim)

    context = self.o(context)

    outputs = (context,) + present_key_value_state

    if output_attentions:
        outputs = outputs + (weights,)
    if self.has_relative_attention_bias:
        outputs = outputs + (position_bias,)
    return outputs
