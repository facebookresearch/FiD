# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing (https://pytorch.org/docs/stable/checkpoint.html) in encoder.
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage (only used when cross-attention scores are saved to train a retriever)
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per passage.
        This scalar can be seen as a similarity score between the question and the input passage.
        It is obtained by averaging the cross-attention scores obtained on the first decoded token 
        over heads, layers, and tokens of the input passage.
        More details in Distilling Knowledge from Reader to Retriever: https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1) #batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        bsz, total_length = input_ids.shape #total length = n_passages * passage_length
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs 

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper applied to each block of the encoder to enable the use of checkpoint 
    (https://pytorch.org/docs/stable/checkpoint.html) 
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):

        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    model_output = module(*inputs, **kwargs)
                    output = ()
                    empty_tensor = torch.tensor(
                        [], 
                        device=model_output[0].device, 
                        dtype=torch.float, 
                        requires_grad=True
                    )
                    for x in model_output:
                        if x is None:
                            output = output + (empty_tensor,) #checkpoint doesn't work if there is tensor with requires_grad=False
                        else:
                            output = output + (x,)
                    return output
                return custom_forward

            out = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.module), 
                hidden_states, 
                attention_mask, 
                position_bias
            )
            outputs = ()
            for x in out:
                if len(x.size()) == 0:
                    outputs = outputs + (None,)
                else:
                    outputs = outputs + (x,)
        else:
            outputs = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return outputs


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
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
    This only works for computing cross attention over the input
    """
    assert(kv != None)
    assert(head_mask == None)
    assert(position_bias != None or self.has_relative_attention_bias)

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
       scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output

