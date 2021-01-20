import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig, BertConfig

class RetrieverConfig(BertConfig):

    def __init__(self, output_size=768, apply_mask_question=False, apply_mask_passage=False, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.apply_mask_question = apply_mask_question
        self.apply_mask_passage = apply_mask_passage


class Retriever(PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)

        self.config = config
        self.apply_mask_passage = self.config.apply_mask_passage
        self.apply_mask_question = self.config.apply_mask_question
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        self.proj = nn.Linear(self.model.config.hidden_size, self.config.output_size)
        self.norm = nn.LayerNorm(self.config.output_size)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(
        self,
        question_ids=None,
        question_mask=None,
        passage_ids=None,
        passage_mask=None,
        gold_score=None,
    ):
        assert question_ids is not None or passage_ids is not None

        if question_ids is not None:
            question_output = self.model(input_ids=question_ids, attention_mask=question_mask if self.apply_mask_question else None)[0]
            question_output = self.proj(question_output)
            question_output = self.norm(question_output)
            question_output = torch.mean(question_output, dim=1)

        if passage_ids is None:
            return question_output
        else:
            bsz, n_passages, plen = passage_ids.size()
            passage_ids = passage_ids.view(bsz * n_passages, plen)
            passage_mask = passage_mask.view(bsz * n_passages, plen)
            passage_output = self.model(input_ids=passage_ids, attention_mask=passage_mask if self.apply_mask_passage else None)[0]
            passage_output = passage_output.view(bsz, n_passages, plen, -1)

            passage_output = self.proj(passage_output)
            passage_output = self.norm(passage_output)

            #passage_output = passage_output[:, :, 0]
            passage_output = torch.mean(passage_output, dim=2)
        
        if question_ids is None:
            return passage_output

        score = torch.einsum('bd,bid->bi', question_output, passage_output)
        score = score / np.sqrt(question_output.size(-1))
        outputs = (score,)
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
            outputs = (loss,) + outputs

        return outputs 

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)