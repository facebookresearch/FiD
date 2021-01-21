import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig, BertConfig

class RetrieverConfig(BertConfig):

    def __init__(self, indexing_dimension=768, apply_mask_question=False, apply_mask_passage=False, passage_maxlength=200, question_maxlength=40, projection=True, **kwargs):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_mask_question = apply_mask_question
        self.apply_mask_passage = apply_mask_passage
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection


class Retriever(PreTrainedModel):

    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert config.projection or config.indexing_dimension == 768, 'If no projection then indexing dimension must be equal to 768'
        self.config = config
        self.apply_mask_passage = self.config.apply_mask_passage
        self.apply_mask_question = self.config.apply_mask_question
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if not self.config.projection:
            self.proj = nn.Linear(self.model.config.hidden_size, self.config.indexing_dimension)
            self.norm = nn.LayerNorm(self.config.output_size)
        self.loss_fct = torch.nn.KLDivLoss()

    def score(self, question_ids, question_mask, passage_ids, passage_mask, gold_score=None):
        question_output = self.embed_questions(question_ids, question_mask)
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_passages(passage_ids, passage_mask)
        passage_output = passage_output.view(bsz, n_passages, -1)

        score = torch.einsum('bd,bid->bi', question_output, passage_output)
        score = score / np.sqrt(question_output.size(-1))
        outputs = (score,)
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
            outputs = (loss,) + outputs

        return outputs 

    def embed_passages(self, passage_ids, passage_mask):
        passage_output = self.model(input_ids=passage_ids, attention_mask=passage_mask if self.apply_mask_passage else None)
        if not passage_output is tuple:
            passage_output.to_tuple()
        passage_output = passage_output[0]
        if not self.config.projection:
            passage_output = self.proj(passage_output)
            passage_output = self.norm(passage_output)
        passage_output = torch.mean(passage_output, dim=-2)
        return passage_output

    def embed_questions(self, question_ids, question_mask):
        question_output = self.model(input_ids=question_ids, attention_mask=question_mask if self.apply_mask_question else None)
        if not question_output is tuple:
            question_output.to_tuple()
        question_output = question_output[0]
        if not self.config.projection:
            question_output = self.proj(question_output)
            question_output = self.norm(question_output)
        question_output = torch.mean(question_output, dim=-2)
        return question_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)