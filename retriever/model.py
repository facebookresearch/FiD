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
        self.apply_passage_mask = self.config.apply_passage_mask
        self.apply_question_mask = self.config.apply_question_mask
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained('bert-base-uncased')
        else:
            self.model = transformers.BertModel(config)
        if not self.config.projection:
            self.proj = nn.Linear(self.model.config.hidden_size, self.config.indexing_dimension)
            self.norm = nn.LayerNorm(self.config.output_size)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self, question_ids, question_mask, passage_ids, passage_mask, gold_score=None):
        question_output = self.embed_text(text_ids=question_ids, text_mask=question_mask, apply_mask=self.apply_question_mask)
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(text_ids=passage_ids, text_mask=passage_mask, apply_mask=self.apply_passage_mask)

        score = torch.einsum('bd,bid->bi', question_output, passage_output.view(bsz, n_passages, -1))
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False):
        text_output = self.model(input_ids=text_ids, attention_mask=passage_mask if apply_mask else None)
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if not self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)
        text_output = torch.mean(text_output, dim=-2)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)