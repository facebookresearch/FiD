import torch
import random
import json
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, n_context=None, no_title=False, question_prefix='question:', title_prefix='title:', passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.no_title = no_title
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']
        question = self.question_prefix + question
        if 'ctxs' in example and self.n_context is not None:
            contexts = example['ctxs'][:self.n_context]
            passages, scores = [], []
            for i in range(len(contexts)):
                text = contexts[i]['text']
                to_concatenate = []
                if not self.no_title:
                    title = contexts[i]['title']
                    to_concatenate.append(self.title_prefix)
                    to_concatenate.append(title)
                to_concatenate.append(self.passage_prefix)
                to_concatenate.append(text)
                text = ' '.join(to_concatenate)
                passages.append(text)
                scores.append(contexts[i]['score'])
            scores = torch.tensor(scores)
        else:
            passages, scores = None, None
        return {'index':index, 'question':question, 'passages':passages, 'scores':scores}
 
    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            scores = [ctx['score'] for ctx in ex['ctxs']]
            idx = np.argsort(-np.array(scores))
            ctxs = [ex['ctxs'][k] for k in idx]
            ex['ctxs'] = ctxs


    def get_example(self, index):
        return self.data[index]


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(question, pad_to_max_length=True, return_tensors="pt", max_length=self.question_maxlength, truncation=True)
        question_ids, question_mask = question['input_ids'], question['attention_mask']
        question_mask = question_mask.bool()
        scores = [ex['scores'] for ex in batch]
        if scores[0] is None:
            scores = None
        else:
            scores = torch.stack(scores, dim=0)
        batch_text_passages = [ex['passages'] for ex in batch]
        if not batch_text_passages[0] is None:
            batch_passage_ids, batch_passage_masks = self.encode_passages(batch_text_passages, self.tokenizer)
        else:
            batch_passage_ids, batch_passage_masks = None, None

        return (index, question_ids, question_mask, batch_passage_ids, batch_passage_masks, scores)
    
    def encode_passages(self, batch_text_passages, tokenizer):
        batch_encoded_passages = []
        batch_passage_ids, batch_passage_masks = [], []
        max_context_length = 0
        for k, text_passages in enumerate(batch_text_passages):
            p = tokenizer.batch_encode_plus(text_passages, max_length=self.passage_maxlength, pad_to_max_length=True, truncation=True, return_tensors='pt')
            p_ids = p['input_ids']
            p_masks = p['attention_mask']
            batch_passage_ids.append(p_ids[None])
            batch_passage_masks.append(p_masks[None])
        batch_passage_ids = torch.cat(batch_passage_ids, dim=0)
        batch_passage_masks = torch.cat(batch_passage_masks, dim=0).bool()
        return batch_passage_ids, batch_passage_masks


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, no_title=False, title_prefix='title:', passage_prefix='context:'):
        self.data = data
        self.no_title = no_title
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        to_concatenate = []
        if not self.no_title:
            to_concatenate.append(self.title_prefix)
            to_concatenate.append(example[2])
        to_concatenate.append(self.passage_prefix)
        to_concatenate.append(example[1])
        text = ' '.join(to_concatenate)
        return example[0], text
 
class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        pids = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus([x[1] for x in batch], pad_to_max_length=True, return_tensors="pt", max_length=self.maxlength, truncation=True)
        text_ids, text_mask = encoded_batch['input_ids'], encoded_batch['attention_mask']
        
        return pids, text_ids, text_mask

def load_data(data_path=None, global_rank=-1, world_size=-1, maxload=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = [] 
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if maxload > -1 and k >= maxload:
            break
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()
    return examples