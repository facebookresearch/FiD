import torch
import random
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, n_context, tokenizer, text_maxlength=250, no_title=False, training=False, retriever_mode=False, 
        question_prefix='question:', title_prefix='title:', passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.no_title = no_title
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.training = training
        self.retriever_mode = retriever_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example['question']
        if 'target' in example:
            target = example['target']
        elif 'answers' in example:
            target = random.choice(example['answers'])
        else:
            target = None

        contexts = example['ctxs'][:self.n_context]

        passages = []
        if len(contexts) == 0:
            to_concatenate = [self.question_prefix, question] 
            text = ' '.join(to_concatenate)
            passages.append(text)
        for i in range(len(contexts)):
            text = contexts[i]['text']
            to_concatenate = [self.question_prefix, question]
            if not self.no_title:
                title = contexts[i]['title']
                to_concatenate.append(self.title_prefix)
                to_concatenate.append(title)
            to_concatenate.append(self.passage_prefix)
            to_concatenate.append(text)
            text = ' '.join(to_concatenate)
            passages.append(text)

        return {'index':index, 'question':question, 'target':target, 'passages':passages}

    def get_example(self, index):
        return self.data[index]


class Collator(object):
    def __init__(self, text_maxlength, tokenizer):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        question = [ex['question'] for ex in batch]
        target = [ex['target'] for ex in batch] 
        if target[0] is not None:
            target = self.tokenizer.batch_encode_plus(target, pad_to_max_length=True, return_tensors="pt")
            target_ids, target_mask = target["input_ids"], target["attention_mask"]
        else:
            target_ids, target_mask = None, None

        batch_text_passages = [ex['passages'] for ex in batch]
        batch_passage_ids, batch_passage_masks = self.encode_passages(batch_text_passages, self.tokenizer)

        return (index, target_ids, target_mask.bool(), batch_passage_ids, batch_passage_masks.bool())

    def encode_passages(self, batch_text_passages, tokenizer):
        batch_encoded_passages = []
        max_context_length = 0
        batch_passage_ids, batch_passage_masks = [], []
        for k, text_passages in enumerate(batch_text_passages):
            p = tokenizer.batch_encode_plus(text_passages, max_length=self.text_maxlength, pad_to_max_length=True, return_tensors='pt', truncation=True)
            p_ids = p['input_ids']
            p_masks = p['attention_mask']
            batch_passage_ids.append(p_ids[None])
            batch_passage_masks.append(p_masks[None])

        batch_passage_ids = torch.cat(batch_passage_ids, dim=0)
        batch_passage_masks = torch.cat(batch_passage_masks, dim=0)
        return batch_passage_ids, batch_passage_masks


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])
        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(question, pad_to_max_length=True, return_tensors="pt", max_length=40, truncation=True)
        question_ids, question_mask = question['input_ids'], question['attention_mask']
        question_mask = question_mask.bool()
        scores = [torch.tensor(ex['scores'])[None] for ex in batch]
        scores = torch.cat(scores, dim=0)
        batch_text_passages = [ex['passages'] for ex in batch]
        batch_passage_ids, batch_passage_masks = self.encode_passages(batch_text_passages, self.tokenizer)
        batch_concatenation_ids, batch_concatenation_masks = None, None

        return (index, question_ids, question_mask, batch_passage_ids, batch_passage_masks, scores)
    
    def encode_passages(self, batch_text_passages, tokenizer):
        batch_encoded_passages = []
        batch_passage_ids, batch_passage_masks = [], []
        max_context_length = 0
        for k, text_passages in enumerate(batch_text_passages):
            p = tokenizer.batch_encode_plus(text_passages, max_length=self.max_passage_length, pad_to_max_length=True, truncation=True, return_tensors='pt')
            p_ids = p['input_ids']
            p_masks = p['attention_mask']
            batch_passage_ids.append(p_ids[None])
            batch_passage_masks.append(p_masks[None])
        batch_passage_ids = torch.cat(batch_passage_ids, dim=0)
        batch_passage_masks = torch.cat(batch_passage_masks, dim=0).bool()
        return batch_passage_ids, batch_passage_masks


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
