# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json
import parser
from pathlib import Path
import numpy as np

def select_examples_TQA(data, index):
    selected_data = []
    for k in index:
        ex = data[k]
        q = ex['Question']
        answers = ex['Answer']['Aliases']
        target = ex['Answer']['Value']
        if target.isupper():
            target = target.title()
        selected_data.append(
            {
                'question': q,
                'answers': answers,
                'target': target,
            }
        )
    return selected_data

def select_examples_NQ(data, index):
    selected_data = [
        {
            'question': data[k]['question'],
            'answers': data[k]['answer'],
        }
        for k in index
    ]
    return selected_data

if __name__ == "__main__":
    dir_path = Path(sys.argv[1])
    save_dir = Path(sys.argv[2])

    #load NQ question idx
    with open(dir_path/'NQ.train.idx.json', 'r') as fin:
        NQ_trainidx = json.load(fin)
    with open(dir_path/'NQ.dev.idx.json', 'r') as fin:
        NQ_devidx = json.load(fin)
    with open(dir_path/'NQ.test.idx.json', 'r') as fin:
        NQ_testidx = json.load(fin)

    originaltrain, originaldev = [], []
    with open(dir_path/'NQ-open.dev.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)
    
    with open(dir_path/'NQ-open.train.jsonl') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    NQ_train = select_examples_NQ(originaltrain, NQ_trainidx)
    NQ_dev = select_examples_NQ(originaltrain, NQ_devidx)
    NQ_test = select_examples_NQ(originaldev, NQ_testidx)

    NQ_save_path = save_dir / 'NQ'
    NQ_save_path.mkdir(parents=True, exist_ok=True)

    with open(NQ_save_path/'train.json', 'w') as fout:
        json.dump(NQ_train, fout, indent=4)
    with open(NQ_save_path/'dev.json', 'w') as fout:
        json.dump(NQ_dev, fout, indent=4)
    with open(NQ_save_path/'test.json', 'w') as fout:
        json.dump(NQ_test, fout, indent=4)

    #load Trivia question idx
    with open(dir_path/'TQA.train.idx.json', 'r') as fin:
        TQA_trainidx = json.load(fin)
    with open(dir_path/'TQA.dev.idx.json', 'r') as fin:
        TQA_devidx = json.load(fin)
    with open(dir_path/'TQA.test.idx.json', 'r') as fin:
        TQA_testidx = json.load(fin)

    originaltrain, originaldev = [], []
    with open(dir_path/'triviaqa-unfiltered'/'unfiltered-web-train.json') as fin:
        originaltrain = json.load(fin)['Data']
    
    with open(dir_path/'triviaqa-unfiltered'/'unfiltered-web-dev.json') as fin:
        originaldev = json.load(fin)['Data']

    TQA_train = select_examples_TQA(originaltrain, TQA_trainidx)
    TQA_dev = select_examples_TQA(originaltrain, TQA_devidx)
    TQA_test = select_examples_TQA(originaltrain, TQA_testidx)
   
    TQA_save_path = save_dir / 'TQA'
    TQA_save_path.mkdir(parents=True, exist_ok=True)

    with open(TQA_save_path/'train.json', 'w') as fout:
        json.dump(TQA_train, fout, indent=4)
    with open(TQA_save_path/'dev.json', 'w') as fout:
        json.dump(TQA_dev, fout, indent=4)
    with open(TQA_save_path/'test.json', 'w') as fout:
        json.dump(TQA_test, fout, indent=4)
