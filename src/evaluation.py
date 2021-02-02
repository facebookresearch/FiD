#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import regex
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict
import numpy as np


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

def calculate_matches(all_docs: Dict[object, Tuple[str, str]],
                      answers: List[List[str]],
                      closest_docs: List[Tuple[List[object], List[float]]],
                      workers_num: int,
                      match_type: str) -> QAMatchStats:
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    global dpr_all_documents
    dpr_all_documents = all_docs

    logger.info('Matching answers in top docs...')

    tokenizer = SimpleTokenizer()
    get_score_partial = partial(check_answer, tokenizer=tokenizer)

    answers_docs = zip(answers, closest_docs)

    processes = ProcessPool(processes=workers_num)
    scores = processes.map(get_score_partial, answers_docs)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(answers_docs, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = answers_docs

    global dpr_all_documents
    hits = []

    for doc_id in doc_ids:
        text = dpr_all_documents[doc_id][0]

        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = ' '.join(tokenizer.tokenize(text, uncased=True))

    for answer in answers:
        answer = _normalize(answer)
        answer = ' '.join(tokenizer.tokenize(answer, uncased=True))
        if answer in text:
            return True
    return False

#################################################
########        READER EVALUATION        ########
#################################################

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

####################################################
########        RETRIEVER EVALUATION        ########
####################################################

def eval_batch(scores, inversions, avg_topk, idx_topk):
    for k, s in enumerate(scores):
        s = s.cpu().numpy()
        sorted_idx = np.argsort(-s)
        score(sorted_idx, inversions, avg_topk, idx_topk)

def count_inversions(arr):
    inv_count = 0
    lenarr = len(arr)
    for i in range(lenarr):
        for j in range(i + 1, lenarr):
            if (arr[i] > arr[j]):
                inv_count += 1
    return inv_count

def score(x, inversions, avg_topk, idx_topk):
    x = np.array(x)
    inversions.append(count_inversions(x))
    for k in avg_topk:
        # ratio of passages in the predicted top-k that are
        # also in the topk given by gold score
        avg_pred_topk = (x[:k]<k).mean()
        avg_topk[k].append(avg_pred_topk)
    for k in idx_topk:
        below_k = (x<k)
        # number of passages required to obtain all passages from gold top-k
        idx_gold_topk = len(x) - np.argmax(below_k[::-1])
        idx_topk[k].append(idx_gold_topk)
