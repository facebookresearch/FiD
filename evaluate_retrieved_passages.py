# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import csv
import json
import sys
import logging
import pickle
import time

import numpy as np
import torch

import util

from retriever.standout_validation import calculate_matches
#from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    #top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    top_k_hits = [v / len(data) for v in top_k_hits] 
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_file, maxload=None):
    docs = {}
    logger.info(f'Reading data from: {ctx_file}')
    with open(ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: doc_id, doc_text, title
        for k, row in enumerate(reader):
            if maxload is not None and k == maxload:
                return docs
            if row[0] == 'id':
                continue
            try:
                docs[row[0]] = (row[1], row[2])
            except:
                logger.warning(f'The following input line has not been correctly loaded: {row}')
    return docs


def main(opt):
    util.init_logger(is_main=True)
    with open(opt.data, 'r') as fin:
        data = json.load(fin)
    #all_passages = load_passages(args.passages_path, maxload=args.maxload)

    answers = [ex['answers'] for ex in data]
    questions_doc_hits = validate(data, args.validation_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None)
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")

    args = parser.parse_args()
    main(args)