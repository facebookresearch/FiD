# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging

import numpy as np
import torch

import src.util

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def main(opt):
    logger = src.util.init_logger(is_main=True)
    with open(opt.data, 'r') as fin:
        data = json.load(fin)
    answers = [ex['answers'] for ex in data]
    questions_doc_hits = validate(data, args.validation_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None)
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")

    args = parser.parse_args()
    main(args)
