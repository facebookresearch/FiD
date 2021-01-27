#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib

import argparse
import csv
import logging
import pickle

import numpy as np
import torch

import transformers
import retriever.data
import retriever.model
import util
import slurm

from torch.utils.data import DataLoader



logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

def embed_passages(opt, passages, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    collator = retriever.data.TextCollator(tokenizer, model.config.passage_maxlength)
    dataset = retriever.data.TextDataset(passages, title_prefix='title:', passage_prefix='context:')
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    total = 0
    allids, allembeddings = [], []
    with torch.no_grad():
        for k, (ids, text_ids, text_mask) in enumerate(dataloader):
            embeddings = model.embed_text(
                text_ids=text_ids.to(opt.device), 
                text_mask=text_mask.to(opt.device), 
                apply_mask=model.apply_passage_mask
            )
            embeddings = embeddings.cpu()
            total += len(ids)

            allids.append(ids)
            allembeddings.append(embeddings)
            if k % 10 == 0:
                logger.info('Encoded passages %d', total)

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    allids = [x for idlist in allids for x in idlist]
    return allids, allembeddings


def main(opt):
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    model_class = retriever.model.Retriever
    model, _, _, _, _, _ = util.load(model_class, opt.model_path, opt)
    
    model.eval()
    model = model.to(opt.device)
    if not opt.no_fp16:
        model = model.half()
    if opt.world_size > 1 and opt.local_rank == -1:
        model = torch.nn.DataParallel(model)

    logger.info(f'Passage file: {args.passages_path}')
    passages = []
    with open(args.passages_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for k, row in enumerate(reader):
            if not row[0] == 'id':
                try:
                    passages.append((row[0], row[1], row[2]))
                except:
                    logger.warning(f'The following input line has not been correctly loaded: {row}')
            if k == 1000:
                break

    shard_size = int(len(passages) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size
    if args.shard_id == args.num_shards-1:
        end_idx = len(passages)

    passages = passages[start_idx:end_idx]
    logger.info(f'Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}')

    allids, allembeddings = embed_passages(opt, passages, model, tokenizer)

    file = args.output_path + '_' + str('{0:02d}'.format(args.shard_id))
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving {len(allids)} passage embeddings to {file}')
    with open(file, mode='wb') as f:
        pickle.dump((allids, allembeddings), f)

    logger.info(f'Total passages processed {len(allids)}. Written to {file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--passages_path', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--output_path', required=True, type=str, default=None,
                        help='output .tsv file path to write results to ')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    parser.add_argument('--passage_maxlength', type=int, default=200)
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--main_port", type=int, default=-1)
    parser.add_argument('--no_fp16', action='store_true')
    args = parser.parse_args()

    slurm.init_distributed_mode(args)

    main(args)
