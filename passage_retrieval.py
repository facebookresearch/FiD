# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data
import src.index

from torch.utils.data import DataLoader

from src.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def embed_questions(opt, data, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = src.data.Dataset(data)
    collator = src.data.Collator(opt.question_maxlength, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, _, _, question_ids, question_mask) = batch
            output = model.embed_text(
                text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
                text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
                apply_mask=model.apply_question_mask,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu().numpy()


def index_encoded_data(index, vector_files, maxload=None):
    """
    Indexes encoded passages takes form a list of files
    :param vector_files: file names to get passages vectors from
    :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
    :return:
    """
    buffer = []
    counter = 0
    allids = []
    allembeddings = torch.tensor([])
    for i, file_path in enumerate(vector_files):
        logger.info(f'Loading file {file_path}')
        with open(file_path, 'rb') as fin:
            ids, embeddings = pickle.load(fin)

        index.index_data(ids, embeddings)
        counter += len(ids)
        if maxload is not None and counter >= maxload:
            break
    logger.info('Data indexing completed.')


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
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


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d['ctxs'] =[
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                } for c in range(ctxs_num)
            ] 

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex['ctxs']):
            d['hasanswer'] = hasanswer[i][k]


def main(opt):
    util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = src.data.load_data(opt.data)
    model_class = src.model.Retriever
    model, _, _, _, _, _ = util.load(model_class, opt.model_path, opt)

    model.cuda()
    model.eval()
    if not opt.no_fp16:
        model = model.half()

    index_buffer_sz = args.index_buffer

    index = src.index.DenseFlatIndexer(model.config.indexing_dimension)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    index_path = Path(input_paths[0]).parent / 'index.faiss'
    if args.save_or_load_index and index_path.exists():
        src.index.deserialize_from(index_path)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, maxload=args.maxload)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if args.save_or_load_index:
            src.index.serialize(index_path)

    questions_embedding = embed_questions(opt, data, model, tokenizer)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, args.top_docs) 
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    passages = load_passages(args.passages, maxload=args.maxload)

    add_passages(data, passages, top_ids_and_scores)
    hasanswer = validate(data, args.validation_workers)
    add_hasanswer(data, hasanswer)
    with open(args.output_path, 'w') as fout:
        json.dump(data, fout, indent=4)
    logger.info(f'Saved results to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None,
                        help="Data file containing questions and answers")
    parser.add_argument('--passages', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--passages_embeddings', type=str, default=None,
                        help='Glob path to encoded passages')
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--n-docs', type=int, default=100, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=32,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--per_gpu_batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_batch_size', type=int, default=128, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=50000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--n_centroids", type=str, default='flat', help='Number of centroids used for vector quantization with 8 bits per vector')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--maxload", type=int, default=None)
    parser.add_argument("--top_docs", type=int, default=100)
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--main_port", type=int, default=-1)
    parser.add_argument('--no_fp16', action='store_true')
    parser.add_argument('--passage_maxlength', type=int, default=200)
    parser.add_argument('--question_maxlength', type=int, default=40)

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)