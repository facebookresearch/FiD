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
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding.cpu().numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logger.info(f'Loading file {file_path}')
        with open(file_path, 'rb') as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
        
    logger.info('Data indexing completed.')

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits] 
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


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
    src.util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = src.data.load_data(opt.data)
    model_class = src.model.Retriever
    model = model_class.from_pretrained(opt.model_path)

    model.cuda()
    model.eval()
    if not opt.no_fp16:
        model = model.half()

    index = src.index.Indexer(model.config.indexing_dimension, opt.n_subquantizers, opt.n_bits)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / 'index.faiss'
    if args.save_or_load_index and index_path.exists():
        src.index.deserialize_from(embeddings_dir)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, opt.indexing_batch_size)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if args.save_or_load_index:
            src.index.serialize(embeddings_dir)

    questions_embedding = embed_questions(opt, data, model, tokenizer)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs) 
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    passages = src.util.load_passages(args.passages)
    passages = {x[0]:(x[1], x[2]) for x in passages}

    add_passages(data, passages, top_ids_and_scores)
    hasanswer = validate(data, args.validation_workers)
    add_hasanswer(data, hasanswer)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as fout:
        json.dump(data, fout, indent=4)
    logger.info(f'Saved results to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str, default=None, 
                        help=".json file containing question and answers, similar format to reader data")
    parser.add_argument('--passages', type=str, default=None, help='Path to passages (.tsv file)')
    parser.add_argument('--passages_embeddings', type=str, default=None, help='Glob path to encoded passages')
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--n-docs', type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument('--validation_workers', type=int, default=32,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--per_gpu_batch_size', type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--save_or_load_index", action='store_true', 
                        help='If enabled, save index and load index if it exists')
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--question_maxlength', type=int, default=40, help="Maximum number of tokens in a question")
    parser.add_argument('--indexing_batch_size', type=int, default=50000, help="Batch size of the number of passages indexed")
    parser.add_argument("--n-subquantizers", type=int, default=0, 
                        help='Number of subquantizer used for vector quantization, if 0 flat index is used')
    parser.add_argument("--n-bits", type=int, default=8, 
                        help='Number of bits per subquantizer')


    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)