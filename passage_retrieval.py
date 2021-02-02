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
import transformers

import slurm
import glob

import util
import retriever.data
import retriever.model
import retriever.index

from torch.utils.data import DataLoader

from retriever.qa_validation import calculate_matches

logger = logging.getLogger(__name__)

def embed_questions(opt, data, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = retriever.data.Dataset(data)
    collator = retriever.data.Collator(tokenizer, opt.question_maxlength)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, _, _, _) = batch
            output = model.embed_text(
                text_ids=question_ids.to(opt.device), 
                text_mask=question_mask.to(opt.device), 
                apply_mask=model.apply_question_mask,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Shape of question embeddings: {embedding.size()}')

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
            #ids, embeddings = torch.load(fin)
            ids, embeddings = pickle.load(fin)

        index.index_data(ids, embeddings)
        counter += len(ids)
        if maxload is not None and counter >= maxload:
            break
    logger.info('Data indexing completed.')



def validate(passages, data, result_ctx_ids, workers_num, match_type='string'):
    match_stats = calculate_matches(passages, data, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
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


def save_results(data, passages, top_passages_and_scores, per_question_hits, out_file):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)
        d['ctxs'] =[
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ] 

    with open(out_file, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)


def main(opt):
    util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = retriever.data.load_data(opt.data_path)
    model_class = retriever.model.Retriever
    model, _, _, _, _, _ = util.load(model_class, opt.model_path, opt)

    model.cuda()
    model.eval()
    if not opt.no_fp16:
        model = model.half()

    index_buffer_sz = args.index_buffer

    index = retriever.index.DenseFlatIndexer(model.config.indexing_dimension)

    # index all passages
    input_paths = glob.glob(args.passages_embeddings_path)
    input_paths = sorted(input_paths)
    #logger.info('Reading all passages data from files: %s', input_paths)
    index_path = "_".join(input_paths[0].split("_")[:-1])
    print(args.save_or_load_index, os.path.exists(index_path), index_path)
    if args.save_or_load_index and os.path.exists(index_path+'.index.dpr'):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, maxload=args.maxload)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    # get questions & answers

    questions_embedding = embed_questions(opt, data, model, tokenizer)

    # get top k results
    start_time_retrieval = time.time()
    nbatch = (len(questions_embedding)-1) // args.index_batch_size + 1
    top_ids_and_scores = index.search_knn(questions_embedding, args.top_docs) 
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    all_passages = load_passages(args.passages_path, maxload=args.maxload)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')


    answers = [ex['answers'] for ex in data]
    questions_doc_hits = validate(all_passages, answers, top_ids_and_scores, args.validation_workers)
    save_results(data, all_passages, top_ids_and_scores, questions_doc_hits, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', required=True, type=str, default=None,
                        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--passages_path', required=True, type=str, default=None,
                        help="All passages file in the tsv format: id \\t passage_text \\t title")
    parser.add_argument('--passages_embeddings_path', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='output .tsv file path to write results to ')
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
    slurm.init_distributed_mode(args)
    main(args)