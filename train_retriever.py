# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
from pathlib import Path
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options


def train(model, optimizer, scheduler, global_step,
                    train_dataset, dev_dataset, opt, collator, best_eval_loss):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')
    train_sampler = DistributedSampler(train_dataset) if opt.is_distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, 
        drop_last=True, 
        num_workers=10, 
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while global_step < opt.total_steps:
        if opt.is_distributed > 1:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            global_step += 1
            (idx, question_ids, question_mask, passage_ids, passage_mask, gold_score) = batch
            _, _, _, train_loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=passage_ids.cuda(),
                passage_mask=passage_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            train_loss.backward()

            if global_step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if global_step % opt.eval_freq == 0:
                eval_loss, inversions, avg_topk, idx_topk = evaluate(model, dev_dataset, collator, opt)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if opt.is_main:
                        src.util.save(model, optimizer, scheduler, global_step, best_eval_loss, opt, dir_path, 'best_dev')
                model.train()
                if opt.is_main:
                    log = f"{global_step} / {opt.total_steps}"
                    log += f" -- train: {curr_loss/opt.eval_freq:.6f}"
                    log += f", eval: {eval_loss:.6f}"
                    log += f", inv: {inversions:.1f}"
                    log += f", lr: {scheduler.get_last_lr()[0]:.6f}"
                    for k in avg_topk:
                        log += f" | avg top{k}: {100*avg_topk[k]:.1f}"
                    for k in idx_topk:    
                        log += f" | idx top{k}: {idx_topk[k]:.1f}"
                    logger.info(log)

                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", eval_loss, global_step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), global_step)
                    curr_loss = 0

            if opt.is_main and global_step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, global_step, best_eval_loss, opt, dir_path, f"step-{global_step}")
            if global_step > opt.total_steps:
                break


def evaluate(model, dataset, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=opt.per_gpu_batch_size,
        drop_last=False, 
        num_workers=10, 
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    eval_loss = []

    avg_topk = {k:[] for k in [1, 2, 5] if k <= opt.n_context}
    idx_topk = {k:[] for k in [1, 2, 5] if k <= opt.n_context}
    inversions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, context_ids, context_mask, gold_score) = batch

            _, _, scores, loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=context_ids.cuda(),
                passage_mask=context_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            src.evaluation.eval_batch(scores, inversions, avg_topk, idx_topk)
            total += question_ids.size(0)

    inversions = src.util.weighted_average(np.mean(inversions), total, opt)[0]
    for k in avg_topk:
        avg_topk[k] = src.util.weighted_average(np.mean(avg_topk[k]), total, opt)[0]
        idx_topk[k] = src.util.weighted_average(np.mean(idx_topk[k]), total, opt)[0]

    return loss, inversions, avg_topk, idx_topk

if __name__ == "__main__":
    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    #Load data
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    collator_function = src.data.RetrieverCollator(
        tokenizer, 
        passage_maxlength=opt.passage_maxlength, 
        question_maxlength=opt.question_maxlength
    )
    train_examples = src.data.load_data(opt.train_data)
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size, 
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    global_step = 0
    best_eval_loss = np.inf
    config = src.model.RetrieverConfig(
        indexing_dimension=opt.indexing_dimension,
        apply_question_mask=not opt.no_question_mask,
        apply_passage_mask=not opt.no_passage_mask,
        extract_cls=opt.extract_cls,
        projection=not opt.no_projection,
    )
    model_class = src.model.Retriever
    if not directory_exists and opt.model_path == "none":
        model = model_class(config, initialize_wBERT=True)
        src.util.set_dropout(model, opt.dropout)
        model = model.to(opt.device)
        optimizer, scheduler = src.util.set_optim(opt, model)
    elif opt.model_path == "none":
        load_path = dir_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, global_step, best_eval_loss = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {dir_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, global_step, best_eval_loss = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.proj = torch.nn.Linear(768, 256)
    model.norm = torch.nn.LayerNorm(256)
    model.config.indexing_dimension = 256
    model = model.to(opt.device)
    optimizer, scheduler = src.util.set_optim(opt, model)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[opt.local_rank], 
            output_device=opt.local_rank, 
            find_unused_parameters=True,
        )
    
    train(
        model, 
        optimizer,
        scheduler,
        global_step,
        train_dataset,
        eval_dataset,
        opt,
        collator_function,
        best_eval_loss
    )