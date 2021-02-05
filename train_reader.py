# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import slurm
import logging
import util
import numpy as np
from options import Options
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from reader.model import EncoderWrapper
import reader.evaluation
import reader.data
import reader.model
from pathlib import Path


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em):

    if opt.is_main:
        tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)

    train_sampler = DistributedSampler(train_dataset) if opt.is_distributed \
        else RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, drop_last=True, num_workers=10, collate_fn=collator)

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        if opt.is_distributed:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1
            (idx, labels, context_ids, context_mask) = batch
            n_passages = context_ids.size(1)
            if hasattr(model, "module"):
                model.module.encoder.n_passages = n_passages
            else:
                model.encoder.n_passages = n_passages

            train_loss = model.forward(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            train_loss = util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    tb_logger.add_scalar("Evaluation", dev_em, step)
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        util.save(model, optimizer, scheduler, step, best_dev_em, opt, dir_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0

            if opt.is_main and step % opt.save_freq == 0:
                util.save(model, optimizer, scheduler, step, best_dev_em, opt, dir_path, f"step-{step}")
            if step > opt.total_steps:
                break


def evaluate(model, dataset, tokenizer, collator, opt):

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, 
        sampler=sampler, 
        batch_size=opt.per_gpu_batch_size,
        drop_last=False, 
        num_workers=10, 
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, context_ids, context_mask) = batch
            model.encoder.n_passages = context_ids.size(1)

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = reader.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    logger = util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)


    model_name = 't5-' + opt.model_size
    model_class = reader.model.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = reader.data.Collator(opt.text_maxlength, tokenizer)

    train_examples = reader.data.load_data(opt.train_data, maxload=opt.maxload)
    train_dataset = reader.data.Dataset(
        train_examples, 
        opt.n_context, 
        tokenizer, 
        text_maxlength=opt.text_maxlength
    )
    eval_examples = reader.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size, #use the global rank and world size attibutes to split the eval set on multiple gpus
        maxload=opt.maxload
    )     
    eval_dataset = reader.data.Dataset(
        eval_examples, 
        opt.n_context, 
        tokenizer, 
        text_maxlength=opt.text_maxlength
    )

    step = 0
    best_dev_em = 0.

    if not directory_exists and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = reader.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = util.set_optim(opt, model)
    elif opt.model_path == "none":
        load_path = dir_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = util.load(
            model_class, load_path, opt, reset_params=False
        )
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = util.load(
            model_class, opt.model_path, opt, reset_params=True,
        )
        logger.info(f"Model loaded from {opt.model_path}") 

    model.set_checkpoint(opt.use_checkpoint)


    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model, 
        optimizer, 
        scheduler, 
        step,
        train_dataset, 
        eval_dataset, 
        opt, 
        collator, 
        best_dev_em
    )
