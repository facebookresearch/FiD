import os
import time
import sys
import torch
import transformers
import slurm
import logging
import util
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from options import Options
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from fidt5 import FiDT5
from fidbart import BartForConditionalGeneration
import evaluation
import data

logger = logging.getLogger(__name__)


def train_evaluate(model, optimizer, scheduler, global_step,
                    train_dataset, dev_dataset, opt, collator_function, best_dev_em):

    if opt.is_master:
        tb_logger = SummaryWriter(os.path.join(opt.checkpoint_dir, opt.name))

    train_sampler = (RandomSampler(train_dataset) if opt.local_rank == -1 or opt.world_size == 1
        else DistributedSampler(train_dataset))
    dev_sampler = SequentialSampler(dev_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, drop_last=True, num_workers=20, collate_fn=collator_function)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=opt.per_gpu_batch_size,
        drop_last=True, num_workers=20, collate_fn=collator_function)

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while global_step < opt.total_step:
        if opt.world_size > 1:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            global_step += 1
            (idx, answer_ids, answer_mask, context_ids, context_mask) = batch
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            labels = answer_ids.masked_fill(~answer_mask, -100)
            if hasattr(model, "module"):
                if opt.model_type == 'bart':
                    model.module.model.encoder.n_passages = context_ids.size(1)
                elif opt.model_type == 't5':
                    model.module.encoder.n_passages = context_ids.size(1)
            else:
                if opt.model_type == 'bart':
                    model.model.encoder.n_passages = context_ids.size(1)
                elif opt.model_type == 't5':
                    model.encoder.n_passages = context_ids.size(1)
            context_ids, context_mask = context_ids.cuda(), context_mask.cuda()
            context_ids = context_ids.view(context_ids.size(0), -1)
            context_mask = context_mask.view(context_mask.size(0), -1)
            if 'bart' in opt.model_type:
                decoder_input_ids = answer_ids[:, :-1]
                decoder_attention_mask = answer_mask[:, :-1]
                labels = labels[:, 1:]
            else:
                decoder_input_ids = None


            model.zero_grad()
            outputs = model(
                input_ids=context_ids,
                attention_mask=context_mask,
                decoder_attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            train_loss = outputs[0]

            train_loss.backward()
            util.clip_gradients(model, opt.clip)
            optimizer.step()

            scheduler.step()

            train_loss = util.average_master(train_loss, opt)
            curr_loss += train_loss.item()

            if global_step % opt.eval_freq == 0:
                dev_em = evaluate(model, dev_dataset, dev_dataloader, tokenizer, opt)
                if opt.is_master:
                    tb_logger.add_scalar("Evaluation", dev_em, global_step)
                if dev_em > best_dev_em:
                    best_dev_em = dev_em
                    if opt.is_master:
                        model_to_save = model.module if hasattr(model, "module") else model
                        util.save(model_to_save, optimizer, scheduler, global_step, best_dev_em, opt, dir_path, 'best_dev')
                model.train()
            if opt.is_master and global_step % opt.eval_freq == 0:
                logger.info(
                    "%d / %d -- train = %.3f | evaluation = %.3f | lr = %.6f"
                    % (global_step, opt.total_step, curr_loss / (opt.eval_freq), dev_em, scheduler.get_last_lr()[0])
                )
                tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), global_step)
                curr_loss = 0

            if opt.is_master and global_step % (50*opt.eval_freq) == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                util.save(model_to_save, optimizer, scheduler, global_step, best_dev_em, opt, dir_path, "step-%s" % global_step)
            if global_step > opt.total_step:
                break


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    ems = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, answer_ids, answer_mask, context_ids, context_mask) = batch
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            answer_ids.masked_fill_(~answer_mask, -100)
            context_ids = context_ids.cuda()
            context_mask = context_mask.cuda()
            if opt.model_type == 'bart':
                model.model.encoder.n_passages = context_ids.size(1)
            elif opt.model_type == 't5':
                model.encoder.n_passages = context_ids.size(1)
            context_ids = context_ids.view(context_ids.size(0), -1)
            context_mask = context_mask.view(context_mask.size(0), -1)

            outputs = model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                max_length=50,
                decoder_start_token_id= tokenizer.bos_token_id,
                decoder_end_token_id=tokenizer.eos_token_id,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k]).answers
                #print(o, ans, gold)
                ems_score = evaluation.ems(ans, gold)
                total+=1
                ems.append(ems_score)
            if opt.is_master and (i + 1) % opt.eval_print_freq == 0:
                logger.info("%d / %d -- average = %.3f" % (i + 1, len(dataloader), np.mean(ems)))

    score, total = util.weighted_average(np.mean(ems), total, opt)
    return score


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    logger.info("Distributed training")

    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)

    assert opt.model_type == 'bart' or opt.model_type == 't5', 'Expected model type bart or t5'
    if 'bart' in opt.model_type:
        model_name = 'facebook/bart-' + opt.model_size
        model_class = BartForConditionalGeneration
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
    elif 't5' in opt.model_type:
        model_name = 't5-' + opt.model_size
        model_class = FiDT5
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    collator_function = data.Collator(opt, tokenizer)

    train_examples = data.load_data(opt.train_data_path)
    train_dataset = data.Dataset(train_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)
    dev_examples = data.load_data(opt.dev_data_path, global_rank=opt.global_rank, world_size=opt.world_size) #use the global rank and world size attibutes to split the dev set on multiple gpus
    dev_dataset = data.Dataset(dev_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title)

    directory_exists = os.path.exists(dir_path)
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    os.makedirs(dir_path, exist_ok=True)
    if not directory_exists and opt.is_master:
        options.print_options(opt)
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    file_handler = logging.FileHandler(filename=os.path.join(dir_path, "run.log"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if opt.is_master else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()


    global_step = 0
    best_dev_em = 0.

    if not directory_exists and opt.model_path == "none":
        model = model_class.from_pretrained(model_name) 
        model = model.to(opt.local_rank)
        optimizer, scheduler = util.set_optim(opt, model)
    elif opt.model_path == "none":
        model, optimizer, scheduler, opt_checkpoint, global_step, best_dev_em = util.load(
            model_class, dir_path, opt, reset_params=False, name="latest",
        )
        logger.info("Model loaded from %s" % dir_path)
    else:
        model, optimizer, scheduler, opt_checkpoint, global_step, best_dev_em = util.load(
            model_class, opt.model_path, opt, reset_params=True, name="latest",
        )
        logger.info("Model loaded from %s" % opt.model_path) 

    
    if opt.model_type == 'bart':
        model.model.encoder.n_passages = opt.n_context
        if opt.use_checkpointing:
            model.model.encoder.checkpoint = True
    elif opt.model_type == 't5':
        if opt.use_checkpointing:
            model.encoder.checkpoint = True
        model.encoder.n_passages = opt.n_context


    if opt.world_size > 1 and opt.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train_evaluate(model, optimizer, scheduler, global_step,
        train_dataset, dev_dataset, opt, collator_function, best_dev_em)
