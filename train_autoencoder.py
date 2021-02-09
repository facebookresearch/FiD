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
from autoencoder.model import AutoEncoder, AutoEncoderConfig
import autoencoder.data as data

logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


def train_evaluate(model, optimizer, scheduler, global_step,
                    train_dataset, dev_dataset, opt, collator_function, best_dev_em):

    if opt.is_master:
        tb_logger = SummaryWriter(os.path.join(opt.checkpoint_dir, opt.name))

    train_sampler = (RandomSampler(train_dataset) if opt.local_rank == -1 or opt.world_size == 1
        else DistributedSampler(train_dataset))
    dev_sampler = SequentialSampler(dev_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=opt.per_gpu_batch_size, drop_last=True, num_workers=10, collate_fn=collator_function)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=opt.per_gpu_batch_size,
        drop_last=True, num_workers=10, collate_fn=collator_function)

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while global_step < opt.total_step:
        if opt.world_size > 1:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            global_step += 1
            (idx, text_ids, text_mask) = batch
            text_ids, text_mask = text_ids.cuda(), text_mask.bool().cuda()
            labels = text_ids.clone().masked_fill(~text_mask, -100)

            outputs = model(
                input_ids=text_ids,
                attention_mask=text_mask,
                labels=labels,
            )
            train_loss = outputs[0]
            train_loss.backward()
            util.clip_gradients(model, opt.clip)
            optimizer.step()
            model.zero_grad()
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
    loss = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, text_ids, text_mask) = batch
            text_ids, text_mask = text_ids.cuda(), text_mask.bool().cuda()
            labels = text_ids.masked_fill(~text_mask, -100)

            outputs = model(
                input_ids=text_ids,
                attention_mask=text_mask,
                labels=labels,
            )
            loss.append(text_ids.size(0)*outputs[0])
            total += text_ids.size(0) 
                
    score, total = util.weighted_average(np.mean(loss), total, opt)
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

    tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base') 
    model_class = AutoEncoder

    collator_function = data.Collator(opt, tokenizer)

    examples = data.load_wikipedia('/private/home/gizacard/DPR/data/wikipedia_split/psgs_w100.tsv')
    train_dataset = data.Dataset(examples)
    dev_dataset = data.Dataset(examples[:1000])

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
        model = model_class(AutoEncoderConfig())
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