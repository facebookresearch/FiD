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
import retriever.evaluation
import retriever.data
import retriever.model

logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


tok = transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def train_evaluate(model, optimizer, scheduler, global_step,
                    train_dataset, dev_dataset, opt, collator_function, best_eval_loss):

    if opt.is_main:
        tb_logger = SummaryWriter(os.path.join(opt.checkpoint_dir, opt.name))
    train_sampler = DistributedSampler(train_dataset) if opt.is_distributed else RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=opt.per_gpu_batch_size, 
        drop_last=True, num_workers=10, collate_fn=collator_function)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=opt.per_gpu_batch_size,
        drop_last=False, num_workers=10, collate_fn=collator_function)


    loss, curr_loss = 0.0, 0.0
    epoch = 1
    #eval_loss, top_metric, avg_metric = evaluate(model, dev_dataset, dev_dataloader, tokenizer, opt)
    model.train()
    while global_step < opt.total_steps:
        if opt.is_distributed > 1:
            train_sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            global_step += 1
            (idx, question_ids, question_mask, context_ids, context_mask, gold_score) = batch
            question_ids, question_mask = question_ids.cuda(), question_mask.cuda()
            context_ids, context_mask = context_ids.cuda(), context_mask.cuda()
            gold_score = gold_score.cuda()
            #print(tok.decode(context_ids[0, 0]))
            #print(tok.decode(question_ids[0]))
            outputs = model.score(
                question_ids=question_ids,
                question_mask=question_mask,
                passage_ids=context_ids,
                passage_mask=context_mask,
                gold_score=gold_score,
            )
            train_loss = outputs[0]

            train_loss.backward()

            if global_step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if global_step % opt.eval_freq == 0:
                eval_loss, inversions, top_metric, avg_metric = evaluate(model, dev_dataset, dev_dataloader, tokenizer, opt)
                if opt.is_main:
                    tb_logger.add_scalar("Evaluation", eval_loss, global_step)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if opt.is_main:
                        model_to_save = model.module if hasattr(model, "module") else model
                        util.save(model_to_save, optimizer, scheduler, global_step, best_eval_loss, opt, dir_path, 'best_dev')
                model.train()
            if opt.is_main and global_step % opt.eval_freq == 0:
                message = f"{global_step} / {opt.total_steps} -- train: {curr_loss/opt.eval_freq:.6f}, eval: {eval_loss:.6f}, inv: {inversions:.1f}, lr: {scheduler.get_last_lr()[0]:.6f}"
                for k in top_metric:
                    message += f"| top{k}: {100*top_metric[k]:.1f} | avg{k}: {avg_metric[k]:.1f}"
                logger.info(message)
                tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), global_step)
                curr_loss = 0

            if opt.is_main and global_step % opt.save_freq == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                util.save(model_to_save, optimizer, scheduler, global_step, best_eval_loss, opt, dir_path, f"step-{global_step}")
            if global_step > opt.total_steps:
                break


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    eval_loss = []

    top_metric, avg_metric = {}, {}
    inversions = []
    metric_at = [i for i in [1, 2, 5] if i <= opt.n_context]
    for k in metric_at:
        top_metric[k] = []
        avg_metric[k] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, context_ids, context_mask, gold_score) = batch
            question_ids, question_mask = question_ids.cuda(), question_mask.cuda()
            context_ids, context_mask = context_ids.cuda(), context_mask.cuda()
            gold_score = gold_score.cuda()

            outputs = model.score(
                question_ids=question_ids,
                question_mask=question_mask,
                passage_ids=context_ids,
                passage_mask=context_mask,
                gold_score=gold_score,
            )
            loss = outputs[0]
            scores = outputs[1]
            for k, s in enumerate(scores):
                curr_idx = idx[k]
                has_answer = [dataset.data[curr_idx]['ctxs'][k]['has_answer'] for k in range(opt.n_context)]
                ass = [dataset.data[curr_idx]['ctxs'][k]['score'] for k in range(opt.n_context)]
                s = s.cpu().numpy()
                sorted_idx = np.argsort(-s)
                sorted_has_answer = np.array(has_answer)[sorted_idx]
                inv, el_topm, el_avgm = retriever.evaluation.score(sorted_idx, metric_at)
                inversions.append(inv)
                for k in top_metric:
                    top_metric[k].append(el_topm[k])
                    avg_metric[k].append(el_avgm[k]) 
                eval_loss.append(loss.item())

            total += question_ids.size(0)

    inversions = util.weighted_average(np.mean(inversions), total, opt)[0]
    for k in top_metric:
        top_metric[k] = util.weighted_average(np.mean(top_metric[k]), total, opt)[0]
        avg_metric[k] = util.weighted_average(np.mean(avg_metric[k]), total, opt)[0]


    return loss, inversions, top_metric, avg_metric

if __name__ == "__main__":
    options = Options(option_type='retriever')
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    logger.info("Distributed training")

    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    collator_function = retriever.data.Collator(tokenizer, passage_maxlength=opt.passage_maxlength, question_maxlength=opt.question_maxlength)

    train_examples = retriever.data.load_data(opt.train_data_path, maxload=opt.maxload)
    train_dataset = retriever.data.Dataset(train_examples, opt.n_context)
    eval_examples = retriever.data.load_data(opt.eval_data_path, global_rank=opt.global_rank, world_size=opt.world_size, maxload=opt.maxload) #use the global rank and world size attibutes to split the dev set on multiple gpus
    eval_dataset = retriever.data.Dataset(eval_examples, opt.n_context)
    logger.info(f"Number of examples in train set: {len(train_dataset)}.")
    logger.info(f"Number of examples in eval set: {len(eval_dataset)}.")

    directory_exists = os.path.exists(dir_path)
    if opt.is_distributed:
        torch.distributed.barrier()
    os.makedirs(dir_path, exist_ok=True)
    if not directory_exists and opt.is_main:
        options.print_options(opt)
    if opt.is_distributed:
        torch.distributed.barrier()
    file_handler = logging.FileHandler(filename=os.path.join(dir_path, "run.log"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if opt.is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    if opt.is_distributed > 1:
        torch.distributed.barrier()

    global_step = 0
    best_eval_loss = np.inf
    config = retriever.model.RetrieverConfig(indexing_dimension=opt.indexing_dimension)
    model_class = retriever.model.Retriever
    if not directory_exists and opt.model_path == "none":
        model = model_class(config, initialize_wBERT=True)
        util.set_dropout(model, opt.dropout)
        model = model.to(opt.device)
        optimizer, scheduler = util.set_optim(opt, model)
    elif opt.model_path == "none":
        model, optimizer, scheduler, opt_checkpoint, global_step, best_eval_loss = util.load(model_class, os.path.join(dir_path, 'checkpoint', 'latest'), opt, reset_params=False)
        logger.info("Model loaded from %s" % dir_path)
    else:
        model, optimizer, scheduler, opt_checkpoint, global_step, best_eval_loss = util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info("Model loaded from %s" % opt.model_path)


    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True,
        )
    
    train_evaluate(model, optimizer, scheduler, global_step,
        train_dataset, eval_dataset, opt, collator_function, best_eval_loss)