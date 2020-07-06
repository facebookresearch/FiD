import os
import time
import sys
import torch
import transformers
import slurm
import json
import logging
import data
import util
from fid import T5MergeForConditionalGeneration
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from options import Options
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import evaluation
logger = logging.getLogger(__name__)

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    answers = []
    ems = []
    results = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (
                idx,
                question_ids,
                question_mask,
                answer_ids,
                answer_mask,
                context_ids,
                context_mask,
            ) = batch
            question_ids, question_mask = question_ids.cuda(), question_mask.cuda()
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            context_ids = [c.cuda() if c is not None else None for c in context_ids]
            context_mask = [c.bool().cuda() if c is not None else None for c in context_mask]

            outputs = model.generate(
                input_ids=question_ids,
                attention_mask=question_mask,
                context_ids=context_ids,
                context_mask=context_mask,
                max_length=50,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.get_example(idx[k])
                question = example.question
                gold = example.answers
                id = example.id
                ems_score = evaluation.ems(ans, gold)
                results.append(str(id) + "\t" + ans)

                #print(o)
                #print(ems_score, question, 'answer', ans, 'gold', gold)
                #print(ems_score, 'answer', ans, 'gold', gold)

                ems.append(ems_score)

                #with open('results_%s_%s/results_%d.txt'%(opt.dataset, opt.model_size, opt.global_rank), 'a') as f:
                #    f.write(str(id) + "\t" + ans + '\n')
            if (i + 1) % opt.eval_print_freq == 0:
                logger.warning(
                    "%d, %d / %d -- average = %.3f" % (opt.global_rank, i + 1, len(dataloader), np.mean(ems))
                )

            total += question_ids.size(0)
    
    logger.warning("%d, total %d -- average = %.3f" % (opt.global_rank, total, np.mean(ems)))
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    t_loss = torch.tensor([np.mean(ems) * total], device=question_ids.device)
    t_total = torch.tensor([total], device=question_ids.device)
    t_loss = util.sum_master(t_loss, opt)
    t_total = util.sum_master(t_total, opt)
    logger.info(t_total)
    logger.info(t_loss / t_total)
    return (t_loss / t_total).item(), None
    #return np.mean(ems), "\n".join(results)


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    logger.info("Distributed training")

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)

    if not opt.model_type == 'bart':
        model_name = "t5-" + opt.model_size
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    else:
        model_name = "bart-large"
        tokenizer = transformers.BartTokenizer.from_pretrained('bart-large')
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    collator_function = data.Collator(tokenizer, opt.max_passage_length)
    test_examples = data.load_data(opt.test_data_path, global_rank=opt.global_rank, world_size=opt.world_size)
    test_dataset = data.Dataset(test_examples, opt.n_context, tokenizer, opt.max_passage_length, opt.no_title, )

    test_sampler = SequentialSampler(test_dataset) 
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=opt.per_gpu_batch_size,
        shuffle=False, num_workers=20, collate_fn=collator_function)

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

    model_class = T5MergeForConditionalGeneration

    logger.info("Start eval")
    model = model_class.from_pretrained(os.path.join(opt.eval_dir, 'best_dev'))#'step-'+str(opt.eval_step)))
    model.merge_encoding = opt.merge_encoding
    model.set_extra_tokens(opt.extra_tokens)
    if opt.model_type == "context":
        model.init_cache_layers(opt.cache_layer_weights)
    model.alpha = opt.alpha
    model.theta = opt.theta
    model.cache_layer = opt.cache_layer
    #model.cache.n_head = opt.cache_n_head
    model.merged_qc = opt.merged_qc
    if opt.model_type == 'context':
        model.decoder.set_topk(opt.topk)
        model.decoder.set_selection_type(opt.selection_type)
    model = model.cuda()

    ems, results = evaluate(model, test_dataset, test_dataloader, tokenizer, opt)
    
    logger.info("Epoch %d | valid %.4f" % (opt.eval_epoch, ems))