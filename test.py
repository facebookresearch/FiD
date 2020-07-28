import os
import sys
import torch
import transformers
import slurm
import logging
import data
import util
from fid3 import T5MergeForConditionalGeneration
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from options import Options
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import evaluation
import kilt.postprocess as kp
logger = logging.getLogger(__name__)

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if opt.write_test_results:
        write_path = os.path.join(opt.checkpoint_dir, opt.name, 'test_results')
        fw = open(os.path.join(write_path, '%d.txt'%opt.global_rank), 'w')
    if hasattr(model, "module"):
        model = model.module
    total = 0
    answers = []
    ems = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, question_ids, question_mask, answer_ids,
                answer_mask, context_ids, context_mask) = batch
            question_ids, question_mask = question_ids.cuda(), question_mask.cuda()
            answer_ids, answer_mask = answer_ids.cuda(), answer_mask.bool().cuda()
            context_ids = [c.cuda()[None] if c is not None else None for c in context_ids]
            context_mask = [c.bool().cuda()[None] if c is not None else None for c in context_mask]
            context_ids = torch.cat(context_ids, dim=0)
            context_mask = torch.cat(context_mask, dim=0)
            context_ids = context_ids.view(context_ids.size(0), -1)
            context_mask = context_mask.view(context_mask.size(0), -1)

            #outputs = model.generate(
            #    input_ids=question_ids,
            #    attention_mask=question_mask,
            #    context_ids=context_ids,
            #    context_mask=context_mask,
            #    max_length=50,
            #)
            outputs = model.generate(
                input_ids=context_ids,
                attention_mask=context_mask,
                max_length=50,
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.get_example(idx[k])
                question = example.question
                gold = example.answers
                id = example.id
                ems_score = evaluation.ems(ans, gold)
                ems.append(ems_score)

                if opt.write_test_results:
                    fw.write(str(id) + "\t" + ans + '\n')

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                logger.warning(
                    "%d, %d / %d -- average = %.3f" % (opt.global_rank, i + 1, len(dataloader), np.mean(ems))
                )

    logger.warning("%d, total %d -- average = %.3f" % (opt.global_rank, total, np.mean(ems)))
    if opt.world_size > 1 and not opt.local_rank == -1:
        torch.distributed.barrier()
    t_loss = torch.tensor([np.mean(ems) * total], device=question_ids.device)
    t_total = torch.tensor([total], device=question_ids.device)
    t_loss = util.sum_master(t_loss, opt)
    t_total = util.sum_master(t_total, opt)
    logger.info('total number of example %d'%t_total.item())
    return (t_loss / t_total).item()


if __name__ == "__main__":
    options = Options()
    opt = options.parse()
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)
    logger.info("Distributed training")

    dir_path = os.path.join(opt.checkpoint_dir, opt.name)

    model_name = "t5-" + opt.model_size
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    #model_name = "bart-large"
    #tokenizer = transformers.BartTokenizer.from_pretrained('bart-large')

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
    if opt.write_test_results:
        os.makedirs(os.path.join(dir_path, 'test_results'), exist_ok=True)
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
    model = model_class.from_pretrained(os.path.join(opt.model_path, 'checkpoint', 'best_dev'))
    model = model.cuda()
    model.encoder.nc = opt.n_context

    ems = evaluate(model, test_dataset, test_dataloader, tokenizer, opt)

    if opt.write_test_results and opt.is_master:
        glob_path = os.path.join(opt.checkpoint_dir, opt.name, 'test_results', '*')
        kilt_write_path = os.path.join(opt.checkpoint_dir, opt.name, 'output_kilt_format.jsonl')
        kp.write_kilt_format(glob_path, kilt_write_path) 

    logger.info("EM %.6f" % (ems))
