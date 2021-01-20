import os
import errno
import torch
import sys
import logging
from pathlib import Path
import torch.distributed as dist

logger = logging.getLogger(__name__)

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, best_dev_em, opt, dir_path, name):
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_dev_em": best_dev_em,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    #epoch_path = os.path.join(dir_path, "checkpoint", name)#str(epoch))
    #epoch_path = os.path.realpath(epoch_path)
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path) #, map_location="cuda:"+str(opt.local_rank))
    logger.info("loading checkpoint %s" %optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=opt.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    best_dev_em = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    model = model.to(opt.device) 
    return model, optimizer, scheduler, opt_checkpoint, step, best_dev_em

############ OPTIM


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self, optimizer, warmup_steps, t_total, min_ratio, fixed_lr, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * float(step) / float(
                max(1, self.warmup_steps)
            ) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(
            0.0,
            1.0
            + float((self.min_ratio - 1) * (step - self.warmup_steps))
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate

def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=600, t_total=15000, min_ratio=0., fixed_lr=False)
    return optimizer, scheduler


def print_parameters(net, log_dir, verbose=False):
    file_name = os.path.join(log_dir, "opt.txt")
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = "[Network] Total number of parameters : %.6f M" % (num_params / 1e6)
    print(message)
    if verbose:
        print(net)
    sys.stdout.flush()


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x

def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x

def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device="cuda:"+str(opt.local_rank))
    t_total = torch.tensor([count], device="cuda:"+str(opt.local_rank))
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()

def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    with open(output_path, 'w') as outfile:
        for path in files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()