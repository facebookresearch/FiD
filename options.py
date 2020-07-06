import argparse
import os
import torch


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/gizacard', help='models are saved here')
        parser.add_argument('--corpus_type', type=str, default='char')
        parser.add_argument('--dataset', type=str, default='triviaqa')
        parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        parser.add_argument('--test_data_path', type=str, default='none', help='path of test data')
        parser.add_argument('--train_data_path', type=str, default='none', help='path of train data')
        parser.add_argument('--dev_data_path', type=str, default='none', help='path of dev data')
        parser.add_argument('--eval_dir', type=str, default='none', help='path for retraining')
        parser.add_argument('--eval_epoch', type=int, default=-1)
        parser.add_argument('--eval_start_epoch', type=int, default=1)
        parser.add_argument('--model_type', type=str, default='standard') 
        parser.add_argument('--model_size', type=str, default='base') 
        parser.add_argument('--epoch', type=int, default=-1)

        # dataset parameters
        parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument('--max_answer_length', type=int, default=64)
        parser.add_argument('--max_question_length', type=int, default=-1)
        parser.add_argument('--max_context_length', type=int, default=-1)
        parser.add_argument('--cache_layer', type=int, default=11)
        parser.add_argument('--switch_layer', type=int, default=-1)
        parser.add_argument('--topk', type=int, default=32)
        parser.add_argument('--selection_type', type=str, default='none')
        parser.add_argument('--title_option', type=str, default='add')
        parser.add_argument('--cache_n_head', type=int, default=1)
        parser.add_argument('--cache_layer_weights', type=int, default=-1)
        parser.add_argument('--extra_tokens', type=int, default=-1)
        parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        parser.add_argument('--merged_qc', action='store_true')
        parser.add_argument('--reduce_qk', action='store_true')
        parser.add_argument('--n_context', type=int, default=1)
        parser.add_argument('--compression_type', type=str, default='none')
        parser.add_argument('--total_step', type=int, default=1000)
        parser.add_argument('--global_step', type=int, default=0)
        parser.add_argument('--max_passage_length', type=int, default=250, 
                            help='maximum number of tokens in the passages (question included)')


        parser.add_argument('--maxload', type=int, default=-1)
        parser.add_argument('--bptt', type=int, default=128, help='context size')
        parser.add_argument('--mem_sz', type=int, default=128, help='context size')
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument("--master_port", type=int, default=-1,
                            help="Master port (for multi-node SLURM jobs)")
        parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
        parser.add_argument('--eval_single_segment', action='store_true')
        # training parameters
        parser.add_argument('--alpha', type=float, default=0.)
        parser.add_argument('--theta', type=float, default=1.0)
        parser.add_argument('--optim', type=str, default='adam', help='optimizer')
        parser.add_argument('--scheduler', type=str, default='linear', help='optimizer')
        parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
        parser.add_argument('--min_lr', type=float, default=0., help='learning rate')
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)
        parser.add_argument('--eps', type=float, default=1e-8)
        parser.add_argument('--warmup', type=int, default=4000, help='warmup duration')
        parser.add_argument('--n_epochs', type=int, default=50, help='number of training epochs')
        parser.add_argument('--clip', type=float, default=1.)
        parser.add_argument('--merge_encoding', action='store_true')
        parser.add_argument('--fixed_lr', action='store_true')
        parser.add_argument('--print_freq', type=int, default=2000)
        parser.add_argument('--eval_print_freq', type=int, default=2000)
        parser.add_argument('--save_freq', type=int, default=5, help='frequency of model saves')
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--normalized_target', action='store_true')
        # model parameters
        self.initialized = True
        return parser

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        model_dir = os.path.join(expr_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(expr_dir, 'models'))
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        opt = parser.parse_args()
        if torch.cuda.is_available():
            opt.use_cuda = True
        else:
            opt.use_cuda = False

        return opt
