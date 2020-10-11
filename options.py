import argparse
import os


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialize(parser)

    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        parser.add_argument('--model_path', type=str, default='none', help='path for retraining')
        parser.add_argument('--train_data_path', type=str, default='none', help='path of train data')
        parser.add_argument('--dev_data_path', type=str, default='none', help='path of dev data')
        parser.add_argument('--test_data_path', type=str, default='none', help='path of test data')
        parser.add_argument('--model_type', type=str, default='t5')
        parser.add_argument('--model_size', type=str, default='base')
        parser.add_argument('--write_test_results', action='store_true', help='save test results')

        # dataset parameters
        parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        parser.add_argument('--n_context', type=int, default=1)
        parser.add_argument('--total_step', type=int, default=1000)
        parser.add_argument('--reload_step', type=int, default=-1, help='reload model at step <reload_step>')
        parser.add_argument('--max_passage_length', type=int, default=250, 
                            help='maximum number of tokens in the passages (question included)')
        parser.add_argument('--checkpointing_encoder', action='store_true', help='trades memory for compute')
        parser.add_argument('--checkpointing_decoder', action='store_true', help='trades memory for compute')

        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument("--master_port", type=int, default=-1,
                            help="Master port (for multi-node SLURM jobs)")
        parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        parser.add_argument('--eval_freq', type=int, default=500, 
                            help='evaluate model every <eval_freq> steps during training')
        parser.add_argument('--eval_print_freq', type=int, default=1000, 
                            help='print intermdiate results of evaluation every <eval_print_freq> steps')
                            
        return parser

    def print_options(self, opt):
        message = ''
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>40}: {:<40}{}\n'.format(str(k), str(v), comment)

        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        model_dir = os.path.join(expr_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(expr_dir, 'models'))
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        opt = self.parser.parse_args()
        return opt
