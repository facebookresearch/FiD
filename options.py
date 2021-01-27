import argparse
import os


class Options():
    def __init__(self, option_type='reader'):
        self.option_type = option_type
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        
    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')
        self.parser.add_argument('--write_crossattention_scores', action='store_true', help='save dataset with cross-attention scores')

    def add_reader_options(self):
        self.parser.add_argument('--train_data_path', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data_path', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--text_maxlength', type=int, default=200, help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)

    def add_retriever_options(self):
        self.parser.add_argument('--train_data_path', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data_path', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--indexing_dimension', type=int, default=768)
        self.parser.add_argument('--no_projection', action='store_true', help='No addition Linear layer and layernorm, only works if indexing size equals 768')
        self.parser.add_argument('--question_maxlength', type=int, default=40, help='maximum number of tokens in questions')
        self.parser.add_argument('--passage_maxlength', type=int, default=200, help='maximum number of tokens in passages')
        self.parser.add_argument('--apply_question_mask', action='store_true')
        self.parser.add_argument('--apply_passage_mask', action='store_true')
        self.parser.add_argument('--no_title', action='store_true', help='article titles not included in passages')
        self.parser.add_argument('--n_context', type=int, default=1)


    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                            help="Master port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=500, 
                            help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=5000, 
                            help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000, 
                            help='print intermdiate results of evaluation every <eval_print_freq> steps')


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
