import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = parse_args()

ROOT_PATH = ""
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(ROOT_PATH, 'runs')
FILE_PATH = join(ROOT_PATH, 'checkpoints')



config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon']
all_models  = ['mf', 'eghg']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['EGHG_n_layers']= args.layer
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['load_adj'] = args.load_adj
config['cache'] = args.cache
config['dropadj'] = args.dropadj
config['Hadj'] = args.Hadj
config['k_G'] = args.k_G
config['k_HG'] = args.k_HG
config['useW'] = args.useW
config['useA'] = args.useA
config['useT'] = args.useT
config['Enhanced'] = args.Enhanced
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
GPU_id = args.GPU

dataset = args.dataset
model_name = args.model
# if dataset not in all_dataset:
#     raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

