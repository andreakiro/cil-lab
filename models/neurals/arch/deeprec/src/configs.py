# Configuration panel for hard params
#####################################

from easydict import EasyDict as edict
from pathlib import Path

config = edict()

# GENERAL CONFIGS

config.USERS = 10000
config.MOVIES = 1000

config.RANDOM_SEED = 42
config.TRAIN_SIZE = 0.8

config.DATA_DIR = 'data'
config.LOG_DIR = 'logs'
config.OUT_DIR = 'logs'
config.SUB_DIR = Path(config.OUT_DIR, 'out-subs')

config.TRAIN_FILE = 'training.data'
config.EVAL_FILE = 'validation.data'
config.TEST_DATA = 'testing.data'
config.LOG_FILE = 'logs-{rname}-{epochs}.pickle'

config.TRAIN_DATA = Path(config.DATA_DIR, config.TRAIN_FILE)
config.EVAL_DATA = Path(config.DATA_DIR, config.EVAL_FILE)
config.TEST_DATA = Path(config.DATA_DIR, config.TEST_DATA)

config.PRINT_FREQ = 50 # print training loss every x batch (in epoch)
config.EVAL_FREQ = 1 # evaluate the model every x epochs (out epoch)

# LIGHTGCN SPECIFIC

config.lightgcn = edict()

config.lightgcn.DATA_DIR = Path(config.DATA_DIR, 'lightgcn')

config.lightgcn.R_MATRIX_TRAIN = Path(config.lightgcn.DATA_DIR, 'r-matrix-train.npz')
config.lightgcn.R_MATRIX_EVAL =  Path(config.lightgcn.DATA_DIR, 'r-matrix-eval.npz')
config.lightgcn.R_MASK_TRAIN =  Path(config.lightgcn.DATA_DIR, 'r-mask-train.npz')
config.lightgcn.R_MASK_EVAL =  Path(config.lightgcn.DATA_DIR, 'r-mask-eval.npz')
