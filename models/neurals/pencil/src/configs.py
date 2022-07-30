# Configuration panel for hard params
#####################################

from easydict import EasyDict as edict
from pathlib import Path

config = edict()

# GENERAL CONFIGS

config.NUM_USERS = 10000
config.NUM_MOVIES = 1000

config.RANDOM_SEED = 42
config.TRAIN_SIZE = 0.8
# full absolute: 1176951

train_split = 'full' if config.TRAIN_SIZE > 1 else f'{int(config.TRAIN_SIZE*100)}-{100 - int(config.TRAIN_SIZE*100)}'

# FILENAMES

config.DATA_DIR = 'data'
config.OUT_DIR = 'logs'
config.SPLIT_DIR = Path(config.DATA_DIR, f'split-{train_split}')

config.TRAIN_FILE = 'training.data'
config.EVAL_FILE = 'validation.data'
config.TEST_FILE = 'testing.data'

config.LOG_FILE = 'logs-losses-{epochs}.pickle'
config.SUB_FILE = 'sub-{model}.csv'

config.TRAIN_DATA = Path(config.SPLIT_DIR, config.TRAIN_FILE)
config.EVAL_DATA = Path(config.SPLIT_DIR, config.EVAL_FILE)
config.TEST_DATA = Path(config.SPLIT_DIR, config.TEST_FILE)

# LIGHTGN

config.lightgcn = edict()

config.lightgcn.DATA_DIR = Path(config.DATA_DIR, 'lightgcn')
config.lightgcn.R_MATRIX_TRAIN = Path(config.lightgcn.DATA_DIR, 'r-matrix-train.npz')
config.lightgcn.R_MATRIX_EVAL = Path(config.lightgcn.DATA_DIR, 'r-matrix-eval.npz')
config.lightgcn.R_MASK_TRAIN = Path(config.lightgcn.DATA_DIR, 'r-mask-train.npz')
config.lightgcn.R_MASK_EVAL = Path(config.lightgcn.DATA_DIR, 'r-mask-eval.npz')