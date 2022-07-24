# Main file handling light GCN model
####################################

import argparse
import torch
import wandb
import numpy as np

from src.proc.training.lightgcn import train_lightgcn
from src.proc.testing.lightgcn import test_lightgcn
from src.configs import config

#######################################
############## CUDA FLAG ##############
#######################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device (as per availability)')

#######################################
############### PARSER ################
#######################################

parser = argparse.ArgumentParser(description='cil-neurals')

# run details and execution environment
parser.add_argument('--device', type=str, default=device, help='manual override of device', choices={'cpu', 'cuda'})
parser.add_argument('--wandb', type=str, default='offline', help='set wandb online or offline', choices={'online', 'offline'})
parser.add_argument('--mode', type=str, default='train', help='define run mode', choices={'train', 'test'})
parser.add_argument('--rname', type=str, help='name of the experiment when wandb is offline')
parser.add_argument('--save', type=bool, default=True, help='whether or not to save models, turn off for cluster optimization')

# model architecture for lightgcn
parser.add_argument('--num_layers', type=int, default=4, help='num of layers i.e. iterations of agg function in model')
parser.add_argument('--emb_size', type=int, default=64, help='layer embedding size of the model')

# training parameters
parser.add_argument('--epochs', type=int, default=50, help='maximum number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='global learning rate for training')
parser.add_argument('--batch_size', type=int, default=2048, help='global batch size for training')

# PATHS to data and logs
parser.add_argument('--path_to_model', type=str, help='restoring model from checkpoint e.g. when testing')
parser.add_argument('--r_train_path', type=str, default=config.lightgcn.R_MATRIX_TRAIN, help='path to R matrix needed for training')
parser.add_argument('--r_mask_train_path', type=str, default=config.lightgcn.R_MASK_TRAIN, help='path to R mask matrix needed for training')

args = parser.parse_args()

#######################################
################ WANDB ################
#######################################

wandb_config = {
  # model architecture
  'architecture': 'lightgcn',
  'num_layers': args.num_layers,
  'emb_size': args.emb_size,
  # training hyperparams
  'epochs': args.epochs,
  'learning_rate': args.learning_rate,
  'batch_size': args.batch_size,
}

wandb.init(
    project='cil-lab-lightgcn',
    config = wandb_config,
    mode = args.wandb,
    job_type = args.mode,
    resume = 'auto',
)

args.rname = args.rname if args.wandb == 'offline' else wandb.run.name

#######################################
################ SEED #################
#######################################

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

#######################################
################ TRAIN ################
#######################################

if __name__ == '__main__':
  if args.mode == 'train':
    train_lightgcn(args)
  elif args.mode == 'test':
    test_lightgcn(args)
  else:
    raise ValueError(f'Invalid run mode choices!')
