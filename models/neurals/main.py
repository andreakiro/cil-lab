"""
Main Driver for neural-based techniques
Networks that can be used with this module:
  - DeepRec (NVIDIA) - deep autoencoder for CF
  - LightGCN - graph convolution network for CF
"""

import os
import torch
import pathlib
import argparse
import numpy as np
import traceback

from src.configs import config
from src.register import register
from src.wnb import activate_wnb
from src.parser import *

#######################################
############### OPTIONS ###############
#######################################

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

use_gpu = torch.cuda.is_available()  # global flag
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device for this run')

#######################################
############### PARSER ################
#######################################

parser = argparse.ArgumentParser(description='cil-neurals')

# execution environment
parser.add_argument('--device', type=str, default=device, help='manual override of device', choices={'cpu', 'cuda'})
parser.add_argument('--wandb', type=str, default='offline', help='set wandb online or offline', choices={'online', 'offline'})
parser.add_argument('--save', type=bool, default=True, help='whether or not to save models, turn off for cluster optimization')

# create model's subparsers
subparsers = parser.add_subparsers(dest='model', help='model selection')
deeprec_ps = fill_deeprec_ps(subparsers.add_parser('deeprec', help='running deeprec model'))
lightgcn_ps = fill_lightgcn_ps(subparsers.add_parser('lightgcn', help='running lightgcn model'))

args = parser.parse_args()

#######################################
################# MAIN ################
#######################################

def main():
    args.model_output = pathlib.Path(config.OUT_DIR, args.model, args.rname)
    if args.mode == 'train' and os.path.exists(args.model_output):
        print(f'{args.model_output} already exists, please change name')
        return

    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    
    try:
        activate_wnb(args)
        os.makedirs(args.model_output, exist_ok=True)
        func = register[args.model][args.mode]
        func(args) # call train / test on model
    except Exception as error:
        print(f'Error: {error}')
        print(traceback.format_exc())

if __name__ == '__main__':
    main()