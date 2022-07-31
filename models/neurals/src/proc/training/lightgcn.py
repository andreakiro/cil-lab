"""
Training procedure for LightGCN model architeture
Adapted from github.com/gusye1234/LightGCN-PyTorch
Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
"""

import torch
import wandb
import time
import numpy as np

from torch import optim

from src.models.lightgcn import LightGCN, RMSEloss
from src.loader.lightgcn import DataLoaderLightGCN
from src.helpers.io import *

#######################################
################ TRAIN ################
#######################################

def train_lightgcn(args):
    model = LightGCN(args)
    if args.path_to_model is not None:
        model.load_state_dict(torch.load(args.path_to_model))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataloder = DataLoaderLightGCN(args, split='train')
    tdl = train_dataloder.get()
    len_tdl = train_dataloder.size()

    # list of epoch num when we'll save model
    epoch_checkpoints = [round(args.epochs/args.num_checkpoints * (x + 1), 0) for x in range(args.num_checkpoints)]
    
    wandb.watch(model)
    model.to(args.device)
    model.train()
    logs = dict()
    st = time.time()

    for i_epoch in range(args.epochs):
        epoch = i_epoch + 1
        e_st = time.time()
        
        num_batches = len_tdl / args.batch_size
        print(f'Starting epoch {epoch:3} of {args.epochs:3}')
        print(f'{num_batches:.0f} batches of {args.batch_size} elems')

        wandb.log({'epoch': epoch})
        training_loss = 0.0
        epoch_loss = 0.0
        
        for i_batch, batch in enumerate(tdl):
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            optimizer.zero_grad()
            scores = model(batch[:, :2])
            loss = RMSEloss(scores, batch[:, 2])
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            epoch_loss += loss.item()
            
            if i_batch % args.print_freq == (args.print_freq - 1):
                print(f'Batch {i_batch + 1:3}: training loss {training_loss / args.print_freq:.3f}')
                wandb.log({'train_loss': training_loss / args.print_freq})
                training_loss = 0.0
        
        epoch_loss /= len(tdl)
        logs[epoch] = {'train_loss': epoch_loss}
        print(f'Epoch {epoch:3}: training loss {epoch_loss:.5f}')
        if np.isnan(epoch_loss): # early termination
            wandb.finish()
            return

        if i_epoch % args.eval_freq == (args.eval_freq - 1):
            print(f'Starting evaluation of epoch {epoch:3}')
            logs[epoch]['eval_loss'] = evaluate(args, model, epoch)
            model.train() # revert to train mode

        if epoch in epoch_checkpoints and args.save:
            save_model(args, model, epoch)

        print(f'Epoch {epoch:3} finished in {time.time() - e_st:.2f} seconds')

    save_log_losses(args, logs)
    print(f'Finished all in {time.time() - st:.2f} seconds')

#######################################
################ EVAL #################
#######################################

def evaluate(args, model, epoch):
    model.eval() # turn to eval mode
    eval_dataloder = DataLoaderLightGCN(args, split='eval')
    edl = eval_dataloder.get()
    with torch.no_grad():
        rmse_eval = 0.0

        for _, batch in enumerate(edl):
            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSEloss(scores, batch[:, 2])
            rmse_eval += loss.item()

        rmse_eval /= len(edl)
        print(f'Epoch {epoch:3}: evaluation loss {rmse_eval:.5f}')
        wandb.log({'eval_loss': rmse_eval})

    return rmse_eval