# Training procedure for LightGCN model architeture
# Adapted from github.com/gusye1234/LightGCN-PyTorch
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

import os
import torch
import wandb
import time
import pickle
import numpy as np

from torch import optim
from pathlib import Path

from src.models.lightgcn import LightGCN
from src.helpers.RMSE import RMSELoss
from src.loader.lightgcn import DataLoaderLightGCN
from src.configs import config

#######################################
################ TRAIN ################
#######################################

def train_lightgcn(args):
    model = LightGCN(args)
    if args.path_to_model is not None:
        model.load_state_dict(torch.load(args.path_to_model))

    RMSE = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataloder = DataLoaderLightGCN(args, split='train')
    tdl = train_dataloder.get()
    len_tdl = train_dataloder.size()
    
    wandb.watch(model)
    model.to(args.device)
    model.train()
    logs = dict()
    st = time.time()

    for i_epoch in range(args.epochs):
        e_start_time = time.time()
        num_batches = len_tdl / args.batch_size
        print(f'Starting epoch {i_epoch + 1:3} of {args.epochs:3}')
        print(f'{num_batches:.0f} batches of {args.batch_size} elems')
        wandb.log({'epoch': (i_epoch + 1)})
        training_loss = 0.0
        epoch_loss = 0.0
        
        for i_batch, batch in enumerate(tdl):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            epoch_loss += loss.item()
            
            if i_batch % args.print_freq == (args.print_freq - 1):
                print(f'Batch {i_batch + 1:3}: training loss {training_loss / args.print_freq:.3f}')
                wandb.log({'train_loss': training_loss / args.print_freq})
                training_loss = 0.0
        
        epoch_loss /= len(tdl)
        logs[i_epoch + 1] = {'train_loss': epoch_loss}
        print(f'Epoch {i_epoch + 1:3}: training loss {epoch_loss:.5f}')
        if np.isnan(epoch_loss): # early termination
            wandb.finish()
            return

        if i_epoch % args.eval_freq == (args.eval_freq - 1):
            print(f'Starting evaluation of epoch {i_epoch + 1:3}')
            logs[i_epoch + 1]['eval_loss'] = evaluate(args, model, i_epoch + 1)
            if args.save:
                save_model(args, model, i_epoch + 1)
            model.train() # revert to train mode

        e_end_time = time.time()
        print(f'Epoch {i_epoch + 1:3} finished in {e_end_time - e_start_time:.2f} seconds')

    print(f'Finished all in {time.time() - st:.2f} seconds')
    path_logs = Path(args.out_path, config.LOG_FILE.format(epochs=args.epochs))
    with open(path_logs, 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#######################################
################ EVAL #################
#######################################

def evaluate(args, model, epoch):
    model.eval() # turn to eval mode
    eval_dataloder = DataLoaderLightGCN(args, split='eval')
    edl = eval_dataloder.get()
    with torch.no_grad():
        RMSE = RMSELoss()
        rmse_eval = 0.0

        for _, batch in enumerate(edl):
            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            rmse_eval += loss.item()

        rmse_eval /= len(edl)
        print(f'Epoch {epoch:3}: evaluation loss {rmse_eval:.5f}')
        wandb.log({'eval_loss': rmse_eval})

    return rmse_eval

#######################################
############### HELPERS ###############
#######################################

def save_model(args, model, epoch):
    filename = f'epoch_{epoch}.model'
    path_to_model = Path(config.OUT_DIR, args.rname, filename)
    os.makedirs(Path(config.OUT_DIR, args.rname), exist_ok=True)
    torch.save(model.state_dict(), path_to_model)
    print(f'Saving model archive at {path_to_model}')