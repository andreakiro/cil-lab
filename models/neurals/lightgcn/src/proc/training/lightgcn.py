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
from src.losses.RMSE import RMSELoss
from src.data.dataloader import get_dataloader
from src.configs import config

def train_lightgcn(args):
    model = LightGCN(args)
    if args.path_to_model is not None:
        model.load_state_dict(torch.load(args.path_to_model))

    RMSE = RMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataloader, len_tdl = get_dataloader(args, split='train')
    
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
        
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            epoch_loss += loss.item()
            
            if i_batch % config.PRINT_FREQ == (config.PRINT_FREQ - 1):
                print(f'Batch {i_batch + 1:3}: training loss {training_loss / config.PRINT_FREQ:.3f}')
                wandb.log({'train_loss': training_loss / config.PRINT_FREQ})
                training_loss = 0.0
        
        epoch_loss /= len(train_dataloader)
        logs[i_epoch + 1] = {'train_loss': epoch_loss}
        print(f'Epoch {i_epoch + 1:3}: training loss {epoch_loss:.5f}')
        if np.isnan(epoch_loss): # early termination
            wandb.finish()
            return

        if i_epoch % config.EVAL_FREQ == (config.EVAL_FREQ - 1):
            print(f'Starting evaluation of epoch {i_epoch + 1:3}')
            logs[i_epoch + 1]['eval_loss'] = evaluate(args, model, i_epoch + 1)
            if args.save:
                save_model(args, model, i_epoch + 1)
            model.train() # revert to train mode

        e_end_time = time.time()
        print(f'Epoch {i_epoch + 1:3} finished in {e_end_time - e_start_time:.2f} seconds')

    print(f'Finished all in {time.time() - st:.2f} seconds')
    with open('logs.pickle', 'wb') as handle:
        pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(args, model, epoch):
    model.eval() # turn to eval mode
    evaluate_dataloader, _ = get_dataloader(args, split='eval')
    with torch.no_grad():
        RMSE = RMSELoss()
        rmse_eval = 0.0

        for _, batch in enumerate(evaluate_dataloader):
            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            rmse_eval += loss.item()

        rmse_eval /= len(evaluate_dataloader)
        print(f'Epoch {epoch:3}: evaluation loss {rmse_eval:.5f}')
        wandb.log({'eval_loss': rmse_eval})

    return rmse_eval

def save_model(args, model, epoch):
    filename = f'epoch_{epoch}.model'
    path_to_model = Path(config.OUT_DIR, args.rname, filename)
    os.makedirs(Path(config.OUT_DIR, args.rname), exist_ok=True)
    torch.save(model.state_dict(), path_to_model)
    print(f'Saving model archive at {path_to_model}')