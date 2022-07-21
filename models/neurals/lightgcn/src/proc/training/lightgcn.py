# Training procedure for LightGCN model architeture
# Adapted from github.com/gusye1234/LightGCN-PyTorch
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

import os
import torch
import wandb

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
    train_dataloader = get_dataloader(args, split='train')
    
    wandb.watch(model)
    model.to(args.device)
    model.train()

    for i_epoch in range(args.epochs):
        training_loss = 0.0
        print(f'Starting epoch {i_epoch + 1} of {args.epochs}')
        
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            if torch.cuda.is_available():
                batch = batch.cuda()

            scores = model(batch[:, :2])
            loss = RMSE(scores, batch[:, 2])
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            
            if i_batch % config.PRINT_FREQ == (config.PRINT_FREQ - 1):
                print(f'lightgcn training loss {(training_loss / config.PRINT_FREQ):.5f}')
                wandb.log({'train_loss': training_loss / config.PRINT_FREQ})
                training_loss = 0.0

        if i_epoch % config.EVAL_FREQ == (config.EVAL_FREQ - 1):
            evaluate(args, model)
            save_model(args, model, i_epoch)
            model.train() # revert to train mode

def evaluate(args, model):
    model.eval() # turn to eval mode
    evaluate_dataloader = get_dataloader(args, split='eval')
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
        print(f'lightgcn evaluation loss {rmse_eval:.5f}')
        wandb.log({'eval_loss': rmse_eval})

def save_model(args, model, epoch):
    filename = f'epoch_{epoch}.model'
    path_to_model = Path(config.OUT_DIR, args.rname, filename)
    os.makedirs(Path(config.OUT_DIR, args.rname), exist_ok=True)
    torch.save(model.state_dict(), path_to_model)
    print(f'Saving model archive at {path_to_model}')