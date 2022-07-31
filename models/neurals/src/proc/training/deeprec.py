"""
Training algorithm for DeepRec model
Adapted from github.com/NVIDIA/DeepRecommender
Copyright (c) 2017 NVIDIA Corporation
"""

import time
import math
import wandb
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import src.models.deeprec as deeprec
from src.helpers.optimizer import *
from src.helpers.io import *

def train_deeprec(args):
  train_data_layer, eval_data_layer = deeprec.load_data(
    args, int(args.batch_size), args.path_to_eval_data)
  
  layer_sizes = ([
    train_data_layer.vector_dim] 
    + [int(l) for l in args.hidden_layers.split(',')
  ])

  model = deeprec.AutoEncoder(
    layer_sizes = layer_sizes,
    nl_type = args.activation,
    is_constrained = args.constrained,
    dp_drop_prob = args.dropout,
    last_layer_activations = not args.skip_last_layer_nl
  )

  if args.path_to_model is not None:
    model.load_state_dict(torch.load(args.path_to_model))

  optimizer = set_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
  if args.optimizer == 'momentum':
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)

  if args.noise_prob > 0.0:
    dp = nn.Dropout(p=args.noise_prob)

  wandb.watch(model)
  model.to(args.device)
  model.train()
  logs = dict()
  st = time.time()

  # list of epoch num when we'll save model
  epoch_checkpoints = [round(args.epochs/args.num_checkpoints * (x + 1), 0) for x in range(args.num_checkpoints)]
  print('Starting training for {} epochs'.format(args.epochs))

  for i_epoch in range(int(args.epochs)):
    epoch = i_epoch + 1
    e_st = time.time()

    wandb.log({'epoch': epoch})
    training_loss = 0.0
    epoch_loss = 0.0
    denom = 0.0

    for i_batch, batch in enumerate(train_data_layer.iterate_one_epoch()):
      if torch.cuda.is_available():
        batch = batch.cuda()

      inputs = Variable(batch.to_dense())    
      optimizer.zero_grad()
      outputs = model(inputs)
      loss, num_ratings = deeprec.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      training_loss += loss.item()
      epoch_loss += loss.item()
      denom += 1

      if args.dense_refeeding_steps > 0:
        for t in range(args.dense_refeeding_steps):
          inputs = Variable(outputs.data)
          if args.noise_prob > 0.0:
            inputs = dp(inputs)
          
          optimizer.zero_grad()
          outputs = model(inputs)
          loss, num_ratings = deeprec.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          optimizer.step()

    if args.optimizer == 'momentum':
      scheduler.step()

    wandb.log({'train_RMSE': math.sqrt(epoch_loss / denom), 'epoch': epoch})
    print(f'Epoch {epoch} training RMSE loss: {math.sqrt(epoch_loss / denom):.2f}')
    logs[epoch] = {'train_loss': epoch_loss / denom}
    
    if np.isnan(math.sqrt(epoch_loss / denom)):
      wandb.finish() # early termination
      return

    if epoch % args.eval_freq == 0:
      if args.path_to_eval_data != '':
        eval_loss = evaluate(model, eval_data_layer, args.device)
        wandb.log({'val_RMSE': eval_loss, 'epoch': epoch})
        print(f'Epoch {epoch} evaluation loss: {eval_loss:.2f}')
        logs[epoch]['eval_loss'] = eval_loss
        model.train() # revert to train mode

    print('Epoch {} finished in {:.2f} seconds'.format(epoch, time.time() - e_st))

    if epoch in epoch_checkpoints:
      save_model(args, model, epoch)
  
  save_log_losses(args, logs)
  wandb.finish()
  print(f'Finished all in {time.time() - st:.2f} seconds')

#######################################
################ EVAL #################
#######################################

def evaluate(model, evaluation_data_layer, device):
  with torch.no_grad():
    model.eval()
    total_epoch_loss = 0.0
    denom = 0.0

    for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
      inputs = Variable(src.cuda().to_dense() if device == 'cuda' else src.to_dense())
      targets = Variable(eval.cuda().to_dense() if device == 'cuda' else eval.to_dense())
      outputs = model(inputs)
      loss, num_ratings = deeprec.MSEloss(outputs, targets)
      total_epoch_loss += loss.item()
      denom += num_ratings.item()

  return math.sqrt(total_epoch_loss / denom)
