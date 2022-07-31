# Copyright (c) 2017 NVIDIA Corporation
#######################################

import os
import copy
import time

import glob
from pathlib import Path
from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import src.loader.deeprec as L
import src.models.deeprec as deeprec
from src.configs import config

import wandb as wandb
import pickle

#######################################
############### HELPERS ###############
#######################################

def set_optimizer(model, optimizer, lr, wd, mom=0.9):
  optimizers = {
    'adam': optim.Adam(
      model.parameters(),
      lr=lr,
      weight_decay=wd,
    ),

    'adagrad': optim.Adagrad(
      model.parameters(),
      lr=lr,
      weight_decay=wd,
    ),

    'momentum': optim.SGD(
      model.parameters(),
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),

    'rmsprop': optim.RMSprop(
      model.parameters(),
      lr=lr,
      momentum=mom,
      weight_decay=wd,
    ),
  }

  try:
    return optimizers[optimizer]
  except ValueError:
    return ValueError('Unknown optimizer.')

def print_details_layers(dl):
  print('Data loaded..!')
  print('Total items found: {}'.format(len(dl.data.keys())))
  print('Vector dim: {}'.format(dl.vector_dim))

def evaluate(encoder, evaluation_data_layer, device):
  with torch.no_grad():
    # evaluate the encoder
    encoder.eval()
    denom = 0.0
    total_epoch_loss = 0.0

    for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
      inputs = Variable(src.cuda().to_dense() if device == 'cuda' else src.to_dense())
      targets = Variable(eval.cuda().to_dense() if device == 'cuda' else eval.to_dense())
      outputs = encoder(inputs)
      loss, num_ratings = deeprec.MSEloss(outputs, targets)
      total_epoch_loss += loss.item()
      denom += num_ratings.item()

  return sqrt(total_epoch_loss / denom)

#######################################
################ TRAIN ################
#######################################

def train_deeprec(args):
  # NVIDIA params
  train_params = {
    'batch_size': int(args.batch_size),
    'data_file': args.path_to_train_data,
    'major': args.major,
    'itemIdInd': 1,
    'userIdInd': 0,
  }

  # define run name of experiment
  l = glob.glob(args.logdir + '/*')
  filtered = [x for x in l if 'experiment' in x]
  exp_name = 'experiment_' + str(len(filtered))
  run_name = exp_name if wandb.run.name is None else wandb.run.name
  
  # loads training data
  print('Loading training data...')
  data_layer = L.UserItemRecDataProvider(params=train_params)
  #print_details_layers(data_layer)

  # loads validation data
  if args.path_to_eval_data != '':
    print('Loading validation data...')
    eval_params = copy.deepcopy(train_params)
    eval_params["data_file"] = args.path_to_eval_data
    eval_data_layer = L.UserItemRecDataProvider(
      params=eval_params,
      user_id_map=data_layer.userIdMap,  # the mappings are provided
      item_id_map=data_layer.itemIdMap,
    )
    eval_data_layer.src_data = data_layer.data
    #print_details_layers(eval_data_layer)

  # define layers
  layer_sizes = (
    [data_layer.vector_dim]
    + [args.layer1_dim]
    + [args.layer2_dim]
  )

  if args.layer3_dim != 0:
    layer_sizes = layer_sizes + [args.layer3_dim]

  # initialize model
  autoenc = deeprec.AutoEncoder(
    layer_sizes = layer_sizes,
    nl_type = args.activation,
    is_constrained = args.constrained,
    dp_drop_prob = args.dropout,
    last_layer_activations = not args.skip_last_layer_nl
  )

  # watch model with wandb
  wandb.watch(autoenc)

  # prepare model checkpoints
  model_output = Path(args.logdir, run_name)
  model_checkpoints = Path(args.out_path, 'checkpoints')
  os.makedirs(model_checkpoints, exist_ok=True)

  # retrieve saved model
  if model_output.is_file():
    print('Loading model from: {}'.format(model_output))
    autoenc.load_state_dict(torch.load(model_output))

  # print architecture
  # print(autoenc)
  
  if args.device == 'cuda':
    # trying to use gpus
    gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    print('Using GPUs: {}'.format(gpu_ids))
  
    if len(gpu_ids) > 1:
      autoenc = nn.DataParallel(autoenc, device_ids=gpu_ids)

    autoenc = autoenc.cuda()
  
  # set training optimizer
  optimizer = set_optimizer(
    autoenc,
    args.optimizer, 
    args.learning_rate, 
    args.weight_decay
  )

  if args.optimizer == 'momentum':
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)

  if args.noise_prob > 0.0:
    dp = nn.Dropout(p=args.noise_prob)

  # list of epoch num when we'll save model
  chkpts = [args.epochs/args.num_checkpoints * x for x in range(1, args.num_checkpoints)]

  autoenc.train()

  # starts training the model
  logs = dict()
  print('Starting training for {} epochs'.format(args.epochs))
  for epoch in range(int(args.epochs)):
    #print('Doing epoch {} of {}'.format(epoch, args.epochs))
    e_start_time = time.time()
    total_epoch_loss = 0.0
    denom = 0.0

    for _, mb in enumerate(data_layer.iterate_one_epoch()):
      inputs = Variable(mb.cuda().to_dense() if args.device == 'cuda' else mb.to_dense())
      optimizer.zero_grad()
      outputs = autoenc(inputs)
      loss, num_ratings = deeprec.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      total_epoch_loss += loss.item()
      denom += 1

      if args.dense_refeeding_steps > 0:
        # magic data augmentation trick (dense refeeding)
        for t in range(args.dense_refeeding_steps):
          inputs = Variable(outputs.data)
          if args.noise_prob > 0.0:
            inputs = dp(inputs)
          optimizer.zero_grad()
          outputs = autoenc(inputs)
          loss, num_ratings = deeprec.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          optimizer.step()

    if args.optimizer == 'momentum':
      scheduler.step()

    e_end_time = time.time()
    wandb.log({'train_RMSE': sqrt(total_epoch_loss / denom), 'epoch': epoch})
    print('Epoch {} finished in {:.2f} seconds'.format(epoch, e_end_time - e_start_time))
    print('\tTRAINING RMSE loss: {:.2f}'.format(sqrt(total_epoch_loss / denom)))
    logs[epoch + 1] = {'train_loss': total_epoch_loss / denom}

    # early termination
    if np.isnan(sqrt(total_epoch_loss / denom)):
      wandb.finish()
      return

    # evaluate model
    if (epoch + 1) % args.evaluation_frequency == 0:
      if args.path_to_eval_data != '':
        eval_loss = evaluate(autoenc, eval_data_layer, args.device)
        wandb.log({'val_RMSE': eval_loss, 'epoch': epoch})
        print('\tEVALUATION RMSE loss: {:.2f}'.format(eval_loss))
        logs[epoch + 1]['eval_loss'] = eval_loss
        autoenc.train()

    # save checkpoint
    if epoch in chkpts:
      version = 'epoch_' + str(epoch) + '.model'
      torch.save(autoenc.state_dict(), Path(model_checkpoints, version))

  path_logs = Path(args.out_path, config.LOG_FILE.format(epochs=args.epochs))
  with open(path_logs, 'wb') as handle:
      pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

  # save std model
  print('Saving model to {}'.format(Path(args.out_path, 'last.model')))
  torch.save(autoenc.state_dict(), Path(args.out_path, 'last.model'))

  # # save onnx model
  # dummy_input = Variable(torch.randn(params['batch_size'], data_layer.vector_dim).type(torch.float))
  # torch.onnx.export(rencoder.float(), dummy_input.cuda() if cuda else dummy_input, model_checkpoint + ".onnx", verbose=True)
  # print("ONNX model saved to {}!".format(model_checkpoint + ".onnx"))

  # close training
  wandb.finish()
