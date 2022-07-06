# Copyright (c) 2017 NVIDIA Corporation
#######################################

import os
import copy
import time

import argparse
from pathlib import Path
from math import sqrt

import torch
import torch.nn as nn
from torch.autograd import Variable

import src.layers as L
import src.model as deeprec
from src.helpers import *

import wandb as wandb

#######################################
############### PARSER ################
#######################################

parser = argparse.ArgumentParser(description='deeprec')
parser.add_argument("--learning_rate", type=float, default=0.00001, metavar="N", help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N', help='L2 weight decay')
parser.add_argument("--dropout", type=float, default=0.0, metavar="N", help="dropout drop probability")
parser.add_argument('--noise_prob', type=float, default=0.0, metavar='N', help='noise probability')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='global batch size')
parser.add_argument('--summary_frequency', type=int, default=100, metavar='N', help='how often to save summaries')
parser.add_argument('--aug_step', type=int, default=-1, metavar='N', help='do data augmentation every X step')
parser.add_argument('--constrained', action='store_true', help='constrained autoencoder')
parser.add_argument('--skip_last_layer_nl', action='store_true', help='if present, decoder\'s last layer will not apply non-linearity function')
parser.add_argument('--num_epochs', type=int, default=5, metavar='N', help='maximum number of epochs')
parser.add_argument('--save_every', type=int, default=3, metavar='N', help='save every N number of epochs')
parser.add_argument('--optimizer', type=str, default="momentum", metavar='N', help='optimizer kind: adam, momentum, adagrad or rmsprop')
parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N', help='hidden layer sizes, comma-separated')
parser.add_argument('--gpu_ids', type=str, default="0", metavar='N', help='comma-separated gpu ids to use for data parallel training')
parser.add_argument('--path_to_train_data', type=str, default="data/train90", metavar='N', help='Path to training data')
parser.add_argument('--path_to_eval_data', type=str, default="data/valid", metavar='N', help='Path to evaluation data')
parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N', help='type of the non-linearity used in activations')
parser.add_argument('--logdir', type=str, default="logs", metavar='N', help='where to save model and write logs')
parser.add_argument("--dense_refeeding_steps", type=int, default=3, metavar="N", help="do data augmentation every X step")
parser.add_argument( "--layer1_dim",
    type=str,
    default="256",
    metavar="N",
    help="hidden layer 1 size",)
parser.add_argument("--layer2_dim",
    type=str,
    default="32",
    metavar="N",
    help="hidden layer 2 size",)
parser.add_argument("--layer3_dim",
    type=str,
    default="0",
    metavar="N",
    help="hidden layer 3 size",)

args = parser.parse_args()
print(args)

#######################################
############## CUDA FLAG ##############
#######################################

cuda = torch.cuda.is_available()

if cuda:
  print('GPU is available') 
else: 
  print('GPU is not available')

#######################################
################ TRAIN ################
#######################################

def train():
  # setup wandb
  wandb.init(
      project = "deeprec",
      config={
        "batch_size": args.batch_size,
        "layer1_dim": args.layer1_dim,
        "layer2_dim": args.layer2_dim,
        "layer3_dim": args.layer3_dim,
        "activation": args.non_linearity_type,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "noise_prob": args.noise_prob,
        "dropout": args.dropout,
        "dense_refeeding_steps": args.dense_refeeding_steps,
      },
    )

  # std params
  params = dict()
  params['batch_size'] = int(wandb.config["batch_size"])
  params['data_dir'] =  args.path_to_train_data
  params['major'] = 'users'
  params['itemIdInd'] = 1
  params['userIdInd'] = 0

  # loads training data
  print("Loading training data")
  data_layer = L.UserItemRecDataProvider(params=params)
  print_details_layers(data_layer)

  # loads validation data
  if args.path_to_eval_data != "":
      print("Loading validation data")
      eval_params = copy.deepcopy(params)
      eval_params["data_dir"] = args.path_to_eval_data
      eval_data_layer = L.UserItemRecDataProvider(
          params=eval_params,
          user_id_map=data_layer.userIdMap,  # the mappings are provided
          item_id_map=data_layer.itemIdMap,
      )
      eval_data_layer.src_data = data_layer.data
      print_details_layers(eval_data_layer)
  else:
      print("Skipping eval data")

  # define layers
  layer_sizes = (
    [data_layer.vector_dim]
    + [int(wandb.config["layer1_dim"])]
    + [int(wandb.config["layer2_dim"])]
  )

  if (wandb.config["layer3_dim"]) != "0":
    layer_sizes = layer_sizes + [int(wandb.config["layer3_dim"])]

  # initialize model
  rencoder = deeprec.AutoEncoder(
    layer_sizes=layer_sizes,
    nl_type=args.non_linearity_type,
    is_constrained=args.constrained,
    dp_drop_prob=wandb.config["dropout"],
    last_layer_activations=not args.skip_last_layer_nl
  )

  # watch model with wandb
  wandb.watch(rencoder)

  # prepare model checkpoints
  os.makedirs(args.logdir, exist_ok=True)
  model_checkpoint = args.logdir + "/model"
  path_to_model = Path(model_checkpoint)

  # retrieve saved model
  if path_to_model.is_file():
    print("Loading model from: {}".format(model_checkpoint))
    rencoder.load_state_dict(torch.load(model_checkpoint))

  print('######################################################')
  print('################### AutoEncoder Model ################')
  print('######################################################')
  print(rencoder)
  print('######################################################')
  print('######################################################')
  print('######################################################')

  # trying to use gpus
  gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
  print('Using GPUs: {}'.format(gpu_ids))
  
  if len(gpu_ids) > 1:
    rencoder = nn.DataParallel(rencoder, device_ids=gpu_ids)
  
  if cuda:
    rencoder = rencoder.cuda()
  
  # set training optimizer
  optimizer = set_optimizer(
    args.optimizer, 
    wandb.config["learning_rate"], 
    wandb.config["weight_decay"], 
    rencoder
  )

  if args.optimizer == "momentum":
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)

  if args.noise_prob > 0.0:
    dp = nn.Dropout(p=wandb.config["noise_prob"])

  # training indicators
  t_loss = 0.0
  t_loss_denom = 0.0
  global_step = 0

  # starts training the model
  for epoch in range(int(args.num_epochs)):
    print("Doing epoch {} of {}".format(epoch, args.num_epochs))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0

    for i, mb in enumerate(data_layer.iterate_one_epoch()):
      inputs = Variable(mb.cuda().to_dense() if cuda else mb.to_dense())
      optimizer.zero_grad()
      outputs = rencoder(inputs)
      loss, num_ratings = deeprec.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      global_step += 1
      t_loss += loss.item()
      t_loss_denom += 1

      if i % args.summary_frequency == 0:
        rmse = sqrt(t_loss / t_loss_denom)
        print("t_loss_denom: ", t_loss_denom)
        print("[%d, %5d] RMSE: %.7f" % (epoch, i, rmse))
        t_loss = 0
        t_loss_denom = 0.0

      total_epoch_loss += loss.item()
      denom += 1

      if wandb.config["dense_refeeding_steps"] > 0:
        # magic data augmentation trick happen here (dense refeeding)
        for t in range(wandb.config["dense_refeeding_steps"]):
          inputs = Variable(outputs.data)
          if args.noise_prob > 0.0:
            inputs = dp(inputs)
          optimizer.zero_grad()
          outputs = rencoder(inputs)
          loss, num_ratings = deeprec.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          # wandb.log({"MSE_loss": loss})
          loss.backward()
          optimizer.step()

    if args.optimizer == "momentum":
      scheduler.step()

    e_end_time = time.time()
    wandb.log({"train_RMSE": sqrt(total_epoch_loss / denom)})
    print(" denom:", denom)
    print("Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}".format(epoch, e_end_time - e_start_time, sqrt(total_epoch_loss / denom)))

    # evaluate model
    if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
      if args.path_to_eval_data != "":
        eval_loss = evaluate(rencoder, eval_data_layer, cuda)
        wandb.log({"val_RMSE": eval_loss, "epoch": epoch})
        print("Epoch {} EVALUATION LOSS: {}".format(epoch, eval_loss))
    else:
      print("Skipping evaluation")

  # save std model 
  print("Saving model to {}".format(model_checkpoint + ".last"))
  torch.save(rencoder.state_dict(), model_checkpoint + ".last")

  # # save onnx model
  # dummy_input = Variable(torch.randn(params['batch_size'], data_layer.vector_dim).type(torch.float))
  # torch.onnx.export(rencoder.float(), dummy_input.cuda() if cuda else dummy_input, model_checkpoint + ".onnx", verbose=True)
  # print("ONNX model saved to {}!".format(model_checkpoint + ".onnx"))

  # close training
  print("Done")
  quit()

if __name__ == '__main__':
  train()
