# File to run to train deeprec model
####################################

import argparse
import torch

from src.train import train

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
parser.add_argument( "--layer1_dim", type=str, default="256", metavar="N", help="hidden layer 1 size",)
parser.add_argument("--layer2_dim", type=str, default="32", metavar="N", help="hidden layer 2 size",)
parser.add_argument("--layer3_dim", type=str, default="0", metavar="N", help="hidden layer 3 size",)

args = parser.parse_args()
#print(args)

#######################################
############### PARAMS ################
#######################################

wandb_config = {
  # model architecture
  'architecture': 'deeprec',
  'activation': args.non_linearity_type,
  'layer1_dim': args.layer1_dim,
  'layer2_dim': args.layer2_dim,
  'layer3_dim': args.layer3_dim,
  # training hyperparams
  'optimizer': args.optimizer,
  'learning_rate': args.learning_rate,
  'batch_size': args.batch_size,
  # regularization params
  'weight_decay': args.weight_decay,
  'P_dropout': args.dropout,
  'P_noise': args.noise_prob,
  'dense_refeeding_steps': args.dense_refeeding_steps,
}

nvidia_params = {
  'batch_size': int(args.batch_size),
  'data_dir': args.path_to_train_data,
  'major': 'users',
  'itemIdInd': 1,
  'userIdInd': 0,
}

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

if __name__ == '__main__':
    train(args, wandb_config, nvidia_params, cuda)
