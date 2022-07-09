# File to run to train deeprec model
####################################

import argparse
import torch

from src.train import train

#######################################
############### PARSER ################
#######################################

parser = argparse.ArgumentParser(description='deeprec')

# PATHS to data and logs
parser.add_argument('--path_to_train_data', type=str, default='data/training_90.data', help='path to training data')
parser.add_argument('--path_to_eval_data', type=str, default='data/validation_10.data', help='path to evaluation data')
parser.add_argument('--logdir', type=str, default='logs', help='where to save model and write logs')

# model architecture
parser.add_argument('--major', type=str, default='users', help='major of the model (users or items)')
parser.add_argument('--activation', type=str, default='selu', help='type of the non-linearity used in activations')
parser.add_argument('--layer1_dim', type=int, default=256, help='dimension of the hidden layer 1')
parser.add_argument('--layer2_dim', type=int, default=32, help='dimension of the hidden layer 2')
parser.add_argument('--layer3_dim', type=int, default=0, help='dimension of the hidden layer 3')

# training parameters
parser.add_argument('--epochs', type=int, default=8, help='maximum number of epochs for training')
parser.add_argument('--optimizer', type=str, default='momentum', help='optimizer kind: adam, momentum, adagrad or rmsprop')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='global learning rate for training')
parser.add_argument('--batch_size', type=int, default=64, help='global batch size for training')

# regularization parameters
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N', help='L2 weight decay')
parser.add_argument("--dropout", type=float, default=0.0, metavar="N", help="dropout drop probability")
parser.add_argument('--noise_prob', type=float, default=0.0, metavar='N', help='noise probability')
parser.add_argument("--dense_refeeding_steps", type=int, default=3, metavar="N", help="do data augmentation every X step")

# other parameters
parser.add_argument('--constrained', action='store_true', help='constrained autoencoder')
parser.add_argument('--skip_last_layer_nl', action='store_true', help='if present, decoder\'s last layer will not apply non-linearity function')
parser.add_argument('--num_checkpoints', type=int, default=4, help='number of saved model checkpoints (including last)')
parser.add_argument('--evaluation_frequency', type=int, default=2, help='frequency (epoch-based) of model evaluation')
parser.add_argument('--summary_frequency', type=int, default=100, metavar='N', help='how often to save summaries')
parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N', help='hidden layer sizes, comma-separated')
parser.add_argument('--gpu_ids', type=str, default='0', help='comma-separated gpu ids to use for data parallel training')


args = parser.parse_args()
#print(args)

#######################################
############### PARAMS ################
#######################################

wandb_config = {
  # model architecture
  'architecture': 'deeprec',
  'major': args.major,
  'activation': args.activation,
  'layer1_dim': args.layer1_dim,
  'layer2_dim': args.layer2_dim,
  'layer3_dim': args.layer3_dim,
  # training hyperparams
  'epochs': args.epochs,
  'optimizer': args.optimizer,
  'learning_rate': args.learning_rate,
  'batch_size': args.batch_size,
  # regularization params
  'weight_decay': args.weight_decay,
  'dropout': args.dropout,
  'noise_prob': args.noise_prob,
  'dense_refeeding_steps': args.dense_refeeding_steps,
}

nvidia_params = {
  'batch_size': int(args.batch_size),
  'data_file': args.path_to_train_data,
  'major': args.major,
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
