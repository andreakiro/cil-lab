"""
DeepRec model PyTorch implementation
Adapted from github.com/NVIDIA/DeepRecommender
Copyright (c) 2017 NVIDIA Corporation
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
import src.loader.deeprec as L
from src.helpers.activation import activation

#######################################
############ DEEPREC MODEL ############
#######################################

class AutoEncoder(nn.Module):

    def __init__(
        self,
        layer_sizes, 
        nl_type = 'selu', 
        is_constrained = True,
        dp_drop_prob = 0.0,
        last_layer_activations = True
    ):
        """
        Describes an AutoEncoder model
        :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x)
        For example: [10000, 1024, 512] will result in:
        - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
        - decoder 2 layers: 512x1024 and 1024x10000.
        :param nl_type: (default 'selu') Type of no-linearity
        :param is_constrained: (default: True) Should constrain decoder weights
        :param dp_drop_prob: (default: 0.0) Dropout drop probability
        :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
        """

        super(AutoEncoder, self).__init__()

        # standard parameters
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        self.drop = nn.Dropout(p=dp_drop_prob)
        self._last = len(layer_sizes) - 2
        self._nl_type = nl_type
        self.is_constrained = is_constrained
        self.layer_sizes = layer_sizes
        
        # W (mask) encoding
        self.encode_w = nn.ParameterList([nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i])) for i in range(len(layer_sizes) - 1)])
        for ind, w in enumerate(self.encode_w):
            weight_init.xavier_uniform_(w)

        # B (bias) encoding
        self.encode_b = nn.ParameterList([nn.Parameter(torch.zeros(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)])

        # W (mask) decoding
        reversed_enc_layers = list(reversed(layer_sizes))
        if not is_constrained:
            self.decode_w = nn.ParameterList([nn.Parameter(torch.rand(reversed_enc_layers[i + 1], reversed_enc_layers[i])) for i in range(len(reversed_enc_layers) - 1)])
            for ind, w in enumerate(self.decode_w):
                weight_init.xavier_uniform_(w)

        # B (bias) decoding
        self.decode_b = nn.ParameterList([nn.Parameter(torch.zeros(reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

        #self.print_architecture()

    def encode(self, x):
        # apply encoding layers to x
        for ind, w in enumerate(self.encode_w):
            x = activation(input=F.linear(input=x, weight=w, bias=self.encode_b[ind]), kind=self._nl_type)
        
        # dropout regularization
        if self._dp_drop_prob > 0:
            x = self.drop(x)
        
        return x

    def decode(self, z):
        # apply decoding layers to z
        if self.is_constrained:
            # constrained autoencode re-uses weights from encoder
            for ind, w in enumerate(list(reversed(self.encode_w))):
                # last layer or decoder should not apply non linearities
                nl_kind = self._nl_type if ind != self._last or self._last_layer_activations else 'none'
                z = activation(input=F.linear(input=z, weight=w.transpose(0, 1), bias=self.decode_b[ind]), kind=nl_kind)
        else:
            for ind, w in enumerate(self.decode_w):
                # last layer or decoder should not apply non linearities
                nl_kind = self._nl_type if ind != self._last or self._last_layer_activations else 'none'
                z = activation(input=F.linear(input=z, weight=w, bias=self.decode_b[ind]), kind=nl_kind)

        return z

    def forward(self, x):
        return self.decode(self.encode(x))

#######################################
############### HELPERS ###############
#######################################

def MSEloss(inputs, targets, size_average=False):
    # computes MSE loss during training
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings
    
def load_data(args, batch_size, eval_path):
    train_params = {
        'batch_size': batch_size,
        'data_file': args.path_to_train_data,
        'major': args.major,
        'itemIdInd': 1,
        'userIdInd': 0,
    } # NVIDIA params

    print('Loading training data...')
    train_data_layer = L.UserItemRecDataProvider(params=train_params)

    eval_data_layer = None
    if args.path_to_eval_data != '':
        print('Loading evaluation data...')
        eval_params = copy.deepcopy(train_params)
        eval_params['data_file'] = eval_path
        eval_data_layer = L.UserItemRecDataProvider(
            params=eval_params,
            user_id_map=train_data_layer.userIdMap,  # the mappings are provided
            item_id_map=train_data_layer.itemIdMap,
        )
        eval_data_layer.src_data = train_data_layer.data

    return train_data_layer, eval_data_layer