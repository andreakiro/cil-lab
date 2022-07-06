# Copyright (c) 2017 NVIDIA Corporation
#######################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

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
                weight_init.xavier_uniform(w)

        # B (bias) decoding
        self.decode_b = nn.ParameterList([nn.Parameter(torch.zeros(reversed_enc_layers[i + 1])) for i in range(len(reversed_enc_layers) - 1)])

        print("******************************")
        print("******************************")

        print(layer_sizes)
        print("Dropout drop probability: {}".format(self._dp_drop_prob))

        print("Encoder pass:")
        for ind, w in enumerate(self.encode_w):
            print(w.data.size())
            print(self.encode_b[ind].size())

        print("Decoder pass:")
        if self.is_constrained:
            print('Decoder is constrained')
            for ind, w in enumerate(list(reversed(self.encode_w))):
                print(w.transpose(0, 1).size())
                print(self.decode_b[ind].size())
        else:
            for ind, w in enumerate(self.decode_w):
                print(w.data.size())
                print(self.decode_b[ind].size())

        print("******************************")
        print("******************************")

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

def activation(input, kind):
    # select activation func.
    if kind == 'selu':
        return F.selu(input)
    elif kind == 'relu':
        return F.relu(input)
    elif kind == 'relu6':
        return F.relu6(input)
    elif kind == 'sigmoid':
        return F.sigmoid(input)
    elif kind == 'tanh':
        return F.tanh(input)
    elif kind == 'elu':
        return F.elu(input)
    elif kind == 'lrelu':
        return F.leaky_relu(input)
    elif kind == 'swish':
        return input*F.sigmoid(input)
    elif kind == 'none':
        return input
    else:
        raise ValueError('Unknown non-linearity type')

def MSEloss(inputs, targets, size_average=False):
    # computes MSE loss during training
    mask = targets != 0
    num_ratings = torch.sum(mask.float())
    criterion = nn.MSELoss(reduction='sum' if not size_average else 'mean')
    return criterion(inputs * mask.float(), targets), Variable(torch.Tensor([1.0])) if size_average else num_ratings
    