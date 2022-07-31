"""
Helper methods for activation selection
To be used in the training algorithms
"""

import torch
import torch.nn.functional as F

def activation(input, kind):
    # select activation func.
    if kind == 'selu':
        return F.selu(input)
    elif kind == 'relu':
        return F.relu(input)
    elif kind == 'relu6':
        return F.relu6(input)
    elif kind == 'sigmoid':
        return torch.sigmoid(input)
    elif kind == 'tanh':
        return torch.tanh(input)
    elif kind == 'elu':
        return F.elu(input)
    elif kind == 'lrelu':
        return F.leaky_relu(input)
    elif kind == 'swish':
        return input*torch.sigmoid(input)
    elif kind == 'none':
        return input
    else:
        raise ValueError('Unknown non-linearity type')