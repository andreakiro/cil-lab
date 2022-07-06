# Copyright (c) 2017 NVIDIA Corporation
#######################################

import numpy as np
from math import sqrt
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import src.model as deeprec

#######################################
############### HELPERS ###############
#######################################

def evaluate(encoder, evaluation_data_layer, cuda):
    # evaluate the encoder
    encoder.eval()
    denom = 0.0
    total_epoch_loss = 0.0

    for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
        inputs = Variable(src.cuda().to_dense() if cuda else src.to_dense())
        targets = Variable(eval.cuda().to_dense() if cuda else eval.to_dense())
        outputs = encoder(inputs)
        loss, num_ratings = deeprec.MSEloss(outputs, targets)
        total_epoch_loss += loss.item()
        denom += num_ratings.item()

    return sqrt(total_epoch_loss / denom)

def set_optimizer2(optimizer, lr, weight_decay, rencoder):
    if optimizer == "adam":
        optimizer = optim.Adam(rencoder.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        optimizer = optim.Adagrad(rencoder.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "momentum":
        optimizer = optim.SGD(rencoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        optimizer = optim.RMSprop(rencoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise  ValueError('Unknown optimizer kind')

def set_optimizer(optimizer, lr, weight_decay, rencoder):
    """Returns autoencoder optimizer"""
    optimizers = {
        "adam": optim.Adam(rencoder.parameters(), lr=lr, weight_decay=weight_decay),
        "adagrad": optim.Adagrad(
            rencoder.parameters(), lr=lr, weight_decay=weight_decay
        ),
        "momentum": optim.SGD(
            rencoder.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        ),
        "rmsprop": optim.RMSprop(
            rencoder.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        ),
    }

    try:
        return optimizers[optimizer]
    except ValueError:
        raise ValueError("Unknown optimizer kind")

def print_details_layers(dl):
    print("Data loaded!")
    print("Total items found: {}".format(len(dl.data.keys())))
    print("Vector dim: {}".format(dl.vector_dim))
