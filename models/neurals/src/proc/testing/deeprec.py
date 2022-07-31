"""
Testing algorithm for DeepRec model
Adapted from github.com/NVIDIA/DeepRecommender
Copyright (c) 2017 NVIDIA Corporation
"""

import torch
from torch.autograd import Variable

import src.loader.deeprec as L
import src.models.deeprec as deeprec
from src.helpers.io import *

def test_deeprec(args):
    train_data_layer, eval_data_layer = deeprec.load_data(
        args, 1, args.path_to_test_data) # batch size of 1 

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

    # load weights from saved model
    print('Loading model from {}'.format(args.path_to_model))
    model.load_state_dict(torch.load(args.path_to_model))
    model.eval()
    
    if args.device == 'cuda':
        model = model.cuda()

    inv_userIdMap = {v: k for k, v in train_data_layer.userIdMap.items()}
    inv_itemIdMap = {v: k for k, v in train_data_layer.itemIdMap.items()}

    preds = dict()

    for i, ((out, src), majorInd) in enumerate(eval_data_layer.iterate_one_epoch_eval(for_inf=True)):
        inputs = Variable(src.cuda().to_dense() if args.device == 'cuda' else src.to_dense())
        targets_np = out.to_dense().numpy()[0, :]
        outputs = model(inputs).cpu().data.numpy()[0, :]
        non_zeros = targets_np.nonzero()[0].tolist()
        major_key = inv_userIdMap[majorInd]  # user
        preds[major_key] = []
        for ind in non_zeros:
            preds[major_key].append((inv_itemIdMap[ind], outputs[ind]))
        if i % 1000 == 0:
            print(f'Predicted {i} batches yet')

    save_submission(args, preds)
