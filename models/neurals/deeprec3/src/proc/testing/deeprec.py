from src.data.converter import convert2CILdictionary
import src.data.layers as L
import src.models.deeprec as deeprec
import numpy as np
import copy
import torch
from torch.autograd import Variable

#######################################
################ EVAL #################
#######################################

def eval(args, params, cuda):
    params["batch_size"] = 1

    # loads training data
    print('Loading training data...')
    data_layer = L.UserItemRecDataProvider(params=params)

    # loads evaluation (sub) data
    print("Loading submission data")
    eval_params = copy.deepcopy(params)
    eval_params["data_file"] = args.path_to_eval_data
    eval_data_layer = L.UserItemRecDataProvider(
      params=eval_params,
      user_id_map=data_layer.userIdMap,  # the mappings are provided
      item_id_map=data_layer.itemIdMap,
    )
    eval_data_layer.src_data = data_layer.data

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

    # load weights from saved model
    print('Loading model from {}'.format(args.path_to_model))
    autoenc.load_state_dict(torch.load(args.path_to_model))
    autoenc.eval()
    
    if cuda:
        autoenc = autoenc.cuda()

    inv_userIdMap = {v: k for k, v in data_layer.userIdMap.items()}
    inv_itemIdMap = {v: k for k, v in data_layer.itemIdMap.items()}

    preds = dict()

    for _, ((out, src), majorInd) in enumerate(eval_data_layer.iterate_one_epoch_eval(for_inf=True)):
        inputs = Variable(src.cuda().to_dense() if cuda else src.to_dense())
        targets_np = out.to_dense().numpy()[0, :]
        outputs = autoenc(inputs).cpu().data.numpy()[0, :]
        non_zeros = targets_np.nonzero()[0].tolist()
        major_key = inv_userIdMap[majorInd]  # user
        preds[major_key] = []
        for ind in non_zeros:
            preds[major_key].append((inv_itemIdMap[ind], outputs[ind]))

    preds = convert2CILdictionary(preds)

    # write predictions
    with open(args.out_file, 'w') as f:
        f.write('Id,Prediction\n')
        for item in preds:
            for user, rating in preds[item]:
                rating = np.clip(rating, 1.0, 5.0)
                f.write('r{}_c{},{}\n'.format(user, item, rating)) 
    
