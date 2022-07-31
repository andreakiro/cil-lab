"""
Helper methods for general i/o 
To be used when saving files to disk
"""

import os
import pickle
import torch

import numpy as np
from pathlib import Path
from src.configs import config

def cil_dict(dictionary):
    newdictionary = dict()
    for user in dictionary:
        for item, rating in dictionary[user]:
            if item not in newdictionary:
                newdictionary[item] = []
            newdictionary[item].append((user, rating))

    for item in newdictionary:
        newdictionary[item] = sorted(newdictionary[item])
    # sort by item, then by user, as in the original csv

    return dict(sorted(newdictionary.items()))

def save_model(args, model, epoch):
    filename = f'epoch_{epoch}.model'
    path_to_model = Path(config.OUT_DIR, args.model, args.rname, filename)
    os.makedirs(Path(config.OUT_DIR, args.model, args.rname), exist_ok=True)
    torch.save(model.state_dict(), path_to_model)
    print(f'Saving model archive at {path_to_model}')

def save_log_losses(args, logs):
  path_logs = Path(args.model_output, config.LOG_FILE.format(epochs=args.epochs))
  with open(path_logs, 'wb') as handle:
      pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print(f'Saving logs of losses at {path_logs}')

def save_submission(args, preds):
    preds = cil_dict(preds)
    out_file = Path(args.model_output, config.SUB_FILE.format(model=args.model, rname=args.rname))
    with open(out_file, 'w') as f:
        f.write('Id,Prediction\n')
        for item in preds:
            for user, rating in preds[item]:
                rating = np.clip(rating, 1.0, 5.0)
                f.write('r{}_c{},{}\n'.format(user, item, rating))
    print(f'Saving submission file at {out_file}')