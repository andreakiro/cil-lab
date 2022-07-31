# Testing procedure for LightGCN model architeture
# Adapted from github.com/gusye1234/LightGCN-PyTorch
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.lightgcn import LightGCN
from src.loader.lightgcn import DataLoaderLightGCN
from src.configs import config

#######################################
################ TEST #################
#######################################

def test_lightgcn(args):
    model = LightGCN(args)
    model.load_state_dict(torch.load(args.path_to_model))
    test_dataloader = DataLoaderLightGCN(args, split='test', shuffle=False)
    tdl = test_dataloader.get()

    model.eval()
    model.to(args.device)

    ratings = []
    with torch.no_grad():
        for batch in tqdm(tdl):
            if torch.cuda.is_available():
                batch = batch.cuda()
            ratings.extend(model(batch[:, :2]).tolist())

    # clip ratings
    ratings = [np.clip(x, 1.0, 5.0) for x in ratings]

    # assert predictions fit the submission file
    sub_data = pd.read_csv(config.TEST_DATA)
    assert len(ratings) == len(sub_data["rating"])
    generate_submission(args, sub_data, ratings)

def generate_submission(args, sub_data, ratings):
    sub_data['Prediction'] = ratings
    sub_data['Id'] = (
        'r'
        + sub_data['user'].apply(lambda x: str(x + 1))
        + '_'
        + 'c'
        + sub_data['movie'].apply(lambda x: str(x + 1))
    )
    sub_data = sub_data[['Id', 'Prediction']]
    # sub_data['Prediction'] = sub_data.apply(lambda r: np.clip(r['Prediction'], 1.0, 5.0), axis=1)

    # sub_name = args.path_to_model.split('.')[0].replace('/', '-') + '.csv'
    sub_name = config.SUB_FILE.format(args.model)
    out_path = os.path.join(args.out_path, sub_name)
    sub_data.to_csv(out_path, float_format='%.5f', index=False)
    print(f'Saving submission file at {os.path.join(config.SUB_DIR, sub_name)}')
