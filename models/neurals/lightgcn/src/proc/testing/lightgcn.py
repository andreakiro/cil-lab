# Testing procedure for LightGCN model architeture
# Adapted from github.com/gusye1234/LightGCN-PyTorch
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

import os
import torch
import pandas as pd
from tqdm import tqdm

from src.models.lightgcn import LightGCN
from src.data.dataloader import get_dataloader
from src.configs import config

def test_lightgcn(args):
    model = LightGCN(args)
    model.load_state_dict(torch.load(args.path_to_model))
    test_dataloader = get_dataloader(args, split='test', shuffle=False)

    model.eval()
    model.to(args.device)

    ratings = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            ratings.extend(model(batch[:, :2]).tolist())

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

    os.makedirs(config.SUB_DIR, exist_ok=True)
    sub_name = args.path_to_model.split('.')[0].replace('/', '-') + '.csv'
    sub_data.to_csv(os.path.join(config.SUB_DIR, sub_name), index=False)
    print(f'Saving submission file at {os.path.join(config.SUB_DIR, sub_name)}')
