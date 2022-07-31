"""
Preprocessing step to get consumable data
Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from src.configs import config
from src.register import register

def extract_users_items_predictions(data_df, offset=0):
    reg = r'r(\d+)_c(\d+)' # parses a row of the "Id" column

    users, movies = [
        np.squeeze(arr)
        for arr in np.split(data_df.Id.str.extract(reg).values.astype(int) - offset, 2, axis=-1)
    ]

    ratings = data_df.Prediction.values
    return users, movies, ratings

def format_and_save(users, movies, ratings, filename):
    column_names = ['user', 'movie', 'rating']
    dataset = np.column_stack((users, movies, ratings))
    dataframe = pd.DataFrame(data=dataset)
    dataframe.columns = column_names
    dataframe.to_csv(filename, index=False)
    print(f'Saved at {filename}')

def preprocess_lightgcn(train_pd, val_pd):
    # extract (users, movies, ratings) from datasets
    train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd, offset=1)
    val_users, val_movies, val_ratings = extract_users_items_predictions(val_pd, offset=1)

    # create and save full training matrix of observed ratings
    filled_training_matrix = np.full((config.NUM_USERS, config.NUM_MOVIES), 0)
    training_mask = np.full((config.NUM_USERS, config.NUM_MOVIES), 0)
    for user, movie, rating in zip(train_users, train_movies, train_ratings):
        filled_training_matrix[user][movie] = rating
        training_mask[user][movie] = 1

    os.makedirs(config.lightgcn.DATA_DIR, exist_ok=True)
    sp.save_npz(config.lightgcn.R_MATRIX_TRAIN, sp.csr_matrix(filled_training_matrix))
    sp.save_npz(config.lightgcn.R_MASK_TRAIN, sp.csr_matrix(filled_training_matrix))

    # create and save full validation matrix of observed ratings
    filled_validation_matrix = np.full((config.NUM_USERS, config.NUM_MOVIES), 0)
    val_mask = np.full((config.NUM_USERS, config.NUM_MOVIES), 0)
    for user, movie, rating in zip(val_users, val_movies, val_ratings):
        filled_validation_matrix[user][movie] = rating
        val_mask[user][movie] = 1

    sp.save_npz(config.lightgcn.R_MATRIX_EVAL, sp.csr_matrix(filled_validation_matrix))
    sp.save_npz(config.lightgcn.R_MASK_EVAL, sp.csr_matrix(val_mask))
    print(f'Saved all lighgcn related at {config.lightgcn.DATA_DIR}')

def preprocess_data(args):
    # get raw data and inputs
    train_data = pd.read_csv(register.datapaths.cil)
    test_data = pd.read_csv(register.datapaths.cil_sample)

    # split dataset into train and val
    train_size = args.split
    train_pd, val_pd = train_test_split(
        train_data,
        train_size=train_size,
        random_state=config.RANDOM_SEED
    )

    # extract (users, movies, ratings) from datasets
    train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
    val_users, val_movies, val_ratings = extract_users_items_predictions(val_pd)
    test_users, test_movies, test_ratings = extract_users_items_predictions(test_data)

    # output directories
    suffix = 'full' if args.split > 1.0 else f'{int(args.split*100)}-{100 - int(args.split*100)}'
    SPLIT_DIR = Path(config.DATA_DIR, f'split-{suffix}')
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # define and save dataframes
    format_and_save(train_users, train_movies, train_ratings, Path(SPLIT_DIR, config.TRAIN_FILE))
    format_and_save(val_users, val_movies, val_ratings, Path(SPLIT_DIR, config.EVAL_FILE))
    format_and_save(test_users, test_movies, test_ratings, Path(SPLIT_DIR, config.TEST_FILE))

    if args.model in ['lightgcn', 'all']:
        preprocess_lightgcn(train_pd, val_pd)

parser = argparse.ArgumentParser(description='cil-preprocess')
parser.add_argument('--model', type=str, help='selection of model for data preparation', choices={'all', 'deeprec', 'lightgcn'}, required=True)
parser.add_argument('--split', type=str, default=config.TRAIN_SIZE, help='size of the trainng split (0.0-1.0) for percentage, full for everything')
args = parser.parse_args()

if __name__ == '__main__':
    args.split = 1176951 if args.split == 'full' else float(args.split)
    type_sp = 'absolute' if args.split > 1.0 else 'percentage'
    validation_split = 1176952 - args.split if args.split > 1.0 else round(1.0 - args.split, 2)
    print(f'Using {type_sp} split to preprocess data for {args.model} models')
    print(f'training split: {args.split }, validation split: {validation_split}')
    preprocess_data(args)