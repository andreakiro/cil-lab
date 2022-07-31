# Preprocessing step to get data in LightGCN format
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

# TODO module error

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from ..configs import config

def extract_users_items_predictions(data_df):
    reg = r"r(\d+)_c(\d+)" # parses a row of the "Id" column

    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_df.Id.str.extract(reg).values.astype(int) - 1, 2, axis=-1
        )
    ]

    ratings = data_df.Prediction.values
    return users, movies, ratings

def preprocess_data():
    # get raw data and inputs
    raw_data = pd.read_csv("../../../data/data_train.csv") # TODO
    number_of_users, number_of_movies = config.USERS, config.MOVIES
    os.makedirs(config.lightgcn.DATA_DIR, exist_ok=True)

    # split dataset into train and val
    train_size = config.TRAIN_SIZE
    train_pd, val_pd = train_test_split(
        raw_data,
        train_size=train_size,
        random_state=config.RANDOM_SEED
    )

    # compute mean of respective datasets
    mean_train = np.mean(train_pd.Prediction.values)
    mean_val = np.mean(val_pd.Prediction.values)

    # extract (users, movies, ratings) from datasets
    train_users, train_movies, train_ratings = extract_users_items_predictions(train_pd)
    val_users, val_movies, val_ratings = extract_users_items_predictions(val_pd)

    # define and save train dataframe
    column_names = ["user", "movie", "rating"]
    train_dataset = np.column_stack((train_users, train_movies, train_ratings))
    train_df = pd.DataFrame(data=train_dataset)
    train_df.columns = column_names
    train_df.to_csv(config.TRAIN_DATA, index=False)

    # define and save val dataframe
    val_dataset = np.column_stack((val_users, val_movies, val_ratings))
    val_df = pd.DataFrame(data=val_dataset)
    val_df.columns = column_names
    val_df.to_csv(config.EVAL_DATA, index=False)

    # create and save full training matrix of observed ratings
    filled_training_matrix = np.full((number_of_users, number_of_movies), 0)
    training_mask = np.full((number_of_users, number_of_movies), 0)
    for user, movie, rating in zip(train_users, train_movies, train_ratings):
        filled_training_matrix[user][movie] = rating
        training_mask[user][movie] = 1

    sp.save_npz(config.lightgcn.R_MATRIX_TRAIN, sp.csr_matrix(filled_training_matrix))
    sp.save_npz(config.lightgcn.R_MASK_TRAIN, sp.csr_matrix(filled_training_matrix))

    # create and save full validation matrix of observed ratings
    filled_validation_matrix = np.full((number_of_users, number_of_movies), 0)
    val_mask = np.full((number_of_users, number_of_movies), 0)
    for user, movie, rating in zip(val_users, val_movies, val_ratings):
        filled_validation_matrix[user][movie] = rating
        val_mask[user][movie] = 1

    sp.save_npz(config.lightgcn.R_MATRIX_EVAL, sp.csr_matrix(filled_validation_matrix))
    sp.save_npz(config.lightgcn.R_MASK_EVAL, sp.csr_matrix(val_mask))

    # create and save submission data
    sub_pd = pd.read_csv("../../../data/sampleSubmission.csv") # TODO
    sub_users, sub_movies, sub_ratings = extract_users_items_predictions(sub_pd)
    sub_dataset = np.column_stack((sub_users, sub_movies, sub_ratings))
    sub_df = pd.DataFrame(data=sub_dataset)
    sub_df.columns = column_names
    sub_df.to_csv(config.TEST_DATA, index=False)

if __name__ == '__main__':
    preprocess_data()