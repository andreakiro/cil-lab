# Copyright github.com/LucaMalagutti/CIL-ETHZ-2021
##################################################

import random
import sys
import os
from math import floor
import pandas as pd
from src.configs import config
from sklearn.model_selection import train_test_split
import numpy as np

def print_stats(data):
    total_ratings = 0
    print("STATS")
    for user in data:
        total_ratings += len(data[user])
    print("Total Ratings: {}".format(total_ratings))
    print("Total User count: {}".format(len(data.keys())))

def save_data_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as out:
        for item in data:
            for user, rating in data[item]:
                out.write("{}\t{}\t{}\n".format(user, item, rating))

def convert2CILdictionary(dictionary):
    """
    Converts dictionary to newdictionary with items as keys, (user, rating) tuples as values.
    Sorts the items and for each item sorts the (user, rating) tuples by user.
    @param dictionary: dictionary with users as keys, (item, rating) tuples as values.
    """
    newdictionary = dict()
    for user in dictionary:
        for item, rating in dictionary[user]:
            if item not in newdictionary:
                newdictionary[item] = []
            newdictionary[item].append((user, rating))

    # sort by item, then by user, as in the original csv
    for item in newdictionary:
        newdictionary[item] = sorted(newdictionary[item])
    return dict(sorted(newdictionary.items()))

def extract_users_items_predictions(data_df):
    reg = r"r(\d+)_c(\d+)" # parses a row of the "Id" column

    users, movies = [
        np.squeeze(arr)
        for arr in np.split(
            data_df.Id.str.extract(reg).values.astype(int), 2, axis=-1
            # data_df.Id.str.extract(reg).values.astype(int) - 1, 2, axis=-1
        )
    ]

    ratings = data_df.Prediction.values
    return users, movies, ratings

def preprocess_deeprec():
    # get raw data
    raw_data = pd.read_csv("../../../data/data_train.csv") # TODO
    umber_of_users, number_of_movies = config.USERS, config.MOVIES
    os.makedirs(config.lightgcn.DATA_DIR, exist_ok=True)

    print(config.TRAIN_SIZE)
    print(config.RANDOM_SEED)

    # split dataset into train and val
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

    # create and save submission data
    sub_pd = pd.read_csv("../../../data/sampleSubmission.csv") # TODO
    sub_users, sub_movies, sub_ratings = extract_users_items_predictions(sub_pd)
    sub_dataset = np.column_stack((sub_users, sub_movies, sub_ratings))
    sub_df = pd.DataFrame(data=sub_dataset)
    sub_df.columns = column_names
    sub_df.to_csv(config.TEST_DATA, index=False)

def main(args):
    inpt = args[1]
    out_prefix_train = "data/training_90"
    out_prefix_valid = "data/validation_10"
    out_prefix_submission = "data/submission"

    # 0.9 for 90%, 1.0 for 100% train and no validation
    percent = 0.9
    if len(args) > 2:
        if args[2] == "submission":
            # take all the ratings to generate the submission file
            percent = 1
            out_prefix_train = "data/train100/CIL_data100"

    data = dict()

    total_rating_count = 0
    with open(inpt, "r") as inpt_f:  # ratings.csv headers: userId,movieId,rating
        for line in inpt_f:
            if "Id" in line:
                continue
            parts = line.split(",")
            useritem = parts[0].split("_")
            user = int(useritem[0][1:])
            item = int(useritem[1][1:])
            rating = float(parts[1])

            total_rating_count += 1
            if user not in data:
                data[user] = []
            data[user].append((item, rating))

    print("STATS")
    print("Total Ratings: {}".format(total_rating_count))

    training_data = dict()
    validation_data = dict()
    train_set_items = set()

    random.seed(1234)

    for user in data.keys():
        if len(data[user]) < 2:
            print(
                "WARNING, userId {} has less than 2 ratings, skipping user...".format(
                    user
                )
            )
            continue
        ratings = data[user]
        if len(args) <= 2:
            random.shuffle(ratings)
        last_train_ind = floor(percent * len(ratings))
        training_data[user] = ratings[:last_train_ind]
        for rating_item in ratings[:last_train_ind]:
            train_set_items.add(rating_item[0])  # keep track of items from training set

        validation_data[user] = ratings[last_train_ind:]

    # remove items not not seen in training set
    for user, userRatings in validation_data.items():
        validation_data[user] = [
            rating for rating in userRatings if rating[0] in train_set_items
        ]

    # Saves train and evaluation data files
    if len(args) <= 2:
        print("Training Data")
        print_stats(training_data)
        save_data_to_file(
            convert2CILdictionary(training_data), out_prefix_train + ".data" #+ ".train"
        )
        print("Validation Data")
        print_stats(validation_data)
        save_data_to_file(
            convert2CILdictionary(validation_data), out_prefix_valid + ".data" #+ ".valid"
        )
    # Saves non-split train data file
    elif args[2] == "submission" and inpt[-9:] == 'train.csv':
        print("Training Data 100%")
        print_stats(training_data)
        save_data_to_file(
            convert2CILdictionary(training_data), out_prefix_train + ".data" #+ ".train"
        )
    # Saves submission data file
    elif args[2] == "submission":
        print("Submission Data")
        print_stats(training_data)
        save_data_to_file(
            convert2CILdictionary(training_data), out_prefix_submission + ".data" #+ ".submission"
        )
    else:
        print("Invalid arguments:", args[2])

if __name__ == "__main__":
    preprocess_deeprec()
