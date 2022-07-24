from calendar import EPOCH
from configparser import ExtendedInterpolation
from sklearn.model_selection import train_test_split
from utils.utils import get_input_matrix, generate_submission, load_data, load_submission_data, submit_on_kaggle
from utils.config import *
import numpy as np
import pandas as pd
from models.matrix_factorization import ALS, NMF, SVD, FunkSVD, BFM
# from models.clustering import BCA
import os

N_USERS = 10000
N_MOVIES = 1000
DATA_PATH = 'data/data_train.csv'
SUBMISSION_DATA_PATH = 'data/sampleSubmission.csv'

def main():
    # load data
    print("Loading data...")
    data = load_data(DATA_PATH)
    X, W = get_input_matrix(data)
    # SVD
    # experiments_on_svd_rank(X, W)
    # ALS
    # experiments_on_als_rank(X, W)
    # BFM
    # experiments_on_bfm_rank(X, W, data)
    # experiments_on_bfm_iterations(X, W, data)
    # experiments_on_bfm_options_by_rank(X, W, data)
    # experiments_on_bfm_options_by_iters(X, W, data)
    # experiments_on_ensemble_bfm(X, W, data)
    experiments_on_ensemble_als(data)
    # Predict Kaggle data
    # train_and_run_on_submission_data(X, W, data)


def train_and_run_on_submission_data(X, W, data):
    model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=True, with_iu=True, with_ii=True)
    # Train
    model.fit(X, None, W, data, iter=500)
    # Predict
    X_test = load_submission_data(SUBMISSION_DATA_PATH)
    predictions = model.predict(X_test)
    # Write results to file, ready for Kaggle
    data_pd = data_pd.astype({"Prediction": float}, errors='raise')
    data_pd['Prediction'] = predictions
    data_pd.to_csv('submission.zip', compression='zip', float_format='%.3f', index = None)


def clean_logs():
    os.system("rm log/*")


def experiments_on_funk_rank(X, W):
    i = 0
    for k in range(2, 30, 1):
        print(f"Rank {k}...")
        model = FunkSVD(i, N_USERS, N_MOVIES, k)
        model.fit(X, None, W, 0.2, n_epochs=5000)
        model.log_model_info()
        i += 1


def experiments_on_svd_rank(X, W):
    i = 0
    for k in range(2, 30):
        print(f"Rank {k}")
        model = SVD(i, N_USERS, N_MOVIES, k, verbose=0)
        model.fit(X, None, W, 0.2, imputation='zeros')
        model.log_model_info()
        i+=1


def experiments_on_nmf_rank(X, W):
    i = 0
    for k in range(2, 30):
        print(f"Rank {k}")
        model = NMF(i, N_USERS, N_MOVIES, k)
        model.fit(X, None, W, 0.2, imputation='mean', iter=1024)
        model.log_model_info()
        i+=1


def experiments_on_als_rank(X, W):
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]:
        print(f"Rank {k}...")
        model = ALS(k, N_USERS, N_MOVIES, k, verbose=1)
        model.fit(X, None, W, epochs=20, test_size=0.2, )
        model.log_model_info()

def experiments_on_bfm_rank(X, W, data):
    ranks = range(1, 50, 1)
    for k in ranks:
        model = BFM(k, N_USERS, N_MOVIES, k, verbose=1, with_ord=True, with_ii=True, with_iu=True)
        model.fit(X, None, W, data=data, test_size=0.2, iter=500)
        model.log_model_info()

def experiments_on_bfm_iterations(X, W, data):
    iterations = range(1, 202, 50) + range(251, 1002, 100)
    for i in iterations:
        model = BFM(i, N_USERS, N_MOVIES, 25, verbose=1, with_ord=True, with_ii=True, with_iu=True)
        model.fit(X, None, W, data=data, test_size=0.2, iter=i)
        model.log_model_info(path='./log/log_BFM_iters/')

def experiments_on_bfm_options_by_rank(X, W, data):
    ranks = [1] + list(range(10, 51, 10))
    for i in ranks:
        print('Rank: ' + str(i))
        pattern = [[False, False, False],
                   [False, False, True],
                   [False, True, False],
                   [False, True, True],
                   [True, False, False],
                   [True, False, True],
                   [True, True, False],
                   [True, True, True]]
        for j in pattern:
            model = BFM(i, N_USERS, N_MOVIES, i, verbose=1, with_ord=j[0], with_iu=j[1], with_ii=j[2])
            model.fit(X, None, W, data=data, test_size=0.2, iter=250)
            model.log_model_info(path='./log/log_BFM_options_rank/', options_in_name=True)

def experiments_on_bfm_options_by_iters(X, W, data):
    iters = [1] + list(range(100, 501, 100))
    for i in iters:
        print('Iters: ' + str(i))
        pattern = [[False, False, False],
                   [False, False, True],
                   [False, True, False],
                   [False, True, True],
                   [True, False, False],
                   [True, False, True],
                   [True, True, False],
                   [True, True, True]]
        for j in pattern:
            model = BFM(i, N_USERS, N_MOVIES, 25, verbose=1, with_ord=j[0], with_iu=j[1], with_ii=j[2])
            model.fit(X, None, W, data=data, test_size=0.2, iter=i)
            model.log_model_info(path='./log/log_BFM_options_iters/', options_in_name=True)

def experiments_on_ensemble_bfm(X, W, data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=True, with_iu=True, with_ii=True)
    model.fit(X, None, W, data=(train[:, 0], train[:, 1], train[:, 2]), iter=500)

    X_test = test[:, :2]
    test_predictions = model.predict(X_test)

    np.savetxt('log/ensemble/bfm_preds.csv', test_predictions, header='Prediction', comments='')
    np.savetxt('log/ensemble/test_true.csv', test[:, 2], header='Prediction', comments='')

def experiments_on_ensemble_als(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X, W = get_input_matrix(train)

    model = ALS(3, N_USERS, N_MOVIES, 3, verbose=1)
    model.fit(X, None, W, epochs=20)
    predictions = model.predict(X)

    # Extract the predictions into one array
    test_predictions = []
    for row in test:
        test_predictions.append(predictions[row[0]][row[1]])

    np.savetxt('log/ensemble/als_preds.csv', test_predictions, header='Prediction', comments='')
    
if __name__ == '__main__':
    main()