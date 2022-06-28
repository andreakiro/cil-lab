from calendar import EPOCH
from configparser import ExtendedInterpolation
from utils.utils import get_input_matrix, generate_submission, submit_on_kaggle
from utils.config import *
import numpy as np
from models.matrix_factorization import ALS, NMF, SVD, FunkSVD
from models.clustering import BCA
import os

N_USERS = 10000
N_MOVIES = 1000

def main():
    # load data
    print("Loading data...")
    X, W = get_input_matrix()

    model = FunkSVD(1, 10000, 1000, k=3, verbose=1)
    model.fit(X, None, W, test_size=0.2)


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


if __name__ == '__main__':
    main()