from calendar import EPOCH
from utils.utils import get_data, get_input_matrix, generate_submission, submit_on_kaggle
from utils.config import *
import numpy as np
from models.matrix_factorization import ALS
from models.matrix_factorization import BFM
from models.dimensionality_reduction import SVD
from models.clustering import BCA

N_USERS = 10000
N_MOVIES = 1000

def main():
    # load data
    print("Loading data...")
    data = get_data()
    X, W = get_input_matrix(data)
    # BFM
    experiments_on_bfm_rank(X, W, data)
    # SVD
    # experiments_on_svd_rank(X, W)
    # ALS
    # experiments_on_als_rank(X, W)

def experiments_on_svd_rank(X, W):
    ranks = range(1, 15, 1)
    for k in ranks:
        model = SVD(k, N_USERS, N_MOVIES, k, verbose=0)
        model.fit(X, None, W, 0.2)
        model.log_model_info()

def experiments_on_als_rank(X, W):
    ranks = range(1, 15, 1)
    for k in ranks:
        model = ALS(k, N_USERS, N_MOVIES, k, verbose=0)
        model.fit(X, None, W, epochs=10, test_size=0.2, n_jobs=-1)
        model.log_model_info()

def experiments_on_bfm_rank(X, W, data):
    ranks = range(1, 26, 1)
    for k in ranks:
        model = BFM(k, N_USERS, N_MOVIES, k, verbose=1)
        model.fit(X, None, W, data=data, test_size=0.2, iter=200)
        model.log_model_info()

if __name__ == '__main__':
    main()