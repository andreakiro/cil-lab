import os
from sklearn.model_selection import train_test_split
import numpy as np
from models.matrix_factorization import ALS, NMF, SVD, FunkSVD, BFM
from models.clustering import BCA
from models.similarity import SimilarityMethods, ComprehensiveSimilarityReinforcement
from utils.utils import *
from utils.config import *


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
    # Similarity
    # experiments_on_similarity(X, W)
    # BFM
    # experiments_on_bfm_rank(data)
    # experiments_on_bfm_iterations(data)
    # experiments_on_bfm_options_by_rank(data)
    # experiments_on_bfm_options_by_iters(data)
    # Get BFM predictions to experiment with ensemble weighting
    # experiments_on_ensemble_bfm(data)
    # Get ALS predictions to experiment with ensemble weighting
    # experiments_on_ensemble_als(data)
    # Get similarity predictions to experiment with ensemble weighting
    # experiments_on_ensemble_similarity(data)
    # Predict Kaggle data
    train_and_run_on_submission_data(X, W, data)


def train_and_run_on_submission_data(X, W, data):
    X_test = load_submission_data(SUBMISSION_DATA_PATH)
    W_test = get_test_mask(X_test)
    # BFM predictions
    bfm_model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=True, with_iu=True, with_ii=True)
    bfm_model.fit(data, None, None, iter=500)
    bfm_preds = bfm_model.predict(X_test)
    # Similarity predictions
    sim_model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting='normal', method="item", k=30, signifiance_threshold=None)
    sim_model.fit(X, None, W, log_rmse=False)
    sim_preds_matrix = sim_model.predict(W_test, invert_norm=False)
    sim_preds = get_preds_from_matrix(X_test, sim_preds_matrix)
    # Ensemble
    weights = {'bfm': 100, 'sim': 19}
    ensemble_preds = (np.array(bfm_preds) * weights['bfm'] + np.array(sim_preds) * weights['sim']) / sum(weights.values())
    # Write results to file, ready for Kaggle
    generate_submission(ensemble_preds, SUBMISSION_DATA_PATH)


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
        model.fit(X, None, W, epochs=20, test_size=0.2)
        model.log_model_info()


def experiments_on_bfm_rank(data):
    ranks = range(1, 50, 1)
    for k in ranks:
        model = BFM(k, N_USERS, N_MOVIES, k, verbose=1, with_ord=True, with_ii=True, with_iu=True)
        model.fit(data, None, None, test_size=0.2, iter=500)
        model.log_model_info()


def experiments_on_similarity(X, W):
    
    methods = ["user", "item", "both"]
    similarity_measures = ["cosine", "PCC", "SiGra"]
    weightings = [None, "normal", "significance", "sigmoid"]
    numbers_nn = [1, 3, 6, 10, 30, 10000] #10000 means taking all the neighbors which are positive

    id = 0
    for method in methods:
        for similarity_measure in similarity_measures:
            for weighting in weightings:
                if not (similarity_measure=="SiGra" and weighting!=None): #If sigra, we don't need to try all the different weighting since it will be set to None
                    if weighting == "significance":
                        signifiance_threshold = 7 if method == "user" else 70 if method == "item" else 20
                    else:
                        signifiance_threshold = None
                    
                    for k in numbers_nn:
                        print(f"Train model number {id}")
                        model = SimilarityMethods(id, N_USERS, N_MOVIES, similarity_measure=similarity_measure, weighting=weighting, method=method, k=k, signifiance_threshold=signifiance_threshold)
                        model.fit(X, None, W, 0.2)
                        model.log_model_info()
                        user_similarity, item_similarity = model.get_similarity_matrices()
                        id += 1


def experiments_on_bfm_iterations(data):
    iterations = range(1, 202, 50) + range(251, 1002, 100)
    for i in iterations:
        model = BFM(i, N_USERS, N_MOVIES, 25, verbose=1, with_ord=True, with_ii=True, with_iu=True)
        model.fit(data, None, None, test_size=0.2, iter=i)
        model.log_model_info(path='./log/log_BFM_iters/')


def experiments_on_bfm_options_by_rank(data):
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
            model.fit(data, None, None, test_size=0.2, iter=250)
            model.log_model_info(path='./log/log_BFM_options_rank/', options_in_name=True)


def experiments_on_bfm_options_by_iters(data):
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
            model.fit(data, None, None, test_size=0.2, iter=i)
            model.log_model_info(path='./log/log_BFM_options_iters/', options_in_name=True)


def experiments_on_ensemble_bfm(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=True, with_iu=True, with_ii=True)
    model.fit(train, None, None, iter=500)

    X_test = test[:, :2]
    test_predictions = model.predict(X_test)

    np.savetxt('log/ensemble/bfm_preds.csv', test_predictions, header='Prediction', comments='')
    # Also save true values of predictions for test set
    np.savetxt('log/ensemble/test_true.csv', test[:, 2], header='Prediction', comments='')


def experiments_on_ensemble_similarity(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X, W = get_input_matrix(train)
    W_test = get_test_mask(test)

    weightings = ['normal', None]
    numbers_nn = [30, 10000] #10000 means taking all the neighbors which are positive

    for weighting in weightings:
        for k in numbers_nn:
            print('Weighting: ' + str(weighting) + ', Neighbors: ' + str(k))

            model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting=weighting, method="item", k=k, signifiance_threshold=None)
            model.fit(X, None, W, log_rmse=False)
            predictions = model.predict(W_test, invert_norm=False)

            # Extract the predictions into one array
            test_predictions = []
            for row in test:
                test_predictions.append(predictions[row[0]][row[1]])
            
            np.savetxt('log/ensemble/sim_preds_w_' + str(weighting) + '_n_' + str(k) + '.csv', test_predictions, header='Prediction', comments='')

    model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting='normal', method="both", use_std=True, k=30, user_weight=0.06, signifiance_threshold=None)
    model.fit(X, None, W, log_rmse=False)
    predictions = model.predict(W_test, invert_norm=False)

    # Extract the predictions into one array
    test_predictions = []
    for row in test:
        test_predictions.append(predictions[row[0]][row[1]])
    
    np.savetxt('log/ensemble/sim_preds_w_normal_n_30_improved.csv', test_predictions, header='Prediction', comments='')


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