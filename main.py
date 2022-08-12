from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from models.matrix_factorization import ALS, SVD, FunkSVD, BFM
from models.similarity import SimilarityMethods, ComprehensiveSimilarityReinforcement
from utils import load_data, get_input_matrix, load_submission_data, get_test_mask, get_preds_from_matrix, generate_submission
from utils.config import N_USERS, N_MOVIES, DATA_PATH, SUBMISSION_DATA_PATH
from plots.generate_plots import generate_rank_experiments_plot
import argparse
import os


def main():
    # Recreate the whole log folder structure if necessary to avoid errors
    os.makedirs('log/ensemble', exist_ok=True)
    os.makedirs('log/ensemble_test', exist_ok=True)
    os.makedirs('log/log_BFM_iters', exist_ok=True)
    os.makedirs('log/log_BFM_options_iters', exist_ok=True)
    os.makedirs('log/log_BFM_options_rank', exist_ok=True)

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Reproduce collaborative filtering results.')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--experiments', action='store_true')
    parser.add_argument('--submission', action='store_true')
    args=parser.parse_args()

    if args.submission:
        train_and_run_on_submission_data()
    if args.plot:
        run_plot_experiments()
        generate_rank_experiments_plot()
    if args.ensemble:
        run_ensemble_experiments()
        experiments_on_ensemble_blending()
    if args.experiments:
        run_other_experiments()


def run_plot_experiments():
    print('Running plot experiments')
    data = load_data(DATA_PATH)
    X, W = get_input_matrix(data)
    # SVD
    print('Training SVD on different ranks')
    experiments_on_svd_rank(X, W)
    # ALS
    print('Training ALS on different ranks')
    experiments_on_als_rank(X, W)
    # FunkSVD
    print('Training FunkSVD on different ranks')
    experiments_on_funk_rank(X, W)
    # BFM
    print('Training BFM on different ranks')
    experiments_on_bfm_rank(data)


def run_other_experiments():
    data = load_data(DATA_PATH)
    X, W = get_input_matrix(data)
    # Similarity
    print('Running experiments on similarity models')
    experiments_on_similarity(X, W)
    experiments_on_similarity_nn(X, W)
    experiments_on_similarity_user_weight(X, W)
    # BFM
    print('Running experiments on BFM models')
    experiments_on_bfm_iterations(data)
    experiments_on_bfm_options_by_rank(data)
    experiments_on_bfm_options_by_iters(data)


def run_ensemble_experiments():
    print('Training variations of matrix factorization and similarity models for the ensemble')
    data = load_data(DATA_PATH)
    # Get BFM predictions to experiment with ensemble weighting
    print('Running BFM')
    experiments_on_ensemble_bfm(data)
    # Get ALS predictions to experiment with ensemble weighting
    print('Running ALS')
    experiments_on_ensemble_als(data)
    # Get similarity predictions to experiment with ensemble weighting
    print('Running Similarity')
    experiments_on_ensemble_similarity(data)
    # Get predictions from funkSVD for ensembling
    print('Running FunkSVD')
    experiments_on_ensemble_funksvd(data)
    # Experiment with blending together these predictions
    print('Running model combinations')
    experiments_on_ensemble_blending()


def train_and_run_on_submission_data():
    data = load_data(DATA_PATH)
    # BFM predictions on 80/20 train/test split
    print('Running different BFM options on a 80/20 train/test split')
    experiments_on_ensemble_bfm(data)
    print('Running different similarity options on the same 80/20 train/test split')
    experiments_on_ensemble_similarity(data)

    # Get these predictions, that were written to files
    test = pd.read_csv('log/ensemble/test_true.csv')['Prediction']
    sim_preds2 = pd.read_csv('log/ensemble/sim_preds_w_none_n_30.csv')['Prediction']
    sim_preds3 = pd.read_csv('log/ensemble/sim_preds_w_none_n_10000.csv')['Prediction']
    sim_preds4 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_30.csv')['Prediction']
    sim_preds5 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_10000.csv')['Prediction']
    sim_preds6 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_30_improved.csv')['Prediction']
    bfm1 = pd.read_csv('log/ensemble/bfm_preds_.csv')['Prediction']
    bfm2 = pd.read_csv('log/ensemble/bfm_preds_ii.csv')['Prediction']
    bfm3 = pd.read_csv('log/ensemble/bfm_preds_iu_ii.csv')['Prediction']
    bfm4 = pd.read_csv('log/ensemble/bfm_preds_iu.csv')['Prediction']
    bfm5 = pd.read_csv('log/ensemble/bfm_preds_ord_ii.csv')['Prediction']
    bfm6 = pd.read_csv('log/ensemble/bfm_preds_ord_iu.csv')['Prediction']
    bfm7 = pd.read_csv('log/ensemble/bfm_preds_ord.csv')['Prediction']
    bfm8 = pd.read_csv('log/ensemble/bfm_preds_ord_iu_ii.csv')['Prediction']

    # Build the whole input matrix
    X = np.stack((bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, sim_preds4, sim_preds3, sim_preds2, sim_preds5, sim_preds6), axis=1)
    y = test.values

    # Train the linear regression model on 
    regressor = LinearRegression()
    print('Training the linear regression model')
    regressor.fit(X, y)

    print('Running different BFM options on the whole dataset')
    experiments_on_ensemble_bfm_submission(data)
    print('Running different similarity options on the whole dataset')
    experiments_on_ensemble_similarity_submission(data)

    sim_preds2 = pd.read_csv('log/ensemble_test/sim_preds_w_none_n_30.csv')['Prediction']
    sim_preds3 = pd.read_csv('log/ensemble_test/sim_preds_w_none_n_10000.csv')['Prediction']
    sim_preds4 = pd.read_csv('log/ensemble_test/sim_preds_w_normal_n_30.csv')['Prediction']
    sim_preds5 = pd.read_csv('log/ensemble_test/sim_preds_w_normal_n_10000.csv')['Prediction']
    sim_preds6 = pd.read_csv('log/ensemble_test/sim_preds_w_normal_n_30_improved.csv')['Prediction']
    bfm1 = pd.read_csv('log/ensemble_test/bfm_preds_.csv')['Prediction']
    bfm2 = pd.read_csv('log/ensemble_test/bfm_preds_ii.csv')['Prediction']
    bfm3= pd.read_csv('log/ensemble_test/bfm_preds_iu_ii.csv')['Prediction']
    bfm4= pd.read_csv('log/ensemble_test/bfm_preds_iu.csv')['Prediction']
    bfm5 = pd.read_csv('log/ensemble_test/bfm_preds_ord_ii.csv')['Prediction']
    bfm6 = pd.read_csv('log/ensemble_test/bfm_preds_ord_iu.csv')['Prediction']
    bfm7 = pd.read_csv('log/ensemble_test/bfm_preds_ord.csv')['Prediction']
    bfm8 = pd.read_csv('log/ensemble_test/bfm_preds_ord_iu_ii.csv')['Prediction']

    X_test_all_preds = np.stack((bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, sim_preds4, sim_preds3, sim_preds2, sim_preds5, sim_preds6), axis=1)
    regressor_preds = regressor.predict(X_test_all_preds)

    generate_submission(regressor_preds, SUBMISSION_DATA_PATH, name="final_ensemble.zip")


def experiments_on_funk_rank(X, W):
    i = 0
    for k in range(2, 30, 1):
        print(f"Rank {k}...")
        model = FunkSVD(i, N_USERS, N_MOVIES, k)
        model.fit(X, None, W, 0.2, n_epochs=100)
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
                        id += 1
    
    print(f"Train model number {id}")
    model = ComprehensiveSimilarityReinforcement(id, N_USERS, N_MOVIES, sample_size=15, max_iter=15, verbose=1)
    model.fit(X, None, W, 0.2)
    model.log_model_info()


def experiments_on_similarity_nn(X, W):
    id = 163
    for k in [20, 25, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 200]:
        print(f"Starting model {id}: number neighbors = {k}")
        model = SimilarityMethods(id, N_USERS, N_MOVIES, similarity_measure="PCC", weighting="normal", method="item", k=k, verbose=1)
        model.fit(X, None, W, 0.2)
        model.log_model_info()
        id += 1


def experiments_on_similarity_user_weight(X, W):
    id = 176
    for user_weight in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        print(f"Starting model {id}: user_weight = {user_weight}")
        model = SimilarityMethods(id, N_USERS, N_MOVIES, similarity_measure="PCC", weighting="normal", method="both", k=30, user_weight=user_weight, verbose=1)
        model.fit(X, None, W, 0.2)
        model.log_model_info()
        id += 1


def experiments_on_bfm_iterations(data):
    iterations = range(1, 202, 50) + range(251, 1002, 100)
    for i in iterations:
        print('Iters: ' + str(i))
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


# Generate BFM prediction results for a 80/20 training/test split, to experiment with the ensemble
def experiments_on_ensemble_bfm(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    pattern = {'': [False, False, False],
               'ii': [False, False, True],
               'iu': [False, True, False],
               'iu_ii': [False, True, True],
               'ord': [True, False, False],
               'ord_ii': [True, False, True],
               'ord_iu': [True, True, False],
               'ord_iu_ii': [True, True, True]}
    for desc, i in pattern.items():
        model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=i[0], with_iu=i[1], with_ii=i[2])
        model.fit(train, None, None, iter=500)

        X_test = test[:, :2]
        test_predictions = model.predict(X_test)

        np.savetxt('log/ensemble/bfm_preds_' + desc + '.csv', test_predictions, header='Prediction', comments='')

    # Also save true values of predictions for test set
    np.savetxt('log/ensemble/test_true.csv', test[:, 2], header='Prediction', comments='')


# Generate BFM prediction results from the whole dataset for the submission missing values
def experiments_on_ensemble_bfm_submission(data):
    X_test = load_submission_data(SUBMISSION_DATA_PATH)
    pattern = {'': [False, False, False],
               'ii': [False, False, True],
               'iu': [False, True, False],
               'iu_ii': [False, True, True],
               'ord': [True, False, False],
               'ord_ii': [True, False, True],
               'ord_iu': [True, True, False],
               'ord_iu_ii': [True, True, True]}
    for desc, i in pattern.items():
        model = BFM(50, N_USERS, N_MOVIES, 50, verbose=1, with_ord=i[0], with_iu=i[1], with_ii=i[2])
        model.fit(data, None, None, iter=500)

        test_predictions = model.predict(X_test)

        np.savetxt('log/ensemble_test/bfm_preds_' + desc + '.csv', test_predictions, header='Prediction', comments='')


# Generate similarity prediction results for a 80/20 training/test split, to experiment with the ensemble
def experiments_on_ensemble_similarity(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X, W = get_input_matrix(train)
    W_test = get_test_mask(test)

    weightings = ['normal', None]
    numbers_nn = [30, 10000] #10000 means taking all the neighbors which are positive

    for weighting in weightings:
        for k in numbers_nn:
            print('Weighting: ' + str(weighting).lower() + ', Neighbors: ' + str(k))

            model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting=weighting, method="item", k=k, signifiance_threshold=None)
            model.fit(X, None, W, log_rmse=False)
            predictions = model.predict(W_test, invert_norm=False)

            # Extract the predictions into one array
            test_predictions = []
            for row in test:
                test_predictions.append(predictions[row[0]][row[1]])

            np.savetxt('log/ensemble/sim_preds_w_' + str(weighting).lower() + '_n_' + str(k) + '.csv', test_predictions, header='Prediction', comments='')

    model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting='normal', method="both", use_std=True, k=30, user_weight=0.06, signifiance_threshold=None)
    model.fit(X, None, W, log_rmse=False)
    predictions = model.predict(W_test, invert_norm=False)

    # Extract the predictions into one array
    test_predictions = []
    for row in test:
        test_predictions.append(predictions[row[0]][row[1]])
    
    np.savetxt('log/ensemble/sim_preds_w_normal_n_30_improved.csv', test_predictions, header='Prediction', comments='')


# Generate similarity prediction results from the whole dataset for the submission missing values
def experiments_on_ensemble_similarity_submission(data):
    X, W = get_input_matrix(data)
    test = load_submission_data(SUBMISSION_DATA_PATH)
    W_test = get_test_mask(test)

    weightings = ['normal', None]
    numbers_nn = [30, 10000] #10000 means taking all the neighbors which are positive

    for weighting in weightings:
        for k in numbers_nn:
            print('Weighting: ' + str(weighting).lower() + ', Neighbors: ' + str(k))

            model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting=weighting, method="item", k=k, signifiance_threshold=None)
            model.fit(X, None, W, log_rmse=False)
            predictions = model.predict(W_test, invert_norm=False)

            # Extract the predictions into one array
            test_predictions = []
            for row in test:
                test_predictions.append(predictions[row[0]][row[1]])

            np.savetxt('log/ensemble_test/sim_preds_w_' + str(weighting).lower() + '_n_' + str(k) + '.csv', test_predictions, header='Prediction', comments='')

    model = SimilarityMethods(0, N_USERS, N_MOVIES, similarity_measure="PCC", weighting='normal', method="both", use_std=True, k=30, user_weight=0.06, signifiance_threshold=None)
    model.fit(X, None, W, log_rmse=False)
    predictions = model.predict(W_test, invert_norm=False)

    # Extract the predictions into one array
    test_predictions = []
    for row in test:
        test_predictions.append(predictions[row[0]][row[1]])
    
    np.savetxt('log/ensemble_test/sim_preds_w_normal_n_30_improved.csv', test_predictions, header='Prediction', comments='')


# Generate ALS prediction results for a 80/20 training/test split, to experiment with the ensemble
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


# Generate FunkSVD prediction results for a 80/20 training/test split, to experiment with the ensemble
def experiments_on_ensemble_funksvd(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42, )
    X, W = get_input_matrix(train)

    model = FunkSVD(0, N_USERS, N_MOVIES, 3,)
    model.fit(X, None, W, )
    predictions = model.predict(None)

    # Extract the predictions into one array
    test_predictions = []
    for row in test:
        test_predictions.append(predictions[row[0]][row[1]])

    np.savetxt('log/ensemble/funk_preds.csv', test_predictions, header='Prediction', comments='')


# Experiment with different combinations of models on a 80/20 training/test split
def experiments_on_ensemble_blending(data):
    # Get the predictions that were written to files
    test = pd.read_csv('log/ensemble/test_true.csv')['Prediction']
    als_preds = pd.read_csv('log/ensemble/als_preds.csv')['Prediction']
    funk_pred = pd.read_csv('log/ensemble/funk_preds.csv')['Prediction']
    sim_preds2 = pd.read_csv('log/ensemble/sim_preds_w_none_n_30.csv')['Prediction']
    sim_preds3 = pd.read_csv('log/ensemble/sim_preds_w_none_n_10000.csv')['Prediction']
    sim_preds4 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_30.csv')['Prediction']
    sim_preds5 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_10000.csv')['Prediction']
    sim_preds6 = pd.read_csv('log/ensemble/sim_preds_w_normal_n_30_improved.csv')['Prediction']
    bfm1 = pd.read_csv('log/ensemble/bfm_preds_.csv')['Prediction']
    bfm2 = pd.read_csv('log/ensemble/bfm_preds_ii.csv')['Prediction']
    bfm3 = pd.read_csv('log/ensemble/bfm_preds_iu_ii.csv')['Prediction']
    bfm4 = pd.read_csv('log/ensemble/bfm_preds_iu.csv')['Prediction']
    bfm5 = pd.read_csv('log/ensemble/bfm_preds_ord_ii.csv')['Prediction']
    bfm6 = pd.read_csv('log/ensemble/bfm_preds_ord_iu.csv')['Prediction']
    bfm7 = pd.read_csv('log/ensemble/bfm_preds_ord.csv')['Prediction']
    bfm8 = pd.read_csv('log/ensemble/bfm_preds_ord_iu_ii.csv')['Prediction']

    # Build the whole input matrix
    X = np.stack((bfm1, bfm2, bfm3, bfm4, bfm5, bfm6, bfm7, bfm8, sim_preds4, sim_preds3, sim_preds2, sim_preds5, sim_preds6), axis=1)
    y = test.values

    kf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    results, names = [], []

    models = {
        "LinearReg" : LinearRegression(),
        "Lasso" : Lasso(alpha=0.001),
        "Ridge" : Ridge(alpha=0.01),
        "XGBoost" : XGBRegressor(n_estimators=100, max_depth=7, n_jobs=-1),
        'MLP' : MLPRegressor(random_state=42, max_iter=1000),
        'RF' : RandomForestRegressor(max_depth=2, random_state=0, n_jobs=-1),
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
        if(name == "LinearReg"): model.fit(X, y) 
        if(name == "Lasso"): model.fit(X, y) 
        results.append((-scores))
        names.append(name)
        print(name, ': %.6f (%.6f)' % (np.mean(-scores), np.std(-scores)), "Coef: " + " ".join(["%0.5f" % x for x in model.coef_]) if name == "LinearReg" or name == "Lasso" else "")

    # plot model performance for comparison
    plt.boxplot(results, labels=names, showmeans=True)
    plt.show()

if __name__ == '__main__':
    main()