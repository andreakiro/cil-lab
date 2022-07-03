from calendar import EPOCH
from utils.utils import get_input_matrix, generate_submission, submit_on_kaggle
from utils.config import *
import numpy as np
from models.matrix_factorization import ALS
from models.dimensionality_reduction import SVD
from models.clustering import BCA
from models.similarity import SimilarityMethods, ComprehensiveSimilarityReinforcement

N_USERS = 10000
N_MOVIES = 1000

def main():
    # load data
    print("Loading data...")
    X, W = get_input_matrix()
    # SVD
    experiments_on_svd_rank(X, W)
    # ALS
    experiments_on_als_rank(X, W)
    # Similarity
    experiments_on_similarity(X, W)

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
    
if __name__ == '__main__':
    main()