#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix completion algorithms based on matrix factorization.

Algorithms implemented in this module:
  - Alternate Least-Square (ALS) algorithm
  - Non-negative Matrix Factorization (NMF) algorithm
"""

import numpy as np
from models.base_model import BaseModel
from models.dimensionality_reduction import SVD
import json
from joblib import Parallel, delayed
import os
from sklearn.decomposition import NMF as NMF_sl
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from sys import platform
import myfm
from myfm import RelationBlock
from scipy import sparse as sps
from collections import defaultdict

######################
###      ALS       ###
######################

class ALS(BaseModel):
    """
    ALS model
    ---
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix.
    
    Parameters
    ----------
    model_id : int
        model identification number

    n_users : int
        rows of the input matrix

    n_movies : int
        columns of the input matrix

    k : int
        number of latent factors to use in matrix 
        factorization model (rank)
        
    verbose : int (optional)
        verbose level of the mode, 0 for no verbose, 1 for verbose

    random_state : int (optional)
        random seed for non-deterministic behaviours in the class
    """

    def __init__(self, model_id, n_users, n_movies, k, verbose = 0, random_state = 1):
        super().__init__(model_id = model_id, n_users = n_users, n_movies = n_movies, verbose = verbose, random_state=random_state)
        self.k = k  
        self.model_name = "ALS"
        self.fitted = False
        
    def fit(self, X, y, W, epochs = 10, λ = 0.1, test_size = 0, normalization = 'zscore', imputation = 'zeros', n_jobs = -1):
        """
        Fit the decomposing matrix U and V using ALS optimization algorithm.

        Parameters        
        ----------
        X : np.array(N_USERS, N_MOVIES)
            input matrix

        y : Ignored
            not used, present for API consistency by convention.

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        epochs : int
            number of iterations to train the algorithm

        λ : float
            regularization term for item/user latent factors

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference

        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        
        n_jobs : int (optional)
            number of cores that can be used for parallel optimization;
            set to -1 to use all the available cores in the machine
        """

        self.λ = λ
        self.epochs = epochs
        self.imputation = imputation
        self.normalization = normalization
        X_train, W_train, X_test, W_test = self.train_test_split(X, W, test_size=test_size)
  
        # normalize input matrix
        X_train = self.normalize(X_train, strategy=normalization)

        # impute missing values
        X_train = self.impute_missing_values(X_train, strategy=imputation)

        # perform SVD decomposition
        svd = SVD(0, self.n_users, self.n_movies, self.k)
        svd.fit(X_train, None, W_train)
        U, S, Vt = svd.get_matrices()
        
        # initialize U and V
        self.U = U[:, :self.k]
        self.V = np.dot(S[:self.k, :self.k], Vt[:self.k, :])

        for epoch in range(self.epochs):
            self._als_step(X_train, W_train, n_jobs=n_jobs)
            predictions_train = self.predict(X_train, invert_norm=False)
            predictions_test = self.predict(X_test, invert_norm=True)
            train_rmse = self.score(X_train, predictions_train, W_train)
            val_rmse = self.score(X_test, predictions_test, W_test)
            if self.verbose: print(f"Epoch {epoch+1}, train_rmse: {train_rmse}, val_rmse: {val_rmse}")
            # log rmse
            self.train_rmse.append(train_rmse)
            self.validation_rmse.append(val_rmse)

    
    def predict(self, X, invert_norm=True):
        pred = np.dot(self.U, self.V)
        if invert_norm:
            pred = self.invert_normalization(pred)
        return pred


    def fit_transform(self, X, y, W, epochs = 10, λ = 0.1, test_size = 0, normalization = 'zscore', imputation = 'zeros', n_jobs = -1, invert_norm = True):
        """
        Fit data and return predictions on the same matrix.

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        epochs : int
            number of iterations to train the algorithm

        λ : float
            regularization term for item/user latent factors

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference

        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        
        n_jobs : int (optional)
            number of cores that can be used for parallel optimization;
            set to -1 to use all the available cores in the machine
        """

        self.fit(X, y, W, epochs, λ, test_size=test_size, normalization=normalization, imputation=imputation, n_jobs=n_jobs)
        pred = self.predict(X, invert_norm=invert_norm)
        return pred


    def _als_step(self, X, W, n_jobs):
        """
        Alternating Least Square optimization step.
        """
        # parallel implementation of the loops
        if n_jobs == -1 and (platform == "linux" or platform == "linux2"): num_cores = len(os.sched_getaffinity(0))
        elif n_jobs == -1: num_cores = os.cpu_count() 
        else: num_cores = n_jobs

        inputs = enumerate(W)
        def optimization(i, Wi):
            A = np.dot(self.V, np.dot(np.diag(Wi), self.V.T)) + self.λ * np.eye(self.k)
            B = np.dot(self.V, np.dot(np.diag(Wi), X[i].T))
            return np.linalg.solve(A, B).T

        result = Parallel(n_jobs=num_cores)(delayed(optimization)(i, np.copy(Wi)) for i, Wi in inputs)
        self.U = np.stack(result, axis=0)

        inputs = enumerate(W.T)
        def optimization(j, Wj):
            A = np.dot(self.U.T, np.dot(np.diag(Wj), self.U)) + self.λ * np.eye(self.k)
            B = np.dot(self.U.T, np.dot(np.diag(Wj), X[:, j]))
            return np.linalg.solve(A, B)
        result = Parallel(n_jobs=num_cores)(delayed(optimization)(j, Wj) for j, Wj in inputs)
        self.V = np.stack(result, axis=1)


    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "epochs" : self.epochs,
                "rank" : self.k,
                "regularization" : self.λ,
                "normalization" : self.normalization,
                "imputation" : self.imputation
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")



######################
###      NMF       ###
######################

class NMF(BaseModel):
    """
    NMF model
    ---------
    
    Train a dimensionality reduction model using Non-negative Matrix Factorization from scikit-learn.
    
    Parameters
    ----------
    model_id : int
        model identification number

    n_users : int
        rows of the input matrix

    n_movies : int
        columns of the input matrix

    k : int
        number of latent factors to use in matrix dimensionality reduction (rank)
        
    verbose : int (optional)
        verbose level of the mode, 0 for no verbose, 1 for verbose

    random_state : int (optional)
        random seed for non-deterministic behaviours in the class
    """

    def __init__(self, model_id, n_users, n_movies, k, verbose = 0, random_state=42):
        super().__init__(model_id = model_id, n_users=n_users, n_movies=n_movies, verbose = verbose, random_state=random_state)
        self.k = k  
        self.model_name = "NMF"
        
    def fit(self, X, y, W, test_size = 0, imputation = None, iter = 500):
        """
        Fit the decomposing matrix U and V using ALS optimization algorithm.

        Parameters        
        ----------
        X : np.array(N_USERS, N_MOVIES)
            input matrix

        y : Ignored
            not used, present for API consistency by convention.

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference
        
        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        """
        self.imputation = imputation
        self.iter = iter
        
        X_train, W_train, X_test, W_test = self.train_test_split(X, W, test_size)

        # impute missing values
        if imputation is not None:
            X_train = self.impute_missing_values(X_train, strategy=imputation)

        self.model = NMF_sl(n_components = self.k, max_iter=iter, random_state=self.random_state, init="nndsvd")
        if self.verbose: print("Fitting model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.U = self.model.fit_transform(X_train)
            self.V = self.model.components_
        
        predictions = self.U.dot(self.V)
        # log training and validation rmse
        train_rmse = self.score(X_train, predictions, W_train)
        val_rmse = self.score(X_test, predictions, W_test)
        if self.verbose: print(f"NMF training rmse: {train_rmse}, val rmse: {val_rmse}")
        self.train_rmse.append(train_rmse)
        self.validation_rmse.append(val_rmse)
        

    def predict(self, X): 
        return self.U.dot(self.V)


    def fit_transform(self, X, y, W, test_size = 0, imputation = 'zeros', iter = 500):
        """
        Fit data and return predictions on the same matrix.

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference
        
        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        
        invert_norm : bool
            boolean flag to invert the normalization of the predictions
            set to False if the input data were not normalized
        """

        self.fit(X, y, W, test_size, imputation, iter)
        return self.predict(X)
    
    
    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "rank" : self.k,
                "imputation" : self.imputation,
                "iter" : self.iter
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")
            



######################
###      BFM       ###
######################

class BFM(BaseModel):
    """
    BFM model
    ---------
    
    Train a dimensionality reduction model using a Bayesian Factorization Machine from the myfm library.
    
    Parameters
    ----------
    model_id : int
        model identification number

    n_users : int
        rows of the input matrix

    n_movies : int
        columns of the input matrix

    k : int
        number of latent factors to use in matrix dimensionality reduction (rank)
        
    verbose : int (optional)
        verbose level of the mode, 0 for no verbose, 1 for verbose

    random_state : int (optional)
        random seed for non-deterministic behaviours in the class
    """

    def __init__(self, model_id, n_users, n_movies, k, verbose = 0, random_state=42, with_ord=False, with_iu=False, with_ii=False):
        super().__init__(model_id = model_id, n_users=n_users, n_movies=n_movies, verbose = verbose, random_state=random_state)
        self.k = k
        self.model_name = "BFM"
        self.with_ord = with_ord
        self.with_iu = with_iu
        self.with_ii = with_ii

        
    def fit(self, X, y, W, data, test_size = 0, iter = 500):
        """
        Fit the decomposing matrix U and V using ALS optimization algorithm.

        Parameters        
        ----------
        X : np.array(N_USERS, N_MOVIES)
            input matrix

        y : Ignored
            not used, present for API consistency by convention.

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference
        
        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        """
        self.iter = iter

        # Unpack and concat vectors
        users, movies, predictions = data
        ump = np.column_stack((np.array(users), np.array(movies), np.array(predictions)))

        if test_size > 0.001:
            train, test = train_test_split(ump, test_size=test_size, random_state=self.random_state)
            X_test = test[:, :2]
            y_test = test[:, 2]
        else:
            train = ump

        X_train = train[:, :2]
        y_train = train[:, 2]

        # index "0" is reserved for unknown ids.
        user_to_index = defaultdict(lambda : 0, { uid: i+1 for i,uid in enumerate(np.unique(X_train[:, 0])) })
        movie_to_index = defaultdict(lambda: 0, { mid: i+1 for i,mid in enumerate(np.unique(X_train[:, 1])) })
        USER_ID_SIZE = len(user_to_index) + 1
        MOVIE_ID_SIZE = len(movie_to_index) + 1

        movie_vs_watched = dict()
        user_vs_watched = dict()
        for row in X_train:
            user_id = row[0]
            movie_id = row[1]
            movie_vs_watched.setdefault(movie_id, list()).append(user_id)
            user_vs_watched.setdefault(user_id, list()).append(movie_id)

        # given user/movie ids, add additional infos and return it as sparse
        def augment_user_id(user_ids):
            Xs = []
            X_uid = sps.lil_matrix((len(user_ids), USER_ID_SIZE))
            for index, user_id in enumerate(user_ids):
                X_uid[index, user_to_index[user_id]] = 1
            Xs.append(X_uid)
            if self.with_iu:
                X_iu = sps.lil_matrix((len(user_ids), MOVIE_ID_SIZE))
                for index, user_id in enumerate(user_ids):
                    watched_movies = user_vs_watched.get(user_id, [])
                    normalizer = 1 / max(len(watched_movies), 1) ** 0.5
                    for uid in watched_movies:
                        X_iu[index, movie_to_index[uid]] = normalizer
                Xs.append(X_iu)
            return sps.hstack(Xs, format='csr')

        def augment_movie_id(movie_ids):
            Xs = []
            X_movie = sps.lil_matrix((len(movie_ids), MOVIE_ID_SIZE))
            for index, movie_id in enumerate(movie_ids):
                X_movie[index, movie_to_index[movie_id]] = 1
            Xs.append(X_movie)
            if self.with_ii:
                X_ii = sps.lil_matrix((len(movie_ids), USER_ID_SIZE))
                for index, movie_id in enumerate(movie_ids):
                    watched_users = movie_vs_watched.get(movie_id, [])
                    normalizer = 1 / max(len(watched_users), 1) ** 0.5
                    for uid in watched_users:
                        X_ii[index, user_to_index[uid]] = normalizer
                Xs.append(X_ii)
            return sps.hstack(Xs, format='csr')
        
        train_uid_unique, train_uid_index = np.unique(X_train[:, 0], return_inverse=True)
        train_mid_unique, train_mid_index = np.unique(X_train[:, 1], return_inverse=True)
        user_data_train = augment_user_id(train_uid_unique)
        movie_data_train = augment_movie_id(train_mid_unique)

        test_uid_unique, test_uid_index = np.unique(X_test[:, 0], return_inverse=True)
        test_mid_unique, test_mid_index = np.unique(X_test[:, 1], return_inverse=True)
        user_data_test = augment_user_id(test_uid_unique)
        movie_data_test = augment_movie_id(test_mid_unique)

        block_user_train = RelationBlock(train_uid_index, user_data_train)
        block_movie_train = RelationBlock(train_mid_index, movie_data_train)
        block_user_test = RelationBlock(test_uid_index, user_data_test)
        block_movie_test = RelationBlock(test_mid_index, movie_data_test)

        if self.with_ord:
            self.model = myfm.MyFMOrderedProbit(rank=self.k, random_seed=self.random_state)
        else:
            self.model = myfm.MyFMRegressor(rank=self.k, random_seed=self.random_state)

        # Ordinal classification: shift ratings from 1 -> 5 to 0 -> 4 since classes start at 0
        if self.with_ord:
            y_train = y_train - 1

        self.model.fit(None, y_train, n_iter=self.iter, X_rel=[block_user_train, block_movie_train])

        if test_size > 0.001:
            y_pred = self.predict([block_user_test, block_movie_test])
            # log only validation rmse, we have no training rmse
            val_rmse = self.score(y_test, y_pred)
            if self.verbose: print(f"BFM val rmse: {val_rmse}")
            self.validation_rmse.append(val_rmse)


    def predict(self, X): 
        if self.with_ord:
            ordinal_probs = self.model.predict_proba(None, X)
            ratings = ordinal_probs.dot(np.arange(1, 6))
        else:
            ratings = self.model.predict(None, X)
        return ratings


    def fit_transform(self, X, y, W, data, test_size = 0, iter = 500):
        """
        Fit data and return predictions on the same matrix.

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix

        W : np.array(N_USERS, N_MOVIES)
            mask matrix for observed entries; True entries in the mask corresponds
            to observed values, False entries to unobserved values

        test_size : float [0,1] (optional)
            percentage of the training data to be used as validation split;
            set to 0 when the model has to be used for inference
        
        normalization : str or None
            strategy to be used to normalize the data, None for no normalization
        
        invert_norm : bool
            boolean flag to invert the normalization of the predictions
            set to False if the input data were not normalized
        """

        raise NotImplementedError()
    
    
    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "rank" : self.k,
                "iter" : self.iter,
                "ordinal" : self.with_ord,
                "implicit user info" : self.with_iu,
                "implicit movie info" : self.with_ii
            },
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")


    @staticmethod
    def score(y_true, y_pred):
        """
        Compute the Root Mean Squared Error between two numeric vectors.
        Overridden to use normal vectors without masks.

        Parameters
        ----------
        y_true : np.ndarray
            ground truth array
            
        y_pred : np.ndarray
            predictions array

        """
        rmse = ((y_true - y_pred) ** 2).mean() ** .5
        return rmse
            