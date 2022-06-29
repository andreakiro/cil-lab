#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matrix completion algorithms based on matrix factorization.

Algorithms implemented in this module:
  - Alternate Least-Square (ALS) algorithm
  - Non-negative Matrix Factorization (NMF) algorithm
"""

import numpy as np
import pandas as pd
from models.base_model import BaseModel
from models.dimensionality_reduction import SVD
import json
from joblib import Parallel, delayed
import os
from sklearn.decomposition import NMF as NMF_sl
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
from sys import platform
import myfm

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

    def __init__(self, model_id, n_users, n_movies, k, verbose = 0, random_state=42):
        super().__init__(model_id = model_id, n_users=n_users, n_movies=n_movies, verbose = verbose, random_state=random_state)
        self.k = k
        self.model_name = "BFM"
        
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

        train, test = train_test_split(ump, test_size=test_size, random_state=self.random_state)
        X_train = train[:, :2]
        y_train = train[:, 2]
        X_test = test[:, :2]
        y_test = test[:, 2]

        # One-Hot Encoding
        ohe = OneHotEncoder(handle_unknown='ignore')
        X_train = ohe.fit_transform(X_train)
        X_test = ohe.transform(X_test)

        self.model = myfm.MyFMRegressor(rank=self.k, random_seed=self.random_state)

        if self.verbose: print("Fitting model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X_train, y_train, n_iter=self.iter)

        y_pred = self.predict(X_test)
        
        # log only validation rmse, we have no training rmse
        val_rmse = self.score(y_test, y_pred)
        if self.verbose: print(f"BFM val rmse: {val_rmse}")
        self.validation_rmse.append(val_rmse)
        

    def predict(self, X): 
        return self.model.predict(X)


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

        #self.fit(X, y, W, data, test_size=test_size, iter=iter)
        #return self.predict(X)
        raise NotImplementedError()
    
    
    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "rank" : self.k,
                "iter" : self.iter
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
            