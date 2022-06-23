from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn.metrics import mean_squared_error

class BaseModel(ABC):

    def __init__(self, model_id, n_users, n_movies, verbose = 0, random_state = 42):
        self.model_id = model_id
        # set number of users and movies
        self.n_users = n_users
        self.n_movies = n_movies
        # set training error log
        self.train_rmse = []
        self.validation_rmse = []
        # set verbose level
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, X, y, **kwargs):
        pass

    @abstractmethod
    def log_model_info(self, path = "./log/", format = "json"):
        pass


    def train_test_split(self, X, W, test_size):
        """
        Create training and test matrices.
        """
        np.random.seed(seed=self.random_state)
        if test_size != 0:
            # initialize training and test vectors
            W_train = np.copy(W)
            W_test = np.copy(W)
            X_train = np.copy(X)
            X_test = np.copy(X)
            # get training and test masks
            W_test[W] = np.random.random(*W[W].shape) > test_size
            W_train = np.multiply(W, ~W_test)
            # mask data values
            X_train[~W_train] = np.nan
            X_test[~W_test] = np.nan
        else:
            X_train, W_train = X, W
            X_test, W_test = np.array([]), np.array([])
            
        return X_train, W_train, X_test, W_test
    

    def normalize(self, axis = 0, technique = "zscore"):
        """
        Normalize the input matrix
        """
        assert self.X_train is not None
        if technique == "zscore":
            # save columns mean and std to invert z-score
            self.μ = np.array(np.broadcast_to(np.nanmean(self.X_train, axis=axis)[:], (self.X_train.shape)))
            self.σ = np.array(np.broadcast_to(np.nanstd(self.X_train, axis=axis)[:], (self.X_train.shape)))
            # normalize data using z-score
            self.X_train = stats.zscore(self.X_train, axis=axis, nan_policy='omit')
        else:
            raise ValueError(f"Technique '{technique}' is not valid.")


    def invert_normalization(self, M, technique = "zscore"):
        """
        Invert normalization (mostly for prediction results)
        """
        # normalization was not done in the first place
        assert self.X_train is not None
        if technique == "zscore":
            return np.multiply(M, self.σ) + self.μ 
        else:
            raise ValueError(f"Technique '{technique}' is not valid.")


    def impute_missing_values(self, strategy="zero"):
        """
        Impute missing (unobserved) values in the input matrix
        """
        assert self.X_train is not None
        # impute values using sklearn imputation
        from sklearn.impute import SimpleImputer
        if strategy == "zero":
            self.X_train = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(self.X_train)
        else:
            raise ValueError(f"Strategy '{strategy}' is not valid.")


    @staticmethod
    def score(y_true, y_pred, y_mask):
        """
        Compute the Root Mean Squared Error between two numeric vectors

        Parameters
        ----------
        y_true : np.ndarray
            ground truth array
            
        y_pred : np.ndarray
            predictions array
            
        y_mask : np.ndarray
            mask array for observed entries
        """
        # parameters must be numpy array
        assert isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)
    
        # flatten the matrices
        y_true, y_pred, y_mask = y_true.flatten(), y_pred.flatten(), y_mask.flatten()
        
        if y_mask.shape[0] == 0 or y_true[y_mask].shape[0] == 0: return math.nan

        rmse = np.sqrt(mean_squared_error(y_true[y_mask], y_pred[y_mask]))
        return rmse
