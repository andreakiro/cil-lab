from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

class BaseModel(ABC):
    """
    Template base model.
    """

    def __init__(self, model_id, n_users, n_movies, verbose = 0, random_state = 1):
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
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Predict ratings for every user and item;
        fit method must be called before this, otherwise an exception will be raised.

        Parameters
        ----------
        X : Ignored
            the prediction is always performed on the X used to fit the model
        
        invert_norm : bool
            boolean flag to invert the normalization of the predictions
            set to False if the input data were not normalized
        """
        pass

    @abstractmethod
    def fit_transform(self, X, y, **kwargs):
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
        """
        pass

    @abstractmethod
    def log_model_info(self, path = "./log/", format = "json"):
        """
        Log model and training information.

        Parameters
        ----------
        path : str (optional)
            path to the folder where the logs have to be stored

        format : str (optional)
            format of the log file, supported formats: ['json'] 
        """
        pass


    def train_test_split(self, X, W, test_size):
        """
        Create training and test matrices.
        """
        if test_size != 0:
            # initialize training and test vectors
            W_train = np.copy(W)
            W_test = np.copy(W)
            X_train = np.copy(X)
            X_test = np.copy(X)
            # get training and test masks
            np.random.seed(seed=self.random_state)
            W_train[W] = np.random.random(*W[W].shape) > test_size
            W_test = np.multiply(W, ~W_train)
            # mask data values
            X_train[~W_train] = np.nan
            X_test[~W_test] = np.nan
        else:
            X_train, W_train = X, W
            X_test, W_test = np.array([]), np.array([])
            
        return X_train, W_train, X_test, W_test
    

    def normalize(self, X, axis = 0, strategy = "zscore"):
        """
        Normalize the input matrix.
        """
        if strategy == "zscore":
            # save columns mean and std to invert z-score
            self.μ = np.array(np.broadcast_to(np.nanmean(X, axis=axis)[:], (X.shape)))
            self.μ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(self.μ)
            self.σ = np.array(np.broadcast_to(np.nanstd(X, axis=axis)[:], (X.shape)))
            self.σ = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=1).fit_transform(self.σ)
            # normalize data using z-score
            X = stats.zscore(X, axis=axis, nan_policy='omit')
        elif strategy == "min_max":
            self.min_val = np.nanmin(X)
            self.max_val = np.nanmax(X)
            X = (X-self.min_val)/(self.max_val-self.min_val)
        else:
            raise ValueError(f"Strategy '{strategy}' is not valid.")
        return X


    def invert_normalization(self, M, strategy = "zscore"):
        """
        Invert normalization (mostly for prediction results).
        """
        if strategy == "zscore":
            return np.multiply(M, self.σ) + self.μ 
        elif strategy == "min_max":
            return M*(self.max_val - self.min_val) + self.min_val
        else:
            raise ValueError(f"Strategy '{strategy}' is not valid.")


    def impute_missing_values(self, X, strategy="zeros"):
        """
        Impute missing (unobserved) values in the input matrix.
        """
        # impute values using sklearn imputation
        if strategy == "zeros":
            X = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(X)
        elif strategy == "mean":
            X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
        elif strategy == "most_frequent":
            X = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(X)
        elif strategy == "median":
            X = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(X)
        else:
            raise ValueError(f"Strategy '{strategy}' is not valid.")
        return X


    @staticmethod
    def score(y_true, y_pred, y_mask):
        """
        Compute the Root Mean Squared Error between two numeric vectors.

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
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray) and isinstance(y_mask, np.ndarray)
    
        # flatten the matrices
        y_true, y_pred, y_mask = y_true.flatten(), y_pred.flatten(), y_mask.flatten()
        
        if y_mask.shape[0] == 0 or y_true[y_mask].shape[0] == 0: return math.nan

        rmse = np.sqrt(mean_squared_error(y_true[y_mask], y_pred[y_mask]))
        return rmse
