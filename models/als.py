import numpy as np
from models.base_model import BaseModel
from models.svd import SVD
import json
from joblib import Parallel, delayed
import multiprocessing

class ALS(BaseModel):
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    model_id : int
        model identification number

    n_users : int
        rows of the input matrix

    n_movies : int
        columns of the input matrix

    epochs : int
        number of iterations to train the algorithm
        
    k : int
        number of latent factors to use in matrix 
        factorization model (rank)
        
    λ : float
        regularization term for item/user latent factors

    verbose : int (optional)
        verbose level of the mode, 0 for no verbose, 1 for verbose

    test_size : float [0,1] (optional)
        percentage of the training data to be used as validation split;
        set to 0 when the model has to be used for inference
    
    random_state : int (optional)
        random seed for non-deterministic behaviours in the class
    
    n_jobs : int (optional)
        number of cores that can be used for parallel optimization;
        set to -1 to use all the available cores in the machine
    """

    def __init__(self, model_id, n_users, n_movies, epochs, k, λ, verbose = 0, test_size = 0, random_state = 42, n_jobs = -1):
        super().__init__(model_id = model_id, n_users = n_users, n_movies = n_movies, verbose = verbose, test_size = test_size, random_state=random_state)
        self.λ = λ
        self.epochs = epochs
        self.k = k  
        self.model_name = "ALS"
        self.n_jobs = n_jobs

        
    def fit(self, X, y):
        """
        Fit the decomposing matrix U and V using ALS optimization algorithm

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix
        """

        self.X_train, self.W_train, self.X_test, self.W_test = self.train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
  
        # normalize input matrix
        self.normalize()

        # impute missing values
        self.impute_missing_values()

        # initialize U and V matrices with SVD matrices
        svd = SVD(0, self.n_users, self.n_movies, self.k)
        svd.fit(X, y)
        U, S, Vt = svd.get_matrices()
        self.U = U[:, :self.k]
        self.V = np.dot(S[:self.k, :self.k], Vt[:self.k, :])
  
        for epoch in range(self.epochs):
            self._als_step()
            predictions_train = self.predict(self.X_train, invert_norm=False)
            predictions_test = self.predict(self.X_test, invert_norm=True)
            train_rmse = self.score(self.X_train, predictions_train, self.W_train)
            val_rmse = self.score(self.X_test, predictions_test, self.W_test)
            if self.verbose:    
                print(f"Epoch {epoch+1}, train_rmse: {train_rmse}, val_rmse: {val_rmse}")
            # log rmse
            self.train_rmse.append(train_rmse)
            self.validation_rmse.append(val_rmse)


    def fit_transform(self, X, y):
        """
        Fit data and return predictions on the same matrix

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix
        """
        self.fit(X, y)
        pred = self.predict(X, invert_norm=True)
        return pred


    def _als_step(self):
        """
        Alternating Least Square optimization step
        """
        # parallel implementation of the loops
        if self.n_jobs == -1: num_cores = multiprocessing.cpu_count()
        else: num_cores = self.n_jobs

        inputs = enumerate(self.W_train)
        def optimization(i, Wi):
            A = np.dot(self.V, np.dot(np.diag(Wi), self.V.T)) + self.λ * np.eye(self.k)
            B = np.dot(self.V, np.dot(np.diag(Wi), self.X_train[i].T))
            return np.linalg.solve(A, B).T

        result = Parallel(n_jobs=num_cores)(delayed(optimization)(i, Wi) for i, Wi in inputs)
        self.U = np.stack(result, axis=0)

        inputs = enumerate(self.W_train.T)
        def optimization(j, Wj):
            A = np.dot(self.U.T, np.dot(np.diag(Wj), self.U)) + self.λ * np.eye(self.k)
            B = np.dot(self.U.T, np.dot(np.diag(Wj), self.X_train[:, j]))
            return np.linalg.solve(A, B)
        result = Parallel(n_jobs=num_cores)(delayed(optimization)(j, Wj) for j, Wj in inputs)
        self.V = np.stack(result, axis=1)

    def predict(self, X, invert_norm=True):
        """
        Predict ratings for every user and item

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix
            this parameter is not really used, as the prediction is performed using U and V
            computed during the fit; the parameter is there for compatibility
        """
        pred = np.dot(self.U, self.V)
        if invert_norm:
            pred = self.invert_normalization(pred)
        return pred

    def log_model_info(self, path = "./log/", format = "json"):
        """
        Log model and training information

        Parameters
        ----------
        path : str (optional)
            path to the folder where the logs have to be stored

        format : str (optional)
            format of the log file, supported formats: ['json'] 
        """
        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "epochs" : self.epochs,
                "rank" : self.k,
                "regularization" : self.λ
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")