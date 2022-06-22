import numpy as np
from models.base_model import BaseModel
from models.svd import SVD
import json


class ALS(BaseModel):
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    epochs : int
        number of iterations to train the algorithm
        
    k : int
        number of latent factors to use in matrix 
        factorization model (rank)
        
    λ : float
        regularization term for item/user latent factors

    verbose : int
              verbose level of the mode, 0 for no verbose, 1 for verbose
    """

    def __init__(self, model_id, n_users, n_movies, epochs, k, λ, verbose = 0, test_size = 0, random_state = 42):
        super().__init__(model_id = model_id, n_users = n_users, n_movies = n_movies, verbose = verbose, test_size = test_size, random_state=random_state)
        self.λ = λ
        self.epochs = epochs
        self.k = k  
        self.model_name = "ALS"

        
    def fit(self, X, y):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
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
        """
        self.fit(X, y)
        pred = self.predict(X, invert_norm=True)
        return pred


    def _als_step(self):
        """
        Alternating Least Square optimization step
        """
        # # NOT OPTIMAL ALS
        # # USE ONLY FOR DEBUG BECAUSE IT'S FASTER TO RUN

        # # solve u* as linear system Au* = B
        # A = np.dot(self.V, self.V.T) + self.λ * np.eye(self.k)
        # B = np.dot(self.V, self.X_train.T)
        # self.U = np.linalg.solve(A,B).T

        # # solve v* as linear system Av* = B
        # A = np.dot(self.U.T, self.U) + self.λ * np.eye(self.k)
        # B = np.dot(self.U.T, self.X_train)
        # self.V = np.linalg.solve(A, B)

        # optimize matrix U
        for i, Wi in enumerate(self.W_train):
            A = np.dot(self.V, np.dot(np.diag(Wi), self.V.T)) + self.λ * np.eye(self.k)
            B = np.dot(self.V, np.dot(np.diag(Wi), self.X_train[i].T))
            self.U[i] = np.linalg.solve(A, B).T
        # optimize matrix V
        for j, Wj in enumerate(self.W_train.T):
            A = np.dot(self.U.T, np.dot(np.diag(Wj), self.U)) + self.λ * np.eye(self.k)
            B = np.dot(self.U.T, np.dot(np.diag(Wj), self.X_train[:, j]))
            self.V[:,j] = np.linalg.solve(A, B)

    def predict(self, X, invert_norm=True):
        """
        Predict ratings for every user and item
        """
        pred = np.dot(self.U, self.V)
        if invert_norm:
            pred = self.invert_normalization(pred)
        return pred

    def log_model_info(self, path = "./log/", format = "json"):
        """
        Log model and training information
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