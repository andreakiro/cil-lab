import numpy as np
from sklearn.metrics import mean_squared_error

class ASL:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    k : int
        number of latent factors to use in matrix 
        factorization model (rank)
        
    λ : float
        regularization term for item/user latent factors
    """

    def __init__(self, n_iters, k, λ):
        self.λ = λ
        self.n_iters = n_iters
        self.k = k  
        
    def fit(self, data, U, V):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings. U must be the
        transpose of the matrix given by SVD
        """
        self.n_user, self.n_movies = data.shape
        self.U = U
        self.V = V
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.train_mse_record = []   
        for _ in range(self.n_iters):
            self.U = self._als_step(data.T, self.U, self.V)
            self.V = self._als_step(data, self.V, self.U) 
            predictions = self.predict()
            train_mse = self.compute_mse(data, predictions)
            self.train_mse_record.append(train_mse)
    
    def _als_step(self, data, to_solve, fixed):
        """alternating least square step"""
        to_solve = np.linalg.solve(fixed.dot(fixed.T) + self.λ * np.eye(self.k),
                                   fixed.dot(data))
        return to_solve
    
    def predict(self):
        """predict ratings for every user and item"""
        return self.U.T.dot(self.V)
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse