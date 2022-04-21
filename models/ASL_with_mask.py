import numpy as np
from sklearn.metrics import mean_squared_error

class ASL_with_mask:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix. Use only the intial given entries
    at each iteration to update the U and V vectors.
    
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
        
    def fit(self, data, W, U, V, verbose=False):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_movies = data.shape
        self.U = U
        self.V = V
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.train_mse_record = []   
        for epoch in range(self.n_iters):
            self.U, self.V = self._als_step(data, W, self.U, self.V)
            predictions = self.predict()
            train_mse = self.compute_mse(data, predictions)
            if verbose:
                print(f"Epoch {epoch}, train_mse = {train_mse}")
            self.train_mse_record.append(train_mse)
    
    def _als_step(self, data, W, U, V):
        """alternating least square step"""
        
        
        for i, Wi in enumerate(W):
            U[i] = np.linalg.solve(np.dot(V, np.dot(np.diag(Wi), V.T)) + self.λ * np.eye(self.k),
                                       np.dot(V, np.dot(np.diag(Wi), data[i].T))).T

        for j, Wj in enumerate(W.T):
            V[:,j] = np.linalg.solve(np.dot(U.T, np.dot(np.diag(Wj), U)) + self.λ * np.eye(self.k),
                                     np.dot(U.T, np.dot(np.diag(Wj), data[:, j])))
        
        return U, V
    
    def predict(self):
        """predict ratings for every user and item"""
        return self.U.dot(self.V)
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse