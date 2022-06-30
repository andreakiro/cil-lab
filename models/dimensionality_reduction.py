import numpy as np
from models.base_model import BaseModel
import json

class SVD(BaseModel):
    """
    SVD model
    ---------
    
    Train a dimensionality reduction model using SVD.
    
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
        self.model_name = "SVD"
        self.fitted = False
        
    def fit(self, X, y, W, test_size = 0, normalization = "zscore"):
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
            technique to be used to normalize the data, None for no normalization
        """

        X_train, W_train, X_test, W_test = self.train_test_split(X, W, test_size)

        # the number of singular values must be lower than
        # the lowest dimension of the matrix
        num_singular_values = min(self.n_users, self.n_movies)
        assert (self.k <= num_singular_values)

        # normalize input matrix
        if normalization:
            X_train = self.normalize(X_train, technique=normalization)

        # impute missing values
        X_train = self.impute_missing_values(X_train)

        # decompose the original matrix
        self.U, Σ, self.Vt = np.linalg.svd(X_train, full_matrices=False)

        # keep the top k components
        self.S = np.zeros((self.n_movies, self.n_movies)) 
        self.S[:self.k, :self.k] = np.diag(Σ[:self.k])
        self.fitted = True
        
        # log training and validation rmse
        train_rmse = self.score(X_train, self.predict(X_train, invert_norm=False), W_train)
        val_rmse = self.score(X_test, self.predict(X_test, invert_norm=True), W_test)
        self.train_rmse.append(train_rmse)
        self.validation_rmse.append(val_rmse)
        

    def predict(self, X, invert_norm = True):
        assert self.fitted
        pred = self.U.dot(self.S).dot(self.Vt)
        if invert_norm:
            pred = self.invert_normalization(pred)
        return pred


    def fit_transform(self, X, y, W, test_size = 0, normalization = "zscore", invert_norm = True):
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
            technique to be used to normalize the data, None for no normalization
        
        invert_norm : bool
            boolean flag to invert the normalization of the predictions
            set to False if the input data were not normalized
        """

        self.fit(X, y, W, test_size, normalization)
        return self.predict(X, invert_norm)

    
    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "rank" : self.k,
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")