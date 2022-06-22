import numpy as np
from models.base_model import BaseModel
import json

class SVD(BaseModel):

    def __init__(self, model_id, n_users, n_movies, k, verbose = 0, test_size = 0, random_state=42):
        super().__init__(model_id = model_id, n_users=n_users, n_movies=n_movies, verbose = verbose, test_size = test_size, random_state=random_state)
        self.k = k  
        self.model_name = "SVD"
        
    def fit(self, X, y):

        self.X_train, self.W_train, self.X_test, self.W_test = self.train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # the number of singular values must be lower than
        # the lowest dimension of the matrix
        num_singular_values = min(self.n_users, self.n_movies)
        assert (self.k <= num_singular_values)

        # normalize input matrix
        self.normalize()

        # impute missing values
        self.impute_missing_values()

        # decompose the original matrix
        self.U, Σ, self.Vt = np.linalg.svd(self.X_train, full_matrices=False)

        # keep the top k components
        self.S = np.zeros((self.n_movies, self.n_movies)) 
        self.S[:self.k, :self.k] = np.diag(Σ[:self.k])

        # log training and validation rmse
        train_rmse = self.score(self.X_train, self.predict(self.X_train, invert_norm=False), self.W_train)
        val_rmse = self.score(self.X_test, self.predict(self.X_test, invert_norm=True), self.W_test)
        self.train_rmse.append(train_rmse)
        self.validation_rmse.append(val_rmse)

    def predict(self, X, invert_norm = True):
        pred = self.U.dot(self.S).dot(self.Vt)
        if invert_norm:
            pred = self.invert_normalization(pred)
        return pred

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def get_matrices(self):
        return self.U, self.S, self.Vt
    
    def log_model_info(self, path = "./log/", format = "json"):
        """
        Log model and training information
        """
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