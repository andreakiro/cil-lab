from models.svd import SVD
from models.als import ALS
from models.base_model import BaseModel
import numpy as np 
import warnings
from sklearn.exceptions import ConvergenceWarning
import json

class BCA(BaseModel):
    """
    Block Completion Algorithm from C. Strohmeier and D. Needell, "Clustering of Nonnegative Data and an Application to Matrix Completion" 
    ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 8349-8353.
    """

    def __init__(self, model_id, n_users, n_movies, k, n_cluster, verbose = 0,  random_state = 42):
        super().__init__(model_id = model_id, n_users = n_users, n_movies = n_movies, verbose = verbose, random_state=random_state) 
        self.model_name = "BCA"
        self.k = k
        self.n_cluster = n_cluster

    def fit(self, X, y, W, test_size = 0):
        """
        Fit the decomposing matrix U and V using ALS optimization algorithm

        Parameters
        ----------
        X : pd.Dataframe.Column
            dataframe column containing coordinates of the observed entries in the matrix

        y : int 
            values of the observed entries in the matrix
        """

        X_train, W_train, X_test, W_test = self.train_test_split(X, W, test_size=test_size)

        # first basic completion of the matrix using ALS
        if self.verbose: print("Fitting base SVD model...")
        model = ALS(1, 10000, 1000, k=self.k)
        basic = model.fit_transform(X_train, None, W_train, test_size=0)
        basic = np.clip(basic, 1, 5)

        # apply NMF clustering
        if self.verbose: print("NMF clustering...")
        clusters = self.__nmf_clustering(basic)

        # create matrix
        self.prediction = np.copy(X_train)

        # create block for each cluster and complete the block with ALS
        if self.verbose: print("Fitting blocks with ALS...")
        for c in range(self.n_cluster):
            print(f"    Fitting cluster {c}...")
            block = X_train[clusters == c, :]
            block_W = W[clusters == c, :]
            model = ALS(1, 10000, 1000, k=self.k)
            with warnings.catch_warnings(): 
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.prediction[clusters == c, :] = model.fit_transform(block, None, block_W, epochs=5, test_size=0)
        
        train_rmse = self.score(X_train, self.prediction, W_train)
        val_rmse = self.score(X_test, self.prediction, W_test)
        if self.verbose:    
            print(f"BCA model train_rmse: {train_rmse}, val_rmse: {val_rmse}")
        # log rmse
        self.train_rmse.append(train_rmse)
        self.validation_rmse.append(val_rmse)

        
    def predict(self):
        return self.prediction
    
    def __nmf_clustering(self, X):
        """
        """
        # perform NMF with k hidden features to write X_train = W @ H
        from sklearn.decomposition import NMF
        model = NMF(n_components=self.k, init='nndsvd', max_iter=512, random_state=self.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            W = model.fit_transform(X)
        H = model.components_
        # apply k-means to rows of W
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_cluster).fit(W)
        # return user labels
        return kmeans.labels_


    def fit_transform(self):
        raise NotImplementedError()

    def log_model_info(self, path = "./log/", format = "json"):
        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {     
                "rank" : self.k,
                "clusters" : self.n_cluster
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }
        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")