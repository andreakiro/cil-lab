import numpy as np
from models.base_model import BaseModel
import json
import itertools
from joblib import Parallel, delayed
import os
from sys import platform

class SimilarityMethods(BaseModel):
    """
    Similarity based model
    ---------
    
    Compute missing values using similarity between the items/users or both.
    
    Parameters
    ----------
    model_id : int
        model identification number

    n_users : int
        rows of the input matrix

    n_movies : int
        columns of the input matrix

    similarity_measure : string, either "PCC", "cosine" or "SiGra"
        The method used to compute the similarity.
    
    weighting : string or None, either "weighting", "significance" or "sigmoid"
        The method used to weight the similarity, in order to give less weight to
        users having rated only few common items.

    method: string, either "user", "item" or "both"
        The method used to compute the final prediction. If "user", compute the similarity
        between users only, if "item", between items only and otherwise use both users
        and items.

    use_std: bool (optional)
        Whether we should take into account the standard deviation for the prediction weighting.
        If set to True, we compute the weights of the neighbors using a z-score.
        
    k: int (optional)
        number of nearest neighbors used for the prediction.

    signifiance_threshold: int (optional)
        Only used if weighting is 'significance'. Minimum number of common rated items needed to not have
        a decrease in importance. Needs to be adapted depending on the number of common users/items. For
        user-based similarity, should be around 7, for item-based similarity, around 70. 
    
    statistic_to_use: string, either "mean" or "median"
        Only used if similarity_measure is PCC. The method use to center the data points.
    
    user_weight: int, in interval [0,1] (optional):
        Only used if method is "both": if close to 1, put more weight to the users prediction, if close 
        to 0, put more weight to the items prediction.
    
    n_jobs: int (optional)
        number of cores that can be used for parallel optimization during fitting and prediction.
        set to -1 to use all the available cores in the machine

    verbose : int (optional)
        verbose level of the mode, 0 for no verbose, 1 for verbose

    random_state : int (optional)
        random seed for non-deterministic behaviours in the class

    """
    def __init__(self, model_id, n_users, n_movies, similarity_measure, weighting, method, use_std=False,  k=10, signifiance_threshold=10, statistic_to_use="mean", user_weight=0.5, n_jobs=-1, verbose = 0, random_state=42):
        super().__init__(model_id = model_id, n_users=n_users, n_movies=n_movies, verbose = verbose, random_state=random_state)
        assert similarity_measure == "PCC" or similarity_measure == "cosine" or similarity_measure == "SiGra", "Incorrect similarity_measure, must be either 'PCC', 'cosine' or 'SiGra'"
        assert weighting is None or weighting == "weighting" or weighting == "significance" or weighting == "sigmoid", "Incorrect weighting, must be either None, 'weighting', 'significance' or 'sigmoid'"
        assert method == "user" or method == "item" or method == "both", "Incorrect method, must be either 'user', 'item' or 'both'"

        # SiGra has already some weighting taken into account in its similarity computation
        if similarity_measure == "SiGra":
            weighting = None

        self.similarity_measure = similarity_measure
        self.weighting = weighting
        self.method = method
        self.use_std = use_std
        self.k = k

        if n_jobs == -1 and (platform == "linux" or platform == "linux2"): self.num_cores = len(os.sched_getaffinity(0))
        elif n_jobs == -1: self.num_cores = os.cpu_count() 
        else: self.num_cores = n_jobs

        # Only useful for this model
        if weighting == "significance":
            self.signifiance_threshold = signifiance_threshold
        if similarity_measure == "PCC":
            self.statistic_to_use = statistic_to_use
        if method == "both":
            self.user_weight = user_weight
        
        self.model_name = method + " " + similarity_measure + " similarity with " + weighting + " weighting"
        self.fitted = False

    
    def fit(self, X, y, W, test_size = 0, normalization = None, log_rmse=True):
        """
        Fit the similarity between users/items.

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
        
        log_rmse : bool (optional)
            If set to True, compute the test and validation loss
        """

        X_train, W_train, X_test, W_test = self.train_test_split(X, W, test_size)

        # We used them as neighbors for the prediction, not the test set
        self.X_train = X_train
        self.W_train = W_train
        self.X_test = X_test
        self.W_test = W_test

        # normalize input matrix
        if normalization is not None:
            self.normalization = normalization
            X_train = self.normalize(X_train, technique=normalization)

        # Compute the Similarity/ies:
        self.similarity_users = None
        self.similarity_items = None

        if self.method == "user" or self.method == "both":
            if self.similarity_measure == "PCC":
                self.min_similarity_neighbor = 0
                self.similarity_users = self.__compute_pearson_correlation_coefficient(X_train, user=True)
            elif self.similarity_measure == "cosine":
                self.min_similarity_neighbor = 0.15 # Was chosen by taking almost same number of neighbors as PCC with 0
                self.similarity_users = self.__compute_cosine_similarity(X_train, user=True)
            elif self.similarity_measure == "SiGra":
                self.min_similarity_neighbor = 0.7 # Was chosen by taking almost same number of neighbors as PCC with 0
                self.similarity_users = self.__compute_SigRA(X_train, W_train, user=True)
        if self.method == "item" or self.method == "both":
            if self.similarity_measure == "PCC":
                self.min_similarity_neighbor = 0
                self.similarity_items = self.__compute_pearson_correlation_coefficient(X_train, user=False)
            elif self.similarity_measure == "cosine":
                self.min_similarity_neighbor = 0.15 # Was chosen by taking almost same number of neighbors as PCC with 0
                self.similarity_items = self.__compute_cosine_similarity(X_train, user=False)
            elif self.similarity_measure == "SiGra":
                self.min_similarity_neighbor = 0.7 # Was chosen by taking almost same number of neighbors as PCC with 0
                self.similarity_items = self.__compute_SigRA(X_train, W_train, user=False)
        
        if self.weighting is not None:
            if self.weighting == "weighting":
                self.min_similarity_neighbor = 0.01 # Was chosen by taking almost same number of neighbors as PCC with 0
            else:
                self.min_similarity_neighbor = 0.1 # Was chosen by taking almost same number of neighbors as PCC with 0

            self.similarity_users =  self.__similarity_weighting(self.similarity_users, W_train)
            self.similarity_items =  self.__similarity_weighting(self.similarity_items, W_train)

        self.fitted = True

        # log training and validation rmse
        if log_rmse:
            train_rmse = self.score(X_train, self.predict(W_train, invert_norm=False), W_train)
            val_rmse = self.score(X_test, self.predict(W_test, invert_norm=True if normalization is not None else False), W_test)
            self.train_rmse.append(train_rmse)
            self.validation_rmse.append(val_rmse)
        

    def predict(self, X, invert_norm = True):
        """
        We pass the mask containing the spots that we want our model to predict as X
        """
        assert self.fitted

        mask_to_predict = X
        
        predictions_users, confidence_users = self.__weighted_average_predict(mask_to_predict, self.similarity_users)
        predictions_items, confidence_items = self.__weighted_average_predict(mask_to_predict, self.similarity_items)

        pred = self.__predict_using_users_and_items(predictions_users, predictions_items, confidence_users, confidence_items)

        if invert_norm:
            pred = self.invert_normalization(pred, self.normalization)
        
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
                "similarity_measure": self.similarity_measure,
                "weighting": self.weighting,
                "method": self.method,
                "with std": self.use_std,
                "number nearest neighbors" : self.k,
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }

        if self.weighting == "significance":
            model_info["parameters"]["signifiance_threshold"] = self.signifiance_threshold
        if self.similarity_measure == "PCC":
            model_info["parameters"]["statistic_to_use"] = self.statistic_to_use
        if self.method == "both":
            model_info["parameters"]["user_weight"] = self.user_weight

        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")

    def __compute_cosine_similarity(self, X, user=True):
        '''
        Compute the cosine similarity between every pair of users or items
        
        Parameters
        ----------
        X : np.array(N_USERS, N_MOVIES)
            The matrix with ratings.
            
        user : bool, default True
            a boolean which says whether we want to compute the cosine similarity between every users or 
            between every items.

        Return
        ----------
        similarity: np.array(N_USERS, N_USERS) if user = True else np.array(N_MOVIES, N_MOVIES):
                    The similarity score (between -1 and 1 if X has negative values, where -1 means two vectors 
                    going in the opposite direction or between 0 and 1 if we only have positive values where 0 means
                    orthogonal vectors) for each user-user or item-item pair. The returned matrix is therefore 
                    symmetric.
        '''
        X = np.nan_to_num(X) # Replace Nan by 0 (the dot product will hence be 0, what we want)
        
        if not user:
            X = X.T
        
        similarity = np.zeros((X.shape[0], X.shape[0]))
        all_rows_norm = np.linalg.norm(X, axis=1)

        for i,user in enumerate(X):
            similarity[i, :] = (X@user)/(all_rows_norm*np.linalg.norm(user))
        
        return similarity

    def __compute_pearson_correlation_coefficient(self, X, user=True):
        '''
        Compute the pearson correlation coefficient between every pair of users or items
        
        Parameters
        ----------
        X : np.array(N_USERS, N_MOVIES)
            The matrix with ratings.
            
        user : bool, default True
            a boolean which says whether we want to compute the cosine similarity between every users or 
            between every items.

        Return
        ----------
        similarity: np.array(N_USERS, N_USERS) if user = True else np.array(N_MOVIES, N_MOVIES):
                    The similarity score (between -1 and 1) for each user-user or item-item pair.
                    The returned matrix is therefore symmetric.
        '''
        if not user:
            X = X.T
        
        if self.statistic_to_use == "mean":
            statistic = np.nanmean(X, axis=1)
        elif self.statistic_to_use == "median":
            statistic = np.nanmedian(X, axis=1)
        else:
            raise ValueError(f"{self.statistic_to_use} is not a valid statistic! Should be 'mean' or 'median'")
        
        centered_X = X-statistic.reshape(-1,1)
        
        return self.__compute_cosine_similarity(centered_X, user=True) # Always True since we have already taken the transpose in this method

    def __compute_SigRA(X, W, user=True):
        '''
        Compute the SiGra (https://ieeexplore.ieee.org/document/8250351) similarity between every pair of users or 
        items. Note that this method already uses weighting and hence should not be followed by the 
        similarity_weighting function.
        
        Parameters
        ----------
        X : np.array(N_USERS, N_MOVIES)
            The matrix with ratings.
        
        W : np.array(N_USERS, N_MOVIES):
            The mask containing 1 for rated items and 0 for unrated items.
            
        user : bool, default True
            a boolean which says whether we want to compute the cosine similarity between every users or 
            between every items.

        Return
        ----------
        similarity: np.array(N_USERS, N_USERS) if user = True else np.array(N_MOVIES, N_MOVIES):
                    The similarity score for each user-user or item-item pair.
                    The returned matrix is therefore symmetric.
        '''    
        if not user:
            X = X.T
            W = W.T
        
        similarity = np.zeros((X.shape[0], X.shape[0]))

        number_ratings = np.sum(W, axis=1)
        
        for i, (uw, ux) in enumerate(zip(W, X)):
            use_range = list(range(i, W.shape[0]))
            for e, (vw, vx) in enumerate(zip(W[use_range,:], X[use_range,:])):
                j = use_range[e]
                common_ratings = np.logical_and(uw, vw)
                number_common_ratings = np.sum(common_ratings)
                if number_common_ratings == 0:
                    similarity[i, j] = 0
                else:
                    ratios_sum = np.sum(np.minimum(ux[common_ratings], vx[common_ratings])/np.maximum(ux[common_ratings], vx[common_ratings]))
                    weight = 1.0/(1+np.exp(-(number_ratings[i] + number_ratings[j])/(2*number_common_ratings))) #Why number_common_ratings in the denominator? Would make more sense to inverse numerator and denominator, but like that in the paper
                    similarity[i, j] = weight*ratios_sum/number_common_ratings
                
                similarity[j, i] = similarity[i, j]
    
        return similarity
        
    def __similarity_weighting(self, similarity, W):
        '''
        Weight the similarity matrix based on the number of ratings of each entry. Without weighting, users having 
        just few entries are often considered as closer, which this method tries to prevent.

        Weight depending on self.method: String, either 'weighting', 'significance' or 'sigmoid', default 'weighting'
                        'weighting' weights all entries based on the number of common rated items and number of rated items.
                        'significance' only reduce importance when number of common rated items is below the threshold.
                        'sigmoid' reduces weight when users have only few common rated items. It keeps most of the similarity
                        measure almost untouched and hence is the softest weighting method.
        
        Parameters
        ----------
        similarity : np.array(N_USERS, N_USERS) or np.array(N_MOVIES, N_MOVIES) or None
                    The matrix with similarity between users or movies
            
        W : np.array(N_USERS, N_MOVIES):
            The mask containing 1 for rated items and 0 for unrated items.

        Return
        ----------
        weighted_similarity: np.array(N_USERS, N_USERS) or np.array(N_MOVIES, N_MOVIES) depending of shape of similarity:
                    The weighted similarity score (between -1 and 1) for each user-user or item-item pair.
                    The returned matrix is therefore symmetric.
        '''
        if similarity is None:
            return None

        assert (similarity.shape[0] == W.shape[0] or similarity.shape[0] == W.shape[1]) and similarity.shape[0] == similarity.shape[1]
        
        weighted_similarity = np.zeros_like(similarity)
        if similarity.shape[0] != W.shape[0]:
            W=W.T # We were using the items and not the users for the similarity
        
        number_ratings = np.sum(W, axis=1)
        
        for i, u in enumerate(W):
            use_range = list(range(i, W.shape[0]))
            for e, v in enumerate(W[use_range, :]):
                j = use_range[e]
                number_common_ratings = np.sum(np.logical_and(u, v))
                
                if self.method == "weighting":
                    weight = 2*number_common_ratings/(number_ratings[i] + number_ratings[j]) if (number_ratings[i] + number_ratings[j]) != 0 else 0
                elif self.method == "significance":
                    weight = np.minimum(number_common_ratings, self.signifiance_threshold)/self.signifiance_threshold
                elif self.method == "sigmoid":
                    weight = 1.0/(1+np.exp(-number_common_ratings/2))
                else:
                    raise ValueError(f"{self.method} is not a valid method! Should be 'weighting', 'significance' or 'sigmoid'")

                weighted_similarity[i, j] = weight*similarity[i, j]
                weighted_similarity[j, i] = weighted_similarity[i, j]

        return weighted_similarity

    def __weighted_average_predict(self, mask_to_predict, similarity):
        '''
        Predict the missing values by a weighted average of the ratings of the k nearest neighbors with a weight 
        corresponding to their similarity. Take into account the mean value of the user (respectively item). Uses
        only item based similarity matrix or user based similarity matrix
        
        
        Parameters
        ----------        
        mask_to_predict : np.array(N_USERS, N_MOVIES):
            The mask containing 1 for the values we want to predict.
        
        similarity : np.array(N_USERS, N_USERS) or np.array(N_MOVIES, N_MOVIES) or None
                    The matrix with similarity between users or movies
        
        Return
        ----------
        predictions: np.array(N_USERS, N_MOVIES)
                    The predictions for the missing values
        
        confidence: np.array(N_USERS, N_MOVIES)
                    The confidence based on the similarity of neighbors used to compute it (if neighbors have high
                    similarity, will give high confidence).
        '''
        assert mask_to_predict.shape == self.X_train.shape and mask_to_predict.shape == self.W_train.shape

        if similarity is None:
            return None, None

        MIN_POSSIBLE_RATING = 1
        MAX_POSSIBLE_RATING = 5
        
        was_transposed=False
        printing_interval=200
        
        X = self.X_train.copy()
        W = self.W_train.copy()

        if similarity.shape[0] != W.shape[0]: # We were using the items and not the users for the similarity 
            was_transposed=True
            W=W.T   
            X=X.T
            printing_interval=30

        def parrallel_weigthed_prediction(i):  
            preds = X[i, :].copy()
            confs = np.ones_like(preds) 

            user_mean = np.nanmean(X[i, :]) # Might raise a warning if row full of Nan. Is handled the following line
            user_mean = np.nan_to_num(user_mean, nan=(MAX_POSSIBLE_RATING+MIN_POSSIBLE_RATING)/2) # Replace Nan by mean value if a row was full of Nan

            if self.use_std:
                number_items_rated = np.sum(W[i, :])
                user_std = np.sqrt(np.nansum((X[i, :]-user_mean)**2)/(number_items_rated-1)) if number_items_rated > 1 else 1
            
            for j in range(mask_to_predict.shape[1]):
                if mask_to_predict[i, j]:
                    possible_neighbors = np.where(np.logical_and(W[:, j], similarity[i, :]>self.min_similarity_neighbor))[0]
                    sorted_possible_neighbors = possible_neighbors[np.flip(np.argsort(similarity[i, possible_neighbors]))]
                    nearest_neighbors = sorted_possible_neighbors[:self.k]

                    if nearest_neighbors.shape[0] == 0:
                        predictions[i, j] = user_mean
                    elif self.use_std:                                        
                        neighbors_number_item_rated = np.sum(W[nearest_neighbors, :], axis=1)
                        neighbors_means = np.nanmean(X[nearest_neighbors, :], axis=1) # Might raise a warning if row full of Nan. Is handled the following line
                        neighbors_means = np.nan_to_num(neighbors_means, nan=(MAX_POSSIBLE_RATING+MIN_POSSIBLE_RATING)/2) # Replace Nan by mean value if a row was full of Nan
                        neighbors_number_item_rated[neighbors_number_item_rated<=1]=2 #To avoid division by 0 problems, should not happen frequently, set std to 1 later
                        
                        neighbors_stds = np.sqrt(np.nansum((X[nearest_neighbors, :]-neighbors_means.reshape(-1,1))**2, axis=1)/(neighbors_number_item_rated-1))
                        neighbors_stds[neighbors_number_item_rated<=1] = 1 #Set std to 1 if it was the only rating

                        preds[j] = user_mean + user_std * np.sum(np.multiply(similarity[i, nearest_neighbors], (X[nearest_neighbors, j]-np.nanmean(X[nearest_neighbors, :], axis=1))/neighbors_stds))/np.sum(similarity[i, nearest_neighbors])
                    else:  
                        preds[j] = user_mean + np.sum(np.multiply(similarity[i, nearest_neighbors], X[nearest_neighbors, j]-np.nanmean(X[nearest_neighbors, :], axis=1)))/np.sum(similarity[i, nearest_neighbors])
                    
                    confs[j] = np.sum(similarity[i, nearest_neighbors]) if nearest_neighbors.shape[0] != 0 else 0
                    
            # If more than one core, doesn't print anything anyway.
            if self.num_cores == 1 and self.verbose and (i==X.shape[0]-1 or (not i%printing_interval and i!=0)):
                similarity_type = "user" if not was_transposed else "item"
                print(f"Done with {similarity_type} {i}/{X.shape[0]}")
            
            return (preds, confs)
        
        r = Parallel(n_jobs=self.num_cores)(delayed(parrallel_weigthed_prediction)(i) for i in range(X.shape[0]))
        predictions, confidence  = zip(*r)
        predictions = np.array(predictions)
        confidence = np.array(confidence)
        predictions = np.clip(predictions, MIN_POSSIBLE_RATING, MAX_POSSIBLE_RATING) # Might exceed it, so we clip to correct range
        
        if was_transposed:
            predictions = predictions.T
            confidence = confidence.T
        
        return predictions, confidence

    def __predict_using_users_and_items(self, users_pred, items_pred, users_confidence, items_confidence):
        """
        Compute the final prediction using both user and items predictions.
        
        Parameters
        ----------
        users_pred : np.array(N_USERS, N_MOVIES) or None
                    The matrix of predictions based on the neighbors of the users
        
        items_pred: np.array(N_USERS, N_MOVIES) or None
                    The matrix of predictions based on the neighbors of the items
        
        users_confidence : np.array(N_USERS, N_MOVIES) or None
                    The matrix of confidence based on the neighbors of the users
        
        items_confidence: np.array(N_USERS, N_MOVIES) or None
                    The matrix of confidence based on the neighbors of the items
            
        user_weight: float in range [0,1], default 0.5
            Used to add manually more weight to user or items prediction
        
        Return
        ----------
        final_predictions: np.array(N_USERS, N_MOVIES)
                    The Final prediction
        """
        assert self.user_weight>=0 and self.user_weight<=1
        assert not (users_pred is None and items_pred is None)

        if users_pred is None:
            return items_pred
        if items_pred is None:
            return users_pred
        
        no_predictions = np.where((self.user_weight*users_confidence + (1-self.user_weight)*items_confidence)==0) #Should not happen
        
        #To avoid division by 0, we put same weight to both predictions, which will just be the mean of the user respectively item, so it makes sense
        if no_predictions[0].shape[0] != 0:
            items_confidence[no_predictions]=1
            users_confidence[no_predictions]=1
        
        weight_users = (self.user_weight*users_confidence)/(self.user_weight*users_confidence + (1-self.user_weight)*items_confidence)
        weight_items = ((1-self.user_weight)*items_confidence)/(self.user_weight*users_confidence + (1-self.user_weight)*items_confidence)

        final_predictions = users_pred*weight_users + items_pred*weight_items
        return final_predictions


class ComprehensiveSimilarityReinforcement(SimilarityMethods):
    """
    Class to implement the Comprehensive Similarity Reinforcement (CSR) algorithm, which refine the user similarity using items similarity
    and vice versa. See paper at https://dl.acm.org/doi/pdf/10.1145/3062179 for more informations. By default, use the same parameters as
    in the paper (PCC with weighting and all neighbors), but can be changed to use all the different similarities and weighting of the super class.
    Note that the original algorithm is extremely slow (runs in O((|I|^2)*(|U|^2)*max_iter))), therefore a modification was added, which consists
    of only sampling a random set of users or items at each iteration to update the similarity. This reduces the runtime to 
    O(max_iter*(sample_size^2)*((|I|^2)+(|U|^2)))).
    Most parameters are the same as the super class and won't be reexplained here.
    
    Parameters
    ----------
    alpha: float in range (0,1), (optional)
        Update parameter. If small, the reinforced matrix will change only slightly at each iteration.
    
    epsilon: float, (optional)
        Threshold used to stop the iterations when the Frobenius norm of the difference between two 
        iterations is below it for both user and item matrix.

    max_iter: int, (optional)
        Maximum number of iterations. 

    sample_size: int or None, (optional)
        Number of samples used to compute the similarity at each iteration. If set to None, don't sample (original paper), which is more accurate
        but extremely slow.     
    """
    def __init__(self, model_id, n_users, n_movies, similarity_measure="PCC", weighting="weighting", use_std=False,  k=10000, statistic_to_use="mean", user_weight=0.5, n_jobs=-1, verbose = 0, random_state=42, alpha=0.5 , epsilon=0.1, max_iter=10, sample_size=None):
        super().__init__(model_id=model_id, n_users=n_users, n_movies=n_movies, similarity_measure=similarity_measure, weighting=weighting, method="both", use_std=use_std,  k=k, statistic_to_use=statistic_to_use, user_weight=user_weight, n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        assert alpha >= 0 and alpha <= 1, "Alpha must be between 0 and 1 (both included)."
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.model_name = "CSR"
        self.sample_size = sample_size

    
    def fit(self, X, y, W, test_size = 0, normalization = None, log_rmse=True):
        super().fit(X, y, W, test_size, normalization, log_rmse=False) # We don't need to compute the loss for this model
        self.fitted = False

        self.similarity_users, self.similarity_items =  self.__compute_CSR()

        self.fitted = True

        # log training and validation rmse
        if log_rmse:
            train_rmse = self.score(self.X_train, self.predict(self.W_train, invert_norm=False), self.W_train)
            val_rmse = self.score(self.X_test, self.predict(self.W_test, invert_norm=True if normalization is not None else False), self.W_test)
            self.train_rmse.append(train_rmse)
            self.validation_rmse.append(val_rmse)

    
    def log_model_info(self, path = "./log/", format = "json"):

        model_info = {
            "id" : self.model_id,
            "name" : self.model_name,
            "parameters" : {
                "similarity_measure": self.similarity_measure,
                "weighting": self.weighting,
                "method": self.method,
                "with std": self.use_std,
                "number nearest neighbors" : self.k,
                "user_weight" : self.user_weight,
                "alpha" : self.alpha,
                "epsilon" : self.epsilon,
                "max_iter": self.max_iter,
                "sample_size": self.sample_size
            },
            "train_rmse" : self.train_rmse,
            "val_rmse" : self.validation_rmse
        }

        if self.weighting == "significance":
            model_info["parameters"]["signifiance_threshold"] = self.signifiance_threshold
        if self.similarity_measure == "PCC":
            model_info["parameters"]["statistic_to_use"] = self.statistic_to_use

        if format == "json":
            with open(path + self.model_name + '{0:05d}'.format(self.model_id) + '.json', 'w') as fp:
                json.dump(model_info, fp, indent=4)
        else: 
            raise ValueError(f"{format} is not a valid file format!")

    def __compute_CSR(self):
        """
        Implement the Comprehensive Similarity Reinforcement algorithm, which refine the user similarity using items similarity
        and vice versa. See paper at https://dl.acm.org/doi/pdf/10.1145/3062179 for more informations.
        
        Return
        ----------
        users_reinforced_similarity: np.array(N_USERS, N_USERS)
                    The reinforced user similarity matrix
                    
        items_reinforced_similarity: np.array(N_MOVIES, N_MOVIES)
                    The reinforced item similarity matrix
        """
        users_reinforced_similarity = self.similarity_users.copy()
        items_reinforced_similarity = self.similarity_items.copy()
        
        X = self.normalize(self.X_train, technique = "min_max")
        W = self.W_train.copy()
        
        for iter_cur in range(self.max_iter):
            last_users_reinforced_similarity = users_reinforced_similarity.copy()
            last_items_reinforced_similarity = items_reinforced_similarity.copy()
            
            #Update users
            users_reinforced_similarity = self.__update(X, W, users_reinforced_similarity, items_reinforced_similarity)
            
            if self.verbose:
                print(f"Done with users of iteration {iter_cur+1}/{self.max_iter}")
            
            #Update items
            items_reinforced_similarity = self.__update(X, W, items_reinforced_similarity, users_reinforced_similarity)
            
            if self.verbose:
                print(f"Done with iteration {iter_cur+1}/{self.max_iter}")
            
            #Check for stopping condition
            dU = np.linalg.norm(users_reinforced_similarity-last_users_reinforced_similarity, ord="fro")
            dI = np.linalg.norm(items_reinforced_similarity-last_items_reinforced_similarity, ord="fro")
            
            if dU < self.epsilon and dI < self.epsilon:
                if self.verbose:
                    print(f"Early stopping due to convergence (difference between two runs smaller than epsilon = {self.epsilon})")
                break
            
        return users_reinforced_similarity, items_reinforced_similarity 

    def __update(self, X, W, matrix_to_be_updated, other_matrix):

        matrix_to_update = matrix_to_be_updated.copy()

        # The naming is based on the user to be updated. If the item similarity matrix is passed as matrix_to_update, then the naming of the variable should be equivalent with items and users changed.
        for user_0 in range(matrix_to_be_updated.shape[0]):
                all_rated_items_user0 = np.where(W[user_0, :])[0]
                if all_rated_items_user0.shape[0]!=0: #If == 0, we just don't update it
                    for user_1 in range(user_0+1, matrix_to_be_updated.shape[0]):
                        all_rated_items_user1 = np.where(W[user_1, :])[0]
                        if all_rated_items_user1.shape[0]!=0: #If == 0, we just don't update it
                            
                            # Sample items randomly if self.sample_size is not None. Otherwise keep all the items
                            if self.sample_size is not None:
                                chosen_items_user0 = np.random.choice(all_rated_items_user0, self.sample_size, False) if all_rated_items_user0.shape[0] > self.sample_size else all_rated_items_user0
                                chosen_items_user1 = np.random.choice(all_rated_items_user1, self.sample_size, False) if all_rated_items_user1.shape[0] > self.sample_size else all_rated_items_user1
                            else:
                                chosen_items_user0 = all_rated_items_user0
                                chosen_items_user1 = all_rated_items_user1
                            
                            indices = np.array(list(itertools.product(chosen_items_user0, chosen_items_user1))) # All pairs of indices, shape (chosen_items_user0*chosen_items_user1, 2)
                            pos_neg_indices_users = indices[other_matrix[indices[:, 0], indices[:, 1]]!=self.min_similarity_neighbor]
                            w_users = 1-2*np.abs(X[user_0, pos_neg_indices_users[:,0]]-X[user_1, pos_neg_indices_users[:,1]])
                            total_w_users = np.sum(np.abs(w_users))

                            if total_w_users != 0: #If == 0, we just don't update it
                                update_users = np.sum(w_users*other_matrix[pos_neg_indices_users[:,0], pos_neg_indices_users[:,1]])/total_w_users
                                new_sim_users = (1-self.alpha)*matrix_to_update[user_0, user_1] + self.alpha * update_users
                                matrix_to_update[user_0, user_1] = new_sim_users
                                matrix_to_update[user_1, user_0] = new_sim_users
        
        return matrix_to_update