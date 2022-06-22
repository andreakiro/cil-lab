import pandas as pd
import numpy as np
import math
import os

N_MOVIES = 1000
N_USERS = 10000

def get_input_matrix():
    '''
    Get the input matrix

    Return
    ----------
    (Id, Prediction): Dataframe Column
        The dataframe column containing entries coordinates and the dataframe
        column containing target values
    '''

    data_pd = pd.read_csv('./data/data_train.csv') 
    return data_pd.Id, data_pd.Prediction

def populate_matrices(X, y, n_users, n_movies):
    """
    Populate data and mask matrices from sparse initial data

    Parameters
    ----------
    X: pd.dataFrame(Id)   
        The dataframe column containing entries coordinates
    y: pd.dataFrame(Prediction)   
        The dataframe column containing the target value for each coordinate
    n_users: int 
        number of rows of the matrices
    n_movies: int
        number of columns of the matrices
    
    Return
    ----------
    (data, W): (np.array(N_USERS, N_MOVIES), np.array(N_USERS, N_MOVIES))
        The input array with the true ratings and Nan where no ratings where given and the 
        array containing 1 where the entries are given and 0 otherwise.
    '''
    """
    users, movies = [np.squeeze(arr) for arr in np.split(X.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = y.values
    # create data matrix
    data = np.full((n_users, n_movies), np.nan)
    W = np.full((n_users, n_movies), False)
    # populate data matrix
    for user, movie, pred in zip(users, movies, predictions): 
        data[user][movie] = pred
        W[user][movie] = True if not math.isnan(pred) else False
    return data, W

def generate_submission(predictions, name="submission.zip", compression="zip"):
    '''
    Generate CSV or zip file for submission

    Parameters
    ----------
    predictions: pd.dataFrame(Id, Prediction)   
        The dataframe containing our predictions for each blank entry
    name: str (optional)
        name of the generated file
    compression: str (optional)
        Format of the compression
    '''
    sample = pd.read_csv('data/sampleSubmission.csv') 
    sample = sample.astype({"Prediction": float}, errors='raise')
    import re
    for index, row in sample.iterrows():
        r, c = re.findall(r'r(\d+)_c(\d+)', row["Id"])[0]
        sample.at[index, "Prediction"] = predictions[int(r)-1][int(c)-1]
    sample.to_csv(name, compression=compression, float_format='%.3f', index = None)



def submit_on_kaggle(name="submission.zip", message=None):
    '''
    Submit a solution on kaggle.

    Parameters
    ----------
    name: str (optional)
        name of the file to submit
    message: str (optional)
        Message to use with the submission. Makes easier to 
        understand what each submission is about
    '''
    command = f"kaggle competitions submit -c cil-collaborative-filtering-2022 -f {name}"
    if not message is None:
        command = command + f' -m "{message}"'
    os.system(command)

