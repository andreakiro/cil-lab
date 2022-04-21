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
    (data, W): (np.array(N_USERS, N_MOVIES), np.array(N_USERS, N_MOVIES))
        The input array with the true ratings and Nan where no ratings where given and the 
        array containing 1 where the entries are given and 0 otherwise.
    '''

    data_pd = pd.read_csv('./data/data_train.csv') 

    # get users, movies
    users, movies = [np.squeeze(arr) 
                    for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    # get predictions
    predictions = data_pd.Prediction.values

    # create data matrix
    data = np.full((N_USERS, N_MOVIES), np.nan)
    W = np.full((N_USERS, N_MOVIES), 0)

    # populate data matrix
    for user, movie, pred in zip(users, movies, predictions): 
        data[user][movie] = pred
        W[user][movie] = 1 if not math.isnan(pred) else 0
    
    return (data, W)


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
    predictions.to_csv(name, compression=compression, float_format='%.3f', index = None)

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
        command = command + f" -m {message}"

    os.system(command)

