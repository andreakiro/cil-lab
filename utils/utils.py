import pandas as pd
import numpy as np
import math
import os
from utils.config import N_MOVIES, N_USERS


def get_test_mask(test):
    '''
    From the test variable to predict, create a sparse matrix representation
    '''
    W_test = np.full((N_USERS, N_MOVIES), False)
    for sample in test:
        W_test[sample[0]][sample[1]] = True
    return W_test


def load_data(data_path):
    '''
    Load the training data
    '''
    data_pd = pd.read_csv(data_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values

    data = np.column_stack((np.array(users), np.array(movies), np.array(predictions)))

    return data


def load_submission_data(data_path):
    '''
    Load the submission data
    '''
    data_pd = pd.read_csv(data_path) 
    users, movies = [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    data = np.column_stack((np.array(users), np.array(movies)))
    return data


def get_input_matrix(data):
    '''
    Get the input matrix

    Return
    ----------
    (X, W): (np.array(N_USERS, N_MOVIES), np.array(N_USERS, N_MOVIES))
        The input array with the true ratings and np.nan where no ratings where given and the 
        mask array containing True where the entries are given and False otherwise.
    '''
    users, movies, predictions = data[:, 0], data[:, 1], data[:, 2]

    # create data matrix
    X = np.full((N_USERS, N_MOVIES), np.nan)
    W = np.full((N_USERS, N_MOVIES), False)
    # populate data matrix
    for user, movie, pred in zip(users, movies, predictions): 
        X[user][movie] = pred
        W[user][movie] = True if not math.isnan(pred) else False
    return X, W


def get_preds_from_matrix(data, matrix):
    '''
    From results in a sparse matrix form, return the predictions in a single 1D array
    '''
    sim_preds = []
    for row in data:
        sim_preds.append(matrix[row[0]][row[1]])
    return sim_preds


def generate_submission(predictions, submission_data_path, name="submission.zip", compression="zip", matrix_predictions=False):
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
    sample = pd.read_csv(submission_data_path) 
    sample = sample.astype({"Prediction": float}, errors='raise')
    if matrix_predictions:
        import re
        for index, row in sample.iterrows():
            r, c = re.findall(r'r(\d+)_c(\d+)', row["Id"])[0]
            sample.at[index, "Prediction"] = predictions[int(r)-1][int(c)-1]
    else:
        sample['Prediction'] = predictions
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

