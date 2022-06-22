from calendar import EPOCH
from utils.utils import get_input_matrix, generate_submission, submit_on_kaggle
from utils.config import *
import numpy as np
from models.als import ALS
from models.svd import SVD


def main():
    # load data
    print("Loading data...")
    id, prediction = get_input_matrix()


    for i, k in enumerate(K):
    
        # fit the model
        print(f"Fitting the model for k={k}...")
        
        model = ALS(i, 10000, 1000, EPOCHS, k, λ, verbose = 1, test_size=0.2)
        pred = model.fit_transform(id, prediction)

        if LOG_MODEL_INFO:
            print("Logging model info...")
            model.log_model_info()

        if GENERATE_SUBM:
            print("Generating submission...")
            generate_submission(pred)

        if KAGGLE:
            print("Submitting to Kaggle...")
            submit_on_kaggle(name="submission.zip", message="Testing bash pipeline")
            
    print("Done.")

if __name__ == '__main__':
    main()