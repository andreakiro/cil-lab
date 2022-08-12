EPOCHS = 20
# K = 3
λ = 0.1
# model = ALS(1, 10000, 1000, EPOCHS, K, λ, verbose = 1, test_size=0.2)
K = [3]
KAGGLE = True
GENERATE_SUBM = True
MESSAGE = "Testin refactoring of the model"
LOG_MODEL_INFO = True

# FOR MAIN
N_USERS = 10000
N_MOVIES = 1000
DATA_PATH = 'data/data_train.csv'
SUBMISSION_DATA_PATH = 'data/sampleSubmission.csv'