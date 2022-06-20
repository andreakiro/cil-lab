# Computational Intelligence Lab - FS2022

- [Resources](https://docs.google.com/document/d/1ynT7xilJTBtD7T8KpMyKRjc3CC-wEh-XX7ZgRkh4fyc/edit#heading=h.ajjlw0b7sp4p)

## How to install and run the environment
Verify that conda is installed and running on your system by typing:
```
conda --version
```
To install the environment, simply run the following commands in the main project folder:
```
conda env create
conda activate cil
```

## How to submit to Kaggle leaderboard
Install Kaggle API using the command:
```
pip install kaggle
```
To use the Kaggle API, sign up for a [Kaggle](https://www.kaggle.com) account. Then go to the 'Account' tab of your user profile and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json`.
For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command:
```
chmod 600 ~/.kaggle/kaggle.json
```
Once the token is set up, just use the in-line command:
```
kaggle competitions submit -c cil-collaborative-filtering-2022 -f submission.zip -m "message"
```
to submit the file `submission.zip` to the competition leaderboard.
