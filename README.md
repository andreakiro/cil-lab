# The effectiveness of factorization and similarity blending

* Collaborative Filtering (CF) project and competition
* Computational Intelligence Lab - ETHZ Zurich FS 2022
* Kaggle [leaderboard](https://www.kaggle.com/competitions/cil-collaborative-filtering-2022/leaderboard) for the competition (Group: *Bonjour*)
* Given resources to kick-off the project available [here](https://docs.google.com/document/d/1ynT7xilJTBtD7T8KpMyKRjc3CC-wEh-XX7ZgRkh4fyc/edit#heading=h.ajjlw0b7sp4p)

## Project highlights

* Benchmark of state of the art CF techniques including;
* Matrix factorization, similarity- and neural-based models.
* Show effectiveness of factorization and similarity blending.
* Propose SCSR, a novel stochastic extension of a similarity model;
* Consistently reduce asymptotic complexity of the original algorithm.
* Read the **project paper** for further details: [**`cil-paper.pdf`**](cil-paper.pdf)

## Get started with the code

### For neural-based techniques
* Refer to the [`models/neurals/README.md`](models/neurals/README.md) file
* Includes all instructions to run code and reproduce results
* DeepRec and LightGCN model implementations available

### For blending techniques
* Refer to the [`instructions.md`](instructions.md) of the base project
* Includes similarity-based and matrix factorization models
* Includes instructions to run code and reproduce paper results

## How to install and run the environment
Verify that conda is installed and running on your system by typing:
```
conda --version
```
To install the environment, simply run the following commands in the main project folder:
```
conda env create -f environment.yml
conda activate cil
```

## Euler and Kaggle informations
* More details available in the [`euler-kaggle.md`](euler-kaggle.md) file