## Usage

The following command generates a submission file for the Kaggle predictions, and runs all of the experiments mentioned in the report. Some of these parameters can be omitted, see below for more information.

```
python main.py --submission --plot --ensemble --experiments
```

## Options

```--submission``` (optional)

Train our best blending model on the whole dataset and generate the submission file `final_ensemble.zip`.

```--plot``` (optional)

Train the matrix factorization models SVD, FunkSVD, ALS and BFM with varying ranks, and plot the results.

```--ensemble``` (optional)

Train the matrix factorization and similarity models, and show the results of the following blending methods: Linear Regression, Lasso, Ridge, XGB Regressor, MLP Regressor, and Random Forest Regressor.

```--experiments``` (optional)

Train multiple variations of the BFM and similarity models, and save the results to the `log/` folder.

## Notes

To train from a different dataset or predict other values, update the `config.py` file with the relevant configuration.