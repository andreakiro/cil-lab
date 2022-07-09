### Convert dataset

```
python src/converter.py $PATH_TO_TRAIN_DATA
```

### Model training

```
# run with defaults params
python main.py --logdir logs \
--path_to_train_data $PATH_TRAIN_DATA \
--path_to_eval_data $PATH_EVAL_DATA \
--evaluation_frequency 2 \
--num_checkpoints 4
```

### Sweep contribution

```
chmod +x sweep.sh
sh sweep.sh
```