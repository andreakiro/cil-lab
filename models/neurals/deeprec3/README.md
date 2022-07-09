## The model
- Model is based on deep AutoEncoders.
- Paper available here https://arxiv.org/abs/1708.01715
- Copyright (c) 2017 NVIDIA Corporation

## Requirements
* Python 3.6
* [Pytorch](http://pytorch.org/): `pipenv install`
* CUDA (recommended version >= 8.0)

## Getting Started

### Convert data to NVIDIA standard

```
$ python src/converter.py $PATH_TO_TRAIN_DATA
```

### Training the model

* Run with all defaults parameters
```
$ python main.py --logdir logs \
--path_to_train_data $PATH_TRAIN_DATA \
--path_to_eval_data $PATH_EVAL_DATA \
--evaluation_frequency 2 \
--num_checkpoints 4
```

* Explore available parameters
```
$ python main.py --help 
```

### Run inference on the Test set
* `TODO`

### Compute Test RMSE
* `TODO`

## W&B Sweeps

### Create a new Sweep
```
wandb sweep sweep.yaml
```

### Update your Sweep
* `TODO`

### Run your own agent
* Ping to get `<USERNAME/PROJECTNAME/SWEEPID>`
```
wandb agent --count $NUM <USERNAME/PROJECTNAME/SWEEPID>
```