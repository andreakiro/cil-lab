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
$ wandb sweep $CONFIG
```

### Run your own agent
* Ping to get `<USERNAME/PROJECTNAME/SWEEPID>`
```
$ wandb agent <USERNAME/PROJECTNAME/SWEEPID> --count $NUM
```

### Run agent on cluster
```
$ env2lmod
$ module load gcc/6.3.0 eth_proxy hdf5/1.10.1
$ bsub -W 08:00 -n 1 -R "rusage[mem=8192]" wandb agent <USERNAME/PROJECTNAME/SWEEPID> --count $NUM
```