## The model
- Model is based on Graph Convolutional Networks.
- Paper available here https://arxiv.org/pdf/2002.02126.pdf

## Requirements
* Python 3.6
* [Pytorch](http://pytorch.org/): `pipenv install`
* CUDA (recommended version >= 8.0)

## Getting Started

### Convert consumable data
```
$ python -m src.data.preprocessing.py
```

### Training the model
* Run with all defaults parameters
* `$RUN_NAME` overriden if `wandb` is online
```
$ python main.py --mode train --rname $RUN_NAME 
```

* Explore available parameters
```
$ python main.py --help 
```

### Testing the model
* Run with all default parameters
* Make sure to adjust params if training was tuned
```
$ python main.py --mode test --path_to_model $PATH
```

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