## The models implemented
**DeepRec Nvidia**; a model based on deep autoencoders
* Paper available here https://arxiv.org/abs/1708.01715
* NVIDIA repositiory here https://github.com/NVIDIA/DeepRecommender
* Copyright (c) 2017 NVIDIA Corporation

**LightGCN**; a model based on graph convolutional networks
* Paper available here https://arxiv.org/pdf/2002.02126.pdf
* Original repository here https://github.com/gusye1234/LightGCN-PyTorch
* Inspired implemention from https://github.com/LucaMalagutti/CIL-ETHZ-2021

## Get started with the code
**Install required dependencies:**
* In case you already have your virtual-env ready to go:
```
pip install -r requirements.txt
```
* Otherwise, create one and setup env with the following:
```
conda env create
conda activate test-cil
```

**Preprocess data to be consumed:**
* This should create a new `./neurals/data` folder:
* By default it creates a 80-20 training validation split
* You can tune these values (and seed) in `src.configs.py` or inline
```
python preprocessing.py --model ${all, deeprec, lightgcn} --split $
```

## Reproduce the report results
**Play with the models:**
* Basic command; you need to specify a model, a mode and a run name 
* Path to model argument is useful in test mode (or if you restore a model)
* You can hard-code your (hyper) params in `src.parser.py` or pass them inline
```
# this command will use all defaults (hyper) parameters
python main.py ${deeprec, lightgcn} --mode ${train, test} --rname $ --path_to_model $
python main-py --help 
```

**DeepRec full-loop:**
* Training your model with std (hyper) parameters
* Note that the evaluation frequency is a tunable params
* **Info:** 1 epoch is approx. 2.5s (double when evaluates)
```
python main.py deeprec --mode train --rname richard
```
* Testing (doing inference) of your cool model
* **Very important: run `test` with same params as `train`!**
```
python main.py deeprec --mode test --rname richard --path_to_model logs/deeprec/richard/last.model
```
&rarr; You now have a submission in csv format: `./logs/deeprec/richard/sub-deeprec.csv`

**LightGCN full-loop:**
* Training your model with std (hyper) parameters
* Note that the evaluation frequency is a tunable params
* **Info:** 1 epoch is approx. 150s (evaluation included)
```
python main.py lightgcn --mode train --rname buzz
```
* Testing (doing inference) of your cool model
* **Very important: run `test` with same params as `train`!**
```
python main.py lightgcn --mode test --rname buzz --path_to_model logs/deeprec/buzz/last.model
```
&rarr; You now have a submission in csv format: `./logs/lightgcn/buzz/sub-lightgcn.csv`

## **Some other neural-based methods**
- DeepRec Nvidia (Deep autoencoders) [[Paper](https://arxiv.org/pdf/1708.01715.pdf)] [[Repo](https://github.com/NVIDIA/DeepRecommender)]
- AutoRec (Deep autoencoders) [[Paper](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)] [[Repo](https://github.com/gtshs2/Autorec)]
- Light GCN (Graph neural networks) [[Paper](https://arxiv.org/pdf/2002.02126.pdf)] [[Repo](https://github.com/gusye1234/LightGCN-PyTorch)]
- xDeepFM (Deep factorization machines) [[Paper](https://arxiv.org/pdf/1803.05170.pdf)] [[Repo](https://github.com/Leavingseason/xDeepFM)]
- NCF (Neural collaborative filtering) [[Paper](https://arxiv.org/pdf/1708.05031.pdf)] [[Repo](https://github.com/hexiangnan/neural_collaborative_filtering)]
- KernelNet (Kernelized neural networks) [[Paper](http://proceedings.mlr.press/v80/muller18a.html)] [[Repo](https://github.com/lorenzMuller/kernelNet_MovieLens)]