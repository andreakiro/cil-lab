"""
LightGCN Dataset and Dataloader PyTorch implementation
Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
"""

import torch
import numpy as np
import torch.utils.data as td

#######################################
######## DATA LOADER LILGHTGCN ########
#######################################

class DataLoaderLightGCN():

    def __init__(self, args, split='train', shuffle=True):
        self.dataset = DatasetLightGCN(args, split)
        self.len_set = len(self.dataset)
        self.dataloader = td.DataLoader(
            dataset=self.dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False
        )

    def get(self):
        print(f'Data loading {self.len_set} samples')
        return self.dataloader

    def size(self):
        return self.len_set

#######################################
########## DATASET LILGHTGCN ##########
#######################################

class DatasetLightGCN(td.Dataset):

    def __init__(self, args, split='train'):
        self.split = split

        self.train_df = np.loadtxt(
            open(args.path_to_train_data, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

        self.eval_df = np.loadtxt(
            open(args.path_to_eval_data, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

        self.test_df = np.loadtxt(
            open(args.path_to_test_data, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

    def __getitem__(self, idx):
        if self.split == 'eval':
            return torch.from_numpy(self.eval_df[idx])
        elif self.split == 'test':
            return torch.from_numpy(self.test_df[idx])

        return torch.from_numpy(self.train_df[idx])

    def __len__(self):
        if self.split == 'eval':
            return self.eval_df.shape[0]
        elif self.split == 'test':
            return self.test_df.shape[0]

        return self.train_df.shape[0]

