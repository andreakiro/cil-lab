# CIL Dataset and DataLoader PyTorch implementation
# Dataset of (user_index, item_index, rating) tuples
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
#####################################################

import torch
import numpy as np
import torch.utils.data as td
from src.configs import config

class CILDataset(td.Dataset):

    def __init__(self, split='train'):
        self.split = split

        self.train_df = np.loadtxt(
            open(config.TRAIN_DATA, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

        self.val_df = np.loadtxt(
            open(config.EVAL_DATA, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

        self.test_df = np.loadtxt(
            open(config.TEST_DATA, 'rb'),
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
        )

    def __getitem__(self, idx):
        if self.split == 'eval':
            return torch.from_numpy(self.val_df[idx])
        elif self.split == 'test':
            return torch.from_numpy(self.test_df[idx])

        return torch.from_numpy(self.train_df[idx])

    def __len__(self):
        if self.split == 'eval':
            return self.val_df.shape[0]
        elif self.split == 'test':
            return self.test_df.shape[0]

        return self.train_df.shape[0]

def get_dataloader(args, split='train', shuffle=True):
    dataset = CILDataset(split)
    dataloader = td.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=False,
    )
    
    len_tdl = len(dataset)
    print('Loading data with %d samples' % len_tdl)
    return dataloader, len_tdl
