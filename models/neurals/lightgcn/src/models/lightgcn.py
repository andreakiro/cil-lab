# PyTorch implementation of LightGCN neural model
# Adapted from github.com/gusye1234/LightGCN-PyTorch
# Adapted from github.com/LucaMalagutti/CIL-ETHZ-2021
# Source paper @ https://arxiv.org/pdf/2002.02126.pdf
#####################################################

import numpy as np
import scipy.sparse as sp
from torch import nn
import torch

from src.configs import config

class LightGCN(nn.Module):

    def __init__(self, args):
        super(LightGCN, self).__init__()
        
        # std params
        self.args = args
        self.n_users = config.USERS
        self.n_items = config.MOVIES

        # model layers
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users,
            embedding_dim=args.emb_size
        )

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items,
            embedding_dim=args.emb_size
        )

        # layer initialization
        if args.path_to_model is None:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.A_split = self.build_matrix()

    def build_matrix(self):
        A = sp.dok_matrix(
            (
                self.n_users + self.n_items, 
                self.n_users + self.n_items
            ),
            dtype=np.float32
        )

        A = A.tolil()
        R = sp.load_npz(self.args.r_train_path)

        A[: self.n_users, self.n_users :] = R
        A[self.n_users :, : self.n_users] = R.T

        A_mask = sp.dok_matrix(
            (
                self.n_users + self.n_items, 
                self.n_users + self.n_items
            ),
            dtype=np.float32,
        )

        A_mask = A_mask.tolil()
        R_mask = sp.load_npz(self.args.r_mask_train_path)

        A_mask[: self.n_users, self.n_users :] = R_mask
        A_mask[self.n_users :, : self.n_users] = R_mask.T

        row_sum = np.array(A_mask.sum(axis=1))
        D_inv = np.power(row_sum, -0.5).flatten()
        D_inv[np.isinf(D_inv)] = 0.0
        D = sp.diags(D_inv)

        A_split = D.dot(A).dot(D)
        A_split = A_split.tocsr()

        A_split = self._convert_sp_mat_to_sp_tensor(A_split)

        return A_split.to(self.args.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, batch):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        curr_emb = torch.cat([users_emb, items_emb])
        embs_list = [curr_emb]

        for _ in range(self.args.num_layers):
            # performs graph convolutions
            curr_emb = torch.sparse.mm(self.A_split, curr_emb)
            embs_list.append(curr_emb)

        stacked_embs = torch.stack(embs_list, dim=1)

        e = torch.mean(stacked_embs, dim=1)
        users_emb, items_emb = torch.split(e, [self.n_users, self.n_items])

        users_idx = batch[:, 0]
        items_idx = batch[:, 1]

        # gets embeddings of users and items contained in the batch
        batch_users_emb = users_emb[users_idx.long()]
        batch_items_emb = items_emb[items_idx.long()]

        # predicts ratings for all the user-item pairs in the batch
        scores_matrix = batch_users_emb @ batch_items_emb.T

        # extracts predicted rating for the given batch
        scores = torch.diagonal(scores_matrix, 0)

        return scores
