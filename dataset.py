import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class Dataset(Dataset):
    def __init__(self, X_features, X_mat, y):
        self.X_features = torch.as_tensor(X_features)
        self.X_mat = Tensor(X_mat).transpose(1,3)
        self.y = one_hot(torch.as_tensor(y.astype(np.int))).to(dtype=torch.float32)
    def __len__(self):
        return self.X_features.shape[0]
    def __getitem__(self, idx):
        n = self.X_features[idx]
        m = self.X_mat[idx]
        return n, m, self.y[idx]