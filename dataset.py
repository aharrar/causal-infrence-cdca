import numpy as np
import torch
from torch.utils.data import Dataset


def convert(yf, ycf, t):
    mask0 = t == 0
    mask1 = t == 1
    y0 = np.zeros_like(yf)
    y1 = np.zeros_like(ycf)

    y0[mask1] = ycf[mask1]
    y0[mask0] = yf[mask0]
    y1[mask1] = yf[mask1]
    y1[mask0] = ycf[mask0]

    return y0, y1


def inverce_convert(y0, y1, t):
    mask0 = t == 0
    mask1 = t == 1
    yf = np.zeros_like(y0)
    ycf = np.zeros_like(y1)

    yf[mask0] = y0[mask0]
    yf[mask1] = y1[mask1]
    ycf[mask0] = y1[mask0]
    ycf[mask1] = y0[mask1]

    return yf, ycf


class IhdpDataset(Dataset):
    def __init__(self, x, t, y, mu0, mu1, ycf):
        self.x = x
        self.t = t
        self.y = y
        self.mu0 = mu0
        self.mu1 = mu1
        self.ycf = ycf

    def __len__(self):
        # return self.size
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x[idx], self.t[idx], self.y[idx], self.mu0[idx], self.mu1[idx], self.ycf[idx]
