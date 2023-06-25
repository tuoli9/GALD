from PIL import Image

from torch import nn
import torch

import numpy as np
from einops import rearrange

def drop_patch(img, shuffle_size = 14, ratio = 0.1):
    patch_num = shuffle_size**2
    drop_count = int(ratio * patch_num)
    patch_dim1, patch_dim2 = 224 // shuffle_size, 224 // shuffle_size       # 16

    img = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_dim1, p2=patch_dim2)
    row = np.random.choice(range(patch_num), size=drop_count, replace=False)
    img[:, row, :] = 0.0
    img = rearrange(img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=shuffle_size, w=shuffle_size, p1=patch_dim1, p2=patch_dim2)
    return img


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self, t):
        dtype = t.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=t.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=t.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        t = (t-mean)/std
        return t




def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

