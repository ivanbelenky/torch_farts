from typing import Tuple, List, Sequence

import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F


plt.style.use('dark_background')
BoxType = List[Tuple[int, int]]

if torch.backends.mps.is_available():
    print("Using MPS backend")
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cossim(u, v):
    u = u.reshape(-1)
    v = v.reshape(-1)
    return F.cosine_similarity(u, v, dim=0, eps=1e-8)


def rect_mask(shape, box: BoxType):
    '''Returns a mask of shape `shape` with a rectangle of size `dx` by `dy`
    because we are dumb right?
    '''
    mask = np.ones(shape)
    (lowy, lowx), (upy, upx) = box
    mask[lowy:upy, lowx:upx] = 0.
    return mask


def load_img(path):
    '''reshape maxdim to 256, if image has no channels reshape it
    to (1, h, w)'''
    img = Image.open(path)
    if img.size[0] > 256 or img.size[1] > 256:
        greatest_dim = max(img.size)
        resize_factor = 256/greatest_dim
        img = img.resize((int(img.size[0]*resize_factor), int(img.size[1]*resize_factor)))
    img = np.asarray(img)
    if len(img.shape) == 2:
        img = img.reshape((*img.shape, 1))
    img = torch.from_numpy(img.astype(np.float32))
    img = img.permute(2, 0, 1)
    return img


class SeparationCosineLoss(nn.Module):
    def __init__(self, similarity=0., theta=0.8):
        super(SeparationCosineLoss, self).__init__()
        self.similarity = similarity
        self.separation_loss = nn.MSELoss()
        self.theta = theta
    
    def forward(self, u, v):
        cos_distance_loss = (1-self.theta)*abs(1-cossim(u, v) - self.similarity)
        naive_separation_loss = self.theta*self.separation_loss(u, v)
        return cos_distance_loss + naive_separation_loss #+ mean_loss + std_loss
        
class RLUDConvolver(nn.Module):
    def __init__(self, channels):
        super(RLUDConvolver, self).__init__()
        
        self.directional_conv = []
        for _ in range(4): #rlud ftw
            self.directional_conv.append(nn.Sequential(
                nn.Conv2d(channels, 16, 3, padding=1, device=device),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, 3, padding=1, device=device),
                nn.LeakyReLU(),
            ))

        self.condense_conv = nn.Sequential(
            nn.Conv2d(32*4, 16, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1, device=device),
            nn.ReLU(),
            nn.Conv2d(4, channels, 3, padding=1, device=device),
        )

    def forward(self, right, left, up, down):
        dir_right = self.directional_conv[0](right)
        dir_left = self.directional_conv[1](left)
        dir_up = self.directional_conv[2](up)
        dir_down = self.directional_conv[3](down)
        x = torch.cat((dir_right, dir_left, dir_up, dir_down), dim=1)
        x = self.condense_conv(x)
        return x
    
    @staticmethod
    def caterino(right, left, up, down):
        return torch.cat((right, left, up, down), dim=1)
            
    @staticmethod
    def cossim_slice_idxs(rlud_x, rlud_tgt, first: int=64):
        cossimilarities = F.cosine_similarity(rlud_tgt.reshape(rlud_tgt.shape[0], -1), rlud_x.reshape(rlud_x.shape[0],-1), dim=1, eps=1e-8)
        selected_idxs = torch.argsort(cossimilarities, descending=True)[:first]
        return selected_idxs
    
    @staticmethod
    def rlud_values(img, box, reshape=False):
        '''right left up down values'''
        (lowy, lowx), (upy, upx) = box
        i, j = lowy, lowx
        bh, bw = upy-lowy, upx-lowx

        right = img[:, i:i+bh, j+bw:j+2*bw]
        left = img[:, i:i+bh, j-bw:j]
        up = img[:, i-bh:i, j:j+bw]
        down = img[:, i+bh:i+2*bh, j:j+bw]

        if reshape:
            right = right.reshape(1, *right.shape)
            left = left.reshape(1, *left.shape)
            up = up.reshape(1, *up.shape)
            down = down.reshape(1, *down.shape)

        return right, left, up, down
    
    @staticmethod
    def weighted_dataset(img, box):
        '''idealized version where the masked image corresponds to places where it is possible to 
        displace the mask shape up/down and left/right without going out of bounds
        
        box: np.ndarray
            [left_up_xy, down_right_xy] box
        '''
        h, w = img.shape[1:]
        (lowy, lowx), (upy, upx) = box
        bh, bw = upy-lowy, upx-lowx
        
        if lowx-bw < 0 or upx + bw > w or lowy-bh < 0 or upy + bh > h:
            raise ValueError("unsolvable for now")

        idxs = [(i,j) for j in range(bw, w-2*bw+1) for i in range(bh, h-2*bh+1)
                if not ((lowy <= i < upy and lowx <= j < upx) or
                        ((lowy <= i < upy and (lowx <= j-bw < upx or lowx <= j+bw < upx)) or
                        (lowx <= j < upx and (lowy <= i-bh < upy or lowy <= i+bh < upy))))]

        center, right, left, up, down = [], [], [], [], []
        
        masked_img = img.clone()
        masked_img[:, lowy:upy, lowx:upx] = 0.
        for i, j in idxs:
            center.append(masked_img[:, i:i+bh, j:j+bw])
            r, l, u, d = RLUDConvolver.rlud_values(img, [(i,j), (i+bh, j+bw)])
            right.append(r)
            left.append(l)
            up.append(u)
            down.append(d)

        center = torch.stack(center)
        right = torch.stack(right)
        left = torch.stack(left)
        up = torch.stack(up)
        down = torch.stack(down)

        return center, right, left, up, down
