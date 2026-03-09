import torch

import matplotlib.pyplot as plt
import torch.nn as nn
import time

from torch.linalg import lstsq
import numpy as np


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                          inp.view(-1, self.num_bands, 1)))
        target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                           target.view(-1, self.num_bands, 1)))

        summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
        angle = torch.acos(summation / (input_norm * target_norm))

        return angle

def Nuclear_norm(inputs):
    band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)

class SparseKLloss(nn.Module):
        def __init__(self):
            super(SparseKLloss, self).__init__()

        def __call__(self, input, decay=4e-4):
            input = torch.sum(input, 0, keepdim=True)
            loss = Nuclear_norm(input)
            return decay*loss

def compute_rmse(x_true, x_pre):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt( ((x_true-x_pre)**2).sum()/(img_w*img_h*img_c) )

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(0, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma_reg=1e-7):
        input = torch.sum(input, 0)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma_reg*loss