import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import numpy as np
import time
import math

from mapped_convolution.nn import MappedConvolution, Convolution
from mapped_convolution.util import time_cuda

torch.manual_seed(0)


def pixel_shuffle_resampling(permutation, h, w, kh, kw, ph, pw, sh, sw, dh,
                             dw):

    # Mesh grid
    x = (permutation % w).view(h, w)
    y = (permutation / w).view(h, w)

    # Pad the mesh grid
    padded_x = F.pad(x, (pw, dw * pw, ph, dh * ph), value=-1)
    padded_y = F.pad(y, (pw, dw * pw, ph, dh * ph), value=-1)

    maps = []
    for i in range(0, kh):
        for j in range(0, kw):
            maps.append(
                torch.cat(
                    (padded_x[i:dh * h + i:dh, j:dw * w + j:dw].unsqueeze(-1),
                     padded_y[i:dh * h + i:dh, j:dw * w + j:dw].unsqueeze(-1)),
                    -1).view(h, w, 1, 2))
    return torch.cat(maps, -2).view(h, w, kh * kw, 2)


def profile(layer, data, mapping=None):
    if mapping is not None:
        return time_cuda(layer, [data, mapping])
    else:
        return time_cuda(layer, [data])


bs = 1
in_channels = 10
out_channels = 1
num_trials = 100
interpolation = 'bilinear'
sizes = [(10, 10), (20, 25), (20, 50), (50, 100), (100, 100), (200, 250),
         (200, 500), (500, 1000), (1000, 1000), (2000, 2500)]

# Initialize the layers
mapped_conv = MappedConvolution(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=9,
                                interpolation=interpolation).double().cuda()
conv = Convolution(in_channels, out_channels, (3, 3),
                   padding=1).double().cuda()

store_map_forward = torch.zeros(len(sizes))
store_map_backward = torch.zeros(len(sizes))
store_reg_forward = torch.zeros(len(sizes))
store_reg_backward = torch.zeros(len(sizes))
for i in range(len(sizes)):
    print(sizes[i])
    h, w = sizes[i]

    # Define inputs and gradient
    data = torch.arange(h * w).repeat(bs, in_channels, 1,
                                      1).view(bs, in_channels, h, w).double()
    data.requires_grad = True
    test_grad_output = torch.rand(bs, out_channels, h, w).double()

    # Create mapping function
    perm = torch.randperm(h * w)
    mapping = pixel_shuffle_resampling(perm, h, w, 3, 3, 1, 1, 1, 1, 1,
                                       1).double() + 0.5

    mapping = mapping.cuda()
    data = data.cuda()
    test_grad_output = test_grad_output.cuda()

    map_forward = torch.zeros(num_trials)
    map_backward = torch.zeros(num_trials)
    reg_forward = torch.zeros(num_trials)
    reg_backward = torch.zeros(num_trials)
    for j in range(num_trials):
        out_map_forward, time_map_forward = profile(mapped_conv, data, mapping)
        out_reg_forward, time_reg_forward = profile(conv, data)
        _, time_map_backward = profile(out_map_forward.backward,
                                       test_grad_output)
        _, time_reg_backward = profile(out_reg_forward.backward,
                                       test_grad_output)
        map_forward[j] = time_map_forward
        map_backward[j] = time_map_backward
        reg_forward[j] = time_reg_forward
        reg_backward[j] = time_reg_backward

        store_map_forward[i] = map_forward.mean()
        store_map_backward[i] = map_backward.mean()
        store_reg_forward[i] = reg_forward.mean()
        store_reg_backward[i] = reg_backward.mean()

    print('Mapped Forward Time: ', store_map_forward[i])
    print('Mapped Backward Time: ', store_map_backward[i])
    print('Regular Forward Time: ', store_reg_forward[i])
    print('Regular Backward Time: ', store_reg_backward[i])

stored = torch.stack((store_map_forward, store_map_backward, store_reg_forward,
                      store_reg_backward), 0)

stored.numpy().astype(np.float32).tofile('profile_bilinear.bin')
