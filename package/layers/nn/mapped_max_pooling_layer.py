import torch
import torch.nn as nn

import math

import _mapped_convolution_ext._weighted_mapped_max_pooling as weighted_mapped_max_pool
import _mapped_convolution_ext._mapped_max_pooling as mapped_max_pool
import _mapped_convolution_ext._resample as resample
from .layer_utils import *


class MappedMaxPoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output, idx_mask = weighted_mapped_max_pool.weighted_mapped_max_pool(
                input, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_output, idx_mask = mapped_max_pool.mapped_max_pool(
                input, sample_map, kernel_size, interp)

        self.mark_non_differentiable(idx_mask)

        self.save_for_backward(torch.tensor([input.shape[2], input.shape[3]]),
                               idx_mask, sample_map, torch.tensor(kernel_size),
                               torch.tensor(interp), interp_weights)

        return pooled_output, idx_mask

    @staticmethod
    def backward(self, grad_output, idx_mask_grad=None):
        input_shape, \
            idx_mask, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            grad_input = weighted_mapped_max_pool.weighted_mapped_max_unpool(
                grad_output, idx_mask, sample_map, interp_weights,
                input_shape[0], input_shape[1], kernel_size, interp)
        else:
            grad_input = mapped_max_pool.mapped_max_unpool(
                grad_output, idx_mask, sample_map, input_shape[0],
                input_shape[1], kernel_size, interp)

        return grad_input, None, None, None, None


class MappedMaxPool(nn.Module):

    def __init__(self,
                 kernel_size,
                 interpolation='bilinear',
                 return_indices=False):

        super(MappedMaxPool, self).__init__()

        self.kernel_size = kernel_size
        self.return_indices = return_indices

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, sample_map, interp_weights=None):

        check_args(x, sample_map, interp_weights, None, self.kernel_size)
        pooled, idx_mask = MappedMaxPoolFunction.apply(x, sample_map,
                                                       self.kernel_size,
                                                       self.interp,
                                                       interp_weights)

        if self.return_indices:
            return pooled, idx_mask
        else:
            return pooled


# --------------------------


class MappedMaxUnpoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                idx_mask,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            unpooled_input = weighted_mapped_max_pool.weighted_mapped_max_unpool(
                input, idx_mask, sample_map, interp_weights, input_shape[0],
                input_shape[1], kernel_size, interp)
        else:
            unpooled_input = mapped_max_pool.mapped_max_unpool(
                intput, idx_mask, sample_map, input_shape[0], input_shape[1],
                kernel_size, interp)

        self.save_for_backward(sample_map, torch.tensor(kernel_size),
                               torch.tensor(interp), interp_weights)

        return unpooled_input

    @staticmethod
    def backward(self, grad_output):

        sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            pooled_grad, _ = weighted_mapped_max_pool.weighted_mapped_max_pool(
                grad_output, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_grad, _ = mapped_max_pool.mapped_max_pool(
                grad_output, sample_map, kernel_size, interp)

        return grad_input, None, None, None, None


class MappedMaxUnpool(nn.Module):

    def __init__(self, kernel_size, interpolation='bilinear'):

        super(MappedMaxUnpool, self).__init__()

        self.kernel_size = kernel_size

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, idx_mask, sample_map, interp_weights=None):

        return MappedMaxUnoolFunction.apply(x, idx_mask, sample_map,
                                            self.kernel_size, self.interp,
                                            interp_weights)
