import torch
import torch.nn as nn

import math

import _mapped_convolution_ext._weighted_mapped_avg_pooling as weighted_mapped_avg_pool
import _mapped_convolution_ext._mapped_avg_pooling as mapped_avg_pool
import _mapped_convolution_ext._resample as resample
from .layer_utils import *


class MappedAvgPoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output = weighted_mapped_avg_pool.weighted_mapped_avg_pool(
                input, sample_map, interp_weights, kernel_size, interp)
        else:
            pooled_output = mapped_avg_pool.mapped_avg_pool(
                input, sample_map, kernel_size, interp)

        self.save_for_backward(torch.tensor([input.shape[2], input.shape[3]]),
                               sample_map, torch.tensor(kernel_size),
                               torch.tensor(interp), interp_weights)

        return pooled_output

    @staticmethod
    def backward(self, grad_output):

        input_shape, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            grad_input = weighted_mapped_avg_pool.weighted_mapped_avg_unpool(
                grad_output, sample_map, interp_weights, input_shape[0],
                input_shape[1], kernel_size, interp)
        else:
            grad_input = mapped_avg_pool.mapped_avg_unpool(
                grad_output, sample_map, input_shape[0], input_shape[1],
                kernel_size, interp)

        return grad_input, None, None, None, None


class MappedAvgPool(nn.Module):

    def __init__(self, kernel_size, interpolation='bilinear'):

        super(MappedAvgPool, self).__init__()

        self.kernel_size = kernel_size

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
        return MappedAvgPoolFunction.apply(x, sample_map, self.kernel_size,
                                           self.interp, interp_weights)


class MappedAvgUnpoolFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                oh,
                ow,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        if interp_weights is not None:
            pooled_output = weighted_mapped_avg_pool.weighted_mapped_avg_unpool(
                input, sample_map, oh, ow, interp_weights, kernel_size, interp)
        else:
            pooled_output = mapped_avg_pool.mapped_avg_unpool(
                input, sample_map, oh, ow, kernel_size, interp)

        self.save_for_backward(torch.tensor([input.shape[2], input.shape[3]]),
                               sample_map, torch.tensor(kernel_size),
                               torch.tensor(interp), interp_weights)

        return pooled_output

    @staticmethod
    def backward(self, grad_output):

        input_shape, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            grad_input = weighted_mapped_avg_pool.weighted_mapped_avg_pool(
                grad_output, sample_map, interp_weights, kernel_size, interp)
        else:
            grad_input = mapped_avg_pool.mapped_avg_pool(
                grad_output, sample_map, kernel_size, interp)

        return grad_input, None, None, None, None, None, None, None


class MappedAvgUnpool(nn.Module):

    def __init__(self, kernel_size, interpolation='bilinear'):

        super(MappedAvgUnpool, self).__init__()

        self.kernel_size = kernel_size

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

    def forward(self, x, oh, ow, sample_map, interp_weights=None):
        '''
        x:          batch x channels x input_height x input_width
        oh:         scalar output height
        ow:         scalar output width
        sample_map: input_height x input_width x kernel_size x 2 (x, y)
        interp_weights: [OPTIONAL] input_height x input_width x kernel_size x num_interp_points x 2 (x, y)
        '''

        check_args(x, sample_map, interp_weights, None, self.kernel_size)
        check_input_map_shape(x, sample_map)
        return MappedAvgUnpoolFunction.apply(x, oh, ow, sample_map,
                                             self.kernel_size, self.interp,
                                             interp_weights)
