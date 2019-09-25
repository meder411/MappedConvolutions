import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import math
import time

import _mapped_convolution_ext._mapped_convolution as mapped_conv
import _mapped_convolution_ext._weighted_mapped_convolution as weighted_mapped_conv
import _mapped_convolution_ext._resample as resample
from .layer_utils import *


class MappedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                weight,
                bias,
                sample_map,
                kernel_size,
                interp,
                interp_weights=None):

        self.save_for_backward(input, weight, bias, sample_map,
                               torch.tensor(kernel_size), torch.tensor(interp),
                               interp_weights)

        if interp_weights is not None:
            return weighted_mapped_conv.weighted_mapped_conv_forward(
                input, sample_map, interp_weights, weight, bias, kernel_size,
                interp)

        else:
            return mapped_conv.mapped_conv_forward(input, sample_map, weight,
                                                   bias, kernel_size, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            weight, \
            bias, \
            sample_map, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            grad_input, grad_weight, grad_bias = weighted_mapped_conv.weighted_mapped_conv_backward(
                grad_output, sample_map, interp_weights, input, weight, bias,
                kernel_size, interp)
        else:
            grad_input, grad_weight, grad_bias = mapped_conv.mapped_conv_backward(
                grad_output, sample_map, input, weight, bias, kernel_size,
                interp)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class MappedConvolution(nn.Module):
    '''
    A class that performs the mapped convolution operation. This operations requires a map tensor that maps from the input to some output. The output dimension is determined by the dimension of the map.
    '''

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # One dimension
            interpolation='bilinear',
            bias=True):

        super(MappedConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if interpolation == 'nearest':
            self.interp = 0
        elif interpolation == 'bilinear':
            self.interp = 1
        elif interpolation == 'bispherical':
            self.interp = 2
        else:
            assert False, 'Unsupported interpolation type'

        # Initialize parameters of the layer
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        # Initializes the parameters of the layer
        self.reset_parameters()

        if not bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def reset_parameters(self):
        '''
        Sets up initial weights for the parameters
        '''
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, sample_map, interp_weights=None):
        '''
        x:              batch x channels x input_height x input_width
        sample_map:     output_height x output_width x kernel_size x 2 (x, y)
        interp_weights: [OPTIONAL] output_height x output_width x num_interp_points x 2
        '''

        check_args(x, sample_map, interp_weights, self.in_channels,
                   self.kernel_size)
        return MappedConvolutionFunction.apply(x, self.weight, self.bias,
                                               sample_map, self.kernel_size,
                                               self.interp, interp_weights)
