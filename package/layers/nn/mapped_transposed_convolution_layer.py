import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

import math
import time

import _mapped_convolution_ext._mapped_transposed_convolution as mapped_transposed_conv
import _mapped_convolution_ext._weighted_mapped_transposed_convolution as weighted_mapped_transposed_conv
import _mapped_convolution_ext._resample as resample
from .layer_utils import *


class MappedTransposedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self,
                input,
                weight,
                bias,
                sample_map,
                output_height,
                output_width,
                kernel_size,
                interp,
                interp_weights=None):

        self.save_for_backward(input, sample_map, weight, bias,
                               torch.tensor(kernel_size), torch.tensor(interp),
                               interp_weights)

        if interp_weights is not None:
            return weighted_mapped_transposed_conv.weighted_mapped_transposed_conv_forward(
                input, sample_map, interp_weights, weight, bias, output_height,
                output_width, kernel_size, interp)
        else:
            return mapped_transposed_conv.mapped_transposed_conv_forward(
                input, sample_map, weight, bias, output_height, output_width,
                kernel_size, interp)

    @staticmethod
    def backward(self, grad_output):
        input, \
            sample_map, \
            weight, \
            bias, \
            kernel_size, \
            interp, \
            interp_weights = self.saved_tensors

        if interp_weights is not None:
            grad_input, grad_weight, grad_bias = weighted_mapped_transposed_conv.weighted_mapped_transposed_conv_backward(
                grad_output, sample_map, interp_weights, input, weight, bias,
                kernel_size, interp)
        else:
            grad_input, grad_weight, grad_bias = mapped_transposed_conv.mapped_transposed_conv_backward(
                grad_output, sample_map, input, weight, bias, kernel_size,
                interp)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class MappedTransposedConvolution(nn.Module):
    '''
    Contrary to the standard mapped convolution, for the mapped transposed convolution, the mapping should be a mapping from the output to the input. The map should be the same size as the input, but point to locations in the range of the output.
    '''

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Note the single dimension
            interpolation='bilinear',
            bias=True):
        super(MappedTransposedConvolution, self).__init__()

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
            torch.Tensor(self.in_channels, self.out_channels, self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters()

        if not bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, oh, ow, sample_map, interp_weights=None):
        '''
        x:          batch x channels x input_height x input_width
        sample_map: input_height x input_width x kernel_size x 2 (x, y)
        oh:         scalar output height
        ow:         scalar output width
        interp_weights: [OPTIONAL] input_height x input_width x kernel_size x num_interp_points x 2 (x, y)
        '''

        check_args(x, sample_map, interp_weights, self.in_channels,
                   self.kernel_size)
        check_input_map_shape(x, sample_map)
        return MappedTransposedConvolutionFunction.apply(
            x, self.weight, self.bias, sample_map, oh, ow, self.kernel_size,
            self.interp, interp_weights)
