import torch
import torch.nn as nn

import math

import _mapped_convolution_ext._transposed_convolution as transposed_conv
from .layer_utils import _pair


class TransposedConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, input, weight, bias, kernel_size, stride, padding,
                dilation):
        self.save_for_backward(input, weight, bias, torch.tensor(kernel_size),
                               torch.tensor(stride), torch.tensor(padding),
                               torch.tensor(dilation))
        return transposed_conv.transposed_conv_forward(
            input, weight, bias, kernel_size[0], kernel_size[1], stride[0],
            stride[1], padding[0], padding[1], dilation[0], dilation[1])

    @staticmethod
    def backward(self, grad_output):
        input, weight, bias, kernel_size, stride, padding, dilation = self.saved_tensors
        grad_input, grad_weight, grad_bias = transposed_conv.transposed_conv_backward(
            grad_output, input, weight, bias, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0],
            dilation[1])

        return grad_input, grad_weight, grad_bias, None, None, None, None


class TransposedConvolution(nn.Module):

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Filter size
            stride=1,  # Stride
            padding=0,  # Padding
            dilation=1):  # Dilation

        super(TransposedConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)

        # Initialize parameters of the layer
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, out_channels, self.kernel_size[0],
                         self.kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return TransposedConvolutionFunction.apply(x, self.weight, self.bias,
                                                   self.kernel_size,
                                                   self.stride, self.padding,
                                                   self.dilation)
