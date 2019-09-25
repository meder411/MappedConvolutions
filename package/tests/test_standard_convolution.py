import torch
import torch.nn as nn
import torch.testing as testing
from torch.autograd import gradcheck

import pytest

from mapped_convolution.nn import Convolution, TransposedConvolution

import utils
import parameters as params

bs = 3
in_channels = 2
out_channels = 3


def test_standard_conv_cpu():
    '''Simply compares our result to PyTorch's implementation'''
    input = torch.ones(params.bs, params.in_channels, 8, 8).double()
    input.requires_grad = True

    pytorch_layer = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=2).double()
    my_layer = Convolution(in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           dilation=2).double()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output = pytorch_layer(input)
    my_output = my_layer(input)

    # Run a numerical gradient check
    gradcheck_res = gradcheck(my_layer, (input))

    # Ensure our implementation passes gradcheck
    assert gradcheck_res

    # Make sure they match
    testing.assert_allclose(my_output, pytorch_output)


def test_standard_transposed_conv_cpu():
    '''Simply compares our result to PyTorch's implementation'''
    input = torch.ones(params.bs, params.in_channels, 5, 5).double()
    input.requires_grad = True

    pytorch_layer = torch.nn.ConvTranspose2d(params.in_channels,
                                             params.out_channels,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             dilation=2).double()
    my_layer = TransposedConvolution(params.in_channels,
                                     params.out_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     dilation=2).double()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output = pytorch_layer(input)
    my_output = my_layer(input)

    # Run a numerical gradient check
    gradcheck_res = gradcheck(my_layer, (input))

    # Ensure our implementation passes gradcheck
    assert gradcheck_res

    # Make sure they match
    testing.assert_allclose(my_output, pytorch_output)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# GPU TESTS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_standard_conv_cuda():
    '''Simply compares our result to PyTorch's implementation'''

    input = torch.ones(params.bs, params.in_channels, 8, 8).double().cuda()
    input.requires_grad = True

    pytorch_layer = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=2).double().cuda()
    my_layer = Convolution(in_channels,
                           out_channels,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           dilation=2).double().cuda()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output = pytorch_layer(input)
    my_output = my_layer(input)

    # Run a numerical gradient check
    gradcheck_res = gradcheck(my_layer, (input))

    # Ensure our implementation passes gradcheck
    assert gradcheck_res

    # Make sure they match
    testing.assert_allclose(my_output, pytorch_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_standard_transposed_conv_cuda():
    '''Simply compares our result to PyTorch's implementation'''

    input = torch.ones(params.bs, params.in_channels, 5, 5).double().cuda()
    input.requires_grad = True

    pytorch_layer = torch.nn.ConvTranspose2d(params.in_channels,
                                             params.out_channels,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1,
                                             dilation=2).double().cuda()
    my_layer = TransposedConvolution(params.in_channels,
                                     params.out_channels,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     dilation=2).double().cuda()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output = pytorch_layer(input)
    my_output = my_layer(input)

    # Run a numerical gradient check
    gradcheck_res = gradcheck(my_layer, (input))

    # Ensure our implementation passes gradcheck
    assert gradcheck_res

    # Make sure they match
    testing.assert_allclose(my_output, pytorch_output)