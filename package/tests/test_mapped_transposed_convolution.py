import torch
import torch.testing as testing

import pytest

from mapped_convolution.nn import MappedTransposedConvolution

import utils
import parameters as params

bs = 3
in_channels = 2
out_channels = 3
kernel_size = 4


def test_integer_sampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x2().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map2(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[
        2, 0, 0, 0, 5
    ], [2, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 3, 1, 3, 5]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_bilinear_interpolation_sampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x2().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map3(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[1.0, 0.0, 0.0, 2.5, 2.5], [1.0, 0.0, 0.5, 0.5, 0.0],
         [0.0, 0.5, 1.0, 0.5, 0.0], [1.5, 2.0, 2.0, 4.0, 2.5]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_integer_sampling_non_square_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x3().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map25(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[
        2, 0, 3, 0, 13
    ], [8, 0, 5, 6, 0], [8, 0, 8, 0, 4], [0, 0, 3, 0, 0]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# GPU TESTS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_integer_sampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x2().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map2(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[
        2, 0, 0, 0, 5
    ], [2, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 3, 1, 3, 5]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_bilinear_interpolation_sampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x2().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map3(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[1.0, 0.0, 0.0, 2.5, 2.5], [1.0, 0.0, 0.5, 0.5, 0.0],
         [0.0, 0.5, 1.0, 0.5, 0.0], [1.5, 2.0, 2.0, 4.0, 2.5]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_integer_sampling_non_square_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedTransposedConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_transposed_conv_test(
        layer,
        weight=params.transposed_weights_unit(in_channels, out_channels),
        input=params.transposed_input_2x3().repeat(bs, in_channels, 1, 1),
        oh=4,
        ow=5,
        sample_map=params.sample_map25(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[
        2, 0, 3, 0, 13
    ], [8, 0, 5, 6, 0], [8, 0, 8, 0, 4], [0, 0, 3, 0, 0]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)