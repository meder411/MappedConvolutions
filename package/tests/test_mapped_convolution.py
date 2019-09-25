import torch
import torch.testing as testing

import pytest

from mapped_convolution.nn import MappedConvolution

import utils
import parameters as params

bs = 3
in_channels = 2
out_channels = 3
kernel_size = 4


def test_integer_sampling_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map0(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + in_channels * torch.tensor(
        [[30, 25, 31, 39, 33], [49, 40, 40, 54, 43], [46, 35, 47, 26, 33],
         [50, 36, 27, 40, 45]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_layer_weights_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_0_25(in_channels, out_channels),
        input=params.input_ones().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map0(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * 2.5 * torch.ones(1, 1, 4,
                                                               5).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_bilinear_interpolation_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map1(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[35.75, 16.00, 28.00, 39.25, 45.00], [
            32.25, 32.00, 23.00, 36.75, 29.00
        ], [34.50, 47.00, 31.00, 34.25, 33.75],
         [39.00, 27.00, 35.25, 40.75, 39.50]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_downsampling_with_integer_sampling_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map2(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[48, 50], [28, 57]
                                                            ]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_downsampling_with_bilinear_interpolation_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map3(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[46, 48], [24.5, 55]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_out_of_bounds_sampling_cpu():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map4(),
        cuda=False)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[29, 23, 26, 10, 13], [18, 45, 23, 20, 34], [18, 0, 22, 13, 17],
         [15, 14, 17, 15, 25]]).double()

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

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map0(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + in_channels * torch.tensor(
        [[30, 25, 31, 39, 33], [49, 40, 40, 54, 43], [46, 35, 47, 26, 33],
         [50, 36, 27, 40, 45]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_layer_weights_cuda():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_0_25(in_channels, out_channels),
        input=params.input_ones().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map0(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * 2.5 * torch.ones(
        1, 1, 4, 5).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_bilinear_interpolation_cuda():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map1(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[35.75, 16.00, 28.00, 39.25, 45.00], [
            32.25, 32.00, 23.00, 36.75, 29.00
        ], [34.50, 47.00, 31.00, 34.25, 33.75],
         [39.00, 27.00, 35.25, 40.75, 39.50]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_downsampling_with_integer_sampling_cuda():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map2(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor([[48, 50], [28, 57]
                                                            ]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_downsampling_with_bilinear_interpolation_cuda():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map3(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[46, 48], [24.5, 55]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_out_of_bounds_sampling_cuda():

    # Basic MappedConvolution layer
    layer = MappedConvolution(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_conv_test(
        layer,
        weight=params.weights_unit(in_channels, out_channels),
        input=params.input_4x5().repeat(bs, in_channels, 1, 1),
        sample_map=params.sample_map4(),
        cuda=True)

    # Manually computed correct result
    correct_output = 2 + params.in_channels * torch.tensor(
        [[29, 23, 26, 10, 13], [18, 45, 23, 20, 34], [18, 0, 22, 13, 17],
         [15, 14, 17, 15, 25]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)