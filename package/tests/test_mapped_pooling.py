import torch
import torch.testing as testing

import pytest

from mapped_convolution.nn import MappedMaxPool, MappedAvgPool

import utils
import parameters as params

bs = 3
in_channels = 2
out_channels = 3
kernel_size = 4


def test_max_pool_integer_sampling_cpu():

    # Basic MaxPool layer
    layer = MappedMaxPool(kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map0(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[11, 19, 14, 18, 13], [18, 17, 19, 18, 19],
                                   [19, 12, 19, 10, 17], [18, 19, 10, 17,
                                                          17]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_max_pool_bilinear_interpolation_sampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map1(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[14, 5, 8.25, 13,
                                    16], [9.25, 9, 9.25, 13, 13],
                                   [13, 15, 15, 13, 11], [13, 13, 13, 13,
                                                          15]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_max_pool_integer_downsampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map2(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[19, 17], [19, 19]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_max_pool_bilinear_downsampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map3(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[18.5, 16.5], [18.5, 18.5]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_avg_pool_integer_sampling_cpu():

    # Basic MappedTransposedConvolution layer
    layer = MappedAvgPool(kernel_size=kernel_size).double()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map0(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor(
        [[7.5, 6.25, 7.75, 9.75, 8.25], [12.25, 10., 10., 13.5, 10.75],
         [11.5, 8.75, 11.75, 6.5, 8.25], [12.5, 9., 6.75, 10., 11.25]
         ], ).double()

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
def test_max_pool_integer_sampling_cuda():

    # Basic MaxPool layer
    layer = MappedMaxPool(kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map0(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[11, 19, 14, 18, 13], [18, 17, 19, 18, 19],
                                   [19, 12, 19, 10, 17], [18, 19, 10, 17,
                                                          17]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_max_pool_bilinear_interpolation_sampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map1(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[14, 5, 8.25, 13,
                                    16], [9.25, 9, 9.25, 13, 13],
                                   [13, 15, 15, 13, 11], [13, 13, 13, 13,
                                                          15]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_max_pool_integer_downsampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map2(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[19, 17], [19, 19]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_max_pool_bilinear_downsampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedMaxPool(kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map3(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[18.5, 16.5], [18.5, 18.5]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_avg_pool_integer_sampling_cuda():

    # Basic MappedTransposedConvolution layer
    layer = MappedAvgPool(kernel_size=kernel_size).double().cuda()

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_pool_test(
        layer,
        input=params.input_4x5().repeat(bs, 1, 1, 1),
        sample_map=params.sample_map0(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor(
        [[7.5, 6.25, 7.75, 9.75, 8.25], [12.25, 10., 10., 13.5, 10.75],
         [11.5, 8.75, 11.75, 6.5, 8.25], [12.5, 9., 6.75, 10., 11.25]
         ], ).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)