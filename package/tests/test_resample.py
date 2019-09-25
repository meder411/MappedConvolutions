import torch
import torch.testing as testing

import pytest

from mapped_convolution.nn import Unresample

import utils
import parameters as params

bs = 3
channels = 3
kernel_size = 4


def test_unresample_nearest_integer_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_nearest_real_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[1, 6], [10, 0], [0, 11]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_nearest_out_of_bounds_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [4, 0]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bilinear_integer_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bilinear_real_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0.5, 5.5], [9.5, 5.5], [1.5,
                                                            10.5]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bilinear_out_of_bounds_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [3, 0]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bispherical_integer_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bispherical_real_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0.5, 5.5], [9.5, 9.5], [1.5,
                                                            10.5]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_bispherical_out_of_bounds_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [4.75, 11]]).double()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


def test_unresample_weighted_sampling_cpu():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_4x7().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map5(),
        interp_weights=params.interp_weights0(),
        cuda=False)

    # Manually computed correct result
    correct_output = torch.tensor([[14, 15.8], [9.6, 19.1]]).double()

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
def test_unresample_nearest_integer_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_nearest_real_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[1, 6], [10, 0], [0, 11]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_nearest_out_of_bounds_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('nearest')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [4, 0]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bilinear_integer_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bilinear_real_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0.5, 5.5], [9.5, 5.5],
                                   [1.5, 10.5]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bilinear_out_of_bounds_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bilinear')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [3, 0]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bispherical_integer_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map6(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 5], [9, 11], [3, 10]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bispherical_real_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map7(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0.5, 5.5], [9.5, 9.5],
                                   [1.5, 10.5]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_bispherical_out_of_bounds_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_3x4().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map8(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[0, 0], [0, 0], [4.75, 11]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='CUDA not detected on system')
def test_unresample_weighted_sampling_cuda():

    # Basic Unresample layer
    layer = Unresample('bispherical')

    # Run a forward and backward pass
    output, forward_time, backward_time, gradcheck_res = utils.mapped_resample_test(
        layer,
        input=params.input_4x7().repeat(bs, channels, 1, 1),
        sample_map=params.sample_map5(),
        interp_weights=params.interp_weights0(),
        cuda=True)

    # Manually computed correct result
    correct_output = torch.tensor([[14, 15.8], [9.6, 19.1]]).double().cuda()

    # Assert gradient check has passed
    assert gradcheck_res

    # Assert outputs match
    testing.assert_allclose(output, correct_output)