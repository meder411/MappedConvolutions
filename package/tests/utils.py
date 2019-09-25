import torch
import torch.nn
from torch.autograd import gradcheck

import time

from mapped_convolution.nn import Convolution, TransposedConvolution
import parameters as params


def time_cuda(func, param_list):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        s = time.time()
        output = func(*param_list)
        torch.cuda.synchronize()
        t = time.time() - s
        return output, t
    else:
        s = time.time()
        output = func(*param_list)
        t = time.time() - s
        return output, t


# Closure to see gradients
grads = {}


def save_grad(name):

    def hook(grad):
        grads[name] = grad

    return hook


def print_report(test_report, correct_output, eps, cuda=False):

    if cuda:
        correct_output = correct_output.cuda()

    # Determine the forward pass is correct
    is_forward_correct = ((test_report[0] - correct_output).abs() < eps).all()
    avg_forward_error = (test_report[0] - correct_output).abs().mean()
    pass_forward_check = is_forward_correct.item() > 0.5

    print('Forward Time: ', test_report[1])
    print('Backward Time: ', test_report[2])
    print('Average Forward Error: ', avg_forward_error.item())
    print('Pass Forward Check: ', pass_forward_check)
    if not pass_forward_check:
        print('\tForward Output: ')
        print('\t', test_report[0])
        print('\n\tCorrect Output:')
        print('\t', correct_output)
        exit()
    print('Pass Grad Check: ', test_report[3])

    return pass_forward_check, test_report[3]


def mapped_conv_test(conv_layer,
                     weight,
                     input,
                     sample_map,
                     interp_weights=None,
                     cuda=False):

    # Make sure the input requires a gradient
    input.requires_grad = True

    if cuda:
        input = input.cuda()
        sample_map = sample_map.cuda()
        if interp_weights is not None:
            interp_weights = interp_weights.cuda()

    # Set the layer weights
    bias = params.bias(weight.shape[0])
    conv_layer.weight.data = weight if not cuda else weight.cuda()
    conv_layer.bias.data = bias if not cuda else bias.cuda()

    # Run a forward pass
    output, forward_time = time_cuda(conv_layer,
                                     [input, sample_map, interp_weights])

    # Run a backward pass
    _, backward_time = time_cuda(output.backward, [params.gradients(output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(conv_layer,
                                 (input, sample_map, interp_weights))

    # Return the report
    return output, forward_time, backward_time, gradcheck_result


def mapped_transposed_conv_test(conv_layer,
                                weight,
                                input,
                                oh,
                                ow,
                                sample_map,
                                interp_weights=None,
                                cuda=False):

    # Make sure the input requires a gradient
    input.requires_grad = True

    if cuda:
        input = input.cuda()
        sample_map = sample_map.cuda()

    # Set the layer weights
    bias = params.bias(weight.shape[1])
    conv_layer.weight.data = weight if not cuda else weight.cuda()
    conv_layer.bias.data = bias if not cuda else bias.cuda()

    # Run a forward pass
    output, forward_time = time_cuda(
        conv_layer, [input, oh, ow, sample_map, interp_weights])

    # Run a backward pass
    _, backward_time = time_cuda(output.backward, [params.gradients(output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(conv_layer,
                                 (input, oh, ow, sample_map, interp_weights))

    # Return the report
    return output, forward_time, backward_time, gradcheck_result


def mapped_pool_test(pool_layer,
                     input,
                     sample_map,
                     interp_weights=None,
                     cuda=False):

    # Make sure the input requires a gradient
    input.requires_grad = True

    if cuda:
        input = input.cuda()
        sample_map = sample_map.cuda()
        if interp_weights is not None:
            interp_weights = interp_weights.cuda()

    # Run a forward pass
    output, forward_time = time_cuda(pool_layer,
                                     [input, sample_map, interp_weights])

    # Run a backward pass
    _, backward_time = time_cuda(output.backward, [params.gradients(output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(pool_layer,
                                 (input, sample_map, interp_weights))

    # Return the report
    return output, forward_time, backward_time, gradcheck_result


def mapped_resample_test(resample_layer,
                         input,
                         sample_map,
                         interp_weights=None,
                         cuda=False):

    # Make sure the input requires a gradient
    input.requires_grad = True

    if cuda:
        input = input.cuda()
        sample_map = sample_map.cuda()
        if interp_weights is not None:
            interp_weights = interp_weights.cuda()

    # Run a forward pass
    output, forward_time = time_cuda(resample_layer,
                                     [input, sample_map, interp_weights])

    # Run a backward pass
    _, backward_time = time_cuda(output.backward, [params.gradients(output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(resample_layer,
                                 (input, sample_map, interp_weights))

    # Return the report
    return output, forward_time, backward_time, gradcheck_result


def standard_conv_test(cuda=False):
    '''Simply compares our result to PyTorch's implementation'''
    input = torch.ones(params.bs, params.in_channels, 8, 8).double()
    input.requires_grad = True

    pytorch_layer = torch.nn.Conv2d(params.in_channels,
                                    params.out_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=2).double()
    my_layer = Convolution(params.in_channels,
                           params.out_channels,
                           kernel_size=3,
                           stride=2,
                           padding=1,
                           dilation=2).double()

    if cuda:
        input = input.cuda()
        pytorch_layer = pytorch_layer.cuda()
        my_layer = my_layer.cuda()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output, _ = time_cuda(pytorch_layer, [input])
    my_output, forward_time = time_cuda(my_layer, [input])

    # Run a backward pass
    _, backward_time = time_cuda(my_output.backward,
                                 [torch.ones_like(pytorch_output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(my_layer, (input))

    # Return the report
    return my_output, forward_time, backward_time, gradcheck_result, pytorch_output


def standard_transposed_conv_test(cuda=False):
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

    if cuda:
        input = input.cuda()
        pytorch_layer = pytorch_layer.cuda()
        my_layer = my_layer.cuda()

    # Set the layer weights
    pytorch_layer.weight.data.fill_(1)
    pytorch_layer.bias.data.fill_(2)
    my_layer.weight.data.fill_(1)
    my_layer.bias.data.fill_(2)

    # Run a forward pass
    pytorch_output, _ = time_cuda(pytorch_layer, [input])
    my_output, forward_time = time_cuda(my_layer, [input])

    # Run a backward pass
    _, backward_time = time_cuda(my_output.backward,
                                 [torch.ones_like(pytorch_output)])

    # Run a numerical gradient check
    gradcheck_result = gradcheck(my_layer, (input))

    # Return the report
    return my_output, forward_time, backward_time, gradcheck_result, pytorch_output


def print_test_header(text):
    print('---------------' + text + '---------------')


def print_group_header(text):
    print('----------------------------------------------------')
    print('----------------------------------------------------')
    print('                     ' + text + '                     ')
    print('----------------------------------------------------')
    print('----------------------------------------------------')


def print_device_header(text):
    print('\n')
    print('====================================================')
    print('                     ' + text + '                     ')
    print('====================================================')
    print('\n\n')


def print_group_results(group, forward_checks, backward_checks):
    print('\n')
    print('****************************************************')
    print('* PASSED ALL ' + group + ' FORWARD CHECKS: ', forward_checks)
    print('* PASSED ALL ' + group + ' BACKWARD CHECKS: ', backward_checks)
    print('****************************************************')
    print('\n')


def print_device_result(device, all_checks):
    print('====================================================')
    print('* PASSED ALL ' + device + ' CHECKS: ', all_checks)
    print('====================================================')
    print('\n')
