import torch

import time


def time_cuda(func, param_list):
    '''
	Times the GPU runtime of a function, returning the output and wall-clock time.
	'''
    torch.cuda.synchronize()
    s = time.time()
    output = func(*param_list)
    torch.cuda.synchronize()
    t = time.time() - s
    return output, t


def batched_index_select(input, dim, index):
    '''
	input: B x * x ... x *
	dim: scalar
	index: B x M
	'''
    dim %= input.dim()
    if index.dim() == 2:
        views = [input.shape[0]] + \
         [1 if i != dim else -1 for i in range(1, input.dim())]
    elif index.dim() == 1:
        views = [1 if i != dim else -1 for i in range(input.dim())]
    else:
        assert False, 'index must have 1 or 2 dimensions ({})'.format(
            index.dim())
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def batched_scatter(input, dim, index):
    '''
	input: B x * x ... x *
	dim: scalar
	index: M
	'''
    dim %= input.dim()
    views = [1 if i != dim else -1 for i in range(input.dim())]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    z = torch.zeros_like(input)
    return z.scatter_(dim, index, input)