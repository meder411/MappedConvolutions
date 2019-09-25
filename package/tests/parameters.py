import torch

bs = 3
in_channels = 2
out_channels = 3
kernel_size = 4
eps = 1e-6


def input_ones():
    return torch.ones(1, 1, 4, 5).double()


def input_4x5():
    return torch.arange(20).view(1, 1, 4, 5).double()


def input_4x7():
    return torch.arange(28).view(1, 1, 4, 7).double()


def input_3x4():
    return torch.arange(12).view(1, 1, 3, 4).double()


def transposed_ones():
    return torch.ones(1, 1, 2, 2).double()


def transposed_input_2x2():
    return torch.arange(4).view(1, 1, 2, 2).double()


def transposed_input_2x3():
    return torch.arange(6).view(1, 1, 2, 3).double()


def weights_unit(in_channels, out_channels):
    return torch.ones(out_channels, in_channels, kernel_size).double()


def weights_0_25(in_channels, out_channels):
    w = torch.ones(out_channels, in_channels, kernel_size).double()
    w[..., :kernel_size // 2] *= 0.25
    return w


def transposed_weights_unit(in_channels, out_channels):
    return torch.ones(in_channels, out_channels, kernel_size).double()


def transposed_weights_0_25(in_channels, out_channels):
    return 0.25 * torch.ones(in_channels, out_channels, kernel_size).double()


def gradients(forward_output):
    return torch.ones_like(forward_output).double()


def bias(out_channels):
    return torch.tensor([2.0]).repeat(out_channels).double()


def sample_map0():
    '''
    returns 4 x 5 x 4 x 2
    Simpler view pictured: each row is a set of samples from the input
    Each row's linear index is the index of the output
    [11,  8,  2,  9],
    [ 2, 19,  0,  4],
    [ 5,  3, 14,  9],
    [ 5, 10,  6, 18],
    [ 5,  3, 13, 12],
    [18,  9,  6, 16],
    [ 6,  3, 14, 17],
    [19, 18,  2,  1],
    [18, 10,  9, 17],
    [14,  0, 19, 10],
    [10,  9, 19,  8],
    [ 7, 12, 11,  5],
    [12,  0, 16, 19],
    [ 4,  5, 10,  7],
    [ 6,  8,  2, 17],
    [10,  5, 18, 17],
    [ 3, 10, 19,  4],
    [ 4,  6,  7, 10],
    [17, 10,  6,  7],
    [12,  9,  7, 17],

    This input mapping resamples the inputs at integer locations
    '''

    sample_map0 = torch.zeros(4, 5, 4, 2)

    # X Cooord
    sample_map0[..., 0] = torch.tensor([[[1, 3, 2, 4], [2, 4, 0,
                                                        4], [0, 3, 4, 4],
                                         [0, 0, 1, 3], [0, 3, 3, 2]],
                                        [[3, 4, 1, 1], [1, 3, 4,
                                                        2], [4, 3, 2, 1],
                                         [3, 0, 4, 2], [4, 0, 4, 0]],
                                        [[0, 4, 4, 3], [2, 2, 1,
                                                        0], [2, 0, 1, 4],
                                         [4, 0, 0, 2], [1, 3, 2, 2]],
                                        [[0, 0, 3, 2], [3, 0, 4,
                                                        4], [4, 1, 2, 0],
                                         [2, 0, 1, 2], [2, 4, 2, 2]]])

    # Y Coord
    sample_map0[..., 1] = torch.tensor([[[2, 1, 0, 1], [0, 3, 0,
                                                        0], [1, 0, 2, 1],
                                         [1, 2, 1, 3], [1, 0, 2, 2]],
                                        [[3, 1, 1, 3], [1, 0, 2,
                                                        3], [3, 3, 0, 0],
                                         [3, 2, 1, 3], [2, 0, 3, 2]],
                                        [[2, 1, 3, 1], [1, 2, 2,
                                                        1], [2, 0, 3, 3],
                                         [0, 1, 2, 1], [1, 1, 0, 3]],
                                        [[2, 1, 3, 3], [0, 2, 3,
                                                        0], [0, 1, 1, 2],
                                         [3, 2, 1, 1], [2, 1, 1, 3]]])

    return sample_map0.double()


def sample_map1():
    '''
    returns 4 x 5 x 4 x 2
    This input mapping is the same at sample_map0 except shifted 0.5 right and 0.5 down, resulting in bilinear interpolation and "sampling" from padding
    '''
    sample_map1 = sample_map0() + 0.5
    return sample_map1


def sample_map2():
    '''
    returns 2 x 2 x 4 x 2
    Simpler view pictured: each row is a set of samples from the input
    Each rows linear index is the index of the output
    [17, 19, 11, 1],
    [12, 13, 17, 8],
    [0, 19, 5, 4],
    [16, 4, 18, 19],

    This input mapping down-samples the image and samples at integer locations
    '''
    sample_map2 = torch.zeros(2, 2, 4, 2)

    sample_map2[..., 0] = torch.tensor([[[2, 4, 1, 1], [2, 3, 2, 3]],
                                        [[0, 4, 0, 4], [1, 4, 3, 4]]])
    sample_map2[..., 1] = torch.tensor([[[3, 3, 2, 0], [2, 2, 3, 1]],
                                        [[0, 3, 1, 0], [3, 0, 3, 3]]])

    return sample_map2.double()


def sample_map25():

    sample_map25 = torch.zeros(2, 3, 4, 2)

    sample_map25[..., 0] = torch.tensor([[[0, 4, 2, 3], [2, 4, 0, 2],
                                          [2, 3, 0, 2]],
                                         [[2, 0, 4, 0], [4, 3, 0, 4],
                                          [2, 4, 2, 0]]])
    sample_map25[..., 1] = torch.tensor([[[1, 3, 3, 1], [2, 0, 1, 0],
                                          [0, 1, 0, 2]],
                                         [[3, 1, 0, 2], [0, 1, 1, 2],
                                          [1, 0, 2, 2]]])

    return sample_map25.double()


def sample_map3():
    '''
    returns 2 x 2 x 4 x 2

    This input mapping down-samples the image and samples at a location shifted left by 0.5
    '''
    sample_map3 = sample_map2()
    sample_map3[..., 0] -= 0.5
    return sample_map3


def sample_map4():
    '''
    returns 4 x 5 x 4 x 2
    Simpler view pictured: each row is a set of samples from the input
    Each row's linear index is the index of the output
    [ 4,  7, 10, 12]
    [11, 22, -2, 12]
    [ 8,  1, 17,  8]
    [ 8, 13, 10, 26]
    [ 9,  2, 21, 11]
    [16,  2,  9, 20]
    [15,  1, 13, 16]
    [18, 27,  5, -1]
    [17,  9,  3, 26]
    [17,  3, 17, 13]
    [ 9, 12, 23,  6]
    [ 0, 20,  4, -1]
    [10,  9, 14, 12]
    [ 2,  4, 14, 11]
    [ 9,  6, 11, 21]
    [ 9, -1, 21, 15]
    [11, 18, 22,  3]
    [ 2, -1, 15, 13]
    [20, 13,  9, 15]
    [11,  3, 11, 20]

    This mapping test the boundary cases where the samples fall outside the valid region of the image
    '''

    sample_map4 = torch.zeros(4, 5, 4, 2)

    sample_map4[..., 0] = torch.tensor([[[-1, 2, 0, 2], [1, 2, -2, 2],
                                         [-2, 1, 2, 3], [-2, -2, 0, 1],
                                         [-1, 2, 1, 1]],
                                        [[1, 2, -1, 0], [0, 1, 3, 1],
                                         [3, 2, 0, -1], [2, -1, 3, 1],
                                         [2, -2, 2, -2]],
                                        [[-1, 2, 3, 1], [0, 0, -1, -1],
                                         [0, -1, -1, 2], [2, -1, -1, 1],
                                         [-1, 1, 1, 1]],
                                        [[-1, -1, 1, 0], [1, -2, 2, 3],
                                         [2, -1, 0, -2], [0, -2, -1, 0],
                                         [1, 3, 1, 0]]])

    sample_map4[..., 1] = torch.tensor([[[1, 1, 2, 2], [2, 4, 0,
                                                        2], [2, 0, 3, 1],
                                         [2, 3, 2, 5], [2, 0, 4, 2]],
                                        [[3, 0, 2, 4], [3, 0, 2,
                                                        3], [3, 5, 1, 0],
                                         [3, 2, 0, 5], [3, 1, 3, 3]],
                                        [[2, 2, 4, 1], [0, 4, 1,
                                                        0], [2, 2, 3, 2],
                                         [0, 1, 3, 2], [2, 1, 2, 4]],
                                        [[2, 0, 4, 3], [2, 4, 4,
                                                        0], [0, 0, 3, 3],
                                         [4, 3, 2, 3], [2, 0, 2, 4]]])

    return sample_map4.double()


def sample_map5():
    '''
    # For use with input_4x7() for doubly mapped convolution
    [0,3,9,10]
    [11,19,25,23]
    [7,17,25,27]
    [16,18,24,13]
    '''

    sample_map5 = torch.tensor([[[[2, 0], [5, 1], [6, 2]],
                                 [[0, 3], [0, 1], [5, 2]]],
                                [[[3, 3], [4, 1], [3, 0]],
                                 [[2, 1], [2, 2], [4, 3]]]]).double()

    return sample_map5


def sample_map6():
    '''
    For unresampling
    '''

    sample_map6 = torch.tensor([[[0, 0], [1, 1]], [[1, 2], [3, 2]],
                                [[3, 0], [2, 2]]])

    return sample_map6.double()


def sample_map7():
    '''
    For unresampling
    '''

    sample_map7 = torch.tensor([[[0.5, 0], [1.5, 1]], [[1.5, 2], [3.5, 2]],
                                [[3.5, 0], [2.5, 2]]])

    return sample_map7.double()


def sample_map8():
    '''
    For unresampling
    '''

    sample_map8 = torch.tensor([[[-1, -1], [-1, -1]], [[-1.5, -2], [-2, 3]],
                                [[-0.25, 1], [7, 2]]])

    return sample_map8.double()


def interp_weights0():
    '''
    Vertices:
    [6,0]
    [6,3]
    [0,3]
    [0,0]

    Vertex Mapping:
    [3,0,2], [3,0,2], [3,0,2], [3,0,2]
    [3,0,2], [0,1,2], [0,1,2], [0,1,2]
    [3,0,2], [0,1,2], [0,1,2], [0,1,2]
    [3,0,2], [0,1,2], [0,1,2], [0,1,2]
    '''
    return torch.tensor([[[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
                         [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]]).double()
