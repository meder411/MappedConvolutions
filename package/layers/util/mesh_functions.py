import torch

import _mapped_convolution_ext._sphere as sphere
from mapped_convolution.nn import Unresample, Resample

from .conversions import *


def get_conv_operator_samples(icosphere, deg, dilation=False):
    assert deg > 0 and deg < 3, 'Only 1 and 2 degree operators are implemented'
    if deg == 1:
        op = sphere.get_conv_operator_deg1()
    else:
        op = sphere.get_conv_operator_deg2(dilation)

    barycenters = sphere.get_face_barycenters()
    lonlat = convert_3d_to_spherical(barycenters)  # (lon, lat)
    samples = lonlat[op]
    return convert_spherical_to_image(samples, samples.shape[:2])


def get_downsample_map(hi_res_sphere, low_res_order):
    # Thanks to the way loop subdivision is implemented it's as simple as rearranging an 'arange' call to downsample
    num_hi_res_faces = 20 * (4**(low_res_order + 1))
    op = torch.arange(num_hi_res_faces).view(-1, 4)

    barycenters = hi_res_sphere.get_face_barycenters()
    lonlat = convert_3d_to_spherical(barycenters)  # (lon, lat)
    samples = lonlat[op]
    return convert_spherical_to_image(samples, samples.shape[:2])


def vertices_to_faces(vertices, face_vertex_indices):
    '''
    vertices: B x 3 x 1 x V
    face_vertex_indices: F x 3

    returns B x 3 x 1 x F average of the face vertices
    '''
    unresampler = Unresample('nearest')
    sample_map = torch.stack(
        (face_vertex_indices, torch.zeros_like(face_vertex_indices)),
        -1).unsqueeze(0).float()
    interp_weights = torch.ones(sample_map.shape[:-1]).float() / 3
    if vertices.is_cuda:
        interp_weights = interp_weights.to(vertices.get_device())
    return unresampler(vertices, sample_map, interp_weights)


def faces_to_vertices(faces, adj_face_indices):
    '''
    faces: B x 3 x 1 x F
    adj_face_indices: V x 6

    returns B x 3 x 1 x F average of the face vertices
    '''

    unresampler = Unresample('nearest')
    sample_map = torch.stack(
        (adj_face_indices, torch.zeros_like(adj_face_indices)),
        -1).unsqueeze(0).float()
    interp_weights = torch.ones(sample_map.shape[:-1]).float() / 3
    if faces.is_cuda:
        interp_weights = interp_weights.to(faces.get_device())
    return unresampler(faces, sample_map, interp_weights) / 2
