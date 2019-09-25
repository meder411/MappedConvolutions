import torch
import math

import numpy as np

import _mapped_convolution_ext._cube2rect as _cube2rect

from .conversions import *
from .grids import *


def get_projection_map(shape, kernel_size, proj='grid', stride=1, dilation=1):
    if proj == 'gnomonic':
        kernel_map = reverse_gnomonic_projection_map(shape, kernel_size)
        kernel_map = convert_spherical_to_image(kernel_map,
                                                kernel_map.shape[:2])
    elif proj == 'equirectangular':
        kernel_map = reverse_equirectangular_projection_map(shape, kernel_size)
        kernel_map = convert_spherical_to_image(kernel_map,
                                                kernel_map.shape[:2])
    else:
        kernel_map = grid_projection_map(shape, kernel_size, dilation)
    return kernel_map[::stride, ::stride, ...]


def get_reverse_cube_projection_map(cube_dim, kernel_size, stride=1):
    spherical = reverse_cubemap_projection_map(cube_dim, kernel_size)
    uv, idx = convert_3d_to_cube(convert_spherical_to_3d(spherical), cube_dim)
    kernel_map = cubemap_idx_to_xy(uv, idx, cube_dim)
    return kernel_map[::stride, ::stride, ...]


def get_perspective_projection_map(shape, kernel_size, stride=1, dilation=1):
    kernel_map = perspective_projection_map(shape, kernel_size, dilation)
    return kernel_map[::stride, ::stride, ...]


def grid_projection_map(shape, kernel_size, dilation=1):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lat, lon, res_lat, res_lon = equirectangular_meshgrid(shape)

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Compute the projection onto the tangent plane at the equator
            y[i, j] = cur_i * dilation * res_lat
            x[i, j] = cur_j * dilation * res_lon

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the "projection"
    out_lat = y + lat
    out_lon = x + lon

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_lon, out_lat), -1)


def perspective_projection_map(shape, kernel_size, dilation):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    r, c = torch.meshgrid(torch.arange(H), torch.arange(W))

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Compute the projection onto the tangent plane at the equator
            y[i, j] = cur_i * dilation
            x[i, j] = cur_j * dilation

    # Equalize views
    r = r.view(H, W, 1).float()
    c = c.view(H, W, 1).float()
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    out_r = y + r
    out_c = x + c

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_c, out_r), -1)


def equirectangular_meshgrid(shape):
    H, W = shape
    lat = torch.linspace(math.pi / 2, -math.pi / 2,
                         steps=H).view(-1, 1).expand(-1, W)
    lon = torch.linspace(-math.pi, math.pi,
                         steps=W + 1)[:-1].view(1, -1).expand(H, -1)
    res_lat = math.pi / (H - 1)
    res_lon = 2 * math.pi / W
    return lat, lon, res_lat, res_lon


def reverse_equirectangular_projection_map(shape, kernel_size):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lat, lon, res_lat, res_lon = equirectangular_meshgrid(shape)

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Project the equirectangular image onto the tangent plane at the equator
            x[i, j] = cur_j * res_lon
            y[i, j] = cur_i * res_lat

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    out_lat = y + lat
    out_lon = x / out_lat.cos() + lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_lon, out_lat), -1)


def reverse_gnomonic_projection_map(shape, kernel_size):
    # For convenience
    H, W = shape
    kh, kw = kernel_size

    # Get lat/lon mesh grid and resolution
    lat, lon, res_lat, res_lon = equirectangular_meshgrid(shape)

    # Kernel
    x = torch.zeros(kernel_size)
    y = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)

            # Project the equirectangular image onto the tangent plane at the equator
            x[i, j] = cur_j * res_lon
            y[i, j] = cur_i * res_lat

    # Equalize views
    lat = lat.view(H, W, 1)
    lon = lon.view(H, W, 1)
    x = x.view(1, 1, kh * kw)
    y = y.view(1, 1, kh * kw)

    # Compute the projection back onto sphere
    rho = (x**2 + y**2).sqrt()
    nu = rho.atan()
    out_lat = (nu.cos() * lat.sin() + y * nu.sin() * lat.cos() / rho).asin()
    out_lon = lon + torch.atan2(
        x * nu.sin(),
        rho * lat.cos() * nu.cos() - y * lat.sin() * nu.sin())

    # Handle the (0,0) case
    out_lat[..., [kh * kw // 2]] = lat
    out_lon[..., [kh * kw // 2]] = lon

    # Compensate for longitudinal wrap around
    out_lon = ((out_lon + math.pi) % (2 * math.pi)) - math.pi

    # Return OH, OW, KH*KW, 2
    return torch.stack((out_lon, out_lat), -1)


def cube_to_rect(cubemap, rect_height, rect_width, interpolation='nearest'):
    '''
    cubemap: (1 x cube_dim x 6*cube_dim) cubemap as input with faces laid out as [-z, -x, +z, +x, +y, -y].
    '''
    assert len(cubemap.shape) == 3, 'Cubemape must have 3 dimensions'

    if interpolation == 'nearest':
        interp = 0
    elif interpolation == 'bilinear':
        interp = 1
    else:
        assert False, 'Invalid interpolation option'

    # Compute and return equirectangular image
    return _cube2rect.cube2rect(cubemap, rect_height, rect_width, interp)


def cube_to_3d(depth_cubemap):
    '''
    depth_cubemap: (1 x cube_dim x 6*cube_dim) cubemap as input with faces laid out as [-z, -x, +z, +x, +y, -y].
    '''
    cube_dim = depth_cubemap.shape[1]
    xyz = torch.ones(3, cube_dim, 6 * cube_dim).to(depth_cubemap.get_device())
    for i in range(6):

        # Grid from [-1, 1]
        v, u = torch.meshgrid(torch.arange(cube_dim), torch.arange(cube_dim))
        u = 2. * u.float() / cube_dim - 1.
        v = 2. * v.float() / cube_dim - 1.

        if i == 0:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = -u
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = v
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = -1
        elif i == 1:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = -1
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = v
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = u
        elif i == 2:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = u
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = v
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = 1
        elif i == 3:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = 1
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = v
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = -u
        elif i == 4:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = u
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = 1
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = -v
        elif i == 5:
            xyz[0, :, i * cube_dim:(i + 1) * cube_dim] = u
            xyz[1, :, i * cube_dim:(i + 1) * cube_dim] = -1
            xyz[2, :, i * cube_dim:(i + 1) * cube_dim] = v

    xyz = F.normalize(xyz, dim=0)
    return xyz * depth_cubemap


def reverse_cubemap_projection_map(cube_dim, kernel_size):
    '''
    Gives points on cube map in terms of spherical coordinates. In other words, indexes the cube map by lat/lon
    '''

    # For convenience
    H = cube_dim
    W = 6 * cube_dim
    kh, kw = kernel_size

    # Get the UV meshgrid and angular resolution
    v, u, idx = cube_meshgrid(cube_dim)
    res = cube_meshgrid_resolution(cube_dim)

    # Convert UV to spherical coordinates
    uv = torch.stack((u, v), -1).float()
    lonlat = convert_3d_to_spherical(convert_cube_to_3d(uv, idx, cube_dim))

    # Kernel offsets in spherical coordinates
    dx = torch.zeros(kernel_size)
    dy = torch.zeros(kernel_size)
    for i in range(kh):
        cur_i = i - (kh // 2)
        for j in range(kw):
            cur_j = j - (kw // 2)
            dx[i, j] = cur_j * res
            dy[i, j] = cur_i * res

    # Equalize views
    lon = lonlat[..., 0].view(H, W, 1)
    lat = lonlat[..., 1].view(H, W, 1)
    dx = dx.view(1, 1, kh * kw)
    dy = dy.view(1, 1, kh * kw)

    # Simply add the kernel and projection back onto the cubemap
    out_lat = lat + dy
    out_lon = lon + dx

    # Return the indices in terms of spherical coords
    return torch.stack((out_lon, out_lat), -1)


def cubemap_idx_to_xy(uv, idx, cube_dim):
    '''Returns X, Y coords on the dim x 6*dim cubemap image'''
    u = uv[..., 0]
    v = uv[..., 1]
    idx = idx[..., 0]
    y = v
    x = idx.float() * cube_dim + u
    return torch.stack((x, y), -1)


def perspective_resampling(fov, width, height, theta):
    """Generates (longitude in +/-pi/2, latitude in +/-pi/2) spherical
       coordinates for a given virtual perspective camera"""
    f = 0.5 * width / np.tan(0.5 * np.radians(fov))  # focal length in pixels
    x, y = np.meshgrid(np.linspace(0.5, width - 0.5, width),
                       np.linspace(0.5, height - 0.5, height))
    x = (x - 0.5 * width) / f
    y = (y - 0.5 * height) / f
    z = 1 / np.sqrt(x**2 + y**2 + 1)
    x *= z
    y *= z
    theta = -np.radians(theta)
    x = np.cos(theta) * x - np.sin(theta) * z
    z = np.sin(theta) * x + np.cos(theta) * z
    lonlat = convert_3d_to_spherical(torch.from_numpy(np.dstack((x, y, z))))
    return lonlat
