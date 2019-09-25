# EXAMPLE: Resampling an cubemap to the vertices of an depth scaled icosphere
#
# This example shows how to resample a cubemap to the vertices of an
# icosphere. We then scale the vertices according to provided depth
# information, which reshapes the mesh to the indoor scene it captures. We
# then show how to render back to an equirectangular image and to render the
# surface normals.
# =============================================================================

import torch.nn.functional as F
from mapped_convolution.util import *

from skimage import io

# =================
# PARAMETERS
# =================
order = 7  # Resolution icosphere desired
output_equirect_shape = (512, 1024)  # Output equirectangular image dims
cuda = True  # Whether to use GPU (recommended)

# -----------------------------------------------------------------------------
# Generate an icosphere
# -----------------------------------------------------------------------------
print('Generating icosphere')
icosphere = generate_icosphere(order)

# -----------------------------------------------------------------------------
# Load and process the cubemap image
# -----------------------------------------------------------------------------
print('Loading the cube map data')

# Load the multi-page TIFF image
# Channel 0: RGB
# Channel 1: Depth
# Channel 2: Sematic labels
# Channel 3: Instance labels
tiff = io.MultiImage('inputs/cubemap.tiff')

# Convert RGB image to a torch tensor with dimensions (1, 3, H, W)
cube_rgb = torch.from_numpy(tiff[0]).permute(2, 0, 1).float().unsqueeze(0)
if cuda:
    cube_rgb = cube_rgb.cuda()

# Convert depth image to torch tensor with dimensions (1, 1, H, W)
cube_inv_depth = torch.from_numpy(tiff[1].astype(
    np.int32)).float().unsqueeze(0).unsqueeze(0)
if cuda:
    cube_inv_depth = cube_inv_depth.cuda()

# Convert inverse depth to regular depth
cube_inv_depth[cube_inv_depth == 0] = -1
cube_depth = 1 / cube_inv_depth
cube_depth[cube_depth < 0] = 0

# Convert to metric scale according to min-distance = 0.3m
# This is a sample image from the SUMO dataset
scale = 0.3 * (2**16 - 1)
cube_depth *= scale

# -----------------------------------------------------------------------------
# Resample the image to the sphere
# -----------------------------------------------------------------------------
print('Resampling the image data to the sphere')

# Resample the depth cubemap using barycentric interpolation
rgb_vertices = resample_cube_to_vertex(cube_rgb, icosphere, order)

# Resample the depth cubemap using nearest-neighbor interpolation
depth_vertices = resample_cube_to_vertex(cube_depth, icosphere, order, True)

# Gather remaining info needed for the PLY
rgb_vertices = rgb_vertices.squeeze()  # (3, V)
vertices = icosphere.get_vertices()  # (V, 3)
face_idx = icosphere.get_all_face_vertex_indices()  # (F, 3)

# Write the textured sphere to file
write_ply('outputs/rgb_sphere.ply',
          vertices.transpose(0, 1).numpy(),
          rgb=rgb_vertices.cpu().numpy(),
          faces=face_idx.cpu().numpy(),
          text=False)
print('Textured icosphere written to `outputs/rgb_sphere.ply`')

# -----------------------------------------------------------------------------
# Scale the vertices according to depth
# -----------------------------------------------------------------------------
print('Scaling the vertices according to the depth data')

# Get the vertices of the icosphere (V, 3)
pts = icosphere.get_vertices()
if cuda:
    pts = pts.cuda()

# Scale the vertices by the depth values (V, 1) * (V, 3)
pts = depth_vertices.squeeze().unsqueeze(-1) * pts

# Write the resulting mesh to file
# This mesh is the result of warping the sphere according the depth values for
# each vertices
write_ply('outputs/deformed_sphere.ply',
          pts.cpu().transpose(0, 1).numpy(),
          rgb=rgb_vertices.cpu().numpy(),
          faces=face_idx.cpu().numpy(),
          text=False)
print('Deformed spherical mesh written to `outputs/deformed_sphere.ply`')

# --------------------------------------------------------------------
# Let's also resample the mesh back to an equirectangular image
# --------------------------------------------------------------------
print('Render sphere back into equirectangular image')

# Resample back to an equirectangular image
rgb_rect = resample_vertex_to_rect(rgb_vertices.view(1, 3, 1, -1),
                                   output_equirect_shape, order)

# Save the re-rendered RGB image
io.imsave('outputs/rerendered_rect.png',
          rgb_rect.squeeze().permute(1, 2, 0).byte().cpu().numpy())
print('Rendered equirectangular image written to `outputs/rerendered_rect.png`')

# --------------------------------------------------------------------
# Now that we have the mesh deformed to the proper geometry, let's also compute a surface normal map from the mesh faces
# --------------------------------------------------------------------
print('Render surface normal map into equirectangular image')

# Compute face normals
face_coords = pts[face_idx.to(pts.get_device())]  # (F, 3, 3)
a = face_coords[:, 2, :] - face_coords[:, 1, :]
b = face_coords[:, 0, :] - face_coords[:, 1, :]
face_normals = F.normalize(torch.cross(a, b, dim=-1), p=2, dim=-1)  # (F, 3)

# Compute the vertex normals by averaging the surrounding face normals (V, 3)
adj_idx = icosphere.get_adjacent_face_indices_to_vertices()
vertex_normals = F.normalize(face_normals[adj_idx.to(
    face_normals.get_device())].mean(1),
                             p=2,
                             dim=-1)

# Resample normals back to an equirectangular image to and visualize them
normals_rect = resample_vertex_to_rect(
    vertex_normals.permute(1, 0).contiguous().view(1, 3, 1, -1),
    output_equirect_shape, order)
normals_rect = F.normalize(normals_rect.squeeze(), 2, 0)

# Visualize the normals in RGB in equirectangular format
np_rect = ((normals_rect * 127.5) + 127.5).byte().permute(1, 2, 0).cpu().numpy()
io.imsave('outputs/normals_rect.png', np_rect)
print(
    'Rendered surface normals written to equirectangular image as `outputs/normals_rect.png`'
)