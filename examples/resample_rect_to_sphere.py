# EXAMPLE: Resampling an equirectangular image to the vertices of an icosphere
#
# This example shows how to resample a equirectangular image to the vertices
# of an icosphere.
# =============================================================================

from mapped_convolution.util import *
from skimage import io

# Order of icosphere to generate
order = 7

# Generate an icosphere
print('Generating icosphere')
icosphere = generate_icosphere(order)

print('Loading Earth image')
# Load the equirectangular earth image and convert to torch format
# (1, 3, H, W)
img = torch.from_numpy(io.imread('inputs/earthmap4k.jpg')[..., :3]).permute(
    2, 0, 1).float().unsqueeze(0).cuda()

# Resample the image using barycentric interpolation
# (1, 3, 1, V)
print('Resampling the image data to the sphere')
rgb_vertices = resample_rgb_to_vertex(img, icosphere, order)

# Get necessary data for PLY writing
faces = icosphere.get_all_face_vertex_indices()
vertices = icosphere.get_vertices()
rgb_vertices = rgb_vertices.squeeze()

# Write to file
write_ply('outputs/earth.ply',
          vertices.transpose(0, 1).numpy(),
          rgb=rgb_vertices.cpu().numpy(),
          faces=faces.numpy(),
          text=False)
print('Earth icosphere written to `outputs/earth.ply`')
