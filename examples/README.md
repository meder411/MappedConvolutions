## Examples

Currently there are two examples in this directory showing how to use the resampling operations to go from an image to (spherical) mesh vertex format.

 - **resample_cube_to_sphere_with_depth**: Takes an RGBD cube map input and produces a water-tight mesh of the scene. Subsequently, it renders an equirectangular RGB image as well as an equirectangular surface normal map.

 - **resample_rect_to_sphere**: Takes an RGB equirectangular input and resamples it to the vertices of an icosphere
