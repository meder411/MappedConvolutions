import torch
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

compute_arch = 'compute_61'
include_dir = 'ext_modules/include'
nn_src = 'ext_modules/src/nn/cpp'
nn_src_cuda = 'ext_modules/src/nn/cuda'
util_src = 'ext_modules/src/util/cpp'
util_src_cuda = 'ext_modules/src/util/cuda'


def extension(name, package, source_basename):
    '''Create a build extension. Use CUDA if available, otherwise C++ only'''
    if package == 'nn':
        prefix = nn_src
        prefix_cuda = nn_src_cuda
    elif package == 'util':
        prefix = util_src
        prefix_cuda = util_src_cuda

    if torch.cuda.is_available():
        return CUDAExtension(name=name,
                             sources=[
                                 osp.join(prefix, source_basename + '.cpp'),
                                 osp.join(prefix_cuda, source_basename + '.cu'),
                             ],
                             include_dirs=[include_dir],
                             extra_compile_args={
                                 'cxx': ['-fopenmp', '-O3'],
                                 'nvcc': ['--gpu-architecture=' + compute_arch]
                             })
    else:
        return CppExtension(name=name,
                            sources=[
                                osp.join(prefix, source_basename + '.cpp'),
                            ],
                            include_dirs=[include_dir],
                            define_macros=[('__NO_CUDA__', None)],
                            extra_compile_args={
                                'cxx': ['-fopenmp', '-O3'],
                                'nvcc': []
                            })


setup(
    name='Mapped Convolution',
    version='0.0.2',
    author='Marc Eder',
    description='A PyTorch module for mapped convolutions',
    ext_package='_mapped_convolution_ext',
    ext_modules=[
        # ------------------------------------------------
        # Standard CNN operations
        # ------------------------------------------------
        extension('_convolution', 'nn', 'convolution_layer'),
        extension('_transposed_convolution', 'nn',
                  'transposed_convolution_layer'),

        # ------------------------------------------------
        # Mapped CNN operations
        # ------------------------------------------------
        extension('_mapped_convolution', 'nn', 'mapped_convolution_layer'),
        extension('_mapped_transposed_convolution', 'nn',
                  'mapped_transposed_convolution_layer'),
        extension('_mapped_max_pooling', 'nn', 'mapped_max_pooling_layer'),
        extension('_mapped_avg_pooling', 'nn', 'mapped_avg_pooling_layer'),
        extension('_resample', 'nn', 'resample_layer'),
        extension('_weighted_mapped_convolution', 'nn',
                  'weighted_mapped_convolution_layer'),
        extension('_weighted_mapped_transposed_convolution', 'nn',
                  'weighted_mapped_transposed_convolution_layer'),
        extension('_weighted_mapped_max_pooling', 'nn',
                  'weighted_mapped_max_pooling_layer'),
        extension('_weighted_mapped_avg_pooling', 'nn',
                  'weighted_mapped_avg_pooling_layer'),
        extension('_weighted_resample', 'nn', 'weighted_resample_layer'),
        extension('_voting_resample', 'nn', 'voting_resample_layer'),

        # ------------------------------------------------
        # Miscellaneous operations
        # ------------------------------------------------
        extension('_knn', 'util', 'knn_layer'),
        extension('_cube2rect', 'util', 'cube2rect_layer'),
        CppExtension(name='_sphere',
                     sources=[
                         osp.join(util_src, 'sphere.cpp'),
                     ],
                     include_dirs=[include_dir],
                     extra_compile_args={
                         'cxx': ['-fopenmp', '-O3'],
                         'nvcc': []
                     }),
    ],
    packages=[
        'mapped_convolution',
        'mapped_convolution.nn',
        'mapped_convolution.util',
        'mapped_convolution.metrics',
        'mapped_convolution.loss',
    ],
    package_dir={
        'mapped_convolution': 'layers',
        'mapped_convolution.nn': 'layers/nn',
        'mapped_convolution.util': 'layers/util',
        'mapped_convolution.metrics': 'layers/metrics',
        'mapped_convolution.loss': 'layers/loss',
    },
    cmdclass={'build_ext': BuildExtension},
)
