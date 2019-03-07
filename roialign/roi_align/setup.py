import os
import torch
# from torch.utils.ffi import create_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages


sources = ['src/crop_and_resize.cpp']
include_dirs = ['src/']

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/crop_and_resize_gpu.cpp', 'src/cuda/crop_and_resize_kernel.cu']
    include_dirs += ['src/cuda/']

    setup(
            name='roialign',
            ext_modules=[
                CUDAExtension(
                        name='roialign',
                        include_dirs=include_dirs,
                        sources=sources,
                        extra_compile_args={'cxx': ['-g', '-fopenmp'],
                                            'nvcc': ['-O2']})
            ],
            cmdclass={
                'build_ext': BuildExtension
            })
else:
    setup(
            name='roialign',
            ext_modules=[
                CppExtension(
                        name='roialign',
                        include_dirs=include_dirs,
                        sources=sources,
                        extra_compile_args={'cxx': ['-g', '-fopenmp']})
            ],
            cmdclass={
                'build_ext': BuildExtension
            })
