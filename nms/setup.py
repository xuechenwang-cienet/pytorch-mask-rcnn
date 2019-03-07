import os
import torch
# from torch.utils.ffi import create_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages


sources = ['src/nms.cpp']
include_dirs = ['src/']

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/nms_cuda.cpp', 'src/cuda/nms_kernel.cu']
    include_dirs += ['src/cuda/']

    setup(
            name='nms',
            ext_modules=[
                CUDAExtension(
                        name='nms',
                        include_dirs=include_dirs,
                        sources=sources,
                        extra_compile_args={'cxx': ['-g'],
                                            'nvcc': ['-O2']})
            ],
            cmdclass={
                'build_ext': BuildExtension
            })
else:
    setup(
            name='nms',
            ext_modules=[
                CppExtension(
                        name='nms',
                        include_dirs=include_dirs,
                        sources=sources,
                        extra_compile_args={'cxx': ['-g']})
            ],
            cmdclass={
                'build_ext': BuildExtension
            })
