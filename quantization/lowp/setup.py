import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, 'lowp', 'csrc')
source_cuda = [os.path.join(extensions_dir, 'cuda', filename)
               for filename in ['lowp_cuda.cpp',
                                'lowp_cuda_kernel.cu',
                                'intel_cuda_kernel.cu'
                                ]
               ]

setup(
    name='lowp',
    ext_modules=[
        CUDAExtension('lowp._C', source_cuda),
    ],
    packages=find_packages(exclude=('test',)),
    cmdclass={
        'build_ext': BuildExtension
    })
