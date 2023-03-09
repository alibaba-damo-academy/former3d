import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    
    setup(
        name="former3d", 
        version="0.1", 
        author="Weihao Yuan", 
        packages=["former3d"],
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='ops_cuda',
                module='former3d.net3d.ops',
                sources=[
                    'src/votr_api.cpp',
                    'src/build_mapping.cpp',
                    'src/build_mapping_gpu.cu',
                    'src/build_attention_indices.cpp',
                    'src/build_attention_indices_gpu.cu',
                    'src/group_features.cpp',
                    'src/group_features_gpu.cu',
                ],
            ),
            ],
        )
