from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="rephine_ext",
    ext_modules=[
        CppExtension(
            name="rephine_ext",
            sources=["rephine.cpp"],
            extra_compile_args=["-fopenmp","-D_GLIBCXX_USE_CXX11_ABI=1"],
            extra_link_args=["-fopenmp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

