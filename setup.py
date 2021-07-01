from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys
import os


ext_modules = [
    Pybind11Extension("ldpc_binding",
        [os.path.join("modem", "coding", "ldpc_binding", "ldpc_binding.cpp")],
        include_dirs = ['3rdparty']
    ),
]

setup(
    name="ldpc_binding",
    author="Tobias Kronauer",
    author_email="tobias.kronauer@bi-dd.de",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)