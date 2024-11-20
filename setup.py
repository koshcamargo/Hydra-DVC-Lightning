from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name="rowmean",
    version="0.1",
    ext_modules=[
        Pybind11Extension("rowmean", ["bindings.cpp"])
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
