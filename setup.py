from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

# Use pybind11 helpers when available.
# Run:  pip install . --no-build-isolation
# Or:   python setup.py build_ext --inplace   (legacy)

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    USE_PYBIND_HELPERS = True
except ImportError:
    from setuptools import Extension as Pybind11Extension
    from setuptools.command.build_ext import build_ext as build_ext
    USE_PYBIND_HELPERS = False

CPP_SRC = Path("cpp/src")
CPP_INC = Path("cpp/include")

ext_modules = [
    Pybind11Extension(
        "voxel_core",
        sources=[
            str(CPP_SRC / "noise.cpp"),
            str(CPP_SRC / "wfc.cpp"),
            str(CPP_SRC / "octree.cpp"),
            str(CPP_SRC / "chunk.cpp"),
            str(CPP_SRC / "terrain.cpp"),
            str(CPP_SRC / "lod.cpp"),
            str(CPP_SRC / "world.cpp"),
            str(CPP_SRC / "bindings.cpp"),
        ],
        include_dirs=[str(CPP_INC)],
        extra_compile_args=["-O3", "-std=c++17", "-pthread"]
        if sys.platform != "win32"
        else ["/O2", "/std:c++17"],
        extra_link_args=["-pthread"] if sys.platform != "win32" else [],
    )
]

setup(
    name="bakaliska-voxel",
    version="0.1.0",
    description="Procedural voxel world — Noise + WFC (C++) + PyOpenGL",
    packages=find_packages(where="."),
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pygame>=2.4; python_version < '3.14'",
        "pygame-ce>=2.5; python_version >= '3.14'",
        "PyOpenGL>=3.1",
        "PyOpenGL-accelerate>=3.1",
        "pybind11>=2.11",
    ],
)
