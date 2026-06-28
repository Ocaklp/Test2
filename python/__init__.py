"""
voxel_world – procedural voxel world Python package.

Exposes the C++ engine via the `voxel_core` pybind11 extension,
and provides a PyOpenGL renderer + camera for interactive viewing.
"""

from pathlib import Path
import sys
import os

# Ensure the compiled extension (next to this file) is importable.
_here = Path(__file__).parent.resolve()
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

try:
    import voxel_core  # type: ignore  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "C++ extension 'voxel_core' not found. "
        "Build it with: python setup.py build_ext --inplace"
    ) from exc

__all__ = ["voxel_core"]
