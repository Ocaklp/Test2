"""
Tests for the Python camera module (no OpenGL dependency).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "python"))

import math
import pytest
import numpy as np

from camera import Camera


class TestCamera:
    def test_default_creation(self):
        c = Camera()
        assert c.position[1] == 40.0
        assert c.fov == 70.0

    def test_view_matrix_shape(self):
        c = Camera()
        v = c.view_matrix()
        assert v.shape == (4, 4)
        assert v.dtype == np.float32

    def test_projection_matrix_shape(self):
        c = Camera()
        p = c.projection_matrix(16 / 9)
        assert p.shape == (4, 4)

    def test_mvp_shape(self):
        c = Camera()
        m = c.mvp(16 / 9)
        assert m.shape == (4, 4)

    def test_move_forward(self):
        c = Camera(position=(0, 0, 0), yaw=0, pitch=0)
        before = c.position.copy()
        c.move_forward(1.0)
        assert not np.allclose(c.position, before)

    def test_move_right(self):
        c = Camera(position=(0, 0, 0), yaw=0, pitch=0)
        before = c.position.copy()
        c.move_right(1.0)
        assert not np.allclose(c.position, before)

    def test_move_up(self):
        c = Camera(position=(0, 0, 0))
        c.move_up(5.0)
        assert abs(c.position[1] - 5.0) < 1e-5

    def test_rotate_pitch_clamp(self):
        c = Camera()
        c.rotate(0, 200)
        assert c.pitch <= 89.0
        c.rotate(0, -400)
        assert c.pitch >= -89.0

    def test_chunk_pos(self):
        c = Camera(position=(64, 32, 96))
        cx, cy, cz = c.chunk_pos(chunk_size=32)
        assert abs(cx - 2.0) < 1e-5
        assert abs(cy - 1.0) < 1e-5
        assert abs(cz - 3.0) < 1e-5

    def test_forward_vector_normalized(self):
        c = Camera(yaw=0.0, pitch=0.0)
        f = c.forward_vector()
        assert f.shape == (3,)
        assert abs(float(np.linalg.norm(f)) - 1.0) < 1e-5
