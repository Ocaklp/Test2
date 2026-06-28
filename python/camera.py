"""
camera.py – Simple first-person fly camera for the voxel world renderer.

Produces a view matrix and projection matrix as numpy arrays.
Does NOT depend on PyOpenGL directly so it can be tested headlessly.
"""

import math
import numpy as np


class Camera:
    """
    First-person camera with yaw/pitch control.

    Attributes
    ----------
    position : np.ndarray  shape (3,) float32
    yaw      : float  degrees, rotation around Y axis (left/right)
    pitch    : float  degrees, rotation around X axis (up/down), clamped ±89°
    fov      : float  vertical field-of-view in degrees
    near     : float  near clip plane distance
    far      : float  far clip plane distance
    """

    def __init__(
        self,
        position: tuple = (0.0, 40.0, 0.0),
        yaw: float = -90.0,
        pitch: float = -30.0,
        fov: float = 70.0,
        near: float = 0.1,
        far: float = 2000.0,
    ):
        self.position = np.array(position, dtype=np.float32)
        self.yaw   = yaw
        self.pitch = pitch
        self.fov   = fov
        self.near  = near
        self.far   = far

    # ── Direction vectors ──────────────────────────────────────────────────

    def _front(self) -> np.ndarray:
        yr = math.radians(self.yaw)
        pr = math.radians(self.pitch)
        f = np.array([
            math.cos(pr) * math.cos(yr),
            math.sin(pr),
            math.cos(pr) * math.sin(yr),
        ], dtype=np.float32)
        n = np.linalg.norm(f)
        return f / n if n > 1e-9 else f

    def _right(self) -> np.ndarray:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(self._front(), world_up)
        n = np.linalg.norm(r)
        return r / n if n > 1e-9 else r

    def _up(self) -> np.ndarray:
        return np.cross(self._right(), self._front())

    # ── Movement ───────────────────────────────────────────────────────────

    def move_forward(self, speed: float) -> None:
        self.position += self._front() * speed

    def move_right(self, speed: float) -> None:
        self.position += self._right() * speed

    def move_up(self, speed: float) -> None:
        self.position += np.array([0.0, 1.0, 0.0], dtype=np.float32) * speed

    def rotate(self, dyaw: float, dpitch: float) -> None:
        self.yaw   += dyaw
        self.pitch  = max(-89.0, min(89.0, self.pitch + dpitch))

    # ── Matrices ───────────────────────────────────────────────────────────

    def view_matrix(self) -> np.ndarray:
        """Return a 4×4 column-major view matrix (float32)."""
        f = self._front()
        r = self._right()
        u = self._up()
        p = self.position

        m = np.eye(4, dtype=np.float32)
        m[0, 0] =  r[0]; m[0, 1] =  r[1]; m[0, 2] =  r[2]
        m[1, 0] =  u[0]; m[1, 1] =  u[1]; m[1, 2] =  u[2]
        m[2, 0] = -f[0]; m[2, 1] = -f[1]; m[2, 2] = -f[2]
        m[0, 3] = -float(np.dot(r, p))
        m[1, 3] = -float(np.dot(u, p))
        m[2, 3] =  float(np.dot(f, p))
        return m

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """Return a 4×4 column-major perspective projection matrix (float32)."""
        f   = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        n, far = self.near, self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + n) / (n - far)
        m[2, 3] = (2 * far * n) / (n - far)
        m[3, 2] = -1.0
        return m

    def mvp(self, aspect: float, model: np.ndarray | None = None) -> np.ndarray:
        """Return Model-View-Projection matrix (4×4 float32, column-major)."""
        proj = self.projection_matrix(aspect)
        view = self.view_matrix()
        if model is None:
            model = np.eye(4, dtype=np.float32)
        return proj @ view @ model

    def forward_vector(self) -> np.ndarray:
        """Return normalized forward direction (float32)."""
        return self._front().copy()

    # ── Position in chunk space ────────────────────────────────────────────

    def chunk_pos(self, chunk_size: int = 32) -> tuple:
        """Return camera position in chunk-space as (x, y, z) floats."""
        return (
            float(self.position[0]) / chunk_size,
            float(self.position[1]) / chunk_size,
            float(self.position[2]) / chunk_size,
        )
