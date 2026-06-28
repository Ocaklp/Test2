"""
single_chunk_test.py – Visualise a single terrain-generated chunk from the C++ engine.

Usage
-----
    python python/single_chunk_test.py [--seed N] [--chunk-x N] [--chunk-y N]
                                       [--chunk-z N] [--headless] [--frames N]

Controls (interactive window)
------------------------------
    W/S       – zoom in / out
    A/D       – orbit left / right
    Q/E       – orbit up / down
    Escape    – quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure python/ directory is on the path for local imports.
_here = Path(__file__).parent.resolve()
sys.path.insert(0, str(_here))

# ── Import C++ engine ──────────────────────────────────────────────────────────
try:
    import voxel_core as ve
except ImportError:
    sys.exit(
        "ERROR: Cannot import 'voxel_core'. "
        "Build it first:\n  python setup.py build_ext --inplace"
    )

from camera import Camera
from renderer import VoxelRenderer, _HAS_OPENGL

try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise a single terrain-generated chunk from the C++ engine."
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Terrain generation seed (default: 42)",
    )
    p.add_argument(
        "--chunk-x",
        type=int,
        default=0,
        help="Chunk X coordinate in chunk-space (default: 0)",
    )
    p.add_argument(
        "--chunk-y",
        type=int,
        default=0,
        help="Chunk Y coordinate in chunk-space (default: 0)",
    )
    p.add_argument(
        "--chunk-z",
        type=int,
        default=0,
        help="Chunk Z coordinate in chunk-space (default: 0)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without a window (for testing/CI)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Run for N frames then exit (0 = unlimited, interactive only)",
    )
    return p.parse_args()


# ── Single chunk generation (C++ engine) ──────────────────────────────────────

def make_single_chunk(seed: int, cx: int, cy: int, cz: int) -> ve.Chunk:
    """Generate a terrain chunk at the given chunk coordinates."""
    chunk = ve.Chunk(ve.IVec3(cx, cy, cz))
    gen = ve.TerrainGenerator(seed)
    gen.generate(chunk)
    return chunk


def print_chunk_info(chunk: ve.Chunk, seed: int, cx: int, cy: int, cz: int) -> None:
    """Print information about the generated chunk and its mesh."""
    mesh = chunk.build_mesh()
    vertex_count = len(mesh) // 7  # 7 floats per vertex: x,y,z,nx,ny,nz,biome_id
    origin = chunk.world_origin()

    print("─" * 50)
    print("Single Chunk — C++ Engine Test")
    print("─" * 50)
    print(f"  Seed          : {seed}")
    print(f"  Chunk coord   : ({cx}, {cy}, {cz})")
    print(f"  World origin  : ({origin.x}, {origin.y}, {origin.z})")
    print(f"  Solid count   : {chunk.solid_count()}")
    print(f"  Mesh vertices : {vertex_count}")
    print(f"  Mesh floats   : {len(mesh)}")
    if mesh:
        # x, y, z, nx, ny, nz, biome_id
        print(
            "  First vertex  : x={:.2f} y={:.2f} z={:.2f} "
            "nx={:.2f} ny={:.2f} nz={:.2f} biome={:.0f}".format(*mesh[:7])
        )
    print("─" * 50)


# ── Interactive visualisation ──────────────────────────────────────────────────

def run_interactive(mesh: list, cx: int, cy: int, cz: int, max_frames: int) -> None:
    """Open an OpenGL window and display the single-chunk mesh."""
    if not _HAS_PYGAME:
        print("pygame not installed – cannot run interactive visualisation.")
        return

    renderer = VoxelRenderer(
        width=800, height=600,
        title=f"Single Chunk — ({cx}, {cy}, {cz})",
    )

    # Upload the chunk mesh at its chunk coordinates.
    renderer.upload_chunk_mesh(cx, cy, cz, mesh)

    # Position the camera so it looks toward the centre of the chunk.
    chunk_size = ve.Chunk.CHUNK_SIZE
    centre_x = (cx + 0.5) * chunk_size
    centre_y = (cy + 0.5) * chunk_size
    centre_z = (cz + 0.5) * chunk_size
    camera = Camera(
        position=(centre_x + chunk_size * 2, centre_y + chunk_size, centre_z + chunk_size * 2),
        yaw=-135.0,
        pitch=-20.0,
        fov=60.0,
        near=0.1,
        far=500.0,
    )

    clock          = pygame.time.Clock()
    mouse_captured = False
    frame          = 0
    orbit_speed    = 1.0   # degrees per frame (mouse look)
    move_speed     = 0.5   # world units per frame (keyboard movement)

    print("Controls: W/S=zoom, A/D=orbit, Q/E=tilt, Esc=quit")

    running = True
    while running:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mouse_captured:
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                        mouse_captured = False
                    else:
                        running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                mouse_captured = True
            elif event.type == pygame.VIDEORESIZE:
                renderer.width  = event.w
                renderer.height = event.h

        if mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            camera.rotate(dx * orbit_speed, -dy * orbit_speed)
        else:
            pygame.mouse.get_rel()  # consume to avoid jumps

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:      camera.move_forward( move_speed)
        if keys[pygame.K_s]:      camera.move_forward(-move_speed)
        if keys[pygame.K_a]:      camera.move_right(  -move_speed)
        if keys[pygame.K_d]:      camera.move_right(   move_speed)
        if keys[pygame.K_q]:      camera.move_up(     -move_speed)
        if keys[pygame.K_e]:      camera.move_up(      move_speed)

        renderer.render(camera)
        renderer.swap()

        frame += 1
        if max_frames > 0 and frame >= max_frames:
            break

    renderer.cleanup()


# ── Headless / CI mode ─────────────────────────────────────────────────────────

def run_headless(mesh: list, cx: int, cy: int, cz: int) -> None:
    """Verify mesh upload via the null renderer (no display required)."""
    renderer = VoxelRenderer(headless=True)
    renderer.upload_chunk_mesh(cx, cy, cz, mesh)
    camera = Camera(position=(64.0, 64.0, 64.0), yaw=-135.0, pitch=-20.0)
    renderer.render(camera)
    renderer.swap()
    renderer.cleanup()
    print(f"Headless render complete — mesh has {len(mesh) // 7} vertices.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Generate a single terrain chunk using the C++ engine.
    chunk = make_single_chunk(args.seed, args.chunk_x, args.chunk_y, args.chunk_z)
    print_chunk_info(chunk, args.seed, args.chunk_x, args.chunk_y, args.chunk_z)

    mesh = chunk.build_mesh()
    if not mesh:
        print("WARNING: Mesh is empty – no visible faces generated.")
        return

    headless = args.headless or not _HAS_OPENGL or not _HAS_PYGAME
    if headless:
        run_headless(mesh, args.chunk_x, args.chunk_y, args.chunk_z)
    else:
        run_interactive(mesh, args.chunk_x, args.chunk_y, args.chunk_z, args.frames)


if __name__ == "__main__":
    main()
