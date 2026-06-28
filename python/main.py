"""
main.py – Entry point for the procedural voxel world.

Usage
-----
    python python/main.py [--seed N] [--view-distance N] [--headless]

Controls (when running with OpenGL window)
------------------------------------------
    W/A/S/D   – move camera
    Q/E       – move up / down
    Mouse     – look around (click window to capture)
    F4        – toggle wireframe + print total vertices
    Escape    – release mouse / quit
    +/-       – increase / decrease view distance
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure python/ directory is on the path for local imports.
_here = Path(__file__).parent.resolve()
sys.path.insert(0, str(_here))

# ── Import C++ engine ─────────────────────────────────────────────────────
try:
    import voxel_core as ve
except ImportError:
    sys.exit(
        "ERROR: Cannot import 'voxel_core'. "
        "Build it first:\n  python setup.py build_ext --inplace"
    )

from camera import Camera
from renderer import VoxelRenderer, _HAS_OPENGL

# Try importing pygame for the interactive loop.
try:
    import pygame
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# Maximum number of chunk mesh uploads per sync_meshes() call.
_MAX_UPLOADS_PER_FRAME = 64

# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Procedural voxel world")
    p.add_argument("--seed",          type=int, default=42,
                   help="World generation seed")
    p.add_argument("--view-distance", type=int, default=32,
                   help="Chunk view distance (default: 32)")
    p.add_argument("--threads",       type=int, default=max(1, (os.cpu_count() or 8) - 1),
                   help="Background worker thread count")
    p.add_argument("--headless",      action="store_true",
                   help="Run without a window (for testing/CI)")
    p.add_argument("--frames",        type=int, default=0,
                   help="Run for N frames then exit (0 = unlimited)")
    return p.parse_args()


# ── Mesh sync helper ──────────────────────────────────────────────────────
def sync_meshes(world: ve.World, renderer: VoxelRenderer, camera: Camera, pending_uploads: list, pending_macros: list) -> None:
    # 1. Pop the delta queue: Only handle chunks that just finished building
    if hasattr(world, "pop_mesh_updates"):
        new_meshes = world.pop_mesh_updates()
        pending_uploads.extend(new_meshes)

        uploads_this_frame = 0
        while pending_uploads and uploads_this_frame < _MAX_UPLOADS_PER_FRAME:
            mesh = pending_uploads.pop(0)
            renderer.upload_chunk_mesh(mesh.cx, mesh.cy, mesh.cz, mesh.vertices)
            uploads_this_frame += 1

    # --- NOVÉ: Macro-Chunky (HLOD) ---
    if hasattr(world, "pop_macro_updates"):
        new_macros = world.pop_macro_updates()
        pending_macros.extend(new_macros)
        
    # Nahráváme jen 1 obří makro-chunk za snímek, abychom nezasekli hru
    if pending_macros:
        macro = pending_macros.pop(0)
        # Macro vrací v C++ struct RenderableMesh, kde cx = rx, cz = rz
        renderer.upload_macro_mesh(macro.cx, macro.cz, macro.vertices)
            
    # 2. Get the lightweight list of visible coordinates
    cam_x, cam_y, cam_z = camera.position
    dir_x, dir_y, dir_z = camera.forward_vector()
    
    if hasattr(world, "get_visible_coordinates"):
        visible_coords = world.get_visible_coordinates(
            cam_x, cam_y, cam_z, 
            dir_x, dir_y, dir_z, 
            camera.fov
        )
        # print(f"Visible chunks this frame: {len(visible_coords)}")
        # Tell the renderer what to draw
        renderer.set_visible_chunks(visible_coords)

    # Přidání viditelnosti pro Macro-Chunky
    if hasattr(world, "get_visible_macros"):
        visible_macros = world.get_visible_macros(
            cam_x, cam_y, cam_z, dir_x, dir_y, dir_z, camera.fov
        )
        renderer.set_visible_macros(visible_macros)

# ── Interactive loop ──────────────────────────────────────────────────────

def run_interactive(world: ve.World, camera: Camera,
                    renderer: VoxelRenderer, max_frames: int) -> None:
    if not _HAS_PYGAME:
        print("pygame not installed – cannot run interactive loop.")
        return


    # uploaded: chunk_key → lod_level (int)
    uploaded: set = set()
    mouse_captured = False
    clock = pygame.time.Clock()
    frame = 0
    move_speed = 0.5
    look_speed = 0.15

    print("Controls: WASD=move, QE=up/down, mouse=look, F4=wireframe+verts, Esc=quit")
    pending_uploads: list = [] # NEW: Our waiting list for GPU uploads
    pending_macros: list = []  # NOVÉ: čekající obří Macro-Chunky
    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # ── Events ──────────────────────────────────────────────────────
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
                elif event.key == pygame.K_F4:
                    wireframe = renderer.toggle_wireframe()
                    vertices = renderer.total_vertex_count()
                    mode = "ON" if wireframe else "OFF"
                    print(f"F4: wireframe {mode} | vertices: {vertices}")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                mouse_captured = True
            elif event.type == pygame.VIDEORESIZE:
                renderer.width  = event.w
                renderer.height = event.h

        # ── Mouse look ───────────────────────────────────────────────────
        if mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            camera.rotate(dx * look_speed, -dy * look_speed)
        else:
            pygame.mouse.get_rel()  # consume to avoid jumps

        # ── Keyboard movement ────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        spd  = move_speed * (10.0 if keys[pygame.K_LSHIFT] else 1.0)
        if keys[pygame.K_w]:      camera.move_forward( spd)
        if keys[pygame.K_s]:      camera.move_forward(-spd)
        if keys[pygame.K_a]:      camera.move_right(  -spd)
        if keys[pygame.K_d]:      camera.move_right(   spd)
        if keys[pygame.K_q]:      camera.move_up(     -spd)
        if keys[pygame.K_e]:      camera.move_up(      spd)

        # ── World update ─────────────────────────────────────────────────
        cx, cy, cz = camera.chunk_pos()
        world.update(ve.Vec3(cx * 32, cy * 32, cz * 32))

        # ── Sync meshes ──────────────────────────────────────────────────
        sync_meshes(world, renderer, camera, pending_uploads, pending_macros)

        # ── Render ───────────────────────────────────────────────────────
        renderer.render(camera)
        renderer.swap()

        frame += 1
        if max_frames > 0 and frame >= max_frames:
            break

    renderer.cleanup()


# ── Headless / demo loop ──────────────────────────────────────────────────

def run_headless(world: ve.World, camera: Camera,
                 renderer: VoxelRenderer, frames: int) -> None:
    """Run a fixed number of update+sync cycles without displaying anything."""
    uploaded: dict = {}
    n = frames if frames > 0 else 5

    print(f"Headless mode: running {n} update cycles …")
    for i in range(n):
        cx, cy, cz = camera.chunk_pos()
        world.update(ve.Vec3(cx * 32, cy * 32, cz * 32))
        time.sleep(0.2)  # give worker threads time to load chunks
        print(f"  Frame {i+1}/{n}: {len(uploaded)} meshes active")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Procedural Voxel World  |  seed={args.seed}  "
          f"view_distance={args.view_distance}")

    # ── Create world ──────────────────────────────────────────────────────
    world = ve.World(
        seed          = args.seed,
        view_distance = args.view_distance,
        thread_count  = args.threads,
    )

    # ── Create camera ─────────────────────────────────────────────────────
    camera = Camera(position=(0.0, 50.0, 0.0))

    # ── Create renderer ───────────────────────────────────────────────────
    headless = args.headless or not _HAS_OPENGL or not _HAS_PYGAME
    renderer = VoxelRenderer(headless=headless, view_distance=args.view_distance)

    if headless:
        run_headless(world, camera, renderer, args.frames)
    else:
        run_interactive(world, camera, renderer, args.frames)


if __name__ == "__main__":
    main()