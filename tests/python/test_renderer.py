"""
Tests for the VoxelRenderer (headless / no-display mode only).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "python"))

import pytest
from renderer import VoxelRenderer, BIOME_COLOURS, _NullRenderer


class TestRendererHeadless:
    def test_create_headless(self):
        r = VoxelRenderer(headless=True)
        assert not r.available

    def test_upload_mesh_headless(self):
        r = VoxelRenderer(headless=True)
        mesh = [0.0] * (6 * 2 * 3 * 7)  # 252 floats (single voxel mesh)
        r.upload_chunk_mesh(0, 0, 0, mesh)
        r.render(None)
        r.swap()

    def test_poll_events_headless(self):
        r = VoxelRenderer(headless=True)
        assert r.poll_events() is True

    def test_cleanup_headless(self):
        r = VoxelRenderer(headless=True)
        r.cleanup()  # should not raise

    def test_remove_mesh_headless(self):
        r = VoxelRenderer(headless=True)
        mesh = [0.0] * (6 * 2 * 3 * 7)
        r.upload_chunk_mesh(0, 0, 0, mesh)
        assert (0, 0, 0) in r._null._meshes
        r.remove_chunk_mesh(0, 0, 0)
        assert (0, 0, 0) not in r._null._meshes

    def test_toggle_wireframe_headless(self):
        r = VoxelRenderer(headless=True)
        assert r.toggle_wireframe() is True
        assert r.toggle_wireframe() is False

    def test_total_vertex_count_headless(self):
        r = VoxelRenderer(headless=True)
        mesh_a = [0.0] * (7 * 6)
        mesh_b = [0.0] * (7 * 12)
        r.upload_chunk_mesh(0, 0, 0, mesh_a)
        r.upload_chunk_mesh(1, 0, 0, mesh_b)
        assert r.total_vertex_count() == 18


class TestBiomePalette:
    def test_palette_completeness(self):
        for i in range(9):
            assert i in BIOME_COLOURS

    def test_palette_values_valid(self):
        for key, colour in BIOME_COLOURS.items():
            assert len(colour) == 3
            for ch in colour:
                assert 0.0 <= ch <= 1.0
