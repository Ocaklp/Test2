"""
Python tests for the voxel_core C++ extension via Pybind11.
Run with: python -m pytest tests/python/ -v
"""

import sys
import os
import math
from pathlib import Path

# Ensure the compiled extension is importable.
_proj_root = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(_proj_root))
sys.path.insert(0, str(_proj_root / "python"))

import pytest
import voxel_core as ve


# ─────────────────────────────────────────────────────────────────────────────
# Noise tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNoise:
    def test_noise2_range(self):
        n = ve.Noise(42)
        for i in range(500):
            v = n.noise2(i * 0.1, i * 0.07)
            assert -1.0 <= v <= 1.0, f"noise2 out of range: {v}"

    def test_noise3_range(self):
        n = ve.Noise(42)
        for i in range(200):
            v = n.noise3(i * 0.1, i * 0.07, i * 0.13)
            assert -1.0 <= v <= 1.0, f"noise3 out of range: {v}"

    def test_fbm2_range(self):
        n = ve.Noise(7)
        for i in range(200):
            v = n.fbm2(i * 0.05, i * 0.03, 6)
            assert -1.0 <= v <= 1.0, f"fbm2 out of range: {v}"

    def test_ridged2_range(self):
        n = ve.Noise(7)
        for i in range(200):
            v = n.ridged2(i * 0.05, i * 0.03, 4)
            assert 0.0 <= v <= 1.0, f"ridged2 out of range: {v}"

    def test_deterministic(self):
        a, b = ve.Noise(99), ve.Noise(99)
        assert a.noise2(1.5, 2.3) == b.noise2(1.5, 2.3)
        assert a.noise3(1.5, 2.3, 3.1) == b.noise3(1.5, 2.3, 3.1)

    def test_different_seeds_differ(self):
        assert ve.Noise(0).noise2(1.0, 1.0) != ve.Noise(1).noise2(1.0, 1.0)

    def test_seed_property(self):
        n = ve.Noise(12345)
        assert n.seed == 12345


# ─────────────────────────────────────────────────────────────────────────────
# Voxel type tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVoxel:
    def test_default_air(self):
        v = ve.Voxel()
        assert not v.solid
        assert v.biome == ve.BiomeType.Air

    def test_solid_biome(self):
        v = ve.Voxel(True, ve.BiomeType.Forest)
        assert v.solid
        assert v.biome == ve.BiomeType.Forest

    def test_set_properties(self):
        v = ve.Voxel()
        v.solid = True
        v.biome = ve.BiomeType.Desert
        assert v.solid
        assert v.biome == ve.BiomeType.Desert

    def test_repr(self):
        v = ve.Voxel(True, ve.BiomeType.Plains)
        assert "Voxel" in repr(v)

    def test_biome_enum_values(self):
        assert int(ve.BiomeType.Air)      == 0
        assert int(ve.BiomeType.Ocean)    == 1
        assert int(ve.BiomeType.Snow)     == 8


# ─────────────────────────────────────────────────────────────────────────────
# Octree tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOctree:
    def test_set_get(self):
        t = ve.Octree(32)
        v = ve.Voxel(True, ve.BiomeType.Plains)
        t.set(ve.IVec3(5, 10, 15), v)
        result = t.get(ve.IVec3(5, 10, 15))
        assert result is not None
        assert result.solid
        assert result.biome == ve.BiomeType.Plains

    def test_default_air(self):
        t = ve.Octree(32)
        result = t.get(ve.IVec3(0, 0, 0))
        assert result is not None
        assert not result.solid

    def test_out_of_bounds(self):
        t = ve.Octree(32)
        result = t.get(ve.IVec3(100, 0, 0))
        assert result is None

    def test_compress(self):
        t = ve.Octree(8)
        v = ve.Voxel(True, ve.BiomeType.Snow)
        for x in range(8):
            for y in range(8):
                for z in range(8):
                    t.set(ve.IVec3(x, y, z), v)
        t.compress()
        assert t.root().all_same

    def test_traverse_leaves(self):
        t = ve.Octree(16)
        t.set(ve.IVec3(0, 0, 0), ve.Voxel(True, ve.BiomeType.Forest))
        t.set(ve.IVec3(8, 8, 8), ve.Voxel(True, ve.BiomeType.Mountain))
        leaves = t.traverse_leaves()
        assert len(leaves) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Chunk tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunk:
    def test_chunk_size(self):
        assert ve.Chunk.CHUNK_SIZE == 32

    def test_set_get(self):
        c = ve.Chunk()
        v = ve.Voxel(True, ve.BiomeType.Beach)
        c.set(1, 2, 3, v)
        r = c.get(1, 2, 3)
        assert r.solid
        assert r.biome == ve.BiomeType.Beach

    def test_fill(self):
        c = ve.Chunk()
        c.fill(ve.Voxel(True, ve.BiomeType.Plains))
        assert c.solid_count() == 32 ** 3

    def test_dirty_flag(self):
        c = ve.Chunk()
        assert not c.dirty()
        c.set(0, 0, 0, ve.Voxel(True, ve.BiomeType.Plains))
        assert c.dirty()
        c.clear_dirty()
        assert not c.dirty()

    def test_build_mesh_empty(self):
        c = ve.Chunk()
        assert c.build_mesh() == []

    def test_build_mesh_single_voxel(self):
        c = ve.Chunk()
        c.set(5, 5, 5, ve.Voxel(True, ve.BiomeType.Plains))
        mesh = c.build_mesh()
        # 6 faces × 2 tris × 3 verts × 7 floats = 252
        assert len(mesh) == 252

    def test_world_origin(self):
        c = ve.Chunk(ve.IVec3(2, 0, -1))
        origin = c.world_origin()
        assert origin.x == 64
        assert origin.z == -32

    def test_out_of_bounds_raises(self):
        c = ve.Chunk()
        with pytest.raises(IndexError):
            c.get(-1, 0, 0)
        with pytest.raises(IndexError):
            c.set(ve.Chunk.CHUNK_SIZE, 0, 0, ve.Voxel(True, ve.BiomeType.Plains))


# ─────────────────────────────────────────────────────────────────────────────
# Terrain tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTerrain:
    def test_generate_chunk(self):
        gen = ve.TerrainGenerator(42)
        c = ve.Chunk(ve.IVec3(0, 0, 0))
        gen.generate(c)
        assert c.solid_count() > 0

    def test_surface_height_varied(self):
        gen = ve.TerrainGenerator(1234)
        h1 = gen.surface_height(0.0, 0.0)
        h2 = gen.surface_height(1000.0, 1000.0)
        assert h1 != h2

    def test_biome_sea_level(self):
        gen = ve.TerrainGenerator(0)
        b = gen.biome_at(0.0, 0.0, 0.0)  # y=0 < sea_level=8
        assert b == ve.BiomeType.Ocean

    def test_biome_snow(self):
        gen = ve.TerrainGenerator(0)
        b = gen.biome_at(0.0, 0.0, 100.0)  # high y → Snow
        assert b == ve.BiomeType.Snow

    def test_terrain_params(self):
        p = ve.TerrainParams()
        p.sea_level = 5
        assert p.sea_level == 5
        gen = ve.TerrainGenerator(0, p)
        assert gen.params.sea_level == 5

    def test_seed_preserved(self):
        gen = ve.TerrainGenerator(777)
        assert gen.seed == 777


# ─────────────────────────────────────────────────────────────────────────────
# WFC tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWFC:
    def test_solve_produces_valid_biomes(self):
        solver = ve.WFCSolver(8, 8)
        solver.solve(20)
        for z in range(8):
            for x in range(8):
                b = solver.at(x, z)
                assert b != ve.BiomeType.Air

    def test_pin_respected(self):
        solver = ve.WFCSolver(4, 4, seed=1)
        solver.pin(0, 0, ve.BiomeType.Ocean)
        solver.solve(10)
        assert solver.at(0, 0) == ve.BiomeType.Ocean

    def test_dimensions(self):
        s = ve.WFCSolver(10, 6)
        assert s.width == 10
        assert s.height == 6

    def test_apply_to_chunk(self):
        solver = ve.WFCSolver(32, 32, seed=42)
        solver.solve(5)
        c = ve.Chunk(ve.IVec3(0, 0, 0))
        gen = ve.TerrainGenerator(42)
        gen.generate(c)
        solver.apply_to_chunk(c, 0, 0)
        # After applying, solid voxels should have valid (non-Air) biomes.
        for z in range(ve.Chunk.CHUNK_SIZE):
            for x in range(ve.Chunk.CHUNK_SIZE):
                for y in range(ve.Chunk.CHUNK_SIZE - 1, -1, -1):
                    v = c.get(x, y, z)
                    if v.solid:
                        assert v.biome != ve.BiomeType.Air
                        break


# ─────────────────────────────────────────────────────────────────────────────
# LOD tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLOD:
    def test_lod_near(self):
        lod = ve.lod_level_for(ve.IVec3(0, 0, 0), ve.Vec3(0, 0, 0))
        assert lod == 0

    def test_lod_far(self):
        lod = ve.lod_level_for(ve.IVec3(100, 0, 100), ve.Vec3(0, 0, 0))
        assert lod == 3  # max LOD level

    def test_chunk_to_octree(self):
        c = ve.Chunk(ve.IVec3(0, 0, 0))
        c.set(0, 0, 0, ve.Voxel(True, ve.BiomeType.Plains))
        t = ve.chunk_to_octree(c)
        assert t.root_size() == 32

    def test_build_lod_mesh(self):
        c = ve.Chunk(ve.IVec3(0, 0, 0))
        c.set(0, 0, 0, ve.Voxel(True, ve.BiomeType.Plains))
        t = ve.chunk_to_octree(c)
        mesh = ve.build_lod_mesh(t, 0)
        assert len(mesh) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Mat4 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMat4:
    def test_identity(self):
        m = ve.Mat4()
        lst = m.as_list()
        assert len(lst) == 16
        # Diagonal should be 1
        for i in [0, 5, 10, 15]:
            assert abs(lst[i] - 1.0) < 1e-6

    def test_translation(self):
        m = ve.Mat4.translation(3.0, 4.0, 5.0)
        lst = m.as_list()
        assert abs(lst[12] - 3.0) < 1e-6
        assert abs(lst[13] - 4.0) < 1e-6
        assert abs(lst[14] - 5.0) < 1e-6

    def test_multiply(self):
        a = ve.Mat4.translation(1.0, 0.0, 0.0)
        b = ve.Mat4.translation(2.0, 0.0, 0.0)
        c = a * b
        lst = c.as_list()
        assert abs(lst[12] - 3.0) < 1e-5

    def test_as_numpy(self):
        import numpy as np
        m = ve.Mat4()
        arr = m.as_numpy()
        assert arr.shape == (4, 4)
        assert abs(arr[0, 0] - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# World tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWorld:
    def test_create(self):
        w = ve.World(seed=42, view_distance=2, thread_count=2)
        assert w.seed == 42
        assert w.view_distance == 2

    def test_update_loads_chunks(self):
        import time
        w = ve.World(seed=1, view_distance=2, thread_count=2)
        w.update(ve.Vec3(0, 0, 0))
        time.sleep(1.0)  # allow workers to load
        coords = w.loaded_chunk_coords()
        assert len(coords) >= 3  # at least one chunk

    def test_chunk_cache_reuses_previous_chunks(self):
        import time

        def as_set(coords):
            return {
                (coords[i], coords[i + 1], coords[i + 2])
                for i in range(0, len(coords), 3)
            }

        w = ve.World(seed=7, view_distance=1, thread_count=2)
        w.update(ve.Vec3(0, 0, 0))
        time.sleep(0.6)
        assert (0, 0, 0) in as_set(w.loaded_chunk_coords())

        w.update(ve.Vec3(32 * 8, 0, 0))
        time.sleep(0.6)
        assert (0, 0, 0) not in as_set(w.loaded_chunk_coords())

        # Moving back should make cached chunks visible immediately.
        w.update(ve.Vec3(0, 0, 0))
        assert (0, 0, 0) in as_set(w.loaded_chunk_coords())

    def test_get_set_voxel(self):
        import time
        w = ve.World(seed=5, view_distance=1, thread_count=2)
        w.update(ve.Vec3(0, 0, 0))
        time.sleep(0.5)

        # Read a voxel
        v = w.get_voxel(0, 0, 0)
        # Set a voxel
        w.set_voxel(0, 0, 0, ve.Voxel(True, ve.BiomeType.Desert))
        time.sleep(0.1)
        result = w.get_voxel(0, 0, 0)
        assert result.solid
        assert result.biome == ve.BiomeType.Desert
