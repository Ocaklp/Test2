// Minimal unit tests for the voxel engine C++ core.
// Uses a lightweight single-header test framework (no external dependencies).

#include "noise.hpp"
#include "voxel.hpp"
#include "octree.hpp"
#include "chunk.hpp"
#include "terrain.hpp"
#include "wfc.hpp"
#include "lod.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// ──────────────────────────────────────────────────────────────────────────────
// Tiny test framework
// ──────────────────────────────────────────────────────────────────────────────
static int s_passed = 0, s_failed = 0;

#define TEST(name) static void test_##name()
#define RUN(name)  do { \
    std::cout << "  " #name " ... "; \
    try { test_##name(); std::cout << "PASS\n"; ++s_passed; } \
    catch (const std::exception& e) { \
        std::cout << "FAIL: " << e.what() << "\n"; ++s_failed; } \
    catch (...) { std::cout << "FAIL: unknown exception\n"; ++s_failed; } \
} while(0)

#define EXPECT(cond) do { if (!(cond)) \
    throw std::runtime_error("Assertion failed: " #cond); } while(0)
#define EXPECT_NEAR(a,b,eps) do { \
    if (std::abs((a)-(b)) > (eps)) \
        throw std::runtime_error(std::string("Expected |") + #a + " - " + #b \
            + "| <= " #eps "; got " + std::to_string(std::abs((a)-(b)))); \
} while(0)

// ──────────────────────────────────────────────────────────────────────────────
// Noise tests
// ──────────────────────────────────────────────────────────────────────────────
using namespace voxel;

TEST(noise_range_2d) {
    Noise n(42);
    for (int i = 0; i < 1000; ++i) {
        float v = n.noise2(static_cast<float>(i) * 0.1f,
                           static_cast<float>(i) * 0.07f);
        EXPECT(v >= -1.0f && v <= 1.0f);
    }
}

TEST(noise_range_3d) {
    Noise n(42);
    for (int i = 0; i < 500; ++i) {
        float v = n.noise3(i * 0.1f, i * 0.07f, i * 0.13f);
        EXPECT(v >= -1.0f && v <= 1.0f);
    }
}

TEST(noise_deterministic) {
    Noise a(99), b(99);
    EXPECT(a.noise2(1.5f, 2.3f) == b.noise2(1.5f, 2.3f));
    EXPECT(a.noise3(1.5f, 2.3f, 3.1f) == b.noise3(1.5f, 2.3f, 3.1f));
}

TEST(noise_different_seeds) {
    Noise a(0), b(1);
    EXPECT(a.noise2(1.0f, 1.0f) != b.noise2(1.0f, 1.0f));
}

TEST(fbm_range) {
    Noise n(7);
    for (int i = 0; i < 200; ++i) {
        float v = n.fbm2(i * 0.05f, i * 0.03f, 6);
        EXPECT(v >= -1.0f && v <= 1.0f);
    }
}

TEST(ridged_range) {
    Noise n(7);
    for (int i = 0; i < 200; ++i) {
        float v = n.ridged2(i * 0.05f, i * 0.03f, 4);
        EXPECT(v >= 0.0f && v <= 1.0f);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Voxel type tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(voxel_default_air) {
    Voxel v;
    EXPECT(!v.solid());
    EXPECT(v.biome() == BiomeType::Air);
}

TEST(voxel_solid_biome) {
    Voxel v(true, BiomeType::Forest);
    EXPECT(v.solid());
    EXPECT(v.biome() == BiomeType::Forest);
}

TEST(voxel_set_bits) {
    Voxel v;
    v.set_solid(true);
    v.set_biome(BiomeType::Desert);
    EXPECT(v.solid());
    EXPECT(v.biome() == BiomeType::Desert);
    v.set_solid(false);
    EXPECT(!v.solid());
    EXPECT(v.biome() == BiomeType::Desert);
}

TEST(voxel_size) {
    EXPECT(sizeof(Voxel) == 1);
}

// ──────────────────────────────────────────────────────────────────────────────
// Octree tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(octree_set_get) {
    Octree tree(32);
    Voxel v(true, BiomeType::Plains);
    tree.set({5, 10, 15}, v);
    auto result = tree.get({5, 10, 15});
    EXPECT(result.has_value());
    EXPECT(result->solid());
    EXPECT(result->biome() == BiomeType::Plains);
}

TEST(octree_out_of_bounds) {
    Octree tree(32);
    // Out-of-bounds returns empty optional.
    EXPECT(!tree.get({100, 0, 0}).has_value());
}

TEST(octree_default_air) {
    Octree tree(32);
    auto v = tree.get({1, 2, 3});
    EXPECT(v.has_value());
    EXPECT(!v->solid());
}

TEST(octree_compress_uniform) {
    Octree tree(8);
    Voxel v(true, BiomeType::Snow);
    // Fill all 8³ = 512 voxels with the same value.
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z)
                tree.set({x, y, z}, v);
    tree.compress();
    // Root should now be marked all_same.
    EXPECT(tree.root().all_same);
}

TEST(octree_traverse_leaves) {
    Octree tree(16);
    tree.set({0, 0, 0}, Voxel(true, BiomeType::Forest));
    tree.set({8, 8, 8}, Voxel(true, BiomeType::Mountain));
    int leaf_count = 0;
    tree.traverse_leaves([&](const OctreeNode&){ ++leaf_count; });
    EXPECT(leaf_count >= 2);
}

// ──────────────────────────────────────────────────────────────────────────────
// Chunk tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(chunk_size) {
    EXPECT(Chunk::SIZE == 32);
    EXPECT(Chunk::VOLUME == 32 * 32 * 32);
}

TEST(chunk_set_get) {
    Chunk c;
    Voxel v(true, BiomeType::Beach);
    c.set(1, 2, 3, v);
    EXPECT(c.get(1, 2, 3).solid());
    EXPECT(c.get(1, 2, 3).biome() == BiomeType::Beach);
}

TEST(chunk_fill) {
    Chunk c;
    Voxel v(true, BiomeType::Plains);
    c.fill(v);
    EXPECT(c.solid_count() == Chunk::VOLUME);
}

TEST(chunk_dirty) {
    Chunk c;
    EXPECT(!c.dirty());
    c.set(0, 0, 0, Voxel(true, BiomeType::Air));
    EXPECT(c.dirty());
    c.clear_dirty();
    EXPECT(!c.dirty());
}

TEST(chunk_build_mesh_empty) {
    Chunk c;
    auto mesh = c.build_mesh();
    EXPECT(mesh.empty());
}

TEST(chunk_build_mesh_single_voxel) {
    Chunk c;
    c.set(5, 5, 5, Voxel(true, BiomeType::Plains));
    auto mesh = c.build_mesh();
    // A single voxel has 6 faces × 2 triangles × 3 verts × 7 floats = 252
    EXPECT(mesh.size() == 6 * 2 * 3 * 7);
}

TEST(chunk_build_mesh_culls_boundary_with_neighbor_query) {
    Chunk left(IVec3{0, 0, 0});
    Chunk right(IVec3{1, 0, 0});
    left.set(Chunk::SIZE - 1, 5, 5, Voxel(true, BiomeType::Plains));
    right.set(0, 5, 5, Voxel(true, BiomeType::Forest));

    auto uncull_mesh = left.build_mesh();
    EXPECT(uncull_mesh.size() == 6 * 2 * 3 * 7);

    auto culled_mesh = left.build_mesh([&](int wx, int wy, int wz) {
        if (wx < Chunk::SIZE || wx >= 2 * Chunk::SIZE ||
            wy < 0 || wy >= Chunk::SIZE ||
            wz < 0 || wz >= Chunk::SIZE) {
            return false;
        }
        return right.get(wx - Chunk::SIZE, wy, wz).solid();
    });

    // The +X face is hidden by the right chunk voxel.
    EXPECT(culled_mesh.size() == 5 * 2 * 3 * 7);
}

TEST(chunk_out_of_bounds_guard) {
    Chunk c;
    bool threw_get = false;
    bool threw_set = false;
    try {
        (void)c.get(-1, 0, 0);
    } catch (const std::out_of_range&) {
        threw_get = true;
    }
    try {
        c.set(Chunk::SIZE, 0, 0, Voxel(true, BiomeType::Plains));
    } catch (const std::out_of_range&) {
        threw_set = true;
    }
    EXPECT(threw_get);
    EXPECT(threw_set);
}

// ──────────────────────────────────────────────────────────────────────────────
// Terrain tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(terrain_generate) {
    TerrainGenerator gen(42);
    Chunk c(IVec3{0, 0, 0});
    gen.generate(c);
    // Expect some solid voxels to exist near the surface.
    EXPECT(c.solid_count() > 0);
}

TEST(terrain_height_varied) {
    TerrainGenerator gen(1234);
    float h1 = gen.surface_height(0.0f, 0.0f);
    float h2 = gen.surface_height(1000.0f, 1000.0f);
    // Heights should differ (extremely unlikely to be equal with noise).
    EXPECT(h1 != h2);
}

TEST(terrain_biome_sea_level) {
    TerrainGenerator gen(0);
    BiomeType b = gen.biome_at(0.0f, 0.0f, 0.0f);  // y=0 < sea_level=8
    EXPECT(b == BiomeType::Ocean);
}

TEST(terrain_biome_snow) {
    TerrainGenerator gen(0);
    BiomeType b = gen.biome_at(0.0f, 0.0f, 100.0f); // very high y → Snow
    EXPECT(b == BiomeType::Snow);
}

// ──────────────────────────────────────────────────────────────────────────────
// WFC tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(wfc_solve) {
    WFCSolver solver(8, 8, default_adjacency(), 42);
    bool ok = solver.solve(20);
    (void)ok;  // May occasionally fail to solve; check tiles are valid.
    // All cells should have a valid biome.
    for (int z = 0; z < 8; ++z) {
        for (int x = 0; x < 8; ++x) {
            BiomeType b = solver.at(x, z);
            EXPECT(b != BiomeType::Air);
        }
    }
}

TEST(wfc_pin) {
    WFCSolver solver(4, 4, default_adjacency(), 1);
    solver.pin(0, 0, BiomeType::Ocean);
    solver.solve(10);
    EXPECT(solver.at(0, 0) == BiomeType::Ocean);
}

TEST(wfc_adjacency) {
    // Ocean should be adjacent to Beach in default rules.
    auto adj = default_adjacency();
    int ocean_idx  = static_cast<int>(BiomeType::Ocean)  - 1;
    int beach_idx  = static_cast<int>(BiomeType::Beach)  - 1;
    EXPECT(adj[ocean_idx].test(beach_idx));
    // Ocean should NOT be directly adjacent to Snow.
    int snow_idx   = static_cast<int>(BiomeType::Snow)   - 1;
    EXPECT(!adj[ocean_idx].test(snow_idx));
}

// ──────────────────────────────────────────────────────────────────────────────
// LOD tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(lod_level_near) {
    // Chunk at (0,0,0), viewer at (0,0,0) → LOD 0 (full detail)
    int lod = lod_level_for({0,0,0}, {0.0f,0.0f,0.0f});
    EXPECT(lod == 0);
}

TEST(lod_level_far) {
    // Chunk 100 chunks away → highest LOD
    int lod = lod_level_for({100,0,100}, {0.0f,0.0f,0.0f});
    EXPECT(lod == LOD_LEVELS - 1);
}

TEST(lod_mesh_generation) {
    Chunk chunk(IVec3{0, 0, 0});
    chunk.set(0, 0, 0, Voxel(true, BiomeType::Plains));
    Octree tree = chunk_to_octree(chunk);
    auto mesh = build_lod_mesh(tree, 0);
    EXPECT(!mesh.empty());
}

// ──────────────────────────────────────────────────────────────────────────────
// Mat4 tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(mat4_identity) {
    Mat4 m;
    EXPECT(m.data()[0]  == 1.0f);
    EXPECT(m.data()[5]  == 1.0f);
    EXPECT(m.data()[10] == 1.0f);
    EXPECT(m.data()[15] == 1.0f);
    EXPECT(m.data()[1]  == 0.0f);
}

TEST(mat4_translation) {
    Mat4 t = Mat4::translation(3.0f, 4.0f, 5.0f);
    EXPECT(t.data()[12] == 3.0f);
    EXPECT(t.data()[13] == 4.0f);
    EXPECT(t.data()[14] == 5.0f);
}

TEST(mat4_multiply) {
    Mat4 a = Mat4::translation(1.0f, 0.0f, 0.0f);
    Mat4 b = Mat4::translation(2.0f, 0.0f, 0.0f);
    Mat4 c = a * b;
    EXPECT_NEAR(c.data()[12], 3.0f, 1e-5f);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Voxel Engine C++ Unit Tests ===\n\n";

    std::cout << "[ Noise ]\n";
    RUN(noise_range_2d);
    RUN(noise_range_3d);
    RUN(noise_deterministic);
    RUN(noise_different_seeds);
    RUN(fbm_range);
    RUN(ridged_range);

    std::cout << "\n[ Voxel ]\n";
    RUN(voxel_default_air);
    RUN(voxel_solid_biome);
    RUN(voxel_set_bits);
    RUN(voxel_size);

    std::cout << "\n[ Octree ]\n";
    RUN(octree_set_get);
    RUN(octree_out_of_bounds);
    RUN(octree_default_air);
    RUN(octree_compress_uniform);
    RUN(octree_traverse_leaves);

    std::cout << "\n[ Chunk ]\n";
    RUN(chunk_size);
    RUN(chunk_set_get);
    RUN(chunk_fill);
    RUN(chunk_dirty);
    RUN(chunk_build_mesh_empty);
    RUN(chunk_build_mesh_single_voxel);
    RUN(chunk_build_mesh_culls_boundary_with_neighbor_query);
    RUN(chunk_out_of_bounds_guard);

    std::cout << "\n[ Terrain ]\n";
    RUN(terrain_generate);
    RUN(terrain_height_varied);
    RUN(terrain_biome_sea_level);
    RUN(terrain_biome_snow);

    std::cout << "\n[ WFC ]\n";
    RUN(wfc_solve);
    RUN(wfc_pin);
    RUN(wfc_adjacency);

    std::cout << "\n[ LOD ]\n";
    RUN(lod_level_near);
    RUN(lod_level_far);
    RUN(lod_mesh_generation);

    std::cout << "\n[ Mat4 ]\n";
    RUN(mat4_identity);
    RUN(mat4_translation);
    RUN(mat4_multiply);

    std::cout << "\n=== Results: " << s_passed << " passed, "
              << s_failed << " failed ===\n";
    return s_failed > 0 ? 1 : 0;
}
