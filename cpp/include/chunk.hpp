#pragma once
#include "voxel.hpp"
#include <array>
#include <cstddef>
#include <vector>
#include <cstdint>
#include <functional>
#include <stdexcept>

namespace voxel {

// Fixed-size chunk of CHUNK_SIZE³ binary voxels.
// Voxels are stored in a flat array with Z-major (ZYX) indexing for cache locality
// when iterating over the XY plane (horizontal slices).
constexpr int CHUNK_SIZE = 32;  // must be a power of 2

class Chunk {
public:
    static constexpr int SIZE   = CHUNK_SIZE;
    static constexpr int VOLUME = SIZE * SIZE * SIZE;
    static constexpr int SHIFT  = 5;              // log2(32)

    Chunk();
    explicit Chunk(IVec3 chunk_coord);

    // Flat index: x + SIZE*(y + SIZE*z)
    static bool in_bounds(int x, int y, int z) {
        return (static_cast<unsigned>(x) < SIZE) &&
               (static_cast<unsigned>(y) < SIZE) &&
               (static_cast<unsigned>(z) < SIZE);
    }
    static int idx_unchecked(int x, int y, int z) {
        return x | (y << SHIFT) | (z << (2 * SHIFT));
    }
    static int idx(int x, int y, int z) {
        if (!in_bounds(x, y, z)) {
            throw std::out_of_range("Chunk coordinates out of bounds");
        }
        return idx_unchecked(x, y, z);
    }
    static int idx(const IVec3& p)      { return idx(p.x, p.y, p.z); }

    Voxel  get(int x, int y, int z) const { return voxels_[idx(x, y, z)]; }
    Voxel  get(const IVec3& p)      const { return get(p.x, p.y, p.z); }
    void   set(int x, int y, int z, Voxel v) { voxels_[idx(x, y, z)] = v; dirty_ = true; }
    void   set(const IVec3& p, Voxel v)      { set(p.x, p.y, p.z, v); }

    // Fill entire chunk with a single voxel value.
    void   fill(Voxel v);

    // Returns true when contents changed since last clear_dirty().
    bool   dirty() const { return dirty_; }
    void   clear_dirty()  { dirty_ = false; }

    // Chunk coordinate in chunk-space (multiply by CHUNK_SIZE for world origin).
    IVec3  chunk_coord() const { return chunk_coord_; }
    IVec3  world_origin() const {
        return {chunk_coord_.x * SIZE,
                chunk_coord_.y * SIZE,
                chunk_coord_.z * SIZE};
    }

    // Direct access to the flat voxel array.
    const std::array<Voxel, VOLUME>& voxels() const { return voxels_; }
    std::array<Voxel, VOLUME>&       voxels()       { return voxels_; }

    // Count of solid voxels (used for LOD representative calculation).
    int solid_count() const;

    // Build a greedy-meshed triangle list for this chunk.
    // Returns an interleaved buffer: [x, y, z, nx, ny, nz, biome_id, ...] per vertex.
    // Vertices represent quads split into two triangles each.
    // Optional callback lets callers report solid voxels outside this chunk
    // (used for cross-chunk hidden-face culling).
    std::vector<float> build_mesh(
        const std::function<bool(int, int, int)>& neighbor_solid_query = {}) const;

private:
    std::array<Voxel, VOLUME> voxels_{};
    IVec3 chunk_coord_{};
    bool  dirty_{false};
};

}  // namespace voxel
