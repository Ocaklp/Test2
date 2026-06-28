#pragma once
#include "chunk.hpp"
#include "octree.hpp"
#include <vector>

namespace voxel {

// LOD (Level of Detail) levels – lower index = higher detail.
// LOD 0 : full resolution (32×32×32 voxels per chunk)
// LOD 1 : 2× reduced  (every 2nd voxel)
// LOD 2 : 4× reduced  (every 4th voxel)
// LOD 3 : 8× reduced  (every 8th voxel, i.e. octree depth-3 node)
constexpr int LOD_LEVELS = 4;

// Distance thresholds in *chunk units* beyond which a coarser LOD is used.
constexpr float LOD_DIST[LOD_LEVELS] = {4.0f, 8.0f, 16.0f, 1e9f};

// Determine which LOD level a chunk at `chunk_coord` should use given the
// viewer position in chunk-space.
int lod_level_for(const IVec3& chunk_coord, const Vec3& viewer_chunk_pos);

// Build a down-sampled mesh from an Octree at the requested LOD level.
// LOD 0 → traverse leaf nodes (full detail).
// LOD k → traverse nodes whose size ≈ 2^k voxels.
std::vector<float> build_lod_mesh(const Octree& tree, int lod_level);

// Build an Octree from a Chunk (all solid voxels inserted, then compressed).
Octree chunk_to_octree(const Chunk& chunk);

}  // namespace voxel
