#pragma once
#include "voxel.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <functional>
#include <optional>

namespace voxel {

// Loose octree node.
// Each node covers a cubic AABB and can be subdivided into 8 children
// (indexed by the standard octant convention: X+, Y+, Z+ bits).
// Leaf nodes store a representative Voxel value for the entire cell
// (used for LOD — the parent summarises its children).
struct OctreeNode {
    static constexpr uint8_t CHILDREN = 8;

    IVec3 min_corner;   // inclusive minimum in voxel space
    int   size;         // side length of this node in voxels (always power of 2)
    Voxel representative;     // summary voxel (most-common solid/biome in subtree)
    bool  is_leaf{true};      // true when this node has no children
    bool  all_same{false};    // true when every voxel in node is identical

    std::array<std::unique_ptr<OctreeNode>, CHILDREN> children{};

    OctreeNode(IVec3 min, int sz)
        : min_corner(min), size(sz) {}

    // Octant index given an absolute voxel position within this node.
    int octant(const IVec3& pos) const {
        int half = size / 2;
        return ((pos.x - min_corner.x) >= half ? 1 : 0) |
               (((pos.y - min_corner.y) >= half ? 1 : 0) << 1) |
               (((pos.z - min_corner.z) >= half ? 1 : 0) << 2);
    }

    IVec3 child_min(int oct) const {
        int half = size / 2;
        return {
            min_corner.x + ((oct & 1) ? half : 0),
            min_corner.y + ((oct & 2) ? half : 0),
            min_corner.z + ((oct & 4) ? half : 0)
        };
    }
};

// A sparse octree over a fixed-size voxel volume.
// Root node covers a cube of `root_size` voxels (must be a power of 2).
class Octree {
public:
    explicit Octree(int root_size, IVec3 origin = {0, 0, 0});

    // Insert / update a single voxel.
    void set(const IVec3& pos, Voxel v);

    // Query a single voxel. Returns empty optional if outside bounds.
    std::optional<Voxel> get(const IVec3& pos) const;

    // Attempt to merge uniform children upward (compression pass).
    void compress();

    // Traverse all leaf nodes, calling callback(node).
    void traverse_leaves(const std::function<void(const OctreeNode&)>& cb) const;

    // Traverse nodes whose size is closest to `lod_size` voxels.
    void traverse_lod(int lod_size,
                      const std::function<void(const OctreeNode&)>& cb) const;

    const OctreeNode& root() const { return *root_; }
    int root_size()          const { return root_size_; }
    IVec3 origin()           const { return origin_; }

private:
    std::unique_ptr<OctreeNode> root_;
    int   root_size_;
    IVec3 origin_;

    void set_rec(OctreeNode& node, const IVec3& pos, Voxel v);
    std::optional<Voxel> get_rec(const OctreeNode& node, const IVec3& pos) const;
    void compress_rec(OctreeNode& node);
    void traverse_leaves_rec(const OctreeNode& node,
                              const std::function<void(const OctreeNode&)>& cb) const;
    void traverse_lod_rec(const OctreeNode& node, int lod_size,
                          const std::function<void(const OctreeNode&)>& cb) const;
    bool in_bounds(const IVec3& pos) const;
};

}  // namespace voxel
