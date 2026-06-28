#include "octree.hpp"
#include <stdexcept>
#include <queue>

namespace voxel {

Octree::Octree(int root_size, IVec3 origin)
    : root_size_(root_size), origin_(origin) {
    // root_size must be a power of 2
    root_ = std::make_unique<OctreeNode>(origin, root_size);
}

bool Octree::in_bounds(const IVec3& pos) const {
    return pos.x >= origin_.x && pos.x < origin_.x + root_size_ &&
           pos.y >= origin_.y && pos.y < origin_.y + root_size_ &&
           pos.z >= origin_.z && pos.z < origin_.z + root_size_;
}

void Octree::set(const IVec3& pos, Voxel v) {
    if (!in_bounds(pos)) return;
    set_rec(*root_, pos, v);
}

void Octree::set_rec(OctreeNode& node, const IVec3& pos, Voxel v) {
    if (node.size == 1) {
        node.representative = v;
        return;
    }
    int oct = node.octant(pos);
    if (!node.children[oct]) {
        node.children[oct] = std::make_unique<OctreeNode>(
            node.child_min(oct), node.size / 2);
        node.is_leaf = false;
    }
    set_rec(*node.children[oct], pos, v);

    // Update representative as most-common voxel among children.
    // Simple majority: favour solid over air.
    int solid_count = 0;
    BiomeType dominant = BiomeType::Air;
    for (auto& c : node.children) {
        if (c && c->representative.solid()) {
            ++solid_count;
            dominant = c->representative.biome();
        }
    }
    node.representative = Voxel(solid_count > 4, dominant);
}

std::optional<Voxel> Octree::get(const IVec3& pos) const {
    if (!in_bounds(pos)) return std::nullopt;
    return get_rec(*root_, pos);
}

std::optional<Voxel> Octree::get_rec(const OctreeNode& node,
                                      const IVec3& pos) const {
    if (node.size == 1) return node.representative;
    int oct = node.octant(pos);
    if (!node.children[oct]) return Voxel{};  // default: air
    return get_rec(*node.children[oct], pos);
}

void Octree::compress() { compress_rec(*root_); }

void Octree::compress_rec(OctreeNode& node) {
    if (node.size == 1) { node.all_same = true; return; }

    for (auto& c : node.children)
        if (c) compress_rec(*c);

    // Check if all 8 children exist, are leaves, and have the same voxel.
    bool can_merge = true;
    Voxel ref{};
    bool  first = true;
    for (auto& c : node.children) {
        if (!c) { can_merge = false; break; }
        if (!c->all_same) { can_merge = false; break; }
        if (first) { ref = c->representative; first = false; }
        else if (c->representative != ref) { can_merge = false; break; }
    }
    if (can_merge) {
        for (auto& c : node.children) c.reset();
        node.is_leaf     = true;
        node.all_same    = true;
        node.representative = ref;
    }
}

void Octree::traverse_leaves(
    const std::function<void(const OctreeNode&)>& cb) const {
    traverse_leaves_rec(*root_, cb);
}

void Octree::traverse_leaves_rec(
    const OctreeNode& node,
    const std::function<void(const OctreeNode&)>& cb) const {
    bool has_any_child = false;
    for (auto& c : node.children) {
        if (c) { traverse_leaves_rec(*c, cb); has_any_child = true; }
    }
    if (!has_any_child) cb(node);
}

void Octree::traverse_lod(
    int lod_size,
    const std::function<void(const OctreeNode&)>& cb) const {
    traverse_lod_rec(*root_, lod_size, cb);
}

void Octree::traverse_lod_rec(
    const OctreeNode& node, int lod_size,
    const std::function<void(const OctreeNode&)>& cb) const {
    if (node.size <= lod_size) { cb(node); return; }
    bool has_child = false;
    for (auto& c : node.children) {
        if (c) { traverse_lod_rec(*c, lod_size, cb); has_child = true; }
    }
    if (!has_child) cb(node);  // sparse node – treat as leaf
}

}  // namespace voxel
