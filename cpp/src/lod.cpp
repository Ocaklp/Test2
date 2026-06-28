#include "lod.hpp"
#include <cmath>
#include <vector>

namespace voxel {

int lod_level_for(const IVec3& chunk_coord, const Vec3& viewer_chunk_pos) {
    float dx = static_cast<float>(chunk_coord.x) - viewer_chunk_pos.x;
    float dy = static_cast<float>(chunk_coord.y) - viewer_chunk_pos.y;
    float dz = static_cast<float>(chunk_coord.z) - viewer_chunk_pos.z;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    for (int i = 0; i < LOD_LEVELS - 1; ++i) {
        if (dist <= LOD_DIST[i]) return i;
    }
    return LOD_LEVELS - 1;
}

// Build mesh from octree at given LOD.
// lod_level 0 → leaf nodes (full detail)
// lod_level k → nodes of size 2^k
std::vector<float> build_lod_mesh(const Octree& tree, int lod_level) {
    std::vector<float> out;
    int lod_size = 1 << lod_level;  // 1, 2, 4, 8, …

    tree.traverse_lod(lod_size, [&](const OctreeNode& node) {
        if (!node.representative.solid()) return;

        float s  = static_cast<float>(node.size);
        float x0 = static_cast<float>(node.min_corner.x);
        float y0 = static_cast<float>(node.min_corner.y);
        float z0 = static_cast<float>(node.min_corner.z);
        float biome_id = static_cast<float>(node.representative.biome());

        // Emit a box as 6 quads (12 triangles).
        // Each face: 6 vertices × 7 floats = 42 floats per face.
        // Vertices: [x,y,z, nx,ny,nz, biome_id]
        struct Face { float nx,ny,nz; float p[4][3]; };
        Face faces[6] = {
            // +X
            { 1,0,0, {{x0+s,y0,z0},{x0+s,y0+s,z0},{x0+s,y0+s,z0+s},{x0+s,y0,z0+s}} },
            // -X
            {-1,0,0, {{x0,y0,z0+s},{x0,y0+s,z0+s},{x0,y0+s,z0},{x0,y0,z0}} },
            // +Y
            { 0,1,0, {{x0,y0+s,z0},{x0,y0+s,z0+s},{x0+s,y0+s,z0+s},{x0+s,y0+s,z0}} },
            // -Y
            { 0,-1,0,{{x0,y0,z0+s},{x0,y0,z0},{x0+s,y0,z0},{x0+s,y0,z0+s}} },
            // +Z
            { 0,0,1, {{x0+s,y0,z0+s},{x0+s,y0+s,z0+s},{x0,y0+s,z0+s},{x0,y0,z0+s}} },
            // -Z
            { 0,0,-1,{{x0,y0,z0},{x0,y0+s,z0},{x0+s,y0+s,z0},{x0+s,y0,z0}} }
        };

        for (auto& f : faces) {
            // Two triangles: 0-1-2 and 0-2-3
            const int tri[6] = {0,1,2,0,2,3};
            for (int i : tri) {
                out.push_back(f.p[i][0]); out.push_back(f.p[i][1]);
                out.push_back(f.p[i][2]);
                out.push_back(f.nx); out.push_back(f.ny); out.push_back(f.nz);
                out.push_back(biome_id);
            }
        }
    });

    return out;
}

Octree chunk_to_octree(const Chunk& chunk) {
    Octree tree(Chunk::SIZE, chunk.world_origin());
    for (int z = 0; z < Chunk::SIZE; ++z)
        for (int y = 0; y < Chunk::SIZE; ++y)
            for (int x = 0; x < Chunk::SIZE; ++x) {
                Voxel v = chunk.get(x, y, z);
                if (v.solid()) {
                    IVec3 wp = chunk.world_origin() + IVec3{x, y, z};
                    tree.set(wp, v);
                }
            }
    tree.compress();
    return tree;
}

}  // namespace voxel
