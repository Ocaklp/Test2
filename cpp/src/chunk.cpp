#include "chunk.hpp"
#include <algorithm>
#include <cstring>

namespace voxel {

Chunk::Chunk() { voxels_.fill(Voxel{}); }

Chunk::Chunk(IVec3 chunk_coord) : chunk_coord_(chunk_coord) {
    voxels_.fill(Voxel{});
}

void Chunk::fill(Voxel v) {
    voxels_.fill(v);
    dirty_ = true;
}

int Chunk::solid_count() const {
    int count = 0;
    for (const auto& v : voxels_) count += v.solid() ? 1 : 0;
    return count;
}

// ──────────────────────────────────────────────────────────────────────────────
// Greedy meshing
// For each of the 6 face directions, sweep through slices perpendicular to
// that axis and merge adjacent solid voxels with the same biome into quads.
// Each quad is triangulated and emitted as 6 floats × 4 vertices:
//   [x, y, z, nx, ny, nz, biome_id]  (7 floats per vertex)
// ──────────────────────────────────────────────────────────────────────────────

namespace {

inline void push_quad(std::vector<float>& out,
                      float p[4][3], float nx, float ny, float nz, float biome) {
    // Two triangles: 0-1-2 and 0-2-3
    for (int i : config::QUAD_TRIANGLES) {
        out.push_back(p[i][0]); out.push_back(p[i][1]); out.push_back(p[i][2]);
        out.push_back(nx); out.push_back(ny); out.push_back(nz);
        out.push_back(biome);
        
    }
}

}  // namespace

std::vector<float> Chunk::build_mesh(
    const std::function<bool(int, int, int)>& neighbor_solid_query) const {
    std::vector<float> out;
    out.reserve(SIZE * SIZE * 7 * 6);  // rough upper bound
    const IVec3 world_origin_pos = world_origin();

    // For each of the 6 face directions:
    for (int d = 0; d < 6; ++d) {
        int fa = config::AXIS_MAP[d][0];  // face axis index
        int ua = config::AXIS_MAP[d][1];  // u axis index
        int va = config::AXIS_MAP[d][2];  // v axis index
        int sign = (d % 2 == 0) ? 1 : -1;  // +1 for positive faces

        float nx = config::NORMS[d][0], ny = config::NORMS[d][1], nz = config::NORMS[d][2];

        // Sweep along face axis
        for (int layer = 0; layer < SIZE; ++layer) {
            // Build mask: for each (u,v) cell, store the biome if this face
            // should be drawn, or 0xFF if not.
            std::array<uint8_t, SIZE * SIZE> mask;
            mask.fill(0xFF);

            for (int u = 0; u < SIZE; ++u) {
                for (int v = 0; v < SIZE; ++v) {
                    int abc[3];
                    abc[fa] = layer;
                    abc[ua] = u;
                    abc[va] = v;
                    Voxel cur = get(abc[0], abc[1], abc[2]);
                    if (!cur.solid()) continue;

                    // Check neighbor in normal direction.
                    int nabc[3] = {abc[0], abc[1], abc[2]};
                    nabc[fa] += sign;
                    bool neighbor_solid = false;
                    if (nabc[fa] >= 0 && nabc[fa] < SIZE) {
                        neighbor_solid = get(nabc[0], nabc[1], nabc[2]).solid();
                    } else if (neighbor_solid_query) {
                        neighbor_solid = neighbor_solid_query(
                            world_origin_pos.x + nabc[0],
                            world_origin_pos.y + nabc[1],
                            world_origin_pos.z + nabc[2]);
                    }
                    if (!neighbor_solid) {
                        mask[u + SIZE * v] = static_cast<uint8_t>(cur.biome());
                    }
                }
            }

            // Greedy merge along u first, then v
            for (int v = 0; v < SIZE; ++v) {
                for (int u = 0; u < SIZE; ) {
                    uint8_t biome = mask[u + SIZE * v];
                    if (biome == 0xFF) { ++u; continue; }

                    // Extend in u direction
                    int w = 1;
                    while (u + w < SIZE && mask[u + w + SIZE * v] == biome) ++w;

                    // Extend in v direction
                    int h = 1;
                    bool done = false;
                    while (!done && v + h < SIZE) {
                        for (int k = 0; k < w; ++k) {
                            if (mask[u + k + SIZE * (v + h)] != biome) {
                                done = true; break;
                            }
                        }
                        if (!done) ++h;
                    }

                    // Emit quad at (layer, u, v) with size (w, h)
                    float abc0[3], abc1[3], abc2[3], abc3[3];
                    auto fill_pos = [&](float (&p)[3], int layer_v, int uu, int vv) {
                        p[fa] = static_cast<float>(layer_v + (sign > 0 ? 1 : 0));
                        p[ua] = static_cast<float>(uu);
                        p[va] = static_cast<float>(vv);
                        // Offset by world origin
                        p[0] += static_cast<float>(chunk_coord_.x * SIZE);
                        p[1] += static_cast<float>(chunk_coord_.y * SIZE);
                        p[2] += static_cast<float>(chunk_coord_.z * SIZE);
                    };
                    fill_pos(abc0, layer, u,     v    );
                    fill_pos(abc1, layer, u + w, v    );
                    fill_pos(abc2, layer, u + w, v + h);
                    fill_pos(abc3, layer, u,     v + h);

                    float pts[4][3];
                    std::copy(abc0, abc0+3, pts[0]);
                    std::copy(abc1, abc1+3, pts[1]);
                    std::copy(abc2, abc2+3, pts[2]);
                    std::copy(abc3, abc3+3, pts[3]);

                    if (config::FLIP[d]) std::swap(pts[1], pts[3]);
                    push_quad(out, pts, nx, ny, nz, static_cast<float>(biome));

                    // Clear mask region
                    for (int dv = 0; dv < h; ++dv)
                        for (int du = 0; du < w; ++du)
                            mask[u + du + SIZE * (v + dv)] = 0xFF;

                    u += w;
                }
            }
        }
    }
    return out;
}

}  // namespace voxel
