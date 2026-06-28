#include "wfc.hpp"
#include <algorithm>
#include <cassert>
#include <queue>
#include <stdexcept>

namespace voxel {

// ──────────────────────────────────────────────────────────────────────────────
// Default adjacency rules for WFC biome grid.
// Tile indices correspond to (static_cast<int>(BiomeType) - 1) since Air=0.
// ──────────────────────────────────────────────────────────────────────────────

static inline int tile(BiomeType b) {
    int idx = static_cast<int>(b) - 1;  // shift: Ocean(1)→0, Beach(2)→1 …
    return idx;
}

AdjacencyMatrix default_adjacency() {
    AdjacencyMatrix adj{};
    // Initialise: every tile can be adjacent to itself.
    for (int i = 0; i < WFC_TILE_COUNT; ++i) adj[i].set(i);

    // Biome-to-biome geographic transitions:
    auto allow = [&](BiomeType a, BiomeType b) {
        int ia = tile(a), ib = tile(b);
        adj[ia].set(ib);
        adj[ib].set(ia);
    };

    allow(BiomeType::Ocean,    BiomeType::Beach);
    allow(BiomeType::Beach,    BiomeType::Plains);
    allow(BiomeType::Beach,    BiomeType::Desert);
    allow(BiomeType::Beach,    BiomeType::Forest);
    allow(BiomeType::Plains,   BiomeType::Forest);
    allow(BiomeType::Plains,   BiomeType::Desert);
    allow(BiomeType::Plains,   BiomeType::Tundra);
    allow(BiomeType::Plains,   BiomeType::Mountain);
    allow(BiomeType::Forest,   BiomeType::Mountain);
    allow(BiomeType::Forest,   BiomeType::Tundra);
    allow(BiomeType::Desert,   BiomeType::Mountain);
    allow(BiomeType::Mountain, BiomeType::Snow);
    allow(BiomeType::Tundra,   BiomeType::Snow);

    return adj;
}

// ──────────────────────────────────────────────────────────────────────────────

WFCSolver::WFCSolver(int width, int height,
                     const AdjacencyMatrix& adj, uint64_t seed)
    : width_(width), height_(height), adj_(adj), rng_(seed) {
    grid_.resize(static_cast<size_t>(width * height));
    reset_grid();
}

void WFCSolver::reset_grid() {
    for (auto& cell : grid_) {
        cell.options.set();  // all bits set = all tiles possible
        cell.tile.reset();
    }
}

void WFCSolver::pin(int cx, int cz, BiomeType biome) {
    int idx = cell_idx(cx, cz);
    grid_[idx].options.reset();
    int t = tile(biome);
    if (t >= 0 && t < WFC_TILE_COUNT) grid_[idx].options.set(t);
    grid_[idx].tile = t;
}

bool WFCSolver::solve(int max_restarts) {
    for (int attempt = 0; attempt < max_restarts; ++attempt) {
        if (attempt > 0) reset_grid();

        bool contradiction = false;
        while (true) {
            int idx = min_entropy_cell();
            if (idx == -1) break;  // all collapsed

            if (!collapse(idx)) { contradiction = true; break; }
            if (!propagate(idx)) { contradiction = true; break; }
        }
        if (!contradiction) return true;
    }
    // Fill remaining cells with Plains as fallback.
    for (auto& cell : grid_) {
        if (!cell.collapsed()) {
            cell.tile = tile(BiomeType::Plains);
        }
    }
    return false;
}

int WFCSolver::min_entropy_cell() const {
    int best = -1, best_ent = WFC_TILE_COUNT + 1;
    for (int i = 0; i < static_cast<int>(grid_.size()); ++i) {
        const auto& c = grid_[i];
        if (c.collapsed()) continue;
        int ent = c.entropy();
        if (ent < best_ent) { best_ent = ent; best = i; }
    }
    return best;
}

bool WFCSolver::collapse(int idx) {
    auto& cell = grid_[idx];
    if (cell.options.none()) return false;

    // Uniform random selection among remaining options.
    std::vector<int> opts;
    for (int i = 0; i < WFC_TILE_COUNT; ++i)
        if (cell.options.test(i)) opts.push_back(i);

    std::uniform_int_distribution<int> dist(0, static_cast<int>(opts.size()) - 1);
    int chosen = opts[dist(rng_)];
    cell.options.reset();
    cell.options.set(chosen);
    cell.tile = chosen;
    return true;
}

bool WFCSolver::propagate(int start_idx) {
    // BFS arc-consistency propagation.
    std::queue<int> q;
    q.push(start_idx);

    // 4-connected neighbours (2-D grid).
    const int DX[] = {1, -1, 0, 0};
    const int DZ[] = {0, 0, 1, -1};

    while (!q.empty()) {
        int ci = q.front(); q.pop();
        int cx = ci % width_, cz = ci / width_;
        const auto& src = grid_[ci];

        for (int d = 0; d < 4; ++d) {
            int nx = cx + DX[d], nz = cz + DZ[d];
            if (nx < 0 || nx >= width_ || nz < 0 || nz >= height_) continue;
            int ni = cell_idx(nx, nz);
            auto& nb = grid_[ni];
            if (nb.collapsed()) continue;

            // Compute allowed options for neighbour given source options.
            std::bitset<WFC_TILE_COUNT> allowed;
            for (int st = 0; st < WFC_TILE_COUNT; ++st) {
                if (!src.options.test(st)) continue;
                allowed |= adj_[st];
            }

            std::bitset<WFC_TILE_COUNT> new_opts = nb.options & allowed;
            if (new_opts.none()) return false;  // contradiction

            if (new_opts != nb.options) {
                nb.options = new_opts;
                q.push(ni);
            }
        }
    }
    return true;
}

BiomeType WFCSolver::at(int cx, int cz) const {
    const auto& cell = grid_[cell_idx(cx, cz)];
    if (!cell.tile.has_value()) return BiomeType::Plains;
    int t = *cell.tile;
    if (t < 0 || t >= WFC_TILE_COUNT) return BiomeType::Plains;
    return static_cast<BiomeType>(t + 1);  // re-add Air offset
}

void WFCSolver::apply_to_chunk(Chunk& chunk,
                                int chunk_cx, int chunk_cz,
                                int wfc_origin_cx, int wfc_origin_cz) const {
    // For each voxel column in the chunk, look up the WFC biome and apply it
    // to the topmost solid voxel in the column.
    for (int lx = 0; lx < Chunk::SIZE; ++lx) {
        for (int lz = 0; lz < Chunk::SIZE; ++lz) {
            int wfc_cx = (chunk_cx - wfc_origin_cx) * Chunk::SIZE + lx;
            int wfc_cz = (chunk_cz - wfc_origin_cz) * Chunk::SIZE + lz;
            if (wfc_cx < 0 || wfc_cx >= width_ ||
                wfc_cz < 0 || wfc_cz >= height_)
                continue;

            BiomeType b = at(wfc_cx, wfc_cz);

            // Apply biome to all solid voxels in the column.
            for (int ly = Chunk::SIZE - 1; ly >= 0; --ly) {
                Voxel v = chunk.get(lx, ly, lz);
                if (v.solid()) {
                    v.set_biome(b);
                    chunk.set(lx, ly, lz, v);
                }
            }
        }
    }
}

}  // namespace voxel
