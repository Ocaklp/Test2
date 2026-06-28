#pragma once
#include "voxel.hpp"
#include "chunk.hpp"
#include <array>
#include <bitset>
#include <cstdint>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

namespace voxel {

// ──────────────────────────────────────────────────────────────────────────────
// Tile-based Wave Function Collapse implementation for smooth biome transitions.
//
// The WFC grid is a 2-D lattice of *cells*, each of which must be assigned one
// BiomeType tile (excluding Air).  Adjacency rules express which biome pairs
// are compatible when placed next to each other.  The solver iterates:
//   1. Find the uncollapsed cell with lowest entropy (fewest options).
//   2. Collapse it to one of its remaining options (weighted random).
//   3. Propagate constraints to neighbours.
//   4. Repeat until solved or a contradiction is detected (triggers restart).
// ──────────────────────────────────────────────────────────────────────────────

// Number of usable biome tile types (Air excluded).
static constexpr int WFC_TILE_COUNT = static_cast<int>(BiomeType::COUNT) - 1;

// Which biomes may appear next to each other (symmetric relation).
// Default rules: all biomes can be adjacent to themselves; additionally
// smooth geographic gradients are enforced (e.g. Ocean↔Beach, Beach↔Plains …).
using AdjacencyMatrix =
    std::array<std::bitset<WFC_TILE_COUNT>, WFC_TILE_COUNT>;

AdjacencyMatrix default_adjacency();

// ──────────────────────────────────────────────────────────────────────────────

struct WFCCell {
    std::bitset<WFC_TILE_COUNT> options;  // remaining possible tiles
    std::optional<int>          tile;     // set when collapsed

    bool collapsed()  const { return tile.has_value(); }
    int  entropy()    const { return static_cast<int>(options.count()); }
};

// Solver for a width × height 2-D WFC grid.
class WFCSolver {
public:
    WFCSolver(int width, int height,
              const AdjacencyMatrix& adj = default_adjacency(),
              uint64_t seed = 42);

    // Run the solver; returns false if max_restarts exceeded.
    bool solve(int max_restarts = 10);

    // Query the collapsed biome at cell (cx, cz).  Returns BiomeType::Plains
    // when not yet solved or collapsed to an invalid state.
    BiomeType at(int cx, int cz) const;

    // Seed selected cells with known biomes (must be called before solve()).
    void pin(int cx, int cz, BiomeType biome);

    int width()  const { return width_; }
    int height() const { return height_; }

    // Apply the solved biome map to the top surface layer of a chunk.
    // `chunk_cx` and `chunk_cz` are the chunk's grid coordinates.
    // `wfc_origin_cx` and `wfc_origin_cz` are the WFC grid offset in chunk units.
    void apply_to_chunk(Chunk& chunk, int chunk_cx, int chunk_cz,
                        int wfc_origin_cx = 0, int wfc_origin_cz = 0) const;

private:
    int    width_, height_;
    AdjacencyMatrix adj_;
    std::mt19937_64 rng_;

    std::vector<WFCCell> grid_;

    int cell_idx(int cx, int cz) const { return cz * width_ + cx; }

    void reset_grid();
    // Returns index of lowest-entropy uncollapsed cell, or -1 if done.
    int  min_entropy_cell() const;
    // Collapse cell `idx` to a random allowed option.
    bool collapse(int idx);
    // Arc-consistency propagation from `idx`.  Returns false on contradiction.
    bool propagate(int start_idx);
};

}  // namespace voxel
