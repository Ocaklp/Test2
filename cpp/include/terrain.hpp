#pragma once
#include "noise.hpp"
#include "voxel.hpp"
#include "chunk.hpp"
#include <memory>

namespace voxel {

// Parameters that control terrain shape and biome assignment.
struct TerrainParams {
    float surface_scale  = 0.005f;  // horizontal frequency of height noise
    float biome_scale    = 0.0008f; // frequency of biome selection noise (lower = larger biomes)
    int   sea_level      = 8;       // voxel Y below which Ocean is assigned
    int   beach_height   = 2;       // ±voxels around sea_level that become Beach
    int   snow_level     = 90;      // voxel Y above which Snow is assigned
    int   mountain_level = 35;      // voxel Y above which Mountain is assigned
    int   fbm_octaves    = 4;       // octaves for surface height FBM
    float height_amp     = 28.0f;   // amplitude multiplier for height map
    float height_base    = 14.0f;   // baseline height added to height map
};

// Generates terrain inside a Chunk given world-space noise and parameters.
class TerrainGenerator {
public:
    TerrainGenerator(uint64_t seed, const TerrainParams& params = {});

    // Fill `chunk` with terrain voxels.
    void generate(Chunk& chunk) const;

    // Return the terrain height at a given world-space (x, z) column.
    float surface_height(float wx, float wz) const;

    // Return the biome at a given world-space (x, z) position.
    BiomeType biome_at(float wx, float wz, float wy) const;

    const TerrainParams& params() const { return params_; }
    uint64_t             seed()   const { return seed_; }

private:
    BiomeType macro_biome_at(float wx, float wz) const;
    float biome_surface_height(BiomeType biome, float wx, float wz) const;

    uint64_t      seed_;
    TerrainParams params_;
    Noise         height_noise_;
    Noise         biome_noise_;
};

}  // namespace voxel
