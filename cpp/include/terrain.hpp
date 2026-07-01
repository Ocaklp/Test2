#pragma once
#include "config.hpp"
#include "noise.hpp"
#include "voxel.hpp"
#include "chunk.hpp"
#include <memory>

namespace voxel {

// Parameters that control terrain shape and biome assignment.
struct TerrainParams {
    float surface_scale  = config::DEFAULT_SURFACE_SCALE;  // horizontal frequency of height noise
    float biome_scale    = config::DEFAULT_BIOME_SCALE; // frequency of biome selection noise (lower = larger biomes)
    int   sea_level      = config::DEFAULT_SEA_LEVEL; // voxel Y below which Ocean is assigned
    int   beach_height   = config::DEFAULT_BEACH_HEIGHT; // ±voxels around sea_level that become Beach
    int   snow_level     = config::DEFAULT_SNOW_LEVEL; // voxel Y above which Snow is assigned
    int   mountain_level = config::DEFAULT_MOUNTAIN_LEVEL; // voxel Y above which Mountain is assigned
    int   fbm_octaves    = config::DEFAULT_FBM_OCTAVES;       // octaves for surface height FBM
    float height_amp     = config::DEFAULT_HEIGHT_AMP;   // amplitude multiplier for height map
    float height_base    = config::DEFAULT_HEIGHT_BASE;   // baseline height added to height map
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
