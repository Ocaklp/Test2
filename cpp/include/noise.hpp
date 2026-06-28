#pragma once
#include <array>
#include <cstdint>

namespace voxel {

// OpenSimplex2-style 3D Simplex noise and 2D fractal noise utilities.
// Generates smooth, tileable noise suitable for terrain height and biome maps.
class Noise {
public:
    explicit Noise(uint64_t seed = 0);

    // 2D noise in [-1, 1]
    float noise2(float x, float y) const;

    // 3D noise in [-1, 1]
    float noise3(float x, float y, float z) const;

    // Fractal Brownian Motion: sum of octaves of noise
    float fbm2(float x, float y, int octaves, float persistence = 0.5f,
               float lacunarity = 2.0f) const;

    float fbm3(float x, float y, float z, int octaves,
               float persistence = 0.5f, float lacunarity = 2.0f) const;

    // Ridged noise for mountain-like terrain
    float ridged2(float x, float y, int octaves, float persistence = 0.5f,
                  float lacunarity = 2.0f) const;

    uint64_t seed() const { return seed_; }

private:
    uint64_t seed_;
    std::array<uint8_t, 512> perm_;  // permutation table

    void init_perm();
    float grad3(int hash, float x, float y, float z) const;
    float grad2(int hash, float x, float y) const;
    float raw2(float x, float y) const;
    float raw3(float x, float y, float z) const;
};

}  // namespace voxel
