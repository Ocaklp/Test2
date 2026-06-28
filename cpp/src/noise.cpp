#include "noise.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace voxel {

// ──────────────────────────────────────────────────────────────────────────────
// Simplex-style noise implementation
// Based on the classic Simplex noise algorithm by Ken Perlin.
// ──────────────────────────────────────────────────────────────────────────────

Noise::Noise(uint64_t seed) : seed_(seed) { init_perm(); }

void Noise::init_perm() {
    // Fill [0..255] then Fisher-Yates shuffle seeded by `seed_`.
    for (int i = 0; i < 256; ++i) perm_[i] = static_cast<uint8_t>(i);
    uint64_t s = seed_;
    for (int i = 255; i > 0; --i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        int j = static_cast<int>(s % static_cast<uint64_t>(i + 1));
        std::swap(perm_[i], perm_[j]);
    }
    // Mirror to avoid modular indexing.
    for (int i = 0; i < 256; ++i) perm_[256 + i] = perm_[i];
}

// ──── 2-D Simplex noise ────────────────────────────────────────────────────

static const float F2 = 0.5f * (std::sqrt(3.0f) - 1.0f);
static const float G2 = (3.0f - std::sqrt(3.0f)) / 6.0f;
static const float F3 = 1.0f / 3.0f;
static const float G3 = 1.0f / 6.0f;

float Noise::grad2(int hash, float x, float y) const {
    int h = hash & 7;
    float u = h < 4 ? x : y;
    float v = h < 4 ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

float Noise::raw2(float x, float y) const {
    float s  = (x + y) * F2;
    int   i  = static_cast<int>(std::floor(x + s));
    int   j  = static_cast<int>(std::floor(y + s));
    float t  = static_cast<float>(i + j) * G2;
    float X0 = i - t, Y0 = j - t;
    float x0 = x - X0, y0 = y - Y0;

    int i1 = (x0 > y0) ? 1 : 0;
    int j1 = (x0 > y0) ? 0 : 1;

    float x1 = x0 - i1 + G2, y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2, y2 = y0 - 1.0f + 2.0f * G2;

    int ii = i & 255, jj = j & 255;
    int gi0 = perm_[ii +        perm_[jj]];
    int gi1 = perm_[ii + i1 + perm_[jj + j1]];
    int gi2 = perm_[ii + 1  + perm_[jj + 1]];

    auto contrib = [&](float rx, float ry, int gi) -> float {
        float t = 0.5f - rx * rx - ry * ry;
        if (t < 0.0f) return 0.0f;
        t *= t;
        return t * t * grad2(gi, rx, ry);
    };

    return std::clamp(70.0f * (contrib(x0, y0, gi0) + contrib(x1, y1, gi1) + contrib(x2, y2, gi2)), -1.0f, 1.0f);
}

float Noise::noise2(float x, float y) const { return raw2(x, y); }

// ──── 3-D Simplex noise ────────────────────────────────────────────────────

static const int GRAD3[12][3] = {
    {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
    {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
    {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}
};

float Noise::grad3(int hash, float x, float y, float z) const {
    const int* g = GRAD3[hash % 12];
    return g[0] * x + g[1] * y + g[2] * z;
}

float Noise::raw3(float x, float y, float z) const {
    float s  = (x + y + z) * F3;
    int   i  = static_cast<int>(std::floor(x + s));
    int   j  = static_cast<int>(std::floor(y + s));
    int   k  = static_cast<int>(std::floor(z + s));
    float t  = static_cast<float>(i + j + k) * G3;

    float x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);

    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if      (y0 >= z0) { i1=1;j1=0;k1=0; i2=1;j2=1;k2=0; }
        else if (x0 >= z0) { i1=1;j1=0;k1=0; i2=1;j2=0;k2=1; }
        else               { i1=0;j1=0;k1=1; i2=1;j2=0;k2=1; }
    } else {
        if      (y0 < z0)  { i1=0;j1=0;k1=1; i2=0;j2=1;k2=1; }
        else if (x0 < z0)  { i1=0;j1=1;k1=0; i2=0;j2=1;k2=1; }
        else               { i1=0;j1=1;k1=0; i2=1;j2=1;k2=0; }
    }

    float x1=x0-i1+G3, y1=y0-j1+G3, z1=z0-k1+G3;
    float x2=x0-i2+2*G3, y2=y0-j2+2*G3, z2=z0-k2+2*G3;
    float x3=x0-1+3*G3, y3=y0-1+3*G3, z3=z0-1+3*G3;

    int ii=i&255, jj=j&255, kk=k&255;
    int gi0 = perm_[ii  +perm_[jj  +perm_[kk  ]]]%12;
    int gi1 = perm_[ii+i1+perm_[jj+j1+perm_[kk+k1]]]%12;
    int gi2 = perm_[ii+i2+perm_[jj+j2+perm_[kk+k2]]]%12;
    int gi3 = perm_[ii+1 +perm_[jj+1 +perm_[kk+1 ]]]%12;

    auto contrib = [&](float rx, float ry, float rz, int gi) -> float {
        float t = 0.6f - rx*rx - ry*ry - rz*rz;
        if (t < 0) return 0.0f;
        t *= t;
        return t * t * grad3(gi, rx, ry, rz);
    };

    return std::clamp(32.0f * (contrib(x0,y0,z0,gi0)+contrib(x1,y1,z1,gi1)
                   +contrib(x2,y2,z2,gi2)+contrib(x3,y3,z3,gi3)), -1.0f, 1.0f);
}

float Noise::noise3(float x, float y, float z) const { return raw3(x, y, z); }

// ──── Fractal Brownian Motion ──────────────────────────────────────────────

float Noise::fbm2(float x, float y, int octaves,
                  float persistence, float lacunarity) const {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_val = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value   += raw2(x * frequency, y * frequency) * amplitude;
        max_val += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / max_val;
}

float Noise::fbm3(float x, float y, float z, int octaves,
                  float persistence, float lacunarity) const {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_val = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        value   += raw3(x*frequency, y*frequency, z*frequency) * amplitude;
        max_val += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / max_val;
}

float Noise::ridged2(float x, float y, int octaves,
                     float persistence, float lacunarity) const {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, max_val = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        float n = 1.0f - std::abs(std::clamp(raw2(x * frequency, y * frequency), -1.0f, 1.0f));
        value   += n * amplitude;
        max_val += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / max_val;
}

}  // namespace voxel
