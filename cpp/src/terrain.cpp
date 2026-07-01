#include "config.hpp"
#include "terrain.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>

namespace voxel {

namespace {

constexpr float kTemperatureOffset = config::TEMPERATURE_OFFSET;
constexpr float kMoistureOffset = config::MOISTURE_OFFSET;
constexpr float kContinentalOffset = config::CONTINENTAL_OFFSET;
constexpr float kContinentalScale = config::CONTINENTAL_SCALE;
constexpr float kPeakOffsetX = config::PEAK_OFFSET_X;
constexpr float kPeakOffsetZ = config::PEAK_OFFSET_Z;
constexpr float kBiomeBlendSharpness = config::BIOME_BLEND_SHARPNESS;
constexpr float kTransitionScale = config::TRANSITION_SCALE;
constexpr float kTransitionAmp   = config::TRANSITION_AMP;
constexpr int   kUpsampleStride  = config::UPSAMPLE_STRIDE;
constexpr int   kUpsampleGridSize = config::UPSAMPLE_GRID_SIZE;
constexpr size_t kUpsampleGridArea = config::UPSAMPLE_GRID_AREA;

// Keep compatibility with legacy TerrainParams amplitude/base interpretation.
constexpr float kLegacyHeightAmp = config::LEGACY_HEIGHT_AMP;
constexpr float kLegacyHeightBase = config::LEGACY_HEIGHT_BASE;

struct ClimateSample {
    float temperature;
    float moisture;
    float continentalness;
};

struct AxisUpsampleSample {
    int i0;
    int i1;
    float t;
};

template <int Stride>
std::array<AxisUpsampleSample, Chunk::SIZE> build_axis_upsample_lut() {
    std::array<AxisUpsampleSample, Chunk::SIZE> lut{};
    for (int i = 0; i < Chunk::SIZE; ++i) {
        int cell = i / Stride;
        int local = i - cell * Stride;
        lut[i] = AxisUpsampleSample{
            cell, cell + 1, static_cast<float>(local) / static_cast<float>(Stride)
        };
    }
    return lut;
}

inline float bilerp(const std::array<float, kUpsampleGridArea>& grid,
                    int gx0, int gx1, float tx,
                    int gz0, int gz1, float tz) {
    auto at = [&](int gx, int gz) -> float {
        return grid[gx + kUpsampleGridSize * gz];
    };

    float v00 = at(gx0, gz0);
    float v10 = at(gx1, gz0);
    float v01 = at(gx0, gz1);
    float v11 = at(gx1, gz1);
    float vx0 = v00 + (v10 - v00) * tx;
    float vx1 = v01 + (v11 - v01) * tx;
    return vx0 + (vx1 - vx0) * tz;
}

const config::BiomeProfile& profile_for(BiomeType biome) {
    switch (biome) {
        case BiomeType::Plains:   return config::BIOME_PROFILES[0];
        case BiomeType::Forest:   return config::BIOME_PROFILES[1];
        case BiomeType::Desert:   return config::BIOME_PROFILES[2];
        case BiomeType::Tundra:   return config::BIOME_PROFILES[3];
        case BiomeType::Mountain: return config::BIOME_PROFILES[4];
        default:                  return config::BIOME_PROFILES[0];
    }
}

ClimateSample sample_climate(const TerrainParams& params,
                             const Noise& biome_noise,
                             const Noise& height_noise,
                             float wx, float wz) {
    constexpr float kBiomeScaleMultiplier = config::DEFAULT_BIOME_SCALE_MULTIPLIER;
    float s = params.biome_scale * kBiomeScaleMultiplier;
    float moisture = biome_noise.fbm2((wx + kMoistureOffset) * s, (wz + kMoistureOffset) * s, 2);
    float temp = biome_noise.fbm2((wx + kTemperatureOffset) * s, (wz + kTemperatureOffset) * s, 2);
    float continental = height_noise.fbm2((wx - kContinentalOffset) * s * kContinentalScale,
                                          (wz - kContinentalOffset) * s * kContinentalScale, 2);
    return ClimateSample{temp, moisture, continental};
}

}  // namespace

TerrainGenerator::TerrainGenerator(uint64_t seed, const TerrainParams& params)
    : seed_(seed), params_(params),
      height_noise_(seed),
      biome_noise_(seed ^ config::BIOME_SEED_MODIFIER) {}

BiomeType TerrainGenerator::macro_biome_at(float wx, float wz) const {
    ClimateSample climate = sample_climate(params_, biome_noise_, height_noise_, wx, wz);

    const config::BiomeProfile* best = &config::BIOME_PROFILES.front();
    float best_score = -(
        std::abs(climate.temperature - best->target_temp) +
        std::abs(climate.moisture - best->target_moisture)
    ) + climate.continentalness * best->continental_bias;
    for (size_t i = 1; i < config::BIOME_PROFILES.size(); ++i) {
        const auto& profile = config::BIOME_PROFILES[i];
        float climate_dist = std::abs(climate.temperature - profile.target_temp) +
                             std::abs(climate.moisture - profile.target_moisture);
        float score = -climate_dist + climate.continentalness * profile.continental_bias;
        if (score > best_score) {
            best_score = score;
            best = &profile;
        }
    }
    return best->biome;
}

float TerrainGenerator::biome_surface_height(BiomeType biome, float wx, float wz) const {
    const auto& p = profile_for(biome);

    float base = height_noise_.fbm2(wx * params_.surface_scale,
                                    wz * params_.surface_scale,
                                    params_.fbm_octaves);
    float ridge = height_noise_.ridged2(wx * params_.surface_scale * p.peak_frequency,
                                        wz * params_.surface_scale * p.peak_frequency, 4);
    float detail = height_noise_.fbm2((wx + 4096.0f) * params_.surface_scale * 2.2f,
                                      (wz + 4096.0f) * params_.surface_scale * 2.2f, 2);

    float shaped = base * (1.0f - p.ridge_weight - p.detail_weight) +
                   (ridge * 2.0f - 1.0f) * p.ridge_weight +
                   detail * p.detail_weight;

    float t = std::clamp(0.5f + 0.5f * shaped, 0.0f, 1.0f);
    float h = p.min_height + (p.max_height - p.min_height) * t;

    // Peak layering controls the number and intensity of peaks per biome.
    float peak_bonus = 0.0f;
    float peak_freq = p.peak_frequency;
    for (int i = 0; i < p.peak_layers; ++i) {
        float peak = height_noise_.ridged2(
            (wx + kPeakOffsetX * static_cast<float>(i + 1)) * params_.surface_scale * peak_freq,
            (wz + kPeakOffsetZ * static_cast<float>(i + 1)) * params_.surface_scale * peak_freq,
            2 + i);
        peak_bonus += std::pow(peak, 3.0f) * (p.peak_height / static_cast<float>(i + 1));
        peak_freq *= 1.5f;
    }

    float raw_height = h + peak_bonus;
    float amp_scale = params_.height_amp / kLegacyHeightAmp;
    return params_.height_base + (raw_height - kLegacyHeightBase) * amp_scale;
}

float TerrainGenerator::surface_height(float wx, float wz) const {
    ClimateSample climate = sample_climate(params_, biome_noise_, height_noise_, wx, wz);

    std::array<float, config::BIOME_PROFILES.size()> raw_scores{};
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < config::BIOME_PROFILES.size(); ++i) {
        const auto& profile = config::BIOME_PROFILES[i];
        float climate_dist = std::abs(climate.temperature - profile.target_temp) +
                             std::abs(climate.moisture - profile.target_moisture);
        float score = -climate_dist + climate.continentalness * profile.continental_bias;
        raw_scores[i] = score;
        max_score = std::max(max_score, score);
    }

    float weight_sum = 0.0f;
    float blended_height = 0.0f;
    for (size_t i = 0; i < config::BIOME_PROFILES.size(); ++i) {
        // Softmax-like weighting; subtracting max_score keeps exp() numerically stable.
        float w = std::exp((raw_scores[i] - max_score) * kBiomeBlendSharpness);
        blended_height += w * biome_surface_height(config::BIOME_PROFILES[i].biome, wx, wz);
        weight_sum += w;
    }

    constexpr float kWeightSumEpsilon = 1e-6f;
    if (weight_sum < kWeightSumEpsilon) {
        return biome_surface_height(BiomeType::Plains, wx, wz);
    }
    return blended_height / weight_sum;
}

BiomeType TerrainGenerator::biome_at(float wx, float wz, float wy) const {
    BiomeType base_biome = macro_biome_at(wx, wz);

    // Noise-based offset applied to Y thresholds so that the biome
    // boundaries are wavy surfaces rather than flat horizontal planes.
    float y_offset = biome_noise_.fbm2(wx * kTransitionScale,
                                       wz * kTransitionScale, 2) * kTransitionAmp;

    float adj_sea   = static_cast<float>(params_.sea_level) + y_offset;
    float adj_beach = static_cast<float>(params_.sea_level + params_.beach_height) + y_offset;
    float adj_snow  = static_cast<float>(params_.snow_level) + y_offset;

    if (wy < adj_sea)   return BiomeType::Ocean;
    if (wy <= adj_beach) return BiomeType::Beach;
    if (wy >= adj_snow)  return BiomeType::Snow;
    return base_biome;
}

void TerrainGenerator::generate(Chunk& chunk) const {
    const IVec3 wo = chunk.world_origin();
    chunk.fill(Voxel{});
    auto& voxels = chunk.voxels();

    static const auto x_lut = build_axis_upsample_lut<kUpsampleStride>();
    static const auto z_lut = build_axis_upsample_lut<kUpsampleStride>();

    std::array<float, kUpsampleGridArea> surface_grid{};
    std::array<float, kUpsampleGridArea> transition_grid{};
    for (int gz = 0; gz < kUpsampleGridSize; ++gz) {
        for (int gx = 0; gx < kUpsampleGridSize; ++gx) {
            float wx = static_cast<float>(wo.x + gx * kUpsampleStride);
            float wz = static_cast<float>(wo.z + gz * kUpsampleStride);
            surface_grid[gx + kUpsampleGridSize * gz] = surface_height(wx, wz);
            transition_grid[gx + kUpsampleGridSize * gz] =
                biome_noise_.fbm2(wx * kTransitionScale, wz * kTransitionScale, 2) * kTransitionAmp;
        }
    }

    for (int lx = 0; lx < Chunk::SIZE; ++lx) {
        const auto& xs = x_lut[lx];
        for (int lz = 0; lz < Chunk::SIZE; ++lz) {
            const auto& zs = z_lut[lz];
            float wx = static_cast<float>(wo.x + lx);
            float wz = static_cast<float>(wo.z + lz);

            float surf_h = bilerp(surface_grid, xs.i0, xs.i1, xs.t, zs.i0, zs.i1, zs.t);
            int top_local = static_cast<int>(std::floor(surf_h - static_cast<float>(wo.y)));
            if (top_local < 0) {
                continue;
            }
            top_local = std::min(top_local, Chunk::SIZE - 1);

            BiomeType base_biome = macro_biome_at(wx, wz);
            float y_offset = bilerp(transition_grid, xs.i0, xs.i1, xs.t, zs.i0, zs.i1, zs.t);

            float adj_sea   = static_cast<float>(params_.sea_level) + y_offset;
            float adj_beach = static_cast<float>(params_.sea_level + params_.beach_height) + y_offset;
            float adj_snow  = static_cast<float>(params_.snow_level) + y_offset;

            int ocean_end = static_cast<int>(std::ceil(adj_sea)) - 1 - wo.y;
            int beach_end = static_cast<int>(std::floor(adj_beach)) - wo.y;
            int snow_begin = static_cast<int>(std::ceil(adj_snow)) - wo.y;

            auto fill_run = [&](int start_y, int end_y, Voxel v) {
                start_y = std::max(start_y, 0);
                end_y = std::min(end_y, top_local);
                if (start_y > end_y) {
                    return;
                }
                constexpr int kYStride = (1 << Chunk::SHIFT);
                int idx = Chunk::idx_unchecked(lx, start_y, lz);
                for (int ly = start_y; ly <= end_y; ++ly) {
                    voxels[idx] = v;
                    idx += kYStride;
                }
            };

            // Vertical runs (run-length style): ocean -> beach -> base biome -> snow.
            int cursor = 0;
            fill_run(cursor, ocean_end, Voxel(true, BiomeType::Ocean));
            cursor = ocean_end + 1;
            fill_run(cursor, beach_end, Voxel(true, BiomeType::Beach));
            cursor = beach_end + 1;

            if (snow_begin <= top_local) {
                fill_run(cursor, snow_begin - 1, Voxel(true, base_biome));
                fill_run(snow_begin, top_local, Voxel(true, BiomeType::Snow));
            } else {
                fill_run(cursor, top_local, Voxel(true, base_biome));
            }
        }
    }
}

}  // namespace voxel
