#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include <string_view> // Added for zero-overhead static strings

namespace voxel {

// Forward declaration of BiomeType to use in BiomeProfile
enum class BiomeType : uint8_t;

namespace config {

// ==========================================
// MATH & SYSTEM UTILS
// ==========================================
constexpr float DEG_TO_RAD = 3.14159265f / 180.0f;

// ==========================================
// ENGINE & RENDERER TARGETS
// ==========================================
constexpr int MAX_UPLOADS_PER_FRAME = 64;
constexpr int TARGET_FPS            = 60;

// ==========================================
// LEVEL OF DETAIL (LOD)
// ==========================================
constexpr int   LOD_LEVELS = 4;
inline constexpr float LOD_DIST[4] = {4.0f, 8.0f, 16.0f, 1e9f};

// ==========================================
// CAMERA & PLAYER MOVEMENT
// ==========================================
inline constexpr float CAM_START_POS[3] = {0.0f, 50.0f, 0.0f};
constexpr float CAM_START_YAW   = -90.0f;
constexpr float CAM_START_PITCH = -30.0f;
constexpr float CAM_FOV         = 70.0f;
constexpr float CAM_NEAR_CLIP   = 0.1f;
constexpr float CAM_FAR_CLIP    = 2000.0f;

constexpr float MOVE_SPEED  = 10.5f;
constexpr float SPRINT_MULT = 10.0f;
constexpr float LOOK_SPEED  = 3.15f;

// ==========================================
// ATMOSPHERE & LIGHTING
// ==========================================
inline constexpr float FOG_COLOR[3]         = {0.53f, 0.81f, 0.92f};
constexpr float        FOG_DENSITY          = 0.0012f;
constexpr float        HEIGHT_FOG_BASE_Y    = 25.0f;
constexpr float        HEIGHT_FOG_FALLOFF   = 0.015f;

inline constexpr float SUN_DIR[3]       = {0.6f, 1.0f, 0.4f};
inline constexpr float SUN_COLOR[3]     = {1.1f, 1.0f, 0.85f};
inline constexpr float AMBIENT_COLOR[3] = {0.35f, 0.45f, 0.6f};

// ==========================================
// TEXTURES & ASSETS
// ==========================================
constexpr float TEXTURE_SCALE      = 1.0f / 8.0f;
constexpr int   TEXTURE_RESOLUTION = 16; // 16x16 pixels

// Matches the BiomeType enum index (0 = Air, 1 = Ocean, etc.)
// Using string_view for compile-time string referencing (zero allocation)
inline constexpr std::array<std::string_view, 9> BLOCK_TEXTURES = {
    "",                         // 0 = Air
    "Textures/water.png",       // 1 = Ocean
    "Textures/sand.png",        // 2 = Beach
    "Textures/grass_top.png",   // 3 = Plains
    "Textures/dirt.png",        // 4 = Forest
    "Textures/sand.png",        // 5 = Desert
    "Textures/stone.png",       // 6 = Tundra
    "Textures/gravel.png",      // 7 = Mountain
    "Textures/snow.png"         // 8 = Snow
};

inline constexpr std::array<std::string_view, 6> SKYBOX_FACES = {
    "./Skybox/nx.png", "./Skybox/px.png", "./Skybox/py.png", 
    "./Skybox/ny.png", "./Skybox/nz.png", "./Skybox/pz.png"
};

// ==========================================
// CHUNK & MESHING CONSTANTS
// ==========================================
constexpr int   CHUNK_SIZE  = 32;          // Must be a power of 2
constexpr int   CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
constexpr int   CHUNK_SHIFT = 5;           // log2(CHUNK_SIZE)

// Greedy Meshing Directional Constants
inline constexpr float NORMS[6][3] = {
    { 1.0f,  0.0f,  0.0f}, {-1.0f,  0.0f,  0.0f}, // ±X
    { 0.0f,  1.0f,  0.0f}, { 0.0f, -1.0f,  0.0f}, // ±Y
    { 0.0f,  0.0f,  1.0f}, { 0.0f,  0.0f, -1.0f}  // ±Z
};

inline constexpr int AXIS_MAP[6][3] = {
    {0, 1, 2}, {0, 1, 2},  // ±X: face_axis=0, u=1, v=2
    {1, 0, 2}, {1, 0, 2},  // ±Y: face_axis=1, u=0, v=2
    {2, 0, 1}, {2, 0, 1}   // ±Z: face_axis=2, u=0, v=1
};

inline constexpr bool FLIP[6] = { false, true, true, false, false, true };
inline constexpr int  QUAD_TRIANGLES[6] = { 0, 1, 2, 0, 2, 3 };

// ==========================================
// TERRAIN GENERATION CONSTANTS
// ==========================================
constexpr float TEMPERATURE_OFFSET  = 16384.0f;
constexpr float MOISTURE_OFFSET     = 32768.0f;
constexpr float CONTINENTAL_OFFSET  = 8192.0f;
constexpr float CONTINENTAL_SCALE   = 0.8f;
constexpr float PEAK_OFFSET_X       = 1337.0f;
constexpr float PEAK_OFFSET_Z       = 7331.0f;
constexpr float BIOME_BLEND_SHARPNESS = 2.25f;

constexpr float TRANSITION_SCALE    = 0.025f;
constexpr float TRANSITION_AMP      = 2.5f;

constexpr int   UPSAMPLE_STRIDE     = 4;
constexpr int   UPSAMPLE_GRID_SIZE  = (CHUNK_SIZE / UPSAMPLE_STRIDE) + 1;
constexpr size_t UPSAMPLE_GRID_AREA = static_cast<size_t>(UPSAMPLE_GRID_SIZE) * static_cast<size_t>(UPSAMPLE_GRID_SIZE);

constexpr float LEGACY_HEIGHT_AMP   = 20.0f;
constexpr float LEGACY_HEIGHT_BASE  = 10.0f;

// Seed modifier for biome noise isolation
constexpr uint64_t BIOME_SEED_MODIFIER = 0xCAFEBABE87654321ULL;


constexpr float DEFAULT_BIOME_SCALE_MULTIPLIER  = 0.35f;
// Terrain Params Defaults (from terrain.hpp)
constexpr float DEFAULT_SURFACE_SCALE  = 0.005f;
constexpr float DEFAULT_BIOME_SCALE    = 0.0008f;
constexpr int   DEFAULT_SEA_LEVEL      = 8;
constexpr int   DEFAULT_BEACH_HEIGHT   = 2;
constexpr int   DEFAULT_SNOW_LEVEL     = 90;
constexpr int   DEFAULT_MOUNTAIN_LEVEL = 35;
constexpr int   DEFAULT_FBM_OCTAVES    = 4;
constexpr float DEFAULT_HEIGHT_AMP     = 28.0f;
constexpr float DEFAULT_HEIGHT_BASE    = 14.0f;

// Biome Profiles Structure
struct BiomeProfile {
    BiomeType biome;
    float target_temp;
    float target_moisture;
    float continental_bias;
    float min_height;
    float max_height;
    int   peak_layers;
    float peak_frequency;
    float peak_height;
    float ridge_weight;
    float detail_weight;
};

// Biome configurations (from terrain.cpp)
// 0: Plains, 1: Forest, 2: Desert, 3: Tundra, 4: Mountain
inline constexpr std::array<BiomeProfile, 5> BIOME_PROFILES{{
    {static_cast<BiomeType>(3),  0.05f,  0.05f, -0.15f, -20.0f, 23.0f, 0, 0.4f, 0.0f, 0.15f, 0.10f}, 
    {static_cast<BiomeType>(4),  0.10f,  0.50f,  0.00f,  20.0f, 33.0f, 2, 0.6f, 5.0f, 0.30f, 0.14f},
    {static_cast<BiomeType>(5),  0.70f, -0.55f, -0.05f,   0.0f, 27.0f, 1, 1.1f, 2.8f, 0.14f, 0.11f},
    {static_cast<BiomeType>(6), -0.70f, -0.10f,  0.05f,  30.0f, 38.0f, 2, 1.8f, 6.0f, 0.34f, 0.14f},
    {static_cast<BiomeType>(7),  0.10f, -0.15f,  0.80f,  98.0f, 98.0f, 1, 0.40f,300.0f, 0.62f, 0.18f}
}};

// ==========================================
// WORLD & RENDERING CONSTANTS
// ==========================================
constexpr int   DEFAULT_VIEW_DISTANCE = 8;
constexpr int   DEFAULT_THREAD_COUNT  = 4;

// Macro Chunks (HLOD)
constexpr int   MACRO_CHUNK_SIZE = 256;
constexpr int   MACRO_MESH_STEP  = 16;
constexpr int   MACRO_LOD_LEVEL  = 5;
constexpr int   MACRO_RADIUS     = 5;

// Frustum & Culling
constexpr float FOV_EXPANSION_FACTOR = 2.2f;
constexpr float FRUSTUM_CULL_BUFFER  = 1.5f;     // chunk_size * 1.5f

// Loading Limits
constexpr float MAX_VERTICAL_CHUNK_DELTA = 3.0f;
constexpr int   CHUNK_LOAD_Y_MIN = -2;
constexpr int   CHUNK_LOAD_Y_MAX = 2;

} // namespace config
} // namespace voxel