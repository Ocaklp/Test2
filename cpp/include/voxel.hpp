#pragma once
#include <array>
#include <cstdint>

namespace voxel {

// Biome/terrain types — used for WFC tile constraints and rendering colour.
enum class BiomeType : uint8_t {
    Air    = 0,
    Ocean  = 1,
    Beach  = 2,
    Plains = 3,
    Forest = 4,
    Desert = 5,
    Tundra = 6,
    Mountain = 7,
    Snow   = 8,
    COUNT  = 9
};

// A single binary voxel: occupied flag (1 bit) + biome (7 bits packed together).
// Kept at exactly 1 byte to allow efficient storage in large arrays.
struct Voxel {
    uint8_t data{0};  // bit 7 = solid, bits 6–0 = BiomeType

    Voxel() = default;
    Voxel(bool solid, BiomeType biome) {
        data = static_cast<uint8_t>(
            (static_cast<uint8_t>(solid) << 7) |
            (static_cast<uint8_t>(biome) & 0x7F));
    }

    bool     solid() const { return (data >> 7) & 1; }
    BiomeType biome() const { return static_cast<BiomeType>(data & 0x7F); }

    void set_solid(bool s) {
        if (s) data |= 0x80; else data &= 0x7F;
    }
    void set_biome(BiomeType b) {
        data = (data & 0x80) | (static_cast<uint8_t>(b) & 0x7F);
    }

    bool operator==(const Voxel& o) const { return data == o.data; }
    bool operator!=(const Voxel& o) const { return data != o.data; }
};
static_assert(sizeof(Voxel) == 1, "Voxel must be 1 byte");

// 3-component integer vector for voxel / chunk positions.
struct IVec3 {
    int x{0}, y{0}, z{0};
    IVec3() = default;
    IVec3(int x, int y, int z) : x(x), y(y), z(z) {}
    IVec3 operator+(const IVec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    IVec3 operator-(const IVec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    bool  operator==(const IVec3& o) const { return x == o.x && y == o.y && z == o.z; }
};

// 3-component float vector for world-space positions.
struct Vec3 {
    float x{0}, y{0}, z{0};
    Vec3() = default;
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    float length_sq() const { return dot(*this); }
};

// 4×4 column-major float matrix (for MVP transforms handed to OpenGL).
struct Mat4 {
    std::array<float, 16> m{};

    Mat4() { identity(); }
    void identity() {
        m.fill(0.0f);
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    static Mat4 translation(float tx, float ty, float tz) {
        Mat4 r;
        r.m[12] = tx; r.m[13] = ty; r.m[14] = tz;
        return r;
    }

    static Mat4 scale(float sx, float sy, float sz) {
        Mat4 r;
        r.m[0] = sx; r.m[5] = sy; r.m[10] = sz;
        return r;
    }

    Mat4 operator*(const Mat4& b) const {
        Mat4 c;
        c.m.fill(0.0f);
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                for (int k = 0; k < 4; ++k)
                    c.m[col * 4 + row] += m[k * 4 + row] * b.m[col * 4 + k];
        return c;
    }

    const float* data() const { return m.data(); }
};

}  // namespace voxel
