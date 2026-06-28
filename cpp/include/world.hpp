#pragma once
#include "chunk.hpp"
#include "terrain.hpp"
#include "lod.hpp"
#include "wfc.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace voxel {

// Simple key type for chunk map.
struct ChunkKey {
    int x, y, z;
    bool operator==(const ChunkKey& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

struct ChunkKeyHash {
    size_t operator()(const ChunkKey& k) const {
        size_t h = static_cast<size_t>(k.x);
        h ^= static_cast<size_t>(k.y) * 2654435761ULL + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.z) * 2654435761ULL + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ── Macro-Chunk Structures ──
struct MacroKey {
    int rx, rz; // Region X and Z
    bool operator==(const MacroKey& o) const { return rx == o.rx && rz == o.rz; }
};

struct MacroKeyHash {
    size_t operator()(const MacroKey& k) const {
        return std::hash<int>()(k.rx) ^ (std::hash<int>()(k.rz) << 1);
    }
};

// Per-chunk runtime state.
struct ChunkEntry {
    std::shared_ptr<Chunk>  chunk;
    std::vector<float>      mesh;   // latest built mesh for current LOD
    int                     lod{0};
    bool                    mesh_dirty{true};
    bool                    is_visible{true};
};

// ──────────────────────────────────────────────────────────────────────────────
// Async World Manager
//
// Manages a streaming voxel world around a viewer position.
// Chunk loading / mesh building happen on a background thread pool.
// Python (or C++ callers) query ready meshes from the main thread.
// ──────────────────────────────────────────────────────────────────────────────
struct RenderableMesh {
    int cx;
    int cy;
    int cz;
    int lod_level;
    std::vector<float> vertices;
};

class World {
public:
    explicit World(uint64_t seed = 0,
                   int view_distance = 8,
                   int thread_count = 4);
    ~World();

    // Update the viewer position (chunk-space) and trigger async loads/unloads.
    void update(const Vec3& viewer_pos);

    std::vector<RenderableMesh> pop_mesh_updates();
    std::vector<std::tuple<int, int, int>> get_visible_coordinates(float cx, float cy, float cz, float fx, float fy, float fz, float fov) const;
    

    // Retrieve the mesh for a chunk (empty if not yet loaded).
    std::vector<float> get_mesh(int cx, int cy, int cz);

    // Get list of visible loaded chunk coordinates [x,y,z, x,y,z, ...].
    std::vector<int> loaded_chunk_coords() const;

    // Directly set a voxel in the world (marks chunk as dirty for remesh).
    void set_voxel(int wx, int wy, int wz, Voxel v);

    // Read a voxel from the world.  Returns Air if chunk not loaded.
    Voxel get_voxel(int wx, int wy, int wz) const;

    std::vector<RenderableMesh> get_visible_meshes(
        float cam_x, float cam_y, float cam_z, 
        float dir_x, float dir_y, float dir_z, 
        float fov
    ) const;

    std::vector<RenderableMesh> pop_macro_updates();
    std::vector<std::tuple<int, int>> get_visible_macros(
        float cam_x, float cam_y, float cam_z, 
        float dir_x, float dir_y, float dir_z, float fov
    ) const;

    int    view_distance() const { return view_distance_; }
    uint64_t seed()        const { return seed_; }

private:
    uint64_t          seed_;
    int               view_distance_;
    TerrainGenerator  terrain_gen_;

    // Chunk storage
    mutable std::mutex                                              chunks_mutex_;
    std::unordered_map<ChunkKey, ChunkEntry, ChunkKeyHash>          chunks_;

    // Thread pool
    std::vector<std::thread>   workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex                 queue_mutex_;
    std::condition_variable    queue_cv_;
    std::atomic<bool>          stopping_{false};

    Vec3 viewer_pos_{0, 0, 0};

    void worker_loop();
    void enqueue(std::function<void()> task);

    void load_chunk(ChunkKey key);
    void build_mesh(ChunkKey key, int lod);
    void schedule_missing_chunks(const Vec3& vp);
    void schedule_missing_macros(const Vec3& vp);
    void unload_distant_chunks(const Vec3& vp);

    // Add inside the World class private section:
    std::mutex updates_mutex_;
    std::vector<RenderableMesh> updates_queue_;

    ChunkKey world_to_chunk(int wx, int wy, int wz) const;
    IVec3    chunk_local(int wx, int wy, int wz) const;

    std::unordered_map<MacroKey, std::vector<float>, MacroKeyHash> macro_chunks_;
    std::vector<RenderableMesh> macro_updates_queue_;
    mutable std::mutex macro_mutex_;

    void build_macro_mesh(int rx, int rz);
};

}  // namespace voxel
