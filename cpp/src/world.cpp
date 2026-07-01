#include "config.hpp"
#include "world.hpp"
#include <iostream>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace voxel {

World::World(uint64_t seed, int view_distance, int thread_count)
    : seed_(seed),
      view_distance_(view_distance),
      terrain_gen_(seed) {

    // Start worker threads.
    for (int i = 0; i < thread_count; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

World::~World() {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void World::build_macro_mesh(int rx, int rz) {
    std::vector<float> mesh;
    
    constexpr int step = config::MACRO_MESH_STEP;
    constexpr int macro_size = config::MACRO_CHUNK_SIZE;
    float start_x = static_cast<float>(rx * macro_size);
    float start_z = static_cast<float>(rz * macro_size);

    for (int z = 0; z < macro_size; z += step) {
        for (int x = 0; x < macro_size; x += step) {
            float x0 = start_x + static_cast<float>(x);
            float z0 = start_z + static_cast<float>(z);
            float x1 = start_x + static_cast<float>(x + step);
            float z1 = start_z + static_cast<float>(z + step);

            float y00 = terrain_gen_.surface_height(x0, z0);
            float y10 = terrain_gen_.surface_height(x1, z0);
            float y01 = terrain_gen_.surface_height(x0, z1);
            float y11 = terrain_gen_.surface_height(x1, z1);

            BiomeType b00 = terrain_gen_.biome_at(x0, z0, y00);
            BiomeType b10 = terrain_gen_.biome_at(x1, z0, y10);
            BiomeType b01 = terrain_gen_.biome_at(x0, z1, y01);
            BiomeType b11 = terrain_gen_.biome_at(x1, z1, y11);

            // OPRAVA: Přidán 8. float (AO) do vertexu pro shodu se shaderem!
            auto push_vertex = [&](float vx, float vy, float vz, float nx, float ny, float nz, BiomeType biome) {
                mesh.push_back(vx);
                mesh.push_back(vy);
                mesh.push_back(vz);
                mesh.push_back(nx);
                mesh.push_back(ny);
                mesh.push_back(nz);
                mesh.push_back(static_cast<float>(biome)); 
                mesh.push_back(1.0f); // <-- aAO; // NEW: Očekáváno v renderer.py
            };

            // Trojúhelník 1
            float nx1 = -(y10 - y00) * step;
            float ny1 = static_cast<float>(step * step);
            float nz1 = -(y01 - y00) * step;
            float len1 = std::sqrt(nx1*nx1 + ny1*ny1 + nz1*nz1);
            if (len1 > 0.0001f) { nx1 /= len1; ny1 /= len1; nz1 /= len1; } else { nx1 = 0; ny1 = 1; nz1 = 0; }

            push_vertex(x0, y00, z0, nx1, ny1, nz1, b00);
            push_vertex(x0, y01, z1, nx1, ny1, nz1, b01);
            push_vertex(x1, y10, z0, nx1, ny1, nz1, b10);

            // Trojúhelník 2
            float nx2 = -(y11 - y01) * step;
            float ny2 = static_cast<float>(step * step);
            float nz2 = -(y11 - y10) * step;
            float len2 = std::sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2);
            if (len2 > 0.0001f) { nx2 /= len2; ny2 /= len2; nz2 /= len2; } else { nx2 = 0; ny2 = 1; nz2 = 0; }

            push_vertex(x1, y10, z0, nx2, ny2, nz2, b10);
            push_vertex(x0, y01, z1, nx2, ny2, nz2, b01);
            push_vertex(x1, y11, z1, nx2, ny2, nz2, b11);
        }
    }

    std::lock_guard<std::mutex> lock(macro_mutex_);
    
    voxel::RenderableMesh rm;
    rm.cx = rx;
    rm.cy = 0;
    rm.cz = rz;
    rm.lod_level = 5;
    rm.vertices = std::move(mesh);
    
    macro_updates_queue_.push_back(std::move(rm));
}

std::vector<RenderableMesh> World::pop_macro_updates() {
    std::lock_guard<std::mutex> lock(macro_mutex_);
    auto result = std::move(macro_updates_queue_);
    macro_updates_queue_.clear();
    return result;
}

std::vector<std::tuple<int, int>> World::get_visible_macros(
    float cam_x, float cam_y, float cam_z, 
    float dir_x, float dir_y, float dir_z, float fov) const 
{
    std::vector<std::tuple<int, int>> result;
    std::lock_guard<std::mutex> lock(macro_mutex_);
    
    // Simple radius check: If a macro chunk is outside the detailed render 
    // distance, but inside the HLOD render distance (e.g., 5000 blocks), draw it.
    for (const auto& [key, mesh] : macro_chunks_) {
        // You can add proper frustum culling here later, but for now, 
        // we just pass the coordinates to Python.
        result.push_back({key.rx, key.rz});
    }
    return result;
}
std::vector<RenderableMesh> World::pop_mesh_updates() {
    std::lock_guard<std::mutex> lock(updates_mutex_);
    // Move the data to the return value and clear the queue
    auto result = std::move(updates_queue_);
    updates_queue_.clear();
    return result;
}

// 2. Frustum Culling Coordinate Fetcher
std::vector<std::tuple<int, int, int>> World::get_visible_coordinates(
    float cam_x, float cam_y, float cam_z, 
    float dir_x, float dir_y, float dir_z, 
    float fov) const 
{
    std::vector<std::tuple<int, int, int>> result;
    std::lock_guard<std::mutex> lock(chunks_mutex_);

    float fov_rad = fov * (config::DEG_TO_RAD);
    float cos_limit = std::cos((fov_rad * 0.5f) * config::FOV_EXPANSION_FACTOR);
    const float chunk_size = config::CHUNK_SIZE; // Adjust to your Chunk::SIZE

    for (const auto& [key, entry] : chunks_) {
        // Skip unmeshed chunks
        if (entry.mesh.empty() || !entry.is_visible) continue;

        float center_x = (key.x + 0.5f) * chunk_size;
        float center_y = (key.y + 0.5f) * chunk_size;
        float center_z = (key.z + 0.5f) * chunk_size;

        float to_x = center_x - cam_x;
        float to_y = center_y - cam_y;
        float to_z = center_z - cam_z;
        float dist = std::sqrt(to_x*to_x + to_y*to_y + to_z*to_z);

        if (dist > 1e-6) {
            float dot = (to_x / dist) * dir_x + 
                        (to_y / dist) * dir_y + 
                        (to_z / dist) * dir_z;
            
            if (dot < cos_limit && dist > chunk_size * config::FRUSTUM_CULL_BUFFER) {
                continue; // Behind camera / outside FOV
            }
        }

        // Only returning three integers! Lightning fast.
        result.push_back({key.x, key.y, key.z});
    }

    return result;
}

std::vector<RenderableMesh> World::get_visible_meshes(
    float cam_x, float cam_y, float cam_z, 
    float dir_x, float dir_y, float dir_z, 
    float fov) const 
{
    std::vector<RenderableMesh> result;
    std::lock_guard<std::mutex> lock(chunks_mutex_);

    // Convert FOV to radians and apply the 2.2 expansion factor from your render.py
    float fov_rad = fov * (config::DEG_TO_RAD);
    float cos_limit = std::cos((fov_rad * 0.5f) * config::FOV_EXPANSION_FACTOR);
    
    // Assuming Chunk::SIZE is 32 based on your python code
    const float chunk_size = config::CHUNK_SIZE; 

    for (const auto& [key, entry] : chunks_) {
        // Skip chunks that haven't been meshed yet
        if (entry.mesh.empty()) continue;

        // Find the center of this chunk in world space
        float center_x = (key.x + 0.5f) * chunk_size;
        float center_y = (key.y + 0.5f) * chunk_size;
        float center_z = (key.z + 0.5f) * chunk_size;

        // Vector from camera to chunk center
        float to_x = center_x - cam_x;
        float to_y = center_y - cam_y;
        float to_z = center_z - cam_z;

        float dist = std::sqrt(to_x*to_x + to_y*to_y + to_z*to_z);

        // Frustum Culling
        if (dist > 1e-6) {
            float dot = (to_x / dist) * dir_x + 
                        (to_y / dist) * dir_y + 
                        (to_z / dist) * dir_z;
            
            // If it's outside the FOV and not immediately surrounding the player, cull it
            if (dot < cos_limit && dist > chunk_size * config::FRUSTUM_CULL_BUFFER) {
                continue; 
            }
        }

        // If we survived culling, add it to the render batch
        result.push_back({key.x, key.y, key.z, entry.lod, entry.mesh});
    }

    return result;
}

void World::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [this] {
                return stopping_ || !task_queue_.empty();
            });
            if (stopping_ && task_queue_.empty()) return;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }
        task();
    }
}

void World::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        task_queue_.push(std::move(task));
    }
    queue_cv_.notify_one();
}

ChunkKey World::world_to_chunk(int wx, int wy, int wz) const {
    // Floor division to handle negative coordinates correctly.
    auto floordiv = [](int a, int b) {
        return a / b - (a % b != 0 && (a ^ b) < 0);
    };
    return {floordiv(wx, Chunk::SIZE),
            floordiv(wy, Chunk::SIZE),
            floordiv(wz, Chunk::SIZE)};
}

IVec3 World::chunk_local(int wx, int wy, int wz) const {
    auto mod = [](int a, int b) { return ((a % b) + b) % b; };
    return {mod(wx, Chunk::SIZE), mod(wy, Chunk::SIZE), mod(wz, Chunk::SIZE)};
}

void World::update(const Vec3& viewer_pos) {
    viewer_pos_ = viewer_pos;
    schedule_missing_chunks(viewer_pos);
    unload_distant_chunks(viewer_pos);
    schedule_missing_macros(viewer_pos);
}

void World::schedule_missing_chunks(const Vec3& vp) {
    int vcx = static_cast<int>(std::floor(vp.x / Chunk::SIZE));
    int vcy = static_cast<int>(std::floor(vp.y / Chunk::SIZE));
    int vcz = static_cast<int>(std::floor(vp.z / Chunk::SIZE));

    struct MissingChunk {
        ChunkKey key;
        int dist2;
    };
    std::vector<MissingChunk> missing;

    int vd = view_distance_;
    int vd_squared = vd * vd;
    {
        std::lock_guard<std::mutex> lk(chunks_mutex_);
        for (int dx = -vd; dx <= vd; ++dx) {
            for (int dz = -vd; dz <= vd; ++dz) {
                if (dx * dx + dz * dz > vd_squared) {
                    continue;  // circular horizontal load radius
                }
                for (int dy = config::CHUNK_LOAD_Y_MIN; dy <= config::CHUNK_LOAD_Y_MAX; ++dy) {  // limited vertical range
                    ChunkKey key{vcx + dx, vcy + dy, vcz + dz};
                    auto it = chunks_.find(key);
                    if (it == chunks_.end()) {
                        // Reserve slot so we don't double-schedule.
                        chunks_[key] = ChunkEntry{};
                        missing.push_back({key, dx * dx + dy * dy + dz * dz});
                    } else {
                        it->second.is_visible = true;
                    }
                }
            }
        }
    }

    std::sort(missing.begin(), missing.end(),
              [](const MissingChunk& a, const MissingChunk& b) {
                  return a.dist2 < b.dist2;
              });

    for (const auto& item : missing) {
        ChunkKey key = item.key;
        enqueue([this, key] { load_chunk(key); });
    }
}

void World::unload_distant_chunks(const Vec3& vp) {
    constexpr float kMaxVerticalChunkDelta = config::MAX_VERTICAL_CHUNK_DELTA;
    float vcx = vp.x / Chunk::SIZE;
    float vcy = vp.y / Chunk::SIZE;
    float vcz = vp.z / Chunk::SIZE;
    float limit = static_cast<float>(view_distance_ + 1);

    std::lock_guard<std::mutex> lk(chunks_mutex_);
    for (auto& [key, entry] : chunks_) {
        float dx = static_cast<float>(key.x) - vcx;
        float dy = static_cast<float>(key.y) - vcy;
        float dz = static_cast<float>(key.z) - vcz;
        entry.is_visible = !(std::sqrt(dx * dx + dz * dz) > limit ||
                             std::abs(dy) > kMaxVerticalChunkDelta);
    }
}

void World::load_chunk(ChunkKey key) {
    auto chunk = std::make_shared<Chunk>(IVec3{key.x, key.y, key.z});
    terrain_gen_.generate(*chunk);

    // Skip chunks with no visible faces:
    //  - entirely air  (solid_count == 0): above the terrain surface
    //  - entirely solid (solid_count == VOLUME): buried underground
    // In both cases the greedy mesher would only emit boundary-plane quads that
    // are completely covered by adjacent solid chunks, producing invisible geometry.
    // Leaving the reserved slot (chunk == nullptr) prevents re-scheduling.
    bool skip_mesh = false;
    {
        int solid_count = chunk->solid_count();
        skip_mesh = (solid_count == 0 || solid_count == Chunk::VOLUME);
    }

    // Determine LOD based on viewer position.
    Vec3 viewer = viewer_pos_;
    int  lod = lod_level_for({key.x, key.y, key.z},
                              {viewer.x / Chunk::SIZE,
                               viewer.y / Chunk::SIZE,
                               viewer.z / Chunk::SIZE});
    std::vector<ChunkKey> remesh_targets;
    {
        std::lock_guard<std::mutex> lk(chunks_mutex_);
        auto it = chunks_.find(key);
        if (it != chunks_.end()) {
            // Preserve chunk if it was already populated meanwhile.
            if (it->second.chunk) {
                return;
            }
            it->second.chunk      = chunk;
            it->second.mesh       = {};
            it->second.lod        = lod;
            it->second.mesh_dirty = !skip_mesh;
            if (!skip_mesh) {
                remesh_targets.push_back(key);
                static constexpr std::array<ChunkKey, 6> kNeighbors{{
                    { 1, 0, 0}, {-1, 0, 0},
                    { 0, 1, 0}, { 0,-1, 0},
                    { 0, 0, 1}, { 0, 0,-1}
                }};
                for (const auto& off : kNeighbors) {
                    ChunkKey nk{key.x + off.x, key.y + off.y, key.z + off.z};
                    auto nit = chunks_.find(nk);
                    if (nit != chunks_.end() && nit->second.chunk) {
                        remesh_targets.push_back(nk);
                    }
                }
            }
        }
    }

    if (!remesh_targets.empty()) {
        enqueue([this, remesh_targets] {
            for (const auto& target : remesh_targets) {
                int target_lod = 0;
                {
                    std::lock_guard<std::mutex> lk(chunks_mutex_);
                    auto it = chunks_.find(target);
                    if (it == chunks_.end() || !it->second.chunk) {
                        continue;
                    }
                    target_lod = it->second.lod;
                }
                build_mesh(target, target_lod);
            }
        });
    }
}

void World::build_mesh(ChunkKey key, int lod) {
    std::shared_ptr<Chunk> chunk;
    std::array<std::array<uint8_t, Chunk::SIZE * Chunk::SIZE>, 6> neighbor_masks{};
    for (auto& mask : neighbor_masks) {
        mask.fill(0);
    }

    auto fill_neighbor_mask = [&](const ChunkKey& nk, int dir) {
        auto it = chunks_.find(nk);
        if (it == chunks_.end() || !it->second.chunk) {
            return;
        }
        const auto& n = *it->second.chunk;
        for (int u = 0; u < Chunk::SIZE; ++u) {
            for (int v = 0; v < Chunk::SIZE; ++v) {
                bool solid = false;
                switch (dir) {
                    case 0: solid = n.get(0, u, v).solid(); break;                  // +X
                    case 1: solid = n.get(Chunk::SIZE - 1, u, v).solid(); break;    // -X
                    case 2: solid = n.get(u, 0, v).solid(); break;                  // +Y
                    case 3: solid = n.get(u, Chunk::SIZE - 1, v).solid(); break;    // -Y
                    case 4: solid = n.get(u, v, 0).solid(); break;                  // +Z
                    case 5: solid = n.get(u, v, Chunk::SIZE - 1).solid(); break;    // -Z
                }
                neighbor_masks[dir][u + Chunk::SIZE * v] = solid ? 1 : 0;
            }
        }
    };

    {
        std::lock_guard<std::mutex> lk(chunks_mutex_);
        auto it = chunks_.find(key);
        if (it == chunks_.end() || !it->second.chunk) return;
        chunk = it->second.chunk;
        fill_neighbor_mask({key.x + 1, key.y, key.z}, 0);
        fill_neighbor_mask({key.x - 1, key.y, key.z}, 1);
        fill_neighbor_mask({key.x, key.y + 1, key.z}, 2);
        fill_neighbor_mask({key.x, key.y - 1, key.z}, 3);
        fill_neighbor_mask({key.x, key.y, key.z + 1}, 4);
        fill_neighbor_mask({key.x, key.y, key.z - 1}, 5);
    }

    IVec3 origin = chunk->world_origin();
    auto neighbor_solid_query = [&neighbor_masks, origin](int wx, int wy, int wz) {
        int lx = wx - origin.x;
        int ly = wy - origin.y;
        int lz = wz - origin.z;

        if (lx == Chunk::SIZE && ly >= 0 && ly < Chunk::SIZE && lz >= 0 && lz < Chunk::SIZE) {
            return neighbor_masks[0][ly + Chunk::SIZE * lz] != 0;
        }
        if (lx == -1 && ly >= 0 && ly < Chunk::SIZE && lz >= 0 && lz < Chunk::SIZE) {
            return neighbor_masks[1][ly + Chunk::SIZE * lz] != 0;
        }
        if (ly == Chunk::SIZE && lx >= 0 && lx < Chunk::SIZE && lz >= 0 && lz < Chunk::SIZE) {
            return neighbor_masks[2][lx + Chunk::SIZE * lz] != 0;
        }
        if (ly == -1 && lx >= 0 && lx < Chunk::SIZE && lz >= 0 && lz < Chunk::SIZE) {
            return neighbor_masks[3][lx + Chunk::SIZE * lz] != 0;
        }
        if (lz == Chunk::SIZE && lx >= 0 && lx < Chunk::SIZE && ly >= 0 && ly < Chunk::SIZE) {
            return neighbor_masks[4][lx + Chunk::SIZE * ly] != 0;
        }
        if (lz == -1 && lx >= 0 && lx < Chunk::SIZE && ly >= 0 && ly < Chunk::SIZE) {
            return neighbor_masks[5][lx + Chunk::SIZE * ly] != 0;
        }
        return false;
    };
    std::vector<float> mesh = chunk->build_mesh(neighbor_solid_query);

        // 1. Save it in the main chunk map so C++ knows about it
        {
            std::lock_guard<std::mutex> lk(chunks_mutex_);
            auto it = chunks_.find(key);
            if (it != chunks_.end()) {
                it->second.mesh       = mesh; // We removed std::move here so we can still use it below!
                it->second.lod        = lod;
                it->second.mesh_dirty = false;
            }
        }

        // 2. THIS IS THE MISSING PIECE! Push it to the Delta Queue for Python!
        {
            std::lock_guard<std::mutex> q_lock(updates_mutex_);
            // Now we can safely std::move the mesh into the queue
            updates_queue_.push_back({key.x, key.y, key.z, lod, std::move(mesh)});
        }
    }


std::vector<float> World::get_mesh(int cx, int cy, int cz) {
    std::lock_guard<std::mutex> lk(chunks_mutex_);
    auto it = chunks_.find({cx, cy, cz});
    if (it == chunks_.end()) return {};
    return it->second.mesh;
}

std::vector<int> World::loaded_chunk_coords() const {
    std::lock_guard<std::mutex> lk(chunks_mutex_);
    std::vector<int> result;
    result.reserve(chunks_.size() * 3);
    for (const auto& [key, entry] : chunks_) {
        if (entry.chunk && entry.is_visible) {
            result.push_back(key.x);
            result.push_back(key.y);
            result.push_back(key.z);
        }
    }
    return result;
}

void World::set_voxel(int wx, int wy, int wz, Voxel v) {
    ChunkKey key = world_to_chunk(wx, wy, wz);
    IVec3    local = chunk_local(wx, wy, wz);

    std::shared_ptr<Chunk> chunk;
    {
        std::lock_guard<std::mutex> lk(chunks_mutex_);
        auto it = chunks_.find(key);
        if (it != chunks_.end() && it->second.chunk) {
            chunk = it->second.chunk;
        }
    }

    if (!chunk) {
        auto generated = std::make_shared<Chunk>(IVec3{key.x, key.y, key.z});
        terrain_gen_.generate(*generated);

        std::lock_guard<std::mutex> lk(chunks_mutex_);
        auto& entry = chunks_[key];
        if (!entry.chunk) {
            entry.chunk = generated;
            entry.mesh.clear();
            entry.lod = 0;
            entry.mesh_dirty = true;
        }
        entry.is_visible = true;
        chunk = entry.chunk;
    }
    chunk->set(local, v);

    // Schedule remesh.
    enqueue([this, key] { build_mesh(key, 0); });
}

Voxel World::get_voxel(int wx, int wy, int wz) const {
    ChunkKey key = world_to_chunk(wx, wy, wz);
    IVec3    local = chunk_local(wx, wy, wz);

    std::lock_guard<std::mutex> lk(chunks_mutex_);
    auto it = chunks_.find(key);
    if (it == chunks_.end() || !it->second.chunk) return Voxel{};
    return it->second.chunk->get(local);
}

void World::schedule_missing_macros(const Vec3& vp) {
    // Velikost Macro-Chunku je 256 bloků
    int vrx = static_cast<int>(std::floor(vp.x / static_cast<float>(config::MACRO_CHUNK_SIZE)));
    int vrz = static_cast<int>(std::floor(vp.z / static_cast<float>(config::MACRO_CHUNK_SIZE)));

    // Poloměr 5 znamená 5 * 256 = 1280 bloků na každou stranu!
    int macro_radius = config::MACRO_RADIUS; 

    struct MissingMacro { MacroKey key; int dist2; };
    std::vector<MissingMacro> missing;

    {
        std::lock_guard<std::mutex> lk(macro_mutex_);
        for (int dx = -macro_radius; dx <= macro_radius; ++dx) {
            for (int dz = -macro_radius; dz <= macro_radius; ++dz) {
                MacroKey key{vrx + dx, vrz + dz};
                if (macro_chunks_.find(key) == macro_chunks_.end()) {
                    // Rezervujeme místo, aby se negeneroval dvakrát
                    macro_chunks_[key] = {}; 
                    missing.push_back({key, dx*dx + dz*dz});
                }
            }
        }
    }

    // Seřadíme podle vzdálenosti od hráče
    std::sort(missing.begin(), missing.end(),
              [](const MissingMacro& a, const MissingMacro& b) {
                  return a.dist2 < b.dist2;
              });

    // Odešleme do fronty pracovních vláken
    for (const auto& item : missing) {
        int rx = item.key.rx;
        int rz = item.key.rz;
        enqueue([this, rx, rz] { build_macro_mesh(rx, rz); });
    }
}
}  // namespace voxel