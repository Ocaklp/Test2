#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "noise.hpp"
#include "voxel.hpp"
#include "octree.hpp"
#include "chunk.hpp"
#include "terrain.hpp"
#include "wfc.hpp"
#include "lod.hpp"
#include "world.hpp"

namespace py = pybind11;
using namespace voxel;

PYBIND11_MODULE(voxel_core, m) {
    m.doc() = "Procedural voxel world engine — C++ core exposed via Pybind11";
    // Bind the RenderableMesh struct
    py::class_<RenderableMesh>(m, "RenderableMesh")
        .def_readonly("cx", &RenderableMesh::cx)
        .def_readonly("cy", &RenderableMesh::cy)
        .def_readonly("cz", &RenderableMesh::cz)
        .def_readonly("lod_level", &RenderableMesh::lod_level)
        // Expose vertices as a property so Python can read the array
        .def_property_readonly("vertices", [](const RenderableMesh& rm) {
            return rm.vertices; 
        });

    // ── BiomeType ──────────────────────────────────────────────────────────
    py::enum_<BiomeType>(m, "BiomeType")
        .value("Air",      BiomeType::Air)
        .value("Ocean",    BiomeType::Ocean)
        .value("Beach",    BiomeType::Beach)
        .value("Plains",   BiomeType::Plains)
        .value("Forest",   BiomeType::Forest)
        .value("Desert",   BiomeType::Desert)
        .value("Tundra",   BiomeType::Tundra)
        .value("Mountain", BiomeType::Mountain)
        .value("Snow",     BiomeType::Snow)
        .export_values();

    // ── Voxel ──────────────────────────────────────────────────────────────
    py::class_<Voxel>(m, "Voxel")
        .def(py::init<>())
        .def(py::init<bool, BiomeType>(), py::arg("solid"), py::arg("biome"))
        .def_property("solid",
            &Voxel::solid,
            [](Voxel& v, bool s){ v.set_solid(s); })
        .def_property("biome",
            &Voxel::biome,
            [](Voxel& v, BiomeType b){ v.set_biome(b); })
        .def("__repr__", [](const Voxel& v){
            return "<Voxel solid=" + std::string(v.solid()?"true":"false") +
                   " biome=" + std::to_string(static_cast<int>(v.biome())) + ">";
        });

    // ── IVec3 / Vec3 ──────────────────────────────────────────────────────
    py::class_<IVec3>(m, "IVec3")
        .def(py::init<>())
        .def(py::init<int,int,int>())
        .def_readwrite("x", &IVec3::x)
        .def_readwrite("y", &IVec3::y)
        .def_readwrite("z", &IVec3::z)
        .def("__repr__", [](const IVec3& v){
            return "IVec3(" + std::to_string(v.x) + "," +
                   std::to_string(v.y) + "," + std::to_string(v.z) + ")";
        });

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<float,float,float>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("__repr__", [](const Vec3& v){
            return "Vec3(" + std::to_string(v.x) + "," +
                   std::to_string(v.y) + "," + std::to_string(v.z) + ")";
        });

    // ── Mat4 ───────────────────────────────────────────────────────────────
    py::class_<Mat4>(m, "Mat4")
        .def(py::init<>())
        .def("identity", &Mat4::identity)
        .def_static("translation", &Mat4::translation)
        .def_static("scale", &Mat4::scale)
        .def("__mul__", &Mat4::operator*)
        .def("as_list", [](const Mat4& mat){
            return std::vector<float>(mat.data(), mat.data() + 16);
        })
        .def("as_numpy", [](const Mat4& mat) -> py::array_t<float> {
            py::array_t<float> arr({4, 4});
            auto buf = arr.mutable_unchecked<2>();
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    buf(r, c) = mat.data()[c * 4 + r];
            return arr;
        });

    // ── Noise ──────────────────────────────────────────────────────────────
    py::class_<Noise>(m, "Noise")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("noise2",  &Noise::noise2)
        .def("noise3",  &Noise::noise3)
        .def("fbm2",    &Noise::fbm2,
             py::arg("x"), py::arg("y"), py::arg("octaves"),
             py::arg("persistence") = 0.5f, py::arg("lacunarity") = 2.0f)
        .def("fbm3",    &Noise::fbm3,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("octaves"),
             py::arg("persistence") = 0.5f, py::arg("lacunarity") = 2.0f)
        .def("ridged2", &Noise::ridged2,
             py::arg("x"), py::arg("y"), py::arg("octaves"),
             py::arg("persistence") = 0.5f, py::arg("lacunarity") = 2.0f)
        .def_property_readonly("seed", &Noise::seed);

    // ── Octree ─────────────────────────────────────────────────────────────
    py::class_<OctreeNode>(m, "OctreeNode")
        .def_readonly("min_corner",     &OctreeNode::min_corner)
        .def_readonly("size",           &OctreeNode::size)
        .def_readonly("representative", &OctreeNode::representative)
        .def_readonly("is_leaf",        &OctreeNode::is_leaf)
        .def_readonly("all_same",       &OctreeNode::all_same);

    py::class_<Octree>(m, "Octree")
        .def(py::init<int, IVec3>(),
             py::arg("root_size"), py::arg("origin") = IVec3{})
        .def("set",          &Octree::set)
        .def("get",          &Octree::get)
        .def("compress",     &Octree::compress)
        .def("root_size",    &Octree::root_size)
        .def("origin",       &Octree::origin)
        .def("root", [](const Octree& t) -> const OctreeNode& {
            return t.root();
        }, py::return_value_policy::reference_internal)
        .def("traverse_leaves", [](const Octree& tree) {
            std::vector<py::dict> nodes;
            tree.traverse_leaves([&](const OctreeNode& n){
                py::dict d;
                d["x"]     = n.min_corner.x;
                d["y"]     = n.min_corner.y;
                d["z"]     = n.min_corner.z;
                d["size"]  = n.size;
                d["solid"] = n.representative.solid();
                d["biome"] = static_cast<int>(n.representative.biome());
                nodes.push_back(d);
            });
            return nodes;
        });

    // ── Chunk ──────────────────────────────────────────────────────────────
    py::class_<Chunk>(m, "Chunk")
        .def(py::init<>())
        .def(py::init<IVec3>())
        .def("get",         [](const Chunk& c, int x, int y, int z){
            if (!Chunk::in_bounds(x, y, z)) {
                throw py::index_error("Chunk coordinates out of bounds");
            }
            return c.get(x, y, z);
        })
        .def("set",         [](Chunk& c, int x, int y, int z, Voxel v){
            if (!Chunk::in_bounds(x, y, z)) {
                throw py::index_error("Chunk coordinates out of bounds");
            }
            c.set(x, y, z, v);
        })
        .def("fill",        &Chunk::fill)
        .def("dirty",       &Chunk::dirty)
        .def("clear_dirty", &Chunk::clear_dirty)
        .def("chunk_coord", &Chunk::chunk_coord)
        .def("world_origin",&Chunk::world_origin)
        .def("solid_count", &Chunk::solid_count)
        .def("build_mesh",  [](const Chunk& c) { return c.build_mesh(); })
        .def("SIZE",        [](const Chunk&){ return Chunk::SIZE; })
        .def_property_readonly_static("CHUNK_SIZE",
            [](py::object){ return Chunk::SIZE; });

    // ── TerrainParams ──────────────────────────────────────────────────────
    py::class_<TerrainParams>(m, "TerrainParams")
        .def(py::init<>())
        .def_readwrite("surface_scale",  &TerrainParams::surface_scale)
        .def_readwrite("biome_scale",    &TerrainParams::biome_scale)
        .def_readwrite("sea_level",      &TerrainParams::sea_level)
        .def_readwrite("beach_height",   &TerrainParams::beach_height)
        .def_readwrite("snow_level",     &TerrainParams::snow_level)
        .def_readwrite("mountain_level", &TerrainParams::mountain_level)
        .def_readwrite("fbm_octaves",    &TerrainParams::fbm_octaves)
        .def_readwrite("height_amp",     &TerrainParams::height_amp)
        .def_readwrite("height_base",    &TerrainParams::height_base);

    // ── TerrainGenerator ───────────────────────────────────────────────────
    py::class_<TerrainGenerator>(m, "TerrainGenerator")
        .def(py::init<uint64_t, const TerrainParams&>(),
             py::arg("seed") = 0,
             py::arg("params") = TerrainParams{})
        .def("generate",       &TerrainGenerator::generate)
        .def("surface_height", &TerrainGenerator::surface_height)
        .def("biome_at",       &TerrainGenerator::biome_at)
        .def_property_readonly("seed",   &TerrainGenerator::seed)
        .def_property_readonly("params", &TerrainGenerator::params);

    // ── WFCSolver ──────────────────────────────────────────────────────────
    py::class_<WFCSolver>(m, "WFCSolver")
        // Primary constructor: width, height, optional seed (uses default adjacency).
        .def(py::init([](int w, int h, uint64_t seed) {
                 return std::make_unique<WFCSolver>(w, h, default_adjacency(), seed);
             }),
             py::arg("width"), py::arg("height"), py::arg("seed") = 42)
        .def("solve",          &WFCSolver::solve, py::arg("max_restarts") = 10)
        .def("at",             &WFCSolver::at)
        .def("pin",            &WFCSolver::pin)
        .def("apply_to_chunk", &WFCSolver::apply_to_chunk,
             py::arg("chunk"), py::arg("chunk_cx"), py::arg("chunk_cz"),
             py::arg("wfc_origin_cx") = 0, py::arg("wfc_origin_cz") = 0)
        .def_property_readonly("width",  &WFCSolver::width)
        .def_property_readonly("height", &WFCSolver::height);

    // ── LOD helpers ────────────────────────────────────────────────────────
    m.def("lod_level_for", &lod_level_for,
          py::arg("chunk_coord"), py::arg("viewer_chunk_pos"));
    m.def("build_lod_mesh", &build_lod_mesh,
          py::arg("octree"), py::arg("lod_level"));
    m.def("chunk_to_octree", &chunk_to_octree, py::arg("chunk"));

    // ── World ──────────────────────────────────────────────────────────────
    py::class_<World>(m, "World")
        .def(py::init<uint64_t, int, int>(),
             py::arg("seed")          = 0,
             py::arg("view_distance") = 8,
             py::arg("thread_count")  = 4)
        .def("get_visible_meshes", &World::get_visible_meshes,
             "Returns a frustum-culled list of meshes ready for OpenGL.",
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("fx"), py::arg("fy"), py::arg("fz"),
             py::arg("fov"))
        .def("pop_mesh_updates", &World::pop_mesh_updates,
             "Returns a list of meshes that have been built since the last call.")
        .def("get_visible_coordinates", &World::get_visible_coordinates,
         "Returns a lightweight list of (x,y,z) coordinates visible to the camera.",
         py::arg("cam_x"), py::arg("cam_y"), py::arg("cam_z"),
         py::arg("dir_x"), py::arg("dir_y"), py::arg("dir_z"),
         py::arg("fov"))
         .def("pop_macro_updates", &World::pop_macro_updates)
        .def("get_visible_macros", &World::get_visible_macros,
         py::arg("cam_x"), py::arg("cam_y"), py::arg("cam_z"),
         py::arg("dir_x"), py::arg("dir_y"), py::arg("dir_z"), py::arg("fov"))
        .def("update",               &World::update)
        .def("get_mesh",             &World::get_mesh)
        .def("loaded_chunk_coords",  &World::loaded_chunk_coords)
        .def("set_voxel",            &World::set_voxel)
        .def("get_voxel",            &World::get_voxel)
        .def_property_readonly("seed",          &World::seed)
        .def_property_readonly("view_distance", &World::view_distance);
}
