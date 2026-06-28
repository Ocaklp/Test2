# BcTest – Procedural Voxel World Skeleton

A procedural voxel world skeleton implementing:

* **C++ core** (noise, WFC, binary voxels, octree, LOD, async world) compiled as a
  shared library and exposed to Python via **Pybind11**.
* **Python high-level layer**: PyOpenGL renderer, camera, and world driver.
* **Async + multithreaded** chunk loading with a C++ thread pool.

---

## Architecture

```
BcTest/
├── CMakeLists.txt            # Build system
├── requirements.txt          # Python dependencies
├── cpp/
│   ├── include/
│   │   ├── noise.hpp         # Simplex noise (2-D/3-D, FBM, ridged)
│   │   ├── voxel.hpp         # Binary voxel, BiomeType enum, IVec3, Vec3, Mat4
│   │   ├── octree.hpp        # Sparse octree for spatial acceleration
│   │   ├── chunk.hpp         # Fixed-size 32^3 chunk + greedy meshing
│   │   ├── terrain.hpp       # Noise-based terrain generator
│   │   ├── wfc.hpp           # Wave Function Collapse for biome transitions
│   │   ├── lod.hpp           # LOD system (4 levels) + octree mesh builder
│   │   └── world.hpp         # Async world manager (thread pool)
│   ├── src/                  # Implementation (.cpp files)
│   └── bindings/
│       └── bindings.cpp      # Pybind11 module definition
├── python/
│   ├── __init__.py
│   ├── camera.py             # First-person fly camera
│   ├── renderer.py           # PyOpenGL 3.3 Core renderer
│   └── main.py               # Entry point (--headless flag for CI)
└── tests/
    ├── cpp/
    │   └── test_main.cpp     # 34 C++ unit tests (no external framework)
    └── python/
        ├── test_voxel_engine.py  # Python tests via pytest
        ├── test_camera.py        # Camera tests
        └── test_renderer.py      # Renderer tests
```

---

## Building

### Prerequisites

* CMake >= 3.18
* C++17 compiler (GCC 11+, Clang 13+, MSVC 2019+)
* Python 3.10+ with development headers
* `pybind11` Python package (`pip install pybind11`)

```bash
pip install -r requirements.txt

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
```

This builds:

* `libvoxel_core.a` – static C++ library
* `voxel_engine*.so` – Pybind11 Python extension (copied to `python/`)
* `test_voxel_engine` – C++ test binary

---

## Running

### Interactive (requires PyOpenGL + pygame)

```bash
python python/main.py --seed 42 --view-distance 6
```

**Controls:** `W/A/S/D` move, `Q/E` up/down, mouse look, `Shift` sprint, `F4` wireframe+vertices, `Esc` quit

### Headless (CI-friendly)

```bash
python python/main.py --headless --frames 5
```

---

## Testing

### C++ tests

```bash
cd build && ctest -V
# or directly:
./test_voxel_engine
```

### Python tests

```bash
python -m pytest tests/python/ -v
```

---

## Key Components

### Noise (`cpp/include/noise.hpp`)
Simplex noise (2-D and 3-D) with **FBM** (Fractal Brownian Motion) and **ridged** noise
variants. Used for terrain height maps and biome sampling.

### Binary Voxel (`cpp/include/voxel.hpp`)
Each voxel is **1 byte**: 1 bit for solid/air + 7 bits for `BiomeType`. Efficient
storage in large arrays.

### Octree (`cpp/include/octree.hpp`)
Sparse octree over a power-of-2 volume. Supports `set`, `get`, `compress`
(merge uniform children), and two traversal modes: **leaf** and **LOD** (stop
at nodes of a target size).

### Chunk + Greedy Meshing (`cpp/include/chunk.hpp`)
Fixed 32x32x32 voxel chunk. **Greedy meshing** merges co-planar faces of the same
biome into quads (6 directions), emitting interleaved `[x,y,z, nx,ny,nz, biome_id]`
vertex buffers.

### Terrain Generator (`cpp/include/terrain.hpp`)
Combines FBM height + ridged noise for surface shape, and biome noise
(temperature x moisture) for horizontal biome selection.

### WFC (`cpp/include/wfc.hpp`)
Tile-based **Wave Function Collapse** on a 2-D biome grid. Adjacency rules define
valid biome-to-biome transitions (e.g. Ocean-Beach, Beach-Plains). Propagates
arc-consistency via BFS and restarts on contradiction.

### LOD (`cpp/include/lod.hpp`)
Four LOD levels based on viewer-to-chunk distance. Builds reduced-resolution
meshes from octree nodes of size `2^lod_level`.

### Async World (`cpp/include/world.hpp`)
Maintains a streaming chunk map around the viewer. Chunk loading and mesh building
are dispatched to a configurable **thread pool**; results are safely handed back
via mutex-protected shared data structures.

### PyOpenGL Renderer (`python/renderer.py`)
OpenGL 3.3 Core Profile renderer. One **VAO/VBO** per chunk. A single GLSL shader
pair uses the `biome_id` vertex attribute to look up the biome colour palette and
applies simple directional lighting. Graceful **headless fallback** when OpenGL
is not available.

### Camera (`python/camera.py`)
First-person fly camera with yaw/pitch control. Produces **view** and **perspective
projection** matrices as NumPy float32 arrays for direct upload to the shader.

---

## Literature

### Voxels
- *Voxel-Based Terrain for Real-Time Virtual Simulations* – Eric Lengyel
- *Game Engine Architecture* – Jason Gregory
- *Real-Time Rendering* – Tomas Akenine-Möller, Eric Haines, Naty Hoffman

### Procedural Generation
- *Procedural Content Generation in Games* – Shaker, Togelius, Nelson
- *The Art of Game Design: A Book of Lenses* – Jesse Schell
- *Procedural Generation in Game Design* – Tanya Short, Tarn Adams (eds.)

### Wave Function Collapse
- *Model Synthesis* – Paul Merrell *(original academic paper / thesis)*
- *AI Game Programming Wisdom* – Steve Rabin (ed.)
- *Artificial Intelligence for Games* – Ian Millington, John Funge

### Noise
- *Texturing & Modeling: A Procedural Approach* – Ebert, Musgrave, Peachey, Perlin, Worley
- *The Algorithmic Beauty of Plants* – Prusinkiewicz, Lindenmayer
- *Mathematics for 3D Game Programming and Computer Graphics* – Eric Lengyel
