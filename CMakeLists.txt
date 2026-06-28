cmake_minimum_required(VERSION 3.18)
project(VoxelEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code required for shared libs / Python extensions.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Compiler flags ──────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Allow user to point at pybind11 cmake dir via -Dpybind11_DIR=...
find_package(pybind11 REQUIRED)

# ── Pybind11 extension module ────────────────────────────────────────────────
pybind11_add_module(voxel_core
    cpp/src/noise.cpp
    cpp/src/wfc.cpp
    cpp/src/octree.cpp
    cpp/src/chunk.cpp
    cpp/src/terrain.cpp
    cpp/src/lod.cpp
    cpp/src/world.cpp
    cpp/src/bindings.cpp
)
target_include_directories(voxel_core PRIVATE cpp/include)
target_compile_features(voxel_core PRIVATE cxx_std_17)

# Copy the built extension to the python/ directory so imports work directly.
add_custom_command(TARGET voxel_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:voxel_core>
        ${CMAKE_SOURCE_DIR}/python/$<TARGET_FILE_NAME:voxel_core>
    COMMENT "Copying voxel_core extension to python/"
)

# ── C++ unit tests ───────────────────────────────────────────────────────────
enable_testing()

add_executable(test_voxel_engine
    tests/cpp/test_main.cpp
    cpp/src/noise.cpp
    cpp/src/wfc.cpp
    cpp/src/octree.cpp
    cpp/src/chunk.cpp
    cpp/src/terrain.cpp
    cpp/src/lod.cpp
    cpp/src/world.cpp
)
target_include_directories(test_voxel_engine PRIVATE cpp/include)
target_compile_features(test_voxel_engine PRIVATE cxx_std_17)

add_test(NAME voxel_cpp_tests COMMAND test_voxel_engine)
