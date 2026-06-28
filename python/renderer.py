"""
renderer.py – PyOpenGL renderer for the procedural voxel world.

Architecture
------------
* One VAO/VBO per chunk mesh (uploaded lazily when mesh data arrives).
* A single GLSL shader pair handles all chunks; biome_id is passed as
  a vertex attribute and mapped to a colour in the fragment shader.
* Skybox rendering is handled using a cubemap and a depth function trick.
* Supports graceful fallback when OpenGL is unavailable (headless mode).
"""

from __future__ import annotations

import math
import sys
import struct
from typing import Dict, Tuple, List

import numpy as np

# Optional OpenGL imports – mark unavailable so callers can skip.
try:
    import OpenGL.GL as gl
    import pygame
    _HAS_OPENGL = True
except ImportError:
    _HAS_OPENGL = False

# Biome colour palette (index = BiomeType int value, 0 = Air/background).
BIOME_COLOURS: Dict[int, Tuple[float, float, float]] = {
    0: (0.53, 0.81, 0.92),   # Air → sky blue (background)
    1: (0.10, 0.20, 0.60),   # Ocean
    2: (0.95, 0.90, 0.60),   # Beach
    3: (0.45, 0.75, 0.30),   # Plains
    4: (0.18, 0.50, 0.18),   # Forest
    5: (0.90, 0.75, 0.40),   # Desert
    6: (0.70, 0.85, 0.80),   # Tundra
    7: (0.55, 0.50, 0.45),   # Mountain
    8: (0.95, 0.95, 0.99),   # Snow
}

_SKYBOX_VERTICES = np.array([
    # positions          
    -1.0,  1.0, -1.0,
    -1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
     1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0, -1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0, -1.0,
    -1.0,  1.0,  1.0,
    -1.0, -1.0,  1.0,

     1.0, -1.0, -1.0,
     1.0, -1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0, -1.0,
     1.0, -1.0, -1.0,

    -1.0, -1.0,  1.0,
    -1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
     1.0, -1.0,  1.0,
    -1.0, -1.0,  1.0,

    -1.0,  1.0, -1.0,
     1.0,  1.0, -1.0,
     1.0,  1.0,  1.0,
     1.0,  1.0,  1.0,
    -1.0,  1.0,  1.0,
    -1.0,  1.0, -1.0,

    -1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0, -1.0,
     1.0, -1.0, -1.0,
    -1.0, -1.0,  1.0,
     1.0, -1.0,  1.0
], dtype=np.float32)

_SKYBOX_VERT_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main() {
    TexCoords = aPos;
    vec4 pos = projection * view * vec4(aPos, 1.0);
    // Setting z to w ensures the depth value is always 1.0 (max depth)
    gl_Position = pos.xyww; 
}
"""

_SKYBOX_FRAG_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;
uniform samplerCube skybox;
// Matches the fog color in the world shader
const vec3 FOG_COLOR = vec3(0.53, 0.81, 0.92);

void main() {    
    vec3 skyColor = texture(skybox, TexCoords).rgb;
    
    // Blend the skybox into the fog color near the horizon.
    // This perfectly hides the edge of the world and creates a "massive" scale.
    float horizonBlend = 1.0 - clamp(abs(normalize(TexCoords).y) * 2.5, 0.0, 1.0);
    
    FragColor = vec4(mix(skyColor, FOG_COLOR, horizonBlend), 1.0);
}
"""

_VERT_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aBiome;
layout(location = 3) in float aAO;

uniform mat4 uMVP;
uniform vec3 uCameraPos;

out vec3 vWorldPos;
out vec3 vNormal;
out float vBiome;
out float vAO;
out float vDist; // NEW: Pass distance for volumetric fog

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vWorldPos = aPos;
    vNormal = aNormal;
    vBiome  = aBiome;
    vAO     = aAO;
    vDist   = length(aPos - uCameraPos); // Calculate true distance to camera
}
"""

_FRAG_SHADER = """
#version 330 core
in vec3  vWorldPos;
in vec3  vNormal;
in float vBiome;
in float vAO;
in float vDist;

//uniform vec3 uBiomeColours[9];
uniform sampler2DArray uBlockTextures;
uniform vec3 uLightDir;

uniform vec3 uCameraPos;
uniform float uRenderDistance;
uniform float uIsMacro;

out vec4 FragColor;

const vec3 FOG_COLOR = vec3(0.53, 0.81, 0.92); // Sky color
const float FOG_DENSITY = 0.0012; // Lower = further visibility

void main() {

    if (uIsMacro > 0.5) {
        // Small 0.5 buffer prevents Z-fighting at the boundary
        float dist = length(vec2(vWorldPos.x - uCameraPos.x, vWorldPos.z - uCameraPos.z));
        if (dist < uRenderDistance - 0.5) {
            discard;
        }
    }

// --- TRIPLANAR TEXTURE MAPPING ---
    int texIndex = int(vBiome + 0.5); 

    // Calculate blending weights based on the absolute normal vector
    vec3 blendWeights = abs(vNormal);
    // Force weights to sum to 1.0 to ensure consistent brightness
    blendWeights = blendWeights / (blendWeights.x + blendWeights.y + blendWeights.z);

    // Scale controls how many world-units a texture covers (1.0 = 1 block)
    float texScale = 1.0 / 8.0;

    // Sample the texture array from the 3 orthogonal planes
    vec4 texColorX = texture(uBlockTextures, vec3(vWorldPos.yz * texScale, texIndex));
    vec4 texColorY = texture(uBlockTextures, vec3(vWorldPos.xz * texScale, texIndex));
    vec4 texColorZ = texture(uBlockTextures, vec3(vWorldPos.xy * texScale, texIndex));

    // Blend the samples together
    vec3 baseColor = (texColorX * blendWeights.x + 
                      texColorY * blendWeights.y + 
                      texColorZ * blendWeights.z).rgb;
    // ---------------------------------

    // 2. Lighting
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDir);
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 sunLight = vec3(1.1, 1.0, 0.85) * diff;
    vec3 ambientLight = vec3(0.35, 0.45, 0.6);
    float aoFactor = 0.35 + (vAO * 0.65);
    
    vec3 finalColor = baseColor * (sunLight + ambientLight) * aoFactor;

    // 3. Fog & Atmospheric Scattering
    float fogFactor = 1.0 - exp(-pow(vDist * FOG_DENSITY, 2.0));
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    float heightFog = exp(-max(vWorldPos.y - 25.0, 0.0) * 0.015);
    vec3 horizonFogColor = mix(vec3(0.75, 0.85, 0.95), FOG_COLOR, heightFog);
    
    finalColor = mix(finalColor, horizonFogColor, fogFactor);
    finalColor = pow(finalColor, vec3(1.0 / 1.4));

    FragColor = vec4(finalColor, 1.0);
}
"""


class _NullRenderer:
    """No-op renderer used when OpenGL is unavailable (e.g., CI/headless)."""

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self._meshes: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._wireframe = False

    def upload_chunk_mesh(self, cx: int, cy: int, cz: int, mesh) -> None:
        arr = np.asarray(mesh, dtype=np.float32)
        if arr.size:
            self._meshes[(cx, cy, cz)] = arr

    def remove_chunk_mesh(self, cx: int, cy: int, cz: int) -> None:
        self._meshes.pop((cx, cy, cz), None)

    def render(self, camera) -> None:
        pass  # headless: nothing to draw

    def swap(self) -> None:
        pass

    def poll_events(self) -> bool:
        return True  # keep running

    def cleanup(self) -> None:
        pass

    def set_wireframe(self, enabled: bool) -> bool:
        self._wireframe = bool(enabled)
        return self._wireframe

    def toggle_wireframe(self) -> bool:
        return self.set_wireframe(not self._wireframe)

    def total_vertex_count(self) -> int:
        return sum(len(mesh) // 7 for mesh in self._meshes.values())

    @property
    def available(self) -> bool:
        return False


class VoxelRenderer:
    """
    OpenGL 3.3 Core Profile renderer for voxel meshes.

    Parameters
    ----------
    width, height : window dimensions in pixels
    title         : window title
    headless      : if True, skip window creation (useful for testing)
    """

    def __init__(self, width: int = 1920, height: int = 1080,
                 title: str = "Voxel World", headless: bool = False, view_distance: float = 16.0):
        self.width  = width
        self.height = height
        self._headless = headless or not _HAS_OPENGL
        self._visible_chunks = set()
        self._view_distance = view_distance # Add a small buffer to ensure chunks fully load before they pop in

        if self._headless:
            self._null = _NullRenderer(width, height)
            return

        # ── Init pygame + OpenGL context ──────────────────────────────────
        pygame.init()
        pygame.display.set_caption(title)
        pygame.display.set_mode(
            (width, height),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        self._shader = self._compile_shader_from_src(_VERT_SHADER, _FRAG_SHADER)
        self._u_mvp   = gl.glGetUniformLocation(self._shader, "uMVP")
        self._u_biome = gl.glGetUniformLocation(self._shader, "uBiomeColours")
        self._u_light = gl.glGetUniformLocation(self._shader, "uLightDir")
        
        # NOVÉ: Uniformy pro vyříznutí díry do makro meshe
        self._u_cam_pos     = gl.glGetUniformLocation(self._shader, "uCameraPos")
        self._u_render_dist = gl.glGetUniformLocation(self._shader, "uRenderDistance")
        self._u_is_macro    = gl.glGetUniformLocation(self._shader, "uIsMacro")

        # Upload palette to shader.
        gl.glUseProgram(self._shader)
        texture_files = [
            None,                       # 0 = Air
            "Textures/water.png",       # 1 = Ocean
            "Textures/sand.png",        # 2 = Beach
            "Textures/grass_top.png",   # 3 = Plains
            "Textures/dirt.png",        # 4 = Forest
            "Textures/sand.png",        # 5 = Desert
            "Textures/stone.png",      # 6 = Tundra
            "Textures/gravel.png",       # 7 = Mountain
            "Textures/snow.png"         # 8 = Snow
        ]
        self._block_texture_array = self.load_texture_array(texture_files, tex_size=16)
        gl.glUseProgram(self._shader)
        u_textures = gl.glGetUniformLocation(self._shader, "uBlockTextures")
        gl.glUniform1i(u_textures, 1)  # Texture Unit 1

        # --- THE MISSING SUN ---
        # Fetch the uniform location and set the light direction again
        self._u_light = gl.glGetUniformLocation(self._shader, "uLightDir")
        gl.glUniform3f(self._u_light, 0.6, 1.0, 0.4)
        # Per-chunk GPU buffers: (cx,cy,cz) → (vao, vbo, vertex_count)
        self._buffers: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}
        # NOVÉ: Paměť pro Macro-Chunky
        self._macro_buffers: Dict[Tuple[int,int], Tuple[int,int,int]] = {} 
        self._visible_macros = set()
        self._wireframe = False

        # ── Skybox Setup ──────────────────────────────────────────────────
        self._skybox_shader = self._compile_shader_from_src(_SKYBOX_VERT_SHADER, _SKYBOX_FRAG_SHADER)
        self._u_sky_proj = gl.glGetUniformLocation(self._skybox_shader, "projection")
        self._u_sky_view = gl.glGetUniformLocation(self._skybox_shader, "view")
        self._u_skybox   = gl.glGetUniformLocation(self._skybox_shader, "skybox")

        self._skybox_vao = gl.glGenVertexArrays(1)
        self._skybox_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._skybox_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._skybox_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, _SKYBOX_VERTICES.nbytes, _SKYBOX_VERTICES, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, gl.ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

        faces = [
            "./Skybox/nx.png", "./Skybox/px.png",
            "./Skybox/py.png", "./Skybox/ny.png",
            "./Skybox/nz.png", "./Skybox/pz.png"
        ]
        self._cubemap_texture = self.load_cubemap(faces)
        self._null = None  # type: ignore

    # ── Shader compilation ─────────────────────────────────────────────────

    @staticmethod
    def _compile_shader_from_src(vs_src: str, fs_src: str) -> int:
        def compile_src(src: str, kind: int) -> int:
            s = gl.glCreateShader(kind)
            gl.glShaderSource(s, src)
            gl.glCompileShader(s)
            if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
                err = gl.glGetShaderInfoLog(s).decode()
                raise RuntimeError(f"Shader compile error: {err}")
            return s

        vs = compile_src(vs_src, gl.GL_VERTEX_SHADER)
        fs = compile_src(fs_src, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            err = gl.glGetProgramInfoLog(prog).decode()
            raise RuntimeError(f"Shader link error: {err}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog
    
    def load_cubemap(self, faces: List[str]) -> int:
        """Loads 6 images into an OpenGL cubemap texture.
           Expected order: right, left, top, bottom, front, back.
        """
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture_id)

        for i, face in enumerate(faces):
            img = pygame.image.load(face).convert_alpha()
            img_data = pygame.image.tostring(img, "RGBA", False)
            width, height = img.get_size()
            
            gl.glTexImage2D(
                gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, gl.GL_RGBA, width, height, 0, 
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data
            )

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        return texture_id
    
    def load_texture_array(self, image_paths: list[str], tex_size: int = 16) -> int:
        """Loads a list of images into a GL_TEXTURE_2D_ARRAY."""
        tex_array_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tex_array_id)

        # Allocate storage for the texture array
        layer_count = len(image_paths)
        gl.glTexImage3D(
            gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8, 
            tex_size, tex_size, layer_count, 
            0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None
        )

        # Upload each texture into its respective layer
        for i, img_path in enumerate(image_paths):
            if not img_path:  # Skip Air or empty indices
                continue
            
            img = pygame.image.load(img_path).convert_alpha()
            # Ensure images are scaled to uniform size (e.g. 16x16)
            if img.get_size() != (tex_size, tex_size):
                img = pygame.transform.scale(img, (tex_size, tex_size))
                
            img_data = pygame.image.tostring(img, "RGBA", True)
            gl.glTexSubImage3D(
                gl.GL_TEXTURE_2D_ARRAY, 0, 
                0, 0, i,  # xoffset, yoffset, zoffset (layer i)
                tex_size, tex_size, 1, 
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data
            )

        # Set texture parameters (NEAREST filter for that blocky voxel look)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        
        return tex_array_id

    # ── Mesh management ────────────────────────────────────────────────────

    def set_visible_chunks(self, coords: list[tuple[int, int, int]]) -> None:
        """Updates the list of chunks that should be drawn this frame."""
        if self._headless:
            return
        self._visible_chunks = set(coords)

    def set_visible_macros(self, coords: list[tuple[int, int]]) -> None:
        if self._headless: 
            return
        self._visible_macros = set(coords)

    def upload_macro_mesh(self, rx: int, rz: int, mesh) -> None:
        if self._headless: 
            return
        key = (rx, rz)
        
        # Delete old buffer
        if key in self._macro_buffers:
            vao, vbo, _ = self._macro_buffers.pop(key)
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])

        data = np.asarray(mesh, dtype=np.float32)
        if not data.size: 
            return

        # FIX: Reverted to 8! The C++ macro mesh (world.cpp) DOES append the 8th aAO float.
        vertex_count = data.size // 8  

        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)

        stride = 8 * 4
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(24))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(28))
        gl.glEnableVertexAttribArray(3)

        gl.glBindVertexArray(0)
        self._macro_buffers[key] = (vao, vbo, vertex_count)

    def upload_chunk_mesh(self, cx: int, cy: int, cz: int, mesh) -> None:
        if self._headless:
            self._null.upload_chunk_mesh(cx, cy, cz, mesh)
            return

        key = (cx, cy, cz)
        self.remove_chunk_mesh(cx, cy, cz)  # delete old buffers if any

        data = np.asarray(mesh, dtype=np.float32)
        if not data.size:
            return

        vertex_count = data.size // 7  # 7 floats per vertex

        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)

        stride = 7 * 4  # 7 float32
        # aPos (location 0): 3 floats at offset 0
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                  gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # aNormal (location 1): 3 floats at offset 12
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                  gl.ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        # aBiome (location 2): 1 float at offset 24
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                  gl.ctypes.c_void_p(24))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)
        self._buffers[key] = (vao, vbo, vertex_count)

    def remove_chunk_mesh(self, cx: int, cy: int, cz: int) -> None:
        """Free GPU buffers for the given chunk."""
        if self._headless:
            self._null.remove_chunk_mesh(cx, cy, cz)
            return
        key = (cx, cy, cz)
        if key in self._buffers:
            vao, vbo, _ = self._buffers.pop(key)
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteBuffers(1, [vbo])

    # ── Render loop ────────────────────────────────────────────────────────

    def render(self, camera) -> None:
        if self._headless:
            return

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self._buffers and not self._macro_buffers:
            return

        aspect = self.width / max(1, self.height)
        mvp = camera.mvp(aspect).astype(np.float32, order='C')

        # ── 1. Render World Chunks ───────────────────────────────────────
        gl.glUseProgram(self._shader)
        gl.glUniformMatrix4fv(self._u_mvp, 1, gl.GL_TRUE, mvp)
        gl.glUniform3f(self._u_cam_pos, camera.position[0], camera.position[1], camera.position[2])
        
        chunk_size = 32.0
        gl.glUniform1f(self._u_render_dist, (self._view_distance * chunk_size) - 2.0)

        # NEW: Bind the texture array to texture unit 1
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self._block_texture_array)

        # Cache PyOpenGL functions locally for speed
        bind_vao = gl.glBindVertexArray
        draw_arrays = gl.glDrawArrays
        GL_TRIANGLES = gl.GL_TRIANGLES

        # Safe global override: Detail chunks don't have attribute 3 enabled in their VAO,
        # so they will read this default value of 1.0. Macro chunks DO have attribute 3
        # enabled, so they will ignore this and read straight from their VBO.
        # This completely avoids illegal OpenGL state transitions!
        gl.glVertexAttrib1f(3, 1.0) 

        # -- Vykreslování DETAILNÍCH CHUNKŮ --
        gl.glUniform1f(self._u_is_macro, 0.0)
        for key in self._visible_chunks:
            if key in self._buffers:
                vao, _vbo, vertex_count = self._buffers[key]
                bind_vao(vao)
                draw_arrays(GL_TRIANGLES, 0, vertex_count)

        # -- Vykreslování MAKRO CHUNKŮ --
        gl.glUniform1f(self._u_is_macro, 1.0)
        for key in self._visible_macros:
            if key in self._macro_buffers:
                vao, _vbo, vertex_count = self._macro_buffers[key]
                bind_vao(vao)
                draw_arrays(GL_TRIANGLES, 0, vertex_count)

        bind_vao(0)

        # ── 2. Render Skybox ─────────────────────────────────────────────
        gl.glDepthFunc(gl.GL_LEQUAL)  
        gl.glUseProgram(self._skybox_shader)

        view = camera.view_matrix()
        view[0, 3] = 0.0
        view[1, 3] = 0.0
        view[2, 3] = 0.0
        view_f32 = view.astype(np.float32, order='C')
        
        proj_f32 = camera.projection_matrix(aspect).astype(np.float32, order='C')

        gl.glUniformMatrix4fv(self._u_sky_view, 1, gl.GL_TRUE, view_f32)
        gl.glUniformMatrix4fv(self._u_sky_proj, 1, gl.GL_TRUE, proj_f32)

        bind_vao(self._skybox_vao)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self._cubemap_texture)
        draw_arrays(GL_TRIANGLES, 0, 36)

        bind_vao(0)
        gl.glDepthFunc(gl.GL_LESS)

    def swap(self) -> None:
        if not self._headless:
            pygame.display.flip()

    def poll_events(self) -> bool:
        """Process OS events; return False when the window is closed."""
        if self._headless:
            return True
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
        return True

    def cleanup(self) -> None:
        if not self._headless:
            for vao, vbo, _ in self._buffers.values():
                gl.glDeleteVertexArrays(1, [vao])
                gl.glDeleteBuffers(1, [vbo])
            gl.glDeleteProgram(self._shader)
            if hasattr(self, '_skybox_shader'):
                gl.glDeleteProgram(self._skybox_shader)
            pygame.quit()

    def set_wireframe(self, enabled: bool) -> bool:
        if self._headless:
            return self._null.set_wireframe(enabled)
        self._wireframe = bool(enabled)
        mode = gl.GL_LINE if self._wireframe else gl.GL_FILL
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, mode)
        return self._wireframe

    def toggle_wireframe(self) -> bool:
        if self._headless:
            return self._null.toggle_wireframe()
        return self.set_wireframe(not self._wireframe)

    def total_vertex_count(self) -> int:
        if self._headless:
            return self._null.total_vertex_count()
        return sum(vertex_count for _, _, vertex_count in self._buffers.values())

    @property
    def available(self) -> bool:
        return _HAS_OPENGL and not self._headless