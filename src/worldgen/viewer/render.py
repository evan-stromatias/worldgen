"""Terrain and water shaders, and drawing patches with them.

The per-patch calls go straight to raylib's C functions: pyray's wrappers cost tens of microseconds
a call, which adds up over hundreds of patches a frame.
"""

from collections.abc import Sequence
from importlib.resources import files

import numpy as np
import pyray as rl
import raylib

from worldgen.viewer.config import ViewerSettings
from worldgen.mesh.gpu import GpuMesh, GpuPatch
from worldgen.mesh.builders import sea_frame

_SHADERS = files("worldgen").joinpath("viewer/shaders")
TERRAIN_VS, TERRAIN_FS = _SHADERS.joinpath("viewer_terrain.vs"), _SHADERS.joinpath("viewer_terrain.fs")
WATER_VS, WATER_FS = _SHADERS.joinpath("viewer_water.vs"), _SHADERS.joinpath("viewer_water.fs")
SHALLOW, DEEP, DEEP_AT = (0.27, 0.68, 0.74), (0.06, 0.24, 0.45), 6.0
SEA_REACH = 20000.0  # how far the open sea extends past the world's edges
NO_FOG = (1e9, 2e9)
# Debug colours by level of detail: overview stand-ins, then levels 0, 1, 2, 3, ...
LOD_TINTS = {-1: (0.6, 0.3, 0.8), 0: (0.2, 0.8, 0.3), 1: (0.9, 0.85, 0.2), 2: (0.95, 0.55, 0.15), 3: (0.9, 0.2, 0.2)}


VEC2, VEC3, VEC4, FLOAT = (
    rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2,
    rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3,
    rl.ShaderUniformDataType.SHADER_UNIFORM_VEC4,
    rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
)


class Uniforms:
    """A shader's uniforms by name, with locations looked up once."""

    def __init__(self, shader: rl.Shader):
        self.shader = shader
        self._locations: dict[str, int] = {}
        self._last: dict[str, tuple] = {}

    def set(self, name: str, values: float | Sequence[float] | np.ndarray, kind: int, *, cache: bool = False) -> None:
        """Set a float uniform; with `cache`, skip the call if the value has not changed."""
        floats = tuple(float(v) for v in np.ravel(values))
        if cache and self._last.get(name) == floats:
            return
        self._last[name] = floats
        location = self._locations.get(name)
        if location is None:
            location = self._locations[name] = rl.get_shader_location(self.shader, name)
        raylib.SetShaderValue(self.shader, location, rl.ffi.new("float[]", floats), kind)


class Renderer:
    def __init__(self, settings: ViewerSettings, world_size: float, sea_level: float):
        self.terrain_shader = rl.load_shader(str(TERRAIN_VS), str(TERRAIN_FS))
        self.water_shader = rl.load_shader(str(WATER_VS), str(WATER_FS))
        self.terrain_uniforms, self.water_uniforms = Uniforms(self.terrain_shader), Uniforms(self.water_shader)
        self.terrain_material = rl.load_material_default()
        self.terrain_material.shader = self.terrain_shader
        self.water_material = rl.load_material_default()
        self.water_material.shader = self.water_shader
        for uniforms in (self.terrain_uniforms, self.water_uniforms):
            uniforms.set("sunDir", settings.sun_direction, VEC3)
            uniforms.set("sunColor", settings.sun_color, VEC3)
            uniforms.set("fogColor", settings.sky_color, VEC3)
        self.terrain_uniforms.set("ambientColor", settings.ambient_color, VEC3)
        self.water_uniforms.set("shallowColor", SHALLOW, VEC3)
        self.water_uniforms.set("deepColor", DEEP, VEC3)
        self.water_uniforms.set("deepAt", DEEP_AT, FLOAT)
        self.sea = GpuMesh(sea_frame(world_size, sea_level, SEA_REACH, DEEP_AT * 4))
        self._identity = rl.matrix_identity()
        self._albedo = self.terrain_material.maps[rl.MaterialMapIndex.MATERIAL_MAP_ALBEDO]
        self.wireframe = False
        self.borders = 0.0
        self.lod_colors = False

    def begin(self, eye: np.ndarray, fog_range: tuple[float, float]) -> None:
        """Per-view uniforms; call inside begin_mode_3d before drawing."""
        for uniforms in (self.terrain_uniforms, self.water_uniforms):
            uniforms.set("viewPos", eye, VEC3)
            uniforms.set("fogRange", fog_range, VEC2)
        self.terrain_uniforms.set("borderWidth", self.borders, FLOAT)
        self.terrain_uniforms.set("tint", (0, 0, 0, 0), VEC4, cache=True)
        rl.rl_disable_backface_culling()  # skirts face outwards; nothing here is seen from below anyway
        if self.wireframe:
            rl.rl_enable_wire_mode()

    def end(self) -> None:
        if self.wireframe:
            rl.rl_disable_wire_mode()
        rl.rl_enable_backface_culling()

    def terrain(self, patch: GpuPatch, transform: rl.Matrix) -> None:
        if self.lod_colors:
            self.terrain_uniforms.set("tint", (*LOD_TINTS.get(patch.lod, (0.5, 0.5, 0.5)), 0.55), VEC4, cache=True)
        self.terrain_uniforms.set("atlasTiles", 0.0, FLOAT, cache=True)
        self._albedo.texture = patch.texture.texture
        raylib.DrawMesh(patch.terrain.ptr[0], self.terrain_material, transform)

    def begin_water(self) -> None:
        """Before drawing patch water: lift it off the bed (see viewer_water.vs)."""
        self.water_uniforms.set("lift", 1.0, FLOAT)

    def water(self, patch: GpuPatch, transform: rl.Matrix) -> None:
        if patch.water is not None:
            raylib.DrawMesh(patch.water.ptr[0], self.water_material, transform)

    def open_sea(self) -> None:
        self.water_uniforms.set("lift", 0.0, FLOAT)
        raylib.DrawMesh(self.sea.ptr[0], self.water_material, self._identity)

    def unload(self) -> None:
        self.sea.unload()
        rl.unload_shader(self.terrain_shader)
        rl.unload_shader(self.water_shader)
