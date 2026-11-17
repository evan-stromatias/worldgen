"""Raylib OpenGL rendering for cavegen-ui: 2D plan, 3D orbit, and 3D FPS camera modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyray as rl
import raylib

from worldgen.cave.ui.model.fps_camera import FpsCamera
from worldgen.mesh.builders import MeshArrays
from worldgen.mesh.gpu import GpuMesh

_SHADERS_DIR = Path(__file__).resolve().parent / "shaders"
CAVE_VS = _SHADERS_DIR / "cave.vs"
CAVE_FS = _SHADERS_DIR / "cave.fs"

VEC3 = rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3
FLOAT = rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT


class CaveRenderer:
    """Renders subterranean caverns into a render texture for rayscreen UI display."""

    def __init__(self):
        self.gpu_mesh: GpuMesh | None = None
        self.gpu_water: GpuMesh | None = None
        self.plan_texture: rl.Texture | None = None
        self.target: rl.RenderTexture | None = None
        self.shader: rl.Shader | None = None
        self.material: rl.Material | None = None

        if rl.is_window_ready():
            self._init_gl()

    def _init_gl(self) -> None:
        if CAVE_VS.exists() and CAVE_FS.exists():
            try:
                self.shader = rl.load_shader(str(CAVE_VS), str(CAVE_FS))
            except Exception:  # noqa: BLE001
                self.shader = None

        self.material = rl.load_material_default()
        if self.shader is not None:
            self.material.shader = self.shader

    def upload_meshes(self, mesh_arrays: MeshArrays | None, water_arrays: MeshArrays | None = None) -> None:
        """Upload mesh geometry to GPU on the main thread."""
        if not rl.is_window_ready():
            return

        if self.material is None:
            self._init_gl()

        if self.gpu_mesh is not None:
            self.gpu_mesh.unload()
            self.gpu_mesh = None
        if self.gpu_water is not None:
            self.gpu_water.unload()
            self.gpu_water = None

        if mesh_arrays is not None and len(mesh_arrays.positions) > 0:
            self.gpu_mesh = GpuMesh(mesh_arrays)
        if water_arrays is not None and len(water_arrays.positions) > 0:
            self.gpu_water = GpuMesh(water_arrays)

    def upload_plan(self, rgb: np.ndarray | None) -> None:
        """Upload 2D plan overview texture."""
        if not rl.is_window_ready():
            return

        if self.plan_texture is not None:
            rl.unload_texture(self.plan_texture)
            self.plan_texture = None

        if rgb is not None:
            h, w, _ = rgb.shape
            data = rl.ffi.from_buffer("unsigned char[]", np.ascontiguousarray(rgb))
            image = rl.Image(data, w, h, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
            self.plan_texture = rl.load_texture_from_image(image)
            rl.set_texture_filter(self.plan_texture, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)

    def _ensure_target(self, width: int, height: int) -> rl.RenderTexture | None:
        if not rl.is_window_ready():
            return None
        w = max(2, width)
        h = max(2, height)
        if self.target is None or self.target.texture.width != w or self.target.texture.height != h:
            if self.target is not None:
                rl.unload_render_texture(self.target)
            self.target = rl.load_render_texture(w, h)
        return self.target

    def render_2d_plan(
        self, width: int, height: int, zoom: float, pan: tuple[float, float], world_size_m: float
    ) -> rl.Texture | None:
        """Render 2D plan view of the cave clearance and floor contours."""
        target = self._ensure_target(width, height)
        if target is None:
            return None
        rl.begin_texture_mode(target)
        rl.clear_background(rl.Color(18, 20, 24, 255))

        if self.plan_texture is not None:
            tex_w = float(self.plan_texture.width)
            tex_h = float(self.plan_texture.height)
            fit = min(width / tex_w, height / tex_h) * zoom
            dw = tex_w * fit
            dh = tex_h * fit
            dx = (width - dw) / 2.0 + pan[0]
            dy = (height - dh) / 2.0 + pan[1]

            src = rl.Rectangle(0, 0, tex_w, tex_h)
            dst = rl.Rectangle(dx, dy, dw, dh)
            rl.draw_texture_pro(self.plan_texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

        rl.end_texture_mode()
        return target.texture

    def _set_uniform(self, name: str, values: list[float], kind: int) -> None:
        if self.shader is None:
            return
        loc = rl.get_shader_location(self.shader, name)
        if loc >= 0:
            raylib.SetShaderValue(self.shader, loc, rl.ffi.new("float[]", values), kind)

    def render_3d_orbit(
        self,
        width: int,
        height: int,
        camera: rl.Camera3D,
    ) -> rl.Texture | None:
        """Render 3D orbit view with ambient bounce and directional light."""
        target = self._ensure_target(width, height)
        if target is None or self.material is None:
            return None
        rl.begin_texture_mode(target)
        rl.clear_background(rl.Color(16, 18, 22, 255))

        # Shader uniforms: Orbit mode
        self._set_uniform("ambientColor", [0.45, 0.45, 0.48], VEC3)
        self._set_uniform("lightDir", [-0.3, 1.0, -0.4], VEC3)
        self._set_uniform("lightColor", [0.85, 0.82, 0.78], VEC3)

        rl.begin_mode_3d(camera)
        mat = rl.matrix_identity()
        if self.gpu_mesh is not None:
            rl.draw_mesh(self.gpu_mesh.mesh, self.material, mat)
        if self.gpu_water is not None:
            rl.draw_mesh(self.gpu_water.mesh, self.material, mat)
        rl.end_mode_3d()

        rl.end_texture_mode()
        return target.texture

    def render_3d_fps(
        self,
        width: int,
        height: int,
        fps_cam: FpsCamera,
    ) -> rl.Texture | None:
        """Render 3D first-person interior view with clean global illumination."""
        target = self._ensure_target(width, height)
        if target is None or self.material is None:
            return None
        rl.begin_texture_mode(target)
        rl.clear_background(rl.Color(20, 22, 28, 255))

        pos = fps_cam.position
        target_pos = fps_cam.target
        camera = rl.Camera3D(
            rl.Vector3(pos[0], pos[1], pos[2]),
            rl.Vector3(target_pos[0], target_pos[1], target_pos[2]),
            rl.Vector3(0.0, 1.0, 0.0),
            65.0,  # FOV
            rl.CameraProjection.CAMERA_PERSPECTIVE,
        )

        # Shader uniforms: Global inspectable lighting
        self._set_uniform("ambientColor", [0.45, 0.45, 0.48], VEC3)
        self._set_uniform("lightDir", [0.3, 0.9, 0.4], VEC3)
        self._set_uniform("lightColor", [0.85, 0.82, 0.78], VEC3)

        rl.begin_mode_3d(camera)
        mat = rl.matrix_identity()
        if self.gpu_mesh is not None:
            rl.draw_mesh(self.gpu_mesh.mesh, self.material, mat)
        if self.gpu_water is not None:
            rl.draw_mesh(self.gpu_water.mesh, self.material, mat)
        rl.end_mode_3d()

        rl.end_texture_mode()
        return target.texture

    def close(self) -> None:
        """Unload GPU resources."""
        if not rl.is_window_ready():
            return
        if self.gpu_mesh is not None:
            self.gpu_mesh.unload()
            self.gpu_mesh = None
        if self.gpu_water is not None:
            self.gpu_water.unload()
            self.gpu_water = None
        if self.plan_texture is not None:
            rl.unload_texture(self.plan_texture)
            self.plan_texture = None
        if self.target is not None:
            rl.unload_render_texture(self.target)
            self.target = None
        if self.shader is not None:
            rl.unload_shader(self.shader)
            self.shader = None
