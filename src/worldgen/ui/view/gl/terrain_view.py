"""The map in 3D: the whole terrain as one mesh coloured by the layer, lit and watered by the viewer's shaders,
seen by a camera orbiting the viewpoint's centre. Rendered into a texture the map picture shows. The controller
moves the camera; heights can be exaggerated about sea level, so relief reads on a large terrain.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyray as rl

from rayscreen import Theme
from worldgen.ui.model.map.terrain import Picture
from worldgen.ui.model.map.viewpoint import FOVY, Viewpoint
from worldgen.viewer.config import ViewerSettings
from worldgen.viewer.render import NO_FOG, Renderer
from worldgen.mesh.gpu import GpuPatch


class TerrainView:
    def __init__(self):
        self.settings = ViewerSettings()  # sun, ambient light and sky
        self.picture: Picture | None = None
        self.sea_level = 0.0
        self.exaggeration = 1.0
        self._mesh: GpuPatch | None = None  # the picture's terrain, once uploaded
        self._renderer: Renderer | None = None
        self._renderer_for: tuple[float, float] | None = None  # (world side, sea level): the open sea's frame
        self._target: rl.RenderTexture | None = None

    def set_picture(self, picture: Picture | None, sea_level: float) -> None:
        """Show `picture`'s terrain from now on; it is uploaded when next drawn (main thread)."""
        if self._mesh is not None:
            self._mesh.unload()
            self._mesh = None
        self.picture, self.sea_level = picture, sea_level

    def ground_at(self, x: float, z: float) -> float:
        """The height drawn at (x, z): the ground or the sea's surface, exaggerated about sea level."""
        sea = self.sea_level
        return sea + (max(self.picture.overview.height_at(x, z), sea) - sea) * self.exaggeration

    def render(self, width: int, height: int, viewpoint: Viewpoint, theme: Theme,
               overlay: Callable[[], None]) -> rl.Texture | None:
        """The terrain seen through `viewpoint`, `width` x `height` pixels, with whatever `overlay` draws over it
        (in the texture's pixels); None before a picture is set."""
        if self.picture is None or width < 2 or height < 2:
            return None
        ground_y = self._ground_y(viewpoint)
        eye, target = viewpoint.camera(ground_y)
        camera = rl.Camera3D(rl.Vector3(*eye), rl.Vector3(*target), rl.Vector3(0, 1, 0), FOVY,
                             rl.CameraProjection.CAMERA_PERSPECTIVE)
        self._render(viewpoint, camera, eye, width, height)
        rl.begin_texture_mode(self._target)
        overlay()
        rl.end_texture_mode()
        return self._target.texture

    def pick(self, viewpoint: Viewpoint, width: float, height: float, px: float, py: float) -> tuple[float, float] | None:
        """The world point (x, z) on the ground under pixel (px, py) of the view."""
        if self.picture is None:
            return None
        return viewpoint.pick(self._ground_y(viewpoint), width, height, px, py, self.ground_at)

    def _ground_y(self, viewpoint: Viewpoint) -> float:
        return self.ground_at(*(min(max(c, 0.0), viewpoint.side) for c in viewpoint.centre))

    def _render(self, viewpoint: Viewpoint, camera: rl.Camera3D, eye: np.ndarray, width: int, height: int) -> None:
        if self._target is None or (self._target.texture.width, self._target.texture.height) != (width, height):
            if self._target is not None:
                rl.unload_render_texture(self._target)
            self._target = rl.load_render_texture(width, height)
        if self._renderer_for != (viewpoint.side, self.sea_level):
            if self._renderer is not None:
                self._renderer.unload()
            self._renderer = Renderer(self.settings, viewpoint.side, self.sea_level)
            self._renderer_for = (viewpoint.side, self.sea_level)
        if self._mesh is None:
            self._mesh = GpuPatch(self.picture.terrain)
        renderer, distance = self._renderer, viewpoint.distance()
        rl.begin_texture_mode(self._target)
        rl.clear_background(rl.Color(*(int(c * 255) for c in self.settings.sky_color), 255))
        rl.rl_set_clip_planes(max(distance * 0.01, 1.0), distance * 4 + viewpoint.side * 2)
        rl.begin_mode_3d(camera)
        renderer.begin(eye, NO_FOG)
        transform = self._exaggerate()
        renderer.terrain(self._mesh, transform)
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA)
        renderer.begin_water()
        renderer.water(self._mesh, transform)
        renderer.open_sea()
        rl.end_blend_mode()
        renderer.end()
        rl.end_mode_3d()
        rl.end_texture_mode()

    def _exaggerate(self) -> rl.Matrix:
        """Heights scaled about sea level, as the viewer's minimap scales them."""
        sea = self.sea_level
        return rl.matrix_multiply(
            rl.matrix_multiply(rl.matrix_translate(0, -sea, 0), rl.matrix_scale(1, self.exaggeration, 1)),
            rl.matrix_translate(0, sea, 0),
        )

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.unload()
            self._mesh = None
        if self._renderer is not None:
            self._renderer.unload()
            self._renderer = None
        if self._target is not None:
            rl.unload_render_texture(self._target)
            self._target = None
