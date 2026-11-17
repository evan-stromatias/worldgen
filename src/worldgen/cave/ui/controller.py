"""Controller for cavegen-ui: ties CaveSession, FpsCamera, Orbit Camera, CaveRenderer and rayscreen together."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyray as rl

from rayscreen import Event, ScreenManager, Theme
from worldgen.cave.config import CaveConfig
from worldgen.cave.export import cave_meta, export_cave
from worldgen.cave.ui.model.cave_session import CaveSession
from worldgen.cave.ui.model.fps_camera import FpsCamera
from worldgen.cave.ui.view.cave_screen import CaveScreen, CaveShown
from worldgen.cave.ui.view.gl.cave_render import CaveRenderer
from worldgen.viewer.camera import FlyInput


class CaveController:
    """Coordinates cavegen UI rendering, cameras, events, and background generation."""

    def __init__(
        self,
        config: CaveConfig,
        seed: int,
        out_dir: Path | None = None,
        theme: Theme | None = None,
    ):
        self.config = config
        self.seed = seed
        self.out_dir = out_dir or Path("out") / f"cave-seed-{seed}"
        self.theme = theme or Theme()
        self.export_format: str = "npz"

        self.session = CaveSession(config, seed)
        self.renderer = CaveRenderer()

        self.mode: str = "2d"  # "2d", "3d_orbit", "3d_fps"
        self.cutaway: bool = False

        # 2D Navigation
        self.pan: tuple[float, float] = (0.0, 0.0)
        self.zoom: float = 1.0

        # 3D Orbit Navigation
        self.orbit_yaw: float = 0.8
        self.orbit_pitch: float = 0.55
        mid_m = (config.world.size_cells * config.world.cell_size) / 2.0
        self.orbit_target = np.array([mid_m, 10.0, mid_m], dtype=np.float64)
        self.orbit_dist: float = mid_m * 1.5

        # 3D FPS Navigation
        self.fps_camera = FpsCamera(
            position=np.array([mid_m, 15.0, mid_m], dtype=np.float64),
            yaw=0.0,
            pitch=0.0,
        )

        self.shown = CaveShown(seed=str(seed), mode=self.mode)
        self.screen = CaveScreen(self.shown)
        self.manager = ScreenManager()
        self.manager.reset(self.screen)

        self.rendered_frames: int = 0
        self._last_mouse: tuple[float, float] | None = None
        self._spawn_needed: bool = True

    def before_frame(self) -> None:
        """Called every frame before drawing: update session, navigate cameras, and render active view."""
        dt = rl.get_frame_time()

        # Poll background worker
        if self.session.poll():
            self.renderer.upload_meshes(self.session.mesh_arrays, self.session.water_mesh_arrays)
            self.renderer.upload_plan(self.session.preview_image)
            self._update_metrics()
            if self._spawn_needed and self.session.cave is not None:
                pos, yaw, pitch = self.session.cave.find_spawn_point()
                self.fps_camera.position = pos
                self.fps_camera.yaw = yaw
                self.fps_camera.pitch = pitch
                self._spawn_needed = False

        # Handle user navigation
        self._handle_navigation(dt)

        # Render view into texture
        pw = int(max(2, self.screen.picture.rect.width))
        ph = int(max(2, self.screen.picture.rect.height))

        tex: rl.Texture | None = None
        if self.mode == "2d":
            world_size = self.session.config.world.size_cells * self.session.config.world.cell_size
            tex = self.renderer.render_2d_plan(pw, ph, self.zoom, self.pan, world_size)
        elif self.mode == "3d_orbit":
            cam = self._make_orbit_camera()
            tex = self.renderer.render_3d_orbit(pw, ph, cam)
        elif self.mode == "3d_fps":
            tex = self.renderer.render_3d_fps(pw, ph, self.fps_camera)

        if tex is not None:
            self.screen.picture.texture = tex
            self.screen.picture.size = (tex.width, tex.height)

        self.rendered_frames += 1

    def _make_orbit_camera(self) -> rl.Camera3D:
        cy = math.cos(self.orbit_yaw)
        sy = math.sin(self.orbit_yaw)
        cp = math.cos(self.orbit_pitch)
        sp = math.sin(self.orbit_pitch)

        dx = self.orbit_dist * cp * sy
        dy = self.orbit_dist * sp
        dz = self.orbit_dist * cp * cy

        pos = self.orbit_target + np.array([dx, dy, dz], dtype=np.float64)
        return rl.Camera3D(
            rl.Vector3(pos[0], pos[1], pos[2]),
            rl.Vector3(self.orbit_target[0], self.orbit_target[1], self.orbit_target[2]),
            rl.Vector3(0.0, 1.0, 0.0),
            45.0,
            rl.CameraProjection.CAMERA_PERSPECTIVE,
        )

    def _handle_navigation(self, dt: float) -> None:
        mouse = rl.get_mouse_position()
        rect = self.screen.picture.rect
        inside = rect.x <= mouse.x <= rect.x + rect.width and rect.y <= mouse.y <= rect.y + rect.height
        wheel = rl.get_mouse_wheel_move()

        # Update 2D hover readout
        if self.mode == "2d" and inside and self.session.cave is not None:
            self._update_2d_readout(mouse.x - rect.x, mouse.y - rect.y, rect.width, rect.height)

        # 2D Plan Navigation
        if self.mode == "2d":
            if inside and wheel != 0:
                self.zoom = min(max(self.zoom * (1.15**wheel), 0.2), 20.0)
            if inside and rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT):
                delta = rl.get_mouse_delta()
                self.pan = (self.pan[0] + delta.x, self.pan[1] + delta.y)

        # 3D Orbit Navigation
        elif self.mode == "3d_orbit":
            if inside and wheel != 0:
                self.orbit_dist = min(max(self.orbit_dist * (0.9**wheel), 20.0), 3000.0)
            if inside and (
                rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
                or rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)
            ):
                delta = rl.get_mouse_delta()
                self.orbit_yaw = (self.orbit_yaw - delta.x * 0.005) % (2 * math.pi)
                self.orbit_pitch = min(max(self.orbit_pitch + delta.y * 0.005, 0.05), 1.5)

        # 3D FPS Navigation
        elif self.mode == "3d_fps":
            look_dx, look_dy = 0.0, 0.0
            if inside and (
                rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_RIGHT)
                or rl.is_mouse_button_down(rl.MouseButton.MOUSE_BUTTON_LEFT)
            ):
                delta = rl.get_mouse_delta()
                look_dx, look_dy = delta.x, delta.y

            # Gather movement keys
            move_x, move_y, move_z = 0.0, 0.0, 0.0
            if rl.is_key_down(rl.KeyboardKey.KEY_W):
                move_z += 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_S):
                move_z -= 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_A):
                move_x -= 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_D):
                move_x += 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_SPACE):
                move_y += 1.0
            if rl.is_key_down(rl.KeyboardKey.KEY_LEFT_CONTROL) or rl.is_key_down(rl.KeyboardKey.KEY_C):
                move_y -= 1.0

            boost = rl.is_key_down(rl.KeyboardKey.KEY_LEFT_SHIFT)
            inp = FlyInput(
                look=(look_dx, look_dy),
                move=(move_x, move_y, move_z),
                boost=boost,
                speed_steps=wheel if inside else 0.0,
            )

            cave = self.session.cave
            if cave is not None:
                self.fps_camera.update(dt, inp, cave.floor_at, cave.ceiling_at, cave.is_navigable)

    def _update_2d_readout(self, px: float, py: float, rw: float, rh: float) -> None:
        cave = self.session.cave
        assert cave is not None
        world_size = cave.config.world.size_cells * cave.config.world.cell_size
        fit = min(rw / world_size, rh / world_size) * self.zoom
        dw = world_size * fit
        dh = world_size * fit
        ox = (rw - dw) / 2.0 + self.pan[0]
        oy = (rh - dh) / 2.0 + self.pan[1]

        wx = (px - ox) / fit
        wz = (py - oy) / fit

        if 0 <= wx < world_size and 0 <= wz < world_size:
            hf = cave.floor_at(wx, wz)
            hc = cave.ceiling_at(wx, wz)
            headroom = hc - hf
            nav = cave.is_navigable(wx, wz)
            status_pillar = "Open Passage" if nav else "Solid Pillar / Rock Wall"
            self.shown.readout = f"({wx:.1f}m, {wz:.1f}m) · Floor: {hf:.1f}m · Ceiling: {hc:.1f}m · Headroom: {headroom:.1f}m [{status_pillar}]"
        else:
            self.shown.readout = "Outside cavern bounds"
        self.screen.refresh(self.shown)

    def _update_metrics(self) -> None:
        if self.session.cave is None:
            return
        m = cave_meta(self.session.cave)
        self.shown.volume_m3 = f"{m['volume_m3']:,.0f} m³"
        self.shown.passable_ratio = f"{m['passable_ratio']:.1%}"
        self.shown.min_clearance = f"{m['min_clearance_m']:.1f} m"
        self.shown.max_clearance = f"{m['max_clearance_m']:.1f} m"
        self.shown.status = "Generation complete"
        self.screen.refresh(self.shown)

    def handle(self, events: list[Event]) -> None:
        """Dispatch events from rayscreen widgets."""
        refresh_needed = False
        for ev in events:
            w = ev.widget
            if w == "btn_mode_2d":
                self.mode = "2d"
                self.shown.mode = "2d"
                refresh_needed = True
            elif w == "btn_mode_orbit":
                self.mode = "3d_orbit"
                self.shown.mode = "3d_orbit"
                refresh_needed = True
            elif w == "btn_mode_fps":
                self.mode = "3d_fps"
                self.shown.mode = "3d_fps"
                refresh_needed = True
            elif w == "btn_cutaway":
                self.cutaway = not self.cutaway
                self.shown.cutaway = self.cutaway
                self.session.include_ceiling = not self.cutaway
                self.session.rebuild_mesh()
                if rl.is_window_ready():
                    self.renderer.upload_meshes(self.session.mesh_arrays, self.session.water_mesh_arrays)
                refresh_needed = True
            elif w == "btn_run":
                self.session.start_generation()
                self.shown.status = "Generating..."
                refresh_needed = True
            elif w == "btn_export":
                if self.session.cave is not None:
                    export_cave(self.session.cave, self.out_dir, self.export_format)
                    self.shown.status = f"Exported to {self.out_dir}"
                    refresh_needed = True
            elif w == "input_seed" and isinstance(ev.value, str):
                try:
                    s = int(ev.value)
                    self.session.seed = s
                    self.shown.seed = str(s)
                    refresh_needed = True
                except ValueError:
                    pass
            elif w == "slider_clearance" and isinstance(ev.value, (int, float)):
                v = float(ev.value)
                new_cavern = self.session.config.cavern.model_copy(update={"clearance": v})
                self.session.config = self.session.config.model_copy(update={"cavern": new_cavern})
                self.shown.clearance = v
                refresh_needed = True
            elif w == "slider_min_passage" and isinstance(ev.value, (int, float)):
                v = float(ev.value)
                new_cavern = self.session.config.cavern.model_copy(update={"min_passage": v})
                self.session.config = self.session.config.model_copy(update={"cavern": new_cavern})
                self.shown.min_passage = v
                refresh_needed = True
            elif w == "slider_water_table" and isinstance(ev.value, (int, float)):
                v = float(ev.value)
                new_hydro = self.session.config.hydrology.model_copy(update={"water_table": v})
                self.session.config = self.session.config.model_copy(update={"hydrology": new_hydro})
                self.shown.water_table = v
                refresh_needed = True

        if refresh_needed:
            self.screen.refresh(self.shown)

    def settled(self) -> bool:
        """True when generation has settled and picture is rendered."""
        return self.session.settled() and self.rendered_frames >= 3

    def close(self) -> None:
        """Free resources."""
        self.renderer.close()
