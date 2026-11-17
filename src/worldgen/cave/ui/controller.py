"""Controller for cavegen-ui: coordinates CaveSession, FpsCamera, Orbit Camera, CaveRenderer, dynamic recipe forms and log window."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pyray as rl

import rayscreen
from rayscreen import Column, Event, FileWindow, ScreenManager, Theme, Window
from worldgen.cave.config import CaveConfig
from worldgen.cave.export import CaveExportFormat, cave_meta, export_cave
from worldgen.cave.ui.model.cave_session import CaveSession
from worldgen.cave.ui.model.form import cave_recipe_form
from worldgen.cave.ui.model.fps_camera import FpsCamera
from worldgen.cave.ui.model.recipe import CaveRecipe
from worldgen.cave.ui.view.cave_screen import CaveScreen, CaveShown
from worldgen.cave.ui.view.gl.cave_render import CaveRenderer
from worldgen.cave.ui.view.stages import cave_recipe_tree
from worldgen.cave.ui.view.windows import (
    LOG,
    SAVE_WORLD,
    log_window,
    save_cave_window,
    show_cave_format,
    show_log,
)
from worldgen.ui.controller import log_news
from worldgen.ui.model.log import Log, Source
from worldgen.ui.model.recipe.form import FieldPath, Kind, blank, parse
from worldgen.ui.model.recipe.rules import can_move, can_remove, new_rule
from worldgen.ui.view import events as ev
from worldgen.ui.view.form import FormOptions
from worldgen.ui.view.stages import rule_cards
from worldgen.viewer.camera import FlyInput


class CaveController:
    """Coordinates cavegen UI rendering, cameras, events, background generation, and logging."""

    def __init__(
        self,
        config: CaveConfig,
        seed: int,
        out_dir: Path | None = None,
        log: Log | None = None,
        theme: Theme | None = None,
    ):
        self.recipe = CaveRecipe(None, config.model_dump(mode="json"))
        self.form_spec = cave_recipe_form()
        self.opened_sections: set[FieldPath] = set()

        self.seed = seed
        self.out_dir = out_dir or Path("out") / f"cave-seed-{seed}"
        self.log = log or Log()
        self._log_seen = 0
        if theme is not None:
            self.theme = theme
        else:
            self.theme = Theme()
            self.theme.set_dark(True)
        self.export_format: CaveExportFormat = "npz"
        self._save_to: tuple[Path, CaveExportFormat] | None = None

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

        self.shown = CaveShown(
            seed=str(seed),
            mode=self.mode,
            clearance=config.cavern.clearance,
            min_passage=config.cavern.min_passage,
            water_table=config.hydrology.water_table,
            perimeter_wall=config.cavern.perimeter_wall,
            dark=self.theme.dark,
        )
        self.screen = CaveScreen(self.shown, form=self._build_form_tree())
        self.manager = ScreenManager()
        self.manager.reset(self.screen)

        self.rendered_frames: int = 0
        self._last_mouse: tuple[float, float] | None = None
        self._spawn_needed: bool = True

        self._say("Ready")

    @property
    def config(self) -> CaveConfig:
        return self.recipe.config if self.recipe.config is not None else self.session.config

    def _say(self, message: str, error: bool = False) -> None:
        """Update status line and append entry to log."""
        self.shown.status = message
        self.log.add(Source.UI_ERROR if error else Source.UI, message)

    def _window(self, window_id: str) -> Window | None:
        return next((w for w in self.screen.windows if w.id == window_id), None)

    def _close(self, window_id: str) -> None:
        if (window := self._window(window_id)) is not None:
            self.screen.close_window(window)

    def toggle_log(self) -> None:
        """Open or close the floating log window."""
        if (window := self._window(LOG)) is not None:
            self.screen.close_window(window)
            return
        self._log_seen = self.log.added
        self.screen.open(log_window(self.log.lines()))

    def _form_options(self) -> FormOptions:
        return FormOptions(
            recipe=self.recipe,  # type: ignore[arg-type]
            opened=lambda path, default: path in self.opened_sections,
            rules=rule_cards,
        )

    def _build_form_tree(self) -> Column:
        options = self._form_options()
        statuses = self.session.stage_texts(self.recipe.config)
        return cave_recipe_tree(
            self.form_spec,
            options,
            statuses,
            runnable=self.recipe.valid and not self.session.busy,
        )

    def before_frame(self) -> None:
        """Called every frame before drawing: update session, navigate cameras, and render active view."""
        dt = rl.get_frame_time() if rl.is_window_ready() else 0.016

        # Poll background worker
        if self.session.poll():
            if rl.is_window_ready():
                self.renderer.upload_meshes(self.session.mesh_arrays, self.session.water_mesh_arrays)
                self.renderer.upload_plan(self.session.preview_image)
            self._update_metrics()
            if self._spawn_needed and self.session.cave is not None:
                pos, yaw, pitch = self.session.cave.find_spawn_point()
                self.fps_camera.position = pos
                self.fps_camera.yaw = yaw
                self.fps_camera.pitch = pitch
                self._spawn_needed = False
            if self.session.status.startswith("error"):
                self._say(self.session.status, error=True)
                self._save_to = None
            else:
                self._say("Generation complete")
                if self._save_to is not None and self.session.cave is not None:
                    out_dir, fmt = self._save_to
                    self._save_to = None
                    export_cave(self.session.cave, out_dir, fmt)
                    self._say(f"Exported cave ({fmt}) to {out_dir}")
            self.screen.refresh(self.shown, self._build_form_tree())
        elif self.session.busy:
            # Refresh timer during execution
            self.shown.runnable = False
            self.shown.stoppable = True
            self.shown.status = f"Running {self.session.running_stage}..."
            self.screen.refresh(self.shown, self._build_form_tree())

        # Update log news indicators and active log window
        self.shown.dark = self.theme.dark
        news, errors = log_news(self.log, self._log_seen)
        self.shown.log_news, self.shown.log_errors = news, errors
        window = self._window(LOG)
        if window is not None and self.log.added != self._log_seen:
            self._log_seen = self.log.added
            show_log(window, self.log.lines())

        # Handle user navigation
        if rl.is_window_ready():
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
        if fit <= 0:
            return
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
        self.screen.refresh(self.shown, self._build_form_tree())

    def _update_metrics(self) -> None:
        if self.session.cave is None:
            return
        m = cave_meta(self.session.cave)
        self.shown.volume_m3 = f"{m['volume_m3']:,.0f} m³"
        self.shown.passable_ratio = f"{m['passable_ratio']:.1%}"
        self.shown.min_clearance = f"{m['min_clearance_m']:.1f} m"
        self.shown.max_clearance = f"{m['max_clearance_m']:.1f} m"
        self.shown.runnable = self.recipe.valid and not self.session.busy
        self.shown.stoppable = self.session.busy

    def _run_all(self) -> None:
        if self.recipe.valid and not self.session.busy:
            assert self.recipe.config is not None
            self.session.run_all(self.recipe.config)
            self._say("Generating all stages...")

    def handle(self, events: list[Event]) -> None:
        """Dispatch events from rayscreen widgets."""
        refresh_needed = False

        dispatched = rayscreen.dispatch(self.screen, events)

        # First handle direct tool bar & legacy widgets
        for ev_item in dispatched:
            w = ev_item.widget
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
                self._run_all()
                refresh_needed = True
            elif w == "btn_stop":
                self.session.stop()
                self._say("Stopping...")
                refresh_needed = True
            elif w in ("btn_export", "save-world"):
                if self._window(SAVE_WORLD) is None:
                    self.screen.open(save_cave_window(self.out_dir, self.export_format))
                refresh_needed = True
            elif w == "input_seed" and isinstance(ev_item.value, str):
                try:
                    s = int(ev_item.value)
                    self.session.seed = s
                    self.shown.seed = str(s)
                    self._say(f"Seed set to {s}")
                    refresh_needed = True
                except ValueError:
                    pass
            elif w == "slider_clearance" and isinstance(ev_item.value, (int, float)):
                v = float(ev_item.value)
                self.recipe.set(("cavern", "clearance"), v)
                self.shown.clearance = v
                refresh_needed = True
            elif w == "slider_min_passage" and isinstance(ev_item.value, (int, float)):
                v = float(ev_item.value)
                self.recipe.set(("cavern", "min_passage"), v)
                self.shown.min_passage = v
                refresh_needed = True
            elif w == "slider_water_table" and isinstance(ev_item.value, (int, float)):
                v = float(ev_item.value)
                self.recipe.set(("hydrology", "water_table"), v)
                self.shown.water_table = v
                refresh_needed = True

        # Next dispatch translated form and dialog events
        translated = ev.translate(dispatched, self.form_spec, self.recipe)  # type: ignore[arg-type]
        for event in translated:
            if self.act(event):
                refresh_needed = True

        if refresh_needed:
            self.shown.runnable = self.recipe.valid and not self.session.busy
            self.shown.stoppable = self.session.busy
            self.screen.refresh(self.shown, self._build_form_tree())

    def _export_cave(self) -> None:
        """Export the cave bundle. If stages need running, run them first and save on completion."""
        if not self.recipe.valid:
            self._say("Cannot export: recipe has errors", error=True)
            return
        assert self.recipe.config is not None
        if not self.session.up_to_date(self.recipe.config):
            self._save_to = (self.out_dir, self.export_format)
            self.session.run_all(self.recipe.config)
            self._say("Updating cavern stages before export...")
            return
        if self.session.cave is not None:
            export_cave(self.session.cave, self.out_dir, self.export_format)
            self._say(f"Exported cave ({self.export_format}) to {self.out_dir}")
        else:
            self._save_to = (self.out_dir, self.export_format)
            self.session.run_all(self.recipe.config)
            self._say("Generating cavern before export...")

    def act(self, event: ev.Event) -> bool:
        """Apply an event, returning whether the UI needs a redraw."""
        match event:
            case ev.DialogAsked(window_id):
                if window_id == SAVE_WORLD and self._window(SAVE_WORLD) is None:
                    self.screen.open(save_cave_window(self.out_dir, self.export_format))
                return True
            case ev.DialogClosed(window_id):
                self._close(window_id)
                return True
            case ev.WorldPathTyped(text):
                self.out_dir = Path(text.strip()).expanduser()
                return True
            case ev.FormatChosen(format):
                self.export_format = format  # type: ignore[assignment]
                if isinstance(window := self._window(SAVE_WORLD), FileWindow):
                    show_cave_format(window, self.export_format)
                return True
            case ev.SaveWorld():
                self._close(SAVE_WORLD)
                self._export_cave()
                return True
            case ev.ToggleLog():
                self.toggle_log()
                return True
            case ev.ClearLog():
                self.log.clear()
                self._log_seen = 0
                if (window := self._window(LOG)) is not None:
                    show_log(window, self.log.lines())
                return True
            case ev.CopyLog():
                if rl.is_window_ready():
                    rl.set_clipboard_text(self.log.text())
                return False
            case ev.SectionToggled(path):
                self.opened_sections ^= {path}
                return True
            case ev.TextCommitted(path, text):
                spec, _ = self.form_spec.locate(path)
                val, unreadable = parse(spec.kind, text)  # type: ignore[arg-type]
                self.recipe.set(path, val, unreadable=unreadable)
                return True
            case ev.ValueChosen(path, value):
                self.recipe.set(path, value)
                return True
            case ev.OptionalToggled(path, on):
                spec, rest = self.form_spec.locate(path)
                if on:
                    self.recipe.turn_on(path, blank(spec, rest))
                else:
                    self.recipe.turn_off(path, drop_key=bool(rest) and spec.kind is Kind.WEIGHTS)  # type: ignore[union-attr]
                return True
            case ev.RunStage(stage):
                if self.recipe.valid and not self.session.busy:
                    assert self.recipe.config is not None
                    self.session.run(stage, self.recipe.config)
                    self._say(f"Running {stage}...")
                    return True
            case ev.ResetStage(stage):
                self.recipe.reset(stage)
                self.session.reset_stage(stage)
                self._say(f"{stage} reset to defaults")
                return True
            case ev.RunAll():
                self._run_all()
                return True
            case ev.StopAll():
                self.session.stop()
                self._say("Stopping...")
                return True
            case ev.ThemeSwitched():
                self.theme.set_dark(not self.theme.dark)
                self.shown.dark = self.theme.dark
                return True
            case ev.AddRule():
                rules = self.recipe.get(("materials", "rules"))
                if isinstance(rules, list):
                    rule = new_rule(rules)
                    at = max(len(rules) - 1, 0)
                    self.recipe.insert(("materials", "rules"), at, rule)
                    self.opened_sections.add(("materials", "rules", at))
                    self._say(f"Added {rule['id']}")
                    return True
            case ev.RemoveRule(index):
                rules = self.recipe.get(("materials", "rules"))
                if isinstance(rules, list) and can_remove(rules, index):
                    self.recipe.remove(("materials", "rules"), index)
                    self._say("Removed rule")
                    return True
            case ev.MoveRule(index, to):
                rules = self.recipe.get(("materials", "rules"))
                if isinstance(rules, list) and can_move(rules, index, to):
                    self.recipe.move(("materials", "rules"), index, to)
                    self._say("Reordered rules")
                    return True
            case _:
                return False
        return False

    def settled(self) -> bool:
        """True when generation has settled and picture is rendered."""
        return self.session.settled() and self.rendered_frames >= 3

    def close(self) -> None:
        """Free resources."""
        self.session.close()
        self.renderer.close()
