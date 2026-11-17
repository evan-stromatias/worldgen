from pathlib import Path

from rayscreen import Event, Label
from rayscreen.specs import check_unique_ids, walk
from worldgen.cave.config import parse_config
from worldgen.cave.ui.controller import CaveController
from worldgen.cave.ui.model.cave_session import CaveSession

SMALL_CAVE = {
    "world": {"size_cells": 16, "patch_cells": 16, "cell_size": 1.0},
    "elevation": {"noise": {"frequency": 0.05}},
    "cavern": {
        "clearance": 12.0,
        "ceiling_mode": "inverted",
        "min_passage": 2.5,
        "perimeter_wall": True,
        "wall_thickness": 1,
    },
}


def test_cave_session_sync():
    cfg = parse_config(SMALL_CAVE)
    session = CaveSession(cfg, seed=5)
    assert not session.running

    session.run_sync()
    assert session.cave is not None
    assert session.mesh_arrays is not None
    assert session.preview_image is not None
    assert session.settled()


def test_cave_session_background():
    cfg = parse_config(SMALL_CAVE)
    session = CaveSession(cfg, seed=5)
    session.start_generation()
    assert session.running

    # Wait for thread
    session._thread.join(timeout=5.0)
    assert session.poll()
    assert session.cave is not None
    assert session.mesh_arrays is not None


def test_cave_controller_events(tmp_path: Path):
    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=7, out_dir=tmp_path / "ctrl_out")

    # Dark theme by default
    assert controller.theme.dark is True
    assert controller.shown.dark is True

    # Theme toggle event via button id
    controller.handle([Event("cave_screen", "theme", True)])
    assert controller.theme.dark is False
    assert controller.shown.dark is False

    controller.handle([Event("cave_screen", "theme", True)])
    assert controller.theme.dark is True
    assert controller.shown.dark is True

    # Theme toggle event via ThemeSwitched
    from worldgen.ui.view.events import ThemeSwitched
    controller.act(ThemeSwitched())
    assert controller.theme.dark is False

    # Mode events
    controller.handle([Event("cave_screen", "btn_mode_orbit", True)])
    assert controller.mode == "3d_orbit"

    controller.handle([Event("cave_screen", "btn_mode_fps", True)])
    assert controller.mode == "3d_fps"

    controller.handle([Event("cave_screen", "btn_mode_2d", True)])
    assert controller.mode == "2d"

    # Slider events
    controller.handle([Event("cave_screen", "slider_clearance", 24.0)])
    assert controller.shown.clearance == 24.0

    controller.handle([Event("cave_screen", "slider_min_passage", 3.5)])
    assert controller.shown.min_passage == 3.5

    controller.handle([Event("cave_screen", "slider_water_table", 2.0)])
    assert controller.shown.water_table == 2.0

    # Seed event
    controller.handle([Event("cave_screen", "input_seed", "99")])
    assert controller.session.seed == 99

    # Cutaway event
    controller.session.run_sync(controller.config)
    controller.handle([Event("cave_screen", "btn_cutaway", True)])
    assert controller.cutaway is True

    # Export event: clicking Export cave opens dialog, then confirming exports
    controller.handle([Event("cave_screen", "btn_export", True)])
    from worldgen.cave.ui.view.windows import SAVE_WORLD
    assert controller._window(SAVE_WORLD) is not None

    from worldgen.ui.view import events as ev
    controller.act(ev.SaveWorld())
    assert controller._window(SAVE_WORLD) is None
    assert (tmp_path / "ctrl_out" / "cave.npz").exists()


def test_cave_screen_widget_validity():
    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=1)
    check_unique_ids(controller.screen.root)

    valid_tones = {"ink", "muted", "error", "ok", "warning", "info"}
    for node in walk(controller.screen.root):
        if isinstance(node, Label):
            assert node.tone in valid_tones


def test_cave_controller_stage_events():
    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=1)

    # 1. Section toggle
    assert ("cavern",) not in controller.opened_sections
    controller.handle([Event("cave_screen", "cavern.header", True)])
    assert ("cavern",) in controller.opened_sections

    # 2. Text commit on clearance
    controller.handle([Event("cave_screen", "cavern.clearance", "27.5")])
    assert controller.recipe.get(("cavern", "clearance")) == 27.5

    # 3. Reset stage
    controller.handle([Event("cave_screen", "stage.cavern.reset", True)])
    assert controller.recipe.is_default("cavern")


def test_cave_session_staleness():
    from worldgen.ui.model.stages import Status

    cfg = parse_config(SMALL_CAVE)
    session = CaveSession(cfg, seed=10)

    # Initially all not run
    statuses = session.statuses(cfg)
    assert all(s == Status.NOT_RUN for s in statuses.values())

    # After run_sync, all done
    session.run_sync()
    statuses = session.statuses(cfg)
    assert all(s in (Status.DONE, Status.SKIPPED) for s in statuses.values())

    # Mutate cavern config -> elevation and erosion stay DONE, cavern and downstream become STALE
    cfg_modified = cfg.model_copy(update={"cavern": cfg.cavern.model_copy(update={"clearance": 40.0})})
    new_statuses = session.statuses(cfg_modified)
    assert new_statuses["elevation"] == Status.DONE
    assert new_statuses["cavern"] == Status.STALE
    assert new_statuses["hydrology"] == Status.STALE
    assert new_statuses["climate"] == Status.STALE
    assert new_statuses["materials"] == Status.STALE


def test_cave_controller_log_window():
    from worldgen.ui.view.windows import LOG

    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=1)

    assert controller._window(LOG) is None

    # Toggle open via event
    controller.handle([Event("cave_screen", "log", True)])
    win = controller._window(LOG)
    assert win is not None
    assert win.title == "Log"

    # Log news check
    controller._say("Hello cave log")
    assert any("Hello cave log" in str(line) for line in controller.log.lines())

    # Clear log via event
    controller.handle([Event("cave_screen", "log.clear", True)])
    assert len(controller.log.lines()) == 0

    # Close log via event
    controller.handle([Event("cave_screen", LOG, True)])
    assert controller._window(LOG) is None


def test_cave_controller_export_dialog(tmp_path: Path):
    from worldgen.cave.ui.view.windows import SAVE_WORLD, save_cave_window, show_cave_format
    from worldgen.ui.view import events as ev

    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=3, out_dir=tmp_path / "cave_export")

    # 1. Test save_cave_window builder
    win = save_cave_window(tmp_path / "test_path", "npz")
    assert win.id == SAVE_WORLD
    assert win.title == "Export the world"
    assert "cave.npz" in win.prompt

    show_cave_format(win, "raw")
    assert "cave.json" in win.prompt

    # 2. Open export dialog in controller
    controller.act(ev.DialogAsked(SAVE_WORLD))
    active_win = controller._window(SAVE_WORLD)
    assert active_win is not None

    # 3. Choose raw format
    controller.act(ev.FormatChosen("raw"))
    assert controller.export_format == "raw"
    assert "cave.json" in active_win.prompt

    # 4. Type new path
    new_out = tmp_path / "raw_cave_export"
    controller.act(ev.WorldPathTyped(str(new_out)))
    assert controller.out_dir == new_out

    # 5. Export raw cave bundle
    controller.session.run_sync(controller.config)
    controller.act(ev.SaveWorld())
    assert controller._window(SAVE_WORLD) is None
    assert (new_out / "cave.json").exists()
    assert (new_out / "cave.yaml").exists()
    assert (new_out / "meta.json").exists()


def test_cave_shader_no_spotlight():
    from worldgen.cave.ui.view.gl.cave_render import CAVE_FS

    fs_source = CAVE_FS.read_text(encoding="utf-8")
    assert "lightCutoff" not in fs_source
    assert "isFps" not in fs_source
    assert "spot" not in fs_source
    assert "atten" not in fs_source




