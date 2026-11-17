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
    controller.session.run_sync()
    controller.handle([Event("cave_screen", "btn_cutaway", True)])
    assert controller.cutaway is True

    # Export event
    controller.handle([Event("cave_screen", "btn_export", True)])
    assert (tmp_path / "ctrl_out" / "cave.npz").exists()


def test_cave_screen_widget_validity():
    cfg = parse_config(SMALL_CAVE)
    controller = CaveController(cfg, seed=1)
    check_unique_ids(controller.screen.root)

    valid_tones = {"ink", "muted", "error", "ok", "warning", "info"}
    for node in walk(controller.screen.root):
        if isinstance(node, Label):
            assert node.tone in valid_tones

