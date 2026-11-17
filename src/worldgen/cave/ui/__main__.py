"""Generate and inspect subterranean caves with a UI:

    uv run cavegen-ui
    uv run cavegen-ui --seed 42 --run-all
    uv run cavegen-ui --3d --run-all
    uv run cavegen-ui --fps --run-all

Features a 2D clearance heatmap, 3D orbit inspection with cutaway ceiling,
and an interactive 3D FPS camera inside the cavern with player flashlight and collision.
"""

import argparse
import sys
from pathlib import Path

from worldgen.cave.config import DEFAULT_CAVE_RECIPE, ConfigError, load_config
from worldgen.cave.export import CAVE_EXPORT_FORMATS
from worldgen.cave.ui import app
from worldgen.cave.ui.controller import CaveController
from worldgen.ui.model.log import Log, capture


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cavegen-ui", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Cave recipe (YAML) to start from.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--out", type=Path, help="Where Export cave writes.")
    parser.add_argument(
        "--format",
        choices=CAVE_EXPORT_FORMATS,
        default="npz",
        help="Export format (npz for Python, raw for other engines).",
    )
    parser.add_argument("--width", type=int, default=1600, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Window height in pixels.")
    parser.add_argument("--run-all", action="store_true", help="Generate the cave at start.")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="Start in 3D Orbit view.")
    parser.add_argument("--fps", action="store_true", help="Start in 3D FPS camera view.")
    parser.add_argument("--light", action="store_true", help="Start in light theme (dark by default).")
    parser.add_argument("--log", action="store_true", help="Open the log window from the start.")
    parser.add_argument(
        "--export-dialog",
        action="store_true",
        help="Open the Export cave dialog at start (for scripted screenshots).",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a PNG of the window once rendered, then exit.",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must not be negative")

    config_path = args.config or DEFAULT_CAVE_RECIPE
    try:
        config = load_config(config_path)
    except ConfigError as e:
        sys.exit(str(e))

    log = Log()
    with capture(log):
        theme = app.open_window(args.width, args.height, "subterranean cave generator", (900, 500))
        theme.set_dark(not args.light)  # dark by default
        controller = CaveController(config, args.seed, args.out, log=log, theme=theme)

        try:
            controller.export_format = args.format
            if args.fps:
                controller.mode = "3d_fps"
                controller.shown.mode = "3d_fps"
            elif args.three_d:
                controller.mode = "3d_orbit"
                controller.shown.mode = "3d_orbit"

            if args.log:
                controller.toggle_log()

            if args.export_dialog:
                from worldgen.cave.ui.view.windows import SAVE_WORLD
                from worldgen.ui.view import events as ev

                controller.act(ev.DialogAsked(SAVE_WORLD))

            if args.run_all or args.screenshot:
                # Synchronous run for instant screenshot or start
                controller.session.run_sync()
                controller.renderer.upload_meshes(
                    controller.session.mesh_arrays, controller.session.water_mesh_arrays
                )
                controller.renderer.upload_plan(controller.session.preview_image)
                controller._update_metrics()
                if controller.session.cave is not None:
                    pos, yaw, pitch = controller.session.cave.find_spawn_point()
                    controller.fps_camera.position = pos
                    controller.fps_camera.yaw = yaw
                    controller.fps_camera.pitch = pitch

            app.run(controller, theme, args.screenshot)
        finally:
            app.close(controller, theme)


if __name__ == "__main__":
    main()
