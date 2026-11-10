"""Generate a terrain with a UI: the recipe's sections as forms on the right, the terrain on the left.

    uv run worldgen-ui
    uv run worldgen-ui --config my_terrain.yaml --seed 3

Without --config the recipe starts at the defaults of its models, with no file until Save as writes
one. Each stage has its own Run button. The window is drawn with rayscreen.
"""

import argparse
import sys
from pathlib import Path

from worldgen.generator.config import ConfigError
from worldgen.ui.model.log import Log, capture
from worldgen.ui.model.recipe.recipe import Recipe
from worldgen.ui.view.events import RunAll


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="worldgen-ui", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Terrain recipe (YAML) to start from; the defaults without it.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed; the same seed and recipe give the same terrain.")
    parser.add_argument(
        "--out", type=Path, help="Where Export world writes (default: a new folder in the system's temporary folder)."
    )
    parser.add_argument("--width", type=int, default=1600, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=900, help="Window height in pixels.")
    parser.add_argument("--run-all", action="store_true", help="Run every stage at start, as Run all does.")
    parser.add_argument(
        "--layer", help="Show this map layer once the run settles (with --run-all), e.g. --layer \"land mask\"."
    )
    parser.add_argument("--log", action="store_true", help="Open the log window from the start.")
    parser.add_argument("--3d", dest="three_d", action="store_true", help="Show the map in 3D from the start.")
    parser.add_argument("--light", action="store_true", help="Start in light mode, not dark (the button at the top right switches).")
    parser.add_argument(
        "--screenshot", type=Path, help="Save a PNG of the window once nothing is running and the map is drawn, then exit."
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must not be negative")

    try:
        recipe = Recipe.load(args.config) if args.config else Recipe.defaults()
    except ConfigError as e:
        sys.exit(str(e))
    if args.layer is not None:
        from worldgen.ui.model.map.layers import LAYERS

        if args.layer not in LAYERS:
            parser.error(f"--layer must be one of: {', '.join(LAYERS)}")

    from worldgen.ui import app  # opens no window until asked
    from worldgen.ui.controller import Controller

    log = Log()
    with capture(log):  # what the stages print shows in the log window too
        theme = app.open_window(args.width, args.height, "terrain generator", (900, 500))
        theme.set_dark(not args.light)  # dark by default
        controller = Controller(recipe, args.seed, args.out, log=log, theme=theme)
        try:
            if recipe.seed_hint is not None and args.seed == parser.get_default("seed"):
                controller.session.seed = recipe.seed_hint  # a terrain bundle's seed, unless --seed was given
            controller.three_d = args.three_d
            controller.pending_layer = args.layer
            if args.log:
                controller.toggle_log()
            if args.run_all:
                controller.act(RunAll())
            app.run(controller, theme, args.screenshot)
        finally:
            app.close(controller, theme)

if __name__ == "__main__":
    main()
