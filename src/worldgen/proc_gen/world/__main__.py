"""Generate a terrain from a terrain.yaml recipe and a seed:

    uv run worldgen --seed 7
    uv run worldgen --config my_terrain.yaml --seed 42 --out terrains/archipelago

Writes terrain.yaml (the recipe and the seed), terrain.npz (the arrays, compressed) and preview.png
to the output directory. With --preview-only, writes only preview.png.
"""

import argparse
import sys
from pathlib import Path

from worldgen.config import DEFAULT_RECIPE
from worldgen.proc_gen.world import preview
from worldgen.proc_gen.world.config import ConfigError, load_config
from worldgen.proc_gen.world.terrain_io import export_terrain, generate_terrain


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="worldgen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Terrain recipe (YAML; default: the bundled default recipe).")
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed; the same seed and recipe give the same terrain."
    )
    parser.add_argument(
        "--out", type=Path, help="Output directory (default: out/seed-<seed>)."
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write only preview.png to --out (or the current directory), not the terrain bundle.",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must not be negative")
    args.config = args.config or DEFAULT_RECIPE

    try:
        config = load_config(args.config)
    except ConfigError as e:
        sys.exit(str(e))

    size = config.world.size_cells
    print(f"generating {size}x{size} cells, seed {args.seed}, from {args.config}")
    terrain = generate_terrain(
        config, args.seed, report=lambda stage, seconds: print(f"  {stage:<10} {seconds:6.2f}s")
    )

    if args.preview_only:
        out = args.out or Path(".")
        out.mkdir(parents=True, exist_ok=True)
        preview.write_png(out / "preview.png", preview.render(terrain))
        print(f"wrote {out / 'preview.png'}")
    else:
        out = args.out or Path("out") / f"seed-{args.seed}"
        export_terrain(terrain, out)
        print(f"wrote {out}: terrain.yaml, terrain.npz ({len(terrain.lakes)} lakes, "
              f"{len(terrain.rivers)} rivers), preview.png")
    side = size * config.world.cell_size / 1000
    land = 1 - float(terrain.sea.mean())  # lakes and rivers count as land here; only the sea does not
    print(f"  world {side:g} x {side:g} km = {side * side:.2f} km², land {side * side * land:.2f} km² ({land:.0%})")


if __name__ == "__main__":
    main()
