"""Generate a subterranean cave from a cave.yaml recipe and a seed:

    uv run cavegen --seed 7
    uv run cavegen --config my_cave.yaml --seed 42 --out caves/limestone
    uv run cavegen --seed 7 --format raw

Writes cave.yaml (recipe and seed), cave.npz (arrays, compressed) and preview.png
to the output directory. With --format raw, writes the bundle for clients in other languages instead:
cave.json (the manifest), cave.yaml, meta.json, arrays/*.bin and preview.png.
With --preview-only, writes only preview.png.
"""

import argparse
import sys
from pathlib import Path

from worldgen.cave.config import DEFAULT_CAVE_RECIPE, ConfigError, load_config
from worldgen.cave.export import (
    CAVE_EXPORT_FORMATS,
    bundle_files,
    cave_meta,
    export_cave,
    render_cave_preview,
)
from worldgen.cave.generator import generate_cave
from worldgen.generator.preview import write_png
from worldgen.generator.terrain_io import listed


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cavegen", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="Cave recipe (YAML; default: bundled default cave recipe).")
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed; the same seed and recipe give the same cave."
    )
    parser.add_argument("--out", type=Path, help="Output directory (default: out/cave-seed-<seed>).")
    parser.add_argument(
        "--format",
        choices=CAVE_EXPORT_FORMATS,
        default="npz",
        help="npz: for Python (load_cave, numpy); raw: JSON and raw binary arrays, for other languages.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Write only preview.png to --out (or current directory), not the full cave bundle.",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must not be negative")

    config_path = args.config or DEFAULT_CAVE_RECIPE
    try:
        config = load_config(config_path)
    except ConfigError as e:
        sys.exit(str(e))

    size = config.world.size_cells
    print(f"generating cave {size}x{size} cells, seed {args.seed}, from {config_path}")
    cave = generate_cave(
        config, args.seed, report=lambda stage, seconds: print(f"  {stage:<10} {seconds:6.2f}s")
    )

    if args.preview_only:
        out = args.out or Path(".")
        out.mkdir(parents=True, exist_ok=True)
        preview_img = render_cave_preview(cave)
        write_png(out / "preview.png", preview_img)
        print(f"wrote {out / 'preview.png'}")
    else:
        out = args.out or Path("out") / f"cave-seed-{args.seed}"
        export_cave(cave, out, args.format)
        meta = cave_meta(cave)
        print(
            f"wrote {out}: {listed(bundle_files(args.format))} "
            f"(volume {meta['volume_m3']:,.0f} m³, passable {meta['passable_ratio']:.1%})"
        )


if __name__ == "__main__":
    main()
