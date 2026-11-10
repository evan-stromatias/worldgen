"""The terrain bundle on disk: `terrain.yaml` (the seed and the recipe), `terrain.npz` (the arrays,
compressed deterministically) and `preview.png`.

    <terrain_dir>/
      terrain.yaml
      terrain.npz
      preview.png

`biome_weights`, `slope` and `relief` are not stored: they are pure functions of the stored arrays
(`TerrainData` recomputes them on first use, with no difference).
"""

import dataclasses
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from worldgen.config import VERSION
from worldgen.generator import pipeline, preview
from worldgen.generator.config import ConfigError, TerrainConfig, parse_config
from worldgen.generator.stages.hydrology import Lake
from worldgen.generator.terrain import TerrainData

RECIPE_FILE = "terrain.yaml"
ARRAYS_FILE = "terrain.npz"
PREVIEW_FILE = "preview.png"

# What terrain.npz holds; the rest of TerrainData is derived (slope, relief, biome_weights)
ARRAYS = ("elevation", "water", "sea", "lake_id", "river", "flow_dir", "biome", "color", "temperature", "moisture",
          "sediment")

HEADER = "# A terrain bundle's recipe and seed, written by worldgen.\n\n"


def generate_terrain(
    config: TerrainConfig,
    seed: int,
    report: Callable[[str, float], None] = lambda stage, seconds: None,
) -> TerrainData:
    """Run stages 1-5 (elevation, erosion, hydrology, climate, biomes) on `config` with `seed`."""
    return pipeline.generate(config, seed, report)


def export_terrain(terrain: TerrainData, out_dir: Path | str) -> None:
    """Write `terrain` as a terrain bundle under `out_dir`: terrain.yaml, terrain.npz and preview.png.
    terrain.yaml's `version` is this worldgen's version, whatever the loaded recipe's said, and its
    `client` is the recipe's own, verbatim (left out when empty, as it defaults to)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"seed": terrain.seed, **terrain.config.model_dump(mode="json"), "version": VERSION}
    if not data["client"]:
        del data["client"]
    (out_dir / RECIPE_FILE).write_text(HEADER + yaml.dump(data, sort_keys=False, allow_unicode=True, width=110),
                                        encoding="utf-8")
    meta = {"lakes": [dataclasses.asdict(lake) for lake in terrain.lakes], "rivers": terrain.rivers}
    write_arrays(out_dir / ARRAYS_FILE, terrain, meta)
    preview.write_png(out_dir / PREVIEW_FILE, preview.render(terrain))


def write_arrays(path: Path, terrain: TerrainData, meta: dict) -> None:
    """terrain.npz: the arrays, plus the lakes and rivers metadata as JSON."""
    arrays = {name: getattr(terrain, name) for name in ARRAYS}
    arrays["meta"] = np.array(json.dumps(meta))
    preview.write_npz(path, arrays)


def load_terrain(terrain_dir: Path | str) -> TerrainData:
    """Read a terrain bundle back: terrain.yaml and terrain.npz from `terrain_dir`. The derived arrays
    (slope, relief, biome_weights) are recomputed on first use.

    The bundle's `version` is never checked — whether another worldgen version's bundle still works is
    the caller's call — but every error raised here names it."""
    terrain_dir = Path(terrain_dir)
    recipe_path = terrain_dir / RECIPE_FILE
    try:
        data = yaml.safe_load(recipe_path.read_text())
    except OSError as e:
        raise ConfigError(f"cannot read {recipe_path}: {e.strerror}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{recipe_path} is not valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{recipe_path}: expected a mapping of sections at the top level")
    version = data.get("version", "?")  # only for error messages; never checked
    try:
        seed = int(data.pop("seed"))
    except KeyError:
        raise ConfigError(f"{recipe_path}: no `seed`; pass one with --seed, or use a recipe, not a terrain bundle") from None
    config = parse_config(data, str(recipe_path))
    try:
        with np.load(terrain_dir / ARRAYS_FILE) as archive:
            arrays = {name: archive[name] for name in ARRAYS}
            meta = json.loads(str(archive["meta"]))
        lakes = [Lake(d["id"], d["level"], tuple(d["bbox"]), tuple(d["outlet"]) if d["outlet"] else None, d["area"])
                 for d in meta["lakes"]]
        rivers = [[tuple(p) for p in river] for river in meta["rivers"]]
    except (OSError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as e:
        raise ConfigError(f"{terrain_dir}: cannot read the terrain written by worldgen {version}: {e}") from None
    return TerrainData(config=config, seed=seed, lakes=lakes, rivers=rivers, **arrays)
