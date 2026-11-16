"""The terrain bundle on disk, in one of two formats.

- `npz` (the default, for Python): `terrain.yaml` (the seed and the recipe), `terrain.npz` (the arrays,
  compressed deterministically) and `preview.png`. `load_terrain` reads it back. `biome_weights`, `slope`
  and `relief` are not stored: they are pure functions of the stored arrays (`TerrainData` recomputes them
  on first use, with no difference).
- `raw` (for clients in other languages): `terrain.json`, `terrain.yaml`, `meta.json`, `arrays/*.bin` and
  `preview.png` (`terrain_export`). worldgen only writes it.
"""

import dataclasses
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from worldgen.config import VERSION
from worldgen.generator import pipeline, preview
from worldgen.generator.config import ConfigError, TerrainConfig, parse_config
from worldgen.generator.stages.hydrology import Lake
from worldgen.generator.terrain import TerrainData
from worldgen.generator.terrain_export import ARRAY_FILES, ARRAYS_DIR, MANIFEST_FILE, META_FILE, write_raw

type ExportFormat = Literal["npz", "raw"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("npz", "raw")

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


def bundle_files(format: ExportFormat) -> tuple[str, ...]:
    """The files a bundle of `format` holds, as paths relative to its folder."""
    if format == "raw":
        return (MANIFEST_FILE, RECIPE_FILE, META_FILE, *ARRAY_FILES, PREVIEW_FILE)
    return (RECIPE_FILE, ARRAYS_FILE, PREVIEW_FILE)


def listed(files: tuple[str, ...]) -> str:
    """Bundle files for a message: "terrain.json, meta.json, arrays/*.bin", the array files as one entry."""
    names: list[str] = []
    for name in files:
        shown = f"{ARRAYS_DIR}/*.bin" if name.startswith(f"{ARRAYS_DIR}/") else name
        if shown not in names:
            names.append(shown)
    return ", ".join(names)


def stale_files(format: ExportFormat) -> tuple[str, ...]:
    """The files only the other format writes: exporting `format` removes them, so a folder never holds both."""
    kept = set(bundle_files(format))
    return tuple(name for other in EXPORT_FORMATS if other != format for name in bundle_files(other) if name not in kept)


def export_terrain(terrain: TerrainData, out_dir: Path | str, format: ExportFormat = "npz") -> None:
    """Write `terrain` as a terrain bundle of `format` under `out_dir` (`bundle_files` names the files).

    - terrain.yaml's `version` is this worldgen's version, whatever the loaded recipe's said, and its
      `client` is the recipe's own, verbatim (left out when empty, as it defaults to).
    - The other format's files are removed first, by name and nothing else (`stale_files`), so the folder
      never holds two terrains that disagree."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in stale_files(format):
        (out_dir / name).unlink(missing_ok=True)
    arrays_dir = out_dir / ARRAYS_DIR
    if format != "raw" and arrays_dir.is_dir() and not any(arrays_dir.iterdir()):
        arrays_dir.rmdir()
    data = {"seed": terrain.seed, **terrain.config.model_dump(mode="json"), "version": VERSION}
    if not data["client"]:
        del data["client"]
    (out_dir / RECIPE_FILE).write_text(HEADER + yaml.dump(data, sort_keys=False, allow_unicode=True, width=110),
                                        encoding="utf-8")
    if format == "raw":
        write_raw(out_dir, terrain, bundle_meta(terrain))
    else:
        write_arrays(out_dir / ARRAYS_FILE, terrain, bundle_meta(terrain))
    preview.write_png(out_dir / PREVIEW_FILE, preview.render(terrain))


def bundle_meta(terrain: TerrainData) -> dict:
    """The lakes and rivers, as JSON-ready data: terrain.npz's `meta` entry, and the raw bundle's meta.json."""
    return {"lakes": [dataclasses.asdict(lake) for lake in terrain.lakes], "rivers": terrain.rivers}


def write_arrays(path: Path, terrain: TerrainData, meta: dict) -> None:
    """terrain.npz: the arrays, plus the lakes and rivers metadata as JSON."""
    arrays = {name: getattr(terrain, name) for name in ARRAYS}
    arrays["meta"] = np.array(json.dumps(meta))
    preview.write_npz(path, arrays)


def load_terrain(terrain_dir: Path | str) -> TerrainData:
    """Read an npz terrain bundle back: terrain.yaml and terrain.npz from `terrain_dir`. The derived arrays
    (slope, relief, biome_weights) are recomputed on first use. A raw bundle is refused: it is for other
    languages.

    The bundle's `version` is never checked — whether another worldgen version's bundle still works is
    the caller's call — but every error raised here names it."""
    terrain_dir = Path(terrain_dir)
    if (terrain_dir / MANIFEST_FILE).is_file() and not (terrain_dir / ARRAYS_FILE).exists():
        raise ConfigError(f"{terrain_dir} is a raw export, for clients in other languages; worldgen reads only npz "
                          "bundles (export with --format npz)")
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
