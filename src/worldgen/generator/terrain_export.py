"""The raw terrain bundle, for clients outside Python: a JSON manifest, the lakes and rivers as JSON and one raw
little-endian file per array, readable with a JSON parser and a file read.

    <terrain_dir>/
      terrain.json      # the manifest: the seed, the world, the biomes, and each array's file, dtype and shape
      terrain.yaml      # the recipe and the seed, as in the npz bundle (written by `terrain_io.export_terrain`)
      meta.json         # lakes and rivers
      arrays/<name>.bin # row-major [z][x], little-endian, no header
      preview.png

- worldgen writes it and never reads it back: Python clients read the npz bundle, so there is one loader.
- The derived arrays (slope, relief, biome_weights) are always written: porting `np.gradient`'s edges and the
  biome rules' soft weights to another language is harder than reading a file.
"""

import json
from pathlib import Path

import numpy as np

from worldgen.config import VERSION
from worldgen.generator.stages.hydrology import D8
from worldgen.generator.terrain import TerrainData

MANIFEST_FILE = "terrain.json"
META_FILE = "meta.json"
ARRAYS_DIR = "arrays"
FORMAT, FORMAT_VERSION = "worldgen-raw", 1  # the layout's own version: bumped only when the raw layout changes

# Each manifest dtype as numpy writes it: little-endian whatever the host
DTYPES = {"uint8": "<u1", "uint16": "<u2", "float32": "<f4", "float64": "<f8"}

# What terrain.json says about each array, besides its file and shape; the dtype is the one written
RAW_ARRAYS: dict[str, dict] = {
    "elevation": {"dtype": "float64", "grid": "vertex", "units": "m"},
    "water": {"dtype": "float64", "grid": "vertex", "units": "m", "nan": "dry"},
    "sea": {"dtype": "uint8", "grid": "vertex"},  # bool, as 0 or 1
    "lake_id": {"dtype": "uint16", "grid": "vertex", "nodata": 0},
    "river": {"dtype": "uint8", "grid": "vertex", "nodata": 0},
    "flow_dir": {"dtype": "uint8", "grid": "vertex", "nodata": 255},
    "biome": {"dtype": "uint8", "grid": "cell"},
    "color": {"dtype": "uint8", "grid": "cell", "units": "srgb"},
    "temperature": {"dtype": "float32", "grid": "cell", "units": "degC"},
    "moisture": {"dtype": "float32", "grid": "cell", "range": [0, 1]},
    "sediment": {"dtype": "float32", "grid": "cell", "units": "m"},
    "slope": {"dtype": "float32", "grid": "cell", "units": "deg"},  # derived
    "relief": {"dtype": "float32", "grid": "cell", "units": "m"},
    "biome_weights": {"dtype": "float32", "grid": "cell", "range": [0, 1]},
}

# The array files, as paths relative to the bundle (`terrain_io.bundle_files` lists the rest)
ARRAY_FILES = tuple(f"{ARRAYS_DIR}/{name}.bin" for name in RAW_ARRAYS)


def write_raw(out_dir: Path, terrain: TerrainData, meta: dict) -> None:
    """terrain.json, meta.json and arrays/: everything of the raw bundle but terrain.yaml and preview.png.
    `meta` is the lakes and rivers, as the npz bundle stores them."""
    (out_dir / ARRAYS_DIR).mkdir(exist_ok=True)
    arrays = {}
    for name, spec in RAW_ARRAYS.items():
        array = getattr(terrain, name)
        path = f"{ARRAYS_DIR}/{name}.bin"
        (out_dir / path).write_bytes(np.ascontiguousarray(array, dtype=DTYPES[spec["dtype"]]).tobytes())
        arrays[name] = {**spec, "file": path, "shape": list(array.shape)}
    (out_dir / META_FILE).write_text(_json(meta), encoding="utf-8")
    (out_dir / MANIFEST_FILE).write_text(_json(manifest(terrain, arrays)), encoding="utf-8")


def manifest(terrain: TerrainData, arrays: dict[str, dict]) -> dict:
    """terrain.json: what a client needs to read the bundle without a YAML parser."""
    recipe = terrain.config.model_dump(mode="json")
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "version": VERSION,
        "seed": terrain.seed,
        "world": recipe["world"],
        "client": recipe["client"],
        "biomes": [{"id": rule["id"], "color": rule["ground"]["color"]} for rule in recipe["biomes"]["rules"]],
        "d8": [[dx, dz] for dz, dx in D8],  # hydrology's D8 is (dz, dx), as numpy indexes
        "byte_order": "little",
        "meta": META_FILE,
        "arrays": arrays,
    }


def _json(data: dict) -> str:
    """Sorted keys and a final newline, so the same terrain gives the same bytes; no NaN, which JSON lacks."""
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
