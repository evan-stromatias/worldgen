"""Exporting and loading cave bundles in `npz` (Python) and `raw` (other languages) formats.

Matches worldgen conventions: deterministic timestamps, stale file cleanup, little-endian binary arrays.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from worldgen.cave.config import CaveConfig
from worldgen.cave.generator import CaveData
from worldgen.config import VERSION
from worldgen.generator.preview import hillshade, write_npz, write_png

type CaveExportFormat = Literal["npz", "raw"]
CAVE_EXPORT_FORMATS: tuple[CaveExportFormat, ...] = ("npz", "raw")

RECIPE_FILE = "cave.yaml"
ARRAYS_FILE = "cave.npz"
MANIFEST_FILE = "cave.json"
META_FILE = "meta.json"
PREVIEW_FILE = "preview.png"
ARRAYS_DIR = "arrays"

FORMAT, FORMAT_VERSION = "cavegen-raw", 1

RAW_ARRAYS: dict[str, dict] = {
    "floor_elevation": {"dtype": "float64", "grid": "vertex", "units": "m"},
    "ceiling_elevation": {"dtype": "float64", "grid": "vertex", "units": "m"},
    "clearance": {"dtype": "float32", "grid": "cell", "units": "m"},
    "pillar_mask": {"dtype": "uint8", "grid": "cell"},
    "color": {"dtype": "uint8", "grid": "cell", "units": "srgb"},
}

ARRAY_FILES = tuple(f"{ARRAYS_DIR}/{name}.bin" for name in RAW_ARRAYS)
DTYPES = {"uint8": "<u1", "uint16": "<u2", "float32": "<f4", "float64": "<f8"}

HEADER = "# A subterranean cave bundle's recipe and seed, written by worldgen.\n\n"


def bundle_files(format: CaveExportFormat) -> tuple[str, ...]:
    """The files a bundle of `format` holds, as paths relative to its folder."""
    if format == "raw":
        return (MANIFEST_FILE, RECIPE_FILE, META_FILE, *ARRAY_FILES, PREVIEW_FILE)
    return (RECIPE_FILE, ARRAYS_FILE, PREVIEW_FILE)


def stale_files(format: CaveExportFormat) -> tuple[str, ...]:
    """Files written by the other format that must be unlinked."""
    kept = set(bundle_files(format))
    return tuple(
        name for other in CAVE_EXPORT_FORMATS if other != format for name in bundle_files(other) if name not in kept
    )


def render_cave_preview(cave: CaveData) -> np.ndarray:
    """Render a 2D clearance-tinted and hillshaded preview of the cave."""
    cell_size = cave.config.world.cell_size
    shade = hillshade(cave.floor_elevation, cell_size)[..., None]
    base = cave.color.astype(np.float32) * shade

    # Tint by clearance
    cl_norm = np.clip(cave.clearance / max(float(cave.config.cavern.clearance), 1.0), 0.0, 1.5)[..., None]
    tinted = base * (0.8 + 0.4 * cl_norm)

    # Darken solid pillars
    pillars = cave.pillar_mask == 1
    tinted[pillars] = tinted[pillars] * 0.35 + np.array([20, 20, 25], dtype=np.float32) * 0.65

    # Tint submerged pools
    if cave.config.hydrology.enabled:
        wet = ~np.isnan(cave.water)
        from worldgen.generator.grid import to_cells

        wet_cells = to_cells(wet.astype(np.float64)) >= 0.5
        tinted[wet_cells] = tinted[wet_cells] * 0.4 + np.array([30, 70, 110], dtype=np.float32) * 0.6

    return np.clip(np.rint(tinted), 0, 255).astype(np.uint8)


def cave_meta(cave: CaveData) -> dict:
    """Summary statistics for meta.json."""
    cell_size = cave.config.world.cell_size
    cell_area = cell_size * cell_size
    open_cells = cave.pillar_mask == 0
    total_cells = int(cave.pillar_mask.size)
    passable_cells = int(np.sum(open_cells))
    pillar_cells = total_cells - passable_cells

    # Cave volume estimate in cubic metres
    volume_m3 = float(np.sum(cave.clearance[open_cells]) * cell_area)

    return {
        "seed": cave.seed,
        "total_cells": total_cells,
        "passable_cells": passable_cells,
        "pillar_cells": pillar_cells,
        "passable_ratio": float(passable_cells / max(total_cells, 1)),
        "volume_m3": round(volume_m3, 2),
        "min_clearance_m": float(np.min(cave.clearance)),
        "max_clearance_m": float(np.max(cave.clearance)),
        "mean_clearance_m": float(np.mean(cave.clearance[open_cells])) if passable_cells > 0 else 0.0,
    }


def export_cave(cave: CaveData, out_dir: Path | str, format: CaveExportFormat = "npz") -> None:
    """Export `cave` into `out_dir` in either `npz` or `raw` bundle format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in stale_files(format):
        (out_dir / name).unlink(missing_ok=True)
    arrays_dir = out_dir / ARRAYS_DIR
    if format != "raw" and arrays_dir.is_dir() and not any(arrays_dir.iterdir()):
        arrays_dir.rmdir()

    data = {"seed": cave.seed, **cave.config.model_dump(mode="json"), "version": VERSION}
    if not data["client"]:
        del data["client"]
    (out_dir / RECIPE_FILE).write_text(
        HEADER + yaml.dump(data, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8"
    )

    meta = cave_meta(cave)

    if format == "raw":
        (out_dir / ARRAYS_DIR).mkdir(exist_ok=True)
        arrays_meta = {}
        for name, spec in RAW_ARRAYS.items():
            arr = getattr(cave, name)
            rel_path = f"{ARRAYS_DIR}/{name}.bin"
            (out_dir / rel_path).write_bytes(np.ascontiguousarray(arr, dtype=DTYPES[spec["dtype"]]).tobytes())
            arrays_meta[name] = {**spec, "file": rel_path, "shape": list(arr.shape)}

        (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        manifest = {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "version": VERSION,
            "seed": cave.seed,
            "world": cave.config.world.model_dump(mode="json"),
            "client": cave.config.client,
            "cavern": cave.config.cavern.model_dump(mode="json"),
            "materials": [
                {"id": rule.id, "color": list(rule.ground.color)} for rule in cave.config.materials.rules
            ],
            "byte_order": "little",
            "meta": META_FILE,
            "arrays": arrays_meta,
        }
        (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        arrays = {
            "floor_elevation": cave.floor_elevation,
            "ceiling_elevation": cave.ceiling_elevation,
            "clearance": cave.clearance,
            "pillar_mask": cave.pillar_mask,
            "water": cave.water,
            "material": cave.material,
            "color": cave.color,
            "temperature": cave.temperature,
            "moisture": cave.moisture,
            "sediment": cave.sediment,
        }
        write_npz(out_dir / ARRAYS_FILE, arrays)

    preview_img = render_cave_preview(cave)
    write_png(out_dir / PREVIEW_FILE, preview_img)


def load_cave(directory: Path | str) -> CaveData:
    """Load a cave from an `npz` bundle directory."""
    directory = Path(directory)
    recipe_path = directory / RECIPE_FILE
    arrays_path = directory / ARRAYS_FILE

    raw_yaml = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    seed = int(raw_yaml.pop("seed"))
    raw_yaml.pop("version", None)
    config = CaveConfig.model_validate(raw_yaml)

    with np.load(arrays_path) as data:
        return CaveData(
            seed=seed,
            config=config,
            floor_elevation=data["floor_elevation"],
            ceiling_elevation=data["ceiling_elevation"],
            clearance=data["clearance"],
            pillar_mask=data["pillar_mask"],
            water=data["water"],
            material=data["material"],
            color=data["color"],
            temperature=data["temperature"],
            moisture=data["moisture"],
            sediment=data["sediment"],
        )
