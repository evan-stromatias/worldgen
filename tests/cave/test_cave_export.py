import json
from pathlib import Path

import numpy as np

from worldgen.cave.config import parse_config
from worldgen.cave.export import (
    ARRAY_FILES,
    ARRAYS_FILE,
    MANIFEST_FILE,
    META_FILE,
    PREVIEW_FILE,
    RECIPE_FILE,
    export_cave,
    load_cave,
    render_cave_preview,
)
from worldgen.cave.generator import generate_cave

CAVE_RECIPE = {
    "world": {"size_cells": 16, "patch_cells": 16, "cell_size": 1.0},
    "elevation": {"noise": {"frequency": 0.05}},
    "cavern": {
        "clearance": 12.0,
        "ceiling_mode": "inverted",
        "min_passage": 2.5,
        "perimeter_wall": True,
        "wall_thickness": 1,
    },
    "hydrology": {"enabled": True, "water_table": -1.0},
}


def test_export_and_load_npz_roundtrip(tmp_path: Path):
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=7)

    out_dir = tmp_path / "cave_npz"
    export_cave(cave, out_dir, format="npz")

    assert (out_dir / RECIPE_FILE).exists()
    assert (out_dir / ARRAYS_FILE).exists()
    assert (out_dir / PREVIEW_FILE).exists()

    loaded = load_cave(out_dir)
    assert loaded.seed == cave.seed
    assert loaded.config == cave.config
    np.testing.assert_array_equal(loaded.floor_elevation, cave.floor_elevation)
    np.testing.assert_array_equal(loaded.ceiling_elevation, cave.ceiling_elevation)
    np.testing.assert_array_equal(loaded.clearance, cave.clearance)
    np.testing.assert_array_equal(loaded.pillar_mask, cave.pillar_mask)
    np.testing.assert_array_equal(loaded.color, cave.color)


def test_export_raw_format(tmp_path: Path):
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=7)

    out_dir = tmp_path / "cave_raw"
    export_cave(cave, out_dir, format="raw")

    assert (out_dir / MANIFEST_FILE).exists()
    assert (out_dir / RECIPE_FILE).exists()
    assert (out_dir / META_FILE).exists()
    assert (out_dir / PREVIEW_FILE).exists()

    for rel in ARRAY_FILES:
        assert (out_dir / rel).exists()

    manifest = json.loads((out_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["format"] == "cavegen-raw"
    assert manifest["seed"] == 7
    assert "floor_elevation" in manifest["arrays"]

    meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    assert "volume_m3" in meta
    assert "passable_ratio" in meta


def test_stale_files_cleanup_on_reexport(tmp_path: Path):
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=7)
    out_dir = tmp_path / "switch_format"

    # Export raw first
    export_cave(cave, out_dir, format="raw")
    assert (out_dir / MANIFEST_FILE).exists()
    assert not (out_dir / ARRAYS_FILE).exists()

    # Re-export as npz
    export_cave(cave, out_dir, format="npz")
    assert (out_dir / ARRAYS_FILE).exists()
    assert not (out_dir / MANIFEST_FILE).exists()
    assert not (out_dir / META_FILE).exists()


def test_render_cave_preview():
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=7)

    img = render_cave_preview(cave)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.shape[:2] == (16, 16)
    assert img.dtype == np.uint8
