"""Tests for CaveData: loading, derived properties, spatial queries, and patch streaming."""

from pathlib import Path

import numpy as np
import pytest

from worldgen.cave.config import parse_config
from worldgen.cave.export import export_cave, load_cave
from worldgen.cave.generator import CaveData, generate_cave
from worldgen.cave.patches import CavePatch, patch, patch_at, patches_per_side
from worldgen.generator.config import ConfigError

SMALL_CAVE_CONFIG = {
    "world": {"size_cells": 32, "patch_cells": 16, "cell_size": 2.0},
    "elevation": {"noise": {"frequency": 0.05}},
    "cavern": {
        "clearance": 15.0,
        "ceiling_mode": "inverted",
        "min_passage": 3.0,
        "perimeter_wall": True,
        "wall_thickness": 1,
    },
    "hydrology": {"enabled": True, "water_table": -1.0},
}


@pytest.fixture
def small_cave() -> CaveData:
    cfg = parse_config(SMALL_CAVE_CONFIG)
    return generate_cave(cfg, seed=42)


def test_load_user_test_export_if_present():
    test_dir = Path("out/cave-test-export-python")
    if not test_dir.is_dir():
        pytest.skip("out/cave-test-export-python is not present")

    cave = CaveData.load(test_dir)
    assert cave.seed == 7
    assert cave.config.world.size_cells == 1024
    assert cave.config.world.patch_cells == 128
    assert cave.config.world.cell_size == 1.0

    # Grid shapes
    assert cave.floor_elevation.shape == (1025, 1025)
    assert cave.ceiling_elevation.shape == (1025, 1025)
    assert cave.clearance.shape == (1024, 1024)
    assert cave.pillar_mask.shape == (1024, 1024)
    assert cave.material.shape == (1024, 1024)
    assert cave.color.shape == (1024, 1024, 3)

    # Aliases and derived properties
    np.testing.assert_array_equal(cave.elevation, cave.floor_elevation)
    np.testing.assert_array_equal(cave.biome, cave.material)
    assert cave.slope.shape == (1024, 1024)
    assert cave.relief.shape == (1024, 1024)
    assert cave.material_weights.shape[1:] == (1024, 1024)

    # Patches
    assert cave.patches_per_side == 8
    assert cave.patch_shape == (8, 8)
    p0 = cave.patch((0, 0))
    assert isinstance(p0, CavePatch)
    assert p0.key == (0, 0)
    assert p0.cells == 128
    assert p0.floor_elevation.shape == (131, 131)

    p_at = cave.patch_at(500.0, 300.0)
    assert p_at.key == (3, 2)

    # Spatial query & spawn
    spawn_pos, yaw, pitch = cave.find_spawn_point()
    assert spawn_pos.shape == (3,)
    assert yaw == 0.0 and pitch == 0.0
    assert cave.floor_at(10.0, 10.0) <= cave.ceiling_at(10.0, 10.0) or cave.clearance_at(10.0, 10.0) < 0


def test_cave_data_roundtrip_and_export_method(small_cave: CaveData, tmp_path: Path):
    out = tmp_path / "bundle"
    small_cave.export(out)
    assert (out / "cave.yaml").exists()
    assert (out / "cave.npz").exists()
    assert (out / "preview.png").exists()

    loaded = CaveData.load(out)
    assert loaded.seed == small_cave.seed
    assert loaded.config == small_cave.config

    for name in (
        "floor_elevation",
        "ceiling_elevation",
        "clearance",
        "pillar_mask",
        "water",
        "material",
        "color",
        "temperature",
        "moisture",
        "sediment",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(small_cave, name))

    # Derived arrays are identical
    np.testing.assert_array_equal(loaded.slope, small_cave.slope)
    np.testing.assert_array_equal(loaded.relief, small_cave.relief)
    np.testing.assert_array_equal(loaded.material_weights, small_cave.material_weights)


def test_cave_patches_tile_and_slice(small_cave: CaveData):
    assert patches_per_side(small_cave) == 2
    assert small_cave.patches_per_side == 2
    assert small_cave.patch_shape == (2, 2)

    p00 = small_cave.patch((0, 0))
    p10 = small_cave.patch((1, 0))
    p01 = small_cave.patch((0, 1))
    p11 = small_cave.patch((1, 1))

    assert p00.origin == (0.0, 0.0)
    assert p10.origin == (32.0, 0.0)  # 16 cells * 2.0 cell_size
    assert p01.origin == (0.0, 32.0)
    assert p11.origin == (32.0, 32.0)

    # Check unpadded vertex grids
    np.testing.assert_array_equal(p00.vertex_floor, small_cave.floor_elevation[0:17, 0:17])
    np.testing.assert_array_equal(p00.vertex_ceiling, small_cave.ceiling_elevation[0:17, 0:17])
    np.testing.assert_array_equal(p00.clearance, small_cave.clearance[0:16, 0:16])
    np.testing.assert_array_equal(p00.pillar_mask, small_cave.pillar_mask[0:16, 0:16])

    # Apron check: inner vertices match neighbours
    # p11 has neighbours at north (row 31) and west (col 31)
    np.testing.assert_array_equal(p11.floor_elevation[1:-1, 1:-1], small_cave.floor_elevation[16:33, 16:33])
    np.testing.assert_array_equal(p11.floor_elevation[0, 1:-1], small_cave.floor_elevation[15, 16:33])
    np.testing.assert_array_equal(p11.floor_elevation[1:-1, 0], small_cave.floor_elevation[16:33, 15])


def test_patch_at_and_bounds_check(small_cave: CaveData):
    assert small_cave.patch_at(10.0, 10.0).key == (0, 0)
    assert small_cave.patch_at(40.0, 10.0).key == (1, 0)
    assert small_cave.patch_at(10.0, 40.0).key == (0, 1)
    assert small_cave.patch_at(40.0, 40.0).key == (1, 1)
    # Edge clamping
    assert patch_at(small_cave, 1000.0, 1000.0).key == (1, 1)
    assert patch_at(small_cave, -100.0, -100.0).key == (0, 0)

    with pytest.raises(IndexError, match="outside"):
        patch(small_cave, (2, 0))


def test_mesh_and_water_mesh_methods(small_cave: CaveData):
    mesh = small_cave.mesh(stride=1)
    assert len(mesh.positions) > 0
    assert len(mesh.indices) > 0
    assert len(mesh.normals) == len(mesh.positions)

    wmesh = small_cave.water_mesh(stride=1)
    if wmesh is not None:
        assert len(wmesh.positions) > 0
        assert len(wmesh.indices) > 0


def test_load_cave_error_handling(tmp_path: Path, small_cave: CaveData):
    # Non-existent
    with pytest.raises(ConfigError, match="cannot read"):
        load_cave(tmp_path / "does_not_exist")

    # Raw export rejected
    raw_dir = tmp_path / "raw_cave"
    export_cave(small_cave, raw_dir, format="raw")
    with pytest.raises(ConfigError, match="raw export"):
        load_cave(raw_dir)

    # Missing seed in yaml
    bad_dir = tmp_path / "bad_cave"
    bad_dir.mkdir()
    (bad_dir / "cave.yaml").write_text("world:\n  size_cells: 16\n")
    with pytest.raises(ConfigError, match="no `seed`"):
        load_cave(bad_dir)
