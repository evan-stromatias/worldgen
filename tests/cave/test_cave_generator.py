import numpy as np
import pytest

from worldgen.cave.config import parse_config
from worldgen.cave.generator import generate_cave
from worldgen.generator.grid import to_cells

SMALL_CAVE_RECIPE = {
    "world": {"size_cells": 32, "patch_cells": 16, "cell_size": 2.0},
    "elevation": {
        "max_height": 20.0,
        "noise": {"frequency": 0.05, "octaves": 3},
    },
    "cavern": {
        "clearance": 12.0,
        "ceiling_mode": "inverted",
        "ceiling_scale": 1.0,
        "min_passage": 3.0,
        "perimeter_wall": True,
        "wall_thickness": 2,
    },
    "hydrology": {
        "enabled": True,
        "water_table": -2.0,
    },
}


def test_cave_generation_determinism():
    cfg = parse_config(SMALL_CAVE_RECIPE)
    cave1 = generate_cave(cfg, seed=42)
    cave2 = generate_cave(cfg, seed=42)

    np.testing.assert_array_equal(cave1.floor_elevation, cave2.floor_elevation)
    np.testing.assert_array_equal(cave1.ceiling_elevation, cave2.ceiling_elevation)
    np.testing.assert_array_equal(cave1.clearance, cave2.clearance)
    np.testing.assert_array_equal(cave1.pillar_mask, cave2.pillar_mask)
    np.testing.assert_array_equal(cave1.color, cave2.color)


def test_rotated_ceiling_math():
    cfg = parse_config(SMALL_CAVE_RECIPE)
    cave = generate_cave(cfg, seed=17)

    floor = cave.floor_elevation
    h_rot = np.rot90(floor, 2)
    k = floor.shape[0]

    # Check 180 degree rotation exactness
    for z in range(k):
        for x in range(k):
            assert h_rot[z, x] == floor[k - 1 - z, k - 1 - x]

    # Inverted relief: h_base + clearance + (h_base - h_rot) * s_c
    h_base = float(np.mean(floor))
    expected_ceiling = h_base + cfg.cavern.clearance + (h_base - h_rot) * cfg.cavern.ceiling_scale
    np.testing.assert_allclose(cave.ceiling_elevation, expected_ceiling, rtol=1e-5)


def test_clearance_and_pillar_mask():
    cfg = parse_config(SMALL_CAVE_RECIPE)
    cave = generate_cave(cfg, seed=17)

    # Clearance on cell grid is to_cells(ceiling - floor)
    vertex_clearance = cave.ceiling_elevation - cave.floor_elevation
    expected_cells = to_cells(vertex_clearance).astype(np.float32)
    np.testing.assert_allclose(cave.clearance, expected_cells, rtol=1e-5)

    # Perimeter wall thickness = 2 cells
    t = cfg.cavern.wall_thickness
    assert np.all(cave.pillar_mask[:t, :] == 1)
    assert np.all(cave.pillar_mask[-t:, :] == 1)
    assert np.all(cave.pillar_mask[:, :t] == 1)
    assert np.all(cave.pillar_mask[:, -t:] == 1)

    # Anywhere clearance < min_passage must be a pillar
    interior_low_clearance = (cave.clearance < cfg.cavern.min_passage)
    assert np.all(cave.pillar_mask[interior_low_clearance] == 1)


def test_direct_offset_mode():
    data = dict(SMALL_CAVE_RECIPE)
    data["cavern"] = {
        "clearance": 15.0,
        "ceiling_mode": "offset",
        "min_passage": 2.0,
        "perimeter_wall": False,
    }
    cfg = parse_config(data)
    cave = generate_cave(cfg, seed=9)

    h_rot = np.rot90(cave.floor_elevation, 2)
    expected_ceiling = h_rot + 15.0
    np.testing.assert_allclose(cave.ceiling_elevation, expected_ceiling, rtol=1e-5)


def test_cave_sampling_and_navigation():
    cfg = parse_config(SMALL_CAVE_RECIPE)
    cave = generate_cave(cfg, seed=42)

    # Spawn point
    pos, yaw, pitch = cave.find_spawn_point()
    assert len(pos) == 3
    assert pos[1] >= cave.floor_at(pos[0], pos[2])
    assert cave.is_navigable(pos[0], pos[2])

    hf = cave.floor_at(pos[0], pos[2])
    hc = cave.ceiling_at(pos[0], pos[2])
    assert hc > hf
    assert cave.clearance_at(pos[0], pos[2]) == pytest.approx(hc - hf)
    assert cave.clearance_at(pos[0], pos[2]) >= cfg.cavern.min_passage

    # Out of bounds
    assert not cave.is_navigable(-10.0, -10.0)
    assert not cave.is_navigable(1000.0, 1000.0)


def test_derived_properties():
    cfg = parse_config(SMALL_CAVE_RECIPE)
    cave = generate_cave(cfg, seed=42)

    assert cave.slope.shape == (32, 32)
    assert cave.relief.shape == (32, 32)
    assert cave.material_weights.shape[0] == len(cfg.materials.rules)
    assert np.allclose(np.sum(cave.material_weights, axis=0), 1.0)
