import numpy as np
import pytest

from worldgen.cave.config import parse_config
from worldgen.cave.generator import generate_cave
from worldgen.cave.mesh import build_cave_mesh, build_cave_water_mesh

CAVE_RECIPE = {
    "world": {"size_cells": 16, "patch_cells": 16, "cell_size": 1.0},
    "elevation": {"noise": {"frequency": 0.05}},
    "cavern": {
        "clearance": 10.0,
        "ceiling_mode": "inverted",
        "min_passage": 2.0,
        "perimeter_wall": True,
        "wall_thickness": 1,
    },
    "hydrology": {"enabled": True, "water_table": 0.0},
}


def test_build_cave_mesh_structure():
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=12)

    mesh = build_cave_mesh(cave, stride=1, include_ceiling=True)
    assert mesh.positions.ndim == 2 and mesh.positions.shape[1] == 3
    assert mesh.normals.shape == mesh.positions.shape
    assert mesh.texcoords.shape == (len(mesh.positions), 2)
    assert mesh.colors.shape == (len(mesh.positions), 4)
    assert mesh.indices.ndim == 2 and mesh.indices.shape[1] == 3

    # Check that indices point within valid vertex range
    assert np.all(mesh.indices >= 0)
    assert np.all(mesh.indices < len(mesh.positions))


def test_mesh_ceiling_cutaway():
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=12)

    mesh_full = build_cave_mesh(cave, include_ceiling=True)
    mesh_no_ceil = build_cave_mesh(cave, include_ceiling=False)

    # Floor only has fewer vertices and triangles
    assert len(mesh_no_ceil.positions) < len(mesh_full.positions)
    assert len(mesh_no_ceil.indices) < len(mesh_full.indices)

    # Cutaway height
    mesh_cut = build_cave_mesh(cave, include_ceiling=True, cutaway_height=5.0)
    assert len(mesh_cut.indices) <= len(mesh_full.indices)


def test_stride_validation():
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=12)

    with pytest.raises(ValueError, match="stride"):
        build_cave_mesh(cave, stride=5)  # 5 does not divide 16


def test_water_mesh_builder():
    cfg = parse_config(CAVE_RECIPE)
    cave = generate_cave(cfg, seed=12)

    water_mesh = build_cave_water_mesh(cave, stride=1)
    if water_mesh is not None:
        assert water_mesh.positions.shape[1] == 3
        assert np.allclose(water_mesh.normals[:, 1], 1.0)  # +y upward
