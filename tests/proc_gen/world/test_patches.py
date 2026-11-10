"""Reading a terrain in patches: exact slices of its grids, with the apron for meshing."""

import numpy as np
import pytest

from worldgen.proc_gen.world.patches import patch, patch_at, patches_per_side


def test_the_patches_tile_the_terrain(small_terrain):
    assert patches_per_side(small_terrain) == 4  # 128 cells in 32-cell patches
    assert small_terrain.patches_per_side == 4
    patches = [patch(small_terrain, (px, pz)) for pz in range(4) for px in range(4)]
    assert [p.key for p in patches] == [(px, pz) for pz in range(4) for px in range(4)]


def test_a_patch_is_an_exact_slice(small_terrain):
    p = patch(small_terrain, (1, 2))
    assert p.cells == 32 and p.cell_size == 1.0 and p.origin == (32.0, 64.0)
    np.testing.assert_array_equal(p.vertex_height, small_terrain.elevation[64:97, 32:65])
    np.testing.assert_array_equal(p.water, small_terrain.water[64:97, 32:65])
    np.testing.assert_array_equal(p.biome, small_terrain.biome[64:96, 32:64])
    np.testing.assert_array_equal(p.color, small_terrain.color[64:96, 32:64])
    np.testing.assert_array_equal(p.temperature, small_terrain.temperature[64:96, 32:64])


def test_the_apron_matches_the_neighbours_or_the_edge(small_terrain):
    p = patch(small_terrain, (1, 1))
    assert p.elevation.shape == (35, 35)
    # The apron is the neighbours' edge rows: the patch's own vertices sit inside it
    np.testing.assert_array_equal(p.elevation[1:-1, 1:-1], small_terrain.elevation[32:65, 32:65])
    np.testing.assert_array_equal(p.elevation[0, 1:-1], small_terrain.elevation[31, 32:65])
    edge = patch(small_terrain, (0, 0))
    np.testing.assert_array_equal(edge.elevation[0, :], edge.elevation[1, :])  # the rim repeats at the world's edge
    np.testing.assert_array_equal(edge.elevation[:, 0], edge.elevation[:, 1])


def test_patch_at_finds_the_patch_under_a_point(small_terrain):
    assert patch_at(small_terrain, 40.0, 70.0).key == (1, 2)
    assert patch_at(small_terrain, 0.0, 0.0).key == (0, 0)
    assert patch_at(small_terrain, 127.5, 127.5).key == (3, 3)  # the world edge clamps to the last patch
    assert small_terrain.patch_at(40.0, 70.0).key == (1, 2)
    with pytest.raises(IndexError, match="outside"):
        patch(small_terrain, (4, 0))


def test_a_patch_knows_its_lakes_and_rivers(small_terrain):
    found_lakes, found_rivers = set(), 0
    for pz in range(4):
        for px in range(4):
            p = patch(small_terrain, (px, pz))
            found_lakes.update(lake.id for lake in p.lakes)
            found_rivers += len(p.rivers)
            for river in p.rivers:  # only rivers that pass through
                assert any(32 * px <= x <= 32 * (px + 1) and 32 * pz <= z <= 32 * (pz + 1) for x, z in river)
    assert found_lakes == {lake.id for lake in small_terrain.lakes}
    assert found_rivers >= len(small_terrain.rivers)  # a river crossing a border counts in both patches


def test_a_patch_meshes(small_terrain):
    p = patch(small_terrain, (2, 1))
    mesh = p.mesh(skirt_depth=3.0)
    assert mesh.key == (2, 1) and mesh.terrain.positions.size
    assert mesh.color.shape == (32, 32, 3)
    assert (mesh.water is not None) == np.isfinite(p.water).any()
    # A neighbouring patch shares the edge vertices: no seam
    neighbour = patch(small_terrain, (3, 1))
    np.testing.assert_array_equal(p.vertex_height[:, -1], neighbour.vertex_height[:, 0])
