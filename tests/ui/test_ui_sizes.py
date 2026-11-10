"""What the world's grid settings come to in metres, shown next to their boxes."""

from worldgen.ui.model.recipe.recipe import Recipe
from worldgen.ui.model.recipe.sizes import PATCH_CELLS, SIZE_CELLS, size_notes


def notes(**world) -> dict:
    return size_notes(Recipe(None, {"world": world}))


def test_world_and_patch_sizes():
    found = notes(size_cells=2048, patch_cells=128, cell_size=1.0)
    assert found[SIZE_CELLS] == "~ 2.0 x 2.0 km = 4.19 km2"
    assert found[PATCH_CELLS] == "~ 128 x 128 m = 0.0164 km2"


def test_cell_size_scales_both():
    found = notes(size_cells=256, patch_cells=64, cell_size=2)
    assert found[SIZE_CELLS] == "~ 512 x 512 m = 0.262 km2"
    assert found[PATCH_CELLS] == "~ 128 x 128 m = 0.0164 km2"


def test_nothing_for_values_being_typed():
    assert notes(size_cells="10x", patch_cells=128, cell_size=1.0) == {PATCH_CELLS: "~ 128 x 128 m = 0.0164 km2"}
    assert notes(size_cells=1024, patch_cells=0, cell_size=1.0).keys() == {SIZE_CELLS}
    assert notes(size_cells=1024, patch_cells=128) == {}
