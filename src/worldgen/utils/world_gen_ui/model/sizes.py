"""What the world's grid settings come to in metres, shown next to them as they are typed.

The text is ASCII: raylib's default font has no multiplication sign, approximately-equal sign or superscript two.
"""

from typing import Any

from worldgen.utils.world_gen_ui.model.recipe import FieldPath, Recipe

SIZE_CELLS = ("world", "size_cells")
PATCH_CELLS = ("world", "patch_cells")
CELL_SIZE = ("world", "cell_size")


def size_notes(recipe: Recipe) -> dict[FieldPath, str]:
    """The world's and a patch's side and area, by the field they belong to; left out while a value is not a number."""
    cell = _positive(recipe.get(CELL_SIZE), float)
    size = _positive(recipe.get(SIZE_CELLS), int)
    patch = _positive(recipe.get(PATCH_CELLS), int)
    if cell is None:
        return {}
    notes = {}
    if size is not None:
        notes[SIZE_CELLS] = _square(size * cell)
    if patch is not None:
        notes[PATCH_CELLS] = _square(patch * cell)
    return notes


def _positive(value: Any, kind: type) -> float | None:
    """`value` when it is a number above zero of `kind` (an int counts as a float); None otherwise."""
    kinds = (int, float) if kind is float else (int,)
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        return None
    return value


def _square(side: float) -> str:
    """A square `side` metres across, as "~ 2.0 x 2.0 km = 4.19 km2"."""
    length = f"{side / 1000:.1f} x {side / 1000:.1f} km" if side >= 1000 else f"{side:g} x {side:g} m"
    return f"~ {length} = {(side / 1000) ** 2:.3g} km2"
