"""Writing a generated terrain to disk: a terrain bundle (terrain.yaml with the recipe and the seed,
terrain.npz, preview.png).

The UI's own recipe text (with its comments) becomes the bundle's terrain.yaml, with the seed added on
top, so what the user edited is what the file says.
"""

from pathlib import Path

from worldgen.proc_gen.world.terrain import TerrainData
from worldgen.proc_gen.world.terrain_io import ARRAYS_FILE, RECIPE_FILE, export_terrain


def save_terrain(terrain: TerrainData, recipe_text: str, out: Path) -> Path:
    """Write `terrain` under `out`; the bundle's terrain.yaml is the UI's recipe text with the seed on top."""
    export_terrain(terrain, out)
    (out / RECIPE_FILE).write_text(f"seed: {terrain.seed}\n\n{recipe_text}", encoding="utf-8")
    return out


def occupied(folder: Path) -> bool:
    """Whether `folder` exists and has anything in it."""
    return folder.is_dir() and any(folder.iterdir())


def is_terrain(folder: Path) -> bool:
    """Whether `folder` holds an exported terrain bundle."""
    return (folder / RECIPE_FILE).is_file() and (folder / ARRAYS_FILE).is_file()
