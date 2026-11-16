"""Writing a generated terrain to disk: a terrain bundle in either format (`terrain_io.bundle_files` names the
files), terrain.yaml with the recipe and the seed among them.

The UI's own recipe text (with its comments) becomes the bundle's terrain.yaml, with the seed added on
top, so what the user edited is what the file says.
"""

from pathlib import Path

from worldgen.generator.terrain import TerrainData
from worldgen.generator.terrain_export import MANIFEST_FILE
from worldgen.generator.terrain_io import ARRAYS_FILE, RECIPE_FILE, ExportFormat, export_terrain


def save_terrain(terrain: TerrainData, recipe_text: str, out: Path, format: ExportFormat = "npz") -> Path:
    """Write `terrain` under `out` as a bundle of `format`; its terrain.yaml is the UI's recipe text with the seed
    on top. A raw bundle's terrain.json is made from the parsed recipe, which that text is, so the two agree."""
    export_terrain(terrain, out, format)
    (out / RECIPE_FILE).write_text(f"seed: {terrain.seed}\n\n{recipe_text}", encoding="utf-8")
    return out


def occupied(folder: Path) -> bool:
    """Whether `folder` exists and has anything in it."""
    return folder.is_dir() and any(folder.iterdir())


def terrain_format(folder: Path) -> ExportFormat | None:
    """The format of the terrain bundle `folder` holds, if it holds one."""
    if not (folder / RECIPE_FILE).is_file():
        return None
    if (folder / ARRAYS_FILE).is_file():
        return "npz"
    if (folder / MANIFEST_FILE).is_file():
        return "raw"
    return None
