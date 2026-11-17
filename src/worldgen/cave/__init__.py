"""Procedural subterranean cave generator: dual-shell heightmaps, 180° rotated ceiling,
pillars, speleological materials and interactive FPS camera inspection.

    from worldgen.cave import CaveConfig, CaveData, generate_cave, export_cave, load_cave

    cave = generate_cave(CaveConfig(), seed=7)
    export_cave(cave, "out/cavern")
    cave = CaveData.load("out/cavern")

Also `cavegen` (the CLI) and `cavegen-ui` (the generator with a UI).
"""

from worldgen.cave.config import CaveConfig, load_config, parse_config
from worldgen.cave.export import CAVE_EXPORT_FORMATS, CaveExportFormat, export_cave, load_cave
from worldgen.cave.generator import CaveData, generate_cave
from worldgen.cave.mesh import build_cave_mesh, build_cave_water_mesh
from worldgen.cave.patches import CavePatch, patch, patch_at

__all__ = [
    "CAVE_EXPORT_FORMATS",
    "CaveConfig",
    "CaveData",
    "CaveExportFormat",
    "CavePatch",
    "build_cave_mesh",
    "build_cave_water_mesh",
    "export_cave",
    "generate_cave",
    "load_cave",
    "load_config",
    "parse_config",
    "patch",
    "patch_at",
]
