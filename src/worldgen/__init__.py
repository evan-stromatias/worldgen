"""Generic procedural terrain generation: elevation, erosion, hydrology, climate and biomes (stages 1-5),
from a YAML recipe and a seed.

    from worldgen import TerrainConfig, generate_terrain, export_terrain, load_terrain

    terrain = generate_terrain(TerrainConfig(), seed=7)
    export_terrain(terrain, "out/island")
    terrain = load_terrain("out/island")

Also `worldgen` (the CLI) and `worldgen-ui` (the generator with a UI).
"""

from worldgen.config import VERSION
from worldgen.generator.config import TerrainConfig
from worldgen.generator.patches import Patch
from worldgen.generator.terrain import TerrainData
from worldgen.generator.terrain_io import export_terrain, generate_terrain, load_terrain

__version__ = VERSION

__all__ = ["Patch", "TerrainConfig", "TerrainData", "VERSION", "export_terrain", "generate_terrain", "load_terrain"]
