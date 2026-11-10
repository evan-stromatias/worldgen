"""`TerrainData`: the terrain that stages 1-5 make, in memory.

Vertex grids have (size_cells + 1)² samples; cell grids have size_cells². `slope`, `relief` and
`biome_weights` are derived from the stored arrays, so they are computed on first use and cached
(exported terrain bundles leave them out and recompute them on load, with no difference).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from worldgen.generator.stages import biomes, climate
from worldgen.generator.config import TerrainConfig
from worldgen.generator.grid import to_cells
from worldgen.generator.stages.hydrology import Lake


@dataclass(frozen=True, slots=True)
class TerrainData:
    seed: int
    config: TerrainConfig
    elevation: np.ndarray  # vertex grid, metres
    water: np.ndarray  # vertex grid, water surface height, NaN = dry
    sea: np.ndarray  # vertex grid, bool
    lake_id: np.ndarray  # vertex grid, uint16
    river: np.ndarray  # vertex grid, uint8 strength
    flow_dir: np.ndarray  # vertex grid, uint8 D8 code
    biome: np.ndarray  # cell grid, uint8 index into config.biomes.rules
    color: np.ndarray  # cell grid, uint8 RGB
    temperature: np.ndarray  # cell grid, °C, as the biome rules saw it
    moisture: np.ndarray  # cell grid, 0-1
    sediment: np.ndarray  # cell grid, float32 metres erosion deposited (0 without erosion)
    lakes: list[Lake]
    rivers: list[list[tuple[int, int]]]  # polylines of (x, z) vertex coordinates, source first
    _slope: np.ndarray | None = field(default=None, repr=False)  # cached properties
    _relief: np.ndarray | None = field(default=None, repr=False)
    _biome_weights: np.ndarray | None = field(default=None, repr=False)

    @property
    def slope(self) -> np.ndarray:
        """Cell grid, degrees: the ground's slope, as the biome rules saw it."""
        if self._slope is None:
            object.__setattr__(
                self, "_slope", to_cells(climate.slope_degrees(self.elevation, self.config.world.cell_size)).astype(np.float32)
            )
        return self._slope

    @property
    def relief(self) -> np.ndarray:
        """Cell grid, metres above the lowest ground within biomes.relief_radius."""
        if self._relief is None:
            world = self.config.world
            object.__setattr__(
                self, "_relief",
                biomes.local_relief(self.elevation, world.cell_size, self.config.biomes.relief_radius).astype(np.float32),
            )
        return self._relief

    @property
    def biome_weights(self) -> np.ndarray:
        """(rules, z, x) float32: each rule's share of each cell, summing to one (biomes.soft_weights)."""
        if self._biome_weights is None:
            object.__setattr__(self, "_biome_weights", biomes.soft_weights(self.cell_inputs(), self.config.biomes))
        return self._biome_weights

    def cell_inputs(self) -> biomes.CellInputs:
        """The per-cell values the biome rules test, as the biomes stage built them."""
        world = self.config.world
        cells = to_cells(self.elevation)
        return biomes.CellInputs(
            elevation=cells - world.sea_level,
            temperature=self.temperature.astype(np.float64),
            moisture=self.moisture.astype(np.float64),
            slope=self.slope.astype(np.float64),
            water=biomes.water_kind_cells(self.sea, self.lake_id > 0, self.river > 0),
            sediment=self.sediment.astype(np.float64),
            relief=self.relief.astype(np.float64),
        )

    @property
    def patches_per_side(self) -> int:
        """How many patches of `world.patch_cells` cells tile the terrain's side."""
        return self.config.world.size_cells // self.config.world.patch_cells

    @property
    def patch_shape(self) -> tuple[int, int]:
        """How many patches tile the terrain, along x and along z."""
        return self.patches_per_side, self.patches_per_side

    @classmethod
    def load(cls, directory: Path | str) -> "TerrainData":
        """Read a terrain bundle (a directory written by `export_terrain` or the UI's Export world)."""
        from worldgen.generator.terrain_io import load_terrain

        return load_terrain(directory)

    def patch(self, key: tuple[int, int]):
        """The patch at `key` ((px, pz), 0-based): its share of every grid, and its lakes and rivers."""
        from worldgen.generator.patches import patch

        return patch(self, key)

    def patch_at(self, x: float, z: float):
        """The patch under the world point (x, z), in metres."""
        from worldgen.generator.patches import patch_at

        return patch_at(self, x, z)

    def height_at(self, x: float, z: float) -> float:
        """The ground's height at the world point (x, z), in metres, interpolated as the meshes interpolate it."""
        from worldgen.mesh.overview import sample_grid

        return sample_grid(self.elevation, self.config.world.cell_size, x, z)

    def water_at(self, x: float, z: float) -> float:
        """The water's surface at the world point (x, z), in metres; NaN where dry."""
        from worldgen.mesh.overview import sample_grid

        return sample_grid(self.water, self.config.world.cell_size, x, z)

    def biome_at(self, x: float, z: float) -> str:
        """The biome id of the cell under the world point (x, z), in metres."""
        world = self.config.world
        n = world.size_cells
        cx = min(max(int(x / world.cell_size), 0), n - 1)
        cz = min(max(int(z / world.cell_size), 0), n - 1)
        return self.config.biomes.rules[int(self.biome[cz, cx])].id
