"""Subterranean cave generator: floor heightmap, 180°-rotated ceiling, clearance,
pillars, hydrology, climate and speleological materials.

Deterministic across runs given the same recipe and seed.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from worldgen.cave import pipeline
from worldgen.cave.config import CaveConfig
from worldgen.generator.grid import to_cells
from worldgen.generator.stages import biomes, climate

if TYPE_CHECKING:
    from worldgen.cave.export import CaveExportFormat
    from worldgen.cave.patches import CavePatch
    from worldgen.mesh.builders import MeshArrays

# RNG stream indices for generator stages
STAGE_ELEVATION = 1
STAGE_EROSION = 2
STAGE_MATERIALS = 3


@dataclass(frozen=True, slots=True)
class CaveData:
    """The subterranean cave data in memory.

    Vertex grids have (size_cells + 1)² samples; cell grids have size_cells².
    """

    seed: int
    config: CaveConfig
    floor_elevation: np.ndarray  # vertex grid, float64 metres
    ceiling_elevation: np.ndarray  # vertex grid, float64 metres
    clearance: np.ndarray  # cell grid, float32 metres
    pillar_mask: np.ndarray  # cell grid, uint8: 1 = solid pillar/rock, 0 = open cavern
    water: np.ndarray  # vertex grid, float64 metres (water surface; NaN = dry)
    material: np.ndarray  # cell grid, uint8 index into config.materials.rules
    color: np.ndarray  # cell grid, uint8 RGB
    temperature: np.ndarray  # cell grid, float32 °C
    moisture: np.ndarray  # cell grid, float32 0-1
    sediment: np.ndarray  # cell grid, float32 metres
    _slope: np.ndarray | None = field(default=None, repr=False)
    _relief: np.ndarray | None = field(default=None, repr=False)
    _material_weights: np.ndarray | None = field(default=None, repr=False)

    @property
    def slope(self) -> np.ndarray:
        """Cell grid, degrees: the cave floor slope."""
        if self._slope is None:
            cell_size = self.config.world.cell_size
            object.__setattr__(
                self, "_slope", to_cells(climate.slope_degrees(self.floor_elevation, cell_size)).astype(np.float32)
            )
        return self._slope

    @property
    def relief(self) -> np.ndarray:
        """Cell grid, metres above the lowest floor within materials.relief_radius."""
        if self._relief is None:
            world = self.config.world
            object.__setattr__(
                self,
                "_relief",
                biomes.local_relief(self.floor_elevation, world.cell_size, self.config.materials.relief_radius).astype(
                    np.float32
                ),
            )
        return self._relief

    @property
    def material_weights(self) -> np.ndarray:
        """(rules, z, x) float32: each material rule's share of each cell."""
        if self._material_weights is None:
            object.__setattr__(self, "_material_weights", biomes.soft_weights(self.cell_inputs(), self.config.materials))
        return self._material_weights

    def cell_inputs(self) -> biomes.CellInputs:
        """Per-cell values evaluated by the material rules."""
        world = self.config.world
        cells = to_cells(self.floor_elevation)
        wet_vertex = ~np.isnan(self.water)
        water_kind = biomes.water_kind_cells(
            np.zeros_like(wet_vertex, bool),
            wet_vertex,
            np.zeros_like(wet_vertex, bool),
        )
        return biomes.CellInputs(
            elevation=cells - world.sea_level,
            temperature=self.temperature.astype(np.float64),
            moisture=self.moisture.astype(np.float64),
            slope=self.slope.astype(np.float64),
            water=water_kind,
            sediment=self.sediment.astype(np.float64),
            relief=self.relief.astype(np.float64),
        )

    def floor_at(self, x: float, z: float) -> float:
        """Bilinear height of the cave floor at world coordinates (x, z)."""
        return _sample_grid(self.floor_elevation, self.config.world.cell_size, x, z)

    def ceiling_at(self, x: float, z: float) -> float:
        """Bilinear height of the cave ceiling at world coordinates (x, z)."""
        return _sample_grid(self.ceiling_elevation, self.config.world.cell_size, x, z)

    def clearance_at(self, x: float, z: float) -> float:
        """Vertical headroom between ceiling and floor at world coordinates (x, z)."""
        return self.ceiling_at(x, z) - self.floor_at(x, z)

    def is_navigable(self, x: float, z: float) -> bool:
        """True if (x, z) is inside world bounds, outside pillars, and has clearance >= min_passage."""
        cell_size = self.config.world.cell_size
        size_cells = self.config.world.size_cells
        cx = int(x / cell_size)
        cz = int(z / cell_size)
        if not (0 <= cx < size_cells and 0 <= cz < size_cells):
            return False
        if self.pillar_mask[cz, cx] != 0:
            return False
        return self.clearance[cz, cx] >= self.config.cavern.min_passage

    def find_spawn_point(self) -> tuple[np.ndarray, float, float]:
        """Find a safe spawn position (x, y, z), yaw, and pitch inside the cavern."""
        cell_size = self.config.world.cell_size
        open_mask = self.pillar_mask == 0
        if not np.any(open_mask):
            mid = (self.config.world.size_cells * cell_size) / 2.0
            return np.array([mid, self.floor_at(mid, mid) + 1.7, mid], dtype=np.float64), 0.0, 0.0

        scores = self.clearance.copy()
        scores[~open_mask] = -1.0
        cz, cx = np.unravel_index(np.argmax(scores), scores.shape)
        x = (cx + 0.5) * cell_size
        z = (cz + 0.5) * cell_size
        y = self.floor_at(x, z) + 1.7
        return np.array([x, y, z], dtype=np.float64), 0.0, 0.0

    @property
    def elevation(self) -> np.ndarray:
        """Alias for floor_elevation (metres, vertex grid)."""
        return self.floor_elevation

    @property
    def biome(self) -> np.ndarray:
        """Alias for material (cell grid index into materials rules)."""
        return self.material

    @property
    def patches_per_side(self) -> int:
        """How many patches of `world.patch_cells` cells tile the cave's side."""
        return self.config.world.size_cells // self.config.world.patch_cells

    @property
    def patch_shape(self) -> tuple[int, int]:
        """How many patches tile the cave, along x and along z."""
        return self.patches_per_side, self.patches_per_side

    @classmethod
    def load(cls, directory: Path | str) -> "CaveData":
        """Read a cave bundle (a directory written by `export_cave` or the UI's Export cave)."""
        from worldgen.cave.export import load_cave

        return load_cave(directory)

    def export(self, out_dir: Path | str, format: "CaveExportFormat" = "npz") -> None:
        """Export the cave bundle to `out_dir`."""
        from worldgen.cave.export import export_cave

        export_cave(self, out_dir, format=format)

    def patch(self, key: tuple[int, int]) -> "CavePatch":
        """The patch at `key` ((px, pz), 0-based): its share of floor, ceiling, and materials."""
        from worldgen.cave.patches import patch

        return patch(self, key)

    def patch_at(self, x: float, z: float) -> "CavePatch":
        """The patch covering the point (x, z) in world metres."""
        from worldgen.cave.patches import patch_at

        return patch_at(self, x, z)

    def mesh(
        self,
        stride: int = 1,
        ceiling_offset: float = 0.0,
        include_ceiling: bool = True,
        cutaway_height: float | None = None,
    ) -> "MeshArrays":
        """Build dual-shell cave mesh arrays."""
        from worldgen.cave.mesh import build_cave_mesh

        return build_cave_mesh(
            self,
            stride=stride,
            ceiling_offset=ceiling_offset,
            include_ceiling=include_ceiling,
            cutaway_height=cutaway_height,
        )

    def water_mesh(self, stride: int = 1) -> "MeshArrays | None":
        """Build water mesh arrays if flooded."""
        from worldgen.cave.mesh import build_cave_water_mesh

        return build_cave_water_mesh(self, stride=stride)


def _sample_grid(grid: np.ndarray, cell_size: float, x: float, z: float) -> float:
    rows, cols = grid.shape
    u = np.clip(x / cell_size, 0.0, cols - 1.0001)
    v = np.clip(z / cell_size, 0.0, rows - 1.0001)
    x0, z0 = int(u), int(v)
    x1, z1 = min(x0 + 1, cols - 1), min(z0 + 1, rows - 1)
    fx, fz = u - x0, v - z0
    h00 = grid[z0, x0]
    h10 = grid[z0, x1]
    h01 = grid[z1, x0]
    h11 = grid[z1, x1]
    return float((h00 * (1 - fx) + h10 * fx) * (1 - fz) + (h01 * (1 - fx) + h11 * fx) * fz)


def generate_cave(
    config: CaveConfig,
    seed: int,
    report: Callable[[str, float], None] = lambda stage, seconds: None,
) -> CaveData:
    """Generate a subterranean cave from `config` and `seed`."""
    progress = pipeline.generate(config, seed, report=report)
    assert progress.cave is not None
    return progress.cave
