"""Subterranean cave generator: floor heightmap, 180°-rotated ceiling, clearance,
pillars, hydrology, climate and speleological materials.

Deterministic across runs given the same recipe and seed.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from worldgen.cave.config import CaveConfig
from worldgen.generator.grid import to_cells
from worldgen.generator.stages import biomes, climate, elevation, erosion

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
    world = config.world
    size_cells = world.size_cells

    # 1. Floor Elevation
    t0 = time.perf_counter()
    rng_elevation = np.random.default_rng([seed, STAGE_ELEVATION])
    elev = elevation.generate(world, config.elevation, rng_elevation)
    floor = elev.heights.copy()
    report("elevation", time.perf_counter() - t0)

    # 2. Floor Erosion (optional)
    t0 = time.perf_counter()
    sediment = np.zeros((size_cells, size_cells), np.float32)
    if config.erosion.enabled:
        rng_erosion = np.random.default_rng([seed, STAGE_EROSION])
        eroded = erosion.generate(floor, elev.terraces, world, config.erosion, rng_erosion)
        floor = eroded.heights
        sediment = to_cells(eroded.sediment).astype(np.float32)
    report("erosion", time.perf_counter() - t0)

    # 3. Cavern: 180° rotated ceiling, clearance ΔH, and pillar mask
    t0 = time.perf_counter()
    h_rot = np.rot90(floor, 2)
    cavern = config.cavern
    if cavern.ceiling_mode == "inverted":
        h_base = float(np.mean(floor))
        ceiling = h_base + cavern.clearance + (h_base - h_rot) * cavern.ceiling_scale
    else:
        ceiling = h_rot + cavern.clearance

    clearance_vertex = ceiling - floor
    clearance_cells = to_cells(clearance_vertex).astype(np.float32)

    pillar_mask = (clearance_cells < cavern.min_passage).astype(np.uint8)
    if cavern.perimeter_wall:
        t = cavern.wall_thickness
        pillar_mask[:t, :] = 1
        pillar_mask[-t:, :] = 1
        pillar_mask[:, :t] = 1
        pillar_mask[:, -t:] = 1
    report("cavern", time.perf_counter() - t0)

    # 4. Subterranean Hydrology (water table and sumps)
    t0 = time.perf_counter()
    water = np.full_like(floor, np.nan)
    if config.hydrology.enabled:
        wt = config.hydrology.water_table
        wet = floor < wt
        water[wet] = wt
    report("hydrology", time.perf_counter() - t0)

    # 5. Cave Climate (depth geothermal gradient & humidity)
    t0 = time.perf_counter()
    floor_cells = to_cells(floor)
    depth = np.max(floor) - floor_cells
    temperature = (config.climate.temperature_base + config.climate.geothermal_gradient * depth).astype(np.float32)
    moisture = np.full_like(temperature, config.climate.ambient_humidity)
    if config.hydrology.enabled:
        wet_cells = to_cells((~np.isnan(water)).astype(np.float64))
        moisture = np.clip(moisture + 0.15 * wet_cells.astype(np.float32), 0.0, 1.0)
    report("climate", time.perf_counter() - t0)

    # 6. Speleological Materials (biomes engine reuse)
    t0 = time.perf_counter()
    cell_inputs = biomes.CellInputs(
        elevation=floor_cells - world.sea_level,
        temperature=temperature.astype(np.float64),
        moisture=moisture.astype(np.float64),
        slope=to_cells(climate.slope_degrees(floor, world.cell_size)).astype(np.float64),
        water=biomes.water_kind_cells(
            np.zeros_like(water, bool),
            ~np.isnan(water),
            np.zeros_like(water, bool),
        ),
        sediment=sediment.astype(np.float64),
        relief=biomes.local_relief(floor, world.cell_size, config.materials.relief_radius).astype(np.float64),
    )
    rng_materials = np.random.default_rng([seed, STAGE_MATERIALS])
    material, color = biomes.classify(cell_inputs, config.materials, rng_materials)
    report("materials", time.perf_counter() - t0)

    return CaveData(
        seed=seed,
        config=config,
        floor_elevation=floor,
        ceiling_elevation=ceiling,
        clearance=clearance_cells,
        pillar_mask=pillar_mask,
        water=water,
        material=material,
        color=color,
        temperature=temperature,
        moisture=moisture,
        sediment=sediment,
    )
