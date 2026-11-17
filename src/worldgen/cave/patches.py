"""Read a subterranean cave in patches: fixed-size tiles of `world.patch_cells` cells.

A patch carries the patch's share of every grid the cave has: the floor and ceiling heightmaps
with a one-vertex apron (so edge normals match neighbours when meshed), and the per-cell grids
(clearance, pillar_mask, material, color, temperature, moisture, sediment).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from worldgen.cave.generator import CaveData


@dataclass(frozen=True, slots=True)
class CavePatch:
    """One patch of a subterranean cave, `patch_cells` a side."""

    key: tuple[int, int]  # (px, pz): patch indices, (0, 0) at the cave's north-west corner
    origin: tuple[float, float]  # metres, the patch's north-west corner (x, z)
    cell_size: float  # metres per cell
    cells: int  # cells per side (world.patch_cells)
    floor_elevation: np.ndarray  # (P+3)² float64, metres, with a one-vertex apron (edge-clamped at the world's edges)
    ceiling_elevation: np.ndarray  # (P+3)² float64, metres, with a one-vertex apron (edge-clamped at the world's edges)
    clearance: np.ndarray  # P² float32 metres
    pillar_mask: np.ndarray  # P² uint8 (1 = pillar/rock, 0 = open cavern)
    water: np.ndarray  # (P+1)² float64, water surface height, NaN = dry
    material: np.ndarray  # P² uint8 index into materials rules
    color: np.ndarray  # P² × 3 uint8 RGB
    temperature: np.ndarray  # P² float32 °C
    moisture: np.ndarray  # P² float32 0-1
    sediment: np.ndarray  # P² float32 metres

    @property
    def vertex_floor(self) -> np.ndarray:
        """The floor heightmap without its apron: (P+1)²."""
        return self.floor_elevation[1:-1, 1:-1]

    @property
    def vertex_ceiling(self) -> np.ndarray:
        """The ceiling heightmap without its apron: (P+1)²."""
        return self.ceiling_elevation[1:-1, 1:-1]

    @property
    def elevation(self) -> np.ndarray:
        """Alias for floor_elevation (with apron)."""
        return self.floor_elevation

    @property
    def biome(self) -> np.ndarray:
        """Alias for material."""
        return self.material


def patches_per_side(cave: "CaveData") -> int:
    """How many patches tile the cave's side."""
    return cave.config.world.size_cells // cave.config.world.patch_cells


def patch(cave: "CaveData", key: tuple[int, int]) -> CavePatch:
    """The patch at `key` ((px, pz), 0-based)."""
    world = cave.config.world
    n, p = world.size_cells, world.patch_cells
    px, pz = key
    if not (0 <= px < n // p and 0 <= pz < n // p):
        raise IndexError(f"patch {key} lies outside the {n // p} x {n // p} patch grid")
    x0, z0 = px * p, pz * p

    def vertex_grid(a: np.ndarray) -> np.ndarray:
        return a[z0 : z0 + p + 1, x0 : x0 + p + 1]

    def cell_grid(a: np.ndarray) -> np.ndarray:
        return a[z0 : z0 + p, x0 : x0 + p]

    # The apron: one vertex further each way, clamped at the world's edges (np.pad "edge" repeats the rim)
    zlo, zhi = max(z0 - 1, 0), min(z0 + p + 2, n + 1)
    xlo, xhi = max(x0 - 1, 0), min(x0 + p + 2, n + 1)
    pad_spec = ((zlo - (z0 - 1), (z0 + p + 2) - zhi), (xlo - (x0 - 1), (x0 + p + 2) - xhi))

    floor_elevation = np.pad(cave.floor_elevation[zlo:zhi, xlo:xhi], pad_spec, mode="edge")
    ceiling_elevation = np.pad(cave.ceiling_elevation[zlo:zhi, xlo:xhi], pad_spec, mode="edge")

    return CavePatch(
        key=(px, pz),
        origin=(x0 * world.cell_size, z0 * world.cell_size),
        cell_size=world.cell_size,
        cells=p,
        floor_elevation=floor_elevation,
        ceiling_elevation=ceiling_elevation,
        clearance=cell_grid(cave.clearance),
        pillar_mask=cell_grid(cave.pillar_mask),
        water=vertex_grid(cave.water),
        material=cell_grid(cave.material),
        color=cell_grid(cave.color),
        temperature=cell_grid(cave.temperature),
        moisture=cell_grid(cave.moisture),
        sediment=cell_grid(cave.sediment),
    )


def patch_at(cave: "CaveData", x: float, z: float) -> CavePatch:
    """The patch covering the point (x, z) in world metres."""
    world = cave.config.world
    n, p = world.size_cells, world.patch_cells
    patch_size = p * world.cell_size
    px = int(np.clip(x / patch_size, 0, (n // p) - 1))
    pz = int(np.clip(z / patch_size, 0, (n // p) - 1))
    return patch(cave, (px, pz))
