"""Read a terrain in patches: fixed-size tiles of `world.patch_cells` cells, for games that stream or
load the world a piece at a time.

A patch carries the patch's share of every grid the terrain has: the heightmap with a one-vertex apron
(so its edge normals match its neighbours' when meshed), the water grids without an apron, and the
per-cell grids, plus the records of the lakes and rivers that reach it. `Patch.mesh()` builds the
terrain and water meshes, ready for `worldgen.world.gpu` to upload.

Patches are exact slices of the terrain's own arrays: nothing is resampled, so adjacent patches meet
without a seam.
"""

from dataclasses import dataclass

import numpy as np

from worldgen.proc_gen.world.hydrology import D8, Lake
from worldgen.proc_gen.world.terrain import TerrainData
from worldgen.world.mesh import PatchMeshArrays, build_terrain, build_water, river_flow

# (dx, dz) per D8 code, as river_flow wants them (hydrology's D8 is (dz, dx))
DIRECTIONS = np.array([(dx, dz) for dz, dx in D8], np.float32)


@dataclass(frozen=True, slots=True)
class Patch:
    """One patch of a terrain, `patch_cells` a side."""

    key: tuple[int, int]  # (px, pz): patch indices, (0, 0) at the world's north-west corner
    origin: tuple[float, float]  # metres, the patch's north-west corner (x, z)
    cell_size: float  # metres per cell
    cells: int  # cells per side (world.patch_cells)
    elevation: np.ndarray  # (P+3)² float64, metres, with a one-vertex apron (edge-clamped at the world's edges)
    water: np.ndarray  # (P+1)² float64, water surface height, NaN = dry
    sea: np.ndarray  # (P+1)² bool
    lake_id: np.ndarray  # (P+1)² uint16
    river: np.ndarray  # (P+1)² uint8 strength
    flow_dir: np.ndarray  # (P+1)² uint8 D8 code
    biome: np.ndarray  # P² uint8, index into the recipe's biome rules
    color: np.ndarray  # P² × 3 uint8
    temperature: np.ndarray  # P² float32, °C
    moisture: np.ndarray  # P² float32, 0-1
    sediment: np.ndarray  # P² float32, metres
    lakes: list[Lake]  # the lakes whose bounds reach the patch
    rivers: list[list[tuple[int, int]]]  # the rivers (world vertex coordinates) that pass through the patch

    @property
    def vertex_height(self) -> np.ndarray:
        """The heightmap without its apron: (P+1)²."""
        return self.elevation[1:-1, 1:-1]

    def mesh(self, skirt_depth: float = 3.0) -> PatchMeshArrays:
        """Terrain and water meshes for the patch, river flow included, ready to upload (`worldgen.world.gpu`)."""
        flow = river_flow(self.river, self.flow_dir, DIRECTIONS)
        return PatchMeshArrays(
            key=self.key,
            lod=0,
            terrain=build_terrain(self.elevation, self.cell_size, 1, skirt_depth),
            water=build_water(self.water, self.vertex_height, self.cell_size, 1, flow),
            color=np.ascontiguousarray(self.color),
        )


def patches_per_side(terrain: TerrainData) -> int:
    """How many patches tile the terrain's side."""
    return terrain.config.world.size_cells // terrain.config.world.patch_cells


def patch(terrain: TerrainData, key: tuple[int, int]) -> Patch:
    """The patch at `key` ((px, pz), 0-based)."""
    world = terrain.config.world
    n, p = world.size_cells, world.patch_cells
    px, pz = key
    if not (0 <= px < n // p and 0 <= pz < n // p):
        raise IndexError(f"patch {key} lies outside the {n // p} x {n // p} patch grid")
    x0, z0 = px * p, pz * p

    def vertex_grid(a: np.ndarray) -> np.ndarray:
        return a[z0:z0 + p + 1, x0:x0 + p + 1]

    # The apron: one vertex further each way, clamped at the world's edges (np.pad "edge" repeats the rim)
    zlo, zhi, xlo, xhi = max(z0 - 1, 0), min(z0 + p + 2, n + 1), max(x0 - 1, 0), min(x0 + p + 2, n + 1)
    elevation = np.pad(
        terrain.elevation[zlo:zhi, xlo:xhi],
        ((zlo - (z0 - 1), (z0 + p + 2) - zhi), (xlo - (x0 - 1), (x0 + p + 2) - xhi)),
        mode="edge",
    )

    lake_ids = set(np.unique(vertex_grid(terrain.lake_id))) - {0}
    lakes = [lake for lake in terrain.lakes if lake.id in lake_ids]
    rivers = [river for river in terrain.rivers
              if any(x0 <= x <= x0 + p and z0 <= z <= z0 + p for x, z in river)]
    return Patch(
        key=(px, pz),
        origin=(x0 * world.cell_size, z0 * world.cell_size),
        cell_size=world.cell_size,
        cells=p,
        elevation=elevation,
        water=vertex_grid(terrain.water),
        sea=vertex_grid(terrain.sea),
        lake_id=vertex_grid(terrain.lake_id),
        river=vertex_grid(terrain.river),
        flow_dir=vertex_grid(terrain.flow_dir),
        biome=terrain.biome[z0:z0 + p, x0:x0 + p],
        color=terrain.color[z0:z0 + p, x0:x0 + p],
        temperature=terrain.temperature[z0:z0 + p, x0:x0 + p],
        moisture=terrain.moisture[z0:z0 + p, x0:x0 + p],
        sediment=terrain.sediment[z0:z0 + p, x0:x0 + p],
        lakes=lakes,
        rivers=rivers,
    )


def patch_at(terrain: TerrainData, x: float, z: float) -> Patch:
    """The patch under the world point (x, z), in metres."""
    world = terrain.config.world
    p = world.patch_cells * world.cell_size
    return patch(terrain, (min(int(x / p), patches_per_side(terrain) - 1),
                           min(int(z / p), patches_per_side(terrain) - 1)))
