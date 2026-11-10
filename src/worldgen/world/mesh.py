"""Terrain and water meshes for a patch of ground, at any level of detail.

Pure numpy, so it runs on worker threads; `worldgen.world.gpu` uploads the arrays on the main thread.
"""

from dataclasses import dataclass

import numpy as np

APRON = 1  # extra ring of height samples around a patch's mesh, so its edge normals match its neighbours'


@dataclass(frozen=True, slots=True)
class MeshArrays:
    positions: np.ndarray  # (n, 3) float32, local to the patch's north-west corner
    normals: np.ndarray | None  # (n, 3) float32; for water, the river flow (dx, strength 0-1, dz)
    texcoords: np.ndarray  # (n, 2) float32; for water, x is the water depth (negative past the shoreline)
    indices: np.ndarray  # (m, 3) uint32
    colors: np.ndarray | None = None  # (n, 4) uint8 per-vertex colour (towns), multiplying the texture
    texcoords2: np.ndarray | None = None  # (n, 2) float32, for the reader's shader (the game's grass: how it sways)

    @property
    def nbytes(self) -> int:
        normals = self.normals.nbytes if self.normals is not None else 0
        colors = self.colors.nbytes if self.colors is not None else 0
        texcoords2 = self.texcoords2.nbytes if self.texcoords2 is not None else 0
        return self.positions.nbytes + normals + colors + self.texcoords.nbytes + texcoords2 + self.indices.size * 2


@dataclass(frozen=True, slots=True)
class PatchMeshArrays:
    """Everything the GPU needs for one patch at one level of detail."""

    key: tuple[int, int]
    lod: int
    terrain: MeshArrays
    water: MeshArrays | None
    color: np.ndarray  # texture pixels, (rows, cols, 3) uint8
    ground: np.ndarray | None = None  # (rows, cols) uint8 per cell, for the game's ground textures; read exactly

    @property
    def nbytes(self) -> int:
        """GPU memory this will take, roughly: buffers plus the textures and the colour's mipmaps."""
        water = self.water.nbytes if self.water is not None else 0
        ground = self.ground.nbytes if self.ground is not None else 0
        return self.terrain.nbytes + water + self.color.nbytes * 4 // 3 + ground


def build_terrain(height: np.ndarray, spacing: float, stride: int, skirt_depth: float) -> MeshArrays:
    """Terrain mesh from a height grid with a one-sample apron, (n + 3)², keeping every `stride`-th vertex.

    Normals use full-resolution central differences, reaching into the apron at the edges, so they
    match a neighbour's at every level of detail. A skirt (the border ring dropped `skirt_depth`
    straight down) hides cracks against neighbours drawn at a different level.
    """
    n = height.shape[0] - 1 - 2 * APRON
    if n % stride:
        raise ValueError(f"stride {stride} does not divide {n} cells")
    keep = np.arange(0, n + 1, stride)
    k = len(keep)
    rows, cols = np.meshgrid(keep, keep, indexing="ij")
    h = height[rows + APRON, cols + APRON]

    left, right = height[rows + APRON, cols + APRON - 1], height[rows + APRON, cols + APRON + 1]
    up, down = height[rows + APRON - 1, cols + APRON], height[rows + APRON + 1, cols + APRON]
    normals = np.stack([(left - right) / (2 * spacing), np.ones_like(h), (up - down) / (2 * spacing)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    positions = np.stack([cols * spacing, h, rows * spacing], axis=-1).reshape(-1, 3)
    texcoords = np.stack([cols / n, rows / n], axis=-1).reshape(-1, 2)
    normals = normals.reshape(-1, 3)
    indices = [_grid_triangles(k, np.ones((k - 1, k - 1), bool))]

    # Skirt: each edge's vertices again, lowered, joined to the edge by a strip of quads
    ring = np.arange(k * k).reshape(k, k)
    for edge in (ring[0, :], ring[:, -1], ring[-1, :], ring[:, 0]):
        base = len(positions)
        lowered = positions[edge].copy()
        lowered[:, 1] -= skirt_depth
        positions = np.concatenate([positions, lowered])
        normals = np.concatenate([normals, normals[edge]])
        texcoords = np.concatenate([texcoords, texcoords[edge]])
        top, next_top = edge[:-1], edge[1:]  # neighbours along the edge, which are k apart on the east and west
        bottom = base + np.arange(k - 1)
        indices.append(np.stack([top, bottom, next_top], axis=-1))
        indices.append(np.stack([next_top, bottom, bottom + 1], axis=-1))

    return MeshArrays(
        positions.astype(np.float32),
        normals.astype(np.float32),
        texcoords.astype(np.float32),
        np.concatenate(indices).astype(np.uint32),
    )


def build_water(
    water: np.ndarray, height: np.ndarray, spacing: float, stride: int,
    flow: np.ndarray | None = None,
) -> MeshArrays | None:
    """Water surface over every cell with a wet corner; None if the patch is dry.

    `water` and `height` are (n + 1)² vertex grids (no apron). Texcoord x carries the water depth.
    `flow`, an (n + 1)² × 3 grid of (dx, strength, dz) river flow, goes in the normals.

    The surface reaches one vertex past the wet ones, dipped just under the ground there (`shore`),
    so it meets the bank part-way across the cell: shorelines follow the terrain instead of the grid,
    and rivers one vertex wide still show.
    """
    keep = np.arange(0, water.shape[0], stride)
    k = len(keep)
    w = water[np.ix_(keep, keep)]
    ground = height[np.ix_(keep, keep)]
    wet = ~np.isnan(w)
    if not wet.any():
        return None
    surface = shore(w, ground)
    defined = ~np.isnan(surface)
    cells = defined[:-1, :-1] & defined[:-1, 1:] & defined[1:, :-1] & defined[1:, 1:]
    cells &= wet[:-1, :-1] | wet[:-1, 1:] | wet[1:, :-1] | wet[1:, 1:]
    if not cells.any():
        return None
    rows, cols = np.meshgrid(keep, keep, indexing="ij")
    # Negative at shore vertices (the bank is above the surface), so depth crosses zero at the shoreline
    depth = np.where(wet, np.maximum(w - ground, 0.0), np.where(defined, surface - ground, 0.0))
    surface = np.where(defined, surface, 0.0)
    positions = np.stack([cols * spacing, surface, rows * spacing], axis=-1).reshape(-1, 3)
    texcoords = np.stack([depth, np.zeros_like(depth)], axis=-1).reshape(-1, 2)
    normals = flow[np.ix_(keep, keep)].reshape(-1, 3).astype(np.float32) if flow is not None else None
    return MeshArrays(
        positions.astype(np.float32), normals, texcoords.astype(np.float32), _grid_triangles(k, cells).astype(np.uint32)
    )


SHORE_DIP = 0.05  # metres under the ground that the surface is taken to at a dry vertex beside water


def shore(water: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """`water` with each dry vertex next to wet ones (of its 8 neighbours) given a surface: the lowest
    neighbouring surface, but no higher than just under its own ground. NaN elsewhere."""
    padded = np.pad(water, 1, constant_values=np.nan)
    rows, cols = water.shape
    neighbours = np.stack([padded[1 + dz:1 + dz + rows, 1 + dx:1 + dx + cols]
                           for dz in (-1, 0, 1) for dx in (-1, 0, 1) if dz or dx])
    with np.errstate(all="ignore"):
        lowest = np.nanmin(np.where(np.isnan(neighbours), np.inf, neighbours), axis=0)
    beside = np.isnan(water) & np.isfinite(lowest)
    result = water.copy()
    result[beside] = np.minimum(lowest[beside], ground[beside] - SHORE_DIP)
    return result


def river_flow(
    river: np.ndarray | None, flow_dir: np.ndarray | None, flow_dirs: np.ndarray | None
) -> np.ndarray | None:
    """(n + 1)² × 3 grid of (dx, strength 0-1, dz): each river vertex's downstream direction, from the
    D8 codes (`flow_dirs` maps a code to (dx, dz)). None without river data."""
    if river is None or flow_dir is None or flow_dirs is None:
        return None
    directions = np.zeros((256, 2), np.float32)
    unit = flow_dirs / np.linalg.norm(flow_dirs, axis=1, keepdims=True)
    directions[: len(unit)] = unit
    d = directions[flow_dir]
    strength = river.astype(np.float32) / 255.0
    d[river == 0] = 0.0
    return np.stack([d[..., 0], strength, d[..., 1]], axis=-1)


def sea_frame(world_size: float, sea_level: float, reach: float, depth: float) -> MeshArrays:
    """Open sea around the world: four quads framing it, so it never doubles up with patch water.
    Every vertex has water `depth` (texcoord x)."""
    w, r = world_size, reach
    rects = ((-r, -r, w + r, 0), (-r, w, w + r, w + r), (-r, 0, 0, w), (w, 0, w + r, w))
    positions, indices = [], []
    for x0, z0, x1, z1 in rects:
        base = len(positions)
        positions += [(x0, sea_level, z0), (x0, sea_level, z1), (x1, sea_level, z0), (x1, sea_level, z1)]
        indices += [(base, base + 1, base + 2), (base + 2, base + 1, base + 3)]
    positions = np.array(positions, np.float32)
    texcoords = np.tile(np.array([[depth, 0.0]], np.float32), (len(positions), 1))
    return MeshArrays(positions, np.zeros_like(positions), texcoords, np.array(indices, np.uint32))


def _grid_triangles(k: int, cells: np.ndarray) -> np.ndarray:
    """Two triangles per selected cell of a k × k vertex grid, split along the (x+1, z)-(x, z+1)
    diagonal and wound counter-clockwise seen from above, like raylib's heightmap meshes."""
    a, b = np.nonzero(cells)
    v00 = a * k + b
    v01, v10, v11 = v00 + 1, v00 + k, v00 + k + 1
    return np.concatenate([np.stack([v00, v10, v01], axis=-1), np.stack([v01, v10, v11], axis=-1)])
