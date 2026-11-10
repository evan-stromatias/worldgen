"""The whole terrain at low resolution, for the 3D view and height queries over the ground.

`downsample` builds the overview arrays from the full-resolution grids; `Overview` queries them and
builds the whole terrain as one mesh.
"""

import numpy as np

from worldgen.mesh.builders import APRON, PatchMeshArrays, build_terrain, build_water

MAX_SAMPLES = 512  # overview cells per side, at most


def overview_stride(size_cells: int, max_samples: int = MAX_SAMPLES) -> int:
    """Cells per overview cell: the smallest divisor of `size_cells` that brings the terrain within
    `max_samples` cells per side."""
    for stride in range(1, size_cells + 1):
        if size_cells % stride == 0 and size_cells // stride <= max_samples:
            return stride
    return size_cells


def downsample(
    height: np.ndarray, water: np.ndarray, color: np.ndarray, biome: np.ndarray, stride: int
) -> dict[str, np.ndarray]:
    """Overview arrays from full-resolution vertex grids (height, water) and cell grids (color, biome).

    Heights are point-sampled, so overview vertices sit exactly on real ones; colours are averaged.
    Heights go through float32 first, the precision meshes are built at.
    """
    rows, cols = color.shape[0] // stride, color.shape[1] // stride
    blocks = color.reshape(rows, stride, cols, stride, 3).astype(np.float64).mean(axis=(1, 3))
    return {
        "height": height.astype(np.float32)[::stride, ::stride],
        "water": water.astype(np.float32)[::stride, ::stride],
        "color": np.rint(blocks).astype(np.uint8),
        "biome": biome[::stride, ::stride],
    }


class Overview:
    """The terrain's ground, water, colour and biome at overview resolution."""

    def __init__(
        self,
        stride: int,  # world cells per overview cell
        spacing: float,  # world units between overview samples
        height: np.ndarray,  # (n + 1)² float32
        water: np.ndarray,  # (n + 1)² float32, NaN = dry
        color: np.ndarray,  # n² × 3 uint8
        biome: np.ndarray,  # n² uint8
    ):
        self.stride = stride
        self.spacing = spacing
        self.height = height
        self.water = water
        self.color = color
        self.biome = biome

    def height_at(self, x: float, z: float) -> float:
        return sample_grid(self.height, self.spacing, x, z)

    def whole(self, max_cells: int, skirt_depth: float) -> PatchMeshArrays:
        """The whole terrain as one mesh, for the 3D view, with at most `max_cells` cells per side."""
        n = self.color.shape[0]
        stride = next(s for s in range(1, n + 1) if n % s == 0 and n // s <= max_cells)
        return PatchMeshArrays(
            key=(0, 0),
            lod=-1,
            terrain=build_terrain(np.pad(self.height, APRON, mode="edge"), self.spacing, stride, skirt_depth),
            water=build_water(self.water, self.height, self.spacing, stride),
            color=self.color,
        )


def sample_grid(grid: np.ndarray, spacing: float, x: float, z: float) -> float:
    """Value at local (x, z) of a vertex grid with `spacing` between samples, interpolated over the
    same two triangles per cell the meshes use (split along the (x+1, z)-(x, z+1) diagonal)."""
    rows, cols = grid.shape
    gx = min(max(x / spacing, 0.0), cols - 1.001)
    gz = min(max(z / spacing, 0.0), rows - 1.001)
    ix, iz = int(gx), int(gz)
    fx, fz = gx - ix, gz - iz
    h00, h10 = grid[iz, ix], grid[iz, ix + 1]
    h01, h11 = grid[iz + 1, ix], grid[iz + 1, ix + 1]
    if fx + fz <= 1:
        return float(h00 + (h10 - h00) * fx + (h01 - h00) * fz)
    return float(h11 + (h01 - h11) * (1 - fx) + (h10 - h11) * (1 - fz))
