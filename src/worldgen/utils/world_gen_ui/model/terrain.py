"""A layer as the map shows it: the flat picture for 2D, and the terrain coloured by it for 3D.

The 3D terrain is the whole world at overview resolution (`overview.downsample`, then `Overview.whole`),
from the session's arrays.
"""

import math
from dataclasses import dataclass

import numpy as np

from worldgen.proc_gen.world.config import TerrainConfig
from worldgen.proc_gen.world.pipeline import Progress
from worldgen.utils.world_gen_ui.model.layers import MAX_SIDE, ground_and_water, layer_image
from worldgen.world.mesh import PatchMeshArrays
from worldgen.world.overview import Overview, downsample, overview_stride

MAX_CELLS = 512  # cells per side of the 3D terrain, at most


@dataclass(frozen=True, slots=True)
class Picture:
    image: np.ndarray  # uint8 RGB, north up, at most MAX_SIDE pixels a side
    overview: Overview  # heights for picking in 3D
    terrain: PatchMeshArrays  # the whole terrain as one mesh, coloured by the layer


def picture(layer: str, progress: Progress, config: TerrainConfig, max_side: int = MAX_SIDE) -> Picture:
    """Draw `layer` once, at a pixel per cell, and make both the map image and the 3D terrain from it."""
    image = layer_image(layer, progress, config)
    step = math.ceil(image.shape[0] / max_side)
    overview = layer_overview(progress, config, image)
    return Picture(np.ascontiguousarray(image[::step, ::step]), overview, overview.whole(MAX_CELLS, skirt_depth(config)))


def layer_overview(progress: Progress, config: TerrainConfig, image: np.ndarray) -> Overview:
    """The terrain's ground and water from `progress`, coloured by `image` (a pixel per cell), at overview
    resolution."""
    world = config.world
    ground, water = ground_and_water(progress)
    if water is None:
        water = np.full_like(ground, np.nan)
    stride = overview_stride(world.size_cells)
    arrays = downsample(ground, water, image, np.zeros(image.shape[:2], np.uint8), stride)
    return Overview(stride=stride, spacing=stride * world.cell_size, **arrays)


def skirt_depth(config: TerrainConfig) -> float:
    """Deep enough to close the world's edges when seen from low down."""
    world = config.world
    return max(3.0, 0.005 * world.size_cells * world.cell_size)
