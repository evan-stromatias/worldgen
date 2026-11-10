"""The 2D view's pixels and metres: plan 013's `Viewpoint` seen straight down, north up. At zoom 1 the whole
town or world (`viewpoint.side` metres) fits the view's shorter side. Coordinates are the view's: a world's own,
or a town layout's shifted by the town's reach, so the town lies in [0, side].
"""

from worldgen.geometry import XZ
from worldgen.utils.gen_ui.model.viewpoint import Viewpoint


def metres_per_pixel(viewpoint: Viewpoint, width: float, height: float) -> float:
    return viewpoint.side / viewpoint.zoom / max(min(width, height), 1.0)


def to_world(viewpoint: Viewpoint, width: float, height: float, px: float, py: float) -> XZ:
    """The point under pixel (px, py) of a view `width` x `height`; y grows south, as z does."""
    m = metres_per_pixel(viewpoint, width, height)
    cx, cz = viewpoint.centre
    return cx + (px - width / 2) * m, cz + (py - height / 2) * m
