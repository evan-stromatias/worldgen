"""Where the map looks, in 2D and 3D alike, so that running a stage or switching between 2D and 3D never
moves it: the point in the middle of the view, how far in it is zoomed, and in 3D which way it faces.

In 3D the camera orbits that point on the ground, at the distance that shows as much of the world as 2D
does at the same zoom. Headings are radians clockwise from north (0 looks north, as 2D shows it); pitch is
how far below the horizon the camera looks.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from worldgen.viewer.geometry import ray_plane, screen_ray

FOVY = 45.0  # degrees, the 3D camera's vertical field of view
ZOOM_STEP = 1.25  # per wheel notch
MAX_ZOOM = 64.0  # times the size that fits the view
MIN_PITCH, MAX_PITCH = math.radians(12.0), math.radians(89.0)
PITCH = math.radians(50.0)  # at start


@dataclass(slots=True)
class Viewpoint:
    side: float = 0.0  # metres the world spans, west to east and north to south
    centre: tuple[float, float] = (0.0, 0.0)  # world point (x, z) in the middle of the view, metres
    zoom: float = 1.0  # 1: the whole world fits
    heading: float = 0.0
    pitch: float = PITCH

    def fit(self, side: float) -> None:
        """Look at the whole of a world `side` metres across, unless it is the size already shown."""
        if side != self.side:
            self.side, self.zoom, self.centre = side, 1.0, (side / 2, side / 2)

    def zoom_about(self, notches: float, point: tuple[float, float] | None = None) -> None:
        """Zoom in (notches > 0) or out, keeping `point` where it is on the screen (by default the centre)."""
        before = self.zoom
        self.zoom = min(max(self.zoom * ZOOM_STEP ** notches, 1.0), MAX_ZOOM)
        if point is not None:
            ratio = before / self.zoom
            self.centre = tuple(p + (c - p) * ratio for c, p in zip(self.centre, point, strict=True))
        self._keep_in_view()

    def pan(self, dx: float, dz: float) -> None:
        """Move the centre by (dx, dz) metres."""
        self.centre = (self.centre[0] + dx, self.centre[1] + dz)
        self._keep_in_view()

    def drag(self, dx: float, dy: float, metres_per_pixel: float) -> None:
        """Move the 3D view as the mouse moved (pixels), so the ground follows it: across and along the heading."""
        right = (math.cos(self.heading), math.sin(self.heading))
        ahead = (math.sin(self.heading), -math.cos(self.heading))
        self.pan((-right[0] * dx + ahead[0] * dy) * metres_per_pixel, (-right[1] * dx + ahead[1] * dy) * metres_per_pixel)

    def orbit(self, turn: float, tilt: float) -> None:
        """Turn the 3D camera about the centre (radians clockwise) and tilt it (radians, more is steeper)."""
        self.heading = (self.heading + turn) % (2 * math.pi)
        self.pitch = min(max(self.pitch + tilt, MIN_PITCH), MAX_PITCH)

    def distance(self) -> float:
        """How far the 3D camera is from the centre: as far as shows the ground 2D shows at this zoom."""
        return self.side / self.zoom / (2 * math.tan(math.radians(FOVY) / 2))

    def camera(self, ground_y: float) -> tuple[np.ndarray, np.ndarray]:
        """The 3D camera's eye and target, the target on the ground (at `ground_y`) under the centre."""
        target = np.array([self.centre[0], ground_y, self.centre[1]])
        back = np.array([
            -math.sin(self.heading) * math.cos(self.pitch),
            math.sin(self.pitch),
            math.cos(self.heading) * math.cos(self.pitch),
        ])
        return target + back * self.distance(), target

    def pick(
        self, ground_y: float, width: float, height: float, px: float, py: float, height_at: Callable[[float, float], float]
    ) -> tuple[float, float] | None:
        """The world point (x, z) on the ground under pixel (px, py) of a 3D view `width` x `height` pixels:
        where the ray meets the ground's height, refined a few times as `minimap` picking is."""
        eye, target = self.camera(ground_y)
        direction = screen_ray(eye, target, FOVY, width, height, px, py)
        plane, hit = ground_y, None
        for _ in range(6):
            hit = ray_plane(eye, direction, plane)
            if hit is None:
                return None
            plane = height_at(min(max(hit[0], 0.0), self.side), min(max(hit[2], 0.0), self.side))
        if not (0 <= hit[0] < self.side and 0 <= hit[2] < self.side):
            return None
        return float(hit[0]), float(hit[2])

    def _keep_in_view(self) -> None:
        """Keep some of the world in view."""
        half = self.side / 2
        self.centre = tuple(min(max(c, half - self.side), half + self.side) for c in self.centre)
