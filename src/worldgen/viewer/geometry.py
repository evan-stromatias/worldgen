"""Camera maths with no window: directions, frustum tests, screen rays, minimap camera and picking.

Directions follow the generator: north is -z, east is +x. Headings (yaw) are radians clockwise from
north; pitch is radians above the horizon (negative looks down).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def forward_vector(yaw: float, pitch: float) -> np.ndarray:
    return np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])


def right_vector(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), 0.0, math.sin(yaw)])


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def camera_basis(forward: np.ndarray, up: np.ndarray = UP) -> tuple[np.ndarray, np.ndarray]:
    """Right and screen-up vectors for a camera looking along `forward`."""
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight along `up`: any right will do
        right = np.array([1.0, 0.0, 0.0])
    right = _normalize(right)
    return right, np.cross(right, forward)


@dataclass(frozen=True, slots=True)
class Frustum:
    """The four side planes (and near plane) of a perspective view, as inward normals through the eye."""

    eye: np.ndarray
    normals: np.ndarray  # (5, 3)

    @classmethod
    def from_camera(cls, eye: np.ndarray, forward: np.ndarray, fovy_degrees: float, aspect: float) -> "Frustum":
        forward = _normalize(forward)
        right, up = camera_basis(forward)
        half_v = math.tan(math.radians(fovy_degrees) / 2)
        half_h = half_v * aspect
        normals = []
        for edge, across in ((forward - right * half_h, up), (forward + right * half_h, up),
                             (forward - up * half_v, right), (forward + up * half_v, right)):
            n = _normalize(np.cross(across, edge))
            normals.append(n if n @ forward > 0 else -n)
        normals.append(forward)
        return cls(np.asarray(eye, float), np.array(normals))

    def sees_boxes(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """`sees_box` for many boxes at once: (n, 3) low and high corners -> (n,) bool."""
        visible = np.ones(len(lo), bool)
        for n in self.normals:
            corner = np.where(n > 0, hi, lo)
            visible &= (corner - self.eye) @ n >= 0
        return visible

    def sees_box(self, box: tuple[float, float, float, float, float, float]) -> bool:
        """Conservative: True unless the box is wholly outside one plane."""
        lo, hi = np.array(box[:3]), np.array(box[3:])
        for n in self.normals:
            corner = np.where(n > 0, hi, lo)  # the corner farthest along the plane's normal
            if (corner - self.eye) @ n < 0:
                return False
        return True


def screen_ray(
    eye: np.ndarray, target: np.ndarray, fovy_degrees: float, width: float, height: float, px: float, py: float
) -> np.ndarray:
    """Unit direction through pixel (px, py) of a perspective camera at `eye` looking at `target`."""
    forward = _normalize(np.asarray(target, float) - eye)
    right, up = camera_basis(forward)
    half_v = math.tan(math.radians(fovy_degrees) / 2)
    x = (2 * px / width - 1) * half_v * width / height
    y = (1 - 2 * py / height) * half_v
    return _normalize(forward + right * x + up * y)


def ray_plane(origin: np.ndarray, direction: np.ndarray, plane_y: float) -> np.ndarray | None:
    """Where the ray meets the horizontal plane y = plane_y, or None if it never does."""
    if abs(direction[1]) < 1e-9:
        return None
    t = (plane_y - origin[1]) / direction[1]
    return origin + direction * t if t > 0 else None


@dataclass(frozen=True, slots=True)
class MinimapCamera:
    eye: np.ndarray
    target: np.ndarray
    fovy: float


def minimap_camera(
    world_size: float, sea_level: float, heading: float, fovy: float = 40.0, tilt_degrees: float = 55.0
) -> MinimapCamera:
    """A camera framing the whole world from behind `heading` (0 = from the south, north up), tilted down."""
    centre = np.array([world_size / 2, sea_level, world_size / 2])
    distance = 1.08 * world_size / (2 * math.tan(math.radians(fovy) / 2))
    tilt = math.radians(tilt_degrees)
    behind = np.array([-math.sin(heading), 0.0, math.cos(heading)])
    eye = centre + behind * distance * math.cos(tilt) + UP * distance * math.sin(tilt)
    return MinimapCamera(eye, centre, fovy)


def pick_on_terrain(
    camera: MinimapCamera,
    size: float,
    px: float,
    py: float,
    height_at: Callable[[float, float], float],
    height_scale: float,
    sea_level: float,
    world_size: float,
) -> tuple[float, float] | None:
    """World (x, z) under minimap pixel (px, py), on terrain drawn with heights exaggerated by
    `height_scale` about sea level. Refines a plane intersection a few times against the terrain."""
    direction = screen_ray(camera.eye, camera.target, camera.fovy, size, size, px, py)
    plane = sea_level
    hit = None
    for _ in range(6):
        hit = ray_plane(camera.eye, direction, plane)
        if hit is None:
            return None
        x = min(max(hit[0], 0.0), world_size)
        z = min(max(hit[2], 0.0), world_size)
        plane = sea_level + (height_at(x, z) - sea_level) * height_scale
    if not (0 <= hit[0] <= world_size and 0 <= hit[2] <= world_size):
        return None
    return float(hit[0]), float(hit[2])


def view_footprint(
    eye: np.ndarray, forward: np.ndarray, fovy_degrees: float, aspect: float, ground_y: float, reach: float
) -> list[np.ndarray]:
    """Ground points under the four corners of the view, in order; corners above the horizon are
    placed `reach` away horizontally instead."""
    right, up = camera_basis(_normalize(forward))
    half_v = math.tan(math.radians(fovy_degrees) / 2)
    half_h = half_v * aspect
    points = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        direction = _normalize(forward + right * sx * half_h + up * sy * half_v)
        hit = ray_plane(eye, direction, ground_y) if direction[1] < -0.02 else None
        if hit is None or np.linalg.norm((hit - eye)[[0, 2]]) > reach:
            flat = np.array([direction[0], 0.0, direction[2]])
            flat = _normalize(flat) if np.linalg.norm(flat) > 1e-6 else np.array([0.0, 0.0, -1.0])
            hit = np.array([eye[0], ground_y, eye[2]]) + flat * reach
        points.append(hit)
    return points


def raymarch(
    origin: np.ndarray, direction: np.ndarray, height_at: Callable[[float, float], float], reach: float, step: float
) -> np.ndarray | None:
    """First point along the ray at or below the terrain, within `reach`."""
    t = 0.0
    while t <= reach:
        p = origin + direction * t
        clearance = p[1] - height_at(p[0], p[2])
        if clearance <= 0:
            return p
        t += max(step, clearance * 0.5)
    return None
