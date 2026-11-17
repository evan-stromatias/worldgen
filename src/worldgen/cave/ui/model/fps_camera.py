"""FPS camera for interactive inside-the-cave inspection with ground/ceiling clamping and pillar collision."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from worldgen.viewer.camera import FlyInput, Glide
from worldgen.viewer.geometry import forward_vector, right_vector

MAX_PITCH = math.radians(88.0)
DEFAULT_EYE_HEIGHT = 1.7  # metres above cave floor
DEFAULT_HEAD_MARGIN = 0.3  # metres below cave ceiling


@dataclass(slots=True)
class FpsCamera:
    """Camera designed for first-person subterranean exploration.

    Constrains the eye position between cave floor and ceiling, and prevents walking into solid rock pillars.
    """

    position: np.ndarray  # [x, y, z] in metres
    yaw: float = 0.0  # radians
    pitch: float = 0.0  # radians
    speed: float = 8.0  # metres per second
    boost: float = 2.5  # sprint multiplier
    sensitivity: float = 0.003  # radians per pixel
    eye_height: float = DEFAULT_EYE_HEIGHT
    head_margin: float = DEFAULT_HEAD_MARGIN
    fly_mode: bool = False  # free vertical flying vs grounded walking
    glide: Glide | None = None
    _floor: float = field(default=0.0)
    _ceiling: float = field(default=20.0)

    @property
    def forward(self) -> np.ndarray:
        return forward_vector(self.yaw, self.pitch)

    @property
    def target(self) -> np.ndarray:
        return self.position + self.forward

    def glide_to(self, position: np.ndarray, yaw: float, pitch: float, duration: float = 0.8) -> None:
        """Smoothly ease camera to a target location and orientation."""
        yaw = self.yaw + (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.glide = Glide(
            (self.position.copy(), self.yaw, self.pitch),
            (np.asarray(position, float), yaw, pitch),
            duration,
        )

    def turn(self, look: tuple[float, float]) -> None:
        """Rotate camera by mouse pixel delta (dx, dy)."""
        self.yaw = (self.yaw + look[0] * self.sensitivity) % (2 * math.pi)
        self.pitch = min(max(self.pitch - look[1] * self.sensitivity, -MAX_PITCH), MAX_PITCH)

    def update(
        self,
        dt: float,
        controls: FlyInput,
        floor_at: Callable[[float, float], float],
        ceiling_at: Callable[[float, float], float],
        is_navigable: Callable[[float, float], bool],
    ) -> None:
        """Update position and orientation from controls with terrain collision."""
        if self.glide is not None:
            if any(controls.move) or controls.look != (0.0, 0.0):
                self.glide = None
            else:
                self._advance_glide(dt)

        if self.glide is None:
            self.turn(controls.look)
            self.speed = min(max(self.speed * 1.25**controls.speed_steps, 1.0), 100.0)
            pace = self.speed * (self.boost if controls.boost else 1.0)

            # Horizontal movement vectors
            right = right_vector(self.yaw)
            # Forward in horizontal plane (no vertical tilt during walking)
            forward_h = np.array([math.sin(self.yaw), 0.0, -math.cos(self.yaw)], dtype=np.float64)

            ctrl_right, ctrl_up, ctrl_ahead = controls.move
            horiz_step = (right * ctrl_right + forward_h * ctrl_ahead) * pace * dt

            # Collision test for horizontal step
            curr_x, curr_z = float(self.position[0]), float(self.position[2])
            next_x = curr_x + horiz_step[0]
            next_z = curr_z + horiz_step[2]

            if is_navigable(next_x, next_z):
                self.position[0] = next_x
                self.position[2] = next_z
            elif is_navigable(next_x, curr_z):
                # Slide along X
                self.position[0] = next_x
            elif is_navigable(curr_x, next_z):
                # Slide along Z
                self.position[2] = next_z

            # Vertical movement
            if self.fly_mode:
                self.position[1] += ctrl_up * pace * dt
            else:
                # Walk locked to floor plus eye height
                self.position[1] = floor_at(self.position[0], self.position[2]) + self.eye_height

        # Clamp vertical position between floor and ceiling
        self._floor = floor_at(self.position[0], self.position[2])
        self._ceiling = ceiling_at(self.position[0], self.position[2])

        min_y = self._floor + self.eye_height
        max_y = max(min_y, self._ceiling - self.head_margin)
        self.position[1] = float(np.clip(self.position[1], min_y, max_y))

    def _advance_glide(self, dt: float) -> None:
        g = self.glide
        assert g is not None
        g.elapsed = min(g.elapsed + dt, g.duration)
        t = g.elapsed / g.duration
        eased = t * t * (3 - 2 * t)
        (p0, yaw0, pitch0), (p1, yaw1, pitch1) = g.start, g.end
        self.position = p0 + (p1 - p0) * eased
        self.yaw = (yaw0 + (yaw1 - yaw0) * eased) % (2 * math.pi)
        self.pitch = pitch0 + (pitch1 - pitch0) * eased
        if g.elapsed >= g.duration:
            self.glide = None
