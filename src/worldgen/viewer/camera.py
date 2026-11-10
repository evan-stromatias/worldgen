"""The free-flying camera: input in, position and orientation out, with eased glides for teleports."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from worldgen.viewer.geometry import forward_vector, right_vector

MAX_PITCH = math.radians(89.0)
GROUND_MARGIN = 2.0  # never closer to the terrain than this
ALTITUDE_SPEEDUP = 30.0  # speed grows with height above the ground past this many units


@dataclass(slots=True)
class FlyInput:
    """One frame of input, gathered by the app from raylib."""

    look: tuple[float, float] = (0.0, 0.0)  # mouse movement in pixels, while looking
    move: tuple[float, float, float] = (0.0, 0.0, 0.0)  # right, up, forward: each -1, 0 or 1
    boost: bool = False
    speed_steps: float = 0.0  # scroll wheel notches


@dataclass(slots=True)
class Glide:
    start: tuple[np.ndarray, float, float]
    end: tuple[np.ndarray, float, float]
    duration: float
    elapsed: float = 0.0


@dataclass(slots=True)
class FlyCamera:
    position: np.ndarray
    yaw: float
    pitch: float
    speed: float
    boost: float
    sensitivity: float
    glide: Glide | None = None
    _ground: float = field(default=0.0)

    @property
    def forward(self) -> np.ndarray:
        return forward_vector(self.yaw, self.pitch)

    @property
    def target(self) -> np.ndarray:
        return self.position + self.forward

    def height_above_ground(self) -> float:
        return float(self.position[1] - self._ground)

    def glide_to(self, position: np.ndarray, yaw: float, pitch: float, duration: float) -> None:
        # Turn the short way round
        yaw = self.yaw + (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.glide = Glide(
            (self.position.copy(), self.yaw, self.pitch), (np.asarray(position, float), yaw, pitch), duration
        )

    def face(self, yaw: float, duration: float) -> None:
        """Turn to a heading in place, keeping the pitch."""
        self.glide_to(self.position, yaw, self.pitch, duration)

    def update(self, dt: float, controls: FlyInput, ground_at: Callable[[float, float], float]) -> None:
        if self.glide is not None:
            moving = any(controls.move) or controls.look != (0.0, 0.0)
            if moving:
                self.glide = None  # taking the controls back cancels a glide
            else:
                self._advance_glide(dt)
        if self.glide is None:
            self.turn(controls.look)
            self.speed = min(max(self.speed * 1.25**controls.speed_steps, 1.0), 5000.0)
            self._ground = ground_at(self.position[0], self.position[2])
            pace = self.speed * max(1.0, self.height_above_ground() / ALTITUDE_SPEEDUP)
            if controls.boost:
                pace *= self.boost
            right, up, ahead = controls.move
            step = right_vector(self.yaw) * right + np.array([0.0, 1.0, 0.0]) * up + self.forward * ahead
            self.position = self.position + step * pace * dt
        self._ground = ground_at(self.position[0], self.position[2])
        self.position[1] = max(self.position[1], self._ground + GROUND_MARGIN)

    def turn(self, look: tuple[float, float]) -> None:
        self.yaw = (self.yaw + look[0] * self.sensitivity) % (2 * math.pi)
        self.pitch = min(max(self.pitch - look[1] * self.sensitivity, -MAX_PITCH), MAX_PITCH)

    def place(self, eye: tuple[float, float, float], ground: float) -> None:
        """Put the camera at an eye position (walking), skipping flight and its ground margin."""
        self.glide = None
        self.position = np.array(eye, float)
        self._ground = ground

    def _advance_glide(self, dt: float) -> None:
        g = self.glide
        g.elapsed = min(g.elapsed + dt, g.duration)
        t = g.elapsed / g.duration
        eased = t * t * (3 - 2 * t)
        (p0, yaw0, pitch0), (p1, yaw1, pitch1) = g.start, g.end
        position = p0 + (p1 - p0) * eased
        # Rise along the way, so long hops arc over the terrain instead of skimming it
        position[1] += 0.25 * np.linalg.norm((p1 - p0)[[0, 2]]) * math.sin(math.pi * eased)
        self.position = position
        self.yaw = (yaw0 + (yaw1 - yaw0) * eased) % (2 * math.pi)
        self.pitch = pitch0 + (pitch1 - pitch0) * eased
        if g.elapsed >= g.duration:
            self.glide = None
