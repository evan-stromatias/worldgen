"""Vectorised 2D gradient (Perlin) noise and fractal sums of it.

Written here rather than taken from a library so that a world's output never changes with a
dependency upgrade.
"""

import numpy as np

# Each octave samples a different, far-away region of the noise, so octaves do not line up
_OCTAVE_OFFSET = (31.7, 47.3)


class Perlin:
    """2D gradient noise in [-1, 1], with its own permutation and gradients drawn from `rng`."""

    def __init__(self, rng: np.random.Generator):
        self.perm = np.tile(rng.permutation(256), 2)
        angles = rng.uniform(0, 2 * np.pi, 256)
        self.gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def __call__(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x0, z0 = np.floor(x), np.floor(z)
        fx, fz = x - x0, z - z0
        ix, iz = x0.astype(np.int64) & 255, z0.astype(np.int64) & 255

        def corner(cx: int, cz: int) -> np.ndarray:
            g = self.gradients[self.perm[self.perm[ix + cx] + iz + cz]]
            return g[..., 0] * (fx - cx) + g[..., 1] * (fz - cz)

        u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)  # quintic fade: no creases at cell borders
        v = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
        top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u
        bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u
        return (top + (bottom - top) * v) * np.sqrt(2)  # unit gradients reach ±1/√2


def fbm(
    noise: Perlin,
    x: np.ndarray,
    z: np.ndarray,
    *,
    frequency: float,
    octaves: int = 1,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal Brownian motion: octaves of `noise` at rising frequency and falling amplitude, roughly in [-1, 1]."""
    total = np.zeros(np.shape(x))
    amplitude, weight = 1.0, 0.0
    for octave in range(octaves):
        total += amplitude * noise(
            x * frequency + _OCTAVE_OFFSET[0] * octave, z * frequency + _OCTAVE_OFFSET[1] * octave
        )
        weight += amplitude
        frequency *= lacunarity
        amplitude *= gain
    return total / weight


RIDGE_WEIGHT = 2.0  # how strongly an octave's crest lets the next octave's detail through


def ridged(
    noise: Perlin,
    x: np.ndarray,
    z: np.ndarray,
    *,
    frequency: float,
    octaves: int = 6,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    sharpness: float = 2.0,
) -> np.ndarray:
    """Ridged multifractal noise (Musgrave), roughly in [0, 1]: sharp crests where the noise crosses zero.

    - Each octave is `(1 - |n|) ** sharpness`, so it peaks in a crease, not a dome; a higher `sharpness` narrows the
      crests.
    - Each octave is weighted by the one before, so fine detail gathers on the ridges and the valleys stay smooth.
    """
    total = np.zeros(np.shape(x))
    weight = np.ones(np.shape(x))
    amplitude, norm = 1.0, 0.0
    for octave in range(octaves):
        n = noise(x * frequency + _OCTAVE_OFFSET[0] * octave, z * frequency + _OCTAVE_OFFSET[1] * octave)
        signal = (1 - np.abs(n)) ** sharpness * weight
        weight = np.clip(signal * RIDGE_WEIGHT, 0, 1)
        total += amplitude * signal
        norm += amplitude
        frequency *= lacunarity
        amplitude *= gain
    return total / norm
