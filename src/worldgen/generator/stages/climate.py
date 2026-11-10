"""Stage 4: temperature, moisture and slope, the inputs to the biome rules."""

from dataclasses import dataclass

import numpy as np

from worldgen.generator.config import ClimateSection, WorldSection
from worldgen.generator.grid import box_blur
from worldgen.generator.noise import Perlin, fbm

NOISE_CYCLES = 4.0  # cycles of temperature and moisture noise across the world
NOISE_OCTAVES = 3


@dataclass(frozen=True, slots=True)
class Climate:
    temperature: np.ndarray  # °C
    moisture: np.ndarray  # 0-1
    slope: np.ndarray  # degrees


def generate(
    elevation: np.ndarray,
    wet: np.ndarray,
    rain: np.ndarray,
    world: WorldSection,
    cfg: ClimateSection,
    rng: np.random.Generator,
) -> Climate:
    """`wet` marks sea, lake and river vertices; `rain` is the hydrology stage's precipitation."""
    n = elevation.shape[0]
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    frequency = NOISE_CYCLES / world.size_cells
    temperature_noise, moisture_noise = Perlin(rng), Perlin(rng)

    north = 1 - z / world.size_cells  # 1 on the north edge, 0 on the south edge
    above_sea = np.maximum(elevation - world.sea_level, 0)
    temperature = (
        cfg.base_temperature
        - cfg.latitude_gradient * north
        - cfg.lapse_rate * above_sea / 1000
        + cfg.temperature_noise * fbm(temperature_noise, x, z, frequency=frequency, octaves=NOISE_OCTAVES)
    )

    # Near water: a blurred water mask, so coasts and riverbanks are moist and it fades inland
    m = cfg.moisture
    rain_share = (rain - rain.min()) / max(np.ptp(rain), 1e-12)
    near_water = np.clip(2 * box_blur(wet.astype(np.float64), max(1, round(m.water_distance))), 0, 1)
    moisture = np.clip(
        m.rain_weight * rain_share
        + (1 - m.rain_weight) * near_water
        + m.noise * fbm(moisture_noise, x, z, frequency=frequency, octaves=NOISE_OCTAVES),
        0,
        1,
    )

    return Climate(temperature, moisture, slope_degrees(elevation, world.cell_size))


def slope_degrees(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """The ground's slope in degrees, per vertex of the elevation grid."""
    gz, gx = np.gradient(elevation, cell_size)
    return np.degrees(np.arctan(np.hypot(gx, gz)))
