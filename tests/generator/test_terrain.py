"""Mountain ranges and terraces: the elevation stage's masks."""

import hashlib

import numpy as np
import pytest

from worldgen.generator.config import ConfigError, TerrainConfig, parse_config
from worldgen.generator.grid import min_filter
from worldgen.generator.noise import Perlin, ridged
from worldgen.generator.pipeline import ELEVATION, MOUNTAINS, TERRACES
from worldgen.generator.stages import elevation

# elevation.generate's heights for the default recipe and seed 7, from before mountains and terraces existed
DEFAULT_HEIGHTS = "feaa73afbc90ed918ff0b2c7b31ead779939faf2735375fcfa3e5855dd65b2d5"


def grid(n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    return x, z


def streams(seed: int) -> tuple[np.random.Generator, ...]:
    return tuple(np.random.default_rng([seed, stream]) for stream in (ELEVATION, MOUNTAINS, TERRACES))


def made(data: dict, seed: int = 3) -> elevation.Elevation:
    config = parse_config(data)
    return elevation.generate(config.world, config.elevation, *streams(seed))


def test_ridged_noise_lies_in_zero_to_one_and_repeats():
    x, z = grid()
    first = ridged(Perlin(np.random.default_rng(1)), x, z, frequency=0.02)
    again = ridged(Perlin(np.random.default_rng(1)), x, z, frequency=0.02)
    assert first.min() >= 0 and first.max() <= 1
    np.testing.assert_array_equal(first, again)


def test_sharper_ridges_have_narrower_crests():
    x, z = grid()
    shares = [(ridged(Perlin(np.random.default_rng(1)), x, z, frequency=0.02, sharpness=s) > 0.5).mean() for s in (1, 2, 4)]
    assert shares[0] > shares[1] > shares[2]


def test_without_mountains_or_terraces_the_heights_are_as_before():
    config = TerrainConfig()
    heights = elevation.generate(config.world, config.elevation, *streams(7)).heights
    assert hashlib.sha256(heights.tobytes()).hexdigest() == DEFAULT_HEIGHTS


@pytest.mark.parametrize("coverage", [0.2, 0.5, 0.8])
def test_the_range_mask_covers_its_share(coverage):
    x, z = grid()
    mask = elevation.coverage_mask(Perlin(np.random.default_rng(2)), x, z, 0.01, coverage)
    assert abs((mask > 0.5).mean() - coverage) < 0.05


def test_ranges_are_steeper_than_the_lowland(small_data):
    small_data["elevation"].update({"mask": "continent", "max_height": 120.0,
                                    "mountains": {"enabled": True, "wavelength": 40.0, "range_wavelength": 120.0,
                                                  "coverage": 0.4}})
    result = made(small_data)
    gz, gx = np.gradient(result.heights)
    slope = np.hypot(gx, gz)
    assert slope[result.mountains > 0.8].mean() > 2 * slope[result.mountains < 0.05].mean()
    assert result.heights.max() <= 120.0 + 1e-9  # max_height keeps its meaning


def test_ranges_rise_on_land_only(small_data):
    small_data["elevation"]["mountains"] = {"enabled": True, "wavelength": 40.0, "range_wavelength": 120.0}
    without = made({**small_data, "elevation": {**small_data["elevation"], "mountains": {}}})
    result = made(small_data)
    sea = without.heights < 0
    assert (result.mountains[sea] == 0).all()
    np.testing.assert_array_equal(result.heights < 0, sea)  # the coast stays where it was


def test_terraces_gather_heights_at_whole_steps():
    heights = np.linspace(0, 100, 10001)
    stepped = elevation.terrace(heights, 10.0, 1.0)
    level = stepped / 10.0
    assert (np.abs(level - np.rint(level)) < 0.01).mean() > 0.85
    assert np.all(np.diff(stepped) >= -1e-9)  # still climbing: a step never dips


def test_terraces_change_nothing_outside_their_mask(small_data):
    small_data["elevation"]["terraces"] = {"enabled": True, "step": 4.0, "wavelength": 60.0, "coverage": 0.4}
    without = made({**small_data, "elevation": {**small_data["elevation"], "terraces": {}}})
    result = made(small_data)
    outside = result.terraces == 0
    assert outside.any() and (~outside).any()
    np.testing.assert_array_equal(result.heights[outside], without.heights[outside])


def test_terraces_on_mountains_need_mountains(small_data):
    small_data["elevation"]["terraces"] = {"enabled": True, "where": "mountains"}
    with pytest.raises(ConfigError, match="mountains are not enabled"):
        parse_config(small_data)


def test_min_filter_is_the_least_value_in_the_window():
    values = np.random.default_rng(4).uniform(0, 100, (37, 23))
    for radius in (0, 1, 3, 12):
        padded = np.pad(values, radius, mode="edge")
        expected = np.array([[padded[z:z + 2 * radius + 1, x:x + 2 * radius + 1].min() for x in range(23)]
                             for z in range(37)])
        np.testing.assert_array_equal(min_filter(values, radius), expected)
