"""The erosion stage."""

import math

import numpy as np
import pytest

from worldgen.generator.config import ErosionSection, parse_config
from worldgen.generator.grid import distance_to
from worldgen.generator.pipeline import STAGES, Progress, generate, run_stage
from worldgen.generator.stages import erosion

SETTINGS = {"enabled": True, "droplets": 0.5, "levels": 2, "lifetime": 20, "thermal_iterations": 20}


@pytest.fixture(scope="module")
def recipe():
    from common import SMALL

    data = {**SMALL, "elevation": {**SMALL["elevation"], "max_height": 15.0}, "erosion": SETTINGS}
    return parse_config(data)


@pytest.fixture(scope="module")
def eroded(recipe):
    progress = Progress()
    for stage in STAGES[:2]:
        progress = run_stage(stage, progress, recipe, 3)
    return progress


def test_level_factors():
    assert erosion.level_factors(1) == [1]
    assert erosion.level_factors(3) == [8, 4, 1]


def test_erosion_is_deterministic(recipe, eroded):
    again = run_stage(STAGES[1], Progress(heights=eroded.heights, terraces=eroded.terraces), recipe, 3)
    np.testing.assert_array_equal(again.erosion.heights, eroded.erosion.heights)
    np.testing.assert_array_equal(again.erosion.sediment, eroded.erosion.sediment)


def test_erosion_changes_the_ground_and_keeps_it_finite(eroded):
    assert np.isfinite(eroded.erosion.heights).all()
    assert np.abs(eroded.erosion.heights - eroded.heights).max() > 0.1


def test_erosion_does_not_drown_the_coast(eroded):
    before, after = (eroded.heights < 0).mean(), (eroded.erosion.heights < 0).mean()
    assert after <= before + 0.01


def test_sediment_lies_on_gentle_ground(eroded, recipe):
    sediment, ground = eroded.erosion.sediment, eroded.erosion.heights
    assert (sediment >= 0).all() and sediment.max() > 0
    gz, gx = np.gradient(ground, recipe.world.cell_size)
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    # Inland: droplets reaching the sea drop all they carry on its steep shelf, and on the coast beside it
    land = distance_to(ground <= recipe.world.sea_level, 1.0, 6.0) > 5
    deep = land & (sediment > np.percentile(sediment[land & (sediment > 0)], 75))
    assert np.median(slope[deep]) < np.median(slope[land])


def test_sliding_leaves_no_slope_steeper_than_it_holds():
    n = 65
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    heights = np.maximum(0, 30 - 1.5 * np.hypot(x - 32, z - 32))  # a 56° cone
    cfg = ErosionSection(talus_angle=30, thermal_iterations=1600)  # it creeps there: the cone is a wide one
    erosion.thermal(heights, None, 1.0, -1.0, cfg)
    limit = math.tan(math.radians(30))
    for dz, dx in erosion.NEIGHBOURS:
        drop = heights - np.roll(heights, (-dz, -dx), axis=(0, 1))
        assert (drop[2:-2, 2:-2] / math.hypot(dz, dx)).max() < limit * 1.05


def test_rock_holds_steeper_slopes():
    n = 65
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    cone = np.maximum(0, 30 - 1.5 * np.hypot(x - 32, z - 32))
    cfg = ErosionSection(talus_angle=30, rock_talus_angle=70, thermal_iterations=100)
    soft, rock = cone.copy(), cone.copy()
    erosion.thermal(soft, None, 1.0, -1.0, cfg)
    erosion.thermal(rock, np.ones_like(cone), 1.0, -1.0, cfg)
    np.testing.assert_allclose(rock, cone)  # 56° stands on rock
    assert soft.max() < cone.max() - 1


def test_hydrology_runs_on_eroded_ground(recipe):
    world = generate(recipe, 3)
    assert world.rivers
    for river in world.rivers:
        (x0, z0), (x1, z1) = river[-2], river[-1]
        assert not np.isnan(world.water[z1, x1])  # ends in water
        if not np.isnan(world.water[z0, x0]):  # the smoothed channel can pass a cell beside a traced one
            assert world.water[z1, x1] <= world.water[z0, x0] + 1e-9
    assert world.sediment.shape == world.slope.shape and world.sediment.max() > 0


def test_without_erosion_the_stage_is_skipped(small_config):
    assert not STAGES[1].applies(small_config)


def test_smoothing_leaves_slopes_less_rippled(recipe, eroded):
    """Droplets crossing a slope in parallel cut a washboard about two vertices apart; each level's change is
    blurred before it is added. Measured as the change less its own blur."""
    from worldgen.generator.grid import box_blur

    def ripple(progress) -> float:
        change = progress.erosion.heights - progress.heights
        return float(np.sqrt(np.mean((change - box_blur(change, 2)) ** 2)))

    rough = recipe.model_copy(update={"erosion": recipe.erosion.model_copy(update={"smoothing": 0})})
    unsmoothed = run_stage(STAGES[1], Progress(heights=eroded.heights, terraces=eroded.terraces), rough, 3)
    assert ripple(eroded) < 0.8 * ripple(unsmoothed)
