import numpy as np
import pytest

from common import SEED
from worldgen.generator.pipeline import ELEVATION, HYDROLOGY
from worldgen.generator.stages import elevation, hydrology


@pytest.fixture(scope="module")
def base(small_config):
    return elevation.generate(small_config.world, small_config.elevation, np.random.default_rng([SEED, ELEVATION])).heights


@pytest.fixture(scope="module")
def water(base, small_config):
    return hydrology.generate(
        base, small_config.world, small_config.hydrology, np.random.default_rng([SEED, HYDROLOGY])
    )


def test_world_has_lakes_and_rivers(water):
    assert water.lakes and water.rivers and water.river_path.any()


def test_every_cell_drains_to_a_sink(water):
    """Following flow from any cell ends at the sea or the world edge, never in a pit or a loop."""
    down = hydrology.downstream(water.flow_dir)
    position = np.arange(down.size)
    for _ in range(down.size):
        nxt = np.where(down[position] >= 0, down[position], position)
        if np.array_equal(nxt, position):
            break
        position = nxt
    else:
        pytest.fail("flow has a loop")
    edge = np.zeros(water.sea.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    assert (water.sea | edge).ravel()[position].all()


def test_river_water_never_rises_downstream(water):
    """Along the traced cells that lie in the (smoothed) channel."""
    down = hydrology.downstream(water.flow_dir)
    wet = ~np.isnan(water.water.ravel())
    cells = np.flatnonzero(water.river_path.ravel() & wet)
    nxt = down[cells]
    into_river = (nxt >= 0) & (water.river_path.ravel() & wet)[np.maximum(nxt, 0)]
    here, there = water.water.ravel()[cells[into_river]], water.water.ravel()[nxt[into_river]]
    assert (there <= here + 1e-9).all()


def test_rivers_are_continuous_ribbons(water):
    """Nearly every traced river cell is under water (the smoothed channel can pass a cell beside a
    few), and no dry vertex sits between two wet ones of a channel across a diagonal step."""
    wet = ~np.isnan(water.water) & ~water.sea & (water.lake_id == 0)
    traced = water.river_path & (water.lake_id == 0) & ~water.sea
    assert wet[traced].mean() > 0.95
    channel = water.river > 0
    c00, c01, c10, c11 = channel[:-1, :-1], channel[:-1, 1:], channel[1:, :-1], channel[1:, 1:]
    for diagonal, across in ((c00 & c11, c01 | c10), (c01 & c10, c00 | c11)):  # a bead: wet only on a diagonal
        assert across[diagonal].mean() > 0.99


def test_rivers_end_in_water_at_or_below_them(water):
    for river in water.rivers:
        (x0, z0), (x1, z1) = river[-2], river[-1]
        assert water.water[z1, x1] <= water.water[z0, x0] + 1e-9


def test_lakes_are_flat_at_their_spill_height(base, water, small_config):
    edge = np.zeros(base.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    plain, _ = hydrology.priority_flood(base, edge, small_config.hydrology.epsilon)
    for lake in water.lakes:
        cells = water.lake_id == lake.id
        assert cells.sum() == lake.area >= small_config.hydrology.lake_min_area
        np.testing.assert_array_equal(water.water[cells], lake.level)
        np.testing.assert_array_equal(plain[cells], lake.level)
        assert (water.elevation[cells] < lake.level).all()


def test_wet_cells_are_under_water(water):
    wet = ~np.isnan(water.water)
    assert (water.water[wet] > water.elevation[wet]).all()
    assert wet[water.sea].all() and wet[water.lake_id > 0].all() and wet[water.river > 0].all()


def test_priority_flood_fills_a_pit():
    ground = np.array([[5, 5, 5, 5, 5], [5, 1, 1, 1, 5], [5, 1, 0, 1, 4], [5, 1, 1, 1, 5], [5, 5, 5, 5, 5]], float)
    edge = np.zeros(ground.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    plain, raised = hydrology.priority_flood(ground, edge, 0.001)
    np.testing.assert_array_equal(plain[1:4, 1:4], 4)  # spills over the 4 on the east rim
    assert (raised[1:4, 1:4] > 4).all()
    flow = hydrology.flow_directions(raised, edge)
    assert flow[2, 3] == 0  # east, towards the spill


def test_meander_only_ever_flows_downhill():
    rng = np.random.default_rng(0)
    surface = np.add.outer(np.arange(32.0), np.zeros(32)) + rng.uniform(0, 0.01, (32, 32))  # even slope, rising south
    sinks = np.zeros(surface.shape, bool)
    sinks[0, :] = True
    plain = hydrology.flow_directions(surface, sinks)
    wiggly = hydrology.flow_directions(surface, sinks, rng.normal(0, 5, surface.shape), meander=3.0)
    assert (wiggly != plain).any()
    down = hydrology.downstream(wiggly)
    flows = down >= 0
    assert (surface.ravel()[down[flows]] < surface.ravel()[flows]).all()
    assert (wiggly[1:, :] != hydrology.NO_FLOW).all()  # every non-sink cell still has somewhere to go
