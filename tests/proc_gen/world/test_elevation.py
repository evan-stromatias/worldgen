import numpy as np

from worldgen.proc_gen.world import elevation
from worldgen.proc_gen.world.config import ArchipelagoSection, parse_config


def test_islands_are_placed_apart_and_inside_the_world():
    cfg = ArchipelagoSection(islands=10, size=(0.05, 0.2), spacing=1.0)
    islands = elevation.place_islands(cfg, reach=0.95, rng=np.random.default_rng(1))
    assert islands[0] == (0.5, 0.5, 0.2)
    assert 2 <= len(islands) <= 10
    for i, (x, z, r) in enumerate(islands):
        assert 0.05 <= r <= 0.2
        assert r * 0.95 <= x <= 1 - r * 0.95 and r * 0.95 <= z <= 1 - r * 0.95
        for ox, oz, other in islands[:i]:
            assert np.hypot(x - ox, z - oz) >= (r + other) * 0.95


def test_archipelago_has_land_at_the_centre_and_sea_at_the_edges(small_data):
    small_data["elevation"]["mask"] = "archipelago"
    config = parse_config(small_data)
    heights = elevation.generate(config.world, config.elevation, np.random.default_rng(0)).heights
    sea = config.world.sea_level
    assert heights[64, 64] > sea
    assert (heights[0, :] < sea).all() and (heights[:, 0] < sea).all()
    assert 0.1 < (heights > sea).mean() < 0.9
