import numpy as np

from worldgen.generator.stages import biomes
from worldgen.generator.config import parse_config


def inputs(**values) -> biomes.CellInputs:
    shape = np.shape(next(iter(values.values())))
    fields = {k: np.zeros(shape) for k in ("elevation", "temperature", "moisture", "slope", "sediment", "relief")}
    fields["water"] = np.zeros(shape, np.int8)
    fields.update({k: np.asarray(v, dtype=fields[k].dtype) for k, v in values.items()})
    return biomes.CellInputs(**fields)


def rules(blend_width: float = 0.0):
    return parse_config({"biomes": {"blend_width": blend_width, "rules": [
        {"id": "cliff", "when": {"slope": [40, 90]}, "ground": {"color": [100, 100, 100]}},
        {"id": "high", "when": {"elevation": [10, 20]}, "ground": {"color": [200, 200, 200]}},
        {"id": "lake", "when": {"water": "lake"}, "ground": {"color": [0, 0, 255]}},
        {"id": "default", "ground": {"color": [0, 100, 0]}},
    ]}}).biomes


def test_first_match_wins_and_catch_all_covers_the_rest():
    cells = inputs(slope=[50, 50, 0, 0, 0], elevation=[15, 0, 15, 0, 20])
    index, _ = biomes.classify(cells, rules(), np.random.default_rng(0))
    assert index.tolist() == [0, 0, 1, 3, 1]  # ranges are inclusive: 20 is still "high"


def test_water_condition():
    cells = inputs(water=[0, 3, 1])
    index, _ = biomes.classify(cells, rules(), np.random.default_rng(0))
    assert index.tolist() == [3, 2, 3]


def test_colour_is_exact_inside_and_blends_at_edges():
    cells = inputs(elevation=[15, 10, 9.0, 21.0])
    _, color = biomes.classify(cells, rules(blend_width=0.2), np.random.default_rng(0))
    assert color[0].tolist() == [200, 200, 200]  # well inside "high"
    assert color[1].tolist() == [100, 150, 100]  # on the edge: half "high", half "default"
    assert color[2].tolist() == [0, 100, 0]  # past the blend ramp
    assert color[3].tolist() == [0, 100, 0]


def test_domain_limits_are_not_borders():
    biome_rules = parse_config({"biomes": {"blend_width": 0.2, "rules": [
        {"id": "wet", "when": {"moisture": [0.5, 1.0]}, "ground": {"color": [0, 0, 200]}},
        {"id": "default", "ground": {"color": [0, 0, 0]}},
    ]}}).biomes
    _, color = biomes.classify(inputs(moisture=[1.0]), biome_rules, np.random.default_rng(0))
    assert color[0].tolist() == [0, 0, 200]


def test_water_kind_takes_the_majority_of_corners():
    sea = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], bool)
    none = np.zeros((3, 3), bool)
    kinds = biomes.water_kind_cells(sea, none, none)
    assert kinds.tolist() == [[1, 0], [0, 0]]  # three sea corners; then one


def test_sediment_and_relief_conditions():
    cfg = parse_config({"biomes": {"blend_width": 0.0, "rules": [
        {"id": "alluvial", "when": {"sediment": [0.5, 100]}, "ground": {"color": [1, 1, 1]}},
        {"id": "plateau", "when": {"relief": [20, 1000]}, "ground": {"color": [2, 2, 2]}},
        {"id": "default", "ground": {"color": [0, 0, 0]}},
    ]}}).biomes
    cells = inputs(sediment=[0.0, 0.4, 0.6, 0.0], relief=[0, 0, 30, 30])
    index, _ = biomes.classify(cells, cfg, np.random.default_rng(0))
    assert index.tolist() == [2, 2, 0, 1]
