import numpy as np

from worldgen.cave.config import parse_config
from worldgen.cave.pipeline import CAVE_STAGES, CaveProgress, generate, run_stage

TEST_CONFIG = {
    "world": {"size_cells": 16, "patch_cells": 16, "cell_size": 1.0},
    "elevation": {"noise": {"frequency": 0.05}},
    "cavern": {
        "clearance": 14.0,
        "ceiling_mode": "inverted",
        "min_passage": 2.0,
        "perimeter_wall": True,
        "wall_thickness": 1,
    },
    "hydrology": {"enabled": True, "water_table": -1.0},
    "climate": {"temperature_base": 11.0, "geothermal_gradient": 0.03, "ambient_humidity": 0.85},
}


def test_cave_pipeline_individual_stages():
    cfg = parse_config(TEST_CONFIG)
    seed = 42

    progress = CaveProgress()
    stages = {s.name: s for s in CAVE_STAGES}

    # 1. Elevation
    progress = run_stage(progress, stages["elevation"], cfg, seed)
    assert progress.floor_elev is not None
    assert progress.floor_elev.shape == (17, 17)

    # 2. Erosion
    progress = run_stage(progress, stages["erosion"], cfg, seed)
    assert progress.eroded_floor is not None
    assert progress.sediment is not None
    assert progress.sediment.shape == (16, 16)

    # 3. Cavern
    progress = run_stage(progress, stages["cavern"], cfg, seed)
    assert progress.ceiling_elev is not None
    assert progress.clearance is not None
    assert progress.pillar_mask is not None
    assert progress.clearance.shape == (16, 16)
    assert progress.pillar_mask.shape == (16, 16)

    # Perimeter wall check
    assert np.all(progress.pillar_mask[0, :] == 1)
    assert np.all(progress.pillar_mask[-1, :] == 1)

    # 4. Hydrology
    progress = run_stage(progress, stages["hydrology"], cfg, seed)
    assert progress.water is not None
    assert progress.water.shape == (17, 17)

    # 5. Climate
    progress = run_stage(progress, stages["climate"], cfg, seed)
    assert progress.temperature is not None
    assert progress.moisture is not None
    assert progress.temperature.shape == (16, 16)
    assert progress.moisture.shape == (16, 16)

    # 6. Materials
    progress = run_stage(progress, stages["materials"], cfg, seed)
    assert progress.material is not None
    assert progress.color is not None
    assert progress.cave is not None
    assert progress.color.shape == (16, 16, 3)


def test_cave_pipeline_determinism():
    cfg = parse_config(TEST_CONFIG)
    seed = 77

    # Run all at once
    res_batch = generate(cfg, seed)

    # Run step-by-step
    res_step = CaveProgress()
    for stage in CAVE_STAGES:
        res_step = run_stage(res_step, stage, cfg, seed)

    assert res_batch.cave is not None
    assert res_step.cave is not None
    np.testing.assert_array_equal(res_batch.cave.floor_elevation, res_step.cave.floor_elevation)
    np.testing.assert_array_equal(res_batch.cave.ceiling_elevation, res_step.cave.ceiling_elevation)
    np.testing.assert_array_equal(res_batch.cave.clearance, res_step.cave.clearance)
    np.testing.assert_array_equal(res_batch.cave.pillar_mask, res_step.cave.pillar_mask)
    np.testing.assert_array_equal(res_batch.cave.color, res_step.cave.color)


def test_cave_pipeline_disabled_stages():
    cfg_dict = dict(TEST_CONFIG)
    cfg_dict["erosion"] = {"enabled": False}
    cfg_dict["hydrology"] = {"enabled": False}
    cfg = parse_config(cfg_dict)
    seed = 13

    res = generate(cfg, seed)
    assert res.cave is not None
    assert np.all(res.sediment == 0.0)
    assert np.all(np.isnan(res.water))
