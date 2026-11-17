from pathlib import Path

import pytest

from worldgen.cave.config import (
    DEFAULT_CAVE_RECIPE,
    CaveConfig,
    ConfigError,
    load_config,
    parse_config,
)


def test_default_recipe_loads_and_validates():
    config = load_config(DEFAULT_CAVE_RECIPE)
    assert isinstance(config, CaveConfig)
    assert config.cavern.clearance > 0
    assert config.cavern.ceiling_mode in ("inverted", "offset")
    assert config.world.size_cells >= 16


def test_custom_recipe_parsing():
    data = {
        "cavern": {
            "clearance": 25.0,
            "ceiling_mode": "offset",
            "ceiling_scale": 1.5,
            "min_passage": 3.0,
            "perimeter_wall": True,
            "wall_thickness": 3,
        },
        "hydrology": {
            "enabled": True,
            "water_table": -1.5,
        },
        "climate": {
            "temperature_base": 14.0,
        },
    }
    cfg = parse_config(data)
    assert cfg.cavern.clearance == 25.0
    assert cfg.cavern.ceiling_mode == "offset"
    assert cfg.cavern.wall_thickness == 3
    assert cfg.hydrology.water_table == -1.5


def test_rejects_seed_in_file():
    with pytest.raises(ConfigError, match="seed"):
        parse_config({"seed": 42})


def test_rejects_invalid_values():
    with pytest.raises(ConfigError):
        parse_config({"cavern": {"clearance": -5.0}})

    with pytest.raises(ConfigError):
        parse_config({"cavern": {"ceiling_mode": "invalid_mode"}})

    with pytest.raises(ConfigError):
        parse_config("not a dict")


def test_load_config_nonexistent(tmp_path: Path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nonexistent.yaml")
