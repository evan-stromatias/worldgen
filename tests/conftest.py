"""Small terrains for tests: the fixtures that run the recipe in `common.py`."""

import copy

import pytest

from common import SEED, SMALL
from worldgen.generator.config import TerrainConfig, parse_config
from worldgen.generator.terrain import TerrainData
from worldgen.generator.terrain_io import generate_terrain


@pytest.fixture
def small_data() -> dict:
    """The small recipe as parsed YAML, safe to modify."""
    return copy.deepcopy(SMALL)


@pytest.fixture(scope="session")
def small_config() -> TerrainConfig:
    return parse_config(copy.deepcopy(SMALL))


@pytest.fixture(scope="session")
def small_terrain(small_config) -> TerrainData:
    return generate_terrain(small_config, SEED)
