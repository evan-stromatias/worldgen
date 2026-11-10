"""Run the stages in order on the whole terrain: `generate` runs them all, `run_stage` one at a time."""

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from worldgen.proc_gen.world import biomes, climate, elevation, erosion, hydrology
from worldgen.proc_gen.world.climate import Climate
from worldgen.proc_gen.world.config import TerrainConfig
from worldgen.proc_gen.world.erosion import Erosion
from worldgen.proc_gen.world.grid import to_cells
from worldgen.proc_gen.world.hydrology import Hydrology
from worldgen.proc_gen.world.terrain import TerrainData

# Each stage draws from its own generator, so changing one never reshuffles another's random numbers
ELEVATION, HYDROLOGY, CLIMATE, BIOMES = 1, 3, 4, 5  # streams, not order
MOUNTAINS, TERRACES, EROSION = 10, 11, 12  # elevation's ranges and terraces, and the erosion stage


@dataclass(frozen=True, slots=True)
class Progress:
    """What the stages have made so far. Each stage reads what the stages before it made and adds its own.

    The stages up to climate keep their results for later stages (and for the terrain generator UI to
    draw); from biomes on, `world` is the terrain as it stands.
    """

    heights: np.ndarray | None = None  # elevation: the heightfield, vertex grid, metres
    mountains: np.ndarray | None = None  # elevation: the range mask, vertex grid, 0-1; None without mountains
    terraces: np.ndarray | None = None  # elevation: the terrace mask, vertex grid, 0-1; None without terraces
    erosion: Erosion | None = None  # the eroded ground and its sediment; None when the recipe has no erosion
    hydrology: Hydrology | None = None  # carved ground, water, lakes and rivers
    climate: Climate | None = None  # temperature, moisture and slope
    world: TerrainData | None = None  # from biomes on


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    stream: int  # its random numbers come from np.random.default_rng([seed, stream])
    sections: tuple[str, ...]  # the recipe sections it reads
    run: Callable[[Progress, TerrainConfig, int], Progress]  # (what came before, recipe, seed) -> with its own
    applies: Callable[[TerrainConfig], bool] = lambda config: True  # whether the recipe wants it at all
    skipped: str = ""  # why it does not run when it does not run, as the UI says it


def _elevation(p: Progress, config: TerrainConfig, seed: int) -> Progress:
    made = elevation.generate(
        config.world, config.elevation, np.random.default_rng([seed, ELEVATION]),
        np.random.default_rng([seed, MOUNTAINS]), np.random.default_rng([seed, TERRACES]),
    )
    return dataclasses.replace(p, heights=made.heights, mountains=made.mountains, terraces=made.terraces)


def _erosion(p: Progress, config: TerrainConfig, seed: int) -> Progress:
    """The eroded ground; nothing at all when the recipe has erosion off, however the stage is called."""
    if not config.erosion.enabled:
        return dataclasses.replace(p, erosion=None)
    rng = np.random.default_rng([seed, EROSION])
    return dataclasses.replace(p, erosion=erosion.generate(p.heights, p.terraces, config.world, config.erosion, rng))


def ground(p: Progress) -> np.ndarray:
    """The heightfield hydrology starts from: the eroded one, or elevation's without erosion."""
    return p.erosion.heights if p.erosion is not None else p.heights


def _hydrology(p: Progress, config: TerrainConfig, seed: int) -> Progress:
    rng = np.random.default_rng([seed, HYDROLOGY])
    return dataclasses.replace(p, hydrology=hydrology.generate(ground(p), config.world, config.hydrology, rng))


def _climate(p: Progress, config: TerrainConfig, seed: int) -> Progress:
    water = p.hydrology
    wet = ~np.isnan(water.water)
    rng = np.random.default_rng([seed, CLIMATE])
    return dataclasses.replace(
        p, climate=climate.generate(water.elevation, wet, water.rain, config.world, config.climate, rng)
    )


def _biomes(p: Progress, config: TerrainConfig, seed: int) -> Progress:
    water, weather, world = p.hydrology, p.climate, config.world
    cells = to_cells(water.elevation)
    size = cells.shape
    sediment = to_cells(p.erosion.sediment.astype(np.float64)) if p.erosion is not None else np.zeros(size)
    inputs = biomes.CellInputs(
        elevation=cells - world.sea_level,
        temperature=to_cells(weather.temperature),
        moisture=to_cells(weather.moisture),
        slope=to_cells(weather.slope),
        water=biomes.water_kind_cells(water.sea, water.lake_id > 0, water.river > 0),
        sediment=sediment,
        relief=biomes.local_relief(water.elevation, world.cell_size, config.biomes.relief_radius),
    )
    biome, color = biomes.classify(inputs, config.biomes, np.random.default_rng([seed, BIOMES]))
    generated = TerrainData(
        config=config,
        seed=seed,
        elevation=water.elevation,
        water=water.water,
        sea=water.sea,
        lake_id=water.lake_id,
        river=water.river,
        flow_dir=water.flow_dir,
        biome=biome,
        color=color,
        temperature=inputs.temperature.astype(np.float32),
        moisture=inputs.moisture.astype(np.float32),
        sediment=inputs.sediment.astype(np.float32),
        lakes=water.lakes,
        rivers=water.rivers,
    )
    return dataclasses.replace(p, world=generated)


# In order. Changing one stage's recipe sections changes its output and that of every stage after it.
STAGES = (
    Stage("elevation", ELEVATION, ("world", "elevation"), _elevation),
    Stage("erosion", EROSION, ("world", "erosion"), _erosion, lambda config: config.erosion.enabled, "erosion off"),
    Stage("hydrology", HYDROLOGY, ("world", "hydrology"), _hydrology),
    Stage("climate", CLIMATE, ("world", "climate"), _climate),
    Stage("biomes", BIOMES, ("world", "biomes"), _biomes),
)


def run_stage(
    stage: Stage, progress: Progress, config: TerrainConfig, seed: int,
    report: Callable[[str, float], None] = lambda stage, seconds: None,
) -> Progress:
    """Run one stage on what the stages before it made; `report` is told its name and duration."""
    start = time.perf_counter()
    progress = stage.run(progress, config, seed)
    report(stage.name, time.perf_counter() - start)
    return progress


def generate(
    config: TerrainConfig,
    seed: int,
    report: Callable[[str, float], None] = lambda stage, seconds: None,
) -> TerrainData:
    """Build a terrain from `config` and `seed`; `report` is told each stage's name and duration."""
    progress = Progress()
    for stage in STAGES:
        if not stage.applies(config):
            continue
        progress = run_stage(stage, progress, config, seed, report)
    return progress.world
