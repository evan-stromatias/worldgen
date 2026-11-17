"""Cave generation pipeline: run stages in order on a subterranean cavern.

Stages can be executed sequentially or individually with caching:
elevation -> erosion -> cavern -> hydrology -> climate -> materials
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from worldgen.generator.grid import to_cells
from worldgen.generator.stages import biomes, climate, elevation, erosion

if TYPE_CHECKING:
    from worldgen.cave.config import CaveConfig
    from worldgen.cave.generator import CaveData

# RNG stream indices for generator stages
STAGE_ELEVATION = 1
STAGE_EROSION = 2
STAGE_MATERIALS = 3
STAGE_MOUNTAINS = 10
STAGE_TERRACES = 11


@dataclass(frozen=True, slots=True)
class CaveProgress:
    """Intermediates produced by cave generation stages.

    Each stage reads inputs from preceding stages and produces its own outputs.
    """

    floor_elev: np.ndarray | None = None  # Stage 1: initial floor heightfield (vertex grid)
    terraces: np.ndarray | None = None  # Stage 1: terrace mask (vertex grid)
    mask: np.ndarray | None = None  # Stage 1: land mask (vertex grid)
    eroded_floor: np.ndarray | None = None  # Stage 2: floor after erosion (vertex grid)
    sediment: np.ndarray | None = None  # Stage 2: sediment depth (cell grid)
    ceiling_elev: np.ndarray | None = None  # Stage 3: ceiling heightfield (vertex grid)
    clearance: np.ndarray | None = None  # Stage 3: vertical headroom (cell grid)
    pillar_mask: np.ndarray | None = None  # Stage 3: solid pillar/wall mask (cell grid, uint8)
    water: np.ndarray | None = None  # Stage 4: water surface (vertex grid, NaN where dry)
    temperature: np.ndarray | None = None  # Stage 5: air/rock temperature (cell grid, °C)
    moisture: np.ndarray | None = None  # Stage 5: relative moisture (cell grid, 0-1)
    material: np.ndarray | None = None  # Stage 6: material rule indices (cell grid)
    color: np.ndarray | None = None  # Stage 6: RGB vertex/cell colors (cell grid)
    cave: CaveData | None = None  # Final complete CaveData


def current_floor(p: CaveProgress) -> np.ndarray:
    """Return the active floor heightmap (eroded if available, else raw elevation)."""
    if p.eroded_floor is not None:
        return p.eroded_floor
    if p.floor_elev is not None:
        return p.floor_elev
    raise ValueError("No floor elevation available in CaveProgress")


@dataclass(frozen=True, slots=True)
class CaveStage:
    name: str
    stream: int
    sections: tuple[str, ...]
    run: Callable[[CaveProgress, CaveConfig, int], CaveProgress]
    applies: Callable[[CaveConfig], bool] = lambda config: True
    skipped: str = ""


def _stage_elevation(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    rng = np.random.default_rng([seed, STAGE_ELEVATION])
    rng_mountains = np.random.default_rng([seed, STAGE_MOUNTAINS])
    rng_terraces = np.random.default_rng([seed, STAGE_TERRACES])
    elev = elevation.generate(config.world, config.elevation, rng, rng_mountains, rng_terraces)
    return dataclasses.replace(
        p,
        floor_elev=elev.heights.copy(),
        terraces=elev.terraces,
        mask=elev.mask,
    )


def _stage_erosion(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    assert p.floor_elev is not None
    size_cells = config.world.size_cells
    if not config.erosion.enabled:
        return dataclasses.replace(
            p,
            eroded_floor=p.floor_elev.copy(),
            sediment=np.zeros((size_cells, size_cells), np.float32),
        )

    rng = np.random.default_rng([seed, STAGE_EROSION])
    eroded = erosion.generate(p.floor_elev, p.terraces, config.world, config.erosion, rng)
    return dataclasses.replace(
        p,
        eroded_floor=eroded.heights,
        sediment=to_cells(eroded.sediment).astype(np.float32),
    )


def _stage_cavern(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    floor = current_floor(p)
    h_rot = np.rot90(floor, 2)
    cavern = config.cavern

    if cavern.ceiling_mode == "inverted":
        h_base = float(np.mean(floor))
        ceiling = h_base + cavern.clearance + (h_base - h_rot) * cavern.ceiling_scale
    else:
        ceiling = h_rot + cavern.clearance

    clearance_vertex = ceiling - floor
    clearance_cells = to_cells(clearance_vertex).astype(np.float32)

    pillar_mask = (clearance_cells < cavern.min_passage).astype(np.uint8)
    if cavern.perimeter_wall:
        t = cavern.wall_thickness
        pillar_mask[:t, :] = 1
        pillar_mask[-t:, :] = 1
        pillar_mask[:, :t] = 1
        pillar_mask[:, -t:] = 1

    return dataclasses.replace(
        p,
        ceiling_elev=ceiling,
        clearance=clearance_cells,
        pillar_mask=pillar_mask,
    )


def _stage_hydrology(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    floor = current_floor(p)
    water = np.full_like(floor, np.nan)
    if config.hydrology.enabled:
        wt = config.hydrology.water_table
        wet = floor < wt
        water[wet] = wt

    return dataclasses.replace(p, water=water)


def _stage_climate(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    floor = current_floor(p)
    floor_cells = to_cells(floor)
    depth = np.max(floor) - floor_cells
    temperature = (config.climate.temperature_base + config.climate.geothermal_gradient * depth).astype(np.float32)
    moisture = np.full_like(temperature, config.climate.ambient_humidity)

    if config.hydrology.enabled and p.water is not None:
        wet_cells = to_cells((~np.isnan(p.water)).astype(np.float64))
        moisture = np.clip(moisture + 0.15 * wet_cells.astype(np.float32), 0.0, 1.0)

    return dataclasses.replace(p, temperature=temperature, moisture=moisture)


def _stage_materials(p: CaveProgress, config: CaveConfig, seed: int) -> CaveProgress:
    from worldgen.cave.generator import CaveData

    floor = current_floor(p)
    assert p.ceiling_elev is not None
    assert p.clearance is not None
    assert p.pillar_mask is not None
    assert p.water is not None
    assert p.temperature is not None
    assert p.moisture is not None
    sediment = p.sediment if p.sediment is not None else np.zeros((config.world.size_cells, config.world.size_cells), np.float32)

    world = config.world
    floor_cells = to_cells(floor)
    cell_inputs = biomes.CellInputs(
        elevation=floor_cells - world.sea_level,
        temperature=p.temperature.astype(np.float64),
        moisture=p.moisture.astype(np.float64),
        slope=to_cells(climate.slope_degrees(floor, world.cell_size)).astype(np.float64),
        water=biomes.water_kind_cells(
            np.zeros_like(p.water, bool),
            ~np.isnan(p.water),
            np.zeros_like(p.water, bool),
        ),
        sediment=sediment.astype(np.float64),
        relief=biomes.local_relief(floor, world.cell_size, config.materials.relief_radius).astype(np.float64),
    )

    rng_materials = np.random.default_rng([seed, STAGE_MATERIALS])
    material, color = biomes.classify(cell_inputs, config.materials, rng_materials)

    cave = CaveData(
        seed=seed,
        config=config,
        floor_elevation=floor,
        ceiling_elevation=p.ceiling_elev,
        clearance=p.clearance,
        pillar_mask=p.pillar_mask,
        water=p.water,
        material=material,
        color=color,
        temperature=p.temperature,
        moisture=p.moisture,
        sediment=sediment,
    )

    return dataclasses.replace(p, material=material, color=color, cave=cave)


CAVE_STAGES: tuple[CaveStage, ...] = (
    CaveStage("elevation", STAGE_ELEVATION, ("world", "elevation"), _stage_elevation),
    CaveStage(
        "erosion",
        STAGE_EROSION,
        ("erosion",),
        _stage_erosion,
        applies=lambda config: config.erosion.enabled,
        skipped="erosion disabled",
    ),
    CaveStage("cavern", 0, ("cavern",), _stage_cavern),
    CaveStage(
        "hydrology",
        0,
        ("hydrology",),
        _stage_hydrology,
        applies=lambda config: config.hydrology.enabled,
        skipped="hydrology disabled",
    ),
    CaveStage("climate", 0, ("climate",), _stage_climate),
    CaveStage("materials", STAGE_MATERIALS, ("materials",), _stage_materials),
)


def run_stage(p: CaveProgress, stage: CaveStage, config: CaveConfig, seed: int) -> CaveProgress:
    """Run one stage with its inputs from `p`."""
    return stage.run(p, config, seed)


def generate(
    config: CaveConfig,
    seed: int,
    report: Callable[[str, float], None] = lambda stage, seconds: None,
) -> CaveProgress:
    """Run all stages in order, reporting timings."""
    progress = CaveProgress()
    for stage in CAVE_STAGES:
        t0 = time.perf_counter()
        progress = stage.run(progress, config, seed)
        report(stage.name, time.perf_counter() - t0)
    return progress
