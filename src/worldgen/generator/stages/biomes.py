"""Stage 5: biomes from the rules in world.yaml, one per cell, with blended ground colour."""

from dataclasses import dataclass

import numpy as np

from worldgen.generator.config import BiomeRule, BiomesSection
from worldgen.generator.grid import min_filter, ramp, to_cells

WATER_KINDS = ("none", "sea", "river", "lake")  # also the order of precedence when a cell's corners differ
RANGES = ("elevation", "temperature", "moisture", "slope", "sediment", "relief")
# Values cannot go past these, so a range edge on them is not a border with another biome
DOMAIN = {"moisture": (0.0, 1.0), "slope": (0.0, 90.0), "sediment": (0.0, np.inf), "relief": (0.0, np.inf)}


def local_relief(elevation: np.ndarray, cell_size: float, radius: float) -> np.ndarray:
    """Each cell's height above the lowest ground within `radius` metres (a square), from the vertex grid."""
    cells = to_cells(elevation)
    return cells - min_filter(cells, round(radius / cell_size))


@dataclass(frozen=True, slots=True)
class CellInputs:
    """Per-cell values the rules test."""

    elevation: np.ndarray  # metres above sea level
    temperature: np.ndarray
    moisture: np.ndarray
    slope: np.ndarray
    water: np.ndarray  # index into WATER_KINDS
    sediment: np.ndarray  # metres erosion deposited
    relief: np.ndarray  # metres above the lowest ground within biomes.relief_radius


def water_kind_cells(sea: np.ndarray, lake: np.ndarray, river: np.ndarray) -> np.ndarray:
    """Each cell's water kind: the kind most of its four corner vertices have, ties going to water."""
    vertex_kind = np.zeros(sea.shape, np.int8)
    for code, mask in ((1, sea), (2, river), (3, lake)):
        vertex_kind[mask] = code
    corners = (vertex_kind[:-1, :-1], vertex_kind[:-1, 1:], vertex_kind[1:, :-1], vertex_kind[1:, 1:])
    counts = np.stack([sum((c == code).astype(np.float64) for c in corners) for code in range(len(WATER_KINDS))])
    counts += np.arange(len(WATER_KINDS))[:, None, None] * 0.1  # tie-break towards the later kinds
    return np.argmax(counts, axis=0).astype(np.int8)


def classify(
    inputs: CellInputs, cfg: BiomesSection, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Biome index per cell (first matching rule) and RGB ground colour per cell.

    Colour uses soft matches: each range edge is widened into a ramp `blend_width` of the range
    wide, and rules are cascaded in order (a rule gets what earlier rules left over). The weights
    always sum to one, so colour changes smoothly across every border, and well inside a biome it
    is exactly that biome's colour.
    """
    biome = np.full(inputs.water.shape, -1, np.int16)
    remaining = np.ones(inputs.water.shape)
    color = np.zeros(inputs.water.shape + (3,))
    jitter = np.zeros(inputs.water.shape)
    for index, rule in enumerate(cfg.rules):
        hard, soft = _match(rule, inputs, cfg.blend_width)
        biome[(biome < 0) & hard] = index
        weight = remaining * soft
        color += weight[..., None] * np.array(rule.ground.color, np.float64)
        jitter += weight * rule.ground.jitter
        remaining *= 1 - soft
    variation = 1 + jitter * rng.uniform(-1, 1, jitter.shape)
    return biome.astype(np.uint8), np.clip(np.rint(color * variation[..., None]), 0, 255).astype(np.uint8)


def soft_weights(inputs: CellInputs, cfg: BiomesSection) -> np.ndarray:
    """Each rule's share of each cell, (rules, z, x), as `classify` blends colour with them: they sum to one, so
    anything a biome gives blends across its borders as its colour does."""
    remaining = np.ones(inputs.water.shape)
    weights = np.zeros((len(cfg.rules),) + inputs.water.shape, np.float32)
    for index, rule in enumerate(cfg.rules):
        _, soft = _match(rule, inputs, cfg.blend_width)
        weights[index] = remaining * soft
        remaining *= 1 - soft
    return weights


def _match(rule: BiomeRule, inputs: CellInputs, blend_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Where the rule matches (inclusive ranges), and its soft 0-1 version for colour blending."""
    shape = inputs.water.shape
    hard, soft = np.ones(shape, bool), np.ones(shape)
    when = rule.when
    if when is None:
        return hard, soft
    for key in RANGES:
        bounds = getattr(when, key)
        if bounds is None:
            continue
        lo, hi = bounds
        values = getattr(inputs, key)
        hard &= (values >= lo) & (values <= hi)
        soft *= ramp(values, lo, hi, blend_width * (hi - lo), DOMAIN.get(key))
    if when.water is not None:
        is_kind = inputs.water == WATER_KINDS.index(when.water)
        hard &= is_kind
        soft *= is_kind
    return hard, soft
