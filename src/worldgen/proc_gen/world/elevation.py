"""Stage 1: the base heightfield, with mountain ranges and terraces where the recipe asks for them."""

from dataclasses import dataclass

import numpy as np

from worldgen.proc_gen.world.config import (
    ArchipelagoSection,
    ElevationSection,
    MountainsSection,
    TerracesSection,
    WorldSection,
)
from worldgen.proc_gen.world.grid import smoothstep
from worldgen.proc_gen.world.noise import Perlin, fbm, ridged

LAND_BASE = 0.15  # inside the island mask, the lowest land sits this far up the 0-1 scale
MASK_CYCLES = 3.0  # cycles of outline-distorting noise across the world
WARP_OCTAVES = 3
PLACEMENT_TRIES = 200
ISLAND_OFFSET = 1000.0  # each island's outline samples a different region of the noise
MASK_OCTAVES = 3  # of the range and terrace masks
MASK_BAND = 0.25  # a mask rises from 0 to 1 over this share of the world either side of its coverage
QUANTILE_STRIDE = 4  # masks find their coverage's threshold on every 4th sample each way: the same to a hair, faster
RANGE_FOOT = 0.1  # ranges rise from nothing at the shore to full height this share of the land's height above it
TERRACE_JITTER = 0.5  # strata tilt up and down by this share of a step across the world
TERRACE_JITTER_CYCLES = 4.0  # of the tilt noise, across the world
TERRACE_FOOT = 0.25  # terraces fade in over this share of a step above sea level


@dataclass(frozen=True, slots=True)
class Elevation:
    """The heightfield and the masks that shaped it, kept for the world UI's layers; vertex grids."""

    heights: np.ndarray  # metres
    mask: np.ndarray | None = None  # 0-1, land membership; None for a continent without falloff (land everywhere)
    mountains: np.ndarray | None = None  # 0-1, the range mask; None without mountains
    terraces: np.ndarray | None = None  # 0-1, the terrace mask; None without terraces


def generate(
    world: WorldSection,
    cfg: ElevationSection,
    rng: np.random.Generator,
    mountains_rng: np.random.Generator | None = None,
    terraces_rng: np.random.Generator | None = None,
) -> Elevation:
    """Heights in metres on the vertex grid: real depths below `sea_level`, down to `sea_floor`.

    - The hills are drawn from `rng`. Mountains and terraces draw from their own generators (needed only when the
      recipe turns them on), so a recipe without them gives the same heights as before they existed.
    - The ranges are mixed in before the metre mapping, so `shore`, `max_height` and `sea_floor` keep their meaning;
      the terraces step the heights in metres after it.
    """
    n = world.size_cells + 1
    z, x = np.mgrid[0:n, 0:n].astype(np.float64)
    hills_noise, warp_x_noise, warp_z_noise, mask_noise = (Perlin(rng) for _ in range(4))

    # Domain warp: sample the hills at pushed-around points, which breaks up the grid-aligned look
    warp = {"frequency": cfg.warp.frequency, "octaves": WARP_OCTAVES}
    wx = fbm(warp_x_noise, x, z, **warp) * cfg.warp.strength
    wz = fbm(warp_z_noise, x, z, **warp) * cfg.warp.strength
    hills = fbm(hills_noise, x + wx, z + wz, **cfg.noise.model_dump())
    hills = (hills - hills.min()) / max(np.ptp(hills), 1e-12)

    mask = None
    if cfg.mask == "continent":
        shaped = hills
        if cfg.falloff > 0:
            mask = _edge_falloff(cfg, world, mask_noise, x, z)
            shaped = hills * mask
    else:
        # One island is the archipelago's special case: a single island of radius 0.5 in the centre
        islands = (
            [(0.5, 0.5, 0.5)] if cfg.mask == "island" else place_islands(cfg.archipelago, cfg.mask_falloff[1], rng)
        )
        u, v = x / world.size_cells, z / world.size_cells
        largest = max(radius for _, _, radius in islands)
        sea = np.ones_like(u)  # chance a cell is in no island: islands join smoothly, without a crease
        weight, stature = np.zeros_like(u), np.zeros_like(u)
        for i, (cx, cz, radius) in enumerate(islands):
            # Outline noise sized to the island, so small islands are as ragged as big ones
            frequency = MASK_CYCLES / (2 * radius * world.size_cells)
            bend = 1 + cfg.mask_distortion * fbm(mask_noise, x + ISLAND_OFFSET * i, z, frequency=frequency, octaves=2)
            d = np.hypot(u - cx, v - cz) / radius * bend  # 1 at the island's radius
            island = 1 - smoothstep(*cfg.mask_falloff, d)
            sea *= 1 - island
            weight += island
            stature += island * np.sqrt(radius / largest)  # smaller islands are lower, not steeper
        land = 1 - sea
        mask = land
        height = np.divide(stature, weight, out=np.ones_like(u), where=weight > 0)
        shaped = land * (LAND_BASE + (1 - LAND_BASE) * hills * height)

    ranges = None
    if cfg.mountains.enabled:
        if mountains_rng is None:
            raise ValueError("mountains are enabled but no generator was given for them")
        warped = (x + wx, z + wz) if cfg.mountains.warp else (x, z)
        shaped, ranges = _raise_ranges(shaped, x, z, warped, world, cfg.mountains, cfg.shore, mountains_rng)

    # Shaped values below `shore` are sea floor, above it land; both scales meet at sea level
    sea = world.sea_level
    heights = np.where(
        shaped < cfg.shore,
        cfg.sea_floor + (sea - cfg.sea_floor) * shaped / cfg.shore,
        sea + cfg.max_height * (shaped - cfg.shore) / (1 - cfg.shore),
    )
    steps = None
    if cfg.terraces.enabled:
        if terraces_rng is None:
            raise ValueError("terraces are enabled but no generator was given for them")
        heights, steps = _terrace(heights, x, z, ranges, world, cfg.terraces, terraces_rng)
    return Elevation(heights, mask, ranges, steps)


def _edge_falloff(
    cfg: ElevationSection,
    world: WorldSection,
    mask_noise: Perlin,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """A continent's coast: 0 at the world borders rising to 1 over `falloff` of the width, bent by the same
    outline noise the island masks use, so the land sinks into the sea along all four edges."""
    u, v = x / world.size_cells, z / world.size_cells
    bend = 1 + cfg.mask_distortion * fbm(mask_noise, x, z, frequency=MASK_CYCLES / world.size_cells, octaves=2)
    edge = np.minimum.reduce([u, 1 - u, v, 1 - v])  # share of the world width from the nearest border
    return smoothstep(0.0, cfg.falloff, edge * bend)


def coverage_mask(noise: Perlin, x: np.ndarray, z: np.ndarray, frequency: float, coverage: float) -> np.ndarray:
    """A 0-1 mask of slow blobs covering about `coverage` of the grid: low-frequency fBm, 1 above the threshold that
    leaves that share above it, rising to it smoothly over a band either side."""
    if coverage <= 0:
        return np.zeros_like(x)
    if coverage >= 1:
        return np.ones_like(x)
    values = fbm(noise, x, z, frequency=frequency, octaves=MASK_OCTAVES)
    sample = values[::QUANTILE_STRIDE, ::QUANTILE_STRIDE]
    threshold = np.quantile(sample, 1 - coverage)  # the mask is 0.5 here, so about `coverage` of it is over a half
    lo = np.quantile(sample, max(1 - coverage - MASK_BAND, 0.0))
    hi = np.quantile(sample, min(1 - coverage + MASK_BAND, 1.0))
    half = max((hi - lo) / 2, 1e-12)
    return smoothstep(threshold - half, threshold + half, values)


def _raise_ranges(
    shaped: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    warped: tuple[np.ndarray, np.ndarray],
    world: WorldSection,
    cfg: MountainsSection,
    shore: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """`shaped` (0-1) with ridged ranges risen where the range mask lies, and the mask.

    - The hills' land, above the shore, is pressed into the lower `1 - height` of it, and the ranges fill the
      `height` above: the top is still 1, and the coast stays where it was.
    - Ranges rise from nothing at the shore, so they never stand up out of the sea floor.
    """
    ridge_noise, mask_noise = Perlin(rng), Perlin(rng)
    cell = world.cell_size
    ridges = ridged(ridge_noise, *warped, frequency=cell / cfg.wavelength, octaves=cfg.octaves, gain=cfg.gain,
                    sharpness=cfg.sharpness)
    mask = coverage_mask(mask_noise, x, z, cell / cfg.range_wavelength, cfg.coverage)
    above = np.maximum(shaped - shore, 0) / (1 - shore)  # 0 at the shore, 1 at the top
    foot = smoothstep(0.0, RANGE_FOOT, above)
    mask = mask * foot
    raised = shaped - (1 - shore) * cfg.height * above + (1 - shore) * cfg.height * mask * ridges
    return raised, mask


def terrace(heights: np.ndarray, step: float, sharpness: float, offset: np.ndarray | float = 0.0) -> np.ndarray:
    """`heights` gathered into strata `step` metres apart, each tread level and each riser steep: the riser takes
    up `1 - 0.9 * sharpness` of the climb, smoothstepped, and the rest of it is tread. `offset` (metres) tilts the
    strata."""
    riser = 1 - 0.9 * sharpness
    level = (heights - offset) / step
    floor = np.floor(level)
    climb = smoothstep(1 - riser, 1.0, level - floor)
    return (floor + climb) * step + offset


def _terrace(
    heights: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    ranges: np.ndarray | None,
    world: WorldSection,
    cfg: TerracesSection,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """The heights stepped inside the terrace mask, and the mask (0 below sea level)."""
    tilt_noise, mask_noise = Perlin(rng), Perlin(rng)
    if cfg.where == "all":
        mask = np.ones_like(heights)
    elif cfg.where == "mountains":
        mask = ranges
    else:
        mask = coverage_mask(mask_noise, x, z, world.cell_size / cfg.wavelength, cfg.coverage)
    mask = mask * smoothstep(world.sea_level, world.sea_level + TERRACE_FOOT * cfg.step, heights)
    tilt = TERRACE_JITTER * cfg.step * fbm(tilt_noise, x, z, frequency=TERRACE_JITTER_CYCLES / world.size_cells, octaves=2)
    stepped = terrace(heights, cfg.step, cfg.sharpness, tilt)
    return heights + mask * (stepped - heights), mask


def place_islands(cfg: ArchipelagoSection, reach: float, rng: np.random.Generator) -> list[tuple[float, float, float]]:
    """(centre x, centre z, radius) of each island, as fractions of the world width.

    The largest island sits in the centre. The rest go at random spots, largest first, each kept inside
    the world and away from the others; an island that finds no room after a number of tries is dropped.
    `reach` is where land ends, in radii.
    """
    small, large = cfg.size
    radii = sorted(rng.uniform(small, large, cfg.islands), reverse=True)
    radii[0] = large
    islands = [(0.5, 0.5, radii[0])]
    for radius in radii[1:]:
        margin = min(radius * reach, 0.5)
        for _ in range(PLACEMENT_TRIES):
            cx, cz = rng.uniform(margin, 1 - margin, 2)
            if all(np.hypot(cx - ox, cz - oz) >= cfg.spacing * (radius + other) * reach for ox, oz, other in islands):
                islands.append((float(cx), float(cz), float(radius)))
                break
    return islands
