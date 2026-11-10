"""Pictures of a terrain part-way through generation, one per layer, and what lies under a point.

Every layer is an RGB image with one pixel per cell, north up, lit from the north-west like
`preview.png`. From biomes on, the layers are `preview.render` itself.
"""

import math

import numpy as np

from worldgen.proc_gen.world.config import TerrainConfig
from worldgen.proc_gen.world.grid import to_cells
from worldgen.proc_gen.world.pipeline import Progress, ground
from worldgen.proc_gen.world.preview import DEEP, SHALLOW, hillshade, render, water_over
from worldgen.utils.gen_ui.model.readout import Fact, Readout

# Layer -> the stage whose result it shows
LAYERS = {
    "elevation": "elevation",
    "land mask": "elevation",
    "mountain mask": "elevation",
    "terrace mask": "elevation",
    "erosion": "erosion",
    "erosion change": "erosion",
    "sediment": "erosion",
    "hydrology": "hydrology",
    "temperature": "climate",
    "moisture": "climate",
    "biomes": "biomes",
    "relief": "biomes",
}
MAX_SIDE = 2048  # pixels a layer image may have per side; larger worlds are shown at every 2nd (4th...) cell

# Colour ramps: (value, RGB) stops, interpolated linearly
LAND = ((0.0, (88, 140, 72)), (0.35, (150, 162, 92)), (0.7, (140, 110, 82)), (1.0, (242, 242, 242)))  # x max height
TEMPERATURE = ((-10, (40, 60, 170)), (0, (120, 170, 230)), (10, (238, 238, 218)), (20, (240, 170, 80)), (35, (190, 40, 30)))
MOISTURE = ((0.0, (205, 172, 112)), (0.5, (122, 172, 82)), (1.0, (40, 110, 172)))
MASK = ((0.0, (238, 236, 222)), (0.5, (196, 150, 104)), (1.0, (120, 70, 50)))  # the land, mountain and terrace masks, 0-1
SEDIMENT = ((0.0, (238, 236, 222)), (0.3, (222, 196, 130)), (1.0, (150, 96, 40)))  # x the deepest (99th percentile)
RELIEF = ((0.0, (60, 110, 160)), (0.3, (238, 236, 222)), (1.0, (150, 60, 40)))  # x the highest (99th percentile)
CUT, DEPOSIT = np.array((196, 70, 50)), np.array((50, 110, 200))  # erosion change: ground taken, ground laid down
PAPER = np.array((238, 236, 222))


def layers_for_stage(stage: str) -> list[str]:
    return [layer for layer, of in LAYERS.items() if of == stage]


def has_layer(layer: str, progress: Progress) -> bool:
    """Whether `progress` has what `layer` shows: the mountain and terrace masks only when the recipe made them,
    an erosion layer only when the ground was eroded. (The land mask always shows; a continent without falloff
    is flat land.)"""
    if layer == "mountain mask":
        return progress.mountains is not None
    if layer == "terrace mask":
        return progress.terraces is not None
    if LAYERS.get(layer) == "erosion":
        return progress.erosion is not None
    return True


def render_layer(layer: str, progress: Progress, config: TerrainConfig, max_side: int = MAX_SIDE) -> np.ndarray:
    """The picture of `layer` (uint8 RGB, rows north to south), at most `max_side` pixels a side."""
    image = layer_image(layer, progress, config)
    step = math.ceil(image.shape[0] / max_side)
    return np.ascontiguousarray(image[::step, ::step])


def layer_image(layer: str, progress: Progress, config: TerrainConfig) -> np.ndarray:
    """The picture of `layer` at one pixel per cell."""
    world, cell_size = config.world, config.world.cell_size
    if layer in ("elevation", "erosion"):
        ground_grid = progress.heights if layer == "elevation" or progress.erosion is None else progress.erosion.heights
        image = elevation_image(ground_grid, config)
    elif layer in ("land mask", "mountain mask", "terrace mask"):
        if layer == "land mask":
            mask, flat = progress.mask, 1.0  # a continent without falloff is land everywhere
        else:
            mask, flat = (progress.mountains if layer == "mountain mask" else progress.terraces), 0.0
        values = to_cells(mask) if mask is not None else np.full((world.size_cells,) * 2, flat)
        image = ramp(values, MASK) * hillshade(progress.heights, cell_size)[..., None]
    elif layer == "erosion change":
        image = change_image(progress) * hillshade(progress.erosion.heights, cell_size)[..., None]
    elif layer == "sediment":
        sediment = to_cells(progress.erosion.sediment.astype(np.float64))
        image = ramp(sediment / _top(sediment), SEDIMENT) * hillshade(progress.erosion.heights, cell_size)[..., None]
    elif layer == "relief":
        relief = progress.world.relief.astype(np.float64)
        image = ramp(relief / _top(relief), RELIEF) * hillshade(progress.world.elevation, cell_size)[..., None]
    elif layer == "hydrology":
        water = progress.hydrology
        lightness = 110 + 90 * np.clip(to_cells(water.elevation - world.sea_level) / config.elevation.max_height, 0, 1)
        image = np.repeat(lightness[..., None], 3, axis=-1) * hillshade(water.elevation, cell_size)[..., None]
        water_over(image, water.elevation, water.water, water.river)
    elif layer in ("temperature", "moisture"):
        weather, ground_grid = progress.climate, progress.hydrology.elevation
        values = to_cells(weather.temperature if layer == "temperature" else weather.moisture)
        image = ramp(values, TEMPERATURE if layer == "temperature" else MOISTURE) * hillshade(ground_grid, cell_size)[..., None]
    elif layer in LAYERS:
        return render(progress.world)  # biomes: the terrain as preview.png shows it
    else:
        raise ValueError(f"unknown layer {layer!r}; there are {', '.join(LAYERS)}")
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def elevation_image(heights: np.ndarray, config: TerrainConfig) -> np.ndarray:
    """Land ramped by height up to `max_height`, the sea by depth, hillshaded."""
    world = config.world
    cells = to_cells(heights - world.sea_level)
    land = ramp(cells / config.elevation.max_height, LAND)
    depth = np.clip(cells / min(config.elevation.sea_floor - world.sea_level, -1e-6), 0, 1)[..., None]
    sea = SHALLOW + (DEEP - SHALLOW) * depth
    return np.where((cells < 0)[..., None], sea, land) * hillshade(heights, world.cell_size)[..., None]


def change_image(progress: Progress) -> np.ndarray:
    """What erosion did, unshaded: ground cut away red, ground laid down blue, deeper where the change is deeper (up
    to the 99th percentile of it), paper where nothing changed."""
    change = to_cells(progress.erosion.heights - progress.heights)
    share = np.clip(np.abs(change) / _top(np.abs(change)), 0, 1)[..., None]
    colour = np.where((change < 0)[..., None], CUT, DEPOSIT)
    return PAPER + (colour - PAPER) * share


def _top(values: np.ndarray) -> float:
    """The 99th percentile of `values`, so one deep pit does not wash out the ramp; never zero."""
    return max(float(np.percentile(values, 99)), 1e-6)


def ramp(values: np.ndarray, stops: tuple) -> np.ndarray:
    xs = [x for x, _ in stops]
    return np.stack([np.interp(values, xs, [rgb[c] for _, rgb in stops]) for c in range(3)], axis=-1)


def ground_and_water(progress: Progress) -> tuple[np.ndarray, np.ndarray | None]:
    """The ground and the water surface (NaN where dry) as far as `progress` has them: vertex grids, metres.
    Before hydrology there is no water."""
    if progress.world is not None:
        return progress.world.elevation, progress.world.water
    if progress.hydrology is not None:
        return progress.hydrology.elevation, progress.hydrology.water
    return ground(progress), None


def readout(progress: Progress, config: TerrainConfig, x: float, z: float) -> Readout | None:
    """What lies at world (x, z), in metres, as far as `progress` knows: where it is, then its height, water,
    biome and climate. Keyed by the cell, so a tooltip waits while the mouse stays in one. None off the world."""
    world, cell_size = config.world, config.world.cell_size
    n = world.size_cells
    if not (0 <= x < n * cell_size and 0 <= z < n * cell_size):
        return None
    vx, vz = round(x / cell_size), round(z / cell_size)  # nearest vertex
    cx, cz = min(int(x / cell_size), n - 1), min(int(z / cell_size), n - 1)  # the cell
    generated = progress.world
    surface_ground, surface = ground_and_water(progress)
    facts: list[Fact] = [("ground", f"{surface_ground[vz, vx] - world.sea_level:.1f} m")]
    if surface is not None and not math.isnan(surface[vz, vx]):
        facts.append(("water", f"{surface[vz, vx] - surface_ground[vz, vx]:.1f} m deep"))
    facts += terrain_facts(progress, config, vx, vz)
    if generated is not None:
        facts.append(("biome", config.biomes.rules[int(generated.biome[cz, cx])].id))
    if progress.climate is not None:
        facts += [("temperature", f"{progress.climate.temperature[vz, vx]:.1f} °C"),
                  ("moisture", f"{progress.climate.moisture[vz, vx]:.2f}"),
                  ("slope", f"{progress.climate.slope[vz, vx]:.0f}°")]
    if generated is not None:
        facts.append(("relief", f"{generated.relief[cz, cx]:.1f} m"))
    return Readout(("cell", cx, cz), f"x {x:.0f} m, z {z:.0f} m", facts)


def terrain_facts(progress: Progress, config: TerrainConfig, vx: int, vz: int) -> list[Fact]:
    """How the ground at vertex (vx, vz) was shaped: the land, mountain and terrace masks' shares there, the
    stratum it lies on, and what erosion took or laid down."""
    facts: list[Fact] = []
    if progress.mask is not None:
        facts.append(("land mask", f"{progress.mask[vz, vx]:.0%}"))
    if progress.mountains is not None:
        facts.append(("mountains", f"{progress.mountains[vz, vx]:.0%}"))
    if progress.terraces is not None:
        step = config.elevation.terraces.step
        stratum = math.floor((progress.heights[vz, vx] - config.world.sea_level) / step)
        facts.append(("terraces", f"{progress.terraces[vz, vx]:.0%}, stratum {stratum} ({stratum * step:.0f} m)"))
    if progress.erosion is not None:
        change = progress.erosion.heights[vz, vx] - progress.heights[vz, vx]
        facts += [("erosion", f"{change:+.1f} m"), ("sediment", f"{progress.erosion.sediment[vz, vx]:.1f} m")]
    return facts
