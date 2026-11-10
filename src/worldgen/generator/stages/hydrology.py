"""Stage 3: lakes and rivers.

Fill depressions (priority-flood), route water downhill (D8), accumulate rainfall, then carve
rivers where enough water gathers. Water flows from cell to cell in eight directions, which makes
staircase rivers; channels are carved around smoothed centre lines instead, by distance, so banks
curve and the water surface falls evenly along the flow. The hot loops run over plain Python lists, which is far faster
than indexing numpy arrays one element at a time.
"""

import heapq
import math
from dataclasses import dataclass

import numpy as np

from worldgen.generator.config import HydrologySection, WorldSection
from worldgen.generator.grid import disc_offsets, shift
from worldgen.generator.noise import Perlin, fbm

# D8 flow codes: (dz, dx) of the neighbour water flows to. z grows southwards, x eastwards.
D8 = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))  # E SE S SW W NW N NE
NO_FLOW = 255

MIN_HALF_WIDTH = 0.75  # cells either side of the centre line at least, so a river never breaks into beads
SHORE_SLOPE = 0.5  # metres the ground rises (falls) per cell outside (inside) a channel's edge, at least
BANK_SLOPE = 1.0  # metres it rises (falls) per cell at most, within a cell of the edge
SMOOTHING = 12  # moving-average passes over a centre line (about a Gaussian of 3 cells) before its corners are rounded
MIN_STRENGTH = 0.05  # a river at the threshold still gets a shallow channel
WATER_FRACTION = 0.8  # share of the channel depth that holds water; the rest is bank above it
MOUTH_REACH = 8  # cells a channel runs on into the lake or sea it ends in, shoaling to nothing
RAIN_OCTAVES = 4
MEANDER_OCTAVES = 3
SHADOW_STEPS, SHADOW_STEP = 24, 4  # look this many steps of this many cells upwind for ridges


@dataclass(frozen=True, slots=True)
class Lake:
    id: int
    level: float  # flat water surface, the depression's spill height
    bbox: tuple[int, int, int, int]  # x0, z0, x1, z1 in vertex coordinates, inclusive
    outlet: tuple[int, int] | None  # (x, z) of the cell the lake drains into
    area: int  # vertices under the lake


@dataclass(frozen=True, slots=True)
class Hydrology:
    elevation: np.ndarray  # carved ground, metres
    water: np.ndarray  # water surface, NaN where dry
    sea: np.ndarray  # bool: open sea, connected to the world edge
    lake_id: np.ndarray  # uint16, 0 = none
    river: np.ndarray  # uint8 strength, 0 = none
    river_path: np.ndarray  # bool: river centre lines (the cells flow was traced along)
    flow_dir: np.ndarray  # uint8 D8 code, NO_FLOW at sinks (sea, world edge)
    rain: np.ndarray  # rainfall per vertex, for the climate stage
    lakes: list[Lake]
    rivers: list[list[tuple[int, int]]]  # polylines of (x, z) vertex coordinates, source first


def generate(
    elevation: np.ndarray, world: WorldSection, cfg: HydrologySection, rng: np.random.Generator
) -> Hydrology:
    elevation = elevation.copy()
    rows, cols = elevation.shape
    sea_level = world.sea_level
    rain = _rainfall(elevation, sea_level, cfg, rng)

    edge = np.zeros(elevation.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    plain, raised = priority_flood(elevation, edge, cfg.epsilon)

    # Sea: below sea level and connected to the world edge by cells that are too. The flood from the
    # edge only stays at or under sea level along such a path.
    sea = (elevation < sea_level) & (plain <= sea_level)
    lake_id, lakes_cells = _lakes(elevation, plain, sea, cfg)  # also fills small depressions in `elevation`

    z, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    # Noise scaled so its slopes are about 1 per cell, whatever the frequency
    roughness = fbm(Perlin(rng), x, z, frequency=cfg.meander.frequency, octaves=MEANDER_OCTAVES)
    roughness /= 2 * np.pi * cfg.meander.frequency
    flow_dir = flow_directions(raised, edge | sea, roughness, cfg.meander.strength)
    down = downstream(flow_dir)
    order = np.argsort(-raised, axis=None, kind="stable")  # every cell drains to a strictly lower one
    accumulation = accumulate(order, down, rain)

    lakes = [
        _describe_lake(i + 1, float(plain.flat[cells[0]]), cells, lake_id, down, cols)
        for i, cells in enumerate(lakes_cells)
    ]
    levels = np.full(elevation.shape, np.nan)
    for lake, cells in zip(lakes, lakes_cells):
        levels.flat[cells] = lake.level

    path = (accumulation > cfg.river_threshold) & ~sea & (lake_id == 0)
    strength = np.zeros(elevation.shape)
    if path.any():
        span = max(np.log(accumulation[path].max()) - np.log(cfg.river_threshold), 1e-9)
        strength[path] = np.clip((np.log(accumulation[path]) - np.log(cfg.river_threshold)) / span, 0, 1)

    rivers = _trace_rivers(path, order, down, cols)
    elevation, water, river = _carve(elevation, path, strength, order, down, sea, levels, rivers, world, cfg)
    water[sea] = sea_level
    water[lake_id > 0] = levels[lake_id > 0]

    return Hydrology(
        elevation=elevation,
        water=water,
        sea=sea,
        lake_id=lake_id,
        river=river,
        river_path=path,
        flow_dir=flow_dir,
        rain=rain,
        lakes=lakes,
        rivers=rivers,
    )


def priority_flood(elevation: np.ndarray, seeds: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill every depression up to its spill height, flooding inwards from `seeds` (Barnes et al. 2014).

    Returns the plain filled surface (flat over depressions, so lake levels can be read off it) and
    the ε-raised one, where each cell is at least `epsilon` above the one it was flooded from, so
    water can drain across flats.
    """
    rows, cols = elevation.shape
    # Work on a grid with a one-cell border marked as already flooded, so neighbours need no bounds checks
    pc = cols + 2
    heights = np.pad(elevation, 1).ravel().tolist()
    closed = bytearray(np.pad(np.zeros(elevation.shape, np.uint8), 1, constant_values=1).ravel().tobytes())
    plain, raised = heights[:], heights[:]
    offsets = [dz * pc + dx for dz, dx in D8]

    seed_idx = np.flatnonzero(np.pad(seeds, 1))
    heap = [(heights[i], i) for i in seed_idx.tolist()]
    heapq.heapify(heap)
    for i in seed_idx.tolist():
        closed[i] = 1

    pop, push = heapq.heappop, heapq.heappush
    while heap:
        level, c = pop(heap)
        spill = plain[c]
        step = level + epsilon
        for off in offsets:
            nb = c + off
            if closed[nb]:
                continue
            closed[nb] = 1
            h = heights[nb]
            plain[nb] = max(spill, h)  # not max(): this loop is hot, and the call costs
            r = max(step, h)
            raised[nb] = r
            push(heap, (r, nb))

    def unpad(values: list) -> np.ndarray:
        return np.array(values).reshape(rows + 2, pc)[1:-1, 1:-1]

    return unpad(plain), unpad(raised)


def flow_directions(
    surface: np.ndarray, sinks: np.ndarray, roughness: np.ndarray | None = None, meander: float = 0.0
) -> np.ndarray:
    """D8: each cell flows to a downhill neighbour; `sinks` and cells with none get NO_FLOW.

    Without `roughness`, that is the steepest neighbour. With it (a noise field whose slopes are
    about 1 per cell), each neighbour's drop also gains `meander` times the cell's steepest drop
    times the noise's drop towards it: water follows small imaginary valleys, so rivers wiggle and
    gather instead of running in the straight parallel lines plain D8 draws on even slopes. Scaling
    by the steepest drop makes the effect the same on gentle and steep ground. Only strictly lower
    neighbours are candidates, so flow still always runs downhill and never loops.
    """
    rows, cols = surface.shape

    def drops_of(field: np.ndarray, outside: float) -> list[np.ndarray]:
        padded = np.pad(field, 1, constant_values=outside)
        return [
            (field - padded[1 + dz:1 + dz + rows, 1 + dx:1 + dx + cols]) / np.hypot(dz, dx) for dz, dx in D8
        ]

    drops = drops_of(surface, np.inf)
    scores = drops
    if roughness is not None and meander > 0:
        steepest = np.max(drops, axis=0)
        scores = [d + meander * steepest * r for d, r in zip(drops, drops_of(roughness, 0.0))]
    best = np.full(surface.shape, -np.inf)
    code = np.full(surface.shape, NO_FLOW, np.uint8)
    for k, (drop, score) in enumerate(zip(drops, scores)):
        better = (drop > 0) & (score > best)
        best[better] = score[better]
        code[better] = k
    code[sinks] = NO_FLOW
    return code


def downstream(flow_dir: np.ndarray) -> np.ndarray:
    """Flat index of each cell's downstream neighbour, or -1 at sinks."""
    cols = flow_dir.shape[1]
    step = np.array([dz * cols + dx for dz, dx in D8] + [0])
    codes = np.where(flow_dir == NO_FLOW, len(D8), flow_dir).ravel()
    return np.where(codes == len(D8), -1, np.arange(codes.size) + step[codes])


def accumulate(order: np.ndarray, down: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Total rainfall flowing through each cell: its own plus everything upstream."""
    acc = rain.ravel().tolist()
    down_l = down.tolist()
    for c in order.tolist():
        d = down_l[c]
        if d >= 0:
            acc[d] += acc[c]
    return np.array(acc).reshape(rain.shape)


def _rainfall(elevation: np.ndarray, sea_level: float, cfg: HydrologySection, rng: np.random.Generator) -> np.ndarray:
    """Coarse precipitation: noise around the base, less behind ridges in the wind's lee."""
    rows, cols = elevation.shape
    z, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    noise = fbm(Perlin(rng), x, z, frequency=cfg.rain.frequency, octaves=RAIN_OCTAVES)
    rain = cfg.rain.base * (1 + cfg.rain.noise * noise)
    if cfg.rain_shadow.enabled:
        wx, wz = cfg.rain_shadow.wind
        length = np.hypot(wx, wz)
        wx, wz = wx / length, wz / length
        ridge = np.full(elevation.shape, sea_level)
        for k in range(1, SHADOW_STEPS + 1):
            dist = k * SHADOW_STEP
            ridge = np.maximum(ridge, shift(elevation, round(-wz * dist), round(-wx * dist), sea_level))
        relief = max(float(elevation.max()) - sea_level, 1.0)
        shadow = np.clip((ridge - elevation) / relief, 0, 1)
        rain *= 1 - cfg.rain_shadow.strength * shadow
    return np.maximum(rain, 0.0)


def _lakes(
    elevation: np.ndarray, plain: np.ndarray, sea: np.ndarray, cfg: HydrologySection
) -> tuple[np.ndarray, list[list[int]]]:
    """Label depressions deep and wide enough as lakes; fill the rest into `elevation` (in place).

    A depression is a 4-connected group of cells raised by the fill to one shared spill height.
    Returns the lake id grid and each lake's flat cell indices (first index first).
    """
    rows, cols = elevation.shape
    depression = ((plain > elevation) & ~sea).ravel()
    spill = plain.ravel().tolist()
    seen = bytearray(depression.size)
    flat_elevation = elevation.reshape(-1)  # a view: filling writes through to `elevation`
    lake_id = np.zeros(depression.size, np.uint16)
    lakes = []
    for start in np.flatnonzero(depression).tolist():
        if seen[start]:
            continue
        seen[start] = 1
        level = spill[start]
        cells, stack = [], [start]
        while stack:
            c = stack.pop()
            cells.append(c)
            z, x = divmod(c, cols)
            for nb, ok in ((c - cols, z > 0), (c + cols, z < rows - 1), (c - 1, x > 0), (c + 1, x < cols - 1)):
                if ok and not seen[nb] and depression[nb] and spill[nb] == level:
                    seen[nb] = 1
                    stack.append(nb)
        cells.sort()
        deepest = level - float(flat_elevation[cells].min())
        if deepest > cfg.lake_min_depth and len(cells) >= cfg.lake_min_area:
            if len(lakes) == np.iinfo(np.uint16).max:
                raise ValueError("more than 65535 lakes; raise hydrology.lake_min_area or lake_min_depth")
            lakes.append(cells)
            lake_id[cells] = len(lakes)
        else:
            flat_elevation[cells] = level
    return lake_id.reshape(elevation.shape), lakes


def _describe_lake(id: int, level: float, cells: list[int], lake_id: np.ndarray, down: np.ndarray, cols: int) -> Lake:
    zs, xs = np.divmod(np.array(cells), cols)
    outlet = None
    flat_ids = lake_id.ravel()
    for c in cells:  # in index order, so the choice is deterministic
        d = int(down[c])
        if d >= 0 and flat_ids[d] != id:
            outlet = (d % cols, d // cols)
            break
    return Lake(id, level, (int(xs.min()), int(zs.min()), int(xs.max()), int(zs.max())), outlet, len(cells))


def _carve(
    elevation: np.ndarray,
    path: np.ndarray,
    strength: np.ndarray,
    order: np.ndarray,
    down: np.ndarray,
    sea: np.ndarray,
    levels: np.ndarray,
    rivers: list[list[tuple[int, int]]],
    world: WorldSection,
    cfg: HydrologySection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut river channels and banks; return the carved ground, river water surface and strength (uint8)."""
    sea_level = world.sea_level
    depth = np.where(path, cfg.carve_depth * np.sqrt(np.maximum(strength, MIN_STRENGTH)), 0.0)
    cell_size = world.cell_size
    widest = max((cfg.river_min_width + cfg.river_width_scale) / 2 / cell_size, MIN_HALF_WIDTH)
    half_width = np.where(path, np.maximum((cfg.river_min_width + cfg.river_width_scale * strength) / 2 / cell_size,
                                           MIN_HALF_WIDTH), 0.0)

    # Water along the centre lines: never rising downstream, and never below the lake or sea it ends in
    surface = np.where(path, elevation - (1 - WATER_FRACTION) * depth, np.nan).ravel().tolist()
    path_l, down_l = path.ravel().tolist(), down.tolist()
    terminal = np.where(sea, sea_level, levels).ravel().tolist()  # NaN where water does not end
    river_cells = [c for c in order.tolist() if path_l[c]]
    for c in river_cells:
        d = down_l[c]
        if d >= 0 and path_l[d] and surface[d] > surface[c]:
            surface[d] = surface[c]
    for c in reversed(river_cells):
        d = down_l[c]
        if d < 0:
            continue
        target = surface[d] if path_l[d] else terminal[d]
        if not math.isnan(target) and surface[c] < target:
            surface[c] = target
    surface = np.array(surface).reshape(elevation.shape)
    surface = np.where(path, surface, np.array(terminal).reshape(elevation.shape))  # where rivers end

    # Every cell near a river takes the nearest point of the smoothed centre lines (nearest to its bank,
    # so a wide river wins over a narrow one beside it), and the lowest water of any channel it is in
    open_water = sea | ~np.isnan(levels)
    samples = _centre_lines(rivers, surface, depth, half_width, strength, open_water)
    reach = widest + cfg.bank_falloff + 1
    near, gap, water = _nearest_samples(samples, elevation.shape, reach)
    found = near >= 0
    pick = np.maximum(near, 0)
    distance = np.where(found, gap, np.inf)
    near_water, near_depth, near_width, near_strength = (np.where(found, samples[pick, k], 0.0) for k in (2, 3, 4, 5))
    to_bank = distance - near_width  # negative inside the channel

    channel = (to_bank <= 0) & ~open_water  # a smoothed line can pass a cell or two beside the traced cells
    channel_water = np.where(np.isfinite(water), water, near_water)
    flat = channel_water.ravel()  # path cells in the channel keep the never-rising rule exactly
    wet_l = channel.ravel().tolist()
    for c in river_cells:
        d = down_l[c]
        if d >= 0 and path_l[d] and wet_l[c] and wet_l[d] and flat[d] > flat[c]:
            flat[d] = flat[c]
    channel_water = flat.reshape(elevation.shape)

    # Rounded channels: deepest on the centre line, shallow at the edge. Near the edge the ground is the
    # water level plus between SHORE_SLOPE and BANK_SLOPE times the distance to the bank, on both sides, so
    # between a wet and a dry vertex the ground crosses the water where the smooth bank is, not at a grid line
    profile = np.clip(1 - (distance / np.maximum(near_width, 1e-9)) ** 2, 0.0, 1.0)
    channel_depth = near_depth * profile
    edge = channel_water + SHORE_SLOPE * np.minimum(to_bank, -0.01)
    bed = np.minimum(np.minimum(elevation - channel_depth, channel_water - WATER_FRACTION * channel_depth), edge)
    bed = np.maximum(bed, channel_water + BANK_SLOPE * to_bank)

    # Banks slope down towards the channel, no lower than the shore slope above its water, and without
    # dropping dry land below nearby sea or lake level
    weight = np.clip(1 - to_bank / (cfg.bank_falloff + 1), 0.0, 1.0) ** 2
    lowered = np.maximum(elevation - near_depth * weight, np.minimum(elevation, near_water + SHORE_SLOPE * to_bank))
    lowered = np.where(to_bank < 1, np.minimum(lowered, near_water + BANK_SLOPE * to_bank), lowered)
    floor = np.where(np.isnan(levels), sea_level, np.maximum(levels, sea_level))
    for dz, dx, _ in disc_offsets(cfg.bank_falloff + math.ceil(widest))[1:]:
        floor = np.maximum(floor, np.nan_to_num(shift(levels, dz, dx, np.nan), nan=-np.inf))
    land = ~channel & ~open_water & found
    carved = elevation.copy()
    carved[land] = np.maximum(lowered[land], np.minimum(elevation[land], floor[land]))
    carved[channel] = bed[channel]
    # Past the shore the channel goes on into the lake or sea, so its bed does not rise to a bar across the mouth
    mouth = (to_bank <= 0) & open_water
    carved[mouth] = np.minimum(elevation[mouth], bed[mouth])
    water = np.where(channel, channel_water, np.nan)
    river = np.where(channel, 1 + np.rint(near_strength * 254), 0).astype(np.uint8)
    return carved, water, river


def _centre_lines(rivers: list[list[tuple[int, int]]], surface: np.ndarray, depth: np.ndarray, half_width: np.ndarray,
                  strength: np.ndarray, open_water: np.ndarray) -> np.ndarray:
    """Points along every river's smoothed centre line: rows of (x, z, water, depth, half width, strength).

    Each traced line (cell centres, eight directions) is averaged with its neighbours, then has its
    corners cut (Chaikin), ends fixed; the values along it are smoothed the same way, which keeps the
    water never rising downstream. A line's last point is where it ends (a lake, the sea or another
    river), so it keeps that water and the depth and width of the point before. A line ending in a lake
    or the sea goes on into it (`_mouth`) at that water, its depth fading out."""
    lines = []
    for river in rivers:
        if len(river) < 2:
            continue
        xs, zs = np.array([p[0] for p in river]), np.array([p[1] for p in river])
        values = [grid[zs, xs].astype(float) for grid in (surface, depth, half_width, strength)]
        for v in values[1:]:
            v[-1] = v[-2]
        if np.isnan(values[0][-1]):
            values[0][-1] = values[0][-2]
        line = np.column_stack([xs, zs, *values]).astype(float)
        if open_water[zs[-1], xs[-1]]:
            tail = _mouth(river, open_water)
            fade = 1 - np.arange(1, len(tail) + 1) / (MOUTH_REACH + 1)
            water, end_depth, width, power = line[-1, 2:]
            line = np.vstack([line, np.column_stack([tail, np.full(len(tail), water), end_depth * fade,
                                                     np.full(len(tail), width), np.full(len(tail), power)])])
        for _ in range(SMOOTHING):
            line[1:-1] = (line[:-2] + line[1:-1] + line[2:]) / 3
        for _ in range(2):
            cut = np.empty((2 * (len(line) - 1), line.shape[1]))
            cut[0::2] = 0.75 * line[:-1] + 0.25 * line[1:]
            cut[1::2] = 0.25 * line[:-1] + 0.75 * line[1:]
            line = np.vstack([line[:1], cut, line[-1:]])
        lines.append(line)
    return np.vstack(lines) if lines else np.zeros((0, 6))


def _mouth(river: list[tuple[int, int]], open_water: np.ndarray) -> np.ndarray:
    """(x, z) points a cell apart carrying on from the river's end along its last few cells' heading, up to
    MOUTH_REACH of them, while they stay over open water."""
    rows, cols = open_water.shape
    end = np.array(river[-1], float)
    heading = end - np.array(river[max(len(river) - 4, 0)], float)
    heading /= max(np.hypot(*heading), 1e-9)
    points = []
    for k in range(1, MOUTH_REACH + 1):
        x, z = end + k * heading
        cx, cz = round(x), round(z)
        if not (0 <= cx < cols and 0 <= cz < rows and open_water[cz, cx]):
            break
        points.append((x, z))
    return np.array(points, float).reshape(-1, 2)


def _nearest_samples(samples: np.ndarray, shape: tuple[int, int], reach: float):
    """For every cell within `reach` of a centre-line point: the index of the point nearest its bank
    (distance less half width), the distance to it, and the lowest water of the points whose channel
    covers the cell. -1, inf and inf elsewhere."""
    rows, cols = shape
    near = np.full(rows * cols, -1, np.int64)
    gap = np.full(rows * cols, np.inf)
    water = np.full(rows * cols, np.inf)
    if not len(samples):
        return near.reshape(shape), gap.reshape(shape), water.reshape(shape)
    base_x, base_z = np.rint(samples[:, 0]).astype(np.int64), np.rint(samples[:, 1]).astype(np.int64)
    index = np.arange(len(samples))
    ids, gaps, points = [], [], []
    for dz, dx, _ in disc_offsets(reach):
        x, z = base_x + dx, base_z + dz
        ok = (x >= 0) & (x < cols) & (z >= 0) & (z < rows)
        ids.append(z[ok] * cols + x[ok])
        gaps.append(np.hypot(x[ok] - samples[ok, 0], z[ok] - samples[ok, 1]))
        points.append(index[ok])
    ids, gaps, points = np.concatenate(ids), np.concatenate(gaps), np.concatenate(points)
    to_bank = gaps - samples[points, 4]
    inside = to_bank <= 0
    np.minimum.at(water, ids[inside], samples[points[inside], 2])
    order = np.lexsort((to_bank, ids))
    first = order[np.r_[True, ids[order][1:] != ids[order][:-1]]]
    near[ids[first]], gap[ids[first]] = points[first], gaps[first]
    return near.reshape(shape), gap.reshape(shape), water.reshape(shape)


def _trace_rivers(path: np.ndarray, order: np.ndarray, down: np.ndarray, cols: int) -> list[list[tuple[int, int]]]:
    """Centre lines as polylines, from each source down to where the river ends or joins another."""
    path_l, down_l = path.ravel().tolist(), down.tolist()
    fed = bytearray(len(path_l))  # has a river flowing into it
    for c in np.flatnonzero(path).tolist():
        d = down_l[c]
        if d >= 0:
            fed[d] = 1
    visited = bytearray(len(path_l))
    rivers = []
    for source in order.tolist():  # highest sources first
        if not path_l[source] or fed[source]:
            continue
        visited[source] = 1
        points, c = [source], source
        while (d := down_l[c]) >= 0:
            points.append(d)
            if not path_l[d] or visited[d]:
                break  # reached a lake, the sea, or a river already traced
            visited[d] = 1
            c = d
        rivers.append([(p % cols, p // cols) for p in points])
    return rivers
