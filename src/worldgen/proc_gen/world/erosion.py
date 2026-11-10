"""Stage 2: weathering. Simulated droplets cut gullies and valleys, then loose ground slides to the angle it holds.

The droplets follow Beyer 2015 ("Implementation of a method for hydraulic erosion"), with three departures, the
first two for speed in numpy:

- **Batches.** `batch` droplets step together as arrays, and their changes are added with `np.add.at`, which adds
  in index order, so a run is deterministic. A droplet sees the others' changes one step late.
- **Coarse to fine.** Droplets on a 1 m grid make rills, not valleys the size of mountains, so the ground is eroded
  first at every 8th vertex, then every 4th, then every one (with 3 levels). Each level's change is upsampled
  (Catmull-Rom) onto the full grid, so the noise's fine detail survives on the eroded shapes.
- **Smoothing.** Each level's change is blurred `smoothing` vertices before it is added. Droplets crossing a slope in
  parallel cut it into a washboard of ripples about two vertices apart (their brushes sit on the nearest vertex, so
  the lines alias); the ground's colour by slope turned them into stripes. A brush centred on each droplet's exact
  point also removes them, at twice the time.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from worldgen.proc_gen.world.config import ErosionSection, WorldSection
from worldgen.proc_gen.world.grid import box_blur, disc_offsets, shift

NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
START_WATER, START_SPEED = 1.0, 1.0
BATCH_SPREAD = 256  # a batch has at most one droplet per this many vertices, so few cut the same spot in one step


@dataclass(frozen=True, slots=True)
class Erosion:
    heights: np.ndarray  # vertex grid, metres: the eroded ground
    sediment: np.ndarray  # vertex grid, metres deposited (0 where the ground was cut)


def level_factors(levels: int) -> list[int]:
    """Every how many vertices each level samples, coarsest first: 3 levels are 8, 4, 1."""
    return [2 ** (levels - i + 1) for i in range(1, levels)] + [1]


def generate(
    heights: np.ndarray,
    rock: np.ndarray | None,
    world: WorldSection,
    cfg: ErosionSection,
    rng: np.random.Generator,
) -> Erosion:
    """`heights` eroded by droplets, level by level, then by sliding. `rock` (0-1, the terrace mask, or None) is
    where the steeper `rock_talus_angle` holds. Each pass prints a line, as the other stages print what they did."""
    original = heights
    ground = heights.astype(np.float64).copy()
    levels = level_factors(cfg.levels)
    for number, factor in enumerate(levels, start=1):
        start = time.perf_counter()
        coarse = _downsample(ground, factor)
        eroded = coarse.copy()
        count = round(cfg.droplets * eroded.size)
        droplets(eroded, world.cell_size * factor, world.sea_level, count, cfg, rng)
        change = eroded - coarse
        if cfg.smoothing:
            change = box_blur(change, cfg.smoothing)
        ground += _upsample(change, factor, ground.shape[0])
        print(f"  erosion level {number} of {len(levels)} (every {factor} vertices): {_count(count)} droplets, "
              f"{time.perf_counter() - start:.1f} s")
    if cfg.thermal_iterations:
        start = time.perf_counter()
        thermal(ground, rock, world.cell_size, world.sea_level, cfg)
        print(f"  erosion thermal: {cfg.thermal_iterations} iterations, {time.perf_counter() - start:.1f} s")
    return Erosion(ground, np.maximum(ground - original, 0).astype(np.float32))


def _count(count: int) -> str:
    return f"{count / 1e6:.1f} M" if count >= 1e6 else f"{count / 1e3:.0f} k" if count >= 1e3 else str(count)


def _downsample(heights: np.ndarray, factor: int) -> np.ndarray:
    """Every `factor`-th vertex each way, blurred first so the coarse grid holds the shapes and not their noise."""
    if factor == 1:
        return heights.copy()
    return box_blur(heights, factor // 2)[::factor, ::factor].copy()


def _catmull_rom(t: np.ndarray) -> np.ndarray:
    """The four Catmull-Rom weights of a point `t` (0-1) of the way from the second sample to the third."""
    t2, t3 = t * t, t * t * t
    return np.stack([-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2])


def _upsample(change: np.ndarray, factor: int, size: int) -> np.ndarray:
    """A coarse grid (every `factor`-th vertex) on the full `size`² grid, bicubic, one axis at a time."""
    if factor == 1:
        return change
    position = np.arange(size) / factor
    base = np.floor(position).astype(np.int64)
    weights = _catmull_rom(position - base)  # (4, size)
    last = change.shape[0] - 1
    taps = [np.clip(base + k - 1, 0, last) for k in range(4)]
    rows = sum(weights[k][:, None] * change[taps[k], :] for k in range(4))  # (size, coarse)
    return sum(weights[k][None, :] * rows[:, taps[k]] for k in range(4))


def droplets(
    heights: np.ndarray, spacing: float, sea_level: float, count: int, cfg: ErosionSection, rng: np.random.Generator
) -> None:
    """Erode `heights` (a grid `spacing` metres apart) in place with `count` droplets, `cfg.batch` at a time (fewer on
    a small grid: droplets in one batch that cut the same spot in the same step would dig it twice as deep).

    - A droplet runs downhill on the bilinear gradient, keeping `inertia` of its direction. Going down it picks up
      sediment up to its capacity (`capacity · drop · speed · water`, the drop in metres over the step, so a droplet
      on a coarser grid stands for the rain of a larger area), over a brush of `radius` vertices so it digs no
      one-vertex pits; where it slows, is too full or would go uphill it drops sediment over the same brush.
    - It stops when it leaves the grid (losing what it carries), stands still or runs out of steps; reaching the sea
      or standing still, it drops what it carries there.
    - Only the starting points draw from `rng`, in order, a batch at a time.
    """
    n = heights.shape[0]
    brush = disc_offsets(cfg.radius)
    brush_dz = np.array([dz for dz, _, _ in brush])
    brush_dx = np.array([dx for _, dx, _ in brush])
    brush_weight = np.array([1 - distance / (cfg.radius + 1) for _, _, distance in brush])
    brush_weight /= brush_weight.sum()
    flat = heights.reshape(-1)
    batch = max(1, min(cfg.batch, heights.size // BATCH_SPREAD))
    for first in range(0, count, batch):
        size = min(batch, count - first)
        px, pz = rng.uniform(0, n - 1, size), rng.uniform(0, n - 1, size)
        dx, dz = np.zeros(size), np.zeros(size)
        speed, water, sediment = np.full(size, START_SPEED), np.full(size, START_WATER), np.zeros(size)
        alive = np.ones(size, bool)
        for _ in range(cfg.lifetime):
            live = np.flatnonzero(alive)
            if not live.size:
                break
            x, z = px[live], pz[live]
            height, gx, gz = _sample(heights, x, z)
            ndx = dx[live] * cfg.inertia - gx * (1 - cfg.inertia)
            ndz = dz[live] * cfg.inertia - gz * (1 - cfg.inertia)
            length = np.hypot(ndx, ndz)
            moving = length > 1e-12
            length[~moving] = 1.0
            ndx, ndz = ndx / length, ndz / length
            nx, nz = x + ndx, z + ndz
            inside = moving & (nx >= 0) & (nx < n - 1) & (nz >= 0) & (nz < n - 1)
            new_height = np.full(live.size, np.inf)
            new_height[inside], _, _ = _sample(heights, nx[inside], nz[inside])
            drop = new_height - height  # metres, over one grid step
            carried = sediment[live]
            capacity = np.maximum(-drop, cfg.min_slope * spacing) * speed[live] * water[live] * cfg.capacity
            at_sea = inside & (new_height < sea_level)
            uphill = inside & (drop > 0)
            full = inside & ~uphill & (carried > capacity)
            # What it drops: all at sea, enough to fill the rise going uphill, a share of the excess when full
            deposit = np.zeros(live.size)
            deposit[~moving] = carried[~moving]  # in a pit or on a flat: it fills it
            deposit[at_sea] = carried[at_sea]
            rest = uphill & ~at_sea
            deposit[rest] = np.minimum(drop[rest], carried[rest])
            rest = full & ~at_sea
            deposit[rest] = (carried[rest] - capacity[rest]) * cfg.deposit_rate
            erode = np.zeros(live.size)
            cuts = inside & ~at_sea & ~uphill & ~full
            erode[cuts] = np.minimum((capacity[cuts] - carried[cuts]) * cfg.erode_rate, -drop[cuts])
            _spread(flat, n, x, z, deposit - erode, brush_dz, brush_dx, brush_weight)
            sediment[live] = carried - deposit + erode
            px[live], pz[live], dx[live], dz[live] = nx, nz, ndx, ndz
            speed[live] = np.sqrt(np.maximum(speed[live] ** 2 - np.where(inside, drop, 0) / spacing * cfg.gravity, 0))
            water[live] *= 1 - cfg.evaporation
            alive[live] = inside & ~at_sea


def _sample(heights: np.ndarray, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bilinear height at (x, z), in grid units, and its gradient (metres per grid step) along x and z."""
    ix, iz = x.astype(np.int64), z.astype(np.int64)
    fx, fz = x - ix, z - iz
    h00, h10 = heights[iz, ix], heights[iz, ix + 1]
    h01, h11 = heights[iz + 1, ix], heights[iz + 1, ix + 1]
    gx = (h10 - h00) * (1 - fz) + (h11 - h01) * fz
    gz = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    height = h00 * (1 - fx) * (1 - fz) + h10 * fx * (1 - fz) + h01 * (1 - fx) * fz + h11 * fx * fz
    return height, gx, gz


def _spread(
    flat: np.ndarray, n: int, x: np.ndarray, z: np.ndarray, amount: np.ndarray,
    brush_dz: np.ndarray, brush_dx: np.ndarray, brush_weight: np.ndarray,
) -> None:
    """Add `amount` (negative to take) around each (x, z), spread over the brush round its nearest vertex; what falls
    off the grid is lost. Deposits are spread too, not put on the four vertices around the point: those leave
    one-vertex bumps wherever droplets stop."""
    some = amount != 0
    if not some.any():
        return
    cx, cz, amount = np.rint(x[some]).astype(np.int64), np.rint(z[some]).astype(np.int64), amount[some]
    bz, bx = cz[:, None] + brush_dz[None, :], cx[:, None] + brush_dx[None, :]
    on = (bz >= 0) & (bz < n) & (bx >= 0) & (bx < n)
    np.add.at(flat, (bz * n + bx)[on], (amount[:, None] * brush_weight[None, :])[on])


def thermal(heights: np.ndarray, rock: np.ndarray | None, spacing: float, sea_level: float, cfg: ErosionSection) -> None:
    """Slide ground downhill in place until no slope to a neighbour is much steeper than it holds.

    - Each pass, a vertex steeper than its angle to some of its 8 neighbours sends them `thermal_rate` of half its
      largest excess, shared by how much each is in excess (Musgrave, Kolb and Mace 1989).
    - The angle is `talus_angle`, rising to `rock_talus_angle` inside `rock`.
    - Only land slides, and it measures its slope to the sea against the sea's surface: the shelf under the sea is
      steep on purpose (elevation maps the sea floor apart), and the coast would slide down it.
    """
    soft, hard = math.tan(math.radians(cfg.talus_angle)), math.tan(math.radians(cfg.rock_talus_angle))
    holds = soft if rock is None else soft + (hard - soft) * rock

    def excess(dz: int, dx: int) -> np.ndarray:
        # Worked out twice a pass rather than kept: eight full grids would be a gigabyte at 4096²
        limit = holds * (spacing * math.hypot(dz, dx))
        below = np.maximum(shift(heights, dz, dx, np.inf), sea_level)  # nothing slides off the world
        return np.where(heights > sea_level, np.maximum(heights - below - limit, 0), 0)

    for _ in range(cfg.thermal_iterations):
        total, largest = np.zeros_like(heights), np.zeros_like(heights)
        for dz, dx in NEIGHBOURS:
            over = excess(dz, dx)
            total += over
            np.maximum(largest, over, out=largest)
        moved = cfg.thermal_rate * largest / 2 / np.where(total > 0, total, 1)
        change = np.zeros_like(heights)
        for dz, dx in NEIGHBOURS:
            share = moved * excess(dz, dx)
            change -= share
            change += shift(share, -dz, -dx, 0.0)  # to the neighbour at (z + dz, x + dx)
        heights += change
