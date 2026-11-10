"""Small array helpers shared by the stages.

Grids are indexed [z][x]: rows run north to south, columns west to east. Stages work on the
vertex grid, (size_cells + 1) samples per side; per-cell outputs come from `to_cells`.
"""

import numpy as np


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """0 below edge0, 1 above edge1, smooth in between. edge0 > edge1 gives the falling version."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ramp(values: np.ndarray, lo: float, hi: float, width: float, domain: tuple[float, float] | None) -> np.ndarray:
    """1 inside [lo, hi], 0 outside, ramping over `width` centred on each edge (0.5 exactly at the edge). An edge on
    the `domain`'s bound (values cannot pass it) or at ±inf does not ramp."""
    if width <= 0:
        return ((values >= lo) & (values <= hi)).astype(np.float64)
    out = np.ones(values.shape)
    if (domain is None or lo > domain[0]) and np.isfinite(lo):
        out *= smoothstep(lo - width / 2, lo + width / 2, values)
    if (domain is None or hi < domain[1]) and np.isfinite(hi):
        out *= 1 - smoothstep(hi - width / 2, hi + width / 2, values)
    return out


def distance_to(mask: np.ndarray, spacing: float, reach: float) -> np.ndarray:
    """Euclidean distance from each sample to the nearest True one, in the units of `spacing`; inf past `reach`.

    Exact within `reach`: a pass down and up each column finds the nearest one in the column (a running minimum),
    then each row takes the best of its neighbours up to `reach` away, one offset at a time."""
    rows, cols = mask.shape
    big = float(rows + cols)
    index = np.arange(rows, dtype=np.float64)[:, None]
    below = np.where(mask, index, -big)  # the nearest True at or above each row, as a row index
    above = np.where(mask, index, 2 * big)
    nearest_up = np.maximum.accumulate(below, axis=0)
    nearest_down = np.minimum.accumulate(above[::-1], axis=0)[::-1]
    column = np.minimum(index - nearest_up, nearest_down - index)  # cells, along the column
    squared = column ** 2
    best = squared.copy()
    steps = int(reach / spacing)
    for dx in range(1, steps + 1):
        best[:, dx:] = np.minimum(best[:, dx:], squared[:, :-dx] + dx * dx)
        best[:, :-dx] = np.minimum(best[:, :-dx], squared[:, dx:] + dx * dx)
    distance = np.sqrt(best) * spacing
    distance[distance > reach] = np.inf
    return distance


def shift(a: np.ndarray, dz: int, dx: int, fill: float) -> np.ndarray:
    """out[z, x] = a[z + dz, x + dx], with `fill` where that falls outside the grid."""
    out = np.full_like(a, fill)
    rows, cols = a.shape
    if abs(dz) >= rows or abs(dx) >= cols:
        return out
    out[max(-dz, 0):rows - max(dz, 0), max(-dx, 0):cols - max(dx, 0)] = (
        a[max(dz, 0):rows + min(dz, 0), max(dx, 0):cols + min(dx, 0)]
    )
    return out


def disc_offsets(radius: float) -> list[tuple[int, int, float]]:
    """(dz, dx, distance) of every cell within `radius`, centre first, in a fixed order."""
    r = int(radius)
    offsets = [
        (dz, dx, float(np.hypot(dz, dx)))
        for dz in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if np.hypot(dz, dx) <= radius
    ]
    return sorted(offsets, key=lambda o: (o[2], o[0], o[1]))


def to_cells(a: np.ndarray) -> np.ndarray:
    """Average the four corner vertices of each cell: (n + 1)² vertices -> n² cells."""
    return (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:]) / 4


def box_blur(a: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Repeated box blur (three passes is close to a Gaussian with sigma ≈ radius). Edges are clamped."""
    out = a.astype(np.float64)
    width = 2 * radius + 1
    for _ in range(passes):
        for axis in (0, 1):
            moved = np.moveaxis(out, axis, 0)
            padded = np.pad(moved, ((radius + 1, radius), (0, 0)), mode="edge")
            padded[0] = 0  # cumulative sums start from zero
            sums = np.cumsum(padded, axis=0)
            out = np.moveaxis((sums[width:] - sums[:-width]) / width, 0, axis)
    return out


def min_filter(a: np.ndarray, radius: int) -> np.ndarray:
    """The least value within `radius` samples each way (a square window), clamped at the edges.

    Van Herk and Gil-Werman's running minimum, one axis at a time: blocks of the window's width, each with its
    minimum from the left and from the right, so any window is the smaller of two lookups, however wide it is."""
    if radius <= 0:
        return a.copy()
    out = a
    width = 2 * radius + 1
    for axis in (0, 1):
        moved = np.moveaxis(out, axis, 0)
        length = moved.shape[0]
        blocks = -(-(length + 2 * radius) // width)
        padded = np.pad(moved, ((radius, blocks * width - length - radius), (0, 0)), mode="edge")
        shaped = padded.reshape(blocks, width, -1)
        left = np.minimum.accumulate(shaped, axis=1).reshape(blocks * width, -1)
        right = np.minimum.accumulate(shaped[:, ::-1], axis=1)[:, ::-1].reshape(blocks * width, -1)
        # The window starting at i covers i .. i + width - 1: the right half of i's block and the left of the next
        out = np.moveaxis(np.minimum(right[:length], left[width - 1:width - 1 + length]), 0, axis)
    return out
