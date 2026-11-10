"""A whole-terrain picture for tuning: biome colours, hillshade, and water over them.

Also the deterministic file writers: `write_png` for the preview image and `write_npz` for the
terrain's arrays (byte-for-byte reproducible, so exports and caches compare equal).
"""

import io
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np

from worldgen.proc_gen.world.grid import to_cells
from worldgen.proc_gen.world.terrain import TerrainData

LIGHT = np.array([-1.0, 1.0, -1.0]) / np.sqrt(3)  # from the north-west, 35° up: (x, y, z), z southwards
SHALLOW, DEEP, RIVER = np.array([70, 175, 190]), np.array([25, 70, 130]), np.array([55, 135, 200])
DEEP_AT = 6.0  # metres of water at which the colour is fully deep
WATER_ALPHA = 0.85


def render(terrain: TerrainData) -> np.ndarray:
    """RGB image, one pixel per cell: the ground colours, hillshaded, with the water over them."""
    image = terrain.color * hillshade(terrain.elevation, terrain.config.world.cell_size)[..., None]
    water_over(image, terrain.elevation, terrain.water, terrain.river)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def hillshade(elevation: np.ndarray, cell_size: float) -> np.ndarray:
    """Per cell, the light on the ground from `elevation` (a vertex grid): 1 on flat ground, less on slopes
    facing away from the light, more on those facing it. Multiply colours by it."""
    gz, gx = np.gradient(elevation, cell_size)
    normal = np.stack([-gx, np.ones_like(gx), -gz], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    shade = to_cells(np.clip(normal @ LIGHT, 0, 1)) / LIGHT[1]  # 1 on flat ground
    return np.clip(shade, 0.45, 1.35)


def water_over(image: np.ndarray, elevation: np.ndarray, water: np.ndarray, river: np.ndarray) -> None:
    """Paint water over `image` (float RGB, one pixel per cell): deeper is darker, rivers their own blue.
    `water` is the surface height (NaN where dry) and `river` the strength, both vertex grids like `elevation`."""
    depth = water - elevation
    wet_corners = to_cells((~np.isnan(depth)).astype(np.float64))
    cell_depth = to_cells(np.nan_to_num(depth, nan=0.0)) / np.maximum(wet_corners, 0.25)
    t = np.clip(cell_depth / DEEP_AT, 0, 1)[..., None]
    water_color = SHALLOW + (DEEP - SHALLOW) * t
    is_river = to_cells((river > 0).astype(np.float64)) >= 0.5
    water_color[is_river] = RIVER
    wet = wet_corners >= 0.5
    image[wet] = image[wet] * (1 - WATER_ALPHA) + water_color[wet] * WATER_ALPHA


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an 8-bit RGB image as a PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    height, width, _ = rgb.shape
    raw = np.concatenate([np.zeros((height, 1), np.uint8), rgb.reshape(height, width * 3)], axis=1).tobytes()
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write arrays as a compressed .npz, deterministically: fixed entry order and timestamps, so the same
    terrain gives the same file, byte for byte."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 6
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
