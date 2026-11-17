"""Dual-shell 3D mesh generation for subterranean caves.

Pure numpy so mesh generation runs on worker threads; raylib uploads on the main thread.
Constructs floor, inverted-winding ceiling, and vertical rock skirts for natural pillars and boundaries.
"""

import numpy as np

from worldgen.cave.generator import CaveData
from worldgen.generator.grid import to_cells
from worldgen.mesh.builders import MeshArrays


def build_cave_mesh(
    cave: CaveData,
    stride: int = 1,
    ceiling_offset: float = 0.0,
    include_ceiling: bool = True,
    cutaway_height: float | None = None,
) -> MeshArrays:
    """Build dual-shell cave mesh (floor, ceiling, and vertical pillar/wall skirts).

    - Floor winding is counter-clockwise seen from above.
    - Ceiling winding is inverted (CW seen from above, CCW looking up from inside).
    - Open cells emit floor and ceiling quads; pillar cells have their quads discarded.
    - Vertical skirts seal pillar boundaries and outer perimeter walls.
    """
    cell_size = cave.config.world.cell_size
    n = cave.config.world.size_cells
    if n % stride != 0:
        raise ValueError(f"stride {stride} does not divide size_cells {n}")

    keep = np.arange(0, n + 1, stride)
    k = len(keep)
    rows, cols = np.meshgrid(keep, keep, indexing="ij")
    x = (cols * cell_size).astype(np.float32)
    z = (rows * cell_size).astype(np.float32)

    # Sample heights
    hf = cave.floor_elevation[rows, cols].astype(np.float32)
    hc = (cave.ceiling_elevation[rows, cols] + ceiling_offset).astype(np.float32)

    # Compute floor normals using central differences (pointing up)
    spacing = cell_size * stride
    left = np.pad(hf[:, :-1], ((0, 0), (1, 0)), mode="edge")
    right = np.pad(hf[:, 1:], ((0, 0), (0, 1)), mode="edge")
    up = np.pad(hf[:-1, :], ((1, 0), (0, 0)), mode="edge")
    down = np.pad(hf[1:, :], ((0, 1), (0, 0)), mode="edge")
    floor_normals = np.stack([(left - right) / (2 * spacing), np.ones_like(hf), (up - down) / (2 * spacing)], axis=-1)
    floor_normals /= np.linalg.norm(floor_normals, axis=-1, keepdims=True)

    # Compute ceiling normals (pointing down into the cave)
    left_c = np.pad(hc[:, :-1], ((0, 0), (1, 0)), mode="edge")
    right_c = np.pad(hc[:, 1:], ((0, 0), (0, 1)), mode="edge")
    up_c = np.pad(hc[:-1, :], ((1, 0), (0, 0)), mode="edge")
    down_c = np.pad(hc[1:, :], ((0, 1), (0, 0)), mode="edge")
    ceil_normals = np.stack([(right_c - left_c) / (2 * spacing), -np.ones_like(hc), (down_c - up_c) / (2 * spacing)], axis=-1)
    ceil_normals /= np.linalg.norm(ceil_normals, axis=-1, keepdims=True)

    # Texcoords (u, v) in [0, 1]
    u = (cols / n).astype(np.float32)
    v = (rows / n).astype(np.float32)

    # Per-vertex colors from cave cell colors
    # Pad cell colors to vertex grid
    color_cells = cave.color.astype(np.float32)
    padded_colors = np.pad(color_cells, ((0, 1), (0, 1), (0, 0)), mode="edge")
    sampled_colors = padded_colors[rows, cols]
    floor_colors = np.concatenate([sampled_colors, np.full((k, k, 1), 255, dtype=np.float32)], axis=-1).astype(np.uint8)

    # Ceiling colors slightly shifted for speleological rock contrast
    ceil_colors_rgb = np.clip(sampled_colors * 0.9 + 10, 0, 255)
    ceil_colors = np.concatenate([ceil_colors_rgb, np.full((k, k, 1), 255, dtype=np.float32)], axis=-1).astype(np.uint8)

    # Subsampled pillar mask for open cells
    pillar_cells = cave.pillar_mask[rows[:-1, :-1], cols[:-1, :-1]]
    open_cells = pillar_cells == 0

    positions_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    texcoords_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    indices_list: list[np.ndarray] = []

    # 1. Floor Mesh
    floor_pos = np.stack([x, hf, z], axis=-1).reshape(-1, 3)
    floor_norm = floor_normals.reshape(-1, 3)
    floor_tc = np.stack([u, v], axis=-1).reshape(-1, 2)
    floor_col = floor_colors.reshape(-1, 4)

    positions_list.append(floor_pos)
    normals_list.append(floor_norm)
    texcoords_list.append(floor_tc)
    colors_list.append(floor_col)

    # Floor triangles for open cells
    r_idx, c_idx = np.nonzero(open_cells)
    v00 = r_idx * k + c_idx
    v01 = v00 + 1
    v10 = v00 + k
    v11 = v00 + k + 1

    floor_tri1 = np.stack([v00, v10, v01], axis=-1)
    floor_tri2 = np.stack([v01, v10, v11], axis=-1)
    indices_list.append(np.concatenate([floor_tri1, floor_tri2], axis=0))

    # 2. Ceiling Mesh (optional or cutaway)
    if include_ceiling:
        base_ceil = len(floor_pos)
        ceil_pos = np.stack([x, hc, z], axis=-1).reshape(-1, 3)
        ceil_norm = ceil_normals.reshape(-1, 3)
        ceil_tc = np.stack([u, v], axis=-1).reshape(-1, 2)
        ceil_col = ceil_colors.reshape(-1, 4)

        positions_list.append(ceil_pos)
        normals_list.append(ceil_norm)
        texcoords_list.append(ceil_tc)
        colors_list.append(ceil_col)

        # Inverted winding: (v00, v01, v10) and (v01, v11, v10)
        c00 = base_ceil + v00
        c01 = base_ceil + v01
        c10 = base_ceil + v10
        c11 = base_ceil + v11

        if cutaway_height is not None:
            # Clip ceiling triangles above cutaway height
            cell_hc = hc[r_idx, c_idx]
            visible = cell_hc < cutaway_height
            c00, c01, c10, c11 = c00[visible], c01[visible], c10[visible], c11[visible]

        ceil_tri1 = np.stack([c00, c01, c10], axis=-1)
        ceil_tri2 = np.stack([c01, c11, c10], axis=-1)
        indices_list.append(np.concatenate([ceil_tri1, ceil_tri2], axis=0))

        # 3. Pillar & Perimeter Rock Skirts
        # Vertical quads joining floor to ceiling where an open cell borders a pillar cell or grid boundary
        skirt_pos: list[list[float]] = []
        skirt_norm: list[list[float]] = []
        skirt_tc: list[list[float]] = []
        skirt_col: list[list[float]] = []
        skirt_idx: list[list[int]] = []

        m_cells = k - 1
        for r, c in zip(r_idx, c_idx, strict=False):
            # Check 4 cardinal neighbors
            # North neighbor (r-1, c)
            if r == 0 or pillar_cells[r - 1, c] != 0:
                # Quad between floor (r, c) -> (r, c+1) and ceiling (r, c) -> (r, c+1)
                idx_base = len(floor_pos) + len(ceil_pos) + len(skirt_pos)
                p_f0 = [x[r, c], hf[r, c], z[r, c]]
                p_f1 = [x[r, c + 1], hf[r, c + 1], z[r, c + 1]]
                p_c0 = [x[r, c], hc[r, c], z[r, c]]
                p_c1 = [x[r, c + 1], hc[r, c + 1], z[r, c + 1]]
                n_vec = [0.0, 0.0, 1.0]  # pointing south into cell
                skirt_pos.extend([p_f0, p_f1, p_c1, p_c0])
                skirt_norm.extend([n_vec, n_vec, n_vec, n_vec])
                skirt_tc.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                c_val = list(floor_colors[r, c])
                skirt_col.extend([c_val, c_val, c_val, c_val])
                skirt_idx.append([idx_base, idx_base + 1, idx_base + 2])
                skirt_idx.append([idx_base, idx_base + 2, idx_base + 3])

            # South neighbor (r+1, c)
            if r == m_cells - 1 or pillar_cells[r + 1, c] != 0:
                idx_base = len(floor_pos) + len(ceil_pos) + len(skirt_pos)
                p_f0 = [x[r + 1, c + 1], hf[r + 1, c + 1], z[r + 1, c + 1]]
                p_f1 = [x[r + 1, c], hf[r + 1, c], z[r + 1, c]]
                p_c0 = [x[r + 1, c + 1], hc[r + 1, c + 1], z[r + 1, c + 1]]
                p_c1 = [x[r + 1, c], hc[r + 1, c], z[r + 1, c]]
                n_vec = [0.0, 0.0, -1.0]  # pointing north into cell
                skirt_pos.extend([p_f0, p_f1, p_c1, p_c0])
                skirt_norm.extend([n_vec, n_vec, n_vec, n_vec])
                skirt_tc.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                c_val = list(floor_colors[r + 1, c])
                skirt_col.extend([c_val, c_val, c_val, c_val])
                skirt_idx.append([idx_base, idx_base + 1, idx_base + 2])
                skirt_idx.append([idx_base, idx_base + 2, idx_base + 3])

            # West neighbor (r, c-1)
            if c == 0 or pillar_cells[r, c - 1] != 0:
                idx_base = len(floor_pos) + len(ceil_pos) + len(skirt_pos)
                p_f0 = [x[r + 1, c], hf[r + 1, c], z[r + 1, c]]
                p_f1 = [x[r, c], hf[r, c], z[r, c]]
                p_c0 = [x[r + 1, c], hc[r + 1, c], z[r + 1, c]]
                p_c1 = [x[r, c], hc[r, c], z[r, c]]
                n_vec = [1.0, 0.0, 0.0]  # pointing east into cell
                skirt_pos.extend([p_f0, p_f1, p_c1, p_c0])
                skirt_norm.extend([n_vec, n_vec, n_vec, n_vec])
                skirt_tc.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                c_val = list(floor_colors[r, c])
                skirt_col.extend([c_val, c_val, c_val, c_val])
                skirt_idx.append([idx_base, idx_base + 1, idx_base + 2])
                skirt_idx.append([idx_base, idx_base + 2, idx_base + 3])

            # East neighbor (r, c+1)
            if c == m_cells - 1 or pillar_cells[r, c + 1] != 0:
                idx_base = len(floor_pos) + len(ceil_pos) + len(skirt_pos)
                p_f0 = [x[r, c + 1], hf[r, c + 1], z[r, c + 1]]
                p_f1 = [x[r + 1, c + 1], hf[r + 1, c + 1], z[r + 1, c + 1]]
                p_c0 = [x[r, c + 1], hc[r, c + 1], z[r, c + 1]]
                p_c1 = [x[r + 1, c + 1], hc[r + 1, c + 1], z[r + 1, c + 1]]
                n_vec = [-1.0, 0.0, 0.0]  # pointing west into cell
                skirt_pos.extend([p_f0, p_f1, p_c1, p_c0])
                skirt_norm.extend([n_vec, n_vec, n_vec, n_vec])
                skirt_tc.extend([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
                c_val = list(floor_colors[r, c + 1])
                skirt_col.extend([c_val, c_val, c_val, c_val])
                skirt_idx.append([idx_base, idx_base + 1, idx_base + 2])
                skirt_idx.append([idx_base, idx_base + 2, idx_base + 3])

        if skirt_pos:
            positions_list.append(np.array(skirt_pos, dtype=np.float32))
            normals_list.append(np.array(skirt_norm, dtype=np.float32))
            texcoords_list.append(np.array(skirt_tc, dtype=np.float32))
            colors_list.append(np.array(skirt_col, dtype=np.uint8))
            indices_list.append(np.array(skirt_idx, dtype=np.uint32))

    positions = np.concatenate(positions_list, axis=0).astype(np.float32)
    normals = np.concatenate(normals_list, axis=0).astype(np.float32)
    texcoords = np.concatenate(texcoords_list, axis=0).astype(np.float32)
    colors = np.concatenate(colors_list, axis=0).astype(np.uint8)
    indices = np.concatenate(indices_list, axis=0).astype(np.uint32)

    return MeshArrays(positions=positions, normals=normals, texcoords=texcoords, indices=indices, colors=colors)


def build_cave_water_mesh(cave: CaveData, stride: int = 1) -> MeshArrays | None:
    """Build water surface mesh for flooded subterranean pools and sumps."""
    if not cave.config.hydrology.enabled or not np.any(~np.isnan(cave.water)):
        return None

    wt = float(cave.config.hydrology.water_table)
    cell_size = cave.config.world.cell_size
    n = cave.config.world.size_cells
    keep = np.arange(0, n + 1, stride)
    k = len(keep)
    rows, cols = np.meshgrid(keep, keep, indexing="ij")

    # Wet cell condition: open cell and floor < water table
    pillar_cells = cave.pillar_mask[rows[:-1, :-1], cols[:-1, :-1]]
    hf_cells = to_cells(cave.floor_elevation)[rows[:-1, :-1], cols[:-1, :-1]]
    wet_cells = (pillar_cells == 0) & (hf_cells < wt)

    if not np.any(wet_cells):
        return None

    x = (cols * cell_size).astype(np.float32)
    z = (rows * cell_size).astype(np.float32)
    y = np.full_like(x, wt, dtype=np.float32)

    positions = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (len(positions), 1))
    texcoords = np.stack([(cols / n).astype(np.float32), (rows / n).astype(np.float32)], axis=-1).reshape(-1, 2)
    water_color = np.tile(np.array([[45, 95, 140, 200]], dtype=np.uint8), (len(positions), 1))

    r_idx, c_idx = np.nonzero(wet_cells)
    v00 = r_idx * k + c_idx
    v01 = v00 + 1
    v10 = v00 + k
    v11 = v00 + k + 1

    tri1 = np.stack([v00, v10, v01], axis=-1)
    tri2 = np.stack([v01, v10, v11], axis=-1)
    indices = np.concatenate([tri1, tri2], axis=0).astype(np.uint32)

    return MeshArrays(positions=positions, normals=normals, texcoords=texcoords, indices=indices, colors=water_color)
