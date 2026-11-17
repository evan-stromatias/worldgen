"""Generate a subterranean cave, write it, read it back with CaveData, and inspect a patch.

    uv run python examples/generate_cave.py --no-window

Generates the big hills cave recipe into examples/out/big_hills_cave (seed 42), loads the bundle with
`CaveData.load`, queries the safe player spawn point, takes the patch under a point, and shows
it in a window: its speleological rock colours, hillshaded floor relief, water sumps, and rock pillars.
"""

import sys
from pathlib import Path

import numpy as np

from worldgen.cave import CaveData, export_cave, generate_cave, load_config
from worldgen.generator import preview

HERE = Path(__file__).parent
OUT = HERE / "out" / "big_hills_cave"
SEED = 42
AT = (250.0, 250.0)  # world coordinate (x, z, metres) whose patch to inspect


def main() -> None:
    config = load_config(HERE / "big_hills_cave.yaml")
    print(f"generating cave: {config.world.size_cells}² cells, seed {SEED}")
    cave = generate_cave(config, SEED, report=lambda stage, seconds: print(f"  {stage:<12} {seconds:5.2f}s"))
    export_cave(cave, OUT)
    print(f"wrote {OUT} (cave.yaml, cave.npz, preview.png)")

    # Read the bundle back: CaveData is the public class for a saved cave bundle
    cave = CaveData.load(OUT)
    nx, nz = cave.patch_shape
    size = cave.config.world.patch_cells
    print(f"the cave tiles into {nx} x {nz} patches of {size} cells a side")
    print(f"floor elevation:   {cave.floor_elevation.min():.1f} m to {cave.floor_elevation.max():.1f} m")
    print(f"ceiling elevation: {cave.ceiling_elevation.min():.1f} m to {cave.ceiling_elevation.max():.1f} m")
    print(f"vertical headroom: {cave.clearance.min():.1f} m to {cave.clearance.max():.1f} m")

    spawn_pos, _yaw, _pitch = cave.find_spawn_point()
    clr = cave.clearance_at(float(spawn_pos[0]), float(spawn_pos[2]))
    print(f"safe spawn point:  x={spawn_pos[0]:.1f}m, y={spawn_pos[1]:.1f}m, z={spawn_pos[2]:.1f}m (clearance={clr:.1f}m)")

    patch = cave.patch_at(*AT)
    print(f"at x {AT[0]:.0f} m, z {AT[1]:.0f} m: patch {patch.key}, origin {patch.origin}")
    open_count = int(np.sum(patch.pillar_mask == 0))
    pillar_count = int(np.sum(patch.pillar_mask != 0))
    print(f"patch {patch.key}: {open_count} open cells, {pillar_count} pillar/rock cells")

    if "--no-window" not in sys.argv:
        try:
            show(patch)
        except Exception as e:
            print(f"Skipping window display ({e})")


def show(patch) -> None:
    """Draw the cave patch in a window: floor colors, hillshade, water sumps, and pillars."""
    import pyray as rl

    from worldgen.mesh.gpu import GpuTexture

    # Base ground colors shaded by floor relief
    shade = preview.hillshade(patch.vertex_floor, patch.cell_size)
    image = patch.color.astype(np.float64) * shade[..., None]

    # Water sumps
    wet = ~np.isnan(patch.water[:-1, :-1])
    image[wet] = np.array([45, 95, 140], dtype=np.float64)

    # Darken solid rock / pillars
    pillars = patch.pillar_mask != 0
    image[pillars] *= 0.35

    rgb = np.clip(np.rint(image), 0, 255).astype(np.uint8)

    zoom = 3  # screen pixels per cell
    side = patch.cells * zoom
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(side, side + 90, f"cavegen patch {patch.key}")
    texture = GpuTexture(rgb).texture
    lines = [
        f"patch {patch.key}, {patch.cells} cells a side, from x {patch.origin[0]:.0f} m, z {patch.origin[1]:.0f} m",
        f"floor {patch.vertex_floor.min():.1f} m to {patch.vertex_floor.max():.1f} m | ceiling {patch.vertex_ceiling.min():.1f} m to {patch.vertex_ceiling.max():.1f} m",
        f"clearance {patch.clearance.min():.1f} m to {patch.clearance.max():.1f} m",
    ]
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.Color(20, 22, 28, 255))
        rl.draw_texture_ex(texture, rl.Vector2(0, 0), 0.0, zoom, rl.WHITE)
        for i, line in enumerate(lines):
            rl.draw_text(line, 10, side + 8 + i * 24, 20, rl.Color(220, 218, 206, 255))
        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
