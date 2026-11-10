"""Generate a terrain, write it, read it back, and look at one patch of it.

    uv run python examples/generate.py

Generates the archipelago recipe into examples/out/archipelago (seed 7), loads the bundle with
`TerrainData.load`, takes the patch under a point, and shows it in a window: its ground colours,
hillshaded, with its water — and what the patch knows about itself written in the corner.
"""

from pathlib import Path

import numpy as np

from worldgen import TerrainData, export_terrain, generate_terrain
from worldgen.generator import preview
from worldgen.generator.config import load_config

HERE = Path(__file__).parent
OUT = HERE / "out" / "archipelago"
SEED = 7
AT = (700.0, 300.0)  # the world point (x, z, metres) whose patch to show


def main() -> None:
    config = load_config(HERE / "archipelago.yaml")
    print(f"generating {config.world.size_cells}² cells, seed {SEED}")
    terrain = generate_terrain(config, SEED, report=lambda stage, seconds: print(f"  {stage:<10} {seconds:5.2f}s"))
    export_terrain(terrain, OUT)
    print(f"wrote {OUT} (terrain.yaml, terrain.npz, preview.png)")

    # Read the bundle back: TerrainData is the class for a directory written by export_terrain
    # (or the UI's Export world)
    terrain = TerrainData.load(OUT)
    nx, nz = terrain.patch_shape  # how many patches along x and z
    size = terrain.config.world.patch_cells
    print(f"the terrain tiles into {nx} x {nz} patches of {size} cells a side")
    patch = terrain.patch_at(*AT)
    print(f"at x {AT[0]:.0f} m, z {AT[1]:.0f} m: patch {patch.key}, "
          f"{len(patch.lakes)} lakes and {len(patch.rivers)} rivers")
    show(patch)


def show(patch) -> None:
    """Draw the patch in a window: its colours, hillshaded, with its water, and what it knows."""
    import pyray as rl

    from worldgen.mesh.gpu import GpuTexture

    image = patch.color.astype(np.float64) * preview.hillshade(patch.vertex_height, patch.cell_size)[..., None]
    preview.water_over(image, patch.vertex_height, patch.water, patch.river)
    rgb = np.clip(np.rint(image), 0, 255).astype(np.uint8)

    zoom = 4  # screen pixels per cell
    side = patch.cells * zoom
    rl.set_config_flags(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    rl.init_window(side, side + 90, f"worldgen patch {patch.key}")
    texture = GpuTexture(rgb).texture
    lines = [
        f"patch {patch.key}, {patch.cells} cells a side, from x {patch.origin[0]:.0f} m, z {patch.origin[1]:.0f} m",
        f"ground {patch.vertex_height.min():.1f} m to {patch.vertex_height.max():.1f} m",
        f"{len(patch.lakes)} lakes, {len(patch.rivers)} rivers",
    ]
    while not rl.window_should_close():
        rl.begin_drawing()
        rl.clear_background(rl.Color(32, 36, 44, 255))
        rl.draw_texture_ex(texture, rl.Vector2(0, 0), 0.0, zoom, rl.WHITE)
        for i, line in enumerate(lines):
            rl.draw_text(line, 10, side + 8 + i * 24, 20, rl.Color(238, 236, 222, 255))
        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
