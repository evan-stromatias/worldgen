2026-11-10, 0.1.2:
  - fix outdated tests

2026-11-10, 0.1.1:
  - dummy change

2026-11-10, 0.1.0:
  - Standalone terrain generator: five stages (elevation, erosion, hydrology, climate, biomes) from a
    YAML recipe and a seed
  - `worldgen` CLI and `worldgen-ui` (2D/3D, recipe forms, Export world)
  - The terrain bundle: terrain.yaml + terrain.npz (deterministic) + preview.png
  - `TerrainData`: the bundle in memory, with slope/relief/biome_weights derived on demand, point
    samplers (`height_at`, `water_at`, `biome_at`) and patch reading (`patch`, `patch_at`,
    `patch_shape`, `Patch.mesh`)
