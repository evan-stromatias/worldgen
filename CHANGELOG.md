2026-11-16, 0.3.0:
  - raw export format
  - **Raw terrain bundle for clients in other languages** (C#, C++, Rust, JS…): `worldgen --format raw`,
  - `terrain.json`: a manifest listing each array's file, dtype, shape, grid (vertex or cell), units,
  - `meta.json`: the lakes and the river polylines, the same as the npz bundle's `meta`.
  - `arrays/*.bin`: one file per array, with no header, little-endian and row-major `[z][x]`. The
  - `terrain.yaml` and `preview.png`, as in the npz bundle.
  - The raw export is deterministic: the same recipe and seed give byte-identical files.
  - **Export world in `worldgen-ui`** has a **Format** choice (npz for Python, raw for other languages).
  - `worldgen-ui --format raw` picks the export format, and `--export-dialog` opens Export world after
  - `examples/read_raw.mjs`: reads a raw bundle in plain node, with no libraries.
  - **Exporting one format into a folder removes the other format's files**, so a folder never holds both.
  - `worldgen-ui` asks before writing a different format over an exported folder, and names the files
  - npz is still the default, and its files are byte-for-byte unchanged.
  - worldgen never reads raw bundles back. `load_terrain` refuses one, saying to export with `--format npz`.

2026-11-10, 0.2.0:
  - clients get their own ignored field

2026-11-10, 0.1.3:
  - more info in the tooltips

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
