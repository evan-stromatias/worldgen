# AGENTS.md

A generic procedural terrain generator, offline, from a YAML recipe and a seed: elevation, erosion,
hydrology, climate and biomes, written out as a compact terrain bundle, with a rayscreen UI to tune it.
Python 3.13, numpy, pydantic, raylib (through rayscreen for the UI). GPL-3.0-or-later.

One package, `worldgen`, built with uv. Run everything from the repository root.

## Off limits

Never run a git operation without asking the user first: not only commits and pushes, but stash, checkout,
restore, reset, apply, merge and the like. Read-only git (`status`, `diff`, `log`, `show`) is fine.

## Commands

```sh
uv sync                                                        # install
uv run pytest -q                                               # tests (no window needed)
uv run ruff check src                                          # lint
uv run worldgen --seed 7                                       # generate the default terrain, into out/seed-7
uv run worldgen --config examples/archipelago.yaml --seed 7 --out out/archipelago
uv run worldgen-ui                                             # the generator with a UI: edit, run stages, export
uv run worldgen-ui --config examples/archipelago.yaml --seed 3 --run-all
uv run python examples/generate.py                             # generate, export, load, show one patch in a window
```

The UI's `--screenshot out.png` (with `--run-all`, and `--3d`) takes scripted pictures of the window;
`--layer "land mask"` picks the map layer it settles on.

## Layout

- `src/worldgen/generator/`: the generator. `config.py` is the recipe (`TerrainConfig`: the `world`
  grid and one section per stage, validated with pydantic, YAML-path error messages); `pipeline.py` runs
  the five stages in order; `stages/` holds one module per stage (`elevation`, `erosion`, `hydrology`,
  `climate`, `biomes`), plus `noise.py` (own gradient noise, so a terrain never changes with a dependency
  upgrade), `grid.py` (array helpers), `preview.py` (the hillshaded preview and the deterministic
  `write_npz` / `write_png`), `terrain.py` (`TerrainData`, the terrain in memory, with its derived arrays
  lazy), `terrain_io.py` (`generate_terrain`, `export_terrain`, `load_terrain`: the terrain bundle),
  `patches.py` (reading a terrain in patches), `__main__.py` (the `worldgen` CLI).
  Generation runs once, offline: favour clear code over speed there.
- `src/worldgen/mesh/`: meshes and the overview (`overview.py`, `builders.py` — pure numpy, so they run on
  workers — and `gpu.py`, the raylib upload of them, main thread only). Nothing here loads saved worlds.
- `src/worldgen/viewer/`: the terrain renderer the UI's 3D view uses: `render.py` (terrain and water
  shaders, bundled in `viewer/shaders/`), `config.py` (`ViewerSettings`, `VIEWER_*` env), `camera.py`,
  `geometry.py` (camera maths, no window).
- `src/worldgen/ui/`: the generator with a UI (`worldgen-ui`). MVC: `model/` has no raylib and is tested
  without a window (`model/recipe/` is the draft recipe, its forms built from the pydantic classes —
  `model/recipe/form.py`, so a new recipe field shows up with no UI change — the biome rules, the size
  notes and the tooltips' per-field tuning hints; `model/` the log, the session running the stages on a
  worker, the export; `model/map/` the camera for 2D and 3D, the map's layers and 3D terrain, the readout
  under the mouse); `view/` is rayscreen specs (the form, dialogs, log window, tooltips, the world
  screen), only `view/gl/` makes GL calls; `controller.py` is the only code that changes the model;
  `app.py` is the window's loop. GL calls stay on the main thread.
- `assets/recipes/`: the default recipe the CLI falls back on.
- `examples/`: recipes and `generate.py` (generate, export, load, show a patch).
- `tests/`: pytest, mirroring `src/` (`generator/`, `ui/`, `viewer/`); `common.py` holds the small test
  recipe and the inline executor.

## Conventions

- **Units.** Everything is in metres (1 world unit = 1 m), seconds and degrees where noted. Distances
  counted in grid cells say so.
- **Coordinates.** Terrains run from (0, 0) at the north-west corner; x is east and z is south (north
  is -z); y is up. Grids are `[z][x]`.
- **Determinism.**
  - Each generator stage has its own RNG, `np.random.default_rng([seed, STAGE])`.
  - The same recipe and seed must give byte-identical files (fixed zip timestamps: `preview.write_npz`).
  - New random choices get a new stream, or go after existing draws, so old outputs do not change.
  - A refactor of generation code keeps outputs byte-identical: do not reorder draws, change float
    evaluation order (`a + (b - a) * i / n` rewritten can change the last bit) or swap numpy and Python math.
- **File formats.** Every terrain.yaml carries `version`: the worldgen package's semantic version,
  stamped on export (`worldgen.config.VERSION`, from the installed package metadata). worldgen never
  checks it — whether another version's file still works is the client's call — but when reading a
  bundle fails, the error names the version in the file.
- **raylib.**
  - GL calls only on the main thread; meshes are built as numpy arrays on workers (`worldgen.mesh.builders`)
    and uploaded on the main thread (`worldgen.mesh.gpu`).
  - In per-frame loops over many objects, call `raylib.*` C functions directly; pyray's wrappers
    cost tens of microseconds a call.
- **Settings and data classes.** Settings and recipes use pydantic with YAML-path or env error
  messages. Per-frame components are plain `slots` dataclasses.
- **Style.**
  - Match the surrounding code: short docstrings that say why, comments sparingly, names spelled
    out (short names only for coordinates and indices in a formula: `x`, `z`, `i`).
  - Annotate every parameter, not only public ones. Name recurring shapes (a `type` alias, a
    `NamedTuple` over an anonymous tuple); `Any` only for genuinely untyped values. Use PEP 695 `type`
    statements and generics; break import cycles with `TYPE_CHECKING`.
  - A long docstring is a summary line, then bullets. Never shorten one by cutting its reasons.
  - Split a long function into named steps only where each step has a clear name and input, not if it
    would just pass a dozen locals around. A helper used once stays only if its name explains the step.
  - Search for an existing helper before writing a private one; do not duplicate it under another module.
  - Catch the narrowest exception; every broad `except` says why in its `noqa: BLE001` comment.
  - Prefer a clear loop to a clever one-liner.
  - Lint configuration is in `pyproject.toml` (ruff); the user runs linting themselves, so do not
    run lint fixes unasked.
- **Checking work.** Run `uv run pytest -q`. For anything visual, take a screenshot (`worldgen-ui`'s
  `--screenshot`) and look at it before calling the work done.
- **Keep the README current.** Whenever a change alters how generation works, update "How a terrain is
  made" in `README.md` in the same change: the stages, their order, what each does, the recipe settings
  it names and the files it writes. Check the stage descriptions against the code. Whenever the recipe
  format changes (a key added, renamed, removed or re-defaulted), update the commented
  `assets/recipes/default.yaml` in the same change.
