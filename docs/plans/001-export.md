# Plan: export formats for clients outside Python

Status: implemented. Where the code differs from this plan:

- The raw writer is `terrain_export.write_raw(out_dir, terrain, meta)`, called by `export_terrain(…, "raw")`,
  which writes terrain.yaml and preview.png for both formats; there is no separate `export_raw`.
- `terrain_io.stale_files(format)` and `listed(files)` sit beside `bundle_files`; `listed` shows the array
  files as one `arrays/*.bin` entry in messages.
- Choosing a format updates the open Export world dialog's prompt in place (`windows.show_format`), so the
  folder typed and the listing stay as they were.
- The second-language reader is `examples/read_raw.mjs` (node), and a test runs it when node is installed.
- Every NaN in `water.bin` is already `0x7FF8000000000000` today; nothing normalises it, and the format
  does not promise that bit pattern.

worldgen exports a terrain as a **terrain bundle**: `terrain.yaml`, `terrain.npz` and `preview.png`
(`src/worldgen/generator/terrain_io.py`). It is easy to read from Python and harder from a game written
in C#, C++, Rust, Go, GDScript or JavaScript. This plan:

- describes the current bundle exactly, so it can be read from any language today;
- adds a **raw bundle** (`--format raw`): a JSON manifest, a JSON metadata file and raw little-endian
  arrays, readable with only a JSON parser and a file read;
- lists the code changes, the tests and the open questions.

The current bundle stays the default and keeps its bytes, so `load_terrain`, the UI and every existing
output stay the same.

**The raw bundle is write-only for worldgen.** Clients in other languages read it; Python code (worldgen,
the UI, `load_terrain`) does not. That keeps one loader, and no Python code ever has to turn its bytes
back into a `TerrainData` with the right dtypes. Python users keep the npz bundle.

---

## 1. Conventions shared by every format

These hold for every array in every format below.

| Topic | Rule |
| --- | --- |
| Units | Metres (1 world unit = 1 m), degrees Celsius, degrees of slope. |
| Axes | x runs east, z runs south (north is -z), y is up. (0, 0) is the north-west corner. |
| Memory layout | Row-major `[z][x]`: row `z` is one west-to-east line, and the rows go from north to south. The index of sample (x, z) is `z * width + x`. |
| Byte order | Little-endian, for every multi-byte type. |
| Vertex grid | `(size_cells + 1)²` samples. Sample (x, z) sits at world point `(x * cell_size, z * cell_size)`. |
| Cell grid | `size_cells²` samples. Cell (x, z) covers the square from `(x, z) * cell_size` to `(x + 1, z + 1) * cell_size`. Its value is for the whole cell, and where it comes from the vertex grid it is the mean of the cell's four corners (`grid.to_cells`). |
| Polyline points | `(x, z)` pairs of **vertex** indices, not metres. Multiply by `cell_size` to get metres. |
| Height | Absolute metres. Sea level is `world.sea_level` (default 0), so "height above sea" is `elevation - sea_level`. |
| Determinism | The same recipe and seed give byte-identical files in every format: fixed file order, fixed zip timestamps, sorted JSON keys, no wall-clock values. |

`size_cells` is any multiple of `world.patch_cells` (at least 16), not necessarily a power of two.

### Sampling a vertex grid between samples

Clients that want the same heights as worldgen's own meshes should interpolate the same way
(`worldgen.mesh.overview.sample_grid`). Split each cell into two triangles along the diagonal from
(x+1, z) to (x, z+1):

```
gx = clamp(x_m / cell_size, 0, width - 1.001);   ix = floor(gx);  fx = gx - ix
gz = clamp(z_m / cell_size, 0, height - 1.001);  iz = floor(gz);  fz = gz - iz
h00 = g[iz][ix]     h10 = g[iz][ix+1]
h01 = g[iz+1][ix]   h11 = g[iz+1][ix+1]
if fx + fz <= 1:  h = h00 + (h10 - h00) * fx + (h01 - h00) * fz
else:             h = h11 + (h01 - h11) * (1 - fx) + (h10 - h11) * (1 - fz)
```

The meshes are triangulated along the same diagonal, so a client that meshes the grid should use it too,
or objects placed on the ground will float or sink by a few centimetres on steep cells.

---

## 2. The stored arrays

The same eleven arrays go into every format (`terrain_io.ARRAYS`). The dtypes are the ones the
generator writes today. The examples use a 128-cell world (`size_cells: 128`).

| Name | Grid | dtype (numpy / raw) | Shape | Meaning |
| --- | --- | --- | --- | --- |
| `elevation` | vertex | `<f8` / `float64` | 129 × 129 | Ground height, metres, after erosion and river carving. |
| `water` | vertex | `<f8` / `float64` | 129 × 129 | Water surface height, metres. **NaN where dry.** Sea vertices hold `sea_level`, lake vertices their lake's `level`, and river vertices a surface that falls along the flow. |
| `sea` | vertex | `\|b1` / `uint8` | 129 × 129 | 1 = open sea (water connected to the world edge), 0 = not. One byte per vertex in both formats. |
| `lake_id` | vertex | `<u2` / `uint16` | 129 × 129 | 0 = no lake. Otherwise the `id` of an entry in `lakes` (section 3). |
| `river` | vertex | `\|u1` / `uint8` | 129 × 129 | River strength: 0 = no river, 1-255 = river channel, from weakest to strongest. Strength is the log of the upstream rainfall, scaled between `hydrology.river_threshold` and the largest river. |
| `flow_dir` | vertex | `\|u1` / `uint8` | 129 × 129 | The D8 direction water flows to from each vertex (table below). 255 = no flow (sea, world edge, sinks). |
| `biome` | cell | `\|u1` / `uint8` | 128 × 128 | Index into the recipe's `biomes.rules` list, in recipe order. The first matching rule wins. |
| `color` | cell | `\|u1` / `uint8` | 128 × 128 × 3 | Ground colour, sRGB, 0-255, with the per-cell variation already applied. The channels are interleaved: `R G B R G B …` along each row. |
| `temperature` | cell | `<f4` / `float32` | 128 × 128 | °C, as the biome rules saw it. |
| `moisture` | cell | `<f4` / `float32` | 128 × 128 | 0-1. |
| `sediment` | cell | `<f4` / `float32` | 128 × 128 | Metres of sediment deposited by erosion (0 everywhere when erosion is off). |

### D8 flow codes

`flow_dir` gives the neighbour that water flows to (`stages.hydrology.D8`):

| Code | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 255 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direction | E | SE | S | SW | W | NW | N | NE | none |
| (dx, dz) | (1, 0) | (1, 1) | (0, 1) | (-1, 1) | (-1, 0) | (-1, -1) | (0, -1) | (1, -1) | — |

Remember that z grows southwards, so "S" is +z.

### Derived arrays

Python recomputes these on load, so the npz bundle leaves them out. Porting them is harder than it looks
(`slope` follows `np.gradient`'s one-sided differences at the world edge, `biome_weights` is all of
`biomes.soft_weights`), so **the raw bundle always includes them** (section 4.2).

| Name | Grid | dtype | Shape | How it is made |
| --- | --- | --- | --- | --- |
| `slope` | cell | `float32` | 128 × 128 | Degrees. `atan(|∇elevation|)` per vertex, using `np.gradient` with spacing `cell_size` (central differences inside, one-sided at the edges), then averaged to cells. |
| `relief` | cell | `float32` | 128 × 128 | Metres. The cell's height minus the lowest cell within `biomes.relief_radius` metres (a square window). |
| `biome_weights` | cell | `float32` | rules × 128 × 128 | Each rule's soft share of the cell. The shares sum to 1 (`biomes.soft_weights`). Useful for blending ground textures. |

---

## 3. Lakes and rivers metadata

This is the same data in every format. Today it is the JSON string inside `terrain.npz`'s `meta` entry.
In the raw bundle it becomes `meta.json`.

```json
{
  "lakes": [
    {"id": 1, "level": 30.53807490840879, "bbox": [58, 32, 74, 45], "outlet": [59, 32], "area": 106}
  ],
  "rivers": [
    [[76, 81], [77, 80], [78, 79]]
  ]
}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `lakes[].id` | int ≥ 1 | Matches the values in `lake_id`. |
| `lakes[].level` | float, metres | The lake's flat water surface: the height at which its depression spills over. |
| `lakes[].bbox` | `[x0, z0, x1, z1]` | Inclusive bounds, in vertex indices. |
| `lakes[].outlet` | `[x, z]` or `null` | The vertex the lake drains into. `null` for a lake with no outlet. |
| `lakes[].area` | int | The number of vertices under the lake. |
| `rivers[]` | list of `[x, z]` | One polyline per river, in vertex indices, **source first** (it runs downstream). A polyline ends at the first lake or sea vertex it reaches, at a sink or the world edge, or where it **joins another river**: its last point is then a point of that river's polyline (`hydrology._trace_rivers`). Following tributaries to their junctions gives the river network. |

---

## 4. Formats

### 4.1 `npz`: the current bundle (default, unchanged)

```
<out>/
  terrain.yaml    # the seed, the worldgen version and the full recipe (and `client`, when not empty)
  terrain.npz     # zip of one .npy per array in section 2, plus meta.npy
  preview.png     # 8-bit RGB hillshaded map, one pixel per cell
```

**terrain.yaml.** This is YAML: a header comment, then `seed`, `version`, then one section per recipe
stage (`world`, `elevation`, `erosion`, `hydrology`, `climate`, `biomes`) and `client` when it is not
empty. `world.size_cells`, `world.cell_size`, `world.sea_level` and `world.patch_cells` are what a
reader needs for the grids. `biomes.rules[i].id` names biome index `i`.

**terrain.npz.** This is a standard zip with Deflate compression. Its entries are `<name>.npy`, sorted by
name, all timestamped 1980-01-01 00:00:00. Each `.npy` (format version 1.0) is laid out as:

```
offset 0   6 bytes   magic "\x93NUMPY"
offset 6   1 byte    major version (1)
offset 7   1 byte    minor version (0)
offset 8   2 bytes   header length N, uint16 little-endian
offset 10  N bytes   ASCII Python dict literal, padded with spaces, ending in "\n", e.g.
                     {'descr': '<f8', 'fortran_order': False, 'shape': (129, 129), }
offset 10+N          raw array data, C order, in the byte order `descr` names
```

`descr` codes: `<f8` = float64, `<f4` = float32, `<u2` = uint16, `|u1` = uint8, `|b1` = bool (one byte,
0 or 1). `fortran_order` is always `False`.

**The `meta` entry is a 0-d unicode array** (`descr` `<U6541` or similar): UTF-32LE code units, padded with
NUL characters to the stated length. Decode it as UTF-32LE, strip the trailing NULs and parse the result
as JSON (section 3). This is the part most non-Python `.npy` readers cannot handle, and one reason for the
raw bundle.

Most languages have a `.npy` reader library, but none is needed: parsing the header above is about 30
lines of code in any language.

### 4.2 `raw`: the language-neutral bundle (new)

```
<out>/
  terrain.json      # manifest: everything a reader needs, in JSON
  terrain.yaml      # the full recipe, as in the npz bundle (for humans and for re-running worldgen)
  meta.json         # lakes and rivers (section 3)
  arrays/
    elevation.bin   # one raw little-endian file per array, no header
    water.bin
    sea.bin
    lake_id.bin
    river.bin
    flow_dir.bin
    biome.bin
    color.bin
    temperature.bin
    moisture.bin
    sediment.bin
    slope.bin       # the derived arrays (section 2), always included
    relief.bin
    biome_weights.bin
  preview.png
```

Every `.bin` file is just the array's bytes, row-major `[z][x]` and little-endian, with nothing before or
after them. A file's size is always `product(shape) * size_of(dtype)`, and a reader should check that.
Booleans (`sea`) are written as `uint8` 0/1.

**terrain.json** is written with `json.dumps(…, indent=2, sort_keys=True)`, UTF-8, with a final newline.
An excerpt, exactly as written (most arrays and biomes left out):

```json
{
  "arrays": {
    "elevation": {
      "dtype": "float64",
      "file": "arrays/elevation.bin",
      "grid": "vertex",
      "shape": [
        129,
        129
      ],
      "units": "m"
    },
    "flow_dir": {
      "dtype": "uint8",
      "file": "arrays/flow_dir.bin",
      "grid": "vertex",
      "nodata": 255,
      "shape": [
        129,
        129
      ]
    },
    "water": {
      "dtype": "float64",
      "file": "arrays/water.bin",
      "grid": "vertex",
      "nan": "dry",
      "shape": [
        129,
        129
      ],
      "units": "m"
    }
  },
  "biomes": [
    {
      "color": [
        150,
        140,
        105
      ],
      "id": "sea_floor"
    }
  ],
  "byte_order": "little",
  "client": {},
  "d8": [
    [
      1,
      0
    ]
  ],
  "format": "worldgen-raw",
  "format_version": 1,
  "meta": "meta.json",
  "seed": 3,
  "version": "0.2.0",
  "world": {
    "cell_size": 1.0,
    "patch_cells": 32,
    "sea_level": 0.0,
    "size_cells": 128
  }
}
```

The manifest's fields:

| Field | Meaning |
| --- | --- |
| `format`, `format_version` | Identify this layout. `format_version` changes only when the raw layout itself changes (a field renamed, a dtype changed), independently of worldgen's own `version`. Clients should refuse a `format_version` they do not know. |
| `version` | The worldgen version, stamped on export as for terrain.yaml. As with the npz bundle, worldgen never checks it. |
| `seed`, `world` | The seed, and the `world` section of the recipe. |
| `client` | The recipe's `client` section (the game's own data), verbatim; `{}` when the recipe has none. With it, a client needs no YAML parser at all. |
| `biomes` | `biomes.rules` in order: entry `i` is biome index `i` in `biome` and layer `i` of `biome_weights`, with its `id` and `ground.color`. |
| `d8` | `(dx, dz)` for each D8 code, so clients need not hard-code the table. |
| `meta` | The path of the lakes and rivers file (section 3). |
| `arrays.*.dtype` | One of `uint8`, `uint16`, `float32`, `float64`. |
| `arrays.*.grid` | `vertex` or `cell` (section 1). |
| `arrays.*.shape` | `[rows (z), cols (x)]`, plus `3` for `color`; `[rules, rows, cols]` for `biome_weights`. |
| `arrays.*.nodata` | Integer arrays only: the value that means "none" (`0` for `lake_id` and `river`, `255` for `flow_dir`). Always a number. |
| `arrays.*.nan` | Float arrays only: what NaN means, when it can occur (`"dry"` for `water`). |
| `arrays.*.units`, `arrays.*.range` | Documentation: `units` is one of `m`, `degC`, `deg`, `srgb`; `range` is `[min, max]` where the values are bounded (`moisture`, `biome_weights`: `[0, 1]`). |

Every key is always present where the table says it applies, so a typed client (C# records, Rust
`serde` structs, Go structs) can model the manifest with one type per object and optional fields.

**Reading it, in pseudo-code, any language:**

```
manifest = json.parse(read("terrain.json"))
assert manifest.format == "worldgen-raw" and manifest.format_version == 1
a = manifest.arrays.elevation
bytes = read(a.file)
assert len(bytes) == product(a.shape) * sizeof(a.dtype)
heights = reinterpret_as<float64, little-endian>(bytes)   // heights[z * a.shape[1] + x]
```

In C# that is `MemoryMarshal.Cast<byte, double>`, in Rust `bytemuck::cast_slice` or `f64::from_le_bytes`,
in JS `new Float64Array(buffer)`, in Godot `PackedByteArray.to_float64_array()`. All of these platforms
are little-endian, so no byte swapping is needed.

**Why no compression?** Raw files can be memory-mapped and read with no library at all. A 1024-cell
world is about 8.4 MB for each float64 vertex grid and 38.8 MB for the eleven stored arrays (21 bytes per
vertex and 16 per cell); the derived arrays add 8 bytes per cell plus 4 per cell per biome rule. The npz
bundle of the default recipe compresses the stored arrays to about 14 MB. Anyone who needs a smaller
download can zip the folder.

### 4.3 Later, if a client needs it

Not in this change. Each is additive, and would bump `format_version` where it changes the raw layout.

| Flag | Would add |
| --- | --- |
| `--compress` | gzip each `.bin` as `<name>.bin.gz` (`mtime=0`), with `"compression": "gzip"` in its manifest entry. |
| `--float32` | `elevation` and `water` as float32: half the size, about 0.1 mm precision at 1000 m. |
| `--patches` | `patches/<px>_<pz>/…` with each patch's slices, laid out as `Patch` slices them (heightmap with a one-vertex apron), for clients that stream worlds too large to load whole. |

---

## 5. Code changes

1. **`generator/terrain_export.py`** (new): `export_raw(terrain, out_dir)`. It reuses
   `terrain_io.ARRAYS`, `RECIPE_FILE`/`HEADER` and the meta dict built by `export_terrain` (move that into
   a small `bundle_meta(terrain)` helper in `terrain_io.py`, so both formats share it). The derived arrays come
   from `TerrainData`'s own properties, so there is no second implementation of them.
2. **`generator/terrain_io.py`**: `type ExportFormat = Literal["npz", "raw"]`, shared by the CLI and the UI,
   and `export_terrain(terrain, out_dir, format: ExportFormat = "npz")`, which dispatches to the new
   writer. The npz path stays byte-identical. A `bundle_files(format)` helper names the files each format
   writes, for the stale-file removal and for the UI's messages.
   - **Stale files.** Before writing, the export removes the files that only the *other* format writes,
     by name and nothing else (`terrain.npz`; `terrain.json`, `meta.json` and the known `arrays/*.bin`,
     then `arrays/` if empty). So a folder never holds two formats.
   - **`load_terrain` does not read raw bundles.** Given a folder with `terrain.json` and no
     `terrain.npz`, it fails with a clear `ConfigError`: "<dir> is a raw export, for clients in other
     languages; worldgen reads only npz bundles (export with --format npz)".
3. **`generator/__main__.py`**: `--format npz|raw`.
4. **UI**: a format choice in the Export world dialog (section 5.1).
5. **`examples/`**: a tiny reader in a second language (for example `examples/read_raw.js` or a
   C# snippet), to prove the format works without numpy.
6. **Docs**:
   - `README.md`: replace "What's in terrain.npz" with a "Terrain bundle formats" section, condensed from
     sections 1-4 of this file (including the note that `sea` is one byte, 0 or 1).
   - `README.md` "How a terrain is made": update the list of files the export writes, as AGENTS.md asks.
   - `CHANGELOG.md`: an entry.
   - The recipe format does not change, so `assets/recipes/default.yaml` needs no update.

### 5.1 The UI

Export world (the `save-world` button) opens `windows.save_world_window`, a rayscreen `FileWindow` in
directory mode. `FileWindow` takes `extra` widgets, placed under its Folder row, so the format choice goes
there, and the rest of the export (running stale stages first, the overwrite question, exporting on the
`map-and-save` worker) stays as it is.

```
┌ Export the world ─────────────────────────────────────┐
│ A folder for the terrain bundle: terrain.yaml,        │
│ terrain.npz and preview.png. …            (prompt)    │
│ /home/me/worlds                                       │
│ [folder list]                                         │
│ Folder  [ island                                    ] │
│ Format  [ npz: for Python (worldgen, numpy)       ▾ ] │
│ [ ] Hidden   /home/me/worlds/island                   │
│                                      [Cancel] [Export]│
└───────────────────────────────────────────────────────┘
```

- **`ui/view/windows.py`**:
  - `SAVE_WORLD_FORMAT = f"{SAVE_WORLD}.format"`.
  - `save_world_window(path, format)` adds `extra=[Row(Label("Format"), Dropdown(SAVE_WORLD_FORMAT, …))]`.
    The two choices are "npz: for Python (worldgen, numpy)" and "raw: for other languages (C#, C++, Rust,
    JS…)".
  - The prompt lists the files the chosen format writes (from `bundle_files`), so switching the choice
    shows what will land in the folder.
- **`ui/view/events.py`**: a `FormatChosen(format: ExportFormat)` event. `_translated` maps
  `SAVE_WORLD_FORMAT` to it, as it maps `layer` to `LayerChosen`. It goes in the `Event` union.
- **`ui/controller.py`**, the only code that changes the model:
  - `self.export_format: ExportFormat = "npz"`, set by `FormatChosen`. The dialog is rebuilt from it (the
    `SAVE_WORLD` entry of the window factories passes it), so its prompt follows the choice.
  - `_start_saving` passes `self.export_format` to `save_terrain`.
  - `self._saved` becomes the folder *and* the format last exported. Today a second export to the same
    folder skips the overwrite question; with two formats, exporting raw into the folder just exported as
    npz removes `terrain.npz`, so that case asks first.
  - The overwrite question names what is there and what replaces it: "… already holds a terrain (npz).
    Write a raw export over it? terrain.npz is removed."
  - The "exported the terrain to …" message lists the files of the format written (from
    `bundle_files`), not the npz list it hard-codes today.
- **`ui/model/saving.py`**:
  - `save_terrain(terrain, recipe_text, out, format)` passes `format` to `export_terrain`. It still
    rewrites `terrain.yaml` with the UI's recipe text in both formats. `terrain.json` is built from the
    parsed recipe, so the two agree.
  - `terrain_format(folder) -> ExportFormat | None` replaces `is_terrain`: `"npz"` with `terrain.npz`,
    `"raw"` with `terrain.json`, `None` otherwise. The controller uses it for the overwrite question.
- **`ui/__main__.py`**: `--format npz|raw` sets the dialog's starting choice, so
  `worldgen-ui --out … --format raw` exports raw without touching the dropdown.
- **Not in the UI:** it never opens a raw export. Open still reads recipes only, and nothing in the UI
  loads a bundle.
- **The recipe is not involved.** The format is a property of the export, not of the terrain, so it is not
  a recipe field and does not change the form or `assets/recipes/default.yaml`.

## 6. Tests

### Generator (`tests/generator/`)

No raw loader exists, so the tests read the raw bundle the way a client in another language would:
through the manifest.

- **Manifest-driven read:** generate the small test terrain and `export_raw` it. For each manifest entry,
  read its file with `np.fromfile(path, dtype=np.dtype(entry["dtype"]).newbyteorder("<"))`, check its
  size is `product(shape) * itemsize` and compare it with the terrain's array (cast to the manifest's
  dtype; `equal_nan=True` for `water`). This checks the dtypes, shapes and paths a client relies on, not
  just a Python round trip. The derived arrays are compared with `terrain.slope`, `terrain.relief` and
  `terrain.biome_weights`.
- **meta.json:** equals the npz bundle's meta, lakes and rivers alike.
- **No numpy needed:** read `elevation.bin` with `struct`/`array` alone and compare it with the terrain.
- **Manifest:** the keys are sorted; the `biomes` order and colours match `config.biomes.rules`;
  `client` matches the recipe's; `nodata` is a number on every integer array that has one, and `water`
  has `"nan": "dry"`.
- **Determinism:** exporting twice gives byte-identical directories.
- **npz unchanged:** the npz bundle's bytes match those written before the change (the existing
  determinism test covers this).
- **Stale files:** exporting raw over an npz bundle (and the reverse) leaves only the new format's files.
- **load_terrain on a raw export:** raises `ConfigError` naming the raw format.

### UI (`tests/ui/`, no window, with the inline executor from `common.py`)

- **Event:** a `SAVE_WORLD_FORMAT` dropdown event translates to `FormatChosen("raw")`.
- **Export:** a controller with `export_format = "raw"` exports the small test terrain; the folder holds
  `terrain.json`, `arrays/` and the UI's recipe text in `terrain.yaml`, and no `terrain.npz`.
- **Overwrite question:** exporting raw into the folder just exported as npz asks before writing; exporting
  the same format again does not.
- **Messages:** the dialog's prompt and the "exported" message list the chosen format's files.
- **`terrain_format`:** `"npz"`, `"raw"` and `None` for the three kinds of folder.
- **Visual check:** a screenshot of the Export world dialog with each choice, looked at before calling it
  done (AGENTS.md). `--screenshot` cannot open dialogs today. A `--export-dialog` flag, opening Export world
  at the start as `--log` opens the log window, makes it scriptable:
  `uv run worldgen-ui --run-all --export-dialog --format raw --screenshot out/export-raw.png`.

## 7. Decided

- **Raw is write-only for worldgen.** Python clients use the npz bundle; no `load_terrain` support for raw.
- **float64 for `elevation` and `water`**, matching the npz bundle; `--float32` deferred (section 4.3).
- **The recipe's `client` section goes into terrain.json**, verbatim, so clients need no YAML parser.
  terrain.yaml stays for humans and for re-running worldgen.
- **The derived arrays are always in the raw bundle**, not behind a flag.
- **No compression, no patches** in this change (section 4.3).
- **No engine heightmap** (`.r16` / 16-bit PNG). Clients build their terrain from `elevation.bin`.

## 8. Open questions

- **Endianness:** big-endian hosts are not a target. The manifest says `little`, so one could be added
  later.
