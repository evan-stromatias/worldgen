# worldgen

Generic procedural terrain generation, offline, from a YAML recipe and a seed: elevation, erosion,
hydrology, climate and biomes, written out as a compact terrain bundle, with a UI to tune it.

Everything is in metres. Terrains start at (0, 0) in the north-west corner; x runs east, z runs south
and y points up. The same recipe and seed always give the same terrain, byte for byte.

## How a terrain is made

Five stages run in order, each reading what the stages before it made. Each draws its random numbers
from its own stream, so tuning one stage never reshuffles another's.

1. **Elevation** — the base heightfield. Fractal (fBm) gradient noise, sampled through a domain warp so
   ridges and coasts bend, shaped by a land mask: one island, an archipelago, or a continent that runs
   out to the edges (or sinks them into the sea when the recipe sets `falloff`). A mask is not an image
   to draw or upload: an island is a radial falloff into the sea with a noise-bent outline, tuned by
   recipe keys (`mask_falloff`, `mask_distortion`, the archipelago's island count, size and spacing), so
   it lives in the recipe — stored, loaded and edited with it, in the UI as plain form fields. Optional
   mountain ranges rise where a slow mask puts them (ridged multifractal noise, after Musgrave's
   [*Texturing and Modeling: A Procedural Approach*](https://www.csee.umbc.edu/~ebert/book/book.html),
   with sharp crests), and optional terraces gather heights into strata. The noise is implemented here
   rather than taken from a library, so a terrain never changes with a dependency upgrade.
2. **Erosion** — weathering, when the recipe turns it on. Simulated droplets cut gullies and valleys
   and leave sediment where they slow, following [Beyer 2015, *"Implementation of a method for
   hydraulic erosion"*](docs/implementation%20of%20a%20methode%20for%20hydraulic%20erosion.pdf),
   batched for numpy and run from coarse grids to the full one so the valleys are mountain-sized, not
   rills. Then loose ground slides until no slope is steeper than it holds (talus
   and rock angles), after [Musgrave, Kolb and Mace 1989, *"The synthesis and rendering of eroded
   fractal terrains"*](https://doi.org/10.1145/74333.74337).
3. **Hydrology** — lakes and rivers. Depressions fill to their spill height (priority-flood,
   [Barnes et al. 2014](https://doi.org/10.1016/j.cageo.2013.04.024)) and the deep and wide ones
   become lakes; rain falls (less in the lee of ridges), flows downhill over D8 with a meander noise so
   rivers wiggle and gather, and where enough water gathers a river carves its channel and banks.
4. **Climate** — temperature (base, latitude, lapse rate over height, plus noise), moisture (rain and
   water nearby, plus noise), and slope.
5. **Biomes** — each cell's biome from rules tried in order over elevation, temperature, moisture,
   slope, sediment, relief and water (`sea`/`lake`/`river`/none); the last rule is the catch-all.
   Range edges soften into ramps so neighbouring biomes blend their ground colours, and a per-cell
   jitter keeps the ground from reading flat.

## Install

```sh
uv sync
```

## Generate a world

With the CLI, into `out/` by default:

```sh
uv run worldgen --seed 7                                     # the bundled default recipe, into out/seed-7
uv run worldgen --config examples/archipelago.yaml --seed 42 --out out/archipelago
uv run worldgen --seed 7 --preview-only --out .              # just preview.png
```

With the UI, to tune a recipe stage by stage and watch each one land:

```sh
uv run worldgen-ui                                       # from the defaults
uv run worldgen-ui --config examples/archipelago.yaml --seed 3 --run-all
```

The recipe as forms on the right, the terrain on the left as it stands after each stage, in 2D or 3D. A
layer dropdown switches the map between the heightfield, the land, mountain and terrace masks, erosion's
changes and sediment, the water, the climate and the biomes; hovering a cell reads out what lies under
it. **Export world...** writes the terrain bundle. Opening a bundle's `terrain.yaml` as a recipe also
brings its seed.

![worldgen-ui, 2D: the default island after all five stages](docs/screenshots/ui-2d.png)

![worldgen-ui, 3D: the same island as a terrain mesh](docs/screenshots/ui-3d.png)

From Python:

```python
from worldgen import TerrainConfig, generate_terrain, export_terrain

config = TerrainConfig()  # the defaults; or load_config("examples/archipelago.yaml")
terrain = generate_terrain(config, seed=7)   # stages 1-5 in memory
export_terrain(terrain, "out/island")        # write the bundle
```

Either way the result is a self-contained **terrain bundle**:

```
out/island/
  terrain.yaml   # the recipe and the seed
  terrain.npz    # the arrays, compressed deterministically
  preview.png    # shaded biome & water preview
```

## Load and use a world

`TerrainData` is the class for a bundle on disk:

```python
from worldgen import TerrainData

terrain = TerrainData.load("out/island")

terrain.elevation      # (N+1)² vertex grid, metres
terrain.water          # (N+1)² water surface, NaN = dry
terrain.biome          # N² cell grid, index into terrain.config.biomes.rules
terrain.color          # N² × 3 uint8 ground colours
terrain.slope          # derived, computed on first use: degrees per cell
terrain.relief         # derived: metres above the lowest ground within biomes.relief_radius
terrain.biome_weights  # derived: (rules, N, N) shares of each biome per cell
terrain.lakes          # the lake records: level, bounds, outlet, area
terrain.rivers         # the river polylines, vertex coordinates, source first

# What lies at a world point (x, z), in metres:
terrain.height_at(x, z)  # ground height, interpolated as the meshes interpolate it
terrain.water_at(x, z)   # water surface, NaN where dry
terrain.biome_at(x, z)   # the cell's biome id
```

### Patches

For a game that loads the world a piece at a time, read the terrain in patches of
`world.patch_cells` cells. There are no patch files on disk: the bundle stores one heightmap for the
whole world (`elevation` in `terrain.npz`), and a patch is an exact slice of it, cut on demand — the
heightmap with a one-vertex apron (so meshed patches meet without a seam), the water grids, the
per-cell grids, and the lakes and rivers that reach it:

```python
terrain = TerrainData.load("out/island")
nx, nz = terrain.patch_shape            # how many patches along x and z
patch = terrain.patch_at(500.0, 300.0)  # the patch under this point (metres); or terrain.patch((px, pz))
patch.elevation     # (P+3)² with apron; patch.vertex_height is the (P+1)² heightmap
patch.water         # (P+1)², NaN = dry — plus sea, lake_id, river, flow_dir
patch.biome         # P² — plus color, temperature, moisture, sediment
patch.lakes         # the lake records whose bounds reach the patch
patch.rivers        # the river polylines (world vertex coordinates) passing through it
mesh = patch.mesh() # terrain and water meshes, ready for worldgen.world.gpu's GpuPatch
```

`examples/generate.py` shows the whole flow: generate, export, load, take the patch under a point and
draw it in a window.

### What's in terrain.npz

Whole-terrain arrays, not patch files — patches are exact slices of these, cut on demand. For a world
of `N` cells per side (`world.size_cells`):

| Entry | Shape | Type | Meaning |
|---|---|---|---|
| `elevation` | (N+1)² | float64 | ground height, metres — the heightmap |
| `water` | (N+1)² | float64 | water surface height, `NaN` where dry |
| `sea` | (N+1)² | bool | open sea |
| `lake_id` | (N+1)² | uint16 | lake number, 0 = none |
| `river` | (N+1)² | uint8 | river strength, 0 = none |
| `flow_dir` | (N+1)² | uint8 | D8 flow direction code |
| `biome` | N² | uint8 | biome index per cell, into `biomes.rules` in terrain.yaml |
| `color` | N²×3 | uint8 | ground colour per cell |
| `temperature` | N² | float32 | °C |
| `moisture` | N² | float32 | 0–1 |
| `sediment` | N² | float32 | metres of sediment left by erosion |
| `meta` | — | JSON | lakes (id, level, bbox, outlet, area) and rivers (polylines of vertex coordinates) |

Vertex grids are (N+1)² because they store the corners of the N² cells. `slope`, `relief` and
`biome_weights` are not stored: they are pure functions of the stored arrays, recomputed exactly on
first use.

A `.npz` is a zip of `.npy` files, so plain numpy reads it, no worldgen needed:

```python
import json
import numpy as np

with np.load("out/island/terrain.npz") as data:
    elevation = data["elevation"]
    meta = json.loads(str(data["meta"]))  # {"lakes": [...], "rivers": [...]}
```

## Recipes

A recipe is YAML: the `world` grid (size, patch and cell size, sea level) and one section per stage.
Every key takes its default when left out, so a recipe says only what it changes. A recipe with every
key at its default, commented:

```yaml
# A terrain recipe, every key at its default. Leave a key out and it takes the value shown here.
version: 1              # recipe format; this generator reads only its own
world:
  cell_size: 1.0        # metres per cell
  patch_cells: 128      # cells per patch side, when the terrain is read in patches
  size_cells: 1024      # cells per side; a multiple of patch_cells
  sea_level: 0.0        # height of the sea surface, metres
elevation:
  mask: island          # island | archipelago | continent
  mask_falloff: [0.55, 0.95]  # island radii where land starts and stops falling into the sea
  mask_distortion: 0.15       # how much noise bends the island outline (0-1)
  falloff: 0.0          # continent only: share of the world width the coast lies within; 0 = land to the edges
  shore: 0.08           # shaped noise value (0-1) at which land meets the sea
  max_height: 60.0      # metres above sea level at the highest point
  sea_floor: -12.0      # height of the deepest open sea, metres
  noise:                # the hills' fractal (fBm) noise
    octaves: 6          # layers of noise summed
    frequency: 0.003    # cycles per cell of the first octave
    lacunarity: 2.0     # frequency multiplier from one octave to the next
    gain: 0.5           # amplitude multiplier from one octave to the next
  warp:                 # domain warp: noise pushes where the hills are sampled, so ridges and coasts bend
    strength: 80.0      # how far sample points are pushed, cells
    frequency: 0.002    # cycles per cell of the warp noise
  archipelago:          # used when mask: archipelago
    islands: 7          # islands to place; fewer if they do not fit
    size: [0.1, 0.3]    # island radius range, as a fraction of the world width
    spacing: 0.8        # minimum gap between island centres, in units of their summed outer radii
  mountains:            # ridged-noise ranges, risen only where a slow mask puts them
    enabled: false
    wavelength: 400.0   # spacing of the ridges, metres
    octaves: 6
    gain: 0.5
    sharpness: 2.0      # 1 is a crease, 2 a knife edge
    height: 0.6         # share of max_height the ranges rise above the hills
    coverage: 0.35      # share of the world in ranges
    range_wavelength: 1500.0  # size of the range mask's blobs, metres
    warp: true          # sample the ridges through the hills' warp, so ranges bend with them
  terraces:             # heights gathered into strata: plateaus and mesas with cliff edges
    enabled: false
    step: 12.0          # height of one stratum, metres
    sharpness: 0.7      # 0 smooth slopes, 1 nearly vertical step faces
    where: noise        # all | mountains (the range mask) | noise (a mask of its own)
    wavelength: 800.0   # size of the terrace mask's blobs (where: noise), metres
    coverage: 0.3       # share of the world terraced (where: noise)
erosion:                # droplet erosion, then thermal sliding; off by default
  enabled: false
  droplets: 1.0         # droplets per vertex of each level's grid
  batch: 8192           # droplets that step together; they see each other's changes a step late
  levels: 3             # grids eroded, coarsest first: 3 is every 8th, 4th and 1st vertex
  lifetime: 40          # steps a droplet takes at most
  inertia: 0.1          # how much a droplet keeps its direction rather than turn downhill
  capacity: 1.0         # sediment a droplet can carry, per metre it drops and unit of speed and water
  min_slope: 0.01       # least slope capacity is worked out with, so flats still carry
  erode_rate: 0.3       # share of its spare capacity a droplet erodes each step
  deposit_rate: 0.3     # share of its excess sediment a droplet drops each step
  evaporation: 0.02     # share of its water a droplet loses each step
  gravity: 0.1          # how fast a droplet speeds up going down, per unit of slope
  radius: 2.0           # radius of the brush a droplet erodes with, in vertices of its level's grid
  smoothing: 1          # vertices each level's change is blurred over before it is added; 0 is none
  talus_angle: 35.0     # steepest slope loose ground holds, degrees
  rock_talus_angle: 70.0    # steepest slope inside the terrace mask, degrees, so cliffs survive
  thermal_iterations: 40    # passes of sliding; 0 skips it
  thermal_rate: 0.5     # share of the excess height a pass moves downhill
hydrology:
  epsilon: 0.0001       # priority-flood step, so flats still drain
  lake_min_depth: 0.5   # shallower depressions are filled in, metres
  lake_min_area: 40     # smaller depressions are filled in, cells
  river_threshold: 1500.0   # accumulated rainfall (in cells' worth) at which a river starts
  carve_depth: 2.5      # k in channel depth = k * strength^0.5, metres
  bank_falloff: 3       # cells over which banks slope down to a river
  river_min_width: 1.5  # width of a river where it starts, metres
  river_width_scale: 4.0    # width the strongest river adds, metres; width grows with strength
  rain:
    base: 1.0           # rainfall per cell before noise
    noise: 0.5          # fraction of the base that noise adds or removes
    frequency: 0.001    # cycles per cell of the rain noise
  rain_shadow:          # less rain downwind of high ground
    enabled: true
    wind: [1.0, 0.0]    # prevailing wind direction (x, z)
    strength: 0.6       # fraction of rain lost behind the tallest ridges
  meander:              # bends in rivers
    strength: 2.0       # how far rivers stray from straight downhill; 0 is plain steepest descent
    frequency: 0.03     # cycles per cell of the noise that steers the bends
climate:
  base_temperature: 18.0    # °C at sea level on the south edge
  latitude_gradient: 12.0   # °C colder on the north edge than the south edge
  lapse_rate: 300.0         # °C colder per 1000 m of height; exaggerated so low peaks still get snow
  temperature_noise: 3.0    # °C
  moisture:                 # rain, and water nearby
    rain_weight: 0.6        # share of moisture from rain; the rest is water nearby
    water_distance: 60.0    # how far water nearby keeps land moist, cells
    noise: 0.15             # amplitude of moisture noise
biomes:
  blend_width: 0.08     # fraction of each range over which neighbouring biomes blend colour
  relief_radius: 60.0   # metres around a cell its relief looks for the lowest ground in
  rules:                # tried in order, the first match wins; the last has no `when` and catches all
  - id: sea_floor       # a lowercase name, unique in the file
    when: {water: sea}  # a `when` key is a range [min, max] or water: sea | lake | river | none
    ground: {color: [150, 140, 105], jitter: 0.0}  # RGB ground colour; jitter adds per-cell brightness variation
  - id: lake_bed        # the `when` keys: elevation (m), temperature (°C), moisture (0-1),
    when: {water: lake} # slope (°), sediment (m), relief (m), water
    ground: {color: [96, 92, 70], jitter: 0.0}
  - id: riverbed
    when: {water: river}
    ground: {color: [110, 100, 78], jitter: 0.0}
  - id: beach
    when: {elevation: [-1.0, 1.5], slope: [0.0, 20.0]}
    ground: {color: [205, 190, 140], jitter: 0.04}
  - id: snow
    when: {temperature: [-50.0, -2.0]}
    ground: {color: [235, 238, 242], jitter: 0.0}
  - id: cliff
    when: {slope: [38.0, 90.0]}
    ground: {color: [128, 122, 112], jitter: 0.05}
  - id: forest
    when: {moisture: [0.55, 1.0], temperature: [2.0, 25.0]}
    ground: {color: [78, 118, 60], jitter: 0.07}
  - id: grassland
    when: {moisture: [0.25, 0.55]}
    ground: {color: [112, 152, 82], jitter: 0.07}
  - id: desert
    when: {moisture: [0.0, 0.25], temperature: [18.0, 50.0]}
    ground: {color: [214, 180, 120], jitter: 0.04}
  - id: default         # no `when`: every cell nothing above matched
    ground: {color: [112, 152, 82], jitter: 0.07}
```

See `examples/` for recipes that change a little, and `worldgen.proc_gen.world.config` for the prose
documentation of every field.

## Tests

```sh
uv run pytest
uv run ruff check src
```
