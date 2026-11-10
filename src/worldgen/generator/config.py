"""The terrain recipe: `terrain.yaml`, validated before any generation starts.

One file holds every generation parameter and the biome rules for stages 1-5 (elevation, erosion,
hydrology, climate, biomes). The seed is not in it: it comes from the command line, so one recipe can
make many terrains.

The `version` key records the worldgen package version that wrote the file (semantic versioning,
stamped on export). worldgen never checks it: whether a file from another version still works is the
client's call. When reading fails here, the error names the version in the file.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError, model_validator

from worldgen.config import VERSION


class ConfigError(ValueError):
    """A terrain.yaml that cannot be used; the message names each problem by its YAML path."""


def _ordered(pair: tuple[float, float]) -> tuple[float, float]:
    if pair[0] > pair[1]:
        raise ValueError(f"min {pair[0]} is greater than max {pair[1]}")
    return pair


Range = Annotated[tuple[float, float], AfterValidator(_ordered)]
Channel = Annotated[int, Field(ge=0, le=255)]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorldSection(Section):
    """Size and layout of the world grid."""

    cell_size: float = Field(default=1.0, gt=0, description="Metres per cell.")
    patch_cells: int = Field(default=128, ge=2, description="Cells per patch side, when the terrain is read in patches.")
    size_cells: int = Field(default=1024, ge=16, description="Cells per side; a multiple of patch_cells.")
    sea_level: float = Field(default=0.0, description="Height of the sea surface, in metres.")

    @model_validator(mode="after")
    def _patches_tile(self) -> Self:
        if self.size_cells % self.patch_cells:
            raise ValueError(f"size_cells ({self.size_cells}) must be a multiple of patch_cells ({self.patch_cells})")
        return self


class NoiseSection(Section):
    """Fractal noise for the heightfield: octaves of noise summed, each finer and fainter than the last."""

    octaves: int = Field(default=6, ge=1, le=12, description="Layers of noise summed.")
    frequency: float = Field(default=0.003, gt=0, description="Cycles per cell of the first octave.")
    lacunarity: float = Field(default=2.0, gt=1, description="Frequency multiplier from one octave to the next.")
    gain: float = Field(default=0.5, gt=0, lt=1, description="Amplitude multiplier from one octave to the next.")


class WarpSection(Section):
    """Domain warp: more noise pushes where the height noise is sampled, so ridges and coasts bend."""

    strength: float = Field(default=80.0, ge=0, description="How far sample points are pushed, in cells.")
    frequency: float = Field(default=0.002, gt=0, description="Cycles per cell of the warp noise.")


class ArchipelagoSection(Section):
    """Used when elevation.mask is archipelago."""

    islands: int = Field(default=7, ge=2, le=64, description="Islands to place; fewer if they do not fit.")
    size: Range = Field(
        default=(0.1, 0.3), description="Island radius range, as a fraction of the world width (0.5 = island mask)."
    )
    spacing: float = Field(
        default=0.8, ge=0, description="Minimum gap between island centres, in units of their summed outer radii."
    )

    @model_validator(mode="after")
    def _sizes_fit(self) -> Self:
        if not (self.size[0] > 0 and self.size[1] <= 0.5):
            raise ValueError(f"size must lie within (0, 0.5], got {list(self.size)}")
        return self


class MountainsSection(Section):
    """Mountain ranges: ridged noise with sharp crests, risen only where a slow mask puts ranges, so lowland
    stays between them. Lengths are in metres."""

    enabled: bool = Field(default=False, description="Whether to raise mountain ranges.")
    wavelength: float = Field(default=400.0, gt=0, description="Spacing of the ridges, in metres.")
    octaves: int = Field(default=6, ge=1, le=12, description="Layers of ridged noise summed.")
    gain: float = Field(default=0.5, gt=0, lt=1, description="Amplitude multiplier from one octave to the next.")
    sharpness: float = Field(default=2.0, ge=1, le=6, description="How sharp the crests are: 1 is a crease, 2 a knife edge.")
    height: float = Field(
        default=0.6, gt=0, lt=1,
        description="Share of max_height the ranges rise above the hills; the hills keep the rest.",
    )
    coverage: float = Field(default=0.35, ge=0, le=1, description="Share of the world in ranges.")
    range_wavelength: float = Field(default=1500.0, gt=0, description="Size of the range mask's blobs, in metres.")
    warp: bool = Field(default=True, description="Sample the ridges through the hills' domain warp, so ranges bend with them.")


class TerracesSection(Section):
    """Stepped ground: heights gathered into strata, so a covered hilltop reads as a plateau or mesa with cliff
    edges. Lengths are in metres."""

    enabled: bool = Field(default=False, description="Whether to step the ground.")
    step: float = Field(default=12.0, gt=0, description="Height of one stratum, in metres.")
    sharpness: float = Field(default=0.7, ge=0, le=1, description="0 smooth slopes, 1 nearly vertical step faces.")
    where: Literal["all", "mountains", "noise"] = Field(
        default="noise",
        description="all: all land; mountains: the mountain ranges' mask; noise: a mask of its own (wavelength, coverage).",
    )
    wavelength: float = Field(default=800.0, gt=0, description="Size of the terrace mask's blobs (where: noise), in metres.")
    coverage: float = Field(default=0.3, ge=0, le=1, description="Share of the world terraced (where: noise).")


class ElevationSection(Section):
    """The base heightfield: noise shaped by a land mask."""

    mask: Literal["island", "archipelago", "continent"] = Field(
        default="island",
        description=(
            "island: one island, falling off into sea at the edges; archipelago: several islands; "
            "continent: land out to the edges, unless falloff sinks them into the sea."
        ),
    )
    mask_falloff: Range = Field(
        default=(0.55, 0.95),
        description="Distance from an island's centre (1 = its radius) where land starts and stops falling into sea.",
    )
    mask_distortion: float = Field(default=0.15, ge=0, le=1, description="How much noise bends the island outline.")
    falloff: float = Field(
        default=0.0, ge=0, le=0.5,
        description="mask: continent only — share of the world width the coast lies within; 0 keeps land to the edges.",
    )
    shore: float = Field(default=0.08, gt=0, lt=1, description="Shaped noise value (0-1) at which land meets the sea.")
    max_height: float = Field(default=60.0, gt=0, description="Metres above sea level at the highest point.")
    sea_floor: float = Field(default=-12.0, description="Height of the deepest open sea, in metres.")
    noise: NoiseSection = Field(default_factory=NoiseSection)
    warp: WarpSection = Field(default_factory=WarpSection)
    archipelago: ArchipelagoSection = Field(default_factory=ArchipelagoSection)
    mountains: MountainsSection = Field(default_factory=MountainsSection)
    terraces: TerracesSection = Field(default_factory=TerracesSection)

    @model_validator(mode="after")
    def _terraces_have_mountains(self) -> Self:
        if self.terraces.enabled and self.terraces.where == "mountains" and not self.mountains.enabled:
            raise ValueError("terraces.where is mountains, but mountains are not enabled")
        return self


class ErosionSection(Section):
    """Weathering after elevation: simulated droplets cut gullies and valleys and leave sediment where they slow,
    from a coarse grid to the full one; then loose ground slides until no slope is steeper than it holds."""

    enabled: bool = Field(default=False, description="Whether to erode the ground.")
    droplets: float = Field(default=1.0, ge=0, description="Droplets per vertex of each level's grid.")
    batch: int = Field(
        default=8192, ge=1, description="Droplets that step together; they see each other's changes a step late.",
    )
    levels: int = Field(
        default=3, ge=1, le=6, description="Grids eroded, coarsest first: 3 is the ground at every 8th, 4th and 1st vertex.",
    )
    lifetime: int = Field(default=40, ge=1, description="Steps a droplet takes at most.")
    inertia: float = Field(
        default=0.1, ge=0, lt=1, description="How much a droplet keeps its direction rather than turn downhill.",
    )
    capacity: float = Field(
        default=1.0, gt=0, description="Sediment a droplet can carry, per metre it drops in a step and unit of speed and water.",
    )
    min_slope: float = Field(default=0.01, ge=0, description="Least slope capacity is worked out with, so flats still carry.")
    erode_rate: float = Field(default=0.3, ge=0, le=1, description="Share of its spare capacity a droplet erodes each step.")
    deposit_rate: float = Field(default=0.3, ge=0, le=1, description="Share of its excess sediment a droplet drops each step.")
    evaporation: float = Field(default=0.02, ge=0, lt=1, description="Share of its water a droplet loses each step.")
    gravity: float = Field(default=0.1, gt=0, description="How fast a droplet speeds up going down, per unit of slope.")
    radius: float = Field(
        default=2.0, ge=1, description="Radius of the brush a droplet erodes with, in vertices of its level's grid.",
    )
    smoothing: int = Field(
        default=1, ge=0, le=8,
        description="Vertices each level's change is blurred over before it is added, so slopes are not left rippled; 0 is none.",
    )
    talus_angle: float = Field(default=35.0, gt=0, lt=90, description="Steepest slope loose ground holds, in degrees.")
    rock_talus_angle: float = Field(
        default=70.0, gt=0, lt=90, description="Steepest slope inside the terrace mask, in degrees, so cliffs survive.",
    )
    thermal_iterations: int = Field(default=40, ge=0, description="Passes of sliding; 0 skips it.")
    thermal_rate: float = Field(default=0.5, gt=0, le=1, description="Share of the excess height a pass moves downhill.")


class RainSection(Section):
    """Rainfall over the world, before the rain shadow."""

    base: float = Field(default=1.0, gt=0, description="Rainfall per cell before noise.")
    noise: float = Field(default=0.5, ge=0, le=1, description="Fraction of the base that noise adds or removes.")
    frequency: float = Field(default=0.001, gt=0, description="Cycles per cell of the rain noise.")


class RainShadowSection(Section):
    """Less rain downwind of high ground."""

    enabled: bool = Field(default=True, description="Whether ridges take rain from the land downwind of them.")
    wind: tuple[float, float] = Field(default=(1.0, 0.0), description="Prevailing wind direction (x, z).")
    strength: float = Field(default=0.6, ge=0, le=1, description="Fraction of rain lost behind the tallest ridges.")

    @model_validator(mode="after")
    def _wind_has_direction(self) -> Self:
        if self.wind == (0, 0):
            raise ValueError("wind must not be (0, 0)")
        return self


class MeanderSection(Section):
    """Bends in rivers."""

    strength: float = Field(
        default=2.0, ge=0, description="How far rivers stray from straight downhill; 0 is plain steepest descent."
    )
    frequency: float = Field(default=0.03, gt=0, description="Cycles per cell of the noise that steers the bends.")


class HydrologySection(Section):
    """Lakes and rivers."""

    epsilon: float = Field(default=0.0001, gt=0, description="Priority-flood step, so flats still drain.")
    lake_min_depth: float = Field(default=0.5, ge=0, description="Shallower depressions are filled in, in metres.")
    lake_min_area: int = Field(default=40, ge=1, description="Smaller depressions are filled in, in cells.")
    river_threshold: float = Field(
        default=1500.0, gt=0, description="Accumulated rainfall (in cells' worth) at which a river starts."
    )
    carve_depth: float = Field(default=2.5, ge=0, description="k in channel depth = k * strength^0.5, in metres.")
    bank_falloff: int = Field(default=3, ge=0, description="Cells over which banks slope down to a river.")
    river_min_width: float = Field(
        default=1.5, gt=0, description="Width of a river where it starts, in metres (never under 1.5 cells)."
    )
    river_width_scale: float = Field(
        default=4.0, ge=0, description="Width the strongest river adds, in metres; width grows with strength."
    )
    rain: RainSection = Field(default_factory=RainSection)
    rain_shadow: RainShadowSection = Field(default_factory=RainShadowSection)
    meander: MeanderSection = Field(default_factory=MeanderSection)


class MoistureSection(Section):
    """How moist the ground is: from rain and from water nearby."""

    rain_weight: float = Field(
        default=0.6, ge=0, le=1, description="Share of moisture from rain; the rest is water nearby."
    )
    water_distance: float = Field(default=60.0, gt=0, description="How far water nearby keeps land moist, in cells.")
    noise: float = Field(default=0.15, ge=0, description="Amplitude of moisture noise.")


class ClimateSection(Section):
    """Temperature and moisture, the inputs to the biome rules."""

    base_temperature: float = Field(default=18.0, description="°C at sea level on the south edge.")
    latitude_gradient: float = Field(default=12.0, description="°C colder on the north edge than the south edge.")
    lapse_rate: float = Field(
        default=300.0, ge=0, description="°C colder per 1000 m of height; exaggerated so low peaks still get snow."
    )
    temperature_noise: float = Field(default=3.0, ge=0, description="Amplitude of temperature noise, in °C.")
    moisture: MoistureSection = Field(default_factory=MoistureSection)


class When(Section):
    """Conditions of a biome rule. Ranges are inclusive; a missing key matches anything."""

    elevation: Range | None = Field(default=None, description="Metres above sea level.")
    temperature: Range | None = Field(default=None, description="°C.")
    moisture: Range | None = Field(default=None, description="0-1.")
    slope: Range | None = Field(default=None, description="Degrees.")
    sediment: Range | None = Field(default=None, description="Metres of sediment erosion left on the ground.")
    relief: Range | None = Field(
        default=None, description="Metres above the lowest ground within biomes.relief_radius: a plateau or peak is high.",
    )
    water: Literal["sea", "lake", "river", "none"] | None = Field(
        default=None, description="The water over the cell: sea, lake, river, or none for dry land."
    )

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if all(getattr(self, key) is None for key in type(self).model_fields):
            raise ValueError("`when` needs at least one condition; leave it out for the catch-all biome")
        return self


class Ground(Section):
    """How a biome's ground looks."""

    color: tuple[Channel, Channel, Channel] = Field(description="Ground colour (RGB, 0-255).")
    jitter: float = Field(default=0.0, ge=0, le=0.5, description="Per-cell brightness variation (fraction).")


class BiomeRule(Section):
    """One biome: where it applies and how it looks. The first rule whose `when` matches a cell wins."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Name, unique in the file.")
    when: When | None = Field(default=None, description="Where the biome applies; only the last rule leaves it out.")
    ground: Ground = Field(description="How the ground looks.")


# The biome rules a recipe gets when it gives none; the default recipe lists the same ones (a test checks)
DEFAULT_RULES = (
    {"id": "sea_floor", "when": {"water": "sea"}, "ground": {"color": (150, 140, 105)}},
    {"id": "lake_bed", "when": {"water": "lake"}, "ground": {"color": (96, 92, 70)}},
    {"id": "riverbed", "when": {"water": "river"}, "ground": {"color": (110, 100, 78)}},
    {
        "id": "beach", "when": {"elevation": (-1, 1.5), "slope": (0, 20)}, "ground": {"color": (205, 190, 140), "jitter": 0.04},
    },
    {"id": "snow", "when": {"temperature": (-50, -2)}, "ground": {"color": (235, 238, 242)}},
    {"id": "cliff", "when": {"slope": (38, 90)}, "ground": {"color": (128, 122, 112), "jitter": 0.05}},
    {
        "id": "forest", "when": {"moisture": (0.55, 1.0), "temperature": (2, 25)},
        "ground": {"color": (78, 118, 60), "jitter": 0.07},
    },
    {
        "id": "grassland", "when": {"moisture": (0.25, 0.55)}, "ground": {"color": (112, 152, 82), "jitter": 0.07},
    },
    {
        "id": "desert", "when": {"moisture": (0.0, 0.25), "temperature": (18, 50)},
        "ground": {"color": (214, 180, 120), "jitter": 0.04},
    },
    {"id": "default", "ground": {"color": (112, 152, 82), "jitter": 0.07}},
)


class BiomesSection(Section):
    """Biomes from elevation, temperature, moisture, slope, sediment, relief and water, by rules tried in order."""

    blend_width: float = Field(
        default=0.08, ge=0, le=0.5, description="Fraction of each range over which neighbouring biomes blend colour."
    )
    relief_radius: float = Field(
        default=60.0, gt=0, description="Metres around a cell its relief looks for the lowest ground in (a square).",
    )
    rules: list[BiomeRule] = Field(
        default_factory=lambda: [BiomeRule.model_validate(rule) for rule in DEFAULT_RULES],
        min_length=1,
        max_length=255,
        description="Tried in order; the last has no `when`, so every cell gets a biome.",
    )

    @model_validator(mode="after")
    def _well_formed(self) -> Self:
        seen = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate biome id '{rule.id}'")
            seen.add(rule.id)
        if self.rules[-1].when is not None:
            raise ValueError(f"the last biome ('{self.rules[-1].id}') must have no `when`, so every cell gets a biome")
        for rule in self.rules[:-1]:
            if rule.when is None:
                raise ValueError(
                    f"biome '{rule.id}' has no `when` but is not last; the biomes after it could never match"
                )
        return self


class TerrainConfig(Section):
    """A whole terrain.yaml: the world grid and the stages 1-5 (elevation, erosion, hydrology, climate, biomes)."""

    version: Annotated[str, BeforeValidator(str)] = Field(
        default=VERSION,
        description="The worldgen version that wrote the file, stamped on export; never checked on load.",
    )
    world: WorldSection = Field(default_factory=WorldSection)
    elevation: ElevationSection = Field(default_factory=ElevationSection)
    erosion: ErosionSection = Field(default_factory=ErosionSection)
    hydrology: HydrologySection = Field(default_factory=HydrologySection)
    climate: ClimateSection = Field(default_factory=ClimateSection)
    biomes: BiomesSection = Field(default_factory=BiomesSection)

    @model_validator(mode="before")
    @classmethod
    def _no_seed(cls, data: Any) -> Any:
        if isinstance(data, dict) and "seed" in data:
            raise ValueError("`seed` does not belong in the file; pass it with --seed")
        return data

    @model_validator(mode="after")
    def _sea_floor_below_sea(self) -> Self:
        if self.elevation.sea_floor >= self.world.sea_level:
            raise ValueError(
                f"elevation.sea_floor ({self.elevation.sea_floor}) must be below "
                f"world.sea_level ({self.world.sea_level})"
            )
        return self


def yaml_path(loc: tuple) -> str:
    """A place in a recipe as messages name it: ("biomes", "rules", 2, "when") is biomes.rules[2].when."""
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else f".{part}" if path else str(part)
    return path


def config_errors(error: ValidationError) -> list[tuple[str, str]]:
    """Each problem as (YAML path, message); the path is "" for the recipe as a whole."""
    return [(yaml_path(err["loc"]), err["msg"].removeprefix("Value error, ")) for err in error.errors()]


def parse_config(data: Any, source: str = "terrain.yaml") -> TerrainConfig:
    """Validate already-parsed YAML; raise ConfigError listing every problem, naming the file's `version`
    so a file another worldgen wrote says which."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping of sections at the top level")
    try:
        return TerrainConfig.model_validate(data)
    except ValidationError as e:
        written = f" (written by worldgen {data['version']})" if "version" in data else ""
        lines = [f"  {path}: {message}" if path else f"  {message}" for path, message in config_errors(e)]
        raise ConfigError(f"{source}{written} is invalid:\n" + "\n".join(lines)) from None


def load_config(path: Path) -> TerrainConfig:
    """Read and validate a terrain.yaml."""
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e.strerror}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from None
    return parse_config(data, str(path))
