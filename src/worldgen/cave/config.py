"""The cave recipe: `cave.yaml`, validated before any generation starts.

One file holds generation parameters for the subterranean cave: the floor elevation and erosion,
the 180°-rotated ceiling, clearance, pillar thresholds, subterranean hydrology, cave climate,
and speleological material rules. The seed comes from the command line.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from worldgen.config import VERSION
from worldgen.generator.config import (
    BiomeRule,
    BiomesSection,
    ElevationSection,
    ErosionSection,
    Ground,
    Section,
    WorldSection,
    config_errors,
)

DEFAULT_CAVE_RECIPE = Path(__file__).resolve().parents[3] / "assets" / "recipes" / "cave.yaml"


class ConfigError(ValueError):
    """A cave.yaml that cannot be used; the message names each problem by its YAML path."""


class CavernSection(Section):
    """Ceiling height and pillar clearance parameters."""

    clearance: float = Field(
        default=18.0, gt=0, description="Nominal vertical distance between floor and ceiling, in metres."
    )
    ceiling_mode: Literal["inverted", "offset"] = Field(
        default="inverted",
        description="'inverted': floor relief is inverted (valleys become domes, ridges become stalactites); 'offset': ceiling mirrors floor offset upwards.",
    )
    ceiling_scale: float = Field(default=1.0, gt=0, description="Ceiling relief scale factor.")
    min_passage: float = Field(
        default=2.0, gt=0, description="Minimum clearance in metres for passable cavern; below this cells are solid rock pillars."
    )
    perimeter_wall: bool = Field(
        default=True, description="Whether perimeter boundary cells are sealed as solid rock walls."
    )
    wall_thickness: int = Field(default=2, ge=1, le=32, description="Thickness of perimeter boundary wall, in cells.")


class CaveHydrologySection(Section):
    """Subterranean water table and flooded sumps."""

    enabled: bool = Field(default=True, description="Whether standing water / subterranean pools are present.")
    water_table: float = Field(
        default=-2.0, description="Subterranean water table elevation in metres. Floor below this is submerged in pools/sumps."
    )


class CaveClimateSection(Section):
    """Depth-based temperature and dampness."""

    temperature_base: float = Field(default=12.0, description="Ambient cave temperature at reference elevation, in °C.")
    geothermal_gradient: float = Field(default=0.025, ge=0, description="Temperature increase with depth, °C per metre.")
    ambient_humidity: float = Field(default=0.75, ge=0, le=1, description="Ambient relative humidity (0-1) in dry cavern chambers.")


def default_material_rules() -> list[BiomeRule]:
    """Default speleological material rules for cave surfaces."""
    return [
        BiomeRule(
            id="sump",
            when={"water": "lake"},
            ground=Ground(color=(50, 70, 95), jitter=0.02),
        ),
        BiomeRule(
            id="sediment",
            when={"sediment": (0.3, 100.0)},
            ground=Ground(color=(95, 80, 65), jitter=0.04),
        ),
        BiomeRule(
            id="flowstone",
            when={"slope": (45.0, 90.0)},
            ground=Ground(color=(215, 210, 195), jitter=0.03),
        ),
        BiomeRule(
            id="fungal_grotto",
            when={"moisture": (0.85, 1.0), "temperature": (13.0, 40.0)},
            ground=Ground(color=(65, 130, 90), jitter=0.06),
        ),
        BiomeRule(
            id="crystal_vault",
            when={"relief": (15.0, 100.0)},
            ground=Ground(color=(175, 210, 225), jitter=0.05),
        ),
        BiomeRule(
            id="limestone",
            ground=Ground(color=(135, 130, 125), jitter=0.05),
        ),
    ]


def default_materials() -> BiomesSection:
    return BiomesSection(blend_width=0.08, relief_radius=60.0, rules=default_material_rules())


def default_elevation() -> ElevationSection:
    return ElevationSection(
        mask="continent",
        falloff=0.0,
        shore=0.08,
        max_height=30.0,
        sea_floor=-10.0,
    )


class CaveConfig(BaseModel):
    """A complete cave recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(
        default=VERSION,
        description="The worldgen version that wrote the file, stamped on export; never checked on load.",
    )
    world: WorldSection = Field(default_factory=WorldSection)
    elevation: ElevationSection = Field(default_factory=default_elevation)
    erosion: ErosionSection = Field(default_factory=ErosionSection)
    cavern: CavernSection = Field(default_factory=CavernSection)
    hydrology: CaveHydrologySection = Field(default_factory=CaveHydrologySection)
    climate: CaveClimateSection = Field(default_factory=CaveClimateSection)
    materials: BiomesSection = Field(default_factory=default_materials)
    client: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-owned data; stored and exported verbatim, never read by worldgen.",
    )

    @model_validator(mode="before")
    @classmethod
    def _no_seed(cls, data: Any) -> Any:
        if isinstance(data, dict) and "seed" in data:
            raise ValueError("`seed` does not belong in the file; pass it with --seed")
        return data


def parse_config(data: Any, source: str = "cave.yaml") -> CaveConfig:
    """Validate already-parsed YAML; raise ConfigError listing every problem."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: expected a mapping of sections at the top level")
    try:
        return CaveConfig.model_validate(data)
    except ValidationError as e:
        written = f" (written by worldgen {data['version']})" if "version" in data else ""
        lines = [f"  {path}: {message}" if path else f"  {message}" for path, message in config_errors(e)]
        raise ConfigError(f"{source}{written} is invalid:\n" + "\n".join(lines)) from None


def load_config(path: Path) -> CaveConfig:
    """Read and validate a cave.yaml."""
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e.strerror}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from None
    return parse_config(data, str(path))
