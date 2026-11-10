import pytest
from pydantic import BaseModel, ValidationError

from worldgen.config import DEFAULT_RECIPE
from worldgen.generator.config import (
    BiomeRule,
    ConfigError,
    TerrainConfig,
    config_errors,
    load_config,
    parse_config,
)


def errors_for(data: dict) -> str:
    with pytest.raises(ConfigError) as caught:
        parse_config(data)
    return str(caught.value)


def test_default_recipe_loads():
    config = load_config(DEFAULT_RECIPE)
    assert config.biomes.rules[-1].when is None


def test_only_rules_is_enough(small_data):
    config = parse_config({"biomes": small_data["biomes"]})
    assert config.world.size_cells == 1024


def test_rules_can_be_replaced(small_data):
    config = parse_config({"biomes": small_data["biomes"]})
    assert [rule.id for rule in config.biomes.rules] == [rule["id"] for rule in small_data["biomes"]["rules"]]


def test_seed_is_rejected(small_data):
    small_data["seed"] = 7
    assert "pass it with --seed" in errors_for(small_data)


def test_client_data_is_kept_verbatim(small_data):
    """A client embeds its own recipe's sections under `client`; worldgen stores them, never reads them."""
    client = {"version": 3, "towns": {"count": 5, "capitol": None}, "biome_flora": {"forest": ["oak"]}}
    assert parse_config({**small_data, "client": client}).client == client
    assert parse_config(small_data).client == {}  # left out: an empty mapping


def test_client_must_be_a_mapping(small_data):
    small_data["client"] = "towns"
    assert "client: Input should be a valid dictionary" in errors_for(small_data)


def test_top_level_keys_beside_client_stay_forbidden(small_data):
    """Only `client` loosens the top level, so a typo of a worldgen key is still named."""
    small_data["towns"] = {}
    assert "towns: Extra inputs are not permitted" in errors_for(small_data)


def test_unknown_key_names_its_path(small_data):
    small_data["hydrology"]["lake_min_aera"] = 3
    assert "hydrology.lake_min_aera: Extra inputs are not permitted" in errors_for(small_data)


def test_out_of_range_names_its_path(small_data):
    small_data["hydrology"]["lake_min_area"] = 0
    assert "hydrology.lake_min_area:" in errors_for(small_data)


def test_patches_must_tile_the_world(small_data):
    small_data["world"]["patch_cells"] = 48
    assert "world: size_cells (128) must be a multiple of patch_cells (48)" in errors_for(small_data)


def test_sea_floor_below_sea_level(small_data):
    small_data["elevation"]["sea_floor"] = 1.0
    assert "must be below world.sea_level" in errors_for(small_data)


def test_the_version_is_never_checked(small_data):
    """Compatibility is the client's call: a recipe from another worldgen version still validates."""
    small_data["version"] = "99.0.0"
    assert parse_config(small_data).version == "99.0.0"
    small_data["version"] = 1  # an old int version coerces
    assert parse_config(small_data).version == "1"


def test_an_invalid_recipe_names_the_files_version(small_data):
    small_data["version"] = "0.0.1"
    small_data["world"]["size_cells"] = -5
    message = errors_for(small_data)
    assert "written by worldgen 0.0.1" in message and "world.size_cells" in message


def test_bad_range_names_rule(small_data):
    small_data["biomes"]["rules"][2]["when"]["elevation"] = [3, 1]
    assert "biomes.rules[2].when.elevation: min 3.0 is greater than max 1.0" in errors_for(small_data)


def test_missing_catch_all(small_data):
    small_data["biomes"]["rules"].pop()
    assert "must have no `when`" in errors_for(small_data)


def test_catch_all_must_be_last(small_data):
    rules = small_data["biomes"]["rules"]
    rules.insert(1, {"id": "early", "ground": {"color": [0, 0, 0]}})
    assert "'early' has no `when` but is not last" in errors_for(small_data)


def test_duplicate_id(small_data):
    small_data["biomes"]["rules"][1]["id"] = "sea_floor"
    assert "duplicate biome id 'sea_floor'" in errors_for(small_data)


def test_empty_when(small_data):
    small_data["biomes"]["rules"][2]["when"] = {}
    assert "biomes.rules[2].when: `when` needs at least one condition" in errors_for(small_data)


def test_every_problem_is_reported(small_data):
    small_data["world"]["cell_size"] = -1
    small_data["climate"]["lapse_rate"] = -1
    message = errors_for(small_data)
    assert "world.cell_size" in message and "climate.lapse_rate" in message


def test_unreadable_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("world: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)
    bad.write_text("")
    with pytest.raises(ConfigError, match="expected a mapping"):
        load_config(bad)


def test_archipelago_sizes_must_fit(small_data):
    small_data["elevation"] = {"mask": "archipelago", "archipelago": {"size": [0.1, 0.7]}}
    assert "elevation.archipelago: size must lie within (0, 0.5]" in errors_for(small_data)


def test_errors_by_path(small_data):
    small_data["world"]["cell_size"] = -1
    small_data["elevation"] = {"sea_floor": 5.0}
    with pytest.raises(ValidationError) as caught:
        TerrainConfig.model_validate(small_data)
    errors = dict(config_errors(caught.value))
    assert set(errors) == {"world.cell_size"}  # the cross-section check runs only once the sections are valid
    assert errors["world.cell_size"].startswith("Input should be greater than 0")
    del small_data["world"]["cell_size"]
    with pytest.raises(ValidationError) as caught:
        TerrainConfig.model_validate(small_data)
    assert config_errors(caught.value) == [("", "elevation.sea_floor (5.0) must be below world.sea_level (0.0)")]


def section_classes(model: type[BaseModel] = TerrainConfig):
    yield model
    for field in model.model_fields.values():
        for kind in (field.annotation, *getattr(field.annotation, "__args__", ())):
            if isinstance(kind, type) and issubclass(kind, BaseModel):
                yield from section_classes(kind)


def test_every_section_and_field_is_described():
    """The terrain generator UI shows these as its help text."""
    classes = set(section_classes()) | {BiomeRule}
    assert [c.__name__ for c in classes if not c.__doc__] == []
    missing = [
        f"{c.__name__}.{name}"
        for c in classes
        for name, field in c.model_fields.items()
        if not field.description
        and not (isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel))  # sections have docstrings
    ]
    assert missing == []
