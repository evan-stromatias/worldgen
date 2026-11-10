"""The terrain generator UI's draft recipe: edits by path, problems by path, typed text, load and save."""

import copy

import pytest
import yaml

from worldgen.config import DEFAULT_RECIPE, VERSION
from worldgen.generator.config import ConfigError, load_config
from worldgen.ui.model.recipe.form import Kind, blank, box_kind, parse, recipe_form
from worldgen.ui.model.recipe.recipe import HEADER, Recipe


@pytest.fixture
def recipe() -> Recipe:
    return Recipe.load(DEFAULT_RECIPE)


def test_the_default_recipe_is_valid_and_complete(recipe):
    assert recipe.valid and recipe.problems == {}


def test_the_saved_text_stamps_the_package_version(recipe):
    """Whatever version the file read in said, a save writes this worldgen's."""
    recipe.set(("version",), "9.9.9")
    assert yaml.safe_load(recipe.text())["version"] == VERSION


@pytest.mark.parametrize(
    ("path", "value", "where", "message"),
    [
        (("world", "cell_size"), "abc", "world.cell_size", "valid number"),
        (("world", "cell_size"), -1.0, "world.cell_size", "greater than 0"),
        (("elevation", "mask_falloff", 0), 0.99, "elevation.mask_falloff", "min 0.99 is greater than max 0.95"),
        (("elevation", "sea_floor"), 5.0, "", "must be below world.sea_level"),
        (("biomes", "rules", 0, "id"), "Sea", "biomes.rules[0].id", "should match pattern"),
    ],
)
def test_problems_land_on_their_paths(recipe, path, value, where, message):
    recipe.set(path, value)
    assert not recipe.valid and recipe.config is None
    assert list(recipe.problems) == [where]
    assert message in recipe.problems[where][0]


def test_fixing_a_problem_makes_the_recipe_valid_again(recipe):
    before = recipe.get(("world", "cell_size"))
    recipe.set(("world", "cell_size"), "abc")
    recipe.set(("world", "cell_size"), 2.0)
    assert recipe.valid and recipe.config.world.cell_size == 2.0 != before


def test_unreadable_text_is_kept_with_its_reason(recipe):
    path = ("biomes", "rules", 6, "when", "temperature", 0)  # forest's lower bound
    recipe.set(path, "warm", "not a number")
    assert recipe.get(path) == "warm"
    assert recipe.problems_at(path) and not recipe.valid
    recipe.set(path, 2.0)
    assert recipe.valid and recipe.config.biomes.rules[6].when.temperature == (2.0, 25.0)


def test_set_makes_missing_containers(recipe):
    del recipe.data["climate"]
    recipe.set(("climate", "moisture", "noise"), 0.2)
    assert recipe.valid and recipe.config.climate.moisture.noise == 0.2


def test_turning_off_and_on_gives_the_value_back(recipe):
    path = ("biomes", "rules", 6, "when", "temperature")
    recipe.turn_off(path)
    assert recipe.get(path) is None and recipe.valid
    recipe.turn_on(path, [0.0, 0.0])
    assert recipe.get(path) == [2.0, 25.0]


def test_reset_puts_one_section_back_to_its_defaults():
    recipe = Recipe.defaults()
    recipe.set(("elevation", "noise", "octaves"), "many", "not a whole number")
    recipe.set(("climate", "lapse_rate"), -1)
    assert not recipe.valid and not recipe.is_default("elevation") and not recipe.is_default("climate")
    recipe.reset("elevation")
    assert recipe.is_default("elevation") and recipe.problems_at(("elevation", "noise", "octaves")) == []
    assert not recipe.is_default("climate")
    recipe.reset("climate")
    assert recipe.valid and recipe.data == Recipe.defaults().data


def test_reset_does_not_share_the_defaults():
    recipe = Recipe.defaults()
    recipe.reset("biomes")
    recipe.remove(("biomes", "rules"), 0)
    assert Recipe.defaults().data["biomes"] != recipe.data["biomes"]


def test_save_and_load_give_the_same_recipe(recipe, tmp_path):
    recipe.set(("elevation", "mask"), "archipelago")
    path = tmp_path / "mine.yaml"
    recipe.save(path)
    assert recipe.path == path
    text = path.read_text(encoding="utf-8")
    assert text.startswith(HEADER) and "null" not in text
    assert "mask_falloff: [0.55, 0.95]" in text  # short lists stay on one line


def test_an_invalid_recipe_opens_with_its_problems(tmp_path):
    data = yaml.safe_load(DEFAULT_RECIPE.read_text(encoding="utf-8"))
    broken = copy.deepcopy(data)
    broken["world"]["cell_size"] = "big"
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    recipe = Recipe.load(path)
    assert recipe.data == broken  # kept as written, so the problem shows where it is
    assert list(recipe.problems) == ["world.cell_size"]


def test_files_that_are_not_recipes(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        Recipe.load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("world: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        Recipe.load(bad)
    bad.write_text("- a list\n")
    with pytest.raises(ConfigError, match="expected a mapping"):
        Recipe.load(bad)


@pytest.mark.parametrize(
    ("kind", "text", "value"),
    [
        (Kind.INT, " 42 ", 42),
        (Kind.INT, "4.5", "4.5"),  # left for validation to name
        (Kind.FLOAT, "1e-05", 1e-05),
        (Kind.FLOAT, "", None),
        (Kind.TEXT, " forest ", "forest"),
        (Kind.YAML, "{density: 0.6}", {"density": 0.6}),
        (Kind.YAML, "", None),
    ],
)
def test_parse(kind, text, value):
    assert parse(kind, text) == (value, None)


def test_locate_box_kind_and_blank():
    form = recipe_form()
    spec, rest = form.locate(("biomes", "rules", 3, "ground", "color", 1))
    assert (spec.key, rest, box_kind(spec, rest)) == ("color", (1,), Kind.INT)
    spec, rest = form.locate(("elevation", "mask_falloff", 0))
    assert box_kind(spec, rest) is Kind.FLOAT
    spec, rest = form.locate(("biomes", "rules", 0, "when"))
    assert rest == () and blank(spec) == {}


def test_the_defaults_have_no_file_until_saved(tmp_path):
    recipe = Recipe.defaults()
    assert recipe.path is None
    assert recipe.valid
    assert Recipe.load(DEFAULT_RECIPE).data
    path = tmp_path / "elsewhere" / "mine.yaml"  # its folder does not exist yet
    recipe.save(path)
    assert recipe.path == path and load_config(path) == recipe.config


def test_list_items_move_with_what_is_kept_for_them(recipe):
    rules = ("biomes", "rules")
    warm = (*rules, 6, "when", "temperature", 0)  # forest's, with text that does not read
    recipe.set(warm, "warm", "not a number")
    recipe.turn_off((*rules, 7, "when", "moisture"))  # grassland's, kept to turn on again
    order = recipe.move(rules, 6, 7)
    assert order[6] == 7 and order[7] == 6
    assert [r["id"] for r in recipe.data["biomes"]["rules"][6:8]] == ["grassland", "forest"]
    assert recipe.problems_at((*rules, 7, "when", "temperature", 0)) and not recipe.problems_at(warm)
    recipe.turn_on((*rules, 6, "when", "moisture"), {})
    assert recipe.get((*rules, 6, "when", "moisture")) == [0.25, 0.55]  # grassland's own came back

    recipe.remove(rules, 7)  # forest, with its unreadable text
    assert "forest" not in [r["id"] for r in recipe.data["biomes"]["rules"]] and recipe.valid
    recipe.insert(rules, 0, {"id": "first", "when": {"water": "sea"}, "ground": {"color": [0, 0, 0]}})
    assert recipe.valid and recipe.config.biomes.rules[0].id == "first" and recipe.config.biomes.rules[7].id == "grassland"
