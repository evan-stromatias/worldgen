"""The terrain generator UI's forms, built from the recipe's pydantic classes, and the recipe they show."""

from typing import Annotated

import pytest
from pydantic import Field

from worldgen.config import DEFAULT_RECIPE
from worldgen.proc_gen.world.config import Section, TerrainConfig
from worldgen.utils.world_gen_ui.model.form import Bounds, Kind, SectionSpec, _section, recipe_form, stepped
from worldgen.utils.world_gen_ui.model.recipe import Recipe
from worldgen.utils.world_gen_ui.model.stages import STAGES


@pytest.fixture(scope="module")
def form() -> SectionSpec:
    return recipe_form()


def test_one_section_per_recipe_section_in_order(form):
    assert [s.key for s in form.fields] == ["world", *STAGES]
    assert all(isinstance(s, SectionSpec) for s in form.fields)


def test_every_recipe_value_has_a_field(form):
    """Each leaf of a full recipe is drawn by exactly one field (rules are drawn per rule)."""
    data = Recipe.load(DEFAULT_RECIPE).data
    drawn = {path for path, _ in form.walk()}

    def leaves(value, path):
        if isinstance(value, dict) and path not in drawn:
            for key, child in value.items():
                yield from leaves(child, (*path, key))
        else:
            yield path

    missing = [path for path in leaves(data, ()) if path not in drawn and path != ("version",)]
    assert missing == []


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (("world", "size_cells"), Kind.INT),
        (("world", "cell_size"), Kind.FLOAT),
        (("world", "sea_level"), Kind.FLOAT),
        (("elevation", "mask"), Kind.CHOICE),
        (("elevation", "mask_falloff"), Kind.PAIR),
        (("hydrology", "rain_shadow", "enabled"), Kind.BOOL),
        (("hydrology", "rain_shadow", "wind"), Kind.PAIR),
        (("erosion", "enabled"), Kind.BOOL),
        (("erosion", "droplets"), Kind.FLOAT),
        (("erosion", "batch"), Kind.INT),
        (("biomes", "rules"), Kind.RULES),
    ],
)
def test_kinds(form, path, kind):
    assert form.find(path).kind is kind


def test_details_come_from_the_models(form):
    size = form.find(("world", "size_cells"))
    assert size.description == TerrainConfig.model_fields["world"].annotation.model_fields["size_cells"].description
    assert size.bounds == Bounds(ge=16)
    assert str(form.find(("elevation", "shore")).bounds) == "above 0, below 1"
    assert form.find(("elevation", "mask")).choices == ("island", "archipelago", "continent")
    assert form.find(("elevation", "mask_falloff")).item is Kind.FLOAT
    assert form.find(("elevation",)).description.startswith("The base heightfield")


def test_a_biome_rule(form):
    rule = form.find(("biomes", "rules")).rule
    kinds = {".".join(path): (spec.kind, spec.optional) for path, spec in rule.walk()}
    assert kinds["id"] == (Kind.TEXT, False)
    assert kinds["when.elevation"] == (Kind.PAIR, True)
    assert kinds["when.water"] == (Kind.CHOICE, True)
    assert kinds["ground.color"] == (Kind.COLOR, False)
    assert rule.find(("when",)).optional
    assert rule.find(("ground", "color")).bounds == Bounds(ge=0, le=255)


def test_a_type_without_a_widget_is_an_error():
    class Odd(Section):
        """A section with a field the UI cannot draw."""

        sizes: set[Annotated[int, Field(ge=0)]] = Field(default_factory=set, description="Some sizes.")

    with pytest.raises(TypeError, match="Odd.sizes"):
        _section(Odd, "odd", "Odd")


def test_recipe_get():
    recipe = Recipe.load(DEFAULT_RECIPE)
    assert recipe.get(("world", "size_cells")) == 1024
    assert recipe.get(("biomes", "rules", 0, "id")) == "sea_floor"
    assert recipe.get(("elevation", "mask_falloff", 1)) == 0.95
    assert recipe.get(("biomes", "rules", 99, "id")) is None
    assert recipe.get(("biomes", "rules", 0, "when", "elevation", 0)) is None


def test_size_cells_steps_by_patch_cells_and_comes_after_it(form):
    world = form.find(("world",))
    keys = [f.key for f in world.fields]
    assert keys.index("patch_cells") < keys.index("size_cells")
    assert form.find(("world", "size_cells")).step_by == "patch_cells"
    assert form.find(("world", "patch_cells")).step_by == ""


def test_stepping_goes_to_the_next_multiple():
    bounds = Bounds(ge=16)
    assert stepped(1024, 128, True, bounds) == 1152
    assert stepped(1024, 128, False, bounds) == 896
    assert stepped(1000, 128, True, bounds) == 1024  # a value that is not a multiple snaps to one
    assert stepped(1000, 128, False, bounds) == 896


def test_no_step_out_of_bounds_or_from_what_is_not_a_number():
    assert stepped(128, 128, False, Bounds(ge=16)) is None  # 0 is below 16
    assert stepped(256, 128, True, Bounds(le=256)) is None
    assert stepped("10x", 128, True, Bounds()) is None
    assert stepped(1024, None, True, Bounds()) is None
    assert stepped(1024, 0, True, Bounds()) is None
