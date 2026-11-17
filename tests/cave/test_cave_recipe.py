from pathlib import Path

from worldgen.cave.ui.model.recipe import CaveRecipe


def test_cave_recipe_defaults():
    recipe = CaveRecipe.defaults()
    assert recipe.valid
    assert recipe.config is not None
    assert recipe.problems == {}
    assert recipe.is_default("cavern")
    assert recipe.is_default("elevation")


def test_cave_recipe_get_set():
    recipe = CaveRecipe.defaults()
    assert recipe.get(("cavern", "clearance")) == 40.0

    recipe.set(("cavern", "clearance"), 28.5)
    assert recipe.get(("cavern", "clearance")) == 28.5
    assert not recipe.is_default("cavern")
    assert recipe.valid

    # Set invalid value
    recipe.set(("cavern", "clearance"), -5.0)
    assert not recipe.valid
    assert "cavern.clearance" in recipe.problems


def test_cave_recipe_reset():
    recipe = CaveRecipe.defaults()
    recipe.set(("cavern", "clearance"), 35.0)
    recipe.set(("cavern", "ceiling_mode"), "offset")
    assert not recipe.is_default("cavern")

    recipe.reset("cavern")
    assert recipe.is_default("cavern")
    assert recipe.get(("cavern", "clearance")) == 40.0
    assert recipe.get(("cavern", "ceiling_mode")) == "inverted"


def test_cave_recipe_save_and_load(tmp_path: Path):
    recipe = CaveRecipe.defaults()
    recipe.set(("cavern", "clearance"), 22.0)
    out_file = tmp_path / "my_cave.yaml"

    recipe.save(out_file)
    assert out_file.exists()

    loaded = CaveRecipe.load(out_file)
    assert loaded.valid
    assert loaded.get(("cavern", "clearance")) == 22.0
