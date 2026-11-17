from rayscreen import Label
from rayscreen.specs import check_unique_ids, walk
from worldgen.cave.ui.model.form import cave_recipe_form
from worldgen.cave.ui.model.recipe import CaveRecipe
from worldgen.cave.ui.view.stages import cave_recipe_tree
from worldgen.ui.model.stages import Status
from worldgen.ui.view.form import FormOptions
from worldgen.ui.view.stages import rule_cards


def test_cave_recipe_form_structure():
    spec = cave_recipe_form()
    assert spec.title == "Cave Recipe"
    keys = [f.key for f in spec.fields]
    assert keys == ["world", "elevation", "erosion", "cavern", "hydrology", "climate", "materials"]
    assert "version" not in keys
    assert "client" not in keys


def test_cave_recipe_tree_rendering_all_opened():
    recipe = CaveRecipe.defaults()
    spec = cave_recipe_form()

    # Open all sections
    opened_paths = {
        ("world",),
        ("elevation",),
        ("erosion",),
        ("cavern",),
        ("hydrology",),
        ("climate",),
        ("materials",),
    }

    options = FormOptions(
        recipe=recipe,  # type: ignore[arg-type]
        opened=lambda path, default: path in opened_paths,
        rules=rule_cards,
    )

    statuses = {
        "elevation": (Status.DONE, "done 0.05 s"),
        "erosion": (Status.DONE, "done 0.12 s"),
        "cavern": (Status.STALE, "stale"),
        "hydrology": (Status.NOT_RUN, "not run"),
        "climate": (Status.NOT_RUN, "not run"),
        "materials": (Status.NOT_RUN, "not run"),
    }

    tree = cave_recipe_tree(spec, options, statuses, runnable=True)
    check_unique_ids(tree)

    # Check tones
    valid_tones = {"ink", "muted", "error", "ok", "warning", "info"}
    for node in walk(tree):
        if isinstance(node, Label):
            assert node.tone in valid_tones
