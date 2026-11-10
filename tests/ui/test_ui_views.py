"""The terrain generator UI's screen as rayscreen trees: the whole recipe's form has a widget for every field, its
ids map back to the recipe, rayscreen's events translate into the tool's, and the screen lays out without overlaps,
its sections closed and all open. No window: text width is faked."""

import pytest

from rayscreen import Button, CheckBox, Dropdown, Event, Label, Picture, Rect, Swatch, TextBox, Toggle, arrange
from rayscreen.specs import check_unique_ids, walk
from worldgen.config import DEFAULT_RECIPE
from worldgen.generator.config import yaml_path
from worldgen.ui.controller import Controller
from worldgen.ui.model.recipe.form import Kind, recipe_form
from worldgen.ui.model.recipe.recipe import Recipe
from worldgen.ui.model.recipe.rules import RULES
from worldgen.ui.model.stages import STAGES
from worldgen.ui.view import events as ev
from worldgen.ui.view.form import Target, field_target, parse_yaml_path
from worldgen.ui.view.world_screen import WorldScreen

LEAVES = (Label, Button, Toggle, CheckBox, TextBox, Dropdown, Picture, Swatch)
FORM = recipe_form()


def fake_measure(text: str, size: int) -> float:
    return float(len(text) * 7)


@pytest.fixture
def controller(tmp_path) -> Controller:
    return Controller(Recipe.load(DEFAULT_RECIPE), 7, tmp_path / "terrain")


@pytest.fixture
def opened(controller) -> Controller:
    """Every section and rule open: the whole form at once. (Stages and rules start closed, nested sections open, so
    opening a nested section means leaving it out of `toggled`.)"""
    for path in [(spec.key,) for spec in FORM.fields] + [(*RULES, i) for i in range(len(controller.recipe.get(RULES)))]:
        controller.act(ev.SectionToggled(path))
    controller.show()
    return controller


def form_ids(controller: Controller) -> dict[str, object]:
    return {node.id: node for node in walk(controller.screen.root) if getattr(node, "id", None) is not None}


def test_every_field_of_the_recipe_has_a_widget_with_every_section_open(opened):
    ids = form_ids(opened)
    check_unique_ids(opened.screen.root)
    rules = opened.recipe.get(RULES)
    paths = [*FORM.walk()]
    paths += [((*RULES, i, *path), spec) for i in range(len(rules)) for path, spec in FORM.find(RULES).rule.walk()]
    for path, spec in paths:
        widget_id = yaml_path(path)
        if any(opened.recipe.get(path[:i]) is None for i in range(1, len(path))):
            assert f"{yaml_path(path[:-1])}.unset" in ids or yaml_path(path[:-1]) not in ids  # an optional section left out
            continue
        match spec.kind:
            case Kind.PAIR | Kind.COLOR:
                assert yaml_path((*path, 0)) in ids, widget_id
            case Kind.COUNTS | Kind.WEIGHTS:
                assert all(yaml_path((*path, key)) in ids for key in spec.choices), widget_id
            case Kind.PICK:
                choices = spec.choices or [rule["id"] for rule in rules]  # its own, or the biome ids
                assert all(f"{yaml_path((*path, choice))}.pick" in ids for choice in choices), widget_id
            case Kind.RULES:
                assert f"{widget_id}.add" in ids and f"{yaml_path((*path, 0))}.header" in ids
            case _:
                assert widget_id in ids, widget_id
        if spec.optional:
            assert f"{widget_id}.on" in ids, widget_id


def test_every_form_id_maps_back_to_its_place_in_the_recipe(opened):
    for widget_id, node in form_ids(opened).items():
        if isinstance(node, (TextBox, CheckBox, Dropdown)) and (target := field_target(widget_id, FORM, ())) is not None:
            head = widget_id.removesuffix(f".{target.part}") if target.part else widget_id
            assert yaml_path(target.path) == head
    assert parse_yaml_path("biomes.rules[3].ground.color[1]") == ("biomes", "rules", 3, "ground", "color", 1)
    assert field_target("biomes.rules[3].up", FORM, ()) == Target(("biomes", "rules", 3), "up")
    for not_form in ("seed", "layer", "stage.climate.run", "world-path", ".world", "world[x]"):
        assert field_target(not_form, FORM, ()) is None, not_form


def test_stages_start_closed_and_sections_inside_them_open(controller):
    ids = form_ids(controller)
    assert "hydrology.header" in ids and "hydrology.rain_shadow.header" not in ids  # closed
    controller.act(ev.SectionToggled(("hydrology",)))
    controller.show()
    ids = form_ids(controller)
    assert "hydrology.header" in ids and all(f"stage.{s}.run" in ids for s in STAGES)
    assert any(widget_id.startswith("hydrology.") and isinstance(node, TextBox) for widget_id, node in ids.items())


def test_events_translate_into_the_tools(controller):
    recipe = controller.recipe

    def one(widget: str, value: object = None) -> ev.Event | None:
        found = ev.translate([Event("world", widget, value)], FORM, recipe)
        return found[0] if found else None

    assert one("world.cell_size", "2.5") == ev.TextCommitted(("world", "cell_size"), "2.5")
    assert one("elevation.mask", "continent") == ev.ValueChosen(("elevation", "mask"), "continent")
    assert one("biomes.rules[6].when.slope.on", True) == ev.OptionalToggled(("biomes", "rules", 6, "when", "slope"), True)
    assert one("hydrology.header") == ev.SectionToggled(("hydrology",))
    assert one("biomes.rules[3].up") == ev.MoveRule(3, 2)
    assert one("biomes.rules[3].down") == ev.MoveRule(3, 4)
    assert one("biomes.rules[3].remove") == ev.RemoveRule(3)
    assert one("biomes.rules.add") == ev.AddRule()
    assert one("stage.climate.run") == ev.RunStage("climate") and one("stage.climate.reset") == ev.ResetStage("climate")
    assert one("seed", "12") == ev.SeedCommitted("12")
    assert one("layer", "biomes") == ev.LayerChosen("biomes") and one("layer", "none yet") is None
    assert one("3d", True) == ev.ViewChosen(True) and one("height", "x4") == ev.ExaggerationChosen(4.0)
    assert one("save-world") == ev.DialogAsked("dialog.save-world") and one("dialog.save-world.ok") == ev.SaveWorld()
    assert one("dialog.save-world.cancel") == ev.DialogClosed("dialog.save-world")
    assert one("dialog.overwrite.ok") == ev.Overwrite() and one("dialog.open.cancel") == ev.DialogClosed("dialog.open")
    assert one("map-menu", "centre") is None  # a menu choice needs the point it was opened at
    found = ev.translate([Event("world", "map-menu", "centre", target=(10.0, 20.0))], FORM, recipe)
    assert found == [ev.MenuChosen("centre", (10.0, 20.0))]


def test_typed_text_goes_in_before_the_click_that_left_its_box(controller):
    clicked = Event("world", "biomes.rules[3].up", None)
    typed = Event("world", "biomes.rules[3].id", "woods")  # rayscreen reports it after the click, in tree order
    found = ev.translate([clicked, typed], FORM, controller.recipe)
    assert found == [ev.TextCommitted(("biomes", "rules", 3, "id"), "woods"), ev.MoveRule(3, 2)]


def overlaps(a: Rect, b: Rect) -> bool:
    return a.x < b.x + b.width and b.x < a.x + a.width and a.y < b.y + b.height and b.y < a.y + a.height


def check_layout(screen: WorldScreen, width: float, height: float) -> None:
    """Every widget inside the window and none over another: outside the form, and inside it (laid out whole, at
    the scroll's width)."""
    arrange(screen.root, Rect(0, 0, width, height), fake_measure)
    form = screen.find("form")
    scrolled = {id(node) for node in walk(form.child)}
    for group in ([n for n in walk(screen.root) if id(n) not in scrolled], [n for n in walk(form.child)]):
        leaves = [node for node in group if isinstance(node, LEAVES) and node.rect.width > 0]
        for i, a in enumerate(leaves):
            for b in leaves[i + 1:]:
                assert not overlaps(a.rect, b.rect), (a, b)
    for node in walk(screen.root):
        if isinstance(node, LEAVES) and id(node) not in scrolled and node.rect.width > 0:
            assert node.rect.x >= 0 and node.rect.x + node.rect.width <= width + 0.01, node
            assert node.rect.y >= 0 and node.rect.y + node.rect.height <= height + 0.01, node
        if isinstance(node, LEAVES) and id(node) in scrolled:
            assert form.rect.x <= node.rect.x and node.rect.x + node.rect.width <= form.rect.x + form.rect.width + 0.01, node


@pytest.mark.parametrize("width, height", [(1600, 900), (1100, 700)])
def test_the_screen_lays_out_with_its_sections_closed(controller, width, height):
    controller.act(ev.TextCommitted(("world", "cell_size"), "abc"))  # a problem, for the Problems panel
    controller.show()
    check_layout(controller.screen, width, height)
    picture = controller.screen.picture.rect
    assert picture.width >= width / 2 - 30 and picture.height >= height / 2
    assert controller.screen.find("problems.count") is not None


@pytest.mark.parametrize("width, height", [(1600, 900), (1100, 700)])
def test_the_screen_lays_out_with_every_section_open(opened, width, height):
    check_layout(opened.screen, width, height)


def test_opening_a_section_keeps_the_box_being_edited_and_the_scroll(controller):
    screen = controller.screen
    screen.state.editing = "seed"
    screen.state.scroll_ptr("form")[0].y = -40.0
    controller.act(ev.SectionToggled(("climate",)))
    controller.show()
    assert screen.state.editing == "seed" and screen.state.scroll_ptr("form")[0].y == -40.0
    assert any(getattr(node, "id", "").startswith("climate.") and isinstance(node, TextBox) for node in walk(screen.root))


def test_a_fields_help_is_its_names_tooltip(controller):
    controller.act(ev.SectionToggled(("world",)))
    controller.show()
    name = controller.screen.find("world.cell_size.label")
    assert controller.screen.find("world.cell_size.about") is None  # help takes no room on the form
    texts = [node.text for node in walk(name.tooltip.root) if isinstance(node, Label)]
    assert texts[0] == "Cell size" and "above 0" in texts  # the description, and the bounds
    assert any("bigger world" in text for text in texts)  # what turning it up does


def test_a_sections_help_is_its_headers_tooltip(controller):
    header = controller.screen.find("world.header")
    texts = [node.text for node in walk(header.tooltip.root) if isinstance(node, Label)]
    assert "Size and layout of the world grid." in texts
