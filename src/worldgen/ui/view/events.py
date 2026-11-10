"""What the user did, in the tool's terms: what the controller acts on. `translate` makes them from the events
rayscreen returns, which carry widget ids (a form's are YAML paths, `view/form.py`). No raylib."""

from dataclasses import dataclass
from typing import Any

import rayscreen
from worldgen.ui.model.map.layers import LAYERS
from worldgen.ui.model.recipe.form import Kind, SectionSpec, stepped
from worldgen.ui.model.recipe.recipe import FieldPath, Recipe
from worldgen.ui.model.recipe.rules import RULES
from worldgen.ui.view import form as form_ids
from worldgen.ui.view import windows
from worldgen.ui.view.form import field_target


@dataclass(frozen=True, slots=True)
class TextCommitted:
    """A text box was left with Enter or a click elsewhere; the text is as typed, not parsed."""

    path: FieldPath
    text: str


@dataclass(frozen=True, slots=True)
class ValueChosen:
    """A check box, choice or toggle changed: the value is ready to use."""

    path: FieldPath
    value: Any


@dataclass(frozen=True, slots=True)
class OptionalToggled:
    """A field that may be left out was turned on or off."""

    path: FieldPath
    on: bool


@dataclass(frozen=True, slots=True)
class SeedCommitted:
    text: str


@dataclass(frozen=True, slots=True)
class RunStage:
    stage: str


@dataclass(frozen=True, slots=True)
class ResetStage:
    """Put a stage's recipe section back to its defaults."""

    stage: str


@dataclass(frozen=True, slots=True)
class RunAll:
    pass


@dataclass(frozen=True, slots=True)
class StopAll:
    """Stop the run going on, the stage running included."""


@dataclass(frozen=True, slots=True)
class WorldPathTyped:
    """The world folder box was left: Save world writes, and View opens, this folder."""

    text: str


@dataclass(frozen=True, slots=True)
class SaveWorld:
    pass


@dataclass(frozen=True, slots=True)
class RecipePathTyped:
    """The recipe file box was left: Open reads, and Save as writes, this path."""

    text: str


@dataclass(frozen=True, slots=True)
class OpenRecipe:
    pass


@dataclass(frozen=True, slots=True)
class SaveRecipeAs:
    pass


@dataclass(frozen=True, slots=True)
class LayerChosen:
    layer: str


@dataclass(frozen=True, slots=True)
class AddRule:
    """Add a biome rule before the catch-all."""


@dataclass(frozen=True, slots=True)
class RemoveRule:
    index: int


@dataclass(frozen=True, slots=True)
class MoveRule:
    index: int
    to: int


@dataclass(frozen=True, slots=True)
class ClearLog:
    pass


@dataclass(frozen=True, slots=True)
class CopyLog:
    """Every line of the log, to the clipboard."""


@dataclass(frozen=True, slots=True)
class ToggleLog:
    """Open the log window, or close it."""


@dataclass(frozen=True, slots=True)
class SectionToggled:
    """A section's or biome rule's header was clicked: open it, or close it."""

    path: FieldPath


@dataclass(frozen=True, slots=True)
class DescriptionsShown:
    on: bool


@dataclass(frozen=True, slots=True)
class ViewChosen:
    """2D or 3D."""

    three_d: bool


@dataclass(frozen=True, slots=True)
class ExaggerationChosen:
    """How many times its real height the 3D view draws the ground."""

    factor: float


@dataclass(frozen=True, slots=True)
class ThemeSwitched:
    """Light to dark, or back."""


@dataclass(frozen=True, slots=True)
class DialogAsked:
    """A button that opens a dialog was pressed: Open..., Save as... or Save world..."""

    window: str


@dataclass(frozen=True, slots=True)
class DialogClosed:
    """Cancel, the close button or Esc: forget what the dialog was asking."""

    window: str


@dataclass(frozen=True, slots=True)
class Overwrite:
    """The overwrite question was answered yes."""


@dataclass(frozen=True, slots=True)
class MenuChosen:
    """A choice of the map's context menu, about the world point (x, z) it was opened at."""

    choice: str
    point: tuple[float, float]


Event = (
    TextCommitted | ValueChosen | OptionalToggled | SeedCommitted | RunStage | ResetStage | RunAll | StopAll
    | WorldPathTyped | SaveWorld | RecipePathTyped | OpenRecipe | SaveRecipeAs
    | LayerChosen | AddRule | RemoveRule | MoveRule
    | ClearLog | CopyLog | ToggleLog | SectionToggled | DescriptionsShown | ViewChosen | ExaggerationChosen | ThemeSwitched
    | DialogAsked | DialogClosed | Overwrite | MenuChosen
)

MAP_MENU = "map-menu"
# Buttons and windows whose event is the tool's event as it stands
SIMPLE: dict[str, Event] = {
    "run-all": RunAll(), "stop-all": StopAll(), "theme": ThemeSwitched(), "log": ToggleLog(),
    "log.copy": CopyLog(), "log.clear": ClearLog(), f"{windows.OPEN}.ok": OpenRecipe(), f"{windows.SAVE_AS}.ok": SaveRecipeAs(),
    f"{windows.SAVE_WORLD}.ok": SaveWorld(), f"{windows.OVERWRITE}.ok": Overwrite(),
    "open": DialogAsked(windows.OPEN), "save-as": DialogAsked(windows.SAVE_AS), "save-world": DialogAsked(windows.SAVE_WORLD),
}
DIALOGS = (windows.OPEN, windows.SAVE_AS, windows.SAVE_WORLD, windows.OVERWRITE, windows.LOG)


def translate(events: list[rayscreen.Event], form: SectionSpec, recipe: Recipe) -> list[Event]:
    """The tool's events for what rayscreen reported this frame. Text a click left goes first, at the path it was
    typed at, before that click acts: typing in a rule and then moving it moves the rule with the typed text."""
    found: list[Event] = []
    for raw in events:
        if isinstance(raw.value, str) and raw.widget in (windows.OPEN, windows.SAVE_AS, windows.SAVE_WORLD):
            if raw.widget == windows.OPEN:
                found.extend((RecipePathTyped(raw.value), OpenRecipe()))
            elif raw.widget == windows.SAVE_AS:
                found.extend((RecipePathTyped(raw.value), SaveRecipeAs()))
            elif raw.widget == windows.SAVE_WORLD:
                found.extend((WorldPathTyped(raw.value), SaveWorld()))
            continue
        event = _translated(raw, form, recipe)
        if event is not None:
            found.append(event)
    found.sort(key=lambda event: not isinstance(event, TextCommitted))
    return found


def _translated(event: rayscreen.Event, form: SectionSpec, recipe: Recipe) -> Event | None:
    widget, value = event.widget, event.value
    if (target := field_target(widget, form, ())) is not None:
        return _form_event(target.path, target.part, value, form, recipe)
    if widget in SIMPLE:
        return SIMPLE[widget]
    stage, _, action = widget.removeprefix("stage.").rpartition(".")
    if widget.startswith("stage.") and action in ("run", "reset"):
        return RunStage(stage) if action == "run" else ResetStage(stage)
    match widget:
        case "seed":
            return SeedCommitted(str(value))
        case "layer" if value in LAYERS:
            return LayerChosen(str(value))
        case "2d" | "3d":
            return ViewChosen(widget == "3d")
        case "height":  # "x2"
            return ExaggerationChosen(float(str(value).removeprefix("x")))
        case "descriptions":
            return DescriptionsShown(bool(value))
        case windows.OPEN_PATH | windows.SAVE_AS_PATH:
            return RecipePathTyped(str(value))
        case windows.SAVE_WORLD_PATH:
            return WorldPathTyped(str(value))
    if widget == MAP_MENU:
        return MenuChosen(str(value), event.target) if value is not None and isinstance(event.target, tuple) else None
    window, _, part = widget.rpartition(".")
    if part == "cancel" and window in DIALOGS:
        return DialogClosed(window)
    if widget in DIALOGS:  # closed by its close button or Esc: rayscreen has closed it
        return DialogClosed(widget)
    return None


def _form_event(path: FieldPath, part: str, value: Any, form: SectionSpec, recipe: Recipe) -> Event | None:
    match part:
        case "":  # the field's own control
            spec, rest = form.locate(path)
            if spec.kind in (Kind.BOOL, Kind.CHOICE) and not rest:
                return ValueChosen(path, value)
            return TextCommitted(path, str(value))
        case form_ids.ON:
            return OptionalToggled(path, bool(value))
        case form_ids.HEADER:
            return SectionToggled(path)
        case form_ids.MINUS | form_ids.PLUS:
            spec, _ = form.locate(path)
            new = stepped(recipe.get(path), recipe.get((*path[:-1], spec.step_by)), part == form_ids.PLUS, spec.bounds)
            return None if new is None else ValueChosen(path, new)
        case form_ids.PICK:
            picked, choice = recipe.get(path[:-1]) or [], path[-1]
            spec, _ = form.locate(path[:-1])
            rules = recipe.get(RULES)
            ids = [str(rule.get("id", "")) for rule in rules if isinstance(rule, dict)] if isinstance(rules, list) else []
            order = spec.choices or ids  # its own choices, or the rules' biome ids
            return ValueChosen(path[:-1], [b for b in order if (b in picked) != (b == choice)])  # keep their order
        case form_ids.UP | form_ids.DOWN if isinstance(path[-1], int):
            return MoveRule(path[-1], path[-1] + (-1 if part == form_ids.UP else 1))
        case form_ids.REMOVE if isinstance(path[-1], int):
            return RemoveRule(path[-1])
        case form_ids.ADD:
            return AddRule()
    return None
