"""A recipe's sections as rayscreen specs, built from `model/recipe/form.py`'s specs, so a new recipe field
shows up with no change here. No raylib: the tree is tested without a window.

- Each widget's id is its field's YAML path (`elevation.noise.octaves`, `biomes.rules[3].ground.color[1]`), so
  an event maps back to the field with no lookup table (`field_target`). The parts of one field add a suffix: the
  check box turning a field left out on and off is `.on`, a section's header `.header`, a stepper's buttons
  `.minus` and `.plus`, a biome toggle `.<id>.pick`. Labels hang off the same path (`.label`, `.about`,
  `.problem0`) and emit nothing.
- Text boxes keep what was typed, valid or not, with the problem under them.
- Nested sections are titled panels, or, given `FormOptions.opened`, headers that open and close (a long form).
- Biome rules are the world tool's to draw (`FormOptions.rules`), with its rules for moving them; a form that
  meets rules without it raises, as for any kind it has no widget for.
"""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import yaml

from rayscreen import (
    ICON,
    ROW,
    SMALL,
    Button,
    CheckBox,
    Column,
    Dropdown,
    Fill,
    Grid,
    Label,
    Node,
    Panel,
    Row,
    Swatch,
    TextBox,
    Toggle,
    Tooltip,
)
from rayscreen.specs import walk
from worldgen.generator.config import yaml_path
from worldgen.ui.model.recipe.form import FieldSpec, Kind, SectionSpec, stepped
from worldgen.ui.model.recipe.recipe import FieldPath, Recipe

ON, HEADER, MINUS, PLUS, PICK = "on", "header", "minus", "plus", "pick"  # the parts of a field, as id suffixes
UP, DOWN, REMOVE, ADD = "up", "down", "remove", "add"  # a list's items (the biome rules)
PARTS = frozenset({ON, HEADER, MINUS, PLUS, PICK, UP, DOWN, REMOVE, ADD})
INDENT = 16.0  # pixels a section's fields stand in from its header
PICK_CELL = 150.0  # a biome toggle's cell: as wide as the longest ids
OPEN_ARROW, CLOSED_ARROW = "#116#", "#115#"  # raygui's down and right arrows

type RulesTree = Callable[[FieldSpec, FieldPath, "FormOptions"], list[Node]]  # a list of rules, drawn by its tool
type Opened = Callable[[FieldPath, bool], bool]  # whether a section is open, given whether it starts open


class Target(NamedTuple):
    """What a form widget's event is about: a place in the recipe, and the part of its field ("" for its value)."""

    path: FieldPath
    part: str


@dataclass(frozen=True, slots=True)
class FormOptions:
    """How to draw the form: the recipe it shows, and what the tool adds to the specs."""

    recipe: Recipe
    descriptions: bool = True  # under each field; otherwise its name's tooltip
    defaults: Mapping[FieldPath, Mapping[str, float]] = field(default_factory=dict)  # greyed for a WEIGHTS key left out
    notes: Mapping[FieldPath, str] = field(default_factory=dict)  # beside a field's box: what its value comes to
    choices: tuple[str, ...] = ()  # what a PICK field picks from: the recipe's biome ids
    opened: Opened | None = None  # sections that open and close; None: always open, as titled panels
    rules: RulesTree | None = None


def form_tree(section: SectionSpec, options: FormOptions, path: FieldPath) -> Column:
    """The section's description and fields top to bottom, nested sections in panels or under headers. Raises
    TypeError on a field kind it has no widget for."""
    widget_id = yaml_path(path)
    children: list[Node] = []
    if options.descriptions and section.description:
        children.append(about(widget_id, section.description))
    if section.optional and options.recipe.get(path) is None:
        children.append(Label(f"{widget_id}.unset", "(not set)", size=SMALL, tone="muted"))
        return Column(children=children)
    for spec in section.fields:
        child_path = (*path, spec.key)
        if isinstance(spec, SectionSpec):
            children.append(section_tree(spec, options, child_path))
        else:
            children += _field(spec, child_path, options)
    return Column(children=children)


def section_tree(section: SectionSpec, options: FormOptions, path: FieldPath) -> Node:
    """A nested section: a titled panel, or, when sections open and close, its header and (while open) its fields,
    stood in. An optional section has a check box to set it or leave it out."""
    widget_id, recipe = yaml_path(path), options.recipe
    on = CheckBox(f"{widget_id}.{ON}", "", recipe.get(path) is not None)
    if options.opened is None:
        body = form_tree(section, options, path)
        if section.optional:
            body.children.insert(0, Row(children=[on, Label(f"{widget_id}.on.label", "set", size=SMALL, tone="muted")]))
        return Panel(body, title=section.title)
    opened = options.opened(path, True)
    header: list[Node] = [header_button(widget_id, section.title, opened)]
    if not opened and problems_below(recipe, path):
        header.append(Label(f"{widget_id}.flag", "has problems", size=SMALL, tone="error"))
    if section.optional:
        header.append(on)
    children: list[Node] = [Row(children=header), *problems(widget_id, recipe.problems_at(path))]
    if opened:
        children.append(indented(form_tree(section, options, path)))
    return Column(children=children)


def header_button(widget_id: str, title: str, opened: bool) -> Button:
    """A section's title, reading from the left, with an arrow saying whether it is open; a click opens or closes it."""
    return Button(f"{widget_id}.{HEADER}", f"{OPEN_ARROW if opened else CLOSED_ARROW}{title}", align="left", weight=1)


def indented(node: Column) -> Row:
    node.weight = 1
    return Row(children=[Fill(weight=0, size=INDENT), node])


def field_target(widget_id: str, section: SectionSpec, root: FieldPath) -> Target | None:
    """The field (or section, or list) a form widget's event is about, and which part of it; None for a widget
    outside the form. `section` is the form at `root`: the recipe's whole form at ()."""
    head, _, last = widget_id.rpartition(".")
    part = last if head and last in PARTS else ""
    path = parse_yaml_path(head if part else widget_id)
    if path is None or path[:len(root)] != root or len(path) == len(root):
        return None
    try:
        section.locate(path[len(root):])
    except StopIteration:  # no such key in the form
        return None
    return Target(path, part)


_STEP = re.compile(r"\[(\d+)\]|(\.?)([^.\[\]]+)")


def parse_yaml_path(text: str) -> FieldPath | None:
    """A place in a recipe from its YAML path, the inverse of `config.yaml_path`: biomes.rules[2].when is
    ("biomes", "rules", 2, "when"). None for text that is not one."""
    path: list[str | int] = []
    at = 0
    while at < len(text):
        step = _STEP.match(text, at)
        if step is None:
            return None
        index, dot, key = step.groups()
        if index is not None:
            path.append(int(index))
        elif bool(dot) == (at == 0):  # a dot goes between keys, never first
            return None
        else:
            path.append(key)
        at = step.end()
    return tuple(path) or None


def text_of(value: object) -> str:
    """A value as its box shows it: nothing for a value left out, a mapping or list as one line of YAML, numbers
    as Python writes them (every digit, so showing a float never changes it)."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, default_flow_style=True, width=math.inf, sort_keys=False).strip()
    return str(value)


def problems_below(recipe: Recipe, path: FieldPath) -> bool:
    """Whether the recipe has a problem at `path` or anywhere under it: a closed section says so."""
    prefix = yaml_path(path)
    return any(where == prefix or where.startswith((prefix + ".", prefix + "[")) for where in recipe.problems)


def problems(widget_id: str, messages: list[str]) -> list[Node]:
    return [Label(f"{widget_id}.problem{i}", message, size=SMALL, wrap=True, tone="error") for i, message in enumerate(messages)]


def about(widget_id: str, text: str, bounds: str = "") -> Label:
    """A field's or section's description, muted, its bounds after it. A wrapped label, not a `Text` of spans: a
    long form has a hundred of these, and spans cost twice as much to lay out every frame."""
    return Label(f"{widget_id}.about", text + (f" ({bounds})" if bounds else ""), size=SMALL, wrap=True, tone="muted")


def _field(spec: FieldSpec, path: FieldPath, options: FormOptions) -> list[Node]:
    widget_id, recipe = yaml_path(path), options.recipe
    value = recipe.get(path)
    messages = recipe.problems_at(path)
    match spec.kind:
        case Kind.RULES if options.rules is not None:
            return options.rules(spec, path, options)
        case Kind.COUNTS | Kind.WEIGHTS:
            return _keyed(spec, path, options)
        case Kind.TEXT | Kind.INT | Kind.FLOAT | Kind.YAML if spec.step_by:
            control: Node = _stepper(spec, path, recipe)
        case Kind.TEXT | Kind.INT | Kind.FLOAT | Kind.YAML:
            control = TextBox(widget_id, text_of(value), invalid=bool(messages), weight=1)
        case Kind.BOOL:
            control = CheckBox(widget_id, "", bool(value), weight=1)
        case Kind.CHOICE:
            current = spec.choices.index(str(value)) if str(value) in spec.choices else 0
            control = Dropdown(widget_id, spec.choices, current, weight=1)
        case Kind.PAIR | Kind.COLOR:
            parts = 2 if spec.kind is Kind.PAIR else 3
            messages = messages + [m for i in range(parts) for m in recipe.problems_at((*path, i))]
            boxes: list[Node] = [
                TextBox(yaml_path((*path, i)), text_of(_part(value, i)), invalid=bool(recipe.problems_at((*path, i))), weight=1)
                for i in range(parts)
            ]
            if spec.kind is Kind.COLOR:
                boxes.append(swatch(f"{widget_id}.swatch", value))
            control = Row(children=boxes, weight=1)
        case Kind.PICK:
            picked = value if isinstance(value, list) else []
            toggles: list[Node] = [Toggle(f"{yaml_path((*path, choice))}.{PICK}", choice, choice in picked) for choice in spec.choices or options.choices]
            control = Grid(toggles, cell=PICK_CELL, weight=1)
        case _:
            raise TypeError(f"{widget_id}: the generator UI has no widget for a {spec.kind.name} field")
    if spec.optional:
        if value is None:
            _disable(control)
        control = Row(children=[CheckBox(f"{widget_id}.{ON}", "", value is not None), control], weight=1)
    row: list[Node] = [_name(widget_id, spec, options), control]
    if path in options.notes:
        row.append(Label(f"{widget_id}.note", options.notes[path], size=SMALL, tone="muted"))
    nodes: list[Node] = [Row(children=row), *problems(widget_id, messages)]
    if options.descriptions and spec.description:
        nodes.append(about(widget_id, spec.description, str(spec.bounds)))
    return nodes


def _name(widget_id: str, spec: FieldSpec, options: FormOptions, title: str | None = None) -> Label:
    """A field's name; with descriptions hidden, its description shows as the name's tooltip."""
    title = title or _title(spec.key)
    tip = None
    if not options.descriptions and spec.description:
        bounds = str(spec.bounds)
        tip = Tooltip.text(f"help.{widget_id}", title, spec.description, *([bounds] if bounds else []))
    return Label(f"{widget_id}.label", title, weight=1, tooltip=tip)


def _stepper(spec: FieldSpec, path: FieldPath, recipe: Recipe) -> Row:
    """A box between - and + buttons that move its value to the next multiple of the `step_by` field; a button
    that would leave the bounds, or has no number to step, is greyed."""
    widget_id, value = yaml_path(path), recipe.get(path)
    step = recipe.get((*path[:-1], spec.step_by))
    down, up = (stepped(value, step, upwards, spec.bounds) is not None for upwards in (False, True))
    return Row(children=[
        Button(f"{widget_id}.{MINUS}", "-", enabled=down, width=ROW),
        TextBox(widget_id, text_of(value), invalid=bool(recipe.problems_at(path)), weight=1),
        Button(f"{widget_id}.{PLUS}", "+", enabled=up, width=ROW),
    ], weight=1)


def _keyed(spec: FieldSpec, path: FieldPath, options: FormOptions) -> list[Node]:
    """A row per key: COUNTS a box each (0 for a key left out); WEIGHTS a check box turning the key on, then its
    box, greyed with the default while off."""
    widget_id, recipe = yaml_path(path), options.recipe
    rows: list[Node] = [Row(children=[_name(widget_id, spec, options)])]
    if options.descriptions and spec.description:
        rows.append(about(widget_id, spec.description, str(spec.bounds)))
    values = recipe.get(path) or {}
    defaults = options.defaults.get(path, {})
    for key in spec.choices:
        key_path = (*path, key)
        key_id = yaml_path(key_path)
        messages = recipe.problems_at(key_path)
        if spec.kind is Kind.COUNTS:
            cells: list[Node] = [Fill(weight=0, size=INDENT), Label(f"{key_id}.label", key, weight=1),
                                 TextBox(key_id, text_of(values.get(key, 0)), invalid=bool(messages), weight=1)]
        else:
            on = key in values
            text = text_of(values[key]) if on else text_of(defaults.get(key))
            cells = [CheckBox(key_id + "." + ON, key, on, weight=1), TextBox(key_id, text, invalid=bool(messages), enabled=on, weight=1)]
        rows.append(Row(children=cells))
        rows += problems(key_id, messages)
    rows += problems(widget_id, recipe.problems_at(path))
    return rows


def swatch(widget_id: str, colour: Any) -> Node:
    """A colour's sample while it is a valid RGB triple; an empty square of the same size otherwise."""
    if isinstance(colour, list | tuple) and len(colour) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in colour):
        return Swatch(widget_id, (colour[0], colour[1], colour[2]))
    return Fill(weight=0, size=ICON)


def _disable(node: Node) -> None:
    """Grey every control in `node` that can be: a field left out shows its controls, unusable."""
    for child in walk(node):
        if isinstance(child, (TextBox, CheckBox, Button)):
            child.enabled = False


def _part(value: Any, i: int) -> Any:
    return value[i] if isinstance(value, list | tuple) and i < len(value) else None


def _title(key: str) -> str:
    return key.replace("_", " ").capitalize()
