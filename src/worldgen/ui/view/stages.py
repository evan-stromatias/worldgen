"""The recipe as the form shows it, around `view/form.py`'s form: a block per recipe section in the recipe's order,
each a stage's under a header with its status, Reset and Run; the biome rules as cards, each with its buttons to
move or remove it. No raylib.

Stages start closed, so the list of them fits on screen; sections inside them start open; rules start closed.
"""

from collections.abc import Mapping

from rayscreen import ROW, SMALL, Button, CheckBox, Column, Fill, Label, Node, Row, Tone, Tooltip
from worldgen.generator.config import yaml_path
from worldgen.ui.model.recipe.form import FieldSpec, SectionSpec
from worldgen.ui.model.recipe.recipe import FieldPath
from worldgen.ui.model.recipe.rules import can_move, can_remove
from worldgen.ui.model.stages import STAGES, Status
from worldgen.ui.view.form import (
    ADD,
    DOWN,
    ON,
    REMOVE,
    UP,
    FormOptions,
    field_tip,
    form_tree,
    header_button,
    indented,
    problems,
    problems_below,
    swatch,
)

type StageTexts = Mapping[str, tuple[Status, str]]  # each stage's status, and how to say it ("done 2.31 s")

TONES: dict[Status, Tone] = {
    Status.NOT_RUN: "muted",
    Status.QUEUED: "info",
    Status.RUNNING: "info",
    Status.DONE: "ok",
    Status.STALE: "warning",
    Status.SKIPPED: "muted",
    Status.FAILED: "error",
}
RESET, RUN = "#56#", "#131#Run"  # raygui's undo arrow and play icons
RESET_TIP = Tooltip.text("reset", "Reset", "Back to the default values")
UP_ICON, DOWN_ICON, REMOVE_ICON, ADD_ICON = "#117#", "#116#", "#143#", "#8#"  # raygui's arrows, bin and add-file


def recipe_tree(form: SectionSpec, options: FormOptions, statuses: StageTexts, runnable: bool) -> Column:
    """Every recipe section, a stage's with its controls; `runnable` says whether Run may be pressed."""
    sections = {spec.key: spec for spec in form.fields if isinstance(spec, SectionSpec)}
    blocks: list[Node] = [_block(section, options, statuses, runnable) for section in sections.values()]
    return Column(children=blocks)


def stage_ids(stage: str) -> tuple[str, str, str]:
    """The ids of a stage's status label, Reset and Run, which change without the tree changing shape."""
    return f"stage.{stage}.status", f"stage.{stage}.reset", f"stage.{stage}.run"


def _block(section: SectionSpec, options: FormOptions, statuses: StageTexts, runnable: bool) -> Column:
    key, recipe = section.key, options.recipe
    path: FieldPath = (key,)
    opened = options.opened is not None and options.opened(path, False)
    staged = key in STAGES
    title = f"{STAGES.index(key) + 1} {section.title}" if staged else section.title
    header: list[Node] = [header_button(key, title, opened, section.description)]
    if not opened and problems_below(recipe, path):
        header.append(Label(f"{key}.flag", "has problems", size=SMALL, tone="error"))
    if staged:
        status, text = statuses.get(key, (Status.NOT_RUN, Status.NOT_RUN.value))
        status_id, reset_id, run_id = stage_ids(key)
        header += [
            Label(status_id, text, size=SMALL, tone=TONES[status]),
            Button(reset_id, RESET, enabled=not recipe.is_default(key), width=ROW, tooltip=RESET_TIP),
            Button(run_id, RUN, enabled=runnable),
        ]
    if section.optional:
        header.append(CheckBox(f"{key}.{ON}", "", recipe.get(path) is not None))
    children: list[Node] = [Row(children=header), *problems(key, recipe.problems_at(path))]
    if opened:
        children.append(indented(form_tree(section, options, path)))
    return Column(children=children)


def rule_cards(spec: FieldSpec, path: FieldPath, options: FormOptions) -> list[Node]:
    """The biome rules, a card each, closed until opened: its number and id, whether it has problems, up, down and
    remove (greyed where they would move the catch-all), and its ground's colour. Then a button to add a rule."""
    widget_id, recipe = yaml_path(path), options.recipe
    rules = recipe.get(path)
    rules = rules if isinstance(rules, list) else []
    nodes: list[Node] = [Label(f"{widget_id}.label", f"Rules ({len(rules)})", tooltip=field_tip(widget_id, "Rules", spec))]
    nodes += problems(widget_id, recipe.problems_at(path))
    for i, rule in enumerate(rules):
        rule = rule if isinstance(rule, dict) else {}
        rule_path = (*path, i)
        rule_id = yaml_path(rule_path)
        opened = options.opened is not None and options.opened(rule_path, False)
        title = f"{i + 1}. {rule.get('id', '')}" + ("  (catch-all, always last)" if i == len(rules) - 1 else "")
        header: list[Node] = [header_button(rule_id, title, opened)]
        if not opened and problems_below(recipe, rule_path):
            header.append(Label(f"{rule_id}.flag", "has problems", size=SMALL, tone="error"))
        ground = rule.get("ground")
        header += [
            Button(f"{rule_id}.{UP}", UP_ICON, enabled=can_move(rules, i, i - 1), width=ROW),
            Button(f"{rule_id}.{DOWN}", DOWN_ICON, enabled=can_move(rules, i, i + 1), width=ROW),
            Button(f"{rule_id}.{REMOVE}", REMOVE_ICON, enabled=can_remove(rules, i), width=ROW),
            swatch(f"{rule_id}.colour", ground.get("color") if isinstance(ground, dict) else None),
        ]
        nodes.append(Row(children=header))
        nodes += problems(rule_id, recipe.problems_at(rule_path))
        if opened:
            nodes.append(indented(form_tree(spec.rule, options, rule_path)))
    nodes.append(Row(children=[Fill(), Button(f"{widget_id}.{ADD}", f"{ADD_ICON}Add a biome rule")]))
    return nodes
