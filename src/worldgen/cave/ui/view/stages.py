"""Cave recipe stages as rayscreen specs: status badges, reset and run buttons, and expandable forms."""

from __future__ import annotations

from collections.abc import Mapping

from rayscreen import ROW, SMALL, Button, CheckBox, Column, Label, Node, Row, Tone, Tooltip
from worldgen.ui.model.recipe.form import SectionSpec
from worldgen.ui.model.recipe.recipe import FieldPath
from worldgen.ui.model.stages import Status
from worldgen.ui.view.form import (
    ON,
    FormOptions,
    form_tree,
    header_button,
    indented,
    problems,
    problems_below,
)

type StageTexts = Mapping[str, tuple[Status, str]]

CAVE_STAGE_NAMES: tuple[str, ...] = ("elevation", "erosion", "cavern", "hydrology", "climate", "materials")

TONES: dict[Status, Tone] = {
    Status.NOT_RUN: "muted",
    Status.QUEUED: "info",
    Status.RUNNING: "info",
    Status.DONE: "ok",
    Status.STALE: "warning",
    Status.SKIPPED: "muted",
    Status.FAILED: "error",
}

RESET, RUN = "#56#", "#131#Run"
RESET_TIP = Tooltip.text("reset", "Reset", "Back to default values")


def stage_ids(stage: str) -> tuple[str, str, str]:
    """The ids of a stage's status label, Reset, and Run."""
    return f"stage.{stage}.status", f"stage.{stage}.reset", f"stage.{stage}.run"


def cave_recipe_tree(form: SectionSpec, options: FormOptions, statuses: StageTexts, runnable: bool) -> Column:
    """Build column with all recipe sections and stage-level execution buttons."""
    sections = {spec.key: spec for spec in form.fields if isinstance(spec, SectionSpec)}
    blocks: list[Node] = [_block(section, options, statuses, runnable) for section in sections.values()]
    return Column(children=blocks)


def _block(section: SectionSpec, options: FormOptions, statuses: StageTexts, runnable: bool) -> Column:
    key, recipe = section.key, options.recipe
    path: FieldPath = (key,)
    opened = options.opened is not None and options.opened(path, False)
    staged = key in CAVE_STAGE_NAMES
    title = f"{CAVE_STAGE_NAMES.index(key) + 1}. {section.title}" if staged else section.title
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
