"""The tool's windows, as rayscreen `Window`s: `FileWindow`s for Open and Save as for the recipe and Export world
for the terrain's folder, the questions before writing over a file or a folder, and the log. No raylib.

A dialog's widgets hang off its id (`dialog.open.path`, `dialog.open.ok`); its Cancel button, close button and Esc
all close it.
"""

from collections.abc import Sequence
from typing import NamedTuple

from rayscreen import (
    SMALL,
    Button,
    Column,
    FileWindow,
    Fill,
    Label,
    Node,
    Row,
    Scroll,
    TextBox,
    Tone,
    Tooltip,
    Window,
)
from worldgen.utils.world_gen_ui.model.log import Line, Source

OPEN, SAVE_AS, SAVE_WORLD = "dialog.open", "dialog.save-as", "dialog.save-world"
OPEN_PATH, SAVE_AS_PATH, SAVE_WORLD_PATH = f"{OPEN}.path", f"{SAVE_AS}.path", f"{SAVE_WORLD}.path"
OVERWRITE, LOG = "dialog.overwrite", "log-window"
LOG_SIZE = (620.0, 320.0)
LOG_SHOWN = 300  # the newest lines the log window holds
BOTTOM = -1e9  # a scroll offset past any content: raygui clamps it to the bottom


def buttons(window_id: str, ok: str) -> Row:
    """Cancel and the dialog's own answer, at the right."""
    return Row(children=[Fill(), Button(f"{window_id}.cancel", "Cancel"), Button(f"{window_id}.ok", ok)])


def path_window(window_id: str, title: str, prompt: str, path: str, ok: str, extra: list[Node] | None = None) -> Window:
    """A modal window asking for a path, with anything `extra` between the path and the buttons."""
    children: list[Node] = [Label(f"{window_id}.prompt", prompt, size=SMALL, wrap=True, tone="muted"), TextBox(f"{window_id}.path", path)]
    children += extra or []
    children.append(buttons(window_id, ok))
    return Window(window_id, title, Column(children=children))


def question_window(window_id: str, title: str, text: str, ok: str) -> Window:
    """A modal question, answered by its button or by Cancel."""
    return Window(window_id, title, Column(children=[Label(f"{window_id}.text", text, wrap=True), buttons(window_id, ok)]))


def open_window(path: str) -> FileWindow:
    return FileWindow(
        OPEN,
        "Open a recipe",
        path or ".",
        mode="load_file",
        suffixes=(".yaml", ".yml"),
        prompt="A terrain recipe (.yaml). Its problems, if any, show at the top of the form.",
        ok="Open",
    )


def save_as_window(path: str) -> FileWindow:
    return FileWindow(
        SAVE_AS,
        "Save the recipe as",
        path or "recipe.yaml",
        mode="save_file",
        suffixes=(".yaml", ".yml"),
        prompt="A .yaml file; the recipe is saved with its problems, if any.",
        ok="Save",
    )


def save_world_window(path: str) -> FileWindow:
    prompt = ("A folder for the terrain bundle: terrain.yaml (the recipe and the seed), terrain.npz (the arrays) "
              "and preview.png. Stages that are not up to date run first.")
    return FileWindow(
        SAVE_WORLD,
        "Export the world",
        path or ".",
        mode="directory",
        prompt=prompt,
        ok="Export",
    )


def overwrite_recipe_window(name: str) -> Window:
    return question_window(OVERWRITE, "Overwrite?", f"{name} exists. Saving drops the file's comments. Overwrite?", "Overwrite")


def overwrite_world_window(name: str, what: str) -> Window:
    """`what` the folder holds: "a terrain", or "files"."""
    return question_window(OVERWRITE, "Write over it?", f"{name} already holds {what}. Write the terrain over it?", "Write")


class LogRow(NamedTuple):
    """A line of a log window: its number (the end of its id, so a line keeps its id as older ones drop off), its
    text, its tone, and its tooltip (None: the text is all there is)."""

    number: int
    text: str
    tone: Tone = "ink"
    tooltip: Tooltip | None = None


def log_root(rows: Sequence[LogRow], prefix: str = "log", above: Node | None = None, follow: bool = False) -> Column:
    """The rows, newest last, over Copy (every line, to the clipboard) and Clear; `above` is what goes over them. Its
    widgets hang off `prefix` ("log.scroll", "log.copy"). A `follow` scroll stays at the newest unless scrolled up."""
    labels: list[Node] = [Label(f"{prefix}.{row.number}", row.text, size=SMALL, wrap=True, tone=row.tone,
                                tooltip=row.tooltip) for row in rows]
    return Column(children=[
        *([above] if above is not None else []),
        Scroll(f"{prefix}.scroll", Column(children=labels), weight=1, follow=follow),
        Row(children=[Fill(), Button(f"{prefix}.copy", "Copy"), Button(f"{prefix}.clear", "Clear")]),
    ])


def line_rows(lines: list[Line]) -> list[LogRow]:
    """The newest `LOG_SHOWN` lines of the log as rows, errors in the error tone."""
    shown = lines[-LOG_SHOWN:]
    first = len(lines) - len(shown)
    return [LogRow(first + i, str(line), "error" if line.source in (Source.ERR, Source.UI_ERROR) else "ink")
            for i, line in enumerate(shown)]


def log_window(lines: list[Line]) -> Window:
    """Floating beside the generator's view, which stays usable."""
    window = Window(LOG, "Log", log_root(line_rows(lines)), size=LOG_SIZE, modal=False, resizable=True,
                    minimizable=True)
    window.position = (40.0, 80.0)
    show_log(window, lines)
    return window


def show_log(window: Window, lines: list[Line]) -> None:
    """The log window's lines, as they are now, scrolled to the newest, as a terminal follows what is printed."""
    window.set_root(log_root(line_rows(lines)))
    window.state.scroll_ptr("log.scroll")[0].y = BOTTOM
