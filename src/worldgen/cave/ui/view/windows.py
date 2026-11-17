"""Windows for cavegen-ui: FileWindow for exporting caves, and log window."""

from pathlib import Path

from rayscreen import Dropdown, FileWindow, Label, Row
from worldgen.cave.export import CaveExportFormat, bundle_files
from worldgen.ui.view.windows import (
    FORMAT_CHOICES,
    LOG,
    SAVE_WORLD,
    SAVE_WORLD_FORMAT,
    SAVE_WORLD_PATH,
    log_window,
    show_log,
)

__all__ = [
    "FORMAT_CHOICES",
    "LOG",
    "SAVE_WORLD",
    "SAVE_WORLD_FORMAT",
    "SAVE_WORLD_PATH",
    "log_window",
    "save_cave_prompt",
    "save_cave_window",
    "show_cave_format",
    "show_log",
]


def save_cave_prompt(format: CaveExportFormat) -> str:
    files = ", ".join(bundle_files(format))
    return f"A folder for the cave bundle: {files}. Stages that are not up to date run first."


def save_cave_window(path: str | Path, format: CaveExportFormat = "npz") -> FileWindow:
    """A folder for the cave bundle, and its format."""
    choices = tuple(FORMAT_CHOICES.values())
    return FileWindow(
        SAVE_WORLD,
        "Export the world",
        str(path) or ".",
        mode="directory",
        prompt=save_cave_prompt(format),
        ok="Export",
        extra=[
            Row(
                children=[
                    Label(f"{SAVE_WORLD_FORMAT}.label", "Format"),
                    Dropdown(
                        SAVE_WORLD_FORMAT,
                        choices,
                        choices.index(FORMAT_CHOICES[format]),
                        weight=1,
                    ),
                ]
            )
        ],
    )


def show_cave_format(window: FileWindow, format: CaveExportFormat) -> None:
    """Update prompt when format dropdown changes."""
    window.prompt = save_cave_prompt(format)
    window.refresh()
