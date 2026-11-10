"""The tool's dialogs, as rayscreen `Window`s: `FileWindow`s for Open and Save as for the recipe and Export world
for the terrain's folder, plus the questions before writing over a file or a folder. The log window is `gen_ui`'s.
No raylib.
"""

from rayscreen import FileWindow, Window
from worldgen.utils.gen_ui.view.windows import OVERWRITE, question_window

OPEN, SAVE_AS, SAVE_WORLD = "dialog.open", "dialog.save-as", "dialog.save-world"
OPEN_PATH, SAVE_AS_PATH, SAVE_WORLD_PATH = f"{OPEN}.path", f"{SAVE_AS}.path", f"{SAVE_WORLD}.path"


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
