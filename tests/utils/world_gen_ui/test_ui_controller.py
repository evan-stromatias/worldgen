"""The terrain generator UI's controller acting on events, without a window."""

import copy
import dataclasses
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
import yaml

import rayscreen
from common import SEED, SMALL, inline_executor
from rayscreen import FileWindow
from rayscreen.specs import walk
from worldgen.config import DEFAULT_RECIPE
from worldgen.proc_gen.world.__main__ import main as worldgen_main
from worldgen.proc_gen.world.terrain_io import ARRAYS_FILE, PREVIEW_FILE, RECIPE_FILE, load_terrain
from worldgen.utils.world_gen_ui.controller import Controller
from worldgen.utils.world_gen_ui.model.recipe import Recipe
from worldgen.utils.world_gen_ui.model.stages import Status
from worldgen.utils.world_gen_ui.view import events as ev
from worldgen.utils.world_gen_ui.view import windows
from worldgen.utils.world_gen_ui.view.windows import OVERWRITE


@pytest.fixture
def controller(tmp_path) -> Controller:
    return Controller(Recipe.load(DEFAULT_RECIPE), 7, tmp_path / "terrain")


def test_typed_text_is_parsed_by_the_field_it_is_in(controller):
    for path, text in [
        (("elevation", "mask_falloff", 1), "0.9"),
        (("biomes", "rules", 6, "ground", "color", 0), "80"),
        (("biomes", "rules", 6, "id"), "woods"),
    ]:
        controller.act(ev.TextCommitted(path, text))
    config = controller.recipe.config
    assert config.elevation.mask_falloff == (0.55, 0.9)
    assert config.biomes.rules[6].ground.color == (80, 118, 60)
    assert config.biomes.rules[6].id == "woods"


def test_choices_and_toggles(controller):
    controller.act(ev.ValueChosen(("elevation", "mask"), "continent"))
    controller.act(ev.OptionalToggled(("biomes", "rules", 6, "when", "slope"), True))
    config = controller.recipe.config
    assert config.elevation.mask == "continent"
    assert config.biomes.rules[6].when.slope == (0.0, 0.0)


def test_reset_puts_a_stage_back_to_its_defaults(controller):
    controller.act(ev.ValueChosen(("elevation", "mask"), "continent"))
    controller.act(ev.TextCommitted(("climate", "lapse_rate"), "100"))
    controller.act(ev.ResetStage("elevation"))
    assert controller.recipe.is_default("elevation") and "elevation" in controller.message
    assert controller.recipe.config.climate.lapse_rate == 100


def test_a_bad_seed_is_refused(controller):
    controller.act(ev.SeedCommitted("12"))
    assert controller.session.seed == 12
    controller.act(ev.SeedCommitted("twelve"))
    assert controller.session.seed == 12 and controller.message_is_error


def test_save_as_asks_before_overwriting_and_open_reads_it_back(controller, tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text("# my notes\n")
    controller.act(ev.TextCommitted(("world", "size_cells"), "512"))
    controller.act(ev.RecipePathTyped(str(path)))
    controller.act(ev.SaveRecipeAs())
    assert path.read_text() == "# my notes\n" and "overwrite it?" in controller.message
    assert [w.id for w in controller.screen.windows] == [OVERWRITE]  # the question
    controller.act(ev.Overwrite())
    assert controller.recipe.path == path and not controller.message_is_error and not controller.screen.windows

    controller.act(ev.TextCommitted(("world", "size_cells"), "256"))
    controller.act(ev.OpenRecipe())
    assert controller.recipe.config.world.size_cells == 512


def test_save_as_wants_a_yaml_file(controller, tmp_path):
    controller.act(ev.RecipePathTyped(str(tmp_path / "mine.txt")))
    controller.act(ev.SaveRecipeAs())
    assert controller.message_is_error and not (tmp_path / "mine.txt").exists()


def test_opening_what_is_not_a_recipe_keeps_the_current_one(controller, tmp_path):
    before = controller.recipe
    controller.act(ev.RecipePathTyped(str(tmp_path / "missing.yaml")))
    controller.act(ev.OpenRecipe())
    assert controller.recipe is before and "cannot read" in controller.message


@pytest.fixture
def small_controller(tmp_path) -> Controller:
    """The small test recipe, with stages and map images run at once on this thread."""
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(SMALL)))
    return Controller(Recipe.load(path), SEED, tmp_path / "terrain", executor=inline_executor())


def test_run_follows_the_stages_and_draws_the_map(small_controller):
    c = small_controller
    assert c.layers() == ()
    c.act(ev.RunStage("hydrology"))
    c.update()
    assert c.layer == "hydrology" and c.layers() == ("elevation", "land mask", "hydrology")
    assert c.stage_texts()["hydrology"][1].startswith("done")
    assert c.message.startswith("ran elevation, hydrology in")
    key, job = c._drawing
    assert key[0] == "hydrology" and job.result().image.shape == (128, 128, 3)
    assert job.result().terrain.water is not None  # the 3D terrain, with the lakes and rivers

    c.act(ev.RunAll())
    c.update()
    assert c.layer == "biomes"
    assert c.stage_texts()["erosion"] == (Status.SKIPPED, "skipped: erosion off")
    assert "erosion" not in c.layers()
    c.act(ev.LayerChosen("temperature"))
    c.update()
    assert c._drawing[0][0] == "temperature"


def test_a_pending_layer_is_shown_when_the_run_settles(small_controller):
    c = small_controller
    c.pending_layer = "land mask"
    c.act(ev.RunAll())
    c.update()
    assert c.layer == "land mask"
    assert c._wanted is not None and c._wanted[0] == "land mask"


def test_a_pending_layer_without_a_result_is_ignored(small_controller):
    c = small_controller
    c.pending_layer = "erosion"  # the small recipe has erosion off
    c.act(ev.RunStage("elevation"))
    c.update()
    assert c.layer == "elevation"  # the map follows the run instead


def test_edits_and_a_new_seed_make_stages_stale(small_controller):
    c = small_controller
    c.act(ev.RunAll())
    c.update()
    c.act(ev.TextCommitted(("biomes", "blend_width"), "0.1"))
    texts = c.stage_texts()
    assert texts["climate"][0] is Status.DONE and texts["biomes"][0] is Status.STALE
    c.act(ev.SeedCommitted("11"))
    assert c.session.seed == 11 and c.stage_texts()["elevation"][0] is Status.STALE


def test_stop_all_stops_the_stage_running_and_the_run(tmp_path):
    path = tmp_path / "small.yaml"
    path.write_text(yaml.safe_dump(copy.deepcopy(SMALL)))
    c = Controller(Recipe.load(path), SEED, tmp_path / "terrain")  # stages on a worker thread, as in the app
    started = threading.Event()

    def work_forever(progress, config, seed):
        started.set()
        while True:
            time.sleep(0.001)

    c.session.stages = tuple(dataclasses.replace(s, run=work_forever) if s.name == "climate" else s for s in c.session.stages)
    c.act(ev.RunAll())
    assert started.wait(10)
    c.act(ev.StopAll())
    assert c.message == "stopping" and c.session.stopping
    deadline = time.perf_counter() + 10
    while c.session.busy and time.perf_counter() < deadline:
        c.update()
        time.sleep(0.01)
    assert c.message == "stopped; ran elevation, hydrology"
    assert not c.session.busy and c.runnable
    texts = c.stage_texts()
    assert texts["hydrology"][0] is Status.DONE and texts["climate"][0] is Status.NOT_RUN
    c.session.close()


def test_nothing_runs_while_the_recipe_has_problems(small_controller):
    c = small_controller
    c.act(ev.TextCommitted(("world", "cell_size"), "abc"))
    assert not c.runnable
    c.act(ev.RunAll())
    assert not c.session.busy and c.session.states["elevation"].progress is None


def files(folder: Path) -> dict[str, bytes]:
    return {f.relative_to(folder).as_posix(): f.read_bytes() for f in sorted(folder.rglob("*")) if f.is_file()}


def save(c: Controller, folder: Path) -> None:
    c.act(ev.WorldPathTyped(str(folder)))
    c.act(ev.SaveWorld())
    c.update()  # takes in the run, which starts the export
    c.update()  # takes in the export


def test_export_world_runs_what_is_needed_and_writes_what_the_cli_writes(small_controller, tmp_path, monkeypatch):
    c, out = small_controller, tmp_path / "saved"
    save(c, out)
    assert c.message.startswith("exported the terrain to") and not c.message_is_error
    assert c.up_to_date() and c.layer == "biomes"
    assert {RECIPE_FILE, ARRAYS_FILE, PREVIEW_FILE} == set(files(out))

    # The bundle loads back, seed and all; and opening its terrain.yaml as a recipe hints the seed
    loaded = load_terrain(out)
    assert loaded.seed == SEED and loaded.config.world.size_cells == 128
    assert Recipe.load(out / RECIPE_FILE).seed_hint == SEED

    # The CLI on the same recipe and seed writes the same arrays and preview
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text((out / RECIPE_FILE).read_text().split("\n\n", 1)[1])  # the bundle's, without its seed line
    monkeypatch.setattr(sys, "argv", ["worldgen", "--config", str(recipe), "--seed", str(SEED),
                                      "--out", str(tmp_path / "cli")])
    worldgen_main()
    ours, theirs = files(out), files(tmp_path / "cli")
    assert ours[ARRAYS_FILE] == theirs[ARRAYS_FILE] and ours[PREVIEW_FILE] == theirs[PREVIEW_FILE]


def test_export_world_asks_before_writing_into_a_folder_with_files(small_controller, tmp_path):
    c, out = small_controller, tmp_path / "busy"
    out.mkdir()
    (out / "notes.txt").write_text("mine")
    save(c, out)
    assert c.message_is_error and "write the terrain over it?" in c.message and not (out / ARRAYS_FILE).exists()
    c.act(ev.Overwrite())
    c.update()  # takes in the run, which starts the export
    c.update()  # takes in the export
    assert (out / ARRAYS_FILE).exists() and (out / "notes.txt").read_text() == "mine"
    save(c, out)  # the folder it exported to last: no question
    assert c.message.startswith("exported the terrain to")


def test_the_terrain_folder_starts_as_a_new_empty_temporary_folder(tmp_path):
    c, other = Controller(Recipe.defaults(), 7), Controller(Recipe.defaults(), 7)
    folder = Path(c.world_path)
    assert folder.is_dir() and not any(folder.iterdir())
    assert folder.parent == Path(tempfile.gettempdir()) and folder.name.startswith("worldgen-ui-")
    assert other.world_path != c.world_path  # each start gets its own
    c.act(ev.SeedCommitted("12"))
    assert Path(c.world_path) == folder  # the seed does not move it
    assert Controller(Recipe.defaults(), 7, tmp_path / "w").world_path == str(tmp_path / "w")
    folder.rmdir()
    Path(other.world_path).rmdir()


def test_without_a_file_open_and_save_as_ask_for_a_path(tmp_path):
    c = Controller(Recipe.defaults(), 7, tmp_path / "terrain")
    assert c.recipe_path == ""
    c.act(ev.SaveRecipeAs())
    assert c.message_is_error and "type a path" in c.message
    c.act(ev.OpenRecipe())
    assert c.message_is_error and "type the path" in c.message
    c.act(ev.RecipePathTyped(str(tmp_path / "mine.yaml")))
    c.act(ev.SaveRecipeAs())
    assert c.recipe.path == tmp_path / "mine.yaml" and not c.message_is_error


def test_dialogs_open_file_windows_for_open_save_as_and_export(small_controller, tmp_path):
    c = small_controller
    c.recipe_path = str(tmp_path / "custom")

    # Save as opens a FileWindow in save_file mode and auto-appends .yaml on confirm
    c.handle([rayscreen.Event("world", "save-as", None)])
    assert isinstance(c.screen.window, FileWindow)
    assert c.screen.window.id == windows.SAVE_AS and c.screen.window.mode == "save_file"
    c.handle([rayscreen.Event("world", f"{windows.SAVE_AS}.ok", None)])
    assert c.screen.window is None
    saved_recipe = (tmp_path / "custom.yaml").resolve()
    assert saved_recipe.is_file() and c.recipe.path == saved_recipe

    # Open recipe opens a FileWindow in load_file mode
    c.act(ev.TextCommitted(("world", "size_cells"), "256"))
    c.handle([rayscreen.Event("world", "open", None)])
    assert isinstance(c.screen.window, FileWindow)
    assert c.screen.window.id == windows.OPEN and c.screen.window.mode == "load_file"
    c.handle([rayscreen.Event("world", f"{windows.OPEN}.ok", None)])
    assert c.screen.window is None
    assert c.recipe.config.world.size_cells == SMALL["world"]["size_cells"]

    # Export world... opens a FileWindow in directory mode
    c.handle([rayscreen.Event("world", "save-world", None)])
    assert isinstance(c.screen.window, FileWindow)
    assert c.screen.window.id == windows.SAVE_WORLD and c.screen.window.mode == "directory"
    c.handle([rayscreen.Event("world", f"{windows.SAVE_WORLD}.cancel", None)])
    assert c.screen.window is None


def rule_ids(c: Controller) -> list[str]:
    return [rule["id"] for rule in c.recipe.data["biomes"]["rules"]]


def test_rules_are_added_before_the_catch_all_moved_and_removed(controller):
    before = rule_ids(controller)
    controller.act(ev.AddRule())
    ids = rule_ids(controller)
    assert ids == [*before[:-1], "biome_11", before[-1]] and controller.recipe.valid
    assert ("biomes", "rules", 9) in controller.toggled  # opened, to edit

    controller.act(ev.MoveRule(9, 8))
    assert rule_ids(controller)[8:10] == ["biome_11", before[8]]
    assert ("biomes", "rules", 8) in controller.toggled  # stays open as it moves
    controller.act(ev.MoveRule(8, 10))  # past the catch-all: refused
    controller.act(ev.MoveRule(10, 9))  # the catch-all: refused
    controller.act(ev.RemoveRule(10))
    assert rule_ids(controller)[-1] == before[-1] and len(rule_ids(controller)) == 11

    controller.act(ev.RemoveRule(8))
    assert rule_ids(controller) == before and controller.recipe.valid


def test_a_rule_change_makes_biomes_stale(small_controller):
    c = small_controller
    c.act(ev.RunAll())
    c.update()
    c.act(ev.MoveRule(0, 1))
    texts = c.stage_texts()
    assert texts["climate"][0] is Status.DONE and texts["biomes"][0] is Status.STALE


def test_only_a_right_click_on_the_map_tells_what_lies_there(small_controller, monkeypatch):
    c = small_controller
    c.act(ev.RunAll())
    c.update()
    centre = (64.0, 64.0)  # the middle of the small terrain, wherever the mouse is on the map
    monkeypatch.setattr(c, "_map_point", lambda x, y: centre)
    menu = c._menu_at(300.0, 300.0)
    assert menu.title == "x 64 m, z 64 m" and menu.target == centre
    names = [node.text for node in walk(menu.info) if getattr(node, "id", "").endswith(".name")]
    assert names[0] == "ground" and {"biome", "temperature", "moisture", "slope"} <= set(names)
    assert [item.id for item in menu.items] == ["copy", "centre", "whole"]
    assert c.screen.hover(300.0, 300.0) is None  # no tooltip over the map
    c.show()
    assert "x 64 m" not in c.shown.status  # nor does the status line read out what is under the mouse
