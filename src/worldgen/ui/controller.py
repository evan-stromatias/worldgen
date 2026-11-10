"""The only code that changes the model. Each frame, before rayscreen draws: take in what the stages and the
picture worker finished (a layer's picture), move the view with the mouse while it is the map's, render the
terrain into the map picture's texture and show the model's values on the screen. After rayscreen draws:
translate the events it returned into the tool's (`view/events.py`) and act on them.

Stages run on the session's worker. Map pictures (the 2D image and the 3D terrain) are made, and terrains
exported, on a second worker, so a large terrain never stalls the window; pictures are uploaded on the main
thread.
"""

import math
import tempfile
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

import pyray as rl

import rayscreen
from rayscreen import Column, Menu, MenuItem, ScreenManager, Theme, Window, mouse_free
from worldgen.generator.config import ConfigError
from worldgen.ui.model.recipe.form import Kind, blank, box_kind, parse, recipe_form
from worldgen.ui.model.map.layers import LAYERS, has_layer, layers_for_stage, readout
from worldgen.ui.model.log import Log, Source
from worldgen.ui.model.map.plan_view import metres_per_pixel, to_world
from worldgen.ui.model.map.readout import Readout
from worldgen.ui.model.recipe.recipe import FieldPath, Recipe, renumbered
from worldgen.ui.model.recipe.rules import RULES, can_move, can_remove, new_rule
from worldgen.ui.model.saving import is_terrain, occupied, save_terrain
from worldgen.ui.model.session import Session, StageState
from worldgen.ui.model.recipe.sizes import size_notes
from worldgen.ui.model.stages import STAGES, Status
from worldgen.ui.model.map.terrain import Picture, picture
from worldgen.ui.model.map.viewpoint import FOVY, Viewpoint
from worldgen.ui.view import events as ev
from worldgen.ui.view import windows
from worldgen.ui.view.bars import Shown
from worldgen.ui.view.form import FormOptions
from worldgen.ui.view.gl.map_render import MapRender
from worldgen.ui.view.stages import recipe_tree, rule_cards
from worldgen.ui.view.tips import facts_table
from worldgen.ui.view.windows import LOG, OVERWRITE, log_window, show_log
from worldgen.ui.view.world_screen import WorldScreen

ImageKey = tuple[str, int]  # (layer, id of the progress it pictures)
LEFT, MIDDLE = rl.MouseButton.MOUSE_BUTTON_LEFT, rl.MouseButton.MOUSE_BUTTON_MIDDLE
TURN = 0.006  # radians per pixel of a turning drag


class Controller:
    def __init__(
        self, recipe: Recipe, seed: int, out: Path | None = None, executor: Executor | None = None, log: Log | None = None,
        theme: Theme | None = None,
    ):
        """`out` is where Export world writes; without it, a new empty folder in the system's temporary folder.
        `executor` runs stages, and map images and exporting; by default each has a worker thread of its own.
        `log` is what the log window shows; the UI's messages are added to it. `theme` is the screens' (light,
        and no window needed, by default)."""
        self.log = log or Log()
        self.theme = theme or Theme()
        self.recipe = recipe
        self.session = Session(seed, executor)
        out = out or Path(tempfile.mkdtemp(prefix="worldgen-ui-"))
        self.world_path = _shown(out)  # where Export world writes
        self.form = recipe_form()
        self.layer: str | None = None  # the layer the map shows
        self.pending_layer: str | None = None  # a layer to show once nothing is running (the UI's --layer)
        self.three_d = False
        self.exaggeration = 1.0
        self.descriptions = True
        self.toggled: set[FieldPath] = set()  # sections and rules opened or closed by hand
        self.viewpoint = Viewpoint()  # kept across pictures while the world's size stays the same
        self.map = MapRender()  # no GL until it draws
        self._ran: list[str] = []  # stages done in the run going on, for the message when it ends
        self._stopped = False  # Stop all was pressed during the run going on
        self.recipe_path = _shown(recipe.path) if recipe.path else ""  # what Open reads and Save as writes
        self.message, self.message_is_error = "", False
        self._asking: tuple[str, Path] | None = None  # what the overwrite question is about: "recipe" or "world", and where
        self._titled: str | None = None  # the window's title
        self._jobs = executor or ThreadPoolExecutor(1, thread_name_prefix="map-and-save")  # map images, exporting
        self._save_to: Path | None = None  # Export world was pressed: export here once the stages are up to date
        self._saving: tuple[Path, Future] | None = None
        self._saved: Path | None = None  # the folder last exported to
        self._wanted: ImageKey | None = None  # the image the map should show
        self._drawing: tuple[ImageKey, Future] | None = None
        self._on_map: ImageKey | None = None  # the image the map shows
        self._drag: tuple[int, tuple[float, float]] | None = None  # the button held on the map, and where the mouse was
        self._log_seen = 0  # the log's `added` when it was last seen
        self._form_dirty = False
        self.manager = ScreenManager()
        self.shown = Shown()
        self.screen = WorldScreen(self._form_tree(), self.shown, self._menu_at)
        self.manager.reset(self.screen)
        self.show()

    # --- each frame, before rayscreen draws -----------------------------------------------------------

    def before_frame(self) -> None:
        """Take in finished work, move and draw the map (GL: main thread), and show the model on the screen."""
        self.update()
        self._upload()
        title = f"terrain generator - {self.recipe.path.name if self.recipe.path else 'untitled'}"
        if self._titled != title:
            self._titled = title
            rl.set_window_title(title)
        self._navigate()
        self._draw_map()
        self.show()

    def show(self) -> None:
        """The model's values onto the screen, rebuilding the form after an edit (no GL)."""
        self._refresh_shown()
        dirty, self._form_dirty = self._form_dirty, False
        self.screen.show(self.shown, self._form_tree() if dirty else None)
        window = self._window(LOG)
        if window is not None and self.log.added != self._log_seen:
            self._log_seen = self.log.added
            show_log(window, self.log.lines())

    def update(self) -> None:
        """Take in what the stages have finished, and start drawing the map image that is wanted. No GL calls."""
        ended = self.session.poll()
        for name in ended:
            state = self.session.states[name]
            if state.status is Status.FAILED:
                self._say(f"{name} failed: {state.error}", error=True)
            elif state.status is Status.DONE:
                self.layer = layers_for_stage(name)[0]  # follow the run
                self._ran.append(name)
                self.log.add(Source.UI, f"  {name:<10} {state.seconds:6.2f}s")  # as worldgen prints it
            elif state.status is Status.SKIPPED:
                self.log.add(Source.UI, f"  {name:<10} {self._skipped(name)}")
        if self._stopped and not self.session.busy:
            self._stopped = False
            if not self.message_is_error:
                self._say(f"stopped; ran {', '.join(self._ran)}" if self._ran else "stopped")
            self._ran = []
        if ended and not self.session.busy and self._ran:
            seconds = sum(self.session.states[name].seconds for name in self._ran)
            if not self.message_is_error:
                self._say(f"ran {', '.join(self._ran)} in {seconds:.1f} s")
            self._ran = []
        if self._save_to is not None and not self.session.busy:  # Export world was waiting for this run
            out, self._save_to = self._save_to, None
            if self.up_to_date():
                self._start_saving(out)
            elif not self.message_is_error:
                self._say("not saved: the recipe changed while the stages ran; press Export world again", error=True)
        if self._saving is not None:
            self._saved_or_failed()
        if self.pending_layer is not None and not self.session.busy:
            layer, self.pending_layer = self.pending_layer, None
            if layer in self.layers():
                self.layer = layer
        if self.layer is not None and self._layer_state(self.layer).progress is None:
            self.layer = None
        self._wanted = None
        if self.layer is not None:
            state = self._layer_state(self.layer)
            self._wanted = (self.layer, id(state.progress))
            if self._wanted != self._on_map and (self._drawing is None or self._drawing[0] != self._wanted):
                self._drawing = (self._wanted, self._jobs.submit(picture, self.layer, state.progress, state.config))

    def _upload(self) -> None:
        """Hand a finished picture to the map (GL: main thread only)."""
        if self._wanted is None and self._on_map is not None:
            self.map.set_picture(None)
            self._on_map = None
        if self._drawing is None or not self._drawing[1].done():
            return
        key, job = self._drawing
        self._drawing = None
        if key != self._wanted:
            return  # drawn for a layer no longer shown
        try:
            drawn: Picture = job.result()
        except Exception as e:  # noqa: BLE001 - shown, not fatal
            self._say(f"cannot draw the {key[0]} layer: {e}", error=True)
            return
        world = self._layer_state(key[0]).config.world
        self.viewpoint.fit(world.size_cells * world.cell_size)  # kept while the world's size stays the same
        self.map.set_picture(drawn, world.sea_level)
        self._on_map = key

    def _navigate(self) -> None:
        """Pan, zoom, turn and tilt while the mouse belongs to the map (not a widget, a window or a menu)."""
        rect = self.screen.picture.rect
        mouse = rl.get_mouse_position()
        inside = rect.x <= mouse.x < rect.x + rect.width and rect.y <= mouse.y < rect.y + rect.height
        free = inside and self.map.shown and mouse_free(self.manager, self.screen, self.theme)
        if free and (wheel := rl.get_mouse_wheel_move()):
            point = None if self.three_d else to_world(self.viewpoint, rect.width, rect.height, mouse.x - rect.x, mouse.y - rect.y)
            self.viewpoint.zoom_about(wheel, point)
        if free and self._drag is None:
            alt = rl.is_key_down(rl.KeyboardKey.KEY_LEFT_ALT) or rl.is_key_down(rl.KeyboardKey.KEY_RIGHT_ALT)
            if rl.is_mouse_button_pressed(MIDDLE) or (alt and rl.is_mouse_button_pressed(LEFT)):
                self._drag = (MIDDLE, (mouse.x, mouse.y))
            elif rl.is_mouse_button_pressed(LEFT):
                self._drag = (LEFT, (mouse.x, mouse.y))
        if self._drag is None:
            return
        button, (x, y) = self._drag
        if not rl.is_mouse_button_down(LEFT if button == LEFT or not rl.is_mouse_button_down(MIDDLE) else MIDDLE):
            self._drag = None
            return
        dx, dy = mouse.x - x, mouse.y - y
        self._drag = (button, (mouse.x, mouse.y))
        if button == MIDDLE and self.three_d:
            self.viewpoint.orbit(dx * TURN, dy * TURN)
        elif self.three_d:
            seen = 2 * self.viewpoint.distance() * math.tan(math.radians(FOVY) / 2)  # metres from top to bottom
            self.viewpoint.drag(dx, dy, seen / max(rect.height, 1.0))
        else:
            m = metres_per_pixel(self.viewpoint, rect.width, rect.height)
            self.viewpoint.pan(-dx * m, -dy * m)

    def _draw_map(self) -> None:
        picture = self.screen.picture
        self.map.terrain.exaggeration = self.exaggeration
        texture = self.map.draw(int(picture.rect.width), int(picture.rect.height), self.viewpoint, self.three_d,
                                self._world_extent(), self.theme)
        picture.texture = texture
        if texture is not None:
            picture.size = (texture.width, texture.height)

    def settled(self) -> bool:
        """Nothing running or saving and the map shows what it should, at its size (a resize redraws it a frame
        late): the moment for a scripted screenshot."""
        rect = self.screen.picture.rect
        return (not self.session.busy and self._save_to is None and self._saving is None
                and self._wanted == self._on_map and self._drawing is None
                and self.screen.picture.size == (int(rect.width), int(rect.height)))

    # --- what the screen shows ------------------------------------------------------------------------

    @property
    def runnable(self) -> bool:
        return self.recipe.valid and not self.session.busy and self._save_to is None and self._saving is None

    @property
    def savable(self) -> bool:
        return self.runnable and bool(self.world_path)

    def up_to_date(self) -> bool:
        """Every stage has run with the recipe and seed as they are now."""
        statuses = self.session.statuses(self.recipe.config) if self.recipe.valid else {}
        return bool(statuses) and all(s in (Status.DONE, Status.SKIPPED) for s in statuses.values())

    def layers(self) -> tuple[str, ...]:
        """The layers whose stage has a result to show."""
        return tuple(layer for layer in LAYERS if self._layer_state(layer).progress is not None
                     and self._layer_state(layer).status is not Status.SKIPPED
                     and has_layer(layer, self._layer_state(layer).progress))

    def stage_texts(self) -> dict[str, tuple[Status, str]]:
        texts = {}
        for name, status in self.session.statuses(self.recipe.config).items():
            state = self.session.states[name]
            text = {
                Status.DONE: f"done {state.seconds:.2f} s",
                Status.RUNNING: f"{'stopping' if self.session.stopping else 'running'} {self.session.elapsed():.1f} s",
                Status.SKIPPED: self._skipped(name),
            }.get(status, status.value)
            texts[name] = (status, text)
        return texts

    def _skipped(self, stage: str) -> str:
        return "skipped: " + next(s.skipped for s in self.session.stages if s.name == stage)

    def _layer_state(self, layer: str) -> StageState:
        return self.session.states[LAYERS[layer]]

    def _refresh_shown(self) -> None:
        shown = self.shown
        shown.seed = str(self.session.seed)
        shown.runnable, shown.stoppable = self.runnable, self.session.busy and not self.session.stopping
        shown.savable = self.savable
        shown.descriptions, shown.dark = self.descriptions, self.theme.dark
        shown.recipe_path, shown.world_path = self.recipe_path, self.world_path
        shown.layers, shown.layer = self.layers(), self.layer
        shown.three_d, shown.exaggeration = self.three_d, self.exaggeration
        shown.statuses = self.stage_texts()
        shown.at_defaults = {stage: self.recipe.is_default(stage) for stage in STAGES}
        shown.problems = [(where, message) for where, messages in self.recipe.problems.items() for message in messages]
        shown.running = "idle"
        if self.session.stopping:
            shown.running = f"stopping {self.session.running or 'the run'} ({self.session.elapsed():.1f} s)"
        elif self.session.running:
            shown.running = f"running {self.session.running} ({self.session.elapsed():.1f} s)"
        elif self._saving:
            shown.running = "exporting the terrain"
        if self.message:
            shown.status, shown.status_tone = self.message, "error" if self.message_is_error else "ok"
        else:
            shown.status, shown.status_tone = self._summary(), "ink"
        shown.log_news, shown.log_errors = log_news(self.log, self._log_seen if self._window(LOG) is None else self.log.added)

    def _summary(self) -> str:
        if not self.recipe.valid:
            return "the recipe has problems: see the top of the form"
        return "" if self.layers() else "no terrain generated yet"

    def _world_extent(self) -> float | None:
        """The world's side in metres, while the recipe's values make sense."""
        if self.recipe.config is None:
            return None
        world = self.recipe.config.world
        return world.size_cells * world.cell_size

    def _form_tree(self) -> Column:
        return recipe_tree(self.form, self._form_options(), self.stage_texts(), self.runnable)

    def _form_options(self) -> FormOptions:
        rules = self.recipe.get(RULES)
        ids = tuple(str(rule.get("id", "")) for rule in rules if isinstance(rule, dict)) if isinstance(rules, list) else ()
        return FormOptions(self.recipe, self.descriptions, notes=size_notes(self.recipe), choices=ids,
                           opened=lambda path, start_open: (path in self.toggled) != start_open, rules=rule_cards)

    def _map_point(self, x: float, y: float) -> tuple[float, float] | None:
        """The world point under screen pixel (x, y), or None off the world shown."""
        rect = self.screen.picture.rect
        if not self.map.shown or not (rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height):
            return None
        px, py = x - rect.x, y - rect.y
        if self.three_d:
            return self.map.terrain.pick(self.viewpoint, rect.width, rect.height, px, py)
        hit = to_world(self.viewpoint, rect.width, rect.height, px, py)
        return hit if 0 <= hit[0] < self.viewpoint.side and 0 <= hit[1] < self.viewpoint.side else None

    def _readout_at(self, x: float, y: float) -> Readout | None:
        """What lies under screen pixel (x, y) on the map, as far as the layer's stage knows."""
        point = self._map_point(x, y)
        if point is None or self.layer is None:
            return None
        state = self._layer_state(self.layer)
        return readout(state.progress, state.config, *point)

    def _menu_at(self, x: float, y: float) -> Menu | None:
        """A right-click on the map: what lies there (the facts the status line reads out), then what to do there."""
        point = self._map_point(x, y)
        if point is None:
            return None
        found = self._readout_at(x, y)
        items = [MenuItem("copy", "Copy coordinates"), MenuItem("centre", "Centre here"), MenuItem("whole", "Show the whole world")]
        title = found.title if found is not None else f"x {point[0]:.0f} m, z {point[1]:.0f} m"
        info = facts_table("menu", found.facts) if found is not None and found.facts else None
        return Menu(ev.MAP_MENU, items, title=title, target=point, info=info)

    # --- after rayscreen draws: what the user did -----------------------------------------------------

    def handle(self, events: list[rayscreen.Event]) -> None:
        """Act on what rayscreen reported this frame."""
        for event in ev.translate(rayscreen.dispatch(self.screen, events), self.form, self.recipe):
            self.act(event)

    def act(self, event: ev.Event) -> None:
        if isinstance(event, (ev.TextCommitted, ev.ValueChosen, ev.OptionalToggled)):
            self.message = ""  # news about the recipe as it was
        self._form_dirty = True
        match event:
            case ev.TextCommitted(path, text):
                spec, rest = self.form.locate(path)
                value, unreadable = parse(box_kind(spec, rest), text)
                self.recipe.set(path, value, unreadable)
            case ev.ValueChosen(path, value):
                self.recipe.set(path, value)
            case ev.OptionalToggled(path, on):
                spec, rest = self.form.locate(path)
                if on:
                    self.recipe.turn_on(path, blank(spec, rest))
                else:
                    self.recipe.turn_off(path, drop_key=bool(rest) and spec.kind is Kind.WEIGHTS)
            case ev.SeedCommitted(text):
                self._set_seed(text)
            case ev.RunStage(stage):
                self._run(stage)
            case ev.ResetStage(stage):
                self.recipe.reset(stage)
                self._say(f"{stage} is back to its defaults")
            case ev.RunAll():
                self._run(STAGES[-1])
            case ev.StopAll():
                self._stop()
            case ev.LayerChosen(layer):
                self.layer = layer
            case ev.ViewChosen(three_d):
                self.three_d = three_d
            case ev.ExaggerationChosen(factor):
                self.exaggeration = factor
            case ev.SectionToggled(path):
                self.toggled ^= {path}
            case ev.DescriptionsShown(on):
                self.descriptions = on
            case ev.ThemeSwitched():
                self.theme.set_dark(not self.theme.dark)
            case ev.RecipePathTyped(text):
                self.recipe_path = text.strip()
            case ev.OpenRecipe():
                self._open()
            case ev.SaveRecipeAs():
                self._save()
            case ev.WorldPathTyped(text):
                self.world_path = text.strip()
            case ev.SaveWorld():
                self._save_world()
            case ev.Overwrite():
                self._overwrite()
            case ev.DialogAsked(window):
                self._ask(window)
            case ev.DialogClosed(window):
                self._close(window)
                if window == OVERWRITE:
                    self._asking = None
            case ev.ToggleLog():
                self.toggle_log()
            case ev.ClearLog():
                self.log.clear()
                self._log_seen = 0
            case ev.CopyLog():
                rl.set_clipboard_text(self.log.text())
            case ev.MenuChosen(choice, point):
                self._menu_choice(choice, point)
            case ev.AddRule() | ev.RemoveRule() | ev.MoveRule():
                self.message = ""
                self._edit_rules(event)

    def _edit_rules(self, event: ev.AddRule | ev.RemoveRule | ev.MoveRule) -> None:
        """Add, remove or move a biome rule, keeping the catch-all last; rules opened in the form stay open."""
        rules = self.recipe.get(RULES)
        if not isinstance(rules, list):
            return
        match event:
            case ev.AddRule():
                rule = new_rule(rules)
                at = max(len(rules) - 1, 0)
                self._renumber(RULES, self.recipe.insert(RULES, at, rule))
                self.toggled.add((*RULES, at))  # open, to edit its `when`
                self._say(f"added {rule['id']} before the catch-all; set where it applies under `when`")
            case ev.RemoveRule(index) if can_remove(rules, index):
                name = rules[index].get("id") if isinstance(rules[index], dict) else None
                self._renumber(RULES, self.recipe.remove(RULES, index))
                self._say(f"removed {name}" if name else "removed the rule")
            case ev.MoveRule(index, to) if can_move(rules, index, to):
                self._renumber(RULES, self.recipe.move(RULES, index, to))

    def _renumber(self, list_path: FieldPath, order: dict[int, int | None]) -> None:
        """Keep sections opened by hand open as the items of the list at `list_path` move (old index -> new)."""
        self.toggled = {moved for path in self.toggled if (moved := renumbered(path, list_path, order)) is not None}

    def _run(self, stage: str) -> None:
        if not self.runnable:
            return
        plan = self.session.run(stage, self.recipe.config)
        self._ran = []
        self._say(f"running {', '.join(plan)}")

    def _stop(self) -> None:
        if not self.session.stop():
            return
        self._stopped = True
        if self._save_to is not None:
            self._save_to = None
            self._say("stopping; the terrain will not be exported")
        else:
            self._say("stopping")

    def _set_seed(self, text: str) -> None:
        try:
            seed = int(text.strip())
        except ValueError:
            seed = -1
        if seed < 0:
            self._say(f"the seed must be a whole number, 0 or more, not {text.strip()!r}", error=True)
        else:
            self.session.seed = seed

    def _menu_choice(self, choice: str, point: tuple[float, float]) -> None:
        x, z = point
        match choice:
            case "copy":
                rl.set_clipboard_text(f"{x:.0f},{z:.0f}")
                self._say(f"copied {x:.0f},{z:.0f}")
            case "centre":
                self.viewpoint.centre = point
            case "whole":
                self.viewpoint.zoom, self.viewpoint.centre = 1.0, (self.viewpoint.side / 2, self.viewpoint.side / 2)

    # --- files and windows ----------------------------------------------------------------------------

    def _window(self, window_id: str) -> Window | None:
        return next((w for w in self.screen.windows if w.id == window_id), None)

    def _close(self, window_id: str) -> None:
        if (window := self._window(window_id)) is not None:
            self.screen.close_window(window)

    def _ask(self, window_id: str) -> None:
        """Open a dialog with the path it is about."""
        make = {windows.OPEN: lambda: windows.open_window(self.recipe_path),
                windows.SAVE_AS: lambda: windows.save_as_window(self.recipe_path),
                windows.SAVE_WORLD: lambda: windows.save_world_window(self.world_path)}
        if self._window(window_id) is None:
            self.screen.open(make[window_id]())

    def toggle_log(self) -> None:
        if (window := self._window(LOG)) is not None:
            self.screen.close_window(window)
            return
        self._log_seen = self.log.added
        self.screen.open(log_window(self.log.lines()))

    def _open(self) -> None:
        if not self.recipe_path:
            self._say("type the path of a recipe, then press Open", error=True)
            return
        path = Path(self.recipe_path).expanduser()
        try:
            recipe = Recipe.load(path)
        except ConfigError as e:
            self._say(str(e).splitlines()[0], error=True)
            return
        self.recipe = recipe
        if recipe.seed_hint is not None:
            self.session.seed = recipe.seed_hint  # a terrain bundle's terrain.yaml brings its seed along
        self.toggled.clear()
        self._close(windows.OPEN)
        problems = sum(map(len, recipe.problems.values()))
        self._say(f"opened {_shown(path)}" + (f", which has {problems} problems" if problems else ""), error=bool(problems))

    def _save(self, overwrite: bool = False) -> None:
        if not self.recipe_path:
            self._say("type a path for the recipe (a .yaml file), then press Save", error=True)
            return
        path = Path(self.recipe_path).expanduser()
        if path.suffix.lower() not in (".yaml", ".yml"):
            self._say(f"a recipe is a .yaml file, not {path.name}", error=True)
            return
        if path.exists() and not overwrite:
            self._asking = ("recipe", path)
            self.screen.open(windows.overwrite_recipe_window(path.name))
            self._say(f"{_shown(path)} exists, and saving drops its comments: overwrite it?", error=True)
            return
        try:
            self.recipe.save(path)
        except OSError as e:
            self._say(f"cannot write {path}: {e.strerror}", error=True)
            return
        self.recipe_path = _shown(path)
        self._close(windows.SAVE_AS)
        problems = "" if self.recipe.valid else " (with its problems; it will not generate until they are fixed)"
        self._say(f"saved {_shown(path)}{problems}")

    def _overwrite(self) -> None:
        """The overwrite question was answered yes: save what it was about."""
        asking, self._asking = self._asking, None
        self._close(OVERWRITE)
        if asking is None:
            return
        what, path = asking
        if what == "recipe":
            self.recipe_path = str(path)
            self._save(overwrite=True)
        else:
            self.world_path = str(path)
            self._save_world(overwrite=True)

    def _save_world(self, overwrite: bool = False) -> None:
        """Export the terrain bundle to the terrain folder, once every stage is up to date: those that are
        not run first."""
        if not self.savable:
            return
        out = Path(self.world_path).expanduser()
        if occupied(out) and out != self._saved and not overwrite:
            self._asking = ("world", out)
            what = "a terrain" if is_terrain(out) else "files"
            self.screen.open(windows.overwrite_world_window(_shown(out), what))
            self._say(f"{_shown(out)} already holds {what}: write the terrain over it?", error=True)
            return
        self._close(windows.SAVE_WORLD)
        if self.up_to_date():
            self._start_saving(out)
        else:
            self._run(STAGES[-1])
            self._save_to = out
            self._say(f"{self.message}, then exporting to {_shown(out)}")

    def _start_saving(self, out: Path) -> None:
        terrain = self.session.states[STAGES[-1]].progress.world
        self._saving = (out, self._jobs.submit(save_terrain, terrain, self.recipe.text(), out))
        self._say(f"exporting the terrain to {_shown(out)}")

    def _saved_or_failed(self) -> None:
        """Take in an export that has finished, if one has."""
        out, job = self._saving
        if not job.done():
            return
        self._saving = None
        try:
            job.result()
        except Exception as e:  # noqa: BLE001 - shown, not fatal
            self._say(f"cannot export the terrain to {_shown(out)}: {e}", error=True)
            return
        self._saved = out
        self._say(f"exported the terrain to {_shown(out)}: terrain.yaml, terrain.npz, preview.png")

    def _say(self, message: str, error: bool = False) -> None:
        """Put `message` in the status line, and in the log."""
        self.message, self.message_is_error = message, error
        self.log.add(Source.UI_ERROR if error else Source.UI, message)

    def close(self) -> None:
        """Stop the workers and free the map's textures (GL: before the window closes)."""
        self.session.close()
        if isinstance(self._jobs, ThreadPoolExecutor):
            self._jobs.shutdown(wait=False, cancel_futures=True)  # an export already going finishes first
        self.map.close()


def log_news(log: Log, seen: int) -> tuple[str, bool]:
    """What the log has had since `seen` (its `added` when last seen), and whether errors came: the Log button's news."""
    new = log.added - seen
    unseen = log.lines()[-new:] if new > 0 else []
    errors = sum(line.source in (Source.ERR, Source.UI_ERROR) for line in unseen)
    if errors:
        return f"{errors} new error{'s' if errors > 1 else ''}", True
    return (f"{len(unseen)} new" if unseen else ""), False


def _shown(path: Path) -> str:
    """`path` relative to the working directory when it is under it, as the user would type it."""
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)
