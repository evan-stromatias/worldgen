"""The world screen's bars as rayscreen specs: the map's (layer, 2D or 3D, the 3D heights), the recipe's (seed,
Run all and Stop all, the recipe's file, the terrain's folder, the light/dark button at the top right), the
problems, the status line; and `Shown`, the model's values they display. No raylib.
"""

from dataclasses import dataclass, field

from rayscreen import (
    GAP,
    MOON,
    SMALL,
    SUN,
    Button,
    CheckBox,
    Column,
    Dropdown,
    Fill,
    Label,
    Node,
    Panel,
    Picture,
    Row,
    Scroll,
    Span,
    Split,
    Table,
    Text,
    TextBox,
    Toggle,
    Tone,
    icon_text,
)
from rayscreen.specs import walk
from worldgen.ui.model.map.layers import LAYERS
from worldgen.ui.model.stages import Status
from worldgen.ui.view.stages import TONES, stage_ids

HALVES = "halves"  # the split between the map and the recipe
HALVES_MINIMUM = (320.0, 560.0)  # pixels: the map, and the recipe's bars
NO_LAYER = "none yet"
EXAGGERATIONS = (1.0, 2.0, 4.0)  # 3D heights, times the real ones
LISTED_PROBLEMS = 8  # in the Problems panel; the rest are counted
THEME = "theme"  # the theme button's id, and its event
ICON_BUTTON = 34.0  # a button showing one of raygui's 16-pixel icons


def theme_button(dark: bool) -> Button:
    return Button(THEME, theme_text(dark), width=ICON_BUTTON)


def theme_text(dark: bool) -> str:
    """The theme button's icon: what it switches to, the sun in dark mode and the moon in light."""
    return icon_text(SUN if dark else MOON)


@dataclass(slots=True)
class Shown:
    """What the world screen shows from the model, refreshed by the controller every frame. The screen keeps no
    model of its own: it draws this and reports events."""

    seed: str = "7"
    runnable: bool = False  # stages may run now
    stoppable: bool = False  # a run is going on that may be stopped
    savable: bool = False  # a terrain may be exported now
    descriptions: bool = True
    dark: bool = True
    recipe_path: str = ""
    world_path: str = ""
    layers: tuple[str, ...] = ()  # the layers whose stage has a result
    layer: str | None = None
    three_d: bool = False
    exaggeration: float = 1.0
    statuses: dict[str, tuple[Status, str]] = field(default_factory=dict)
    at_defaults: dict[str, bool] = field(default_factory=dict)  # by stage: its section is as the defaults have it
    problems: list[tuple[str, str]] = field(default_factory=list)  # every problem in the recipe: (YAML path, message)
    status: str = ""  # the latest news
    status_tone: Tone = "ink"
    running: str = ""
    log_news: str = ""  # lines the log has had since it was last seen
    log_errors: bool = False

    def shape(self) -> tuple:
        """What changes the tree's shape (rather than a text in it): the screen rebuilds when it changes."""
        return self.layers, self.three_d, tuple(self.problems)


def map_bar(shown: Shown) -> Row:
    """Which layer, 2D or 3D, and in 3D how many times their height the hills are drawn."""
    choices = tuple(name for name in LAYERS if name in shown.layers) or (NO_LAYER,)
    children: list[Node] = [
        Label("layer.label", "Layer"),
        Dropdown("layer", choices, choices.index(shown.layer) if shown.layer in choices else 0),
        Toggle("2d", "2D", not shown.three_d),
        Toggle("3d", "3D", shown.three_d),
    ]
    if shown.three_d:
        heights = tuple(f"x{factor:g}" for factor in EXAGGERATIONS)
        children += [Label("height.label", "Height"), Dropdown("height", heights, _exaggeration(shown.exaggeration))]
    return Row(children=children)


def seed_bar(shown: Shown) -> Row:
    """The seed, running every stage or stopping the run, the descriptions, and at the top right light or dark."""
    return Row(children=[
        Label("seed.label", "Seed"),
        TextBox("seed", shown.seed, weight=1),
        Button("run-all", "#131#Run all", enabled=shown.runnable),
        Button("stop-all", "#133#Stop all", enabled=shown.stoppable),
        CheckBox("descriptions", "descriptions", shown.descriptions),
        theme_button(shown.dark),
    ])


def files_bars(shown: Shown) -> list[Node]:
    """The recipe's file with Open and Save as, and the terrain's folder with Export world."""
    return [
        Row(children=[
            Label("recipe-file.label", "Recipe"),
            Label("recipe-path", _recipe_path(shown.recipe_path), size=SMALL, tone="muted", weight=1, elide=True),
            Button("open", "#1#Open..."),
            Button("save-as", "#2#Save as..."),
        ]),
        Row(children=[
            Label("world-folder.label", "Terrain"),
            Label("world-path", shown.world_path, size=SMALL, tone="muted", weight=1, elide=True),
            Button("save-world", "#2#Export world...", enabled=shown.savable),
        ]),
    ]


def problems_panel(problems: list[tuple[str, str]]) -> Panel:
    """Every problem in the recipe, so none goes unseen in a closed section: the first few as a list, each its YAML
    path muted and its message in ink, then how many more."""
    count = len(problems)
    rows = [
        [Label(f"problems.{i}.bullet", "–", height=SMALL + 4, size=SMALL, tone="muted"),
         Text.of(f"problems.{i}", *([Span(f"{where}: ", "muted")] if where else []), message, size=SMALL)]
        for i, (where, message) in enumerate(problems[:LISTED_PROBLEMS])
    ]
    children: list[Node] = [
        Label("problems.count", f"{count} problem{'s' if count > 1 else ''}: the stages cannot run until they are fixed", tone="error"),
        Table(rows, gap=GAP),
    ]
    if count > LISTED_PROBLEMS:
        children.append(Label("problems.more", f"and {count - LISTED_PROBLEMS} more", size=SMALL, tone="muted"))
    return Panel(Column(children=children, gap=2), title="Problems")


def status_line(shown: Shown) -> Row:
    """The log's button and its news, what is running, then the latest news, last as the
    longest."""
    return Row(children=[
        Button("log", "#15#Log"),  # raygui's info icon
        Label("log-news", shown.log_news, size=SMALL, tone="error" if shown.log_errors else "muted"),
        Fill(weight=0, size=2 * GAP),
        Label("running", shown.running, size=SMALL, tone="muted"),
        Fill(weight=0, size=2 * GAP),
        Label("status", shown.status, tone=shown.status_tone, weight=1),
    ])


def screen_tree(picture: Picture, form: Column, shown: Shown) -> Column:
    """The map and the recipe side by side over the status line, the bar between them dragged to share the width."""
    right: list[Node] = [seed_bar(shown), *files_bars(shown)]
    if shown.problems:
        right.append(problems_panel(shown.problems))
    right.append(Scroll("form", form, weight=1))
    return Column(children=[
        Split(HALVES, Column(children=[map_bar(shown), picture], weight=1), Column(children=right, weight=1),
              minimum=HALVES_MINIMUM, weight=1),
        status_line(shown),
    ])


def refresh(nodes: dict[str, Node], shown: Shown) -> None:
    """Set the texts and states that change without changing the tree's shape."""
    if isinstance(seed := nodes.get("seed"), TextBox):
        seed.text = shown.seed
    for node_id, enabled in (("run-all", shown.runnable), ("stop-all", shown.stoppable), ("save-world", shown.savable)):
        if isinstance(button := nodes.get(node_id), Button):
            button.enabled = enabled
    if isinstance(descriptions := nodes.get("descriptions"), CheckBox):
        descriptions.checked = shown.descriptions
    if isinstance(theme := nodes.get("theme"), Button):
        theme.text = theme_text(shown.dark)
    for node_id, text in (("recipe-path", _recipe_path(shown.recipe_path)), ("world-path", shown.world_path)):
        if isinstance(label := nodes.get(node_id), Label):
            label.text = text
    if isinstance(layer := nodes.get("layer"), Dropdown) and shown.layer in layer.choices:
        layer.current = layer.choices.index(shown.layer)
    for node_id, active in (("2d", not shown.three_d), ("3d", shown.three_d)):
        if isinstance(toggle := nodes.get(node_id), Toggle):
            toggle.active = active
    if isinstance(height := nodes.get("height"), Dropdown):
        height.current = _exaggeration(shown.exaggeration)
    for stage, (status, text) in shown.statuses.items():
        status_id, reset_id, run_id = stage_ids(stage)
        if isinstance(label := nodes.get(status_id), Label):
            label.text, label.tone = text, TONES[status]
        if isinstance(reset := nodes.get(reset_id), Button):
            reset.enabled = not shown.at_defaults.get(stage, True)
        if isinstance(run := nodes.get(run_id), Button):
            run.enabled = shown.runnable
    for node_id, text, tone in (("status", shown.status, shown.status_tone), ("running", shown.running, "muted"),
                                ("log-news", shown.log_news, "error" if shown.log_errors else "muted")):
        if isinstance(label := nodes.get(node_id), Label):
            label.text, label.tone = text, tone


def ids(root: Node) -> dict[str, Node]:
    return {node_id: node for node in walk(root) if (node_id := getattr(node, "id", None)) is not None}


def _exaggeration(factor: float) -> int:
    return EXAGGERATIONS.index(factor) if factor in EXAGGERATIONS else 0


def _recipe_path(path: str) -> str:
    return path or "the defaults (no file until Save as)"
