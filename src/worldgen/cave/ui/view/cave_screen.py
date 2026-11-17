"""Rayscreen interface spec for cavegen-ui: interactive dual-pane view (viewport on left, stage form on right)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rayscreen import (
    MOON,
    PAD,
    SMALL,
    SUN,
    Button,
    Column,
    Fill,
    Label,
    Node,
    Panel,
    Picture,
    Row,
    Screen,
    Scroll,
    Split,
    TextBox,
    Toggle,
    icon_text,
)
from worldgen.ui.model.stages import Status

if TYPE_CHECKING:
    pass

HALVES = "cave_halves"
THEME = "theme"
ICON_BUTTON = 34.0


def theme_button(dark: bool) -> Button:
    return Button(THEME, theme_text(dark), width=ICON_BUTTON)


def theme_text(dark: bool) -> str:
    """The theme button's icon: what it switches to, the sun in dark mode and the moon in light."""
    return icon_text(SUN if dark else MOON)


@dataclass(slots=True)
class CaveShown:
    """Values displayed on the UI screens."""

    seed: str = "7"
    mode: str = "2d"  # "2d", "3d_orbit", "3d_fps"
    cutaway: bool = False
    status: str = "Ready"
    readout: str = "Hover over cave for readout"
    runnable: bool = True
    stoppable: bool = False
    volume_m3: str = "—"
    passable_ratio: str = "—"
    min_clearance: str = "—"
    max_clearance: str = "—"
    clearance: float = 40.0
    ceiling_mode: str = "inverted"
    min_passage: float = 3.0
    water_table: float = -5.0
    perimeter_wall: bool = True
    statuses: dict[str, tuple[Status, str]] = field(default_factory=dict)
    log_news: str = ""
    log_errors: bool = False
    dark: bool = True


class CaveScreen(Screen):
    """The main window screen for cavegen-ui."""

    def __init__(self, shown: CaveShown, form: Column | None = None):
        self.picture = Picture("cave_view", fit="stretch", flipped=True, weight=1)
        self.shown = shown
        self._form = form or Column(children=[])
        super().__init__("cave_screen", self._build_tree(shown, self._form), margin=PAD)

    def refresh(self, s: CaveShown, form: Column | None = None) -> None:
        """Update shown state and optional form tree."""
        if form is not None:
            self._form = form
        self.shown = s
        self.set_root(self._build_tree(s, self._form))

    def _build_tree(self, s: CaveShown, form: Column) -> Node:
        # Left side: View toolbar + Picture + Status / Readout
        mode_2d = Toggle("btn_mode_2d", "2D Plan", active=(s.mode == "2d"))
        mode_orbit = Toggle("btn_mode_orbit", "3D Orbit", active=(s.mode == "3d_orbit"))
        mode_fps = Toggle("btn_mode_fps", "3D FPS", active=(s.mode == "3d_fps"))
        cutaway_toggle = Toggle("btn_cutaway", "Cutaway Ceiling", active=s.cutaway)

        toolbar = Row(children=[mode_2d, mode_orbit, mode_fps, Fill(), cutaway_toggle])
        status_bar = Row(
            children=[
                Button("log", "#15#Log"),
                Label("log-news", s.log_news, size=SMALL, tone="error" if s.log_errors else "muted"),
                Fill(weight=0, size=8.0),
                Label("status_text", s.status, tone="muted"),
                Fill(),
                Label("readout_text", s.readout, size=13),
            ]
        )
        left = Column(children=[toolbar, self.picture, status_bar], weight=1)

        # Right side: Top action bar (Seed, Run all, Stop, Export cave, Theme toggle) + Scrollable Stage Form + Collapsible Metrics
        seed_box = TextBox("input_seed", s.seed, weight=1.0)
        run_all_btn = Button("run-all", "#131#Run all", enabled=s.runnable)
        stop_btn = Button("stop-all", "Stop", enabled=s.stoppable)
        export_btn = Button("btn_export", "Export cave...")
        theme_btn = theme_button(s.dark)
        top_bar = Row(children=[Label("seed_label", "Seed:"), seed_box, run_all_btn, stop_btn, export_btn, theme_btn])

        stats = Panel(
            title="Cavern Metrics",
            child=Column(
                children=[
                    Row(children=[Label("lbl_v", "Estimated Volume:"), Fill(), Label("val_v", s.volume_m3)]),
                    Row(children=[Label("lbl_p", "Passable Cavern:"), Fill(), Label("val_p", s.passable_ratio)]),
                    Row(children=[Label("lbl_min_c", "Min Clearance:"), Fill(), Label("val_min_c", s.min_clearance)]),
                    Row(children=[Label("lbl_max_c", "Max Clearance:"), Fill(), Label("val_max_c", s.max_clearance)]),
                ]
            ),
        )

        scroll_form = Scroll("cave_form_scroll", form, weight=1)
        right = Column(children=[top_bar, scroll_form, stats], weight=1)

        return Split(HALVES, left, right, minimum=(320.0, 360.0), weight=1)
