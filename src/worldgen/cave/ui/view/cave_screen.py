"""Rayscreen interface spec for cavegen-ui: 2D plan / 3D orbit / 3D FPS views and parameter controls."""

from dataclasses import dataclass

from rayscreen import (
    PAD,
    Button,
    Column,
    Fill,
    Label,
    Node,
    Panel,
    Picture,
    Row,
    Screen,
    Slider,
    Split,
    TextBox,
    Toggle,
)

HALVES = "cave_halves"


@dataclass(slots=True)
class CaveShown:
    """Values displayed on the UI screens."""

    seed: str = "7"
    mode: str = "2d"  # "2d", "3d_orbit", "3d_fps"
    status: str = "Ready"
    readout: str = "Hover over cave for readout"
    volume_m3: str = "—"
    passable_ratio: str = "—"
    min_clearance: str = "—"
    max_clearance: str = "—"
    clearance: float = 18.0
    ceiling_mode: str = "inverted"
    min_passage: float = 2.0
    water_table: float = -2.0
    perimeter_wall: bool = True
    cutaway: bool = False


class CaveScreen(Screen):
    """The main window screen for cavegen-ui."""

    def __init__(self, shown: CaveShown):
        self.picture = Picture("cave_view", fit="stretch", flipped=True, weight=1)
        self.shown = shown
        super().__init__("cave_screen", self._build_tree(shown), margin=PAD)

    def _build_tree(self, s: CaveShown) -> Node:
        # Left side: View toolbar + Picture + Status / Readout
        mode_2d = Toggle("btn_mode_2d", "2D Plan", active=(s.mode == "2d"))
        mode_orbit = Toggle("btn_mode_orbit", "3D Orbit", active=(s.mode == "3d_orbit"))
        mode_fps = Toggle("btn_mode_fps", "3D FPS", active=(s.mode == "3d_fps"))
        cutaway_toggle = Toggle("btn_cutaway", "Cutaway Ceiling", active=s.cutaway)

        toolbar = Row(children=[mode_2d, mode_orbit, mode_fps, Fill(), cutaway_toggle])
        status_bar = Row(
            children=[
                Label("status_text", s.status, tone="muted"),
                Fill(),
                Label("readout_text", s.readout, size=13),
            ]
        )
        left = Column(children=[toolbar, self.picture, status_bar], weight=1)

        # Right side: Generation controls & Statistics
        run_btn = Button("btn_run", "Run all")
        export_btn = Button("btn_export", "Export cave")
        top_bar = Row(children=[Label("seed_label", "Seed:"), TextBox("input_seed", s.seed, weight=1), run_btn, export_btn])

        # Controls panel
        clearance_slider = Slider("slider_clearance", "Headroom", s.clearance, 4.0, 50.0)
        min_passage_slider = Slider("slider_min_passage", "Min Passage", s.min_passage, 0.5, 8.0)
        water_table_slider = Slider("slider_water_table", "Water Table", s.water_table, -10.0, 20.0)

        controls = Panel(
            title="Cavern Parameters",
            child=Column(
                children=[
                    clearance_slider,
                    min_passage_slider,
                    water_table_slider,
                ]
            ),
        )

        # Statistics panel
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

        hints = Panel(
            title="Navigation Hints",
            child=Column(
                children=[
                    Label("h1", "2D: Drag to pan, wheel to zoom.", size=13),
                    Label("h2", "3D Orbit: Drag to orbit, wheel to zoom.", size=13),
                    Label("h3", "3D FPS: WASD walk, mouse look, wheel speed.", size=13),
                ]
            ),
        )

        right = Column(children=[top_bar, controls, stats, hints], width=420.0)

        return Split(HALVES, left, right, ratio=0.7)

    def refresh(self, s: CaveShown) -> None:
        """Update screen tree when shown state changes."""
        self.shown = s
        self.set_root(self._build_tree(s))
