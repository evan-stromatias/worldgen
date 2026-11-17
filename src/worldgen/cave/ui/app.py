"""Application loop for cavegen-ui, delegating window management and frame timing to rayscreen."""

from worldgen.cave.ui.controller import CaveController
from worldgen.ui.app import close, open_window, run

__all__ = ["CaveController", "close", "open_window", "run"]
