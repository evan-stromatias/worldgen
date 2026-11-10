"""The world screen: the map in the left half under its bar, the recipe in the right, the status line under both.

The screen draws what the controller shows it (`Shown`) and reports events; it changes no model. The map shows no
tooltip: a right-click opens a menu of what lies there and what to do there, which is the controller's to say
(the `menu` it is given), as the game says what lies under the mouse in its world.
"""

from collections.abc import Callable

from rayscreen import PAD, Column, Menu, Picture, Screen
from worldgen.utils.world_gen_ui.view.bars import Shown, ids, refresh, screen_tree

type MenuAt = Callable[[float, float], Menu | None]  # screen pixels -> the menu for what is there


class WorldScreen(Screen):
    def __init__(self, form: Column, shown: Shown, menu: MenuAt):
        self.picture = Picture("map", fit="stretch", flipped=True, weight=1)  # a render texture the picture's size
        self._menu = menu
        self._shape = shown.shape()
        super().__init__("world", screen_tree(self.picture, form, shown), margin=PAD)
        self._nodes = ids(self.root)

    def show(self, shown: Shown, form: Column | None = None) -> None:
        """Show the model's values; a new `form`, or a change of shape, rebuilds the tree (keeping what is being
        typed and where the form is scrolled to)."""
        if form is not None or shown.shape() != self._shape:
            self._shape = shown.shape()
            self.set_root(screen_tree(self.picture, form if form is not None else self._form(), shown))
            self._nodes = ids(self.root)
        refresh(self._nodes, shown)

    def _form(self) -> Column:
        scroll = self._nodes["form"]
        return scroll.child  # Scroll > the form

    def menu(self, x: float, y: float) -> Menu | None:
        return self._menu(x, y)
