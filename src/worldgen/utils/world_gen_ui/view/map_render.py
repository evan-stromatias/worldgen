"""The terrain, rendered into a texture the size of the world screen's map `Picture`, so rayscreen sees no
difference between 2D and 3D. With `terrain_view`, the only modules of the tool that make GL calls (main thread).

- 2D: the layer's picture through the `Viewpoint`, north up, with a scale bar.
- 3D: `TerrainView`'s terrain.
- Before any stage has run: the world's outline from the recipe, and its size.
- Over every view, at the bottom left, how the mouse moves it.
"""

from __future__ import annotations

import math

import pyray as rl

from rayscreen import SMALL, TEXT, Theme
from worldgen.utils.gen_ui.model.plan_view import metres_per_pixel
from worldgen.utils.gen_ui.model.viewpoint import Viewpoint
from worldgen.utils.world_gen_ui.model.terrain import Picture
from worldgen.utils.world_gen_ui.view.terrain_view import TerrainView
from worldgen.world.gpu import GpuTexture

BACKGROUND = rl.Color(32, 36, 44, 255)
GRID = rl.Color(70, 78, 92, 255)
INK = rl.Color(175, 182, 195, 255)
EMPTY = "Nothing generated yet: press Run on a stage, or Run all."
HINTS = {False: "drag: pan · wheel: zoom · right-click: menu",
         True: "drag: pan · middle-drag or Alt-drag: turn and tilt · wheel: zoom · right-click: menu"}
SCALE_BAR = 120  # pixels the 2D scale bar may span, at most
MARGIN = 10  # pixels from the texture's edge to the scale bar and the hint
LABEL_TEXT = rl.Color(250, 246, 232, 255)
LABEL_BACKGROUND = rl.Color(20, 24, 32, 190)


class MapRender:
    def __init__(self) -> None:
        self.terrain = TerrainView()
        self.texture: GpuTexture | None = None  # the layer's picture, once uploaded
        self._target: rl.RenderTexture | None = None

    @property
    def shown(self) -> bool:
        """Whether a layer's picture is shown (rather than the outline)."""
        return self.texture is not None

    def set_picture(self, picture: Picture | None, sea_level: float = 0.0) -> None:
        """Show `picture` from now on, uploading it now; None goes back to the outline."""
        if self.texture is not None:
            self.texture.unload()
            self.texture = None
        if picture is not None:
            self.texture = GpuTexture(picture.image)
        self.terrain.set_picture(picture, sea_level)

    def draw(self, width: int, height: int, viewpoint: Viewpoint, three_d: bool, outline: float | None,
             theme: Theme) -> rl.Texture | None:
        """The map, `width` x `height` pixels: the layer in 2D or 3D, or the outline of the world the recipe
        makes."""
        if width < 2 or height < 2:
            return None
        hint = HINTS[three_d]
        if three_d and self.texture is not None:
            return self.terrain.render(width, height, viewpoint, theme, lambda: _hint(theme, hint, height))
        if self._target is None or (self._target.texture.width, self._target.texture.height) != (width, height):
            if self._target is not None:
                rl.unload_render_texture(self._target)
            self._target = rl.load_render_texture(width, height)
        rl.begin_texture_mode(self._target)
        if self.texture is not None:
            rl.clear_background(BACKGROUND)
            self._world(width, height, viewpoint, theme)
            _hint(theme, hint, height)
        else:
            rl.clear_background(BACKGROUND)
            if outline is not None:
                _outline(width, height, outline, theme)
        rl.end_texture_mode()
        return self._target.texture

    def _world(self, width: int, height: int, viewpoint: Viewpoint, theme: Theme) -> None:
        scale = 1 / metres_per_pixel(viewpoint, width, height)  # pixels a metre
        x0, y0 = width / 2 - viewpoint.centre[0] * scale, height / 2 - viewpoint.centre[1] * scale
        texture = self.texture.texture
        side = viewpoint.side * scale
        rl.draw_texture_pro(texture, rl.Rectangle(0, 0, texture.width, texture.height), rl.Rectangle(x0, y0, side, side),
                            rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.draw_rectangle_lines_ex(rl.Rectangle(x0, y0, side, side), 1, GRID)
        _scale_bar(width, height, scale, theme)

    def close(self) -> None:
        self.set_picture(None)
        self.terrain.close()
        if self._target is not None:
            rl.unload_render_texture(self._target)
            self._target = None


def bar_length(longest: float) -> float:
    """The longest of 1, 2 or 5 times a power of ten metres that is at most `longest` metres."""
    power = 10.0 ** math.floor(math.log10(longest))
    return max(step * power for step in (1, 2, 5) if step * power <= longest)


def _scale_bar(width: int, height: int, scale: float, theme: Theme) -> None:
    """A round length at the bottom right, chosen for the zoom, so sizes on the map can be judged by eye."""
    metres = bar_length(SCALE_BAR / scale)
    span = metres * scale
    right, y = width - MARGIN, height - MARGIN
    caption = f"{metres / 1000:g} km" if metres >= 1000 else f"{metres:g} m"
    theme.draw_text(caption, right - span / 2 - theme.text_width(caption, SMALL) / 2, y - 6 - SMALL, SMALL, LABEL_TEXT)
    rl.draw_rectangle_rec(rl.Rectangle(right - span, y - 2, span, 3), LABEL_TEXT)
    for x in (right - span, right - 1):
        rl.draw_rectangle_rec(rl.Rectangle(x - 1, y - 7, 3, 8), LABEL_TEXT)


def _hint(theme: Theme, hint: str, height: int) -> None:
    """How the mouse moves the map, on a dark backdrop to read on any ground."""
    y = height - MARGIN - SMALL
    rl.draw_rectangle_rec(rl.Rectangle(MARGIN - 4, y - 3, theme.text_width(hint, SMALL) + 8, SMALL + 6), LABEL_BACKGROUND)
    theme.draw_text(hint, MARGIN, y, SMALL, INK)


def _outline(width: int, height: int, side: float, theme: Theme) -> None:
    """The world's square, captioned, with how to start."""
    size = max(min(width, height) - 2 * MARGIN - 4 * SMALL, 1)
    x0, y0 = (width - size) / 2, (height - size) / 2
    rl.draw_rectangle_lines_ex(rl.Rectangle(x0, y0, size, size), 1, INK)
    caption = f"{side / 1000:g} x {side / 1000:g} km; north is up"
    theme.draw_text(caption, x0, y0 + size + 6, SMALL, INK)
    theme.draw_text(EMPTY, (width - theme.text_width(EMPTY, TEXT)) / 2, height / 2 - TEXT, TEXT, INK)
