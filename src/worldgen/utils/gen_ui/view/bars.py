"""What every generator UI's bars share, as rayscreen specs: the light/dark button at the window's top right. No raylib."""

from rayscreen import MOON, SUN, Button, icon_text

THEME = "theme"  # the button's id, and its event
ICON_BUTTON = 34.0  # a button showing one of raygui's 16-pixel icons


def theme_button(dark: bool) -> Button:
    return Button(THEME, theme_text(dark), width=ICON_BUTTON)


def theme_text(dark: bool) -> str:
    """The theme button's icon: what it switches to, the sun in dark mode and the moon in light."""
    return icon_text(SUN if dark else MOON)
