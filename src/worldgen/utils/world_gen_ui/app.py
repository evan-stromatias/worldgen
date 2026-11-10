"""The window and its loop, as rayscreen's README has a game do it: the controller renders its view into its
texture, rayscreen draws the screens over it, then the controller takes the events rayscreen returns."""

from pathlib import Path
from typing import Protocol

import pyray as rl

from rayscreen import HIGHDPI, Event, ScreenManager, Theme, follow_wayland_mouse, frame

SCREENSHOT_FRAMES = 3  # frames drawn before a scripted screenshot, so the layout has settled
# The frame cap while the user acts, and otherwise. rayscreen lays the screens out every frame (about 10 ms
# for the world screen, in Python), holding the GIL the generator's workers need: idle frames are kept few
FPS, IDLE_FPS = 60, 12
QUIET = 0.5  # seconds after the last input before the cap drops


class Controller(Protocol):
    """What the loop drives: each frame, `before_frame` (take in finished work, render the view, show the model),
    rayscreen's frame over `manager`, then `handle` with its events; `settled` says when a scripted screenshot
    may be taken."""

    manager: ScreenManager

    def before_frame(self) -> None: ...

    def handle(self, events: list[Event]) -> None: ...

    def settled(self) -> bool: ...

    def close(self) -> None: ...


def open_window(width: int, height: int, title: str, minimum: tuple[int, int], maximized: bool = False) -> Theme:
    """Open the window, at least `minimum` pixels, and set rayscreen's style; returns the theme the screens draw with."""
    rl.set_trace_log_level(rl.TraceLogLevel.LOG_WARNING)
    # HIGHDPI and the mouse fix in `run`: under display scaling clicks land where they aim (rayscreen.display)
    flags = rl.ConfigFlags.FLAG_WINDOW_RESIZABLE | rl.ConfigFlags.FLAG_MSAA_4X_HINT | HIGHDPI
    if maximized:
        flags |= rl.ConfigFlags.FLAG_WINDOW_MAXIMIZED
    rl.set_config_flags(flags)
    rl.init_window(width, height, title)
    rl.set_window_min_size(*minimum)
    if maximized:
        rl.maximize_window()
    rl.set_target_fps(FPS)
    rl.set_exit_key(rl.KeyboardKey.KEY_NULL)  # Esc closes a window or leaves a text box, never the app
    theme = Theme()
    theme.apply_style()
    return theme


def run(controller: Controller, theme: Theme, screenshot: Path | None = None) -> None:
    """Until the window closes; with `screenshot`, save a frame once the controller has settled (the terrain
    drawn, the stages run), then stop. The frame rate drops while the user does nothing, to leave the workers
    the GIL."""
    frames, last_input = 0, 0.0
    while not rl.window_should_close():
        follow_wayland_mouse()
        rl.begin_drawing()
        rl.clear_background(theme.background)
        controller.before_frame()
        controller.handle(frame(controller.manager, theme))
        frames += 1
        if screenshot is not None and frames >= SCREENSHOT_FRAMES and controller.settled():
            rl.rl_draw_render_batch_active()  # flush, or the image misses what is still batched
            image = rl.load_image_from_screen()
            rl.export_image(image, str(screenshot))
            rl.unload_image(image)
            rl.end_drawing()
            break
        rl.end_drawing()
        if _input(controller):
            last_input = rl.get_time()
        rl.set_target_fps(FPS if rl.get_time() - last_input < QUIET and screenshot is None else IDLE_FPS)


def _input(controller: Controller) -> bool:
    """The user moved the mouse, scrolled or held a button this frame, or is typing in a box. (Keys are not
    polled: raylib's key and character queues are raygui's to read.)"""
    animating = getattr(controller, "animating", False)
    if callable(animating):
        animating = animating()
    if animating:
        return True
    delta = rl.get_mouse_delta()
    buttons = any(rl.is_mouse_button_down(button) for button in (rl.MouseButton.MOUSE_BUTTON_LEFT, rl.MouseButton.MOUSE_BUTTON_RIGHT,
                                                                  rl.MouseButton.MOUSE_BUTTON_MIDDLE))
    screen = controller.manager.top
    typing = screen is not None and (screen.state.editing is not None or any(w.state.editing is not None for w in screen.windows))
    return bool(delta.x or delta.y or rl.get_mouse_wheel_move() or buttons or typing)


def close(controller: Controller, theme: Theme) -> None:
    controller.close()
    theme.unload()
    rl.close_window()
