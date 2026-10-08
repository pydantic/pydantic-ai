"""`LocalComputer`: this machine's display, mouse, and keyboard.

External assumptions, last verified 2026-10 against the installed packages:

- `mss.MSS().monitors[0]` is the union of all displays and `[1]` the primary one, in the
  coordinate space the OS uses for pointer positions (points on macOS, pixels elsewhere).
  `grab` may return more pixels than the monitor size on a high-DPI display, so the
  screenshot is always resized to the computed model space. Source:
  <https://python-mss.readthedocs.io/api.html>.
- `pynput` positions the pointer in that same OS coordinate space, presses keys by `Key`
  member name, and needs an X server on Linux (Wayland is not supported). Its backend is
  chosen at import time and raises `ImportError` without a display, so it is imported on
  first use rather than with this module. Source: <https://pynput.readthedocs.io/en/latest/limitations.html>.
- macOS gates screen capture behind Screen Recording and synthetic input behind
  Accessibility, granted per terminal app. Without them capture returns only the wallpaper
  and input is dropped, with no error, so both are checked up front through
  `CGPreflightScreenCaptureAccess` and `AXIsProcessTrusted` (pyobjc, a pynput dependency on
  macOS).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

from anyio import to_thread

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.computer_use._computer import ComputerError, MouseButton, ScrollDirection

try:
    import mss
    from PIL import Image
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'mss and Pillow are required for LocalComputer. '
        'Install them with: pip install "pydantic-ai-harness[computer-use]"'
    ) from _import_error

if importlib.util.find_spec('pynput') is None:  # pragma: no cover
    raise ImportError(
        'pynput is required for LocalComputer. Install it with: pip install "pydantic-ai-harness[computer-use]"'
    )

if TYPE_CHECKING:
    from pynput.keyboard import Key, KeyCode
    from pynput.mouse import Controller as MouseController

_KEY_NAMES: dict[str, str] = {
    'ctrl': 'ctrl',
    'control': 'ctrl',
    'shift': 'shift',
    'alt': 'alt',
    'option': 'alt',
    'opt': 'alt',
    'cmd': 'cmd',
    'command': 'cmd',
    'meta': 'cmd',
    'super': 'cmd',
    'win': 'cmd',
    'windows': 'cmd',
    'enter': 'enter',
    'return': 'enter',
    'tab': 'tab',
    'esc': 'esc',
    'escape': 'esc',
    'space': 'space',
    'backspace': 'backspace',
    'delete': 'delete',
    'del': 'delete',
    'home': 'home',
    'end': 'end',
    'pageup': 'page_up',
    'pgup': 'page_up',
    'pagedown': 'page_down',
    'pgdn': 'page_down',
    'up': 'up',
    'arrowup': 'up',
    'down': 'down',
    'arrowdown': 'down',
    'left': 'left',
    'arrowleft': 'left',
    'right': 'right',
    'arrowright': 'right',
    'capslock': 'caps_lock',
    'insert': 'insert',
    'printscreen': 'print_screen',
    'menu': 'menu',
    **{f'f{number}': f'f{number}' for number in range(1, 21)},
}

_DRAG_STEPS = 10
_POINTER_SETTLE_SECONDS = 0.05


@dataclass(frozen=True)
class _Geometry:
    """A monitor's position in OS coordinates and the size screenshots are scaled to."""

    left: int
    top: int
    width: int
    height: int
    scaled_width: int
    scaled_height: int

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        if x >= self.scaled_width or y >= self.scaled_height:
            raise ComputerError(
                f'({x}, {y}) is outside the {self.scaled_width}x{self.scaled_height} screenshot. '
                'Use coordinates from the latest screenshot.'
            )
        return (
            self.left + round(x * self.width / self.scaled_width),
            self.top + round(y * self.height / self.scaled_height),
        )


def _split_chord(keys: Sequence[str]) -> list[str]:
    """Accept `['ctrl+c']` as well as `['ctrl', 'c']`; a lone or trailing `+` is the plus key (`'ctrl++'`)."""
    split: list[str] = []
    for key in keys:
        if len(key) > 1 and '+' in key:
            split.extend(part for part in key.split('+') if part)
            if key.endswith('++'):
                split.append('+')
        else:
            split.append(key)
    return split


def _resolve_key(name: str, *, in_chord: bool) -> Key | KeyCode:
    from pynput.keyboard import Key, KeyCode

    if len(name) == 1:
        return KeyCode.from_char(name.lower() if in_chord else name)
    normalized = name.lower().replace(' ', '').replace('_', '').replace('-', '')
    member = _KEY_NAMES.get(normalized)
    if member is None or member not in Key.__members__:
        raise ComputerError(
            f'Unknown key {name!r}. Use a single character or one of: '
            'enter, tab, escape, backspace, delete, space, up, down, left, right, home, end, '
            'pageup, pagedown, f1-f12, ctrl, shift, alt, cmd.'
        )
    return Key[member]


def _check_macos_permissions() -> None:
    """Fail with setup steps when macOS would otherwise capture only the wallpaper or drop input."""
    quartz = importlib.import_module('Quartz')
    application_services = importlib.import_module('ApplicationServices')
    missing: list[str] = []
    if not quartz.CGPreflightScreenCaptureAccess():
        quartz.CGRequestScreenCaptureAccess()
        missing.append('Screen Recording')
    if not application_services.AXIsProcessTrusted():
        missing.append('Accessibility')
    if missing:
        raise UserError(
            f'LocalComputer needs the macOS {" and ".join(missing)} permission. Open System Settings > '
            'Privacy & Security, allow the app running Python (your terminal or IDE) under '
            f'{" and ".join(missing)}, then restart that app.'
        )


@dataclass
class LocalComputer:
    """This machine's display, mouse, and keyboard, through mss, pynput, and Pillow.

    Screenshots are scaled down to fit `max_width` by `max_height` and coordinates
    are mapped back, so the model works in one space whatever the display's
    resolution or pixel density. Smaller screenshots cost fewer tokens; models
    locate targets most reliably at or below about 1280x800.

    The pointer and keyboard are the user's own: actions land on whichever window is
    in front, and moving the mouse during a run moves it out from under the model.
    On Linux it needs an X server; Wayland sessions are not supported. On macOS the
    app running Python needs the Screen Recording and Accessibility permissions.
    """

    monitor: int = 1
    """Which display to drive, as an `mss` monitor index: `1` is the primary display."""

    max_width: int = 1280
    """The widest screenshot sent to the model, in pixels."""

    max_height: int = 800
    """The tallest screenshot sent to the model, in pixels."""

    _permissions_checked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.monitor < 1:
            raise UserError('monitor must be 1 or more: index 0 is every display combined, which cannot be clicked.')
        if self.max_width < 1 or self.max_height < 1:
            raise UserError('max_width and max_height must be positive.')

    async def screenshot(self) -> bytes:
        """Capture the display, scaled to fit `max_width` by `max_height`, as PNG bytes."""
        return await to_thread.run_sync(self._screenshot)

    async def click(
        self, x: int, y: int, *, button: MouseButton = 'left', count: int = 1, modifiers: Sequence[str] = ()
    ) -> None:
        """Click at a screenshot position, holding `modifiers` for the duration."""
        await to_thread.run_sync(self._click, x, y, button, count, tuple(modifiers))

    async def move(self, x: int, y: int) -> None:
        """Move the pointer to a screenshot position."""
        await to_thread.run_sync(self._move, x, y)

    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        """Drag with the left button through screenshot positions."""
        await to_thread.run_sync(self._drag, tuple(path))

    async def scroll(self, x: int, y: int, *, direction: ScrollDirection, amount: int) -> None:
        """Scroll `amount` wheel clicks with the pointer at a screenshot position."""
        await to_thread.run_sync(self._scroll, x, y, direction, amount)

    async def type_text(self, text: str) -> None:
        """Type text at the keyboard focus."""
        await to_thread.run_sync(self._type_text, text)

    async def press_keys(self, keys: Sequence[str]) -> None:
        """Press `keys` together, then release them in reverse order."""
        await to_thread.run_sync(self._press_keys, tuple(keys))

    def _prepare(self) -> None:
        if not self._permissions_checked:
            if sys.platform == 'darwin':
                _check_macos_permissions()
            self._permissions_checked = True

    def _geometry(self) -> _Geometry:
        screen = mss.MSS()
        try:
            monitors = screen.monitors
        finally:
            screen.close()
        if self.monitor >= len(monitors):
            raise UserError(f'Monitor {self.monitor} does not exist; this machine has {len(monitors) - 1}.')
        monitor = monitors[self.monitor]
        width, height = monitor['width'], monitor['height']
        scale = min(1.0, self.max_width / width, self.max_height / height)
        return _Geometry(
            left=monitor['left'],
            top=monitor['top'],
            width=width,
            height=height,
            scaled_width=max(1, round(width * scale)),
            scaled_height=max(1, round(height * scale)),
        )

    def _screenshot(self) -> bytes:
        self._prepare()
        geometry = self._geometry()
        screen = mss.MSS()
        try:
            shot = screen.grab(
                {'left': geometry.left, 'top': geometry.top, 'width': geometry.width, 'height': geometry.height}
            )
        finally:
            screen.close()
        image = Image.frombytes('RGB', shot.size, shot.rgb)
        size = (geometry.scaled_width, geometry.scaled_height)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue()

    def _pointer(self, x: int, y: int) -> MouseController:
        from pynput.mouse import Controller

        self._prepare()
        position = self._geometry().to_screen(x, y)
        mouse = Controller()
        mouse.position = position
        time.sleep(_POINTER_SETTLE_SECONDS)
        return mouse

    def _click(self, x: int, y: int, button: MouseButton, count: int, modifiers: tuple[str, ...]) -> None:
        from pynput.keyboard import Controller
        from pynput.mouse import Button

        held = [_resolve_key(name, in_chord=True) for name in _split_chord(modifiers)]
        mouse = self._pointer(x, y)
        keyboard = Controller()
        pressed: list[Key | KeyCode] = []
        try:
            for key in held:
                keyboard.press(key)
                pressed.append(key)
            mouse.click(Button[button], count)
        finally:
            for key in reversed(pressed):
                keyboard.release(key)

    def _move(self, x: int, y: int) -> None:
        self._pointer(x, y)

    def _drag(self, path: tuple[tuple[int, int], ...]) -> None:
        from pynput.mouse import Button

        geometry = self._geometry()
        points = [geometry.to_screen(x, y) for x, y in path]
        mouse = self._pointer(*path[0])
        mouse.press(Button.left)
        try:
            previous = points[0]
            for point in points[1:]:
                for step in range(1, _DRAG_STEPS + 1):
                    mouse.position = (
                        round(previous[0] + (point[0] - previous[0]) * step / _DRAG_STEPS),
                        round(previous[1] + (point[1] - previous[1]) * step / _DRAG_STEPS),
                    )
                    time.sleep(_POINTER_SETTLE_SECONDS / _DRAG_STEPS)
                previous = point
        finally:
            mouse.release(Button.left)

    def _scroll(self, x: int, y: int, direction: ScrollDirection, amount: int) -> None:
        mouse = self._pointer(x, y)
        dx, dy = {'up': (0, amount), 'down': (0, -amount), 'left': (-amount, 0), 'right': (amount, 0)}[direction]
        mouse.scroll(dx, dy)

    def _type_text(self, text: str) -> None:
        from pynput.keyboard import Controller

        self._prepare()
        try:
            Controller().type(text)
        except Controller.InvalidCharacterException as exc:
            typed, character = exc.args
            raise ComputerError(
                f'Typed the first {typed} characters, then could not type {character!r}. Type only what comes after it.'
            ) from exc

    def _press_keys(self, keys: tuple[str, ...]) -> None:
        from pynput.keyboard import Controller

        names = _split_chord(keys)
        resolved = [_resolve_key(name, in_chord=len(names) > 1) for name in names]
        self._prepare()
        keyboard = Controller()
        pressed: list[Key | KeyCode] = []
        try:
            for key in resolved:
                keyboard.press(key)
                pressed.append(key)
        finally:
            for key in reversed(pressed):
                keyboard.release(key)
