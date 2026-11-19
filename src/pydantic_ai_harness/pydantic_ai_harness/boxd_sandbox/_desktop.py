"""`BoxdDesktop` controls a boxd machine's desktop through boxd's computer API (`machines.computer`).

These facts about boxd's desktop were checked on 2026-11-13 against boxd 0.3.2 on boxd's production cluster. Re-run
the live display tests in `tests/harness/boxd_sandbox/test_boxd_live.py` before changing how actions map to calls.

- Every machine has a desktop on X display `:100`. The console's Desktop view shows the same desktop. The first
  computer call starts it at 1280x800, in about 3 seconds on a new machine. It has no window manager, so a window
  opens at the top left without a title bar.
- `set_screen_size` fixes the screen size until the desktop restarts. Someone watching in the console then sees the
  screen scaled, and doesn't resize it. Every call returns the screen size and the pointer position.
- Screenshots are PNGs at the screen size. A `region` screenshot shows that region at full resolution.
- `type` types any Unicode text, emoji included. It types a newline as Enter and a tab as Tab. One call takes at most
  10,000 characters, and 10,000 characters take about 46 seconds.
- `key` takes X key names (keysyms). Several joined with `+` press a combination, and `U` plus a code point presses
  any character. It refuses the plain names `pause`, `numlock`, `scrolllock` and the numpad and media key names. It
  also refuses a `repeat` over 100.
- `scroll` turns at most 100 wheel clicks per call. `drag` takes 2 to 100 points.
- Mouse buttons are `left`, `middle`, `right`, `back` and `forward`. `click`, `scroll` and `drag` hold modifier keys
  during the event. `move` can't hold modifier keys, and no call holds a key down from one call to the next.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING

import anyio
from boxd import AuthenticationError, PermissionDeniedError

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness._png import PNG_SIGNATURE
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError

if TYPE_CHECKING:
    from boxd import ComputerActionResult
    from boxd.resources.computer import AsyncComputer

DISPLAY = ':100'
"""The X display number of the desktop on every boxd machine."""

_RELEASE_TIMEOUT = 10
"""Seconds to spend letting go of the mouse button after a drag fails or is cancelled."""

_TYPED_PER_CALL = 10_000
"""Characters typed per call. boxd takes no more."""

_SCROLLED_PER_CALL = 100
"""Most wheel clicks per `scroll` call."""

_REPEATS_PER_CALL = 100
"""Most key presses per `key` call."""

_DRAG_POINTS_PER_CALL = 100
"""Most points per `drag` call. A longer drag presses, moves and releases the button in separate calls."""

_KEYSYMS = {
    'pause': 'Pause',
    'numlock': 'Num_Lock',
    'scrolllock': 'Scroll_Lock',
    **{f'numpad{n}': f'KP_{n}' for n in range(10)},
    'numpad_enter': 'KP_Enter',
    'numpad_add': 'KP_Add',
    'numpad_subtract': 'KP_Subtract',
    'numpad_multiply': 'KP_Multiply',
    'numpad_divide': 'KP_Divide',
    'numpad_decimal': 'KP_Decimal',
    'volumemute': 'XF86AudioMute',
    'volumeup': 'XF86AudioRaiseVolume',
    'volumedown': 'XF86AudioLowerVolume',
    'mediaplay': 'XF86AudioPlay',
    'medianext': 'XF86AudioNext',
    'mediaprev': 'XF86AudioPrev',
    # Sent as keys, not characters, because browsers ignore the Linefeed keysym.
    '\n': 'Return',
    '\t': 'Tab',
}
"""Key names that boxd accepts only as X keysyms, mapped to those keysyms. boxd takes every other name as is."""


def _keysym(key: str) -> str:
    if key in _KEYSYMS:
        return _KEYSYMS[key]
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A symbol such as `+` would be read as the joiner in `ctrl++`. `U002B` can't be misread.
        return f'U{ord(key):04X}'
    return key


def _combination(keys: Sequence[str]) -> str:
    return '+'.join(_keysym(key) for key in keys)


def _png(result: ComputerActionResult) -> bytes:
    if result.screenshot is None or not result.screenshot.data.startswith(PNG_SIGNATURE):
        raise WorkspaceError('boxd did not return a PNG screenshot.')
    return result.screenshot.data


class BoxdDesktop:
    """Controls a boxd machine's desktop with boxd's computer API.

    `attached` returns the API and the machine's ID. It is called again for every action, because the backend
    reconnects with a new client after it closes the old one.
    """

    def __init__(self, attached: Callable[[], tuple[AsyncComputer, str]]):
        self._attached = attached

    async def display_size(self) -> tuple[int, int]:
        computer, machine_id = self._attached()
        result = await computer.cursor_position(machine_id)
        return result.screen_width, result.screen_height

    async def screenshot(self) -> bytes:
        computer, machine_id = self._attached()
        return _png(await computer.screenshot(machine_id))

    async def zoom(self, region: Region) -> bytes:
        computer, machine_id = self._attached()
        area = (region.left, region.top, region.right - region.left, region.bottom - region.top)
        return _png(await computer.screenshot(machine_id, region=area))

    async def cursor_position(self) -> Point:
        computer, machine_id = self._attached()
        cursor = (await computer.cursor_position(machine_id)).cursor
        return Point(x=cursor.x, y=cursor.y)

    async def act(self, action: DisplayAction) -> None:
        computer, machine_id = self._attached()
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await computer.click(machine_id, x, y, button=button, count=count, modifiers=modifiers)
            case MouseMove(x=x, y=y, modifiers=modifiers):
                if modifiers:
                    raise UnsupportedActionError("boxd's desktop cannot hold keys while it moves the pointer.")
                await computer.move(machine_id, x, y)
            case Drag(path=path, modifiers=modifiers):
                await self._drag(computer, machine_id, path, modifiers)
            case MouseDown(button=button):
                await computer.mouse_down(machine_id, button=button)
            case MouseUp(button=button):
                await computer.mouse_up(machine_id, button=button)
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                sign = -1 if direction in ('up', 'left') else 1
                vertical = direction in ('up', 'down')
                for start in range(0, amount, _SCROLLED_PER_CALL):
                    clicks = sign * min(_SCROLLED_PER_CALL, amount - start)
                    dx, dy = (0, clicks) if vertical else (clicks, 0)
                    await computer.scroll(machine_id, x, y, dx=dx, dy=dy, modifiers=modifiers)
            case TypeText(text=text):
                for start in range(0, len(text), _TYPED_PER_CALL):
                    await computer.type(machine_id, text[start : start + _TYPED_PER_CALL])
            case PressKeys(keys=keys, repeat=repeat):
                combination = _combination(keys)
                for start in range(0, repeat, _REPEATS_PER_CALL):
                    await computer.key(machine_id, combination, repeat=min(_REPEATS_PER_CALL, repeat - start))
            case HoldKeys(keys=keys, seconds=seconds):
                await computer.key(machine_id, _combination(keys), hold_ms=max(1, round(seconds * 1000)))
            case _:
                raise UnsupportedActionError(f"boxd's desktop cannot perform {action!r}.")

    async def _drag(
        self, computer: AsyncComputer, machine_id: str, path: Sequence[Point], modifiers: Sequence[str]
    ) -> None:
        points = [(point.x, point.y) for point in path]
        if len(points) <= _DRAG_POINTS_PER_CALL:
            await computer.drag(machine_id, points, modifiers=modifiers)
            return
        if modifiers:
            raise UnsupportedActionError(
                f"boxd's desktop can hold keys during a drag of at most {_DRAG_POINTS_PER_CALL} points."
            )
        # Over 100 points: press, then one `move` call per point, then release. No pauses are added, and a browser
        # still sees the drag start and land.
        start, end = path[0], path[-1]
        try:
            # Inside the `try`, so a press that boxd made before a cancel is still let go.
            await computer.mouse_down(machine_id, start.x, start.y)
            for point in path[1:]:
                await computer.move(machine_id, point.x, point.y)
            await computer.mouse_up(machine_id, end.x, end.y)
        except BaseException as error:
            released = False
            with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                await computer.mouse_up(machine_id)
                released = True
            # A credentials error keeps its own type, so the backend still ends the run on it.
            if (
                released
                or not isinstance(error, Exception)
                or isinstance(error, AuthenticationError | PermissionDeniedError)
            ):
                raise
            raise WorkspaceError(
                f'{error}. Releasing the left mouse button failed too, so it may still be held.'
            ) from error
