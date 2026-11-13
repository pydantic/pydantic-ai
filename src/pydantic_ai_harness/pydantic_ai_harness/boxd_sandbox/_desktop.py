"""`BoxdDesktop`: display control for a boxd machine, through boxd's computer API (`machines.computer`).

How boxd's desktop behaves, checked 2026-11-13 against boxd 0.3.2 on boxd's production cluster. Re-run the live display
tests in `tests/harness/boxd_sandbox/test_boxd_live.py` before changing the mapping:

- Every machine has a desktop on X display `:100`, also shown in the console's Desktop view. The first computer call
  starts it, in about 3 seconds on a new machine, at 1280x800. It has no window manager, so a window opens at the top
  left without a title bar.
- `set_screen_size` holds the screen at a size until the desktop restarts, so someone watching in the console sees it
  scaled rather than resizing it. Every call answers with the screen's size and the pointer's position.
- Screenshots are PNGs at the screen's size; a `region` screenshot is the region at full resolution.
- `type` types any Unicode, emoji included, newlines as Enter and tabs as Tab, at most 10,000 characters a call;
  2,000 take about 9 seconds.
- `key` takes X keysyms, several joined with `+` for a combination, and `U` plus a code point for any character. It
  refuses the bare names `pause`, `numlock`, `scrolllock` and the numpad and media key names, and a `repeat` over 100.
- `scroll` turns at most 100 wheel clicks a call, and `drag` takes 2 to 100 points.
- Mouse buttons are `left`, `middle`, `right`, `back` and `forward`. `click`, `scroll` and `drag` hold modifiers
  around the event; `move` takes none, and there is no call that holds a key down across calls.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseUp,
    MoveMouse,
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
"""The X display of every boxd machine's desktop."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing the mouse button after a failed or cancelled drag."""

_TYPED_PER_CALL = 2_000
"""Characters typed by one call: boxd takes at most 10,000, and 2,000 take about 9 seconds."""

_SCROLLED_PER_CALL = 100
"""Wheel clicks one call may turn."""

_REPEATS_PER_CALL = 100
"""Times one call may press a key."""

_DRAG_POINTS_PER_CALL = 100
"""Points one `drag` call may take; a longer drag presses, moves and releases in separate calls."""

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
}
"""Normalized key names boxd knows only by their X keysym. It takes the other names as they are."""


def _keysym(key: str) -> str:
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A symbol such as `+` would read as the join in `ctrl++`; `U002B` can't.
        return f'U{ord(key):04X}'
    return _KEYSYMS.get(key, key)


def _combination(keys: Sequence[str]) -> str:
    return '+'.join(_keysym(key) for key in keys)


def _modifiers(modifiers: Sequence[str]) -> list[str]:
    # Normalized to `ctrl`, `shift`, `alt` and `super`, which boxd takes as they are.
    return list(modifiers)


def _png(result: ComputerActionResult) -> bytes:
    if result.screenshot is None or not result.screenshot.data.startswith(PNG_SIGNATURE):
        raise WorkspaceError('boxd did not return a PNG screenshot.')
    return result.screenshot.data


class BoxdDesktop:
    """Controls a boxd machine's desktop with boxd's computer API.

    `attached` returns the API and the machine's id. It is read on every call, since the backend attaches again with
    a new client after closing the old one.
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
                await computer.click(machine_id, x, y, button=button, count=count, modifiers=_modifiers(modifiers))
            case MoveMouse(x=x, y=y, modifiers=modifiers):
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
                for start in range(0, amount, _SCROLLED_PER_CALL):
                    clicks = sign * min(_SCROLLED_PER_CALL, amount - start)
                    if direction in ('up', 'down'):
                        await computer.scroll(machine_id, x, y, dy=clicks, modifiers=_modifiers(modifiers))
                    else:
                        await computer.scroll(machine_id, x, y, dx=clicks, modifiers=_modifiers(modifiers))
            case TypeText(text=text):
                for start in range(0, len(text), _TYPED_PER_CALL):
                    await computer.type(machine_id, text[start : start + _TYPED_PER_CALL])
            case PressKeys(keys=keys, repeat=repeat):
                combination = _combination(keys)
                for start in range(0, repeat, _REPEATS_PER_CALL):
                    await computer.key(machine_id, combination, repeat=min(_REPEATS_PER_CALL, repeat - start))
            case HoldKeys(keys=keys, seconds=seconds):
                await computer.key(machine_id, _combination(keys), hold_ms=round(seconds * 1000))
            case _:
                raise UnsupportedActionError(f"boxd's desktop cannot perform {action!r}.")

    async def _drag(
        self, computer: AsyncComputer, machine_id: str, path: Sequence[Point], modifiers: Sequence[str]
    ) -> None:
        points = [(point.x, point.y) for point in path]
        if len(points) <= _DRAG_POINTS_PER_CALL:
            await computer.drag(machine_id, points, modifiers=_modifiers(modifiers))
            return
        if modifiers:
            raise UnsupportedActionError(
                f"boxd's desktop can hold keys during a drag of at most {_DRAG_POINTS_PER_CALL} points."
            )
        # More than 100 moves, each its own call: a browser sees the drag start and land without added pauses.
        start, end = path[0], path[-1]
        await computer.mouse_down(machine_id, start.x, start.y)
        try:
            for point in path[1:]:
                await computer.move(machine_id, point.x, point.y)
            await computer.mouse_up(machine_id, end.x, end.y)
        except BaseException:
            with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                await computer.mouse_up(machine_id)
            raise
