"""`BoxdDisplayDriver` controls a boxd machine's desktop through boxd's computer API (`machines.computer`).

These facts about boxd's desktop were checked on 2026-11-13 against boxd 0.3.2 on boxd's production cluster. Re-run
the live display tests in `tests/harness/boxd_sandbox/test_boxd_live.py` before changing how actions map to calls.

- Every machine has a desktop on X display `:100`. The console's Desktop view shows the same desktop. The first
  computer call starts it at 1280x800, in about 3 seconds on a new machine.
- Checked 2026-11-20: the desktop is xpra in seamless mode, and xpra holds the window manager's role, so a second
  window manager such as fluxbox exits at start. A panel (tint2) and xterm do run, started with boxd's own
  `boxd-desktop-launch`. Windows get no title bars, and a window opens where it asks to be, or at the top left. A
  tint2 panel asked for the bottom edge showed at the top. A new window stays unviewable, so it can't be given the
  keyboard, until the next computer call, which shows it; any call does, even `cursor_position`. Clicking a window
  gives it the keyboard. xpra sets the `RESOURCE_MANAGER` X property, so xterm ignores `~/.Xdefaults` and draws a
  small bitmap font unless told a font.
- `set_screen_size` fixes the screen size until the desktop restarts. Someone watching in the console then sees the
  screen scaled, and doesn't resize it. Every call returns the screen size and the pointer position.
- Screenshots are PNGs at the screen size. A `region` screenshot shows that region at full resolution.
- `type` types any Unicode text, emoji included. It types a newline as Enter and a tab as Tab. One call takes at most
  10,000 characters, and 10,000 characters take about 46 seconds.
- `key` takes X key names (keysyms). Several joined with `+` press a combination, and `U` plus a code point presses
  any character. It refuses the plain names `pause`, `numlock` and `scrolllock`, and a `repeat` over 100.
- `scroll` turns at most 100 wheel clicks per call. `drag` takes 2 to 100 points.
- Mouse buttons are `left`, `middle`, `right`, `back` and `forward`. `click`, `scroll` and `drag` hold modifier keys
  during the event. `move` can't hold modifier keys, and no call holds a key down from one call to the next.
"""

from __future__ import annotations as _annotations

import shlex
from collections.abc import Awaitable, Callable, Sequence
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
    NamedKey,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError
from pydantic_ai_harness.computer_use._display import RELEASE_TIMEOUT
from pydantic_ai_harness.computer_use._png import PNG_SIGNATURE
from pydantic_ai_harness.computer_use._x11 import TINT2_BUTTON_STYLE

if TYPE_CHECKING:
    from boxd import ComputerActionResult
    from boxd.resources.computer import AsyncComputer

DISPLAY = ':100'
"""The X display number of the desktop on every boxd machine."""

# Ten times the 30 seconds the spec's install, update included, took live, so a second install outwaits the first.
_APT_GET = 'DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300'
_INSTALL_PANEL_PACKAGES = f'{_APT_GET} install -y --no-install-recommends tint2 xterm'

INSTALL_PANEL = f'{_INSTALL_PANEL_PACKAGES} || {{ {_APT_GET} update && {_INSTALL_PANEL_PACKAGES}; }}'
"""Installs the panel and the terminal it opens, as root. boxd's image has xterm; on 2026-11-20 tint2 needed an update
of the package lists first. The lock timeout makes a second install wait for one already running."""

DESKTOP_LOG = '/tmp/pydantic-ai-desktop.log'
"""Where the panel's install and the panel itself write, so a panel that didn't come up says why."""

# Not `--onlyvisible`: the panel's window stays unviewable until the next computer call.
_PANEL_SHOWS = f'DISPLAY={DISPLAY} xdotool search --class tint2 >/dev/null 2>&1'

_PANEL_DIR = '$HOME/.config/pydantic-ai-desktop'

_PANEL_HEIGHT = 40

# Opens xterm below the panel and as large as the rest of the screen, and gives it the keyboard. Nothing else on the
# desktop places, sizes or focuses a new window. The font is named because xterm ignores `~/.Xdefaults` here; 14 is
# the size the other sandboxes' desktops use, since models misread a smaller g as 9. The window can be focused only
# once it is viewable, at the next computer call, so focusing is retried. Both waits end if xterm exits.
_OPEN_TERMINAL_SCRIPT = f"""#!/bin/sh
xterm -fa Monospace -fs 14 & pid=$!
until w=$(xdotool search --pid $pid 2>/dev/null | head -1); [ -n "$w" ]; do kill -0 $pid 2>/dev/null || exit; sleep 0.1; done
eval "$(xdotool getdisplaygeometry --shell)"
xdotool windowmove "$w" 0 {_PANEL_HEIGHT} windowsize "$w" "$WIDTH" $((HEIGHT - {_PANEL_HEIGHT}))
until xdotool windowfocus "$w" 2>/dev/null; do kill -0 $pid 2>/dev/null || exit; sleep 0.1; done
wait $pid
"""

# Opens boxd's own browser, agent-browser's headed Chromium, as its `agent-browser.desktop` does, and moves its window
# below the panel and as large as the rest of the screen. It opens at the top left otherwise, over the panel. The
# desktop puts a window back at the top left when it first shows it, at the next computer call, so the move waits for
# that. The wait ends if the window closes.
_OPEN_BROWSER_SCRIPT = f"""#!/bin/sh
agent-browser open >/dev/null || exit
w=$(xdotool search --class Chromium-browser | tail -1)
until xwininfo -id "$w" 2>/dev/null | grep -q IsViewable; do xwininfo -id "$w" >/dev/null 2>&1 || exit; sleep 0.1; done
eval "$(xdotool getdisplaygeometry --shell)"
xdotool windowmove "$w" 0 {_PANEL_HEIGHT} windowsize "$w" "$WIDTH" $((HEIGHT - {_PANEL_HEIGHT})) windowfocus "$w"
"""

# A panel along the top with a button labelled Terminal, and a button per open window. A labelled button, not
# tint2's icon-only launcher: in live runs GPT-5.5 didn't try the bare icon until it had tried several key shortcuts.
_TINT2RC = (
    # Backgrounds 1 (the panel) and 2 (the buttons), numbered in the order they're written.
    'rounded = 0',
    'border_width = 0',
    'background_color = #2b2b2b 100',
    'border_color = #2b2b2b 0',
    'rounded = 0',
    'border_width = 1',
    'background_color = #444444 100',
    'border_color = #888888 100',
    'panel_position = top left horizontal',
    f'panel_size = 100% {_PANEL_HEIGHT}',
    'panel_padding = 4 4 6',
    'panel_background_id = 1',
    'button = new',
    'button_icon = mini.xterm',
    'button_text = Terminal',
    *TINT2_BUTTON_STYLE,
    f'button_lclick_command = {_PANEL_DIR}/terminal',
    'taskbar_padding = 2 2 4',
    'task_text = 1',
    'task_icon = 1',
    'task_font_color = #ffffff 100',
    'task_maximum_size = 220 34',
)

_BROWSER_BUTTON = (
    'button = new',
    'button_icon = web-browser',
    'button_text = Chromium',
    *TINT2_BUTTON_STYLE,
    f'button_lclick_command = {_PANEL_DIR}/browser',
)
"""A second button, labelled with the browser's name like the Terminal one, for when the machine has agent-browser."""

WRITE_PANEL_SETTINGS = (
    f'mkdir -p "{_PANEL_DIR}" && printf %s {shlex.quote(_OPEN_TERMINAL_SCRIPT)} >"{_PANEL_DIR}/terminal" && '
    f'printf %s {shlex.quote(_OPEN_BROWSER_SCRIPT)} >"{_PANEL_DIR}/browser" && '
    f'chmod +x "{_PANEL_DIR}/terminal" "{_PANEL_DIR}/browser" && {{ printf "%s\\n" {shlex.join(_TINT2RC)}; '
    # Each `P` in `panel_items` is the next button.
    f'if command -v agent-browser >/dev/null; then printf "%s\\n" {shlex.join(_BROWSER_BUTTON)} "panel_items = PPT"; '
    f'else echo "panel_items = PT"; fi; }} >"{_PANEL_DIR}/tint2rc"'
)
"""Writes the panel's settings and the scripts its buttons run, with a browser button only when the machine has
agent-browser, so it is checked at every panel start."""

START_PANEL = (
    # One start at a time, so two can't both find no panel and start two. tint2 must not hold the lock.
    f'exec >>{DESKTOP_LOG} 2>&1 9>/tmp/.pydantic-ai-panel.lock; ! command -v flock >/dev/null || flock 9; '
    f'command -v tint2 >/dev/null && command -v xterm >/dev/null || sudo -n sh -c {shlex.quote(INSTALL_PANEL)}; '
    f'{WRITE_PANEL_SETTINGS} || exit; '
    f'if command -v tint2 >/dev/null && ! {_PANEL_SHOWS}; then '
    f'boxd-desktop-launch "tint2 -c {_PANEL_DIR}/tint2rc >>{DESKTOP_LOG} 2>&1" 9>&-; i=0; '
    f'until {_PANEL_SHOWS} || [ $i -ge 50 ]; do sleep 0.2; i=$((i+1)); done; fi; '
    f'{_PANEL_SHOWS}'
)
"""Installs the panel and xterm unless both are there (with passwordless `sudo`), writes the panel's settings, and
starts the panel with `boxd-desktop-launch` unless a tint2 panel already runs. Waits up to 10 seconds for its window
(under a second live) and exits 0 once it is there. A machine created with `desktop` also installs both from its spec
at boot; the second install waits for apt's lock."""

_MAX_CHARS_PER_CALL = 10_000

_MAX_SCROLL_CLICKS_PER_CALL = 100

_MAX_KEY_REPEATS_PER_CALL = 100

_MAX_DRAG_POINTS_PER_CALL = 100
"""A longer drag presses, moves and releases the button in separate calls."""

_NAMED_KEYSYMS: dict[NamedKey, str] = {'pause': 'Pause', 'numlock': 'Num_Lock', 'scrolllock': 'Scroll_Lock'}

_RENAMED_KEYS: dict[str, str] = {
    **_NAMED_KEYSYMS,
    # Sent as keys, not characters, because browsers ignore the Linefeed keysym.
    '\n': 'Return',
    '\t': 'Tab',
}
"""Key names that boxd accepts only as X keysyms, mapped to those keysyms. boxd takes every other name as is."""


def _keysym(key: str) -> str:
    if key in _RENAMED_KEYS:
        return _RENAMED_KEYS[key]
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A symbol such as `+` would be read as the joiner in `ctrl++`. `U002B` can't be misread.
        return f'U{ord(key):04X}'
    return key


def _key_combination(keys: Sequence[str]) -> str:
    return '+'.join(_keysym(key) for key in keys)


def _screenshot_png(result: ComputerActionResult) -> bytes:
    if result.screenshot is None or not result.screenshot.data.startswith(PNG_SIGNATURE):
        raise WorkspaceError('boxd did not return a PNG screenshot.')
    return result.screenshot.data


class BoxdDisplayDriver:
    """Controls a boxd machine's desktop with boxd's computer API.

    `get_computer_api` returns the API and the machine's ID. It is called again for every action, because the backend
    reconnects with a new client after it closes the old one. The backend also starts the panel in it, so only a call
    that uses the screen waits for the panel.
    """

    def __init__(self, get_computer_api: Callable[[], Awaitable[tuple[AsyncComputer, str]]]):
        self._get_computer_api = get_computer_api

    async def display_size(self) -> tuple[int, int]:
        computer, machine_id = await self._get_computer_api()
        result = await computer.cursor_position(machine_id)
        return result.screen_width, result.screen_height

    async def screenshot(self) -> bytes:
        computer, machine_id = await self._get_computer_api()
        return _screenshot_png(await computer.screenshot(machine_id))

    async def zoom(self, region: Region) -> bytes:
        computer, machine_id = await self._get_computer_api()
        area = (region.left, region.top, region.right - region.left, region.bottom - region.top)
        return _screenshot_png(await computer.screenshot(machine_id, region=area))

    async def cursor_position(self) -> Point:
        computer, machine_id = await self._get_computer_api()
        cursor = (await computer.cursor_position(machine_id)).cursor
        return Point(x=cursor.x, y=cursor.y)

    async def act(self, action: DisplayAction) -> None:
        computer, machine_id = await self._get_computer_api()
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await computer.click(machine_id, x, y, button=button, count=count, modifiers=modifiers)
            case MouseMove(x=x, y=y):
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
                for start in range(0, amount, _MAX_SCROLL_CLICKS_PER_CALL):
                    clicks = sign * min(_MAX_SCROLL_CLICKS_PER_CALL, amount - start)
                    dx, dy = (0, clicks) if vertical else (clicks, 0)
                    await computer.scroll(machine_id, x, y, dx=dx, dy=dy, modifiers=modifiers)
            case TypeText(text=text):
                for start in range(0, len(text), _MAX_CHARS_PER_CALL):
                    await computer.type(machine_id, text[start : start + _MAX_CHARS_PER_CALL])
            case PressKeys(keys=keys, repeat=repeat):
                combination = _key_combination(keys)
                for start in range(0, repeat, _MAX_KEY_REPEATS_PER_CALL):
                    await computer.key(machine_id, combination, repeat=min(_MAX_KEY_REPEATS_PER_CALL, repeat - start))
            case HoldKeys(keys=keys, seconds=seconds):
                await computer.key(machine_id, _key_combination(keys), hold_ms=max(1, round(seconds * 1000)))
            case _:
                raise UnsupportedActionError(f"boxd's desktop cannot perform {action!r}.")

    async def _drag(
        self, computer: AsyncComputer, machine_id: str, path: Sequence[Point], modifiers: Sequence[str]
    ) -> None:
        points = [(point.x, point.y) for point in path]
        if len(points) <= _MAX_DRAG_POINTS_PER_CALL:
            await computer.drag(machine_id, points, modifiers=modifiers)
            return
        if modifiers:
            raise UnsupportedActionError(
                f"boxd's desktop can hold keys during a drag of at most {_MAX_DRAG_POINTS_PER_CALL} points."
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
            with anyio.move_on_after(RELEASE_TIMEOUT, shield=True), suppress(Exception):
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
                f'A long drag failed ({error}), and releasing the left mouse button failed too, so it may still be held.'
            ) from error
