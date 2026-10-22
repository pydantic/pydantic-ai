"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import base64
import binascii
import re
from contextlib import suppress

import anyio

from .computer import (
    Click,
    ComputerAction,
    Drag,
    HoldKey,
    KeyPress,
    MouseButton,
    MouseDown,
    MouseUp,
    MoveMouse,
    Scroll,
    TypeText,
)
from .protocol import (
    CommandResult,
    SupportsCommands,
    SupportsComputerUse,
    UnsupportedComputerActionError,
    WorkspaceError,
)

__all__ = ('X11Computer',)

_BUTTONS: dict[MouseButton, str] = {'left': '1', 'middle': '2', 'right': '3', 'back': '8', 'forward': '9'}

_KEYSYMS = {
    'enter': 'Return',
    'escape': 'Escape',
    'tab': 'Tab',
    'backspace': 'BackSpace',
    'delete': 'Delete',
    'insert': 'Insert',
    'home': 'Home',
    'end': 'End',
    'pageup': 'Prior',
    'pagedown': 'Next',
    'up': 'Up',
    'down': 'Down',
    'left': 'Left',
    'right': 'Right',
    **{f'f{n}': f'F{n}' for n in range(1, 25)},
    '+': 'plus',
    '-': 'minus',
    '=': 'equal',
    ',': 'comma',
    '.': 'period',
    '/': 'slash',
    '\\': 'backslash',
    ';': 'semicolon',
    "'": 'apostrophe',
    '`': 'grave',
    '[': 'bracketleft',
    ']': 'bracketright',
}
"""Key names whose X keysym is spelled differently; other names (`ctrl`, `a`, `KP_Enter`, ...) are keysyms already."""

_KEYSYM_PATTERN = re.compile(r'[A-Za-z0-9_]+')

_XDOTOOL_COMMANDS = frozenset(
    {
        'behave',
        'behave_screen_edge',
        'click',
        'exec',
        'get_desktop',
        'get_desktop_for_window',
        'get_desktop_viewport',
        'get_num_desktops',
        'getactivewindow',
        'getdisplaygeometry',
        'getmouselocation',
        'getwindowfocus',
        'getwindowgeometry',
        'getwindowname',
        'getwindowpid',
        'help',
        'key',
        'keydown',
        'keyup',
        'mousedown',
        'mousemove',
        'mousemove_relative',
        'mouseup',
        'search',
        'selectwindow',
        'set_desktop',
        'set_desktop_for_window',
        'set_desktop_viewport',
        'set_num_desktops',
        'set_window',
        'sleep',
        'type',
        'version',
        'windowactivate',
        'windowclose',
        'windowfocus',
        'windowkill',
        'windowmap',
        'windowminimize',
        'windowmove',
        'windowraise',
        'windowreparent',
        'windowsize',
        'windowunmap',
    }
)
"""`xdotool help`'s command names: xdotool reads one among key names as the start of the next command, such as `exec`."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_SCREENSHOT = 'd=$(mktemp -d) && scrot "$d/screen.png" && base64 "$d/screen.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
_PNG_END = b'IEND\xaeB`\x82'
"""The final chunk of every PNG, so a truncated capture is caught."""


def _keysym(key: str) -> str:
    keysym = _KEYSYMS.get(key, key)
    if not _KEYSYM_PATTERN.fullmatch(keysym) or keysym in _XDOTOOL_COMMANDS:
        raise UnsupportedComputerActionError(f'X11Computer cannot press the key {key!r}.')
    return keysym


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


class X11Computer(SupportsComputerUse):
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running: starting it is the environment's job (for example an image that runs
    `Xvfb :1`). Needs `xdotool`, `scrot`, `base64`, and `mktemp` in the environment. A backend with a desktop
    implements [`SupportsComputerUse`][pydantic_ai.workspaces.SupportsComputerUse] by delegating to an instance:

    ```python {test="skip" lint="skip"}
    async def screenshot(self) -> bytes:
        return await X11Computer(self).screenshot()
    ```

    Args:
        backend: The environment's commands, which run inside it.
        display: The X display to control, as `DISPLAY` would name it.
    """

    def __init__(self, backend: SupportsCommands, *, display: str = ':1'):
        self._backend = backend
        self._display = display

    async def _run(self, command: list[str] | str) -> CommandResult:
        result = await self._backend.run(command, shell=isinstance(command, str), env={'DISPLAY': self._display})
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Display command failed with exit code {result.exit_code} on display {self._display}: '
                f'{(result.stderr or result.stdout).strip()}'
            )
        return result

    async def _xdotool(self, args: list[str], *, release: list[str] | None = None) -> None:
        """Run one `xdotool` command chain; if it fails or is cancelled, run `release` to let go of what it held."""
        try:
            await self._run(['xdotool', *args])
        except BaseException:
            if release:
                with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                    await self._run(['xdotool', *release])
            raise

    async def _holding(self, keys: tuple[str, ...], args: list[str], *, release: list[str] | None = None) -> None:
        """Run `args` with `keys` held down, released in reverse order, and `release` run as well on failure."""
        keysyms = [_keysym(key) for key in keys]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        chain = ['keydown', *keysyms, *args, *keyup] if keysyms else args
        await self._xdotool(chain, release=[*(release or []), *keyup])

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(['xdotool', 'getdisplaygeometry'])).stdout.split()
        try:
            width, height = output
            return int(width), int(height)
        except ValueError:
            raise WorkspaceError(f'Unexpected `xdotool getdisplaygeometry` output: {" ".join(output)!r}') from None

    async def screenshot(self) -> bytes:
        output = (await self._run(_SCREENSHOT)).stdout
        try:
            data = base64.b64decode(''.join(output.split()), validate=True)
        except (binascii.Error, ValueError):
            data = b''
        if not (data.startswith(_PNG_SIGNATURE) and data.endswith(_PNG_END)):
            raise WorkspaceError('The screenshot command did not print a complete base64-encoded PNG.')
        return data

    async def act(self, action: ComputerAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, keys=keys):
                await self._holding(keys, [*_move(x, y), 'click', '--repeat', str(count), _BUTTONS[button]])
            case MoveMouse(x=x, y=y, keys=keys):
                await self._holding(keys, _move(x, y))
            case Drag(path=path, keys=keys):
                (start_x, start_y), *rest = path
                steps = [arg for x, y in rest for arg in _move(x, y)]
                await self._holding(
                    keys, [*_move(start_x, start_y), 'mousedown', '1', *steps, 'mouseup', '1'], release=['mouseup', '1']
                )
            case MouseDown(button=button):
                await self._xdotool(['mousedown', _BUTTONS[button]])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _BUTTONS[button]])
            case Scroll(x=x, y=y, dx=dx, dy=dy, keys=keys):
                # X maps the wheel to buttons: 4 up, 5 down, 6 left, 7 right.
                clicks: list[str] = []
                if dy:
                    clicks += ['click', '--repeat', str(abs(dy)), '5' if dy > 0 else '4']
                if dx:
                    clicks += ['click', '--repeat', str(abs(dx)), '7' if dx > 0 else '6']
                await self._holding(keys, [*_move(x, y), *clicks])
            case TypeText(text=text):
                await self._xdotool(['type', '--', text])
            case KeyPress(keys=keys, repeat=repeat):
                await self._xdotool(['key', '--repeat', str(repeat), '--', '+'.join(_keysym(key) for key in keys)])
            case HoldKey(keys=keys, seconds=seconds):
                await self._holding(keys, ['sleep', str(seconds)])
            case _:
                raise UnsupportedComputerActionError(f'X11Computer cannot perform {action!r}.')
