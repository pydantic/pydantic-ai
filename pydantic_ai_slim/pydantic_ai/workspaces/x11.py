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
    HoldKeys,
    MouseButton,
    MouseDown,
    MouseUp,
    MoveMouse,
    PressKeys,
    Scroll,
    TypeText,
)
from .protocol import (
    CommandResult,
    SupportsCommands,
    UnsupportedComputerActionError,
    WorkspaceError,
)

__all__ = ('X11Computer',)

_BUTTONS: dict[MouseButton, str] = {'left': '1', 'middle': '2', 'right': '3', 'back': '8', 'forward': '9'}

_WHEEL_BUTTONS = {'up': '4', 'down': '5', 'left': '6', 'right': '7'}
"""X reports the scroll wheel as buttons."""

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
        'getwindowclassname',
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
        'windowlower',
        'windowmap',
        'windowminimize',
        'windowmove',
        'windowquit',
        'windowraise',
        'windowreparent',
        'windowsize',
        'windowstate',
        'windowunmap',
    }
)
"""xdotool's command names, from its newest source: xdotool reads one among key names, in any case, as the start of the
next command, such as `exec`."""

_COMMAND_TIMEOUT = 60
"""Maximum time for one display command; the longest bounded action, typing 2,000 characters, takes about 25 s."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_SCREENSHOT = 'd=$(mktemp -d) && scrot "$d/screen.png" && base64 "$d/screen.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _keysym(key: str) -> str:
    keysym = _KEYSYMS.get(key, key)
    # xdotool reads any argument that names one of its commands, in any case, as that command.
    if not _KEYSYM_PATTERN.fullmatch(keysym) or keysym.lower() in _XDOTOOL_COMMANDS:
        raise UnsupportedComputerActionError(f'X11Computer cannot press the key {key!r}.')
    return keysym


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running: starting it is the environment's job (for example an image that runs
    `Xvfb :1`). Needs `xdotool`, `scrot`, `base64`, and `mktemp` in the environment. Text outside ASCII can lose
    characters: `xdotool` types it by remapping keys on the fly. A backend with a desktop
    implements [`SupportsDisplay`][pydantic_ai.workspaces.SupportsDisplay] by delegating to an instance:

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
        result = await self._backend.run(
            command, shell=isinstance(command, str), env={'DISPLAY': self._display}, timeout=_COMMAND_TIMEOUT
        )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Display command failed with exit code {result.exit_code} on display {self._display}: '
                f'{(result.stderr or result.stdout).strip()}'
            )
        return result

    async def _xdotool(self, args: list[str], *, hold: tuple[str, ...] = (), release: tuple[str, ...] = ()) -> None:
        """Run one `xdotool` command chain with `hold` held down; if it fails or is cancelled, let go of everything."""
        keysyms = [_keysym(key) for key in hold]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        try:
            result = await self._run(['xdotool', *(['keydown', *keysyms] if keysyms else []), *args, *keyup])
        except BaseException:
            if release or keyup:
                with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                    await self._run(['xdotool', *release, *keyup])
            raise
        # xdotool skips a key name the display doesn't know and still succeeds.
        if 'No such key name' in result.stderr:
            raise UnsupportedComputerActionError(f'X11Computer cannot press that key: {result.stderr.strip()}')

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
        if not data.startswith(_PNG_SIGNATURE):
            raise WorkspaceError('The screenshot command did not print a base64-encoded PNG.')
        return data

    async def act(self, action: ComputerAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await self._xdotool([*_move(x, y), 'click', '--repeat', str(count), _BUTTONS[button]], hold=modifiers)
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                await self._xdotool(_move(x, y), hold=modifiers)
            case Drag(path=path, modifiers=modifiers):
                start, *rest = path
                steps = [arg for point in rest for arg in _move(point.x, point.y)]
                await self._xdotool(
                    [*_move(start.x, start.y), 'mousedown', '1', *steps, 'mouseup', '1'],
                    hold=modifiers,
                    release=('mouseup', '1'),
                )
            case MouseDown(button=button):
                await self._xdotool(['mousedown', _BUTTONS[button]])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _BUTTONS[button]])
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                wheel = _WHEEL_BUTTONS[direction]
                await self._xdotool([*_move(x, y), 'click', '--repeat', str(amount), wheel], hold=modifiers)
            case TypeText(text=text):
                await self._xdotool(['type', '--', text])
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ('keyup', *reversed(keysyms))
                await self._xdotool(['key', '--repeat', str(repeat), '--', '+'.join(keysyms)], release=release)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys)
            case _:
                raise UnsupportedComputerActionError(f'X11Computer cannot perform {action!r}.')
