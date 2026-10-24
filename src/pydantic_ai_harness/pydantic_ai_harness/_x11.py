"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import base64
import re
from contextlib import suppress

import anyio

from pydantic_ai.workspaces import (
    Click,
    CommandResult,
    ComputerAction,
    Drag,
    HoldKeys,
    MouseDown,
    MouseUp,
    MoveMouse,
    Point,
    PressKeys,
    Region,
    Scroll,
    SupportsCommands,
    TypeText,
    UnsupportedComputerActionError,
    WorkspaceError,
)

_BUTTONS: dict[str, str] = {'left': '1', 'middle': '2', 'right': '3', 'back': '8', 'forward': '9'}

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

_COMMAND_TIMEOUT = 150
"""Maximum time for one display command; the longest bounded action, typing 2,000 characters outside ASCII, takes
about 100 s."""

_UNICODE_TYPE_DELAY_MS = '50'
"""Milliseconds between characters when the text has any outside ASCII: `xdotool` types those by remapping a key, and
at its default delay of 12 ms the remapping races the keystrokes and drops characters (checked live)."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_SCREENSHOT = 'd=$(mktemp -d) && scrot "$d/screen.png" && base64 "$d/screen.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64."""

_ZOOM = 'd=$(mktemp -d) && scrot -a "$1" "$d/zoom.png" && base64 "$d/zoom.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures the region `$1` names, as `x,y,width,height`, like `_SCREENSHOT`."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _code(codes: dict[str, str], value: str) -> str:
    """The X code for a button or scroll direction, refusing one added after this display was written."""
    if value not in codes:
        raise UnsupportedComputerActionError(f'The display cannot use {value!r}.')
    return codes[value]


def _keysym(key: str) -> str:
    keysym = _KEYSYMS.get(key, key)
    # xdotool reads any argument that names one of its commands, in any case, as that command.
    if not _KEYSYM_PATTERN.fullmatch(keysym) or keysym.lower() in _XDOTOOL_COMMANDS:
        raise UnsupportedComputerActionError(f'The display cannot press the key {key!r}.')
    return keysym


def _png(output: str) -> bytes:
    try:
        data = base64.b64decode(''.join(output.split()), validate=True)
    except ValueError:  # includes `binascii.Error`
        data = b''
    if not data.startswith(_PNG_SIGNATURE):
        raise WorkspaceError('The screenshot command did not print a base64-encoded PNG.')
    return data


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


DISPLAY = ':1'
"""The X display the desktop runs on, as `DISPLAY` names it."""


def start_x_server(log: str) -> str:
    """A shell fragment that starts Xvfb on `DISPLAY`, at the size `$1` names such as `1280x800`, unless one answers.

    It gives up as soon as the X server exits, or after 10 seconds, with the end of `log` as the reason.
    """
    return (
        'if ! xdotool getdisplaygeometry >/dev/null 2>&1; then '
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp >/dev/null 2>{log} & xvfb=$!; i=0; '
        'until xdotool getdisplaygeometry >/dev/null 2>&1; do '
        'if ! kill -0 $xvfb 2>/dev/null || [ $i -ge 100 ]; then '
        f"echo 'the X server did not start:' >&2; tail -n 5 {log} >&2; exit 1; fi; "
        'sleep 0.1; i=$((i+1)); done; fi; '
    )


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `DISPLAY`. Needs `xdotool`, `scrot`, `base64`, and `mktemp` in the
    environment.
    """

    def __init__(self, backend: SupportsCommands):
        self._backend = backend

    async def _run(self, command: list[str] | str) -> CommandResult:
        result = await self._backend.run(
            command, shell=isinstance(command, str), env={'DISPLAY': DISPLAY}, timeout=_COMMAND_TIMEOUT
        )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Display command failed with exit code {result.exit_code} on display {DISPLAY}: '
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
            raise UnsupportedComputerActionError(f'The display cannot press that key: {result.stderr.strip()}')

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(['xdotool', 'getdisplaygeometry'])).stdout.split()
        try:
            width, height = output
            return int(width), int(height)
        except ValueError:
            raise WorkspaceError(f'Unexpected `xdotool getdisplaygeometry` output: {" ".join(output)!r}') from None

    async def screenshot(self) -> bytes:
        return _png((await self._run(_SCREENSHOT)).stdout)

    async def zoom(self, region: Region) -> bytes:
        geometry = f'{region.left},{region.top},{region.right - region.left},{region.bottom - region.top}'
        return _png((await self._run(['sh', '-c', _ZOOM, 'sh', geometry])).stdout)

    async def cursor_position(self) -> Point:
        output = (await self._run(['xdotool', 'getmouselocation', '--shell'])).stdout
        fields = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        try:
            return Point(x=int(fields['X']), y=int(fields['Y']))
        except (KeyError, ValueError):
            raise WorkspaceError(f'Unexpected `xdotool getmouselocation` output: {output.strip()!r}') from None

    async def act(self, action: ComputerAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await self._xdotool(
                    [*_move(x, y), 'click', '--repeat', str(count), _code(_BUTTONS, button)], hold=modifiers
                )
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
                await self._xdotool(['mousedown', _code(_BUTTONS, button)])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _code(_BUTTONS, button)])
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                wheel = _code(_WHEEL_BUTTONS, direction)
                await self._xdotool([*_move(x, y), 'click', '--repeat', str(amount), wheel], hold=modifiers)
            case TypeText(text=text):
                delay = [] if text.isascii() else ['--delay', _UNICODE_TYPE_DELAY_MS]
                await self._xdotool(['type', *delay, '--', text])
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ('keyup', *reversed(keysyms))
                await self._xdotool(['key', '--repeat', str(repeat), '--', '+'.join(keysyms)], release=release)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys)
            case _:
                raise UnsupportedComputerActionError(f'The display cannot perform {action!r}.')
