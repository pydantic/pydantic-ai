"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import base64
import re
from collections.abc import Callable, Sequence
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
    WorkspaceError,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness._drag import DRAG_PRESS_PAUSE, DRAG_STEP_PAUSE, drag_steps

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
    'printscreen': 'Print',
    'pause': 'Pause',
    'menu': 'Menu',
    'capslock': 'Caps_Lock',
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
"""Key names whose X keysym is spelled differently; other names (`ctrl`, `space`, `a`, ...) are keysyms already."""

_KEYSYM_PATTERN = re.compile(r'[A-Za-z0-9_]+')

_OUTSIDE_ASCII = re.compile(r'U(?!00[0-7])[0-9A-F]{4,}')
"""A keysym `_character` gives a character outside ASCII, which has no key until it's mapped to one."""

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
"""Maximum time for one display command; holding keys for the longest wait, 30 s, is the longest bounded action."""

_PRESS_MAPPED = (
    'mods=" $(xmodmap -pm | grep -o \'0x[0-9a-f]*\' | while read h; do printf \'%d \' "$h"; done)"; code=255; map=""; '
    'while [ "$1" != -- ]; do '
    'while case "$mods" in *" $code "*) true;; *) false;; esac; do code=$((code - 1)); done; '
    'map="$map -e \'keycode $code = $1 $1\'"; code=$((code - 1)); shift; done; shift; '
    'eval xmodmap "$map" && sleep 0.2 && xdotool key "$@"'
)
"""Runs as `sh -c PRESS_MAPPED sh <keysym>... -- <xdotool key arguments>...`: maps each keysym before `--` to a
keycode of its own, counting down from 255 past the modifier keys, then runs `xdotool key` with the arguments after
`--`.

`xdotool` alone types a character that has no key by remapping one spare keycode for each keystroke, and an
application that reads the keymap after the keystroke, such as Chrome, then sees a later character, or none when
every keycode is taken, as on E2B's desktop. Mapping a run's characters at once and waiting for applications to
see the new keymap types them reliably (checked live in Chrome and Firefox). The borrowed keycodes, which belong to
rarely used keys such as media keys, stay mapped."""

_MAPPED_PER_COMMAND = 50
"""The most distinct characters one command maps, well within the keycodes there are."""

_TYPING_DELAY_MS = '12'
"""Milliseconds between typed characters, `xdotool type`'s default."""

_LEVEL_SHIFTS = ('Shift_L', 'ISO_Level3_Shift')
"""The modifiers `xdotool` presses to reach a character on a key's other levels, released after a failed typing."""

_SCROLL_TICK_DELAY_MS = '20'
"""Milliseconds between scroll wheel clicks, instead of `xdotool`'s 100 ms between clicks."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_CAPTURE = 'd=$(mktemp -d) && scrot "$@" "$d/s.png" && base64 < "$d/s.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64; arguments
go to scrot, such as `-a x,y,width,height` for a region."""

_CAPTURE_SCALED = (
    'd=$(mktemp -d) && scrot -t "$1" "$d/s.png" && base64 < "$d/s-thumb.png"; status=$?; rm -rf "$d"; exit $status'
)
"""Captures the whole screen and prints a copy scaled to the `WIDTHxHEIGHT` in `$1`, which scrot writes beside it."""

STARTED = 'pydantic-ai-display-started'
"""What a start script prints when it started a new X server, rather than finding one running."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _keysym(key: str) -> str:
    if key in _KEYSYMS:
        return _KEYSYMS[key]
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A character such as `!` or `é`: xdotool presses shift or AltGr when the layout needs it.
        return _character(key)
    # xdotool reads any argument that names one of its commands, in any case, as that command.
    if not _KEYSYM_PATTERN.fullmatch(key) or key.lower() in _XDOTOOL_COMMANDS:
        raise WorkspaceUnsupportedActionError(f'The display cannot press the key {key!r}.')
    return key


def _character(character: str) -> str:
    """The keysym that types `character`; `xdotool` needs one outside ASCII mapped to a key first."""
    if character == '\n':
        return 'Return'  # `xdotool type` sends Linefeed, which browsers ignore
    if character == '\t':
        return 'Tab'
    return f'U{ord(character):04X}'


def _press(keysyms: Sequence[str], args: list[str]) -> list[str]:
    """`xdotool key` with `args`, after mapping the characters outside ASCII among `keysyms` to keys of their own."""
    mapped = [keysym for keysym in dict.fromkeys(keysyms) if _OUTSIDE_ASCII.fullmatch(keysym)]
    return ['sh', '-c', _PRESS_MAPPED, 'sh', *mapped, '--', *args] if mapped else ['xdotool', 'key', *args]


def _png(output: str) -> bytes:
    try:
        data = base64.b64decode(''.join(output.split()), validate=True)
    except ValueError:  # includes `binascii.Error`
        data = b''
    if not data.startswith(_PNG_SIGNATURE):
        raise WorkspaceError('The screenshot command did not print a base64-encoded PNG.')
    return data


def _drag_moves(path: Sequence[Point]) -> list[str]:
    """`xdotool` arguments that move along `path` from its first point, in short steps with pauses between them."""
    return [arg for point in drag_steps(path) for arg in ['sleep', str(DRAG_STEP_PAUSE), *_move(point.x, point.y)]]


def _typing(text: str) -> list[list[str]]:
    """The keysyms that type `text`, split into one command for each 50 distinct characters outside ASCII it maps.

    Most text is one command, so typing it costs one round trip and one wait for the new keymap.
    """
    chunks: list[list[str]] = [[]]
    mapped: set[str] = set()
    for character in text:
        keysym = _character(character)
        if not character.isascii() and keysym not in mapped:
            if len(mapped) == _MAPPED_PER_COMMAND:
                chunks.append([])
                mapped = set()
            mapped.add(keysym)
        chunks[-1].append(keysym)
    return [chunk for chunk in chunks if chunk]


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


DISPLAY = ':1'
"""The X display the desktop runs on, as `DISPLAY` names it."""

_LOCALE = {'LC_ALL': 'C.UTF-8'}
"""`xdotool type` reads its text in the locale's encoding, and stops at the first character outside ASCII under a
POSIX locale, which is what an environment without one set, such as E2B's, has."""


_X_LOCK = f'/tmp/.X{DISPLAY[1:]}-lock'
_X_SOCKET = f'/tmp/.X11-unix/X{DISPLAY[1:]}'


def start_x_server(log: str, *, then: str = '') -> str:
    """A shell fragment that starts Xvfb on `DISPLAY`, at the size `$1` names such as `1280x800`, unless one answers.

    It gives up as soon as the X server exits, or after 10 seconds, with the end of `log` as the reason, and prints
    `STARTED` once a new X server answers. `then` runs then, such as to start a panel.
    """
    return (
        'if ! xdotool getdisplaygeometry >/dev/null 2>&1; then '
        # A lock left by an X server that's gone, such as after `docker stop`, stops Xvfb starting when its PID now
        # belongs to another process. A running X server's lock stays, whichever server it is, such as `Xvnc`.
        f'case "$(cat /proc/$(tr -d " " 2>/dev/null <{_X_LOCK})/comm 2>/dev/null)" in X*) ;; '
        f'*) rm -f {_X_LOCK} {_X_SOCKET} ;; esac; '
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp -noreset >/dev/null 2>{log} & xvfb=$!; i=0; '
        'until xdotool getdisplaygeometry >/dev/null 2>&1; do '
        'if ! kill -0 $xvfb 2>/dev/null || [ $i -ge 100 ]; then '
        f"echo 'the X server did not start:' >&2; tail -n 5 {log} >&2; exit 1; fi; "
        f'sleep 0.1; i=$((i+1)); done; echo {STARTED}; {then} fi; '
    )


async def start_x11(
    backend: SupportsCommands, size: tuple[int, int], script: str, where: Callable[[], str]
) -> tuple[X11Computer, bool]:
    """Run `script`, which starts an X display unless one answers, and control that display.

    `script`, built with `start_x_server`, starts it at the size `$1` names and prints `STARTED` when it started a new
    X server. Returns the display's controller and whether it did. `where` names the environment in errors.
    """
    width, height = size
    result = await backend.run(['sh', '-c', script, 'sh', f'{width}x{height}'], env={'DISPLAY': DISPLAY}, timeout=30)
    if result.exit_code != 0:
        raise WorkspaceUnavailableError(
            f'could not start the display in {where()}: it needs Xvfb, xdotool, xmodmap, scrot and setsid '
            f'({(result.stderr or result.stdout).strip()})'
        )
    return X11Computer(backend), STARTED in result.stdout.split()


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `display`. Needs `xdotool`, `xmodmap`, `scrot`, `base64`, and `mktemp` in the
    environment.
    """

    def __init__(self, backend: SupportsCommands, display: str | None = DISPLAY):
        """`display` is the X display's `DISPLAY`, or `None` when the backend's commands already have it."""
        self._backend = backend
        self._display = display
        self._on = '' if display is None else f' on display {display}'

    async def _run(self, command: list[str] | str) -> CommandResult:
        env = _LOCALE if self._display is None else {'DISPLAY': self._display, **_LOCALE}
        result = await self._backend.run(command, shell=isinstance(command, str), env=env, timeout=_COMMAND_TIMEOUT)
        if result.exit_code == 127:
            raise WorkspaceUnavailableError(
                f'A display command is missing{self._on}: {(result.stderr or result.stdout).strip()}'
            )
        if result.exit_code != 0:
            raise WorkspaceError(
                f'Display command failed with exit code {result.exit_code}{self._on}: '
                f'{(result.stderr or result.stdout).strip()}'
            )
        return result

    async def _xdotool(self, args: list[str], *, hold: tuple[str, ...] = (), release: tuple[str, ...] = ()) -> None:
        """Run one `xdotool` command chain with `hold` held down; if it fails or is cancelled, let go of everything."""
        keysyms = [_keysym(key) for key in hold]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        await self._input(['xdotool', *(['keydown', *keysyms] if keysyms else []), *args, *keyup], [*release, *keyup])

    async def _input(self, command: list[str], release: list[str]) -> None:
        """Run an input command; if it fails or is cancelled, let go of what it may hold with `xdotool <release>`."""
        try:
            result = await self._run(command)
        except BaseException:
            if release:
                with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                    await self._run(['xdotool', *release])
            raise
        # xdotool skips a key name the display doesn't know and still succeeds.
        if 'No such key name' in result.stderr:
            raise WorkspaceUnsupportedActionError(f'The display cannot press that key: {result.stderr.strip()}')

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(['xdotool', 'getdisplaygeometry'])).stdout.split()
        try:
            width, height = output
            return int(width), int(height)
        except ValueError:
            raise WorkspaceError(f'Unexpected `xdotool getdisplaygeometry` output: {" ".join(output)!r}') from None

    async def screenshot(self) -> bytes:
        return _png((await self._run(['sh', '-c', _CAPTURE, 'sh'])).stdout)

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        """The whole display scaled to `width` by `height`."""
        return _png((await self._run(['sh', '-c', _CAPTURE_SCALED, 'sh', f'{width}x{height}'])).stdout)

    async def zoom(self, region: Region) -> bytes:
        geometry = f'{region.left},{region.top},{region.right - region.left},{region.bottom - region.top}'
        return _png((await self._run(['sh', '-c', _CAPTURE, 'sh', '-a', geometry])).stdout)

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
                await self._xdotool([*_move(x, y), 'click', '--repeat', str(count), _BUTTONS[button]], hold=modifiers)
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                await self._xdotool(_move(x, y), hold=modifiers)
            case Drag(path=path, modifiers=modifiers):
                start = path[0]
                await self._xdotool(
                    [*_move(start.x, start.y), 'mousedown', '1', 'sleep', str(DRAG_PRESS_PAUSE), *_drag_moves(path)]
                    + ['sleep', str(DRAG_PRESS_PAUSE), 'mouseup', '1'],
                    hold=modifiers,
                    release=('mouseup', '1'),
                )
            case MouseDown(button=button):
                await self._xdotool(['mousedown', _BUTTONS[button]])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _BUTTONS[button]])
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                wheel = _WHEEL_BUTTONS[direction]
                await self._xdotool(
                    [*_move(x, y), 'click', '--repeat', str(amount), '--delay', _SCROLL_TICK_DELAY_MS, wheel],
                    hold=modifiers,
                )
            case TypeText(text=text):
                for keysyms in _typing(text):
                    # A cancel can land between a key's press and release, and X then repeats the key without end.
                    release = ['keyup', *_LEVEL_SHIFTS, *dict.fromkeys(keysyms)]
                    await self._input(_press(keysyms, ['--delay', _TYPING_DELAY_MS, '--', *keysyms]), release)
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ['keyup', *reversed(keysyms)]
                await self._input(_press(keysyms, ['--repeat', str(repeat), '--', '+'.join(keysyms)]), release)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys)
            case _:
                raise WorkspaceUnsupportedActionError(f'The display cannot perform {action!r}.')
