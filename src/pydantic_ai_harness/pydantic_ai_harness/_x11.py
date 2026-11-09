"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

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
from pydantic_ai_harness._display import ManagedDisplay, png_from_base64
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
"""Key names whose X keysym is spelled differently; `_PLAIN_KEYSYMS` and ASCII letters and digits are keysyms already."""

_PLAIN_KEYSYMS = frozenset({'ctrl', 'shift', 'alt', 'super', 'space'})
"""Key names xdotool takes as they are."""

_COMMAND_TIMEOUT = 150
"""Maximum time for one display command; holding keys for the longest wait, 30 s, is the longest bounded action."""

_TYPING_DELAY_MS = '12'
"""Milliseconds between typed characters, `xdotool type`'s default."""

_LEVEL_SHIFTS = ('Shift_L', 'ISO_Level3_Shift')
"""The modifiers `xdotool` presses to reach a character on a key's other levels, released after a failed typing."""

_RELEASE_MODIFIERS = (
    'keyup',
    *('Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Super_L', 'Super_R', 'ISO_Level3_Shift'),
)
"""Starts every `xdotool` chain that acts: lets go of modifiers an earlier action left down, such as one stopped
between pressing and releasing, which would otherwise change what follows (a stuck Shift types `a` as `A`).

Mouse buttons stay, since `MouseDown` holds one across actions, and so does Caps Lock, which the model may have
turned on."""

_SCROLL_TICK_DELAY_MS = '20'
"""Milliseconds between scroll wheel clicks, instead of `xdotool`'s 100 ms between clicks."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_DISPLAY_SIZE = 'if command -v xwininfo >/dev/null; then xwininfo -root; else xdotool getdisplaygeometry; fi'
"""Prints the root window's size, which spans every monitor as a screenshot does. `xdotool getdisplaygeometry`, used
where `xwininfo` is missing, gives only the first monitor's."""

_ROOT_SIZE = re.compile(r'^\s*(Width|Height): (\d+)$', re.MULTILINE)

_CAPTURE = 'd=$(mktemp -d) && scrot "$@" "$d/s.png" && base64 < "$d/s.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64; arguments
go to scrot, such as `-a x,y,width,height` for a region."""

_CAPTURE_SCALED = (
    'd=$(mktemp -d) && scrot -t "$1" "$d/s.png" && base64 < "$d/s-thumb.png"; status=$?; rm -rf "$d"; exit $status'
)
"""Captures the whole screen and prints a copy scaled to the `WIDTHxHEIGHT` in `$1`, which scrot writes beside it."""

STARTED = 'pydantic-ai-display-started'
"""What a start script prints when it started a new X server, rather than finding one running."""

RECORD_START = f'echo {STARTED}; '
"""What a start script runs once a new X server answers: prints `STARTED`."""


def _keysym(key: str) -> str:
    if key in _KEYSYMS:
        return _KEYSYMS[key]
    if len(key) == 1:
        # A character such as `!`: xdotool presses shift when the layout needs it.
        return key if key.isascii() and key.isalnum() else _character(key)
    if key in _PLAIN_KEYSYMS:
        return key
    # Validation refuses other names; refusing them here too keeps out one xdotool would read as a command, such as
    # `exec`.
    raise WorkspaceUnsupportedActionError(f'The display cannot press the key {key!r}.')


def _character(character: str) -> str:
    """The keysym that types `character`, which must be ASCII.

    `xdotool` types a character that has no key, such as `é`, by remapping a spare key for each keystroke, and an
    application that reads the keymap after the keystroke, such as Chrome, then gets a different character or none.
    """
    if not character.isascii():
        raise WorkspaceUnsupportedActionError(f'The display can only type ASCII characters, not {character!r}.')
    if character == '\n':
        return 'Return'  # `xdotool type` sends Linefeed, which browsers ignore
    if character == '\t':
        return 'Tab'
    return f'U{ord(character):04X}'


def _drag_moves(path: Sequence[Point]) -> list[str]:
    """`xdotool` arguments that move along `path` from its first point, in short steps with pauses between them."""
    return [arg for point in drag_steps(path) for arg in ['sleep', str(DRAG_STEP_PAUSE), *_move(point.x, point.y)]]


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


DISPLAY = ':1'
"""The X display the desktop runs on, as `DISPLAY` names it."""

_ENV = {'DISPLAY': DISPLAY}


_X_LOCK = f'/tmp/.X{DISPLAY[1:]}-lock'
_X_SOCKET_DIR = '/tmp/.X11-unix'
_X_SOCKET = f'{_X_SOCKET_DIR}/X{DISPLAY[1:]}'

_PROGRAMS = ('Xvfb', 'xdotool', 'scrot', 'setsid')
_PACKAGES = 'xvfb xdotool scrot util-linux'
"""The Debian and Ubuntu packages that have `_PROGRAMS`."""

_START_TIMEOUT = 600
"""Seconds a start script gets, which covers installing `_PACKAGES`."""


def start_x_server(log: str, *, then: str = '') -> str:
    """A shell fragment that starts Xvfb on `DISPLAY`, at the size `$1` names such as `1280x800`, unless one answers.

    It first installs any of `_PROGRAMS` that's missing with `apt-get`, as root or through `sudo -n`, and makes the
    directory for the X server's socket, which Xvfb makes only when it runs as root. It gives up as soon as the X
    server exits, or after 10 seconds, with the end of `log` as the reason, and runs `RECORD_START` once a new X
    server answers. `then` runs then, such as to start a panel.
    """
    return (
        'if ! xdotool getdisplaygeometry >/dev/null 2>&1; then '
        f'missing=; for p in {" ".join(_PROGRAMS)}; do command -v $p >/dev/null 2>&1 || missing="$missing $p"; done; '
        # `elevate` prefixes a command that needs root: empty as root, `sudo -n` with passwordless sudo, and
        # `none` when neither works.
        'if [ "$(id -u)" = 0 ]; then elevate=; elif sudo -n true >/dev/null 2>&1; then elevate="sudo -n"; '
        'else elevate=none; fi; '
        'if [ -n "$missing" ]; then '
        'if ! command -v apt-get >/dev/null 2>&1; then '
        """echo "missing$missing, and there's no apt-get to install them with" >&2; exit 1; fi; """
        'if [ "$elevate" = none ]; then '
        """echo "missing$missing, and there's no root or passwordless sudo to install them with" >&2; exit 1; fi; """
        '{ $elevate apt-get update -qq && $elevate env DEBIAN_FRONTEND=noninteractive '
        f'apt-get install -y -qq --no-install-recommends {_PACKAGES}; }} >{log} 2>&1 '
        f'|| {{ echo "could not install$missing:" >&2; tail -n 5 {log} >&2; exit 1; }}; fi; '
        # Without root, a directory the user makes themselves works, so the mkdir runs unprefixed.
        f'if [ ! -d {_X_SOCKET_DIR} ]; then [ "$elevate" = none ] && elevate=; '
        f'$elevate mkdir -p {_X_SOCKET_DIR} && $elevate chmod 1777 {_X_SOCKET_DIR}; fi; '
        # A lock left by an X server that's gone, such as after `docker stop`, stops Xvfb starting when its PID now
        # belongs to another process. A running X server's lock stays, whichever server it is, such as `Xvnc`.
        f'case "$(cat /proc/$(tr -d " " 2>/dev/null <{_X_LOCK})/comm 2>/dev/null)" in X*) ;; '
        f'*) rm -f {_X_LOCK} {_X_SOCKET} ;; esac; '
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp >/dev/null 2>{log} & xvfb=$!; i=0; '
        'until xdotool getdisplaygeometry >/dev/null 2>&1; do '
        'if ! kill -0 $xvfb 2>/dev/null || [ $i -ge 100 ]; then '
        f"echo 'the X server did not start:' >&2; tail -n 5 {log} >&2; exit 1; fi; "
        f'sleep 0.1; i=$((i+1)); done; {RECORD_START}{then} fi; '
    )


def x11_display(
    backend: SupportsCommands,
    size: tuple[int, int],
    script: str,
    where: Callable[[], str],
    *,
    screen_key: Callable[[], object | None],
) -> ManagedDisplay:
    """A backend's display on an X server that `script` starts at the size `$1` names, unless one answers.

    `script` runs `RECORD_START` when it starts a new X server. `where` names the environment in errors.
    """

    async def start() -> X11Computer:
        width, height = size
        # On `DISPLAY` whatever the backend's own env says, so it checks the X server the actions go to.
        result = await backend.run(['sh', '-c', script, 'sh', f'{width}x{height}'], env=_ENV, timeout=_START_TIMEOUT)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'Could not start the display in {where()}: it needs Xvfb, xdotool, scrot and setsid '
                f'({(result.stderr or result.stdout).strip()})'
            )
        if STARTED in result.stdout.splitlines():
            display.started_fresh()
        return X11Computer(backend)

    display = ManagedDisplay(start, screen_key=screen_key)
    return display


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `DISPLAY`. Needs `xdotool`, `scrot`, `base64`, and `mktemp` in the environment.
    """

    def __init__(self, backend: SupportsCommands):
        self._backend = backend

    async def _run(self, command: list[str] | str) -> CommandResult:
        result = await self._backend.run(command, shell=isinstance(command, str), env=_ENV, timeout=_COMMAND_TIMEOUT)
        if result.exit_code == 127:
            raise WorkspaceUnavailableError(
                f'A display command is missing on display {DISPLAY}: {(result.stderr or result.stdout).strip()}'
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
        keydown = ['keydown', *keysyms] if keysyms else []
        await self._input(['xdotool', *_RELEASE_MODIFIERS, *keydown, *args, *keyup], [*release, *keyup])

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

    async def _capture(self, script: str, *args: str) -> bytes:
        """Run a capture script with `args` and read the PNG it prints as base64."""
        return png_from_base64((await self._run(['sh', '-c', script, 'sh', *args])).stdout, 'The screenshot command')

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(['sh', '-c', _DISPLAY_SIZE])).stdout
        root = dict(_ROOT_SIZE.findall(output))
        try:
            width, height = (root['Width'], root['Height']) if root else output.split()
            return int(width), int(height)
        except (KeyError, ValueError):
            raise WorkspaceError(f'Unexpected display size output: {output.strip()!r}') from None

    async def screenshot(self) -> bytes:
        return await self._capture(_CAPTURE)

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        """The whole display scaled to `width` by `height`."""
        return await self._capture(_CAPTURE_SCALED, f'{width}x{height}')

    async def zoom(self, region: Region) -> bytes:
        geometry = f'{region.left},{region.top},{region.right - region.left},{region.bottom - region.top}'
        return await self._capture(_CAPTURE, '-a', geometry)

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
                keysyms = [_character(character) for character in text]
                # A cancel can land between a key's press and release, and X then repeats the key without end.
                release = ['keyup', *_LEVEL_SHIFTS, *dict.fromkeys(keysyms)]
                command = ['xdotool', *_RELEASE_MODIFIERS, 'key', '--delay', _TYPING_DELAY_MS, '--', *keysyms]
                await self._input(command, release)
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ['keyup', *reversed(keysyms)]
                command = ['xdotool', *_RELEASE_MODIFIERS, 'key', '--repeat', str(repeat), '--', '+'.join(keysyms)]
                await self._input(command, release)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys)
            case _:
                raise WorkspaceUnsupportedActionError(f'The display cannot perform {action!r}.')
