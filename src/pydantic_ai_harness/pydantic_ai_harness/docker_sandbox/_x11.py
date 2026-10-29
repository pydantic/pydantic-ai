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

_OUTSIDE_ASCII = re.compile(r'U(?!00[0-7])[0-9A-F]{4,}')
"""A keysym `_character` gives a character outside ASCII, which has no key until it's mapped to one."""

_COMMAND_TIMEOUT = 150
"""Maximum time for one display command; holding keys for the longest wait, 30 s, is the longest bounded action."""

_PRESS_MAPPED = (
    'mods=" $(xmodmap -pm | grep -o \'0x[0-9a-f]*\' | while read h; do printf \'%d \' "$h"; done)"; code=255; map=""; '
    'while [ "$1" != -- ]; do '
    'while case "$mods" in *" $code "*) true;; *) false;; esac; do code=$((code - 1)); done; '
    'map="$map -e \'keycode $code = $1 $1\'"; code=$((code - 1)); shift; done; shift; '
    'eval xmodmap "$map" && sleep 0.2 && xdotool "$@"'
)
"""Runs as `sh -c PRESS_MAPPED sh <keysym>... -- <xdotool arguments>...`: maps each keysym before `--` to a keycode of
its own, counting down from 255 past the modifier keys, then runs `xdotool` with the arguments after `--`.

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

_CAPTURE = 'd=$(mktemp -d) && scrot "$@" "$d/s.png" && base64 < "$d/s.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64; arguments
go to scrot, such as `-a x,y,width,height` for a region."""

_CAPTURE_SCALED = (
    'd=$(mktemp -d) && scrot -t "$1" "$d/s.png" && base64 < "$d/s-thumb.png"; status=$?; rm -rf "$d"; exit $status'
)
"""Captures the whole screen and prints a copy scaled to the `WIDTHxHEIGHT` in `$1`, which scrot writes beside it."""

STARTED = 'pydantic-ai-display-started'
"""What a start script prints when it started a new X server, rather than finding one running."""


def _keysym(key: str) -> str:
    if key in _KEYSYMS:
        return _KEYSYMS[key]
    if len(key) == 1:
        # A character such as `!` or `é`: xdotool presses shift or AltGr when the layout needs it.
        return key if key.isascii() and key.isalnum() else _character(key)
    if key in _PLAIN_KEYSYMS:
        return key
    # Validation refuses other names; refusing them here too keeps out one xdotool would read as a command, such as
    # `exec`.
    raise WorkspaceUnsupportedActionError(f'The display cannot press the key {key!r}.')


def _character(character: str) -> str:
    """The keysym that types `character`; `xdotool` needs one outside ASCII mapped to a key first."""
    if character == '\n':
        return 'Return'  # `xdotool type` sends Linefeed, which browsers ignore
    if character == '\t':
        return 'Tab'
    return f'U{ord(character):04X}'


def _press(keysyms: Sequence[str], args: list[str]) -> list[str]:
    """`xdotool` with `args`, after mapping the characters outside ASCII among `keysyms` to keys of their own."""
    mapped = [keysym for keysym in dict.fromkeys(keysyms) if _OUTSIDE_ASCII.fullmatch(keysym)]
    return ['sh', '-c', _PRESS_MAPPED, 'sh', *mapped, '--', *args] if mapped else ['xdotool', *args]


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

_ENV = {'DISPLAY': DISPLAY, 'LC_ALL': 'C.UTF-8'}
"""`xdotool type` reads its text in the locale's encoding, and stops at the first character outside ASCII under a
POSIX locale, which is what an environment without one set, such as E2B's, has."""


async def start_x11(
    backend: SupportsCommands, size: tuple[int, int], script: str, where: Callable[[], str]
) -> tuple[X11Computer, bool]:
    """Run `script`, which starts an X display of the size `$1` names unless one answers, and control that display.

    Returns its controller and whether a new X server started, which `script` says by printing `STARTED`. `where`
    names the environment in errors.
    """
    width, height = size
    result = await backend.run(['sh', '-c', script, 'sh', f'{width}x{height}'], env={'DISPLAY': DISPLAY}, timeout=30)
    if result.exit_code != 0:
        raise WorkspaceUnavailableError(
            f'Could not start the display in {where()}: it needs Xvfb, xdotool, xmodmap, scrot and setsid '
            f'({(result.stderr or result.stdout).strip()})'
        )
    return X11Computer(backend), STARTED in result.stdout.split()


def x11_display(
    backend: SupportsCommands,
    size: tuple[int, int],
    script: str,
    where: Callable[[], str],
    *,
    screen_key: Callable[[], object | None],
) -> ManagedDisplay:
    """A backend's display on an X server that `script` starts, as `start_x11` runs it."""

    async def start() -> X11Computer:
        computer, started = await start_x11(backend, size, script, where)
        if started:
            display.started_fresh()
        return computer

    display = ManagedDisplay(start, screen_key=screen_key)
    return display


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `DISPLAY`. Needs `xdotool`, `xmodmap`, `scrot`, `base64`, and `mktemp` in the
    environment.
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
        output = (await self._run(['xdotool', 'getdisplaygeometry'])).stdout.split()
        try:
            width, height = output
            return int(width), int(height)
        except ValueError:
            raise WorkspaceError(f'Unexpected `xdotool getdisplaygeometry` output: {" ".join(output)!r}') from None

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
                for keysyms in _typing(text):
                    # A cancel can land between a key's press and release, and X then repeats the key without end.
                    release = ['keyup', *_LEVEL_SHIFTS, *dict.fromkeys(keysyms)]
                    command = [*_RELEASE_MODIFIERS, 'key', '--delay', _TYPING_DELAY_MS, '--', *keysyms]
                    await self._input(_press(keysyms, command), release)
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ['keyup', *reversed(keysyms)]
                command = [*_RELEASE_MODIFIERS, 'key', '--repeat', str(repeat), '--', '+'.join(keysyms)]
                await self._input(_press(keysyms, command), release)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys)
            case _:
                raise WorkspaceUnsupportedActionError(f'The display cannot perform {action!r}.')
