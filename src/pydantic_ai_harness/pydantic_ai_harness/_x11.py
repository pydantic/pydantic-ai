"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import re
from collections.abc import Callable, Sequence
from contextlib import suppress

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
from pydantic_ai.workspaces import CommandResult, SupportsCommands, WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness._display import LazyDisplay, png_from_base64
from pydantic_ai_harness._drag import DRAG_PRESS_PAUSE, DRAG_STEP_PAUSE, drag_steps
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError

DISPLAY = ':1'
"""The X display the desktop runs on, as `DISPLAY` names it."""

DISPLAY_NUMBER = DISPLAY[1:]

START_LOCK = f'exec 9>/tmp/.pydantic-ai-display-{DISPLAY_NUMBER}.lock; ! command -v flock >/dev/null || flock 9; '
"""The first line of a start script: waits for any other start of this display to finish, so two don't both start
one. A program the script starts in the background must close the lock with `9>&-`, or it holds the lock while it
runs."""


def wait_until_ready(ready: str, pid: str, what: str, log: str) -> str:
    """A shell fragment that waits for a program just started in the background, until the command `ready` succeeds.

    It fails once process `pid` exits, or after 10 seconds, saying `what` didn't start and printing the end of `log`.
    `ready` must answer for the new process only: empty any log it reads before starting the program, or the check can
    read the last run's log before the program's redirect replaces it.
    """
    return (
        f'i=0; until {ready}; do '
        f'if ! kill -0 {pid} 2>/dev/null || [ $i -ge 100 ]; then '
        f"echo '{what} did not start:' >&2; tail -n 5 {log} >&2; exit 1; fi; "
        'sleep 0.1; i=$((i+1)); done; '
    )


_MAX_REPEAT = 2**31 - 1
"""The most times xdotool repeats a click or key, since it reads `--repeat` into a C `int`."""


def _repeat(count: int) -> str:
    """`count` as xdotool's `--repeat`, refused when xdotool can't take it."""
    if count > _MAX_REPEAT:
        raise UnsupportedActionError(f'The display can repeat a scroll or key at most {_MAX_REPEAT} times.')
    return str(count)


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
"""Seconds one display command may take, on top of the pauses it asks for, such as a `hold_key`'s seconds."""

_TYPING_DELAY_MS = '12'
"""Milliseconds between typed characters, `xdotool type`'s default, and between the presses of a repeated key."""

_KEY_SECONDS = int(_TYPING_DELAY_MS) / 1000
"""The pause one typed character or one key press asks for."""

_COMMAND_BYTES = 60_000
"""The longest one input command may be, counting each argument as its bytes plus `_ARGUMENT_BYTES`. Longer input,
such as long text or a drag through many points, is split into several commands.

Sprites and Modal refuse commands over 64 KiB; the rest is room for the command's start and the URL's other
parameters."""

_ARGUMENT_BYTES = 6
"""What a transport adds to each argument, such as the `&cmd=` Sprites' exec URL puts before each."""

_LEVEL_SHIFTS = ('Shift_L', 'ISO_Level3_Shift')
"""The modifiers `xdotool` presses to reach a character on a key's other levels, released after a failed typing."""

_RELEASE_MODIFIERS = (
    'keyup',
    *('Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Super_L', 'Super_R', 'ISO_Level3_Shift'),
)
"""Starts every `xdotool` chain that acts: lets go of modifiers an earlier action left down, such as one cancelled
mid-press, since a stuck Shift types `a` as `A`.

Mouse buttons stay down, since `MouseDown` holds one across actions, and so does Caps Lock, which the model may have
turned on."""

_SCROLL_TICK_DELAY_MS = '20'
"""Milliseconds between scroll wheel clicks; `xdotool`'s default is 100."""

_SCROLL_TICK_SECONDS = int(_SCROLL_TICK_DELAY_MS) / 1000

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_DISPLAY_SIZE = ['xwininfo', '-root']
"""Prints the root window's size, which spans every monitor as a screenshot does. `xdotool getdisplaygeometry` gives
only the first monitor's, which would scale a click on a second monitor onto the first."""

_ROOT_SIZE = re.compile(r'^\s*(Width|Height): (\d+)$', re.MULTILINE)

_CAPTURES = '/tmp/pydantic-ai-captures'
"""Where screenshots are written before they are printed. Each capture removes files there older than five minutes,
such as one a cancel left behind."""

_CAPTURE = (
    f'd={_CAPTURES}; mkdir -p "$d" && find "$d" -type f -mmin +5 -exec rm -f {{}} + 2>/dev/null; f="$d/$$.png"; '
    'scrot "$@" "$f" && base64 < "$f"; status=$?; rm -f "$f"; exit $status'
)
"""Captures to a file of its own (scrot picks the format from the `.png` name) and prints it as base64; arguments
go to scrot, such as `-a x,y,width,height` for a region."""

_INPUT_PID = f'/tmp/.pydantic-ai-input-{DISPLAY_NUMBER}'
"""Holds the process id of the input command running on the display; `ComputerUse` runs one at a time."""

_INPUT = f'echo $$ > {_INPUT_PID}; exec "$@"'
"""Runs the input command in `$@` as the process whose id `_INPUT_PID` records."""

_STOP_INPUT = (
    f'p=$(cat {_INPUT_PID} 2>/dev/null) || exit 0; [ "$(cat /proc/$p/comm 2>/dev/null)" = xdotool ] || exit 0; '
    'kill "$p" 2>/dev/null; '
    'while [ -e /proc/$p ] && ! grep -q "^State:[[:space:]]*Z" /proc/$p/status 2>/dev/null; do sleep 0.05; done'
)
"""Stops the input command `_INPUT_PID` names, if it still runs, and waits for it to exit. A cancelled command can
outlive its call and press keys again after they were released."""


def _keysym(key: str) -> str:
    if key in _KEYSYMS:
        return _KEYSYMS[key]
    if len(key) == 1:
        # A character such as `!`: xdotool presses shift when the layout needs it.
        return key if key.isascii() and key.isalnum() else _character(key)
    if key in _PLAIN_KEYSYMS:
        return key
    # Validation refuses other names already; this also keeps out a name xdotool reads as a command, such as `exec`.
    raise UnsupportedActionError(f'The display cannot press the key {key!r}.')


def _character(character: str) -> str:
    """The keysym that types `character`, which must be ASCII.

    `xdotool` types a character that has no key, such as `é`, by remapping a spare key for each keystroke, and an
    application that reads the keymap after the keystroke, such as Chrome, then gets a different character or none.
    """
    if not character.isascii():
        raise UnsupportedActionError(f'The display can only type ASCII characters, not {character!r}.')
    if character == '\n':
        return 'Return'  # `xdotool type` sends Linefeed, which browsers ignore
    if character == '\t':
        return 'Tab'
    return f'U{ord(character):04X}'


def _drag_moves(path: Sequence[Point]) -> list[list[str]]:
    """The `xdotool` steps that move along `path` from its first point, in short moves with pauses between them."""
    return [['sleep', str(DRAG_STEP_PAUSE), *_move(point.x, point.y)] for point in drag_steps(path)]


def _commands(steps: Sequence[list[str]]) -> list[list[str]]:
    """`steps`, each arguments that go together, packed in order into as few commands as keep within `_COMMAND_BYTES`."""
    commands: list[list[str]] = [[]]
    size = 0
    for step in steps:
        step_size = sum(len(arg) + _ARGUMENT_BYTES for arg in step)
        if commands[-1] and size + step_size > _COMMAND_BYTES:
            commands.append([])
            size = 0
        commands[-1] += step
        size += step_size
    return commands


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


_ENV = {'DISPLAY': DISPLAY}


_X_LOCK = f'/tmp/.X{DISPLAY[1:]}-lock'
_X_SOCKET = f'/tmp/.X11-unix/X{DISPLAY[1:]}'


def start_x_server(log: str, *, then: str = '') -> str:
    """A shell fragment that starts Xvfb on `DISPLAY`, at the size `$1` names such as `1280x800`, unless one answers.

    It fails once the X server exits, or after 10 seconds, printing the end of `log`. `then` runs once a new X server
    answers, such as to start a panel.

    It takes `START_LOCK`, so a second start waits and finds the first one's X server. The lock is held until the
    script exits, so the rest of the script runs under it too; anything started in the background must close it with
    `9>&-`.
    """
    return (
        f'{START_LOCK}if ! xdotool getdisplaygeometry >/dev/null 2>&1; then '
        # A stale lock, such as one left by `docker stop`, stops Xvfb starting once its PID belongs to another process.
        # A running X server's lock stays, whichever server it is, such as `Xvnc`.
        f'case "$(cat /proc/$(tr -d " " 2>/dev/null <{_X_LOCK})/comm 2>/dev/null)" in X*) ;; '
        f'*) rm -f {_X_LOCK} {_X_SOCKET} ;; esac; '
        # Without `-noreset`, X resets when its last client exits, such as an `xdotool` command with no window manager
        # running: the pointer goes back to the middle and held buttons are let go.
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp -noreset >/dev/null 2>{log} 9>&- & '
        + wait_until_ready('xdotool getdisplaygeometry >/dev/null 2>&1', '$!', 'the X server', log)
        + f'{then} fi; '
    )


def x11_display(
    backend: SupportsCommands,
    size: tuple[int, int],
    script: str,
    where: Callable[[], str],
) -> LazyDisplay:
    """A backend's display on an X server that `script` starts at the size `$1` names, unless one answers.

    `where` names the environment in errors.
    """

    async def start() -> None:
        width, height = size
        # On `DISPLAY` whatever the backend's own env says, so it checks the X server the actions go to.
        result = await backend.run(['sh', '-c', script, 'sh', f'{width}x{height}'], env=_ENV, timeout=30)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'Could not start the display in {where()}: it needs Xvfb, xdotool, xwininfo, scrot and setsid '
                f'({(result.stderr or result.stdout).strip()})'
            )

    return LazyDisplay(X11Computer(backend), start)


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `DISPLAY`. Needs `xdotool`, `xwininfo` (in Debian's `x11-utils`), `scrot`
    and `base64` in the environment.
    """

    def __init__(self, backend: SupportsCommands):
        self._backend = backend

    async def _run(self, command: list[str] | str, *, pauses: float = 0) -> CommandResult:
        """Run `command`, allowing `pauses`, the seconds it asks to pause for, on top of the usual time limit."""
        timeout = _COMMAND_TIMEOUT + pauses
        result = await self._backend.run(command, shell=isinstance(command, str), env=_ENV, timeout=timeout)
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

    async def _xdotool(
        self, *chains: list[str], hold: tuple[str, ...] = (), release: tuple[str, ...] = (), pauses: float = 0
    ) -> None:
        """Run `xdotool` command chains in turn with `hold` held down; if one fails or is cancelled, let go of everything.

        Several chains are one input split by `_commands`: what it holds stays down from the first to the last.
        """
        keysyms = [_keysym(key) for key in hold]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        keydown = ['keydown', *keysyms] if keysyms else []
        commands = [['xdotool', *chain] for chain in chains]
        commands[0][1:1] = [*_RELEASE_MODIFIERS, *keydown]
        commands[-1] += keyup
        await self._input(commands, [list(release), keyup], pauses)

    async def _input(self, commands: Sequence[list[str]], releases: list[list[str]], pauses: float = 0) -> None:
        """Run input commands in turn; if one fails or is cancelled, stop it and let go of what they may hold.

        Each command may take `pauses` on top of the usual time limit. Each of `releases` is an `xdotool` chain, run
        on its own so one that fails doesn't keep the others from running.
        """
        stderr = ''
        try:
            for command in commands:
                stderr += (await self._run(['sh', '-c', _INPUT, 'sh', *command], pauses=pauses)).stderr
        except BaseException:
            with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True):
                # Before letting go, or the command could press again what was just released.
                with suppress(Exception):
                    await self._run(['sh', '-c', _STOP_INPUT])
                for release in releases:
                    if release:
                        with suppress(Exception):
                            await self._run(['xdotool', *release])
            raise
        # xdotool skips a key name the display doesn't know and still succeeds.
        if 'No such key name' in stderr:
            raise UnsupportedActionError(f'The display cannot press that key: {stderr.strip()}')

    async def _capture(self, script: str, *args: str) -> bytes:
        """Run a capture script with `args` and read the PNG it prints as base64."""
        return png_from_base64((await self._run(['sh', '-c', script, 'sh', *args])).stdout, 'The screenshot command')

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(_DISPLAY_SIZE)).stdout
        root = dict(_ROOT_SIZE.findall(output))
        try:
            return int(root['Width']), int(root['Height'])
        except KeyError:
            raise WorkspaceError(f'Unexpected display size output: {output.strip()!r}') from None

    async def screenshot(self) -> bytes:
        return await self._capture(_CAPTURE)

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

    async def act(self, action: DisplayAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await self._xdotool(
                    [*_move(x, y), 'click', '--repeat', str(count), _BUTTONS[button]],
                    hold=modifiers,
                    release=('mouseup', _BUTTONS[button]),
                )
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                await self._xdotool(_move(x, y), hold=modifiers)
            case Drag(path=path, modifiers=modifiers):
                start = path[0]
                chains = _commands(_drag_moves(path))
                chains[0][:0] = [*_move(start.x, start.y), 'mousedown', '1', 'sleep', str(DRAG_PRESS_PAUSE)]
                chains[-1] += ['sleep', str(DRAG_PRESS_PAUSE), 'mouseup', '1']
                await self._xdotool(
                    *chains,
                    hold=modifiers,
                    release=('mouseup', '1'),
                    pauses=2 * DRAG_PRESS_PAUSE + len(drag_steps(path)) * DRAG_STEP_PAUSE,
                )
            case MouseDown(button=button):
                await self._xdotool(['mousedown', _BUTTONS[button]])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _BUTTONS[button]])
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                wheel = _WHEEL_BUTTONS[direction]
                await self._xdotool(
                    [*_move(x, y), 'click', '--repeat', _repeat(amount), '--delay', _SCROLL_TICK_DELAY_MS, wheel],
                    hold=modifiers,
                    pauses=amount * _SCROLL_TICK_SECONDS,
                )
            case TypeText(text=text):
                for part in _commands([[_character(character)] for character in text]):
                    # A cancel can land between a key's press and release, and X then repeats the key without end.
                    release = ['keyup', *_LEVEL_SHIFTS, *dict.fromkeys(part)]
                    command = ['xdotool', *_RELEASE_MODIFIERS, 'key', '--delay', _TYPING_DELAY_MS, '--', *part]
                    await self._input([command], [release], len(part) * _KEY_SECONDS)
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                release = ['keyup', *reversed(keysyms)]
                command = ['xdotool', *_RELEASE_MODIFIERS, 'key', '--repeat', _repeat(repeat), '--', '+'.join(keysyms)]
                # xdotool pauses after each key of the chord, not once per repeat.
                await self._input([command], [release], repeat * len(keysyms) * _KEY_SECONDS)
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys, pauses=seconds)
            case _:
                raise UnsupportedActionError(f'The display cannot perform {action!r}.')
