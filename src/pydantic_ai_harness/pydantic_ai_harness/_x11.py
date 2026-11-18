"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

import anyio

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
from pydantic_ai.workspaces import CommandResult, WorkspaceError, WorkspaceUnavailableError
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
        raise UnsupportedActionError(f'The display can repeat a click, scroll or key at most {_MAX_REPEAT} times.')
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
"""Key names whose X keysym is spelled differently; the other names, such as `ctrl`, and ASCII letters and digits are
keysyms already."""

_COMMAND_TIMEOUT = 150
"""Seconds one display command may take, on top of the pauses it asks for, such as a `hold_keys`'s seconds."""

_TYPING_DELAY_MS = '12'
"""Milliseconds between typed characters, `xdotool type`'s default, and between the presses of a repeated key."""

_KEY_SECONDS = int(_TYPING_DELAY_MS) / 1000
"""The pause one typed character or one key press asks for."""

_CLICK_SECONDS = 0.1
"""The pause between the clicks of a repeated click, `xdotool click`'s default delay."""

_SCROLL_TICK_DELAY_MS = '20'
"""Milliseconds between scroll wheel clicks; `xdotool`'s default is 100."""

_SCROLL_TICK_SECONDS = int(_SCROLL_TICK_DELAY_MS) / 1000
"""Time per wheel click, as measured: 20 ms each."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_DISPLAY_SIZE = ['xwininfo', '-root']
"""Prints the root window's size, which spans every monitor as a screenshot does. `xdotool getdisplaygeometry` gives
only the first monitor's, which would scale a click on a second monitor onto the first."""

_CAPTURE = (
    'f=/tmp/pydantic-ai-capture-$$.png; scrot --overwrite "$@" "$f" && base64 < "$f"; status=$?; rm -f "$f"; '
    'exit $status'
)
"""Captures to a file of its own (scrot picks the format from the `.png` name) and prints it as base64; arguments
go to scrot, such as `-a x,y,width,height` for a region. `--overwrite` replaces a file a cancel left at a reused
process id; without it scrot saves under another name and the old file would be printed."""

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
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A character such as `!`: xdotool presses shift when the layout needs it.
        return _character(key)
    return key


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


def _move(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


_ENV = {'DISPLAY': DISPLAY}

_START_TIMEOUT = 70
"""Seconds a start script may take: up to three 10-second waits of its own, after up to as long behind `START_LOCK`
for another start, plus 10 seconds for its commands."""


class RunCommand(Protocol):
    """Runs a command in the environment, without starting its display first, as the display's own commands must."""

    def __call__(self, argv: list[str], /, *, env: Mapping[str, str], timeout: float) -> Awaitable[CommandResult]: ...


def x11_display(
    run: RunCommand,
    size: tuple[int, int],
    script: str,
    where: Callable[[], str],
) -> LazyDisplay:
    """A backend's display on an X server that `script` starts at the size `$1` names, unless one runs.

    `where` names the environment in errors.
    """

    async def start() -> None:
        width, height = size
        # On `DISPLAY` whatever the backend's own env says, so it checks the X server the actions go to.
        result = await run(['sh', '-c', script, 'sh', f'{width}x{height}'], env=_ENV, timeout=_START_TIMEOUT)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'Could not start the display in {where()}: it needs Xvfb, xdotool, xwininfo, scrot and setsid '
                f'({(result.stderr or result.stdout).strip()})'
            )

    return LazyDisplay(X11Computer(run), start)


class X11Computer:
    """Controls an X display with `xdotool` and captures it with `scrot`, through a backend's commands.

    The X server must already be running on `DISPLAY`. Needs `xdotool`, `xwininfo` (in Debian's `x11-utils`), `scrot`
    and `base64` in the environment.
    """

    def __init__(self, run: RunCommand):
        self._run_command = run

    async def _run(self, command: list[str], *, pauses: float = 0) -> CommandResult:
        """Run `command`, allowing `pauses`, the seconds it asks to pause for, on top of the usual time limit."""
        timeout = _COMMAND_TIMEOUT + pauses
        result = await self._run_command(command, env=_ENV, timeout=timeout)
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
        self, chain: list[str], hold: tuple[str, ...] = (), release: tuple[str, ...] = (), pauses: float = 0
    ) -> None:
        """Run an `xdotool` command chain with `hold` held down; if it fails or is cancelled, let go of everything.

        `release` is an `xdotool` chain that lets go of what the chain presses; the command may take `pauses` on top of
        the usual time limit.
        """
        keysyms = [_keysym(key) for key in hold]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        keydown = ['keydown', *keysyms] if keysyms else []
        try:
            await self._run(['sh', '-c', _INPUT, 'sh', 'xdotool', *keydown, *chain, *keyup], pauses=pauses)
        except BaseException as error:
            # Each release runs on its own, so one that fails doesn't keep the other from running.
            failures = await self._release([list(release), keyup])
            if failures and isinstance(error, WorkspaceError) and not isinstance(error, WorkspaceUnavailableError):
                raise WorkspaceError(
                    f'{error}. Releasing what it pressed failed too, so a mouse button or key may still be held: '
                    + '; '.join(failures)
                ) from error
            raise

    async def _release(self, releases: list[list[str]]) -> list[str]:
        """Stop the input command, then run each of `releases`; return what failed or didn't get to run in time."""
        failures: list[str] = []
        with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True) as releasing:
            # Stopping comes before letting go, or the command could press again what was just released.
            for cleanup in [['sh', '-c', _STOP_INPUT], *(['xdotool', *release] for release in releases if release)]:
                try:
                    await self._run(cleanup)
                except WorkspaceUnavailableError:
                    raise
                except Exception as e:
                    failures.append(str(e))
        if releasing.cancelled_caught:
            failures.append(f'releasing took longer than {_RELEASE_TIMEOUT} seconds')
        return failures

    async def _capture(self, *args: str) -> bytes:
        """Run `_CAPTURE` with `args` and read the PNG it prints as base64."""
        return png_from_base64((await self._run(['sh', '-c', _CAPTURE, 'sh', *args])).stdout, 'The screenshot command')

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run(_DISPLAY_SIZE)).stdout
        # `[ \t]*` stays on one line; `\s*` would take time growing with the square of the output's length.
        width, height = (
            re.search(rf'^[ \t]*{name}: (\d{{1,5}})$', output, re.MULTILINE) for name in ('Width', 'Height')
        )
        if width is None or height is None:
            raise WorkspaceError(f'Unexpected display size output: {output.strip()!r}')
        size = int(width[1]), int(height[1])
        # X11 sizes are 16-bit; a bigger one would make a drag plan that many points.
        if not all(1 <= side <= 65535 for side in size):
            raise WorkspaceError(f'The display reported an impossible size: {size[0]}x{size[1]}')
        return size

    async def screenshot(self) -> bytes:
        return await self._capture()

    async def zoom(self, region: Region) -> bytes:
        geometry = f'{region.left},{region.top},{region.right - region.left},{region.bottom - region.top}'
        return await self._capture('-a', geometry)

    async def cursor_position(self) -> Point:
        output = (await self._run(['xdotool', 'getmouselocation', '--shell'])).stdout
        x, y = (re.search(rf'^{axis}=(\d{{1,5}})$', output, re.MULTILINE) for axis in ('X', 'Y'))
        if x is None or y is None:
            raise WorkspaceError(f'Unexpected `xdotool getmouselocation` output: {output.strip()!r}')
        return Point(x=int(x[1]), y=int(y[1]))

    async def act(self, action: DisplayAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await self._xdotool(
                    [*_move(x, y), 'click', '--repeat', _repeat(count), _BUTTONS[button]],
                    hold=modifiers,
                    release=('mouseup', _BUTTONS[button]),
                    pauses=count * _CLICK_SECONDS,
                )
            case MouseMove(x=x, y=y, modifiers=modifiers):
                await self._xdotool(_move(x, y), hold=modifiers)
            case Drag(path=path, modifiers=modifiers):
                start = path[0]
                steps = drag_steps(path)
                moves = [arg for point in steps for arg in ['sleep', str(DRAG_STEP_PAUSE), *_move(point.x, point.y)]]
                pause = ['sleep', str(DRAG_PRESS_PAUSE)]
                await self._xdotool(
                    [*_move(start.x, start.y), 'mousedown', '1', *pause, *moves, *pause, 'mouseup', '1'],
                    hold=modifiers,
                    release=('mouseup', '1'),
                    pauses=2 * DRAG_PRESS_PAUSE + len(steps) * DRAG_STEP_PAUSE,
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
                    release=('mouseup', wheel),
                    pauses=amount * _SCROLL_TICK_SECONDS,
                )
            case TypeText(text=text):
                keysyms = [_character(character) for character in text]
                # A cancel can land between a key's press and release, and X then repeats the key without end.
                # `xdotool` holds Shift_L for a shifted character such as `!`.
                await self._xdotool(
                    ['key', '--delay', _TYPING_DELAY_MS, '--', *keysyms],
                    release=('keyup', 'Shift_L', *dict.fromkeys(keysyms)),
                    pauses=len(keysyms) * _KEY_SECONDS,
                )
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                # xdotool pauses after each key of the chord, not once per repeat.
                await self._xdotool(
                    ['key', '--repeat', _repeat(repeat), '--', '+'.join(keysyms)],
                    release=('keyup', *reversed(keysyms)),
                    pauses=repeat * len(keysyms) * _KEY_SECONDS,
                )
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], hold=keys, pauses=seconds)
            case _:
                raise UnsupportedActionError(f'The display cannot perform {action!r}.')
