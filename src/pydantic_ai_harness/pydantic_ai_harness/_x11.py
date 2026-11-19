"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import re
import shlex
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


_MAX_CHAIN_BYTES = 100_000
"""The most bytes of `xdotool` arguments sent in one command. E2B runs a command as one `bash -l -c` argument, which
Linux caps at 131,072 bytes: a live probe started a 130,630-byte line and refused 130,664 bytes with "argument list
too long", and typing 25,000 characters in one command failed. The rest is room for the backend's own wrapping."""


def _chains(steps: list[list[str]], prefix: tuple[str, ...] = ()) -> list[list[str]]:
    """`steps` in order, grouped into `xdotool` chains that each start with `prefix` and fit `_MAX_CHAIN_BYTES`."""
    chains: list[list[str]] = []
    size = _MAX_CHAIN_BYTES
    for step in steps:
        step_size = len(shlex.join(step)) + 1
        if size + step_size > _MAX_CHAIN_BYTES:
            chains.append(list(prefix))
            size = len(shlex.join(prefix))
        chains[-1] += step
        size += step_size
    return chains


_ENV = {'DISPLAY': DISPLAY}

_START_TIMEOUT = 600
"""Seconds a start script may take: it may install `_PACKAGES` (46 seconds on Sprites, live), after waiting behind
`START_LOCK` for another start that installs them. Without an install, a start takes at most 70 seconds."""


_X_LOCK = f'/tmp/.X{DISPLAY_NUMBER}-lock'
_X_SOCKET_DIR = '/tmp/.X11-unix'
_X_SOCKET = f'{_X_SOCKET_DIR}/X{DISPLAY_NUMBER}'

_CLIENTS = 'xdotool xwininfo scrot setsid'
"""The programs the display's commands run: every one but Xvfb, which only a start without a running X server needs."""
_PACKAGES = 'xdotool x11-utils scrot util-linux'
"""The Debian and Ubuntu packages that have `_CLIENTS`. Xvfb's package, `xvfb`, is added when no X server runs."""

_DESKTOP_PROGRAMS = 'fluxbox tint2 xterm'
"""The desktop's window manager, panel and terminal, each in the Debian and Ubuntu package of the same name. `xprop`,
which tells when the window manager is up, is in `x11-utils`."""

_DESKTOP_LOG = '/tmp/pydantic-ai-desktop.log'
"""Where the desktop's install and its programs write, so a desktop that didn't come up says why."""

_PANEL_SHOWS = 'xdotool search --onlyvisible --class tint2 >/dev/null 2>&1'


_FLUXBOX_DIR = '$HOME/.fluxbox'
"""Where fluxbox reads its settings."""

_TINT2_DIR = '${XDG_CONFIG_HOME:-$HOME/.config}/tint2'
"""Where tint2 reads its settings: under `XDG_CONFIG_HOME` when it's set."""


def _write_unless_present(path: str, *lines: str) -> str:
    """A shell fragment that writes `lines` to `path`, unless the file is there already."""
    return f'[ -e "{path}" ] || printf \'%s\\n\' {shlex.join(lines)} >"{path}"; '


_DESKTOP_CONFIG = (
    f'mkdir -p "{_FLUXBOX_DIR}" "{_TINT2_DIR}"; '
    + _write_unless_present(f'{_FLUXBOX_DIR}/init', 'session.screen0.toolbar.visible: false')
    # Every new window fills the screen above the panel, so nothing opens small in a corner.
    + _write_unless_present(f'{_FLUXBOX_DIR}/apps', '[app] (class!=Tint2)', '  [Maximized] {yes}', '[end]')
    # fluxbox asks `fbsetbg` to restore this display's last wallpaper on start. Without one recorded, and without a
    # program that sets a picture as the wallpaper, `fbsetbg` opens a window saying so. This records a plain color, which
    # fluxbox's own `fbsetroot` paints. fluxbox runs it with `DISPLAY` set to `:1.0`.
    + _write_unless_present(f'{_FLUXBOX_DIR}/lastwallpaper', f'fbsetroot|-solid #3b4252||{DISPLAY}.0')
    # A panel along the bottom with a button for each launcher whose program is installed, and a button per window.
    + _write_unless_present(
        f'{_TINT2_DIR}/tint2rc',
        'rounded = 0',
        'border_width = 0',
        'background_color = #2b2b2b 100',
        'border_color = #2b2b2b 0',
        'panel_items = LT',
        'panel_position = bottom left horizontal',
        'panel_size = 100% 40',
        'panel_padding = 4 4 6',
        'panel_background_id = 1',
        'panel_layer = top',
        'strut_policy = follow_size',
        'launcher_icon_size = 32',
        'launcher_padding = 2 4 8',
        'launcher_item_app = /usr/share/applications/firefox-esr.desktop',
        'launcher_item_app = /usr/share/applications/debian-xterm.desktop',
        'taskbar_padding = 2 2 4',
        'task_text = 1',
        'task_icon = 1',
        'task_font_color = #ffffff 100',
        'task_maximum_size = 220 34',
    )
)
"""Writes the desktop's settings where the user the commands run as has fluxbox and tint2 read them, leaving any file
already there alone. A home that can't be written leaves their own defaults."""

_START_WINDOW_MANAGER = (
    f'if command -v fluxbox >/dev/null; then {{ {_DESKTOP_CONFIG}}} 2>/dev/null; '
    # A second fluxbox exits when one runs, so starting it every time recovers one that died. The wait lets the first
    # click focus and raise the window it lands on.
    'setsid fluxbox >>"$desktop_log" 2>&1 9>&- & wm=$!; i=0; '
    'until ! command -v xprop >/dev/null || xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | grep -q "window id" '
    '|| ! kill -0 $wm 2>/dev/null || [ $i -ge 100 ]; do sleep 0.1; i=$((i+1)); done; fi; '
    # tint2 shows only once a window manager answers, so it starts after one.
    f'if command -v tint2 >/dev/null && ! {_PANEL_SHOWS}; then setsid tint2 >>"$desktop_log" 2>&1 9>&- & '
    f'panel=$!; i=0; until {_PANEL_SHOWS} || ! kill -0 $panel 2>/dev/null || [ $i -ge 100 ]; do '
    'sleep 0.1; i=$((i+1)); done; fi; '
)
"""Starts the window manager, unless it's missing, and waits up to 10 seconds for it to answer `xprop`. Then starts
the panel unless it shows or is missing, and waits up to 10 seconds for its window, so the first screenshot isn't
black. Either one exiting, or not answering in time, leaves the display working without it. Their output goes to
`$desktop_log`, which `start_x_server` sets: `_DESKTOP_LOG`, or `/dev/null` when that can't be opened."""


def start_x_server(log: str, *, desktop: bool = True) -> str:
    """A shell fragment that starts Xvfb on `DISPLAY` unless an X server runs, and with `desktop` a desktop on it.

    The screen's size is the one `$1` names, such as `1280x800`. The desktop is a window manager, and a panel with a
    button that opens a terminal.

    Whether an X server runs is read from the display's X lock, not from an X server answering. Pass `desktop=False`
    when the environment starts its own desktop, such as E2B's XFCE.

    It first installs any of the programs it needs that's missing with `apt-get`, as root or through `sudo -n`, and
    fails naming them when it can't: `_CLIENTS`, and Xvfb unless an X server runs. With `desktop`, it then installs the
    desktop's programs, `_DESKTOP_PROGRAMS`, in a separate `apt-get install`, so one the package lists lack never stops
    the display; when they can't be installed, or don't start, the display works without them. It fails once the X
    server exits, or after 10 seconds, printing the end of `log`.

    It takes `START_LOCK`, so a second start waits and finds the first one's install and X server. The lock is held
    until the script exits, so the rest of the script runs under it too; anything started in the background must close
    it with `9>&-`.
    """
    return (
        # Decided from the X lock, not from an X server answering: in a shared network namespace, such as Docker's
        # `--network host`, another X server on the same display answers through Linux's shared abstract socket.
        # A stale lock, such as one left by `docker stop`, names a PID that is gone or belongs to another process.
        # A running X server's lock names it, whichever server it is, such as `Xvnc`.
        f'{START_LOCK}x_runs() {{ case "$(cat /proc/$(tr -d " " 2>/dev/null <{_X_LOCK})/comm 2>/dev/null)" in '
        'X*) return 0;; esac; return 1; }; '
        # `xvfb` is `Xvfb` when the script will start one, and empty when an X server runs; `xvfb_package` likewise.
        'if x_runs; then xvfb=; xvfb_package=; else xvfb=Xvfb; xvfb_package=xvfb; fi; '
        f'missing=; for p in $xvfb {_CLIENTS}; do command -v $p >/dev/null 2>&1 || missing="$missing $p"; done; '
        # `wanted` names the desktop's missing programs, which are installed when they can be and skipped otherwise.
        + (
            f'wanted=; for p in {_DESKTOP_PROGRAMS}; do command -v $p >/dev/null 2>&1 || wanted="$wanted $p"; done; '
            # The desktop writes to `_DESKTOP_LOG` when it's missing or a file that can be written, and to nowhere
            # otherwise, so the log never stops it. Anything else there isn't opened: a named pipe would wait for a
            # reader. The write is tried in a subshell, since a failed redirection would end the script itself in dash.
            f'desktop_log={_DESKTOP_LOG}; {{ [ ! -e "$desktop_log" ] || [ -f "$desktop_log" ]; }} && '
            '( : >>"$desktop_log" ) 2>/dev/null || desktop_log=/dev/null; '
            if desktop
            else 'wanted=; '
        )
        # `elevate` prefixes a command that needs root: empty as root, `sudo -n` with passwordless sudo, and `none`
        # when neither works.
        + 'if [ "$(id -u)" = 0 ]; then elevate=; elif sudo -n true >/dev/null 2>&1; then elevate="sudo -n"; '
        'else elevate=none; fi; '
        'if [ -n "$missing$wanted" ]; then '
        'if ! command -v apt-get >/dev/null 2>&1; then '
        """[ -z "$missing" ] || { echo "the display needs$missing, and there's no apt-get to install them with" >&2; """
        'exit 1; }; '
        'elif [ "$elevate" = none ]; then '
        '[ -z "$missing" ] || '
        """{ echo "the display needs$missing, and there's no root or passwordless sudo to install them with" >&2; """
        'exit 1; }; '
        'else install="$elevate env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends"; '
        # The display's own programs first, failing the start when they can't be installed.
        '{ $elevate apt-get update -qq && '
        f'{{ [ -z "$missing" ] || $install $xvfb_package {_PACKAGES}; }}; }} >{log} 2>&1 || [ -z "$missing" ] || '
        f'{{ echo "could not install$missing:" >&2; tail -n 5 {log} >&2; exit 1; }}; '
        # Then the desktop's, alone, so a package the lists lack only leaves the desktop out. The next start tries again.
        f'[ -z "$wanted" ] || $install $wanted >"$desktop_log" 2>&1 || true; '
        'fi; fi; '
        # Asked again, since the image's own X server may have started during the install: its lock and socket stay.
        'if ! x_runs; then '
        # No X server here holds the lock, and ours doesn't listen on the abstract socket, so one listening there is
        # another namespace's, such as the host's under `--network host`: `xdotool` would reach it before ours.
        f'if grep -q " @{_X_SOCKET}$" /proc/net/unix 2>/dev/null; then echo "another X server, outside this sandbox, '
        f'already uses display {DISPLAY} through a network namespace the sandbox shares, such as the host network" >&2; '
        'exit 1; fi; '
        f'rm -f {_X_LOCK} {_X_SOCKET}; '
        # Xvfb makes the socket directory only as root; without root, one the user makes works, so the mkdir then runs
        # unprefixed. Without `-p`, so a path that appears after the check fails the mkdir instead of being reused.
        f'if [ ! -d {_X_SOCKET_DIR} ]; then [ "$elevate" = none ] && elevate=; '
        f'$elevate mkdir -m 1777 {_X_SOCKET_DIR}; fi; '
        # Without `-noreset`, X resets when its last client exits, such as an `xdotool` command with no window manager
        # running: the pointer goes back to the middle and held buttons are let go.
        # `-nolisten local` closes Linux's abstract socket, which a shared network namespace such as Docker's
        # `--network host` shares with every host user; the socket file in /tmp/.X11-unix stays for local commands.
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp -nolisten local -noreset >/dev/null 2>{log} 9>&- & '
        + wait_until_ready('xdotool getdisplaygeometry >/dev/null 2>&1', '$!', 'the X server', log)
        + 'fi; '
        + (_START_WINDOW_MANAGER if desktop else '')
    )


class RunCommand(Protocol):
    """Runs a command in the environment, without starting its display first, as the display's own commands must."""

    def __call__(self, argv: list[str], /, *, env: Mapping[str, str], timeout: float) -> Awaitable[CommandResult]: ...


def x11_display(
    run: RunCommand,
    size: tuple[int, int],
    script: str,
    where: Callable[[], str],
    *,
    started: Callable[[], None] = lambda: None,
) -> LazyDisplay:
    """A backend's display on an X server that `script` starts at the size `$1` names, unless one runs.

    `where` names the environment in errors. `started` is called after each start that succeeds, such as to forget
    that a program the start installs was missing.
    """

    async def start() -> None:
        width, height = size
        # On `DISPLAY` whatever the backend's own env says, so it checks the X server the actions go to.
        result = await run(['sh', '-c', script, 'sh', f'{width}x{height}'], env=_ENV, timeout=_START_TIMEOUT)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'Could not start the display in {where()}: {(result.stderr or result.stdout).strip()}'
            )
        started()

    return LazyDisplay(X11Computer(run), start)


_START_DESKTOP = start_x_server('/tmp/pydantic-ai-xvfb.log')


def desktop_display(
    run: RunCommand,
    size: tuple[int, int] | None,
    where: Callable[[], str],
    *,
    started: Callable[[], None] = lambda: None,
) -> LazyDisplay | None:
    """A sandbox's desktop on Xvfb at `size`, or `None` without one.

    For a sandbox with no desktop of its own: `start_x_server` installs what's missing on first use. `started` is
    called after each start that succeeds, as for `x11_display`.
    """
    if size is None:
        return None
    return x11_display(run, size, _START_DESKTOP, where, started=started)


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
        self, *chains: list[str], hold: tuple[str, ...] = (), release: tuple[str, ...] = (), pauses: float = 0
    ) -> None:
        """Run `xdotool` command chains in turn with `hold` held down; if one fails or is cancelled, let go of everything.

        `release` is an `xdotool` chain that lets go of what the chains press; each command may take `pauses` on top of
        the usual time limit.
        """
        keysyms = [_keysym(key) for key in hold]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        keydown = ['keydown', *keysyms] if keysyms else []
        try:
            for index, chain in enumerate(chains):
                # Held keys stay down from the first command to the last.
                first = keydown if index == 0 else []
                last = keyup if index == len(chains) - 1 else []
                await self._run(['sh', '-c', _INPUT, 'sh', 'xdotool', *first, *chain, *last], pauses=pauses)
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
                moves = [['sleep', str(DRAG_STEP_PAUSE), *_move(point.x, point.y)] for point in steps]
                pause = ['sleep', str(DRAG_PRESS_PAUSE)]
                await self._xdotool(
                    *_chains([[*_move(start.x, start.y), 'mousedown', '1', *pause], *moves, [*pause, 'mouseup', '1']]),
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
                    *_chains([[keysym] for keysym in keysyms], prefix=('key', '--delay', _TYPING_DELAY_MS, '--')),
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
