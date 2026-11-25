"""`X11DisplayDriver` controls the screen of a Linux environment by running commands in it.

The environment runs an X server, the program that draws the screen.
"""

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
    NamedKey,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import CommandResult, WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use._display import RELEASE_TIMEOUT, LazyDisplay, LiveViewDisplay, png_from_base64
from pydantic_ai_harness.computer_use._drag import DRAG_STEP_PAUSE, drag_steps
from pydantic_ai_harness.computer_use._protocol import DisplayAction, UnsupportedActionError

DISPLAY = ':1'
"""The X display the desktop runs on, as `DISPLAY` names it."""

DISPLAY_NUMBER = DISPLAY[1:]

TAKE_START_LOCK = f'exec 9>/tmp/.pydantic-ai-display-{DISPLAY_NUMBER}.lock; ! command -v flock >/dev/null || flock 9; '
"""The first line of a start script. It waits for any other start of this display to finish, so two scripts don't
both start a display. A program the script starts in the background must close the lock with `9>&-`. Otherwise the
program holds the lock for as long as it runs."""


def wait_until_ready_script(ready_check: str, pid: str, program_name: str, log_path: str) -> str:
    """Shell code that waits until the command `ready_check` succeeds for a program just started in the background.

    It fails when process `pid` exits, or after 10 seconds. It then says `program_name` didn't start and prints the end
    of `log_path`. `ready_check` must succeed only for the new process. If it reads a log, empty that log before
    starting the program. Otherwise the check can read the previous run's log before the program's output replaces it.
    """
    return (
        f'i=0; until {ready_check}; do '
        f'if ! kill -0 {pid} 2>/dev/null || [ $i -ge 100 ]; then '
        f"echo '{program_name} did not start:' >&2; tail -n 5 {log_path} >&2; exit 1; fi; "
        'sleep 0.1; i=$((i+1)); done; '
    )


_MAX_REPEAT = 2**31 - 1
"""The most times xdotool repeats a click or key, since it reads `--repeat` into a C `int`."""


def _repeat(count: int) -> str:
    """`count` as xdotool's `--repeat`, refused when xdotool can't take it."""
    if count > _MAX_REPEAT:
        raise UnsupportedActionError(f'The display can repeat a click, scroll or key at most {_MAX_REPEAT} times.')
    return str(count)


_BUTTONS: dict[str, str] = {'left': '1', 'middle': '2', 'right': '3'}

_WHEEL_BUTTONS = {'up': '4', 'down': '5', 'left': '6', 'right': '7'}
"""X reports the scroll wheel as buttons."""

_NAMED_KEYSYMS: dict[NamedKey, str] = {
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
    'f1': 'F1',
    'f2': 'F2',
    'f3': 'F3',
    'f4': 'F4',
    'f5': 'F5',
    'f6': 'F6',
    'f7': 'F7',
    'f8': 'F8',
    'f9': 'F9',
    'f10': 'F10',
    'f11': 'F11',
    'f12': 'F12',
    'printscreen': 'Print',
    'pause': 'Pause',
    'menu': 'Menu',
    'capslock': 'Caps_Lock',
    'numlock': 'Num_Lock',
    'scrolllock': 'Scroll_Lock',
}

_RENAMED_KEYS: dict[str, str] = {
    **_NAMED_KEYSYMS,
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
"""Key names that X spells differently. X calls a key's name its keysym. The other names, such as `ctrl`, and ASCII
letters and digits are already keysyms."""

_COMMAND_TIMEOUT = 150
"""Seconds one display command may take, plus the pauses it asks for, such as the seconds of a `hold_keys`."""

_KEY_SECONDS = 0.012
"""The pause after each typed character or key press: `xdotool key`'s default of 12 milliseconds."""

_CLICK_SECONDS = 0.1
"""The pause between the clicks of a repeated click, `xdotool click`'s default delay."""

_SCROLL_TICK_DELAY_MS = '20'
"""Milliseconds between scroll wheel clicks; `xdotool`'s default is 100. With none, Chromium lost some of the clicks."""

_SCROLL_TICK_SECONDS = int(_SCROLL_TICK_DELAY_MS) / 1000
"""The pause one wheel click adds to the timeout."""

_PRINT_SCREEN_SIZE = ['xwininfo', '-root']
"""Prints the size of the root window, X's name for the whole screen. It spans every monitor, as a screenshot does.
`xdotool getdisplaygeometry` gives only the first monitor's size, so a click on a second monitor would be scaled onto
the first."""

_SCREENSHOT_AS_BASE64 = (
    'f=/tmp/pydantic-ai-capture-$$.png; scrot --overwrite "$@" "$f" && base64 < "$f"; status=$?; rm -f "$f"; '
    'exit $status'
)
"""Takes a screenshot into a file of its own and prints it as base64. scrot picks the PNG format from the `.png`
name. Arguments go to scrot, such as `-a x,y,width,height` for a region. A cancelled capture can leave its file
behind, and a later capture can get the same process id and so the same file name. `--overwrite` replaces that file.
Without it, scrot saves under another name and the old file would be printed."""

_INPUT_PID_FILE = f'/tmp/.pydantic-ai-input-{DISPLAY_NUMBER}'
"""Holds the process id of the input command running on the display; `ComputerUse` runs one at a time."""

_RUN_INPUT_RECORDING_PID = f'echo $$ > {_INPUT_PID_FILE}; exec "$@"'
"""Runs the input command in `$@` as the process whose id `_INPUT_PID_FILE` records."""

_STOP_INPUT = (
    f'p=$(cat {_INPUT_PID_FILE} 2>/dev/null) || exit 0; [ "$(cat /proc/$p/comm 2>/dev/null)" = xdotool ] || exit 0; '
    'kill "$p" 2>/dev/null; '
    'while [ -e /proc/$p ] && ! grep -q "^State:[[:space:]]*Z" /proc/$p/status 2>/dev/null; do sleep 0.05; done'
)
"""Stops the input command `_INPUT_PID_FILE` names, if it still runs, and waits for it to exit. A cancelled command can
outlive its call and press keys again after they were released."""


def _keysym(key: str) -> str:
    if key in _RENAMED_KEYS:
        return _RENAMED_KEYS[key]
    if len(key) == 1 and not (key.isascii() and key.isalnum()):
        # A character such as `!`: xdotool presses shift when the layout needs it.
        return _keysym_for_character(key)
    return key


def _keysym_for_character(character: str) -> str:
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


def _mousemove_args(x: int | None, y: int | None) -> list[str]:
    return [] if x is None or y is None else ['mousemove', str(x), str(y)]


_MAX_CHAIN_BYTES = 100_000
"""The most bytes of `xdotool` arguments sent in one command. Longer input is split across several commands.

E2B runs a command as a single `bash -l -c` argument, and Linux limits one argument to 131,072 bytes. In a live
probe, a 130,630-byte line ran and a 130,664-byte line failed with "argument list too long". Typing 25,000
characters in one command also failed. The gap between 100,000 and the limit leaves room for the text the backend
wraps around each command."""


def _split_into_commands(steps: list[list[str]], prefix: tuple[str, ...] = ()) -> list[list[str]]:
    """Group `steps`, in order, into `xdotool` commands that each start with `prefix` and fit in `_MAX_CHAIN_BYTES`."""
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


_DISPLAY_ENV = {'DISPLAY': DISPLAY}

_START_TIMEOUT = 600
"""Seconds a start may take. Live, a start that installs the programs took 182 and 208 seconds in a new
`debian:bookworm-slim` container, and a start can wait behind `TAKE_START_LOCK` for another one doing that install.
Without an install, a start takes at most 70 seconds."""


_X_LOCK_FILE = f'/tmp/.X{DISPLAY_NUMBER}-lock'
_X_SOCKET_DIR = '/tmp/.X11-unix'
_X_SOCKET = f'{_X_SOCKET_DIR}/X{DISPLAY_NUMBER}'

_DISPLAY_PROGRAMS = 'xdotool xwininfo scrot setsid'
"""The programs the display's commands run, besides Xvfb."""
_DISPLAY_PACKAGES = 'xdotool x11-utils scrot util-linux'
"""The Debian packages that have `_DISPLAY_PROGRAMS`."""

_APT_FAIL_FAST_OPTIONS = '-o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o Acquire::Retries=1'
"""Options that make an `apt-get` that can't reach its mirrors fail soon, so the start reports the missing programs.

A network that drops packets, such as a Render sandbox's without internet, otherwise runs out `_START_TIMEOUT`: there,
live, each DNS lookup took 80 seconds, and apt looks up its host again for each list and each of its 4 tries. With
these options and `_SHORT_DNS_TIMEOUT_ENV`, the start gave up in 50 to 195 seconds, as it did without the retries, and
a sandbox with internet still installed the packages. One retry keeps a single mirror error from failing the install.
The 20 seconds are for connecting and for each wait for data, not for a whole download, so a slow mirror that keeps
sending still works.
"""

_SHORT_DNS_TIMEOUT_ENV = 'RES_OPTIONS="timeout:2 attempts:2"'
"""Asks each DNS server at most twice, for 2 seconds each, instead of glibc's 5 seconds twice, so one dropped reply
doesn't fail the lookup. A working server answers in milliseconds."""

_DESKTOP_PROGRAMS = 'fluxbox tint2 xterm'
"""The window manager, panel and terminal, each in the Debian package of the same name. `xprop`, used to wait for the
window manager, comes with `x11-utils` in `_DISPLAY_PACKAGES`."""

DESKTOP_PACKAGES = ('xvfb', *_DISPLAY_PACKAGES.split(), *_DESKTOP_PROGRAMS.split())
"""Every Debian package the desktop needs, for a sandbox that builds them into its image."""

_DESKTOP_LOG = '/tmp/pydantic-ai-desktop.log'
"""Where the desktop's install and programs write, so a desktop that didn't come up says why."""

_FAILED_DESKTOP_INSTALL_FILE = '/tmp/pydantic-ai-desktop-install-failed'
"""The desktop programs whose install failed, so later starts skip `apt-get` while the same ones are missing."""

_PANEL_SHOWS = 'xdotool search --onlyvisible --class tint2 >/dev/null 2>&1'


_FLUXBOX_DIR = '$HOME/.fluxbox'
_TINT2_DIR = '${XDG_CONFIG_HOME:-$HOME/.config}/tint2'


def _write_unless_present_script(path: str, *lines: str) -> str:
    """A shell fragment that writes `lines` to `path` unless it exists."""
    return f'[ -e "{path}" ] || printf \'%s\\n\' {shlex.join(lines)} >"{path}"; '


_BROWSERS = (
    ('firefox-esr', 'Firefox'),
    ('firefox', 'Firefox'),
    ('chromium', 'Chromium'),
    ('chromium-browser', 'Chromium'),
    ('google-chrome', 'Chrome'),
)
"""The browsers the panel looks for, first found first, each as its command and the label of its panel button."""

_FIND_BROWSER = (
    'browser=; for b in '
    + ' '.join(f'{command}:{label}' for command, label in _BROWSERS)
    + '; do command -v "${b%%:*}" >/dev/null 2>&1 && { browser=${b%%:*}; browser_label=${b#*:}; break; }; done; '
    # Chromium and Chrome refuse to start as root without `--no-sandbox`, and print "No usable sandbox!" where an
    # unprivileged user can't make a user namespace, as on Daytona and under Docker's default seccomp profile.
    'browser_flag=; case $browser in chrom* | google-chrome) '
    '{ [ "$(id -u)" = 0 ] || ! unshare -Ur true; } >/dev/null 2>&1 && browser_flag=" --no-sandbox";; esac; '
)
"""Sets `browser` to the first of `_BROWSERS` the image has, or to nothing, with its label and any flag it needs."""

TINT2_BUTTON_STYLE = (
    'button_font = Sans 12',
    'button_font_color = #ffffff 100',
    'button_padding = 8 4 6',
    'button_background_id = 2',
)
"""How the panel's labelled buttons look, shared with boxd's panel."""


_WRITE_DESKTOP_SETTINGS = (
    f'mkdir -p "{_FLUXBOX_DIR}" "{_TINT2_DIR}"; '
    + _write_unless_present_script(f'{_FLUXBOX_DIR}/init', 'session.screen0.toolbar.visible: false')
    # Maximize every window but the panel.
    + _write_unless_present_script(f'{_FLUXBOX_DIR}/apps', '[app] (class!=Tint2)', '  [Maximized] {yes}', '[end]')
    # On start, fluxbox has `fbsetbg` restore the last wallpaper. With none recorded and no wallpaper setter installed,
    # `fbsetbg` opens a window saying so. This records a plain color, which fluxbox's own `fbsetroot` paints, under
    # `:1.0`, the `DISPLAY` fluxbox runs it with.
    + _write_unless_present_script(f'{_FLUXBOX_DIR}/lastwallpaper', f'fbsetroot|-solid #3b4252||{DISPLAY}.0')
    # Models misread xterm's default bitmap font, such as `g` as `9`. xterm reads this file however it is opened,
    # unless `xrdb` loaded settings.
    + _write_unless_present_script('$HOME/.Xdefaults', 'XTerm*faceName: Monospace', 'XTerm*faceSize: 14')
    # Models found a button labelled Terminal in fewer steps than an icon alone, so the browser gets one too. Terminal
    # comes first so its place doesn't depend on whether the image has a browser.
    + _FIND_BROWSER
    + f'[ -e "{_TINT2_DIR}/tint2rc" ] || {{ printf \'%s\\n\' '
    + shlex.join(
        (
            # Backgrounds 1 (the panel) and 2 (the buttons), numbered in the order they're written.
            'rounded = 0',
            'border_width = 0',
            'background_color = #2b2b2b 100',
            'border_color = #2b2b2b 0',
            'rounded = 0',
            'border_width = 1',
            'background_color = #444444 100',
            'border_color = #888888 100',
            'panel_position = bottom left horizontal',
            'panel_size = 100% 40',
            'panel_padding = 4 4 6',
            'panel_background_id = 1',
            'panel_layer = top',
            'strut_policy = follow_size',
            'button = new',
            'button_icon = mini.xterm',
            'button_text = Terminal',
            *TINT2_BUTTON_STYLE,
            'button_lclick_command = xterm',
            'taskbar_padding = 2 2 4',
            'task_text = 1',
            'task_icon = 1',
            'task_font_color = #ffffff 100',
            'task_maximum_size = 220 34',
        )
    )
    # Each `P` in `panel_items` is the next button.
    + ' "panel_items = P${browser:+P}T"; [ -z "$browser" ] || printf \'%s\\n\' "button = new" '
    + f'"button_icon = $browser" "button_text = $browser_label" {shlex.join(TINT2_BUTTON_STYLE)} '
    + f'"button_lclick_command = $browser$browser_flag"; }} >"{_TINT2_DIR}/tint2rc"; '
)
"""Writes the fluxbox, tint2 and xterm settings, keeping any file already there."""

_START_DESKTOP_PROGRAMS = (
    f'if command -v fluxbox >/dev/null; then {{ {_WRITE_DESKTOP_SETTINGS}}} 2>/dev/null; '
    # A second fluxbox exits when one runs, so starting it every time recovers one that died. Waiting for it lets the
    # first click focus and raise the window it lands on.
    'setsid fluxbox >&8 2>&1 8>&- 9>&- & wm=$!; i=0; '
    'until ! command -v xprop >/dev/null || xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | grep -q "window id" '
    '|| ! kill -0 $wm 2>/dev/null || [ $i -ge 100 ]; do sleep 0.1; i=$((i+1)); done; fi; '
    # tint2 shows only once a window manager answers. Before running a button's program, tint2 closes every file
    # handle up to its soft open-file limit: under a Kubernetes pod's 1,073,741,816 its Terminal button opened nothing,
    # and at 1,048,576 it opened xterm in under a second. Only tint2's soft limit is capped; the hard limit stays.
    f'if command -v tint2 >/dev/null && ! {_PANEL_SHOWS}; then '
    '( n=$(ulimit -S -n); { [ "$n" = unlimited ] || [ "$n" -gt 1048576 ]; } 2>/dev/null && '
    'ulimit -S -n 1048576 2>/dev/null; exec setsid tint2 ) >&8 2>&1 8>&- 9>&- & '
    f'panel=$!; i=0; until {_PANEL_SHOWS} || ! kill -0 $panel 2>/dev/null || [ $i -ge 100 ]; do '
    'sleep 0.1; i=$((i+1)); done; fi; '
)
"""Starts fluxbox and waits up to 10 seconds for it to answer `xprop`, then tint2 unless it shows, waiting up to 10
seconds for its window so the first screenshot isn't black. Either one missing, exiting or not answering leaves the
display working without it. Their output goes to the desktop log."""


def start_display_script(xvfb_log: str, *, with_desktop: bool = True) -> str:
    """A shell fragment that starts Xvfb on `DISPLAY` unless an X server runs, and with `with_desktop` a desktop on it.

    `$1` is the screen size, such as `1280x800`. The desktop is a window manager and a panel with a Terminal button and,
    when the image has one, a browser button.
    Whether an X server runs is read from the display's X lock. Pass `with_desktop=False` when the environment starts its
    own desktop, such as E2B's XFCE.

    Missing programs are installed with `apt-get`, as root or through `sudo -n`. The fragment fails naming those of
    `_DISPLAY_PROGRAMS` and Xvfb it can't install; without the desktop's programs, the display works without the
    desktop. It fails once the X server exits, or after 10 seconds, printing the end of `xvfb_log`.

    It takes `TAKE_START_LOCK` and holds it until the script exits, so anything the script starts in the background must
    close it with `9>&-`.
    """
    return (
        # Read the X lock instead of asking for an X server. With a shared network, such as Docker's `--network host`,
        # another X server on the same display number answers through Linux's shared abstract socket.
        # A stale lock, such as one `docker stop` leaves, holds a process ID that is gone or now belongs to another
        # program. A running X server's lock holds that server's process ID, whichever server it is, such as `Xvnc`.
        f'{TAKE_START_LOCK}x_server_runs() {{ case "$(cat /proc/$(tr -d " " 2>/dev/null <{_X_LOCK_FILE})/comm 2>/dev/null)" in '
        'X*) return 0;; esac; return 1; }; '
        # Programs in `missing_required` fail the start when they can't be installed; the desktop's, in `missing_desktop`, don't.
        'if x_server_runs; then xvfb=; xvfb_package=; else xvfb=Xvfb; xvfb_package=xvfb; fi; '
        f'missing_required=; for p in $xvfb {_DISPLAY_PROGRAMS}; do command -v $p >/dev/null 2>&1 || missing_required="$missing_required $p"; done; '
        + (
            f'missing_desktop=; for p in {_DESKTOP_PROGRAMS}; do command -v $p >/dev/null 2>&1 || missing_desktop="$missing_desktop $p"; done; '
            # Skip an install that already failed for the same programs.
            f'[ "$missing_desktop" != "$(cat {_FAILED_DESKTOP_INSTALL_FILE} 2>/dev/null)" ] || missing_desktop=; '
            # The desktop writes to file descriptor 8 (9 is the lock): `_DESKTOP_LOG` when it's missing or a writable
            # file, `/dev/null` otherwise. A named pipe would block the open, and a symlink would let a start as root
            # write through it. Opening once, right after the check, means a symlink made later isn't followed. The
            # write is tried in a subshell first because a failed redirection on `exec` ends a dash script; with
            # `command exec` instead, macOS's bash 3.2 hangs at the next `exit`.
            f'exec 8>/dev/null; [ ! -L {_DESKTOP_LOG} ] && {{ [ ! -e {_DESKTOP_LOG} ] || [ -f {_DESKTOP_LOG} ]; }} && '
            f'( : >>{_DESKTOP_LOG} ) 2>/dev/null && exec 8>>{_DESKTOP_LOG}; '
            if with_desktop
            else 'missing_desktop=; '
        )
        + 'if [ "$(id -u)" = 0 ]; then sudo_prefix=; elif sudo -n true >/dev/null 2>&1; then sudo_prefix="sudo -n"; '
        'else sudo_prefix=none; fi; '
        'if [ -n "$missing_required$missing_desktop" ]; then '
        'if ! command -v apt-get >/dev/null 2>&1; then '
        """[ -z "$missing_required" ] || { echo "the display needs$missing_required, and there's no apt-get to install them with" >&2; """
        'exit 1; }; '
        'elif [ "$sudo_prefix" = none ]; then '
        '[ -z "$missing_required" ] || '
        """{ echo "the display needs$missing_required, and there's no root or passwordless sudo to install them with" >&2; """
        'exit 1; }; '
        f'else apt_get() {{ $sudo_prefix env DEBIAN_FRONTEND=noninteractive {_SHORT_DNS_TIMEOUT_ENV} apt-get "$@"; }}; '
        f'install="apt_get install -y -qq --no-install-recommends {_APT_FAIL_FAST_OPTIONS}"; '
        # With only the desktop's programs missing, the update logs to fd 8, since starting Xvfb below empties `xvfb_log`.
        # A failed update clears `updated`, so the install failure after it isn't recorded.
        f'updated=1; if [ -z "$missing_required" ]; then apt_get update -qq {_APT_FAIL_FAST_OPTIONS} >&8 2>&1 || updated=; '
        f'elif ! {{ apt_get update -qq {_APT_FAIL_FAST_OPTIONS} && $install $xvfb_package {_DISPLAY_PACKAGES}; }} >{xvfb_log} 2>&1; then '
        f'echo "could not install$missing_required:" >&2; tail -n 5 {xvfb_log} >&2; exit 1; fi; '
        # The desktop's own install, so a package the lists lack only leaves the desktop out. Its failure is recorded
        # only after a working update, and never through a symlink.
        f'if [ -n "$missing_desktop" ]; then if $install $missing_desktop >&8 2>&1; then rm -f {_FAILED_DESKTOP_INSTALL_FILE} 2>/dev/null; '
        f'elif [ -n "$updated" ] && [ ! -L {_FAILED_DESKTOP_INSTALL_FILE} ]; then '
        f'printf %s "$missing_desktop" 2>/dev/null >{_FAILED_DESKTOP_INSTALL_FILE} || true; fi; fi; '
        'fi; fi; '
        # Check again: the image's own X server may have started during the install, and its lock and socket must stay.
        'if ! x_server_runs; then '
        # Ours doesn't listen on the abstract socket, so an X server there is another namespace's, such as the host's
        # under `--network host`, and `xdotool` would reach it before ours.
        f'if grep -q " @{_X_SOCKET}$" /proc/net/unix 2>/dev/null; then echo "another X server, outside this sandbox, '
        f'already uses display {DISPLAY} through a network namespace the sandbox shares, such as the host network" >&2; '
        'exit 1; fi; '
        f'rm -f {_X_LOCK_FILE} {_X_SOCKET}; '
        # Xvfb makes the socket directory only as root; without root, one the user makes works. No `-p`, so a path
        # that appears after the check fails the mkdir instead of being reused.
        f'if [ ! -d {_X_SOCKET_DIR} ]; then [ "$sudo_prefix" = none ] && sudo_prefix=; '
        f'$sudo_prefix mkdir -m 1777 {_X_SOCKET_DIR}; fi; '
        # Without `-noreset`, X resets whenever its last program disconnects, such as an `xdotool` command when no
        # window manager runs. A reset moves the pointer back to the middle and lets go of held buttons.
        # `-nolisten local` turns off Linux's abstract socket. With a shared network, such as Docker's `--network host`,
        # every user on the host could reach that socket. Commands in the container or sandbox still use the socket
        # file in /tmp/.X11-unix.
        f'setsid Xvfb {DISPLAY} -screen 0 "$1"x24 -nolisten tcp -nolisten local -noreset >/dev/null 2>{xvfb_log} 8>&- 9>&- & '
        + wait_until_ready_script('xdotool getdisplaygeometry >/dev/null 2>&1', '$!', 'the X server', xvfb_log)
        + 'fi; '
        + (_START_DESKTOP_PROGRAMS if with_desktop else '')
    )


class RunCommand(Protocol):
    """Runs a command in the environment, without starting its display first, as the display's own commands must."""

    def __call__(self, argv: list[str], /, *, env: Mapping[str, str], timeout: float) -> Awaitable[CommandResult]: ...


_START_DESKTOP = start_display_script('/tmp/pydantic-ai-xvfb.log')


def x11_display(
    run_command: RunCommand,
    size: tuple[int, int] | None,
    describe_environment: Callable[[], str],
    *,
    start_script: str = _START_DESKTOP,
    live_view: Callable[[], Awaitable[str]] | None = None,
) -> LazyDisplay | None:
    """A backend's display on `DISPLAY` at `size`, or `None` without a size.

    `start_script` starts an X server at the size given as `$1`, unless one already runs; the default also starts a
    desktop. `describe_environment` names the environment in error messages, such as `'E2B sandbox abc'`. The backend
    adds `DISPLAY` to its commands' env, so a program the agent starts from the shell opens on the screen. With
    `live_view`, which makes the URL, the display also implements `SupportsLiveView`.
    """
    if size is None:
        return None

    async def start() -> None:
        width, height = size
        # Always on `DISPLAY`, whatever the backend's own env sets, so it checks the X server the actions go to.
        argv = ['sh', '-c', start_script, 'sh', f'{width}x{height}']
        result = await run_command(argv, env=_DISPLAY_ENV, timeout=_START_TIMEOUT)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(
                f'Could not start the display in {describe_environment()}: {(result.stderr or result.stdout).strip()}'
            )

    if live_view is None:
        return LazyDisplay(X11DisplayDriver(run_command), start)
    return LiveViewDisplay(X11DisplayDriver(run_command), start, live_view=live_view)


class X11DisplayDriver:
    """Controls an X display with `xdotool` and takes screenshots with `scrot`, by running a backend's commands.

    The X server must already be running on `DISPLAY`. The environment needs `xdotool`, `xwininfo` (in Debian's
    `x11-utils`), `scrot` and `base64`.
    """

    def __init__(self, run: RunCommand):
        self._run_in_environment = run

    async def _run_display_command(self, command: list[str], *, pause_seconds: float = 0) -> CommandResult:
        """Run `command`, allowing `pause_seconds`, the seconds it asks to pause for, on top of the usual time limit."""
        timeout = _COMMAND_TIMEOUT + pause_seconds
        result = await self._run_in_environment(command, env=_DISPLAY_ENV, timeout=timeout)
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
        self,
        *chains: list[str],
        held_keys: tuple[str, ...] = (),
        release_chain: tuple[str, ...] = (),
        pause_seconds: float = 0,
    ) -> None:
        """Run `xdotool` command chains one after another, with the `held_keys` held down the whole time.

        If a chain fails or is cancelled, everything is let go. `release_chain` is an `xdotool` chain that lets go of
        what the chains press. Each command gets `pause_seconds` extra seconds on top of the usual time limit.
        """
        keysyms = [_keysym(key) for key in held_keys]
        keyup = ['keyup', *reversed(keysyms)] if keysyms else []
        keydown = ['keydown', *keysyms] if keysyms else []
        try:
            for index, chain in enumerate(chains):
                # The held keys go down in the first command and come up in the last one.
                press_held = keydown if index == 0 else []
                lift_held = keyup if index == len(chains) - 1 else []
                await self._run_display_command(
                    ['sh', '-c', _RUN_INPUT_RECORDING_PID, 'sh', 'xdotool', *press_held, *chain, *lift_held],
                    pause_seconds=pause_seconds,
                )
        except BaseException as error:
            # Each release runs on its own, so one that fails doesn't keep the other from running.
            failures = await self._stop_input_and_release([list(release_chain), keyup])
            if failures and isinstance(error, WorkspaceError) and not isinstance(error, WorkspaceUnavailableError):
                raise WorkspaceError(
                    f'{error}. Releasing what it pressed failed too, so a mouse button or key may still be held: '
                    + '; '.join(failures)
                ) from error
            raise

    async def _stop_input_and_release(self, release_chains: list[list[str]]) -> list[str]:
        """Stop the input command, then run each of `release_chains`. Return what failed or didn't run in time."""
        failures: list[str] = []
        with anyio.move_on_after(RELEASE_TIMEOUT, shield=True) as releasing:
            # Stopping comes before letting go, or the command could press again what was just released.
            for cleanup in [['sh', '-c', _STOP_INPUT], *(['xdotool', *chain] for chain in release_chains if chain)]:
                try:
                    await self._run_display_command(cleanup)
                except WorkspaceUnavailableError:
                    raise
                except Exception as e:
                    failures.append(str(e))
        if releasing.cancelled_caught:
            failures.append(f'releasing took longer than {RELEASE_TIMEOUT} seconds')
        return failures

    async def _screenshot_png(self, *scrot_args: str) -> bytes:
        """Run `_SCREENSHOT_AS_BASE64` with `scrot_args` and read the PNG it prints as base64."""
        return png_from_base64(
            (await self._run_display_command(['sh', '-c', _SCREENSHOT_AS_BASE64, 'sh', *scrot_args])).stdout,
            'The screenshot command',
        )

    async def display_size(self) -> tuple[int, int]:
        output = (await self._run_display_command(_PRINT_SCREEN_SIZE)).stdout
        # `[ \t]*` matches within one line. `\s*` would make the search time grow with the square of the output's
        # length.
        width, height = (
            re.search(rf'^[ \t]*{name}: (\d{{1,5}})$', output, re.MULTILINE) for name in ('Width', 'Height')
        )
        if width is None or height is None:
            raise WorkspaceError(f'Unexpected display size output: {output.strip()!r}')
        size = int(width[1]), int(height[1])
        # X11 sizes fit in 16 bits. A bigger size would make a drag plan that many points.
        if not all(1 <= side <= 65535 for side in size):
            raise WorkspaceError(f'The display reported an impossible size: {size[0]}x{size[1]}')
        return size

    async def screenshot(self) -> bytes:
        return await self._screenshot_png()

    async def zoom(self, region: Region) -> bytes:
        geometry = f'{region.left},{region.top},{region.right - region.left},{region.bottom - region.top}'
        return await self._screenshot_png('-a', geometry)

    async def cursor_position(self) -> Point:
        output = (await self._run_display_command(['xdotool', 'getmouselocation', '--shell'])).stdout
        x, y = (re.search(rf'^{axis}=(\d{{1,5}})$', output, re.MULTILINE) for axis in ('X', 'Y'))
        if x is None or y is None:
            raise WorkspaceError(f'Unexpected `xdotool getmouselocation` output: {output.strip()!r}')
        return Point(x=int(x[1]), y=int(y[1]))

    async def act(self, action: DisplayAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                await self._xdotool(
                    [*_mousemove_args(x, y), 'click', '--repeat', _repeat(count), _BUTTONS[button]],
                    held_keys=modifiers,
                    release_chain=('mouseup', _BUTTONS[button]),
                    pause_seconds=count * _CLICK_SECONDS,
                )
            case MouseMove(x=x, y=y):
                await self._xdotool(_mousemove_args(x, y))
            case Drag(path=path, modifiers=modifiers):
                start = path[0]
                pause = ['sleep', str(DRAG_STEP_PAUSE)]
                # Each move, and letting go, comes after a pause.
                steps = [[*pause, *_mousemove_args(point.x, point.y)] for point in drag_steps(path)]
                await self._xdotool(
                    *_split_into_commands(
                        [[*_mousemove_args(start.x, start.y), 'mousedown', '1'], *steps, [*pause, 'mouseup', '1']]
                    ),
                    held_keys=modifiers,
                    release_chain=('mouseup', '1'),
                    pause_seconds=(len(steps) + 1) * DRAG_STEP_PAUSE,
                )
            case MouseDown(button=button):
                await self._xdotool(['mousedown', _BUTTONS[button]])
            case MouseUp(button=button):
                await self._xdotool(['mouseup', _BUTTONS[button]])
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                wheel = _WHEEL_BUTTONS[direction]
                await self._xdotool(
                    [
                        *_mousemove_args(x, y),
                        'click',
                        '--repeat',
                        _repeat(amount),
                        '--delay',
                        _SCROLL_TICK_DELAY_MS,
                        wheel,
                    ],
                    held_keys=modifiers,
                    release_chain=('mouseup', wheel),
                    pause_seconds=amount * _SCROLL_TICK_SECONDS,
                )
            case TypeText(text=text):
                keysyms = [_keysym_for_character(character) for character in text]
                # If a cancel lands between a key's press and its release, X repeats the key forever.
                # `xdotool` holds Shift_L for a shifted character such as `!`.
                await self._xdotool(
                    *_split_into_commands([[keysym] for keysym in keysyms], prefix=('key', '--')),
                    release_chain=('keyup', 'Shift_L', *dict.fromkeys(keysyms)),
                    pause_seconds=len(keysyms) * _KEY_SECONDS,
                )
            case PressKeys(keys=keys, repeat=repeat):
                keysyms = [_keysym(key) for key in keys]
                # A cancel can land between pressing and releasing, so let go of the keys then too.
                # xdotool pauses after each key of the chord, not once per repeat.
                await self._xdotool(
                    ['key', '--repeat', _repeat(repeat), '--', '+'.join(keysyms)],
                    release_chain=('keyup', *reversed(keysyms)),
                    pause_seconds=repeat * len(keysyms) * _KEY_SECONDS,
                )
            case HoldKeys(keys=keys, seconds=seconds):
                await self._xdotool(['sleep', str(seconds)], held_keys=keys, pause_seconds=seconds)
            case _:
                raise UnsupportedActionError(f'The display cannot perform {action!r}.')
