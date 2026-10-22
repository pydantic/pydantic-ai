"""`X11Computer`: display control for a Linux environment with an X server, through its commands."""

from __future__ import annotations as _annotations

import base64
import binascii

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
    Screenshot,
    Scroll,
    TypeText,
    Wait,
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
}
"""Normalized key names whose X keysym is spelled differently; the rest (`ctrl`, `a`, `minus`, ...) pass through."""

_SCREENSHOT = 'd=$(mktemp -d) && scrot "$d/screen.png" && base64 "$d/screen.png"; status=$?; rm -rf "$d"; exit $status'
"""Captures into a fresh directory (scrot picks the format from the `.png` name) and prints it as base64."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _keysym(key: str) -> str:
    return _KEYSYMS.get(key, key)


def _hold(keys: tuple[str, ...], args: list[str]) -> list[str]:
    """Wrap `xdotool` subcommands in `keydown`/`keyup` for the held keys, released in reverse order."""
    if not keys:
        return args
    syms = [_keysym(key) for key in keys]
    return ['keydown', *syms, *args, 'keyup', *reversed(syms)]


def _move(x: int | None, y: int | None) -> list[str]:
    if x is None or y is None:
        return []
    return ['mousemove', str(x), str(y)]


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

    async def _xdotool(self, args: list[str]) -> CommandResult:
        return await self._run(['xdotool', *args])

    async def display_size(self) -> tuple[int, int]:
        output = (await self._xdotool(['getdisplaygeometry'])).stdout.split()
        try:
            width, height = output
            return int(width), int(height)
        except ValueError:
            raise WorkspaceError(f'Unexpected `xdotool getdisplaygeometry` output: {" ".join(output)!r}') from None

    async def screenshot(self) -> bytes:
        try:
            data = base64.b64decode((await self._run(_SCREENSHOT)).stdout)
        except binascii.Error:
            data = b''
        if not data.startswith(_PNG_SIGNATURE):
            raise WorkspaceError('The screenshot command did not print a base64-encoded PNG.')
        return data

    async def act(self, action: ComputerAction) -> None:
        match action:
            case Click(x=x, y=y, button=button, count=count, keys=keys):
                await self._xdotool(_hold(keys, [*_move(x, y), 'click', '--repeat', str(count), _BUTTONS[button]]))
            case MoveMouse(x=x, y=y, keys=keys):
                await self._xdotool(_hold(keys, _move(x, y)))
            case Drag(path=path, keys=keys):
                (start_x, start_y), *rest = path
                steps = [arg for x, y in rest for arg in _move(x, y)]
                await self._xdotool(_hold(keys, [*_move(start_x, start_y), 'mousedown', '1', *steps, 'mouseup', '1']))
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
                await self._xdotool(_hold(keys, [*_move(x, y), *clicks]))
            case TypeText(text=text):
                await self._xdotool(['type', '--', text])
            case KeyPress(keys=keys, repeat=repeat):
                await self._xdotool(['key', '--repeat', str(repeat), '--', '+'.join(_keysym(key) for key in keys)])
            case HoldKey(keys=keys, seconds=seconds):
                await self._xdotool(_hold(keys, ['sleep', str(seconds)]))
            case Wait(seconds=seconds):
                await anyio.sleep(seconds)
            case Screenshot():
                pass
            case _:
                raise UnsupportedComputerActionError(f'X11Computer cannot perform {action!r}.')
