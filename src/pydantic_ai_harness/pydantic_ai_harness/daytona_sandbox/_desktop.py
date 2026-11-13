"""`DaytonaDesktop`: display control for a Daytona sandbox, through Daytona's computer-use API.

How Daytona's default snapshot behaves. Re-run the live display tests in
`tests/harness/daytona_sandbox/test_daytona_live.py` before changing the mapping:

- `computer_use.start()` returns once the display answers; starting a running desktop is harmless.
- `VNC_RESOLUTION` on a new sandbox sets the display size; without it, or below 1024x768, the size is 1024x768.
- Screenshots are PNGs at the display's size; a region screenshot is the region at full resolution.
- `mouse.scroll`'s amount is wheel clicks; `left` and `right` send X buttons 6 and 7.
- Modifiers given to `click`, `scroll` and `press` are held around the event; `cmd` is `Super_L`.
- `keyboard.down` and `keyboard.up` take letters and named keys, not only modifiers.
- `keyboard.type` types Unicode up to U+FFFF, newlines as Enter, and drops emoji and anything beyond.
- `keyboard.press` refuses shifted symbols such as `!`, X keysyms such as `KP_Enter`, and print screen, pause,
  scroll lock and media keys.
- Its numpad names are `num0` to `num9`, `num_enter`, `num_plus`, `num_minus`, `num_asterisk`, `num_slash`,
  `num_decimal` and `num_lock`; the display refuses `menu`.
- Pressing `f13` to `f24` crashes the computer-use process for good.
- After the sandbox stops and starts, `get_status()` says `inactive` until `start()`. Input sent before that start
  breaks the desktop: every request answers 503 until the sandbox stops and starts again.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from daytona import ScreenshotRegion

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
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness._display import png_from_base64
from pydantic_ai_harness._drag import DRAG_PRESS_PAUSE, DRAG_STEP_PAUSE, drag_steps
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError

if TYPE_CHECKING:
    from daytona import AsyncComputerUse

_REQUEST_TIMEOUT = 30
"""As `_backend._REQUEST_TIMEOUT`, which imports this module."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_TYPE_TIMEOUT = 120
"""Maximum time for one typing request, of at most `_TYPED_PER_REQUEST` characters."""

_TYPED_PER_REQUEST = 2_000
"""Characters typed by one request, well inside `_TYPE_TIMEOUT`: 2,000 take about 11 seconds."""

_SCROLLED_PER_REQUEST = 500
"""Wheel clicks scrolled by one request, well inside `_REQUEST_TIMEOUT`: 1,000 take about 21 seconds."""

_BUTTONS = frozenset({'left', 'middle', 'right'})

_MODIFIERS = {'ctrl': 'ctrl', 'shift': 'shift', 'alt': 'alt', 'super': 'cmd'}

_NAMED_KEYS = frozenset(
    {
        'enter',
        'escape',
        'tab',
        'backspace',
        'delete',
        'insert',
        'home',
        'end',
        'pageup',
        'pagedown',
        'up',
        'down',
        'left',
        'right',
        'space',
        'capslock',
        *(f'f{n}' for n in range(1, 13)),
    }
)
"""Normalized key names Daytona presses under the same name; `f13` to `f24` crash its desktop."""

_RENAMED_KEYS = {
    **{f'numpad{n}': f'num{n}' for n in range(10)},
    'numpad_enter': 'num_enter',
    'numpad_add': 'num_plus',
    'numpad_subtract': 'num_minus',
    'numpad_multiply': 'num_asterisk',
    'numpad_divide': 'num_slash',
    'numpad_decimal': 'num_decimal',
    'numlock': 'num_lock',
}
"""Normalized key names Daytona presses under another name."""

_UNSHIFTED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789`-=[]\\;',./")

_SHIFTED = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))
"""A symbol typed with shift, and the key it is on: Daytona's display has a US keyboard layout."""


def _button(button: str) -> str:
    if button not in _BUTTONS:
        raise UnsupportedActionError(f"Daytona's desktop has no {button!r} mouse button.")
    return button


def _keys(keys: Sequence[str]) -> list[str]:
    """Daytona's names for the keys, in order, with `shift` before a shifted symbol."""
    names: list[str] = []
    for key in keys:
        if key in _MODIFIERS:
            names.append(_MODIFIERS[key])
        elif key in _NAMED_KEYS or key in _UNSHIFTED:
            names.append(key)
        elif key in _RENAMED_KEYS:
            names.append(_RENAMED_KEYS[key])
        elif key in _SHIFTED:
            names += ['shift', _SHIFTED[key]]
        else:
            raise UnsupportedActionError(f"Daytona's desktop cannot press the key {key!r}.")
    return list(dict.fromkeys(names))


class DaytonaDesktop:
    """Controls a Daytona sandbox's desktop, which must already be started, with its computer-use API.

    `computer_use` returns the attached sandbox's API. It is read on every call, since the backend attaches again
    with a new client after closing the old one.
    """

    def __init__(self, computer_use: Callable[[], AsyncComputerUse]):
        self._attached = computer_use

    @property
    def _computer_use(self) -> AsyncComputerUse:
        return self._attached()

    async def _check_running(self) -> None:
        """Refuse an action on a desktop that stopped, before sending it any input.

        The sandbox may stop and start between calls, and input sent before its desktop starts again breaks it.
        """
        status = await self._computer_use.get_status(request_timeout=_REQUEST_TIMEOUT)
        if status.status != 'active':
            raise WorkspaceError(f"Daytona's desktop is not running: its status is {status.status!r}.")

    async def display_size(self) -> tuple[int, int]:
        displays = (await self._computer_use.display.get_info(request_timeout=_REQUEST_TIMEOUT)).displays or []
        display = next((display for display in displays if display.is_active), None)
        if display is None or not display.width or not display.height:
            raise WorkspaceError(f'Daytona reported no display size: {displays!r}')
        return display.width, display.height

    async def screenshot(self) -> bytes:
        response = await self._computer_use.screenshot.take_full_screen(request_timeout=_REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def zoom(self, region: Region) -> bytes:
        area = ScreenshotRegion(
            x=region.left, y=region.top, width=region.right - region.left, height=region.bottom - region.top
        )
        response = await self._computer_use.screenshot.take_region(area, request_timeout=_REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def cursor_position(self) -> Point:
        position = await self._computer_use.mouse.get_position(request_timeout=_REQUEST_TIMEOUT)
        if position.x is None or position.y is None:
            raise WorkspaceError(f'Daytona reported no pointer position: {position!r}')
        return Point(x=position.x, y=position.y)

    async def _at(self, x: int | None, y: int | None) -> tuple[int, int]:
        """The position, or the pointer's: Daytona's click and scroll always take one."""
        if x is None or y is None:
            point = await self.cursor_position()
            return point.x, point.y
        return x, y

    @asynccontextmanager
    async def _holding(self, keys: Sequence[str]) -> AsyncGenerator[None]:
        """Hold Daytona's `keys` down, in order; on any failure or cancel, even while letting go, let go of them all."""
        keyboard = self._computer_use.keyboard
        pressed: list[str] = []
        try:
            for key in keys:
                pressed.append(key)  # before the request, which can press the key and still fail
                await keyboard.down(key, request_timeout=_REQUEST_TIMEOUT)
            yield
            while pressed:
                await keyboard.up(pressed[-1], request_timeout=_REQUEST_TIMEOUT)
                pressed.pop()  # only once it's up, so a failed release is retried below
        except BaseException:
            with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True):
                for key in reversed(pressed):
                    with suppress(Exception):
                        await keyboard.up(key, request_timeout=_REQUEST_TIMEOUT)
            raise

    async def _scroll(self, x: int, y: int, direction: str, amount: int, modifiers: list[str]) -> None:
        """Scroll `amount` wheel clicks, `_SCROLLED_PER_REQUEST` at a time."""
        for start in range(0, amount, _SCROLLED_PER_REQUEST):
            clicks = min(_SCROLLED_PER_REQUEST, amount - start)
            # The one call that reports a failure in its result rather than raising.
            if not await self._computer_use.mouse.scroll(
                x, y, direction, clicks, modifiers=modifiers, request_timeout=_REQUEST_TIMEOUT
            ):
                raise WorkspaceError("Daytona's desktop did not scroll.")

    async def _type(self, text: str) -> None:
        """Type `text`, `_TYPED_PER_REQUEST` characters at a time, pressing Tab for each tab, which `type` refuses."""
        if any(ord(character) > 0xFFFF for character in text):
            raise UnsupportedActionError("Daytona's desktop cannot type emoji or other characters beyond U+FFFF.")
        keyboard = self._computer_use.keyboard
        for index, segment in enumerate(text.split('\t')):
            if index:
                await keyboard.press('tab', request_timeout=_REQUEST_TIMEOUT)
            for start in range(0, len(segment), _TYPED_PER_REQUEST):
                await anyio.lowlevel.checkpoint_if_cancelled()  # a cancel stops here, before the next part is sent
                # Daytona types a part it accepted even if we stop waiting, so a cancel waits for it: the next action
                # must not reach the screen before this text does.
                with anyio.CancelScope(shield=True):
                    await keyboard.type(segment[start : start + _TYPED_PER_REQUEST], request_timeout=_TYPE_TIMEOUT)

    async def act(self, action: DisplayAction) -> None:
        await self._check_running()
        mouse = self._computer_use.mouse
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                button, held = _button(button), _keys(modifiers)
                x, y = await self._at(x, y)
                await mouse.click(x, y, button, clicks=count, modifiers=held, request_timeout=_REQUEST_TIMEOUT)
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                async with self._holding(_keys(modifiers)):
                    await mouse.move(x, y, request_timeout=_REQUEST_TIMEOUT)
            case Drag(path=path, modifiers=modifiers):
                async with self._holding(_keys(modifiers)):
                    await self._drag(path)
            case MouseDown(button=button):
                await mouse.down(button=_button(button), request_timeout=_REQUEST_TIMEOUT)
            case MouseUp(button=button):
                await mouse.up(button=_button(button), request_timeout=_REQUEST_TIMEOUT)
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                held = _keys(modifiers)
                await self._scroll(*await self._at(x, y), direction, amount, held)
            case TypeText(text=text):
                await self._type(text)
            case PressKeys(keys=keys, repeat=repeat):
                await self._press(_keys(keys), repeat)
            case HoldKeys(keys=keys, seconds=seconds):
                async with self._holding(_keys(keys)):
                    await anyio.sleep(seconds)
            case _:
                raise UnsupportedActionError(f"Daytona's desktop cannot perform {action!r}.")

    async def _drag(self, path: Sequence[Point]) -> None:
        mouse, start, end = self._computer_use.mouse, path[0], path[-1]
        try:
            await mouse.down(start.x, start.y, request_timeout=_REQUEST_TIMEOUT)
            await anyio.sleep(DRAG_PRESS_PAUSE)
            for point in drag_steps(path):
                await anyio.sleep(DRAG_STEP_PAUSE)
                await mouse.move(point.x, point.y, request_timeout=_REQUEST_TIMEOUT)
            await anyio.sleep(DRAG_PRESS_PAUSE)
            await mouse.up(end.x, end.y, request_timeout=_REQUEST_TIMEOUT)
        except BaseException:
            with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True), suppress(Exception):
                await mouse.up(request_timeout=_REQUEST_TIMEOUT)
            raise

    async def _press(self, names: list[str], repeat: int) -> None:
        modifiers = [name for name in names if name in _MODIFIERS.values()]
        others = [name for name in names if name not in modifiers]
        for _ in range(repeat):
            if len(others) == 1:
                await self._computer_use.keyboard.press(others[0], modifiers, request_timeout=_REQUEST_TIMEOUT)
            else:  # only modifiers, or several other keys: `press` takes one
                async with self._holding(names):
                    pass
