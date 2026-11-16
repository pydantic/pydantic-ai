"""`DaytonaDesktop`: display control for a Daytona sandbox, through Daytona's computer-use API.

How Daytona's default snapshot behaves. Re-run the live display tests in
`tests/harness/daytona_sandbox/test_daytona_live.py` before changing the mapping:

- `computer_use.start()` returns once the display answers; starting a running desktop is harmless.
- `VNC_RESOLUTION` on a new sandbox sets the display size, such as 800x600; without it, the size is 1024x768.
- Screenshots are PNGs at the display's size; a region screenshot is the region at full resolution.
- `mouse.scroll`'s amount is wheel clicks; `left` and `right` send X buttons 6 and 7.
- Modifiers given to `click`, `scroll` and `press` are held around the event; `cmd` is `Super_L`.
- `keyboard.down` and `keyboard.up` take letters and named keys, not only modifiers.
- `keyboard.type` types Unicode up to U+FFFF and newlines as Enter, and drops characters beyond U+FFFF, such as most
  emoji.
- `keyboard.press` refuses shifted symbols such as `!`, X keysyms such as `KP_Enter`, and print screen, pause,
  scroll lock and media keys (`unsupported key "printscreen"`), and the desktop keeps working.
- The mouse has no `back` or `forward` button (`unsupported mouse button "back"`).
- Its keypad digits are `num0` to `num9`, which follow Num Lock, and Num Lock is `num_lock`. Its other keypad names
  type other keys (`num_plus` types `n`), and the display refuses `menu`.
- Pressing `f13` to `f24` crashes the computer-use process for good.
- After the sandbox stops and starts, `get_status()` says `inactive` until `start()`. Input sent before that start
  breaks the desktop until the sandbox stops and starts again: input answers 500 (`connection is shut down`), and
  `start()` and `get_status()` answer 503.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, TypeVar

import anyio
import anyio.lowlevel
from daytona import (
    DaytonaAuthenticationError,
    DaytonaBadGatewayError,
    DaytonaConnectionError,
    DaytonaForbiddenError,
    DaytonaTimeoutError,
    ScreenshotRegion,
)

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
from pydantic_ai.workspaces import WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness._display import png_from_base64
from pydantic_ai_harness._drag import DRAG_PRESS_PAUSE, DRAG_STEP_PAUSE, drag_steps
from pydantic_ai_harness._workspace_provider import run_to_completion
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError

if TYPE_CHECKING:
    from daytona import AsyncComputerUse

_T = TypeVar('_T')

DISPLAY = ':0'
"""The X display of every Daytona sandbox's desktop."""

REQUEST_TIMEOUT = 30
"""Maximum time for one Daytona API request."""

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_TYPE_TIMEOUT = 120
"""Maximum time for one typing request, of at most `_TYPED_PER_REQUEST` characters."""

_TYPED_PER_REQUEST = 2_000
"""Characters typed by one request, well inside `_TYPE_TIMEOUT`: 2,000 take about 11 seconds."""

_SCROLLED_PER_REQUEST = 500
"""Wheel clicks scrolled by one request, well inside `REQUEST_TIMEOUT`: 1,000 take about 21 seconds."""

_MODIFIERS = frozenset({'ctrl', 'shift', 'alt', 'cmd'})
"""Daytona's names for the modifier keys."""

_RENAMED_KEYS = {'super': 'cmd', **{f'numpad{n}': f'num{n}' for n in range(10)}, 'numlock': 'num_lock'}
"""Normalized key names Daytona presses under another name; it presses the rest under their own, or refuses them."""

_REFUSED_KEYS = frozenset(
    {
        *(f'f{n}' for n in range(13, 25)),
        'numpad_enter',
        'numpad_add',
        'numpad_subtract',
        'numpad_multiply',
        'numpad_divide',
        'numpad_decimal',
    }
)
"""Normalized key names Daytona would get wrong: `f13` to `f24` crash its desktop, and its keypad operator and Enter
keys type other characters."""

_SHIFTED = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))
"""A symbol typed with shift, and the key it is on: Daytona's display has a US keyboard layout."""


async def _send_input(request: Callable[[], Awaitable[_T]]) -> _T:
    """Send one input request to Daytona, which carries out a request it accepted even if we stop waiting.

    So a cancel stops before the request is sent, or waits for it to finish: the next action, perhaps another run's,
    must not reach the screen before it. Every input request goes through here. For the same reason, a request whose
    outcome is unknown ends the run: a timeout (including a 504) doesn't stop Daytona carrying it out, nor does a dropped
    connection, nor the 502 its proxy returns when it loses the sandbox after passing the request on.
    """
    await anyio.lowlevel.checkpoint_if_cancelled()
    try:
        return await run_to_completion(request)
    except (DaytonaTimeoutError, DaytonaConnectionError, DaytonaBadGatewayError) as error:
        raise WorkspaceUnavailableError(
            f"Daytona didn't confirm the input, which may still be running on its desktop: {error}"
        ) from error


async def _let_go(error: BaseException, held: list[str], release: Callable[[str], Awaitable[object]]) -> None:
    """After `error`, let go of each of `held`, last first; a failure that leaves any held says which.

    A cancel or an earlier deadline doesn't stop the releases, and propagates after them, unless a release goes
    unconfirmed: its `WorkspaceUnavailableError` then replaces the cancel, and the run ends either way.
    """
    with anyio.move_on_after(_RELEASE_TIMEOUT, shield=True):
        for name in reversed(list(held)):
            try:
                await _send_input(partial(release, name))
                held.remove(name)
            except WorkspaceUnavailableError:
                raise  # a release that may still be running ends the run, as any unconfirmed input does
            except Exception:
                pass
    # An input that may still be running ends the run, so the model won't act with these held. A credential failure
    # goes on unwrapped, for the backend to clean its text and end the run.
    if (
        held
        and isinstance(error, Exception)
        and not isinstance(error, (WorkspaceUnavailableError, DaytonaAuthenticationError, DaytonaForbiddenError))
    ):
        raise WorkspaceError(
            f'{error}; releasing {" and ".join(held)} failed too, so {"it" if len(held) == 1 else "they"} '
            'may still be held'
        ) from error


def _keys(keys: Sequence[str]) -> list[str]:
    """Daytona's names for the keys, in order, with `shift` before a shifted symbol."""
    names: list[str] = []
    for key in keys:
        if key in _REFUSED_KEYS:
            raise UnsupportedActionError(f"Daytona's desktop cannot press the key {key!r}.")
        if key in _SHIFTED:
            names += ['shift', _SHIFTED[key]]
        else:
            names.append(_RENAMED_KEYS.get(key, key))
    return list(dict.fromkeys(names))


class DaytonaDesktop:
    """Controls a Daytona sandbox's desktop, which must already be started, with its computer-use API.

    `computer_use` returns the attached sandbox's API. It is read on every call, since the backend attaches again
    with a new client after closing the old one.
    """

    def __init__(self, computer_use: Callable[[], AsyncComputerUse]):
        self._computer_use = computer_use

    async def display_size(self) -> tuple[int, int]:
        displays = (await self._computer_use().display.get_info(request_timeout=REQUEST_TIMEOUT)).displays or []
        display = next((display for display in displays if display.is_active), None)
        if display is None or not display.width or not display.height:
            raise WorkspaceError(f'Daytona reported no display size: {displays!r}')
        return display.width, display.height

    async def screenshot(self) -> bytes:
        response = await self._computer_use().screenshot.take_full_screen(request_timeout=REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def zoom(self, region: Region) -> bytes:
        area = ScreenshotRegion(
            x=region.left, y=region.top, width=region.right - region.left, height=region.bottom - region.top
        )
        response = await self._computer_use().screenshot.take_region(area, request_timeout=REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def cursor_position(self) -> Point:
        position = await self._computer_use().mouse.get_position(request_timeout=REQUEST_TIMEOUT)
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
        """Hold Daytona's `keys` down, in order; on any failure or cancel, even while letting go, let go of them all.

        A failure that leaves keys down says which, so the model doesn't type on with them held.
        """
        keyboard = self._computer_use().keyboard
        pressed: list[str] = []
        try:
            for key in keys:
                pressed.append(key)  # before the request, which can press the key and still fail
                await _send_input(partial(keyboard.down, key, request_timeout=REQUEST_TIMEOUT))
            yield
            while pressed:
                await _send_input(partial(keyboard.up, pressed[-1], request_timeout=REQUEST_TIMEOUT))
                pressed.pop()  # only once it's up, so a failed release is retried below
        except BaseException as error:
            await _let_go(error, pressed, partial(keyboard.up, request_timeout=REQUEST_TIMEOUT))
            raise

    async def _scroll(self, x: int, y: int, direction: str, amount: int, modifiers: list[str]) -> None:
        """Scroll `amount` wheel clicks, `_SCROLLED_PER_REQUEST` at a time."""
        for start in range(0, amount, _SCROLLED_PER_REQUEST):
            clicks = min(_SCROLLED_PER_REQUEST, amount - start)
            scroll = partial(
                self._computer_use().mouse.scroll,
                x,
                y,
                direction,
                clicks,
                modifiers=modifiers,
                request_timeout=REQUEST_TIMEOUT,
            )
            # The one call that reports a failure in its result rather than raising.
            if not await _send_input(scroll):
                raise WorkspaceError("Daytona's desktop did not scroll.")

    async def _type(self, text: str) -> None:
        """Type `text`, `_TYPED_PER_REQUEST` characters at a time."""
        if any(ord(character) > 0xFFFF for character in text):
            raise UnsupportedActionError("Daytona's desktop cannot type characters beyond U+FFFF, such as most emoji.")
        keyboard = self._computer_use().keyboard
        for start in range(0, len(text), _TYPED_PER_REQUEST):
            part = text[start : start + _TYPED_PER_REQUEST]
            await _send_input(partial(keyboard.type, part, request_timeout=_TYPE_TIMEOUT))

    async def act(self, action: DisplayAction) -> None:
        mouse = self._computer_use().mouse
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                held = _keys(modifiers)
                x, y = await self._at(x, y)
                await _send_input(
                    partial(mouse.click, x, y, button, clicks=count, modifiers=held, request_timeout=REQUEST_TIMEOUT)
                )
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                async with self._holding(_keys(modifiers)):
                    await _send_input(partial(mouse.move, x, y, request_timeout=REQUEST_TIMEOUT))
            case Drag(path=path, modifiers=modifiers):
                async with self._holding(_keys(modifiers)):
                    await self._drag(path)
            case MouseDown(button=button):
                await _send_input(partial(mouse.down, button=button, request_timeout=REQUEST_TIMEOUT))
            case MouseUp(button=button):
                await _send_input(partial(mouse.up, button=button, request_timeout=REQUEST_TIMEOUT))
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
        mouse, start, end = self._computer_use().mouse, path[0], path[-1]
        try:
            await _send_input(partial(mouse.down, start.x, start.y, request_timeout=REQUEST_TIMEOUT))
            await anyio.sleep(DRAG_PRESS_PAUSE)
            for point in drag_steps(path):
                await anyio.sleep(DRAG_STEP_PAUSE)
                await _send_input(partial(mouse.move, point.x, point.y, request_timeout=REQUEST_TIMEOUT))
            await anyio.sleep(DRAG_PRESS_PAUSE)
            await _send_input(partial(mouse.up, end.x, end.y, request_timeout=REQUEST_TIMEOUT))
        except BaseException as error:
            await _let_go(error, ['the left mouse button'], lambda _: mouse.up(request_timeout=REQUEST_TIMEOUT))
            raise

    async def _press(self, names: list[str], repeat: int) -> None:
        modifiers = [name for name in names if name in _MODIFIERS]
        others = [name for name in names if name not in modifiers]
        for _ in range(repeat):
            if len(others) == 1:
                keyboard = self._computer_use().keyboard
                await _send_input(partial(keyboard.press, others[0], modifiers, request_timeout=REQUEST_TIMEOUT))
            else:  # only modifiers, or several other keys: `press` takes one
                async with self._holding(names):
                    pass
