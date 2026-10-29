"""`DaytonaDesktop`: display control for a Daytona sandbox, through Daytona's computer-use API.

Behavior verified 2026-10-24 against Daytona's default snapshot (daemon behind `daytona` 0.223.1), with `xev`
on the sandbox's display. Re-check these before changing the mapping, by running the live display tests in
`tests/harness/daytona_sandbox/test_daytona_live.py`:

- `computer_use.start()` returns once the display answers, and starting a running desktop is harmless.
- `VNC_RESOLUTION` set on a new sandbox, such as `1280x800`, is the display's size; without it, or with a smaller size such as `640x400`, 1024x768 (checked 2026-10-29).
- Screenshots are PNGs at the display's size; a region screenshot is the region at full resolution.
- `mouse.scroll`'s amount is wheel clicks, and `left`/`right` send X buttons 6 and 7.
- Modifiers given to `click`, `scroll` and `press` are held around the event; `cmd` is `Super_L`.
- `keyboard.down` and `keyboard.up` take letters and named keys, not only modifiers.
- `keyboard.type` types Unicode text up to U+FFFF, newlines as Enter; it drops emoji and anything else beyond.
- `keyboard.press` refuses a shifted symbol such as `!`, and names outside its own list: X keysyms such as `KP_Enter`,
  and print screen, pause, scroll lock and media keys, which it has no name for.
- Its names for the numpad are `num0` to `num9`, `num_enter`, `num_plus`, `num_minus`, `num_asterisk`, `num_slash`,
  `num_decimal` and `num_lock` (checked 2026-10-28); `menu` is refused by the display.
- Pressing `f13` to `f24` crashes the sandbox's computer-use process for good: nothing restarts it.
- After the sandbox stops and starts, `computer_use.get_status()` says `inactive` and `start()` brings the desktop
  back; an input request before that start answers 500 and breaks the desktop for good: every later request answers
  503 until the sandbox stops and starts again (checked 2026-10-28).
- `take_compressed` with a PNG `scale` from 0.1 to 1.0 returns the screen scaled (checked 2026-10-28).
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

import anyio
from daytona import ScreenshotOptions, ScreenshotRegion

from pydantic_ai.workspaces import (
    Click,
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
    TypeText,
    WorkspaceError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness._display import png_from_base64
from pydantic_ai_harness._drag import DRAG_PRESS_PAUSE, DRAG_STEP_PAUSE, drag_steps

if TYPE_CHECKING:
    from daytona import AsyncComputerUse

_REQUEST_TIMEOUT = 30

_RELEASE_TIMEOUT = 10
"""Maximum time spent releasing held keys and buttons after a failed or cancelled action."""

_TYPE_TIMEOUT = 120
"""Typing 2,000 characters, the most one action types, takes about 10 s."""

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
        raise WorkspaceUnsupportedActionError(f"Daytona's desktop has no {button!r} mouse button.")
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
            raise WorkspaceUnsupportedActionError(f"Daytona's desktop cannot press the key {key!r}.")
    return list(dict.fromkeys(names))


class DaytonaDesktop:
    """Controls a Daytona sandbox's desktop, which must already be started, with its computer-use API."""

    def __init__(self, computer_use: AsyncComputerUse):
        self._computer_use = computer_use
        # The sandbox may stop and start between a screenshot and the next action, such as while the model thinks,
        # and input sent to its desktop before the desktop starts again breaks it for good.
        self._check_before_acting = True

    async def _check_running(self) -> None:
        """Refuse the first action after a screenshot on a desktop that stopped, before sending it any input."""
        if not self._check_before_acting:
            return
        status = await self._computer_use.get_status(request_timeout=_REQUEST_TIMEOUT)
        if status.status != 'active':
            raise WorkspaceError(f"Daytona's desktop is not running: its status is {status.status!r}.")
        self._check_before_acting = False

    async def display_size(self) -> tuple[int, int]:
        displays = (await self._computer_use.display.get_info(request_timeout=_REQUEST_TIMEOUT)).displays or []
        display = next((display for display in displays if display.is_active), None)
        if display is None or not display.width or not display.height:
            raise WorkspaceError(f'Daytona reported no display size: {displays!r}')
        return display.width, display.height

    async def screenshot(self) -> bytes:
        response = await self._computer_use.screenshot.take_full_screen(request_timeout=_REQUEST_TIMEOUT)
        self._check_before_acting = True
        return png_from_base64(response.screenshot, 'Daytona')

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        screen_width, screen_height = await self.display_size()
        scale = min(width / screen_width, height / screen_height, 1)
        if scale < 0.1:
            # Daytona scales by a tenth at the least, which would come back larger than asked.
            raise WorkspaceUnsupportedActionError(
                f"Daytona's desktop cannot scale its {screen_width}x{screen_height} screen down to {width}x{height}."
            )
        options = ScreenshotOptions(fmt='png', scale=scale)
        response = await self._computer_use.screenshot.take_compressed(options, request_timeout=_REQUEST_TIMEOUT)
        self._check_before_acting = True
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
        """Hold Daytona's `keys` down, in order; whatever fails or is cancelled, even letting go, lets go of them all."""
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

    async def act(self, action: ComputerAction) -> None:
        await self._check_running()
        mouse, keyboard = self._computer_use.mouse, self._computer_use.keyboard
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                button, held = _button(button), _keys(modifiers)
                x, y = await self._at(x, y)
                await mouse.click(x, y, button, clicks=count, modifiers=held, request_timeout=_REQUEST_TIMEOUT)
            case MoveMouse(x=x, y=y, modifiers=modifiers):
                async with self._holding(_keys(modifiers)):
                    await mouse.move(x, y, request_timeout=_REQUEST_TIMEOUT)
            case Drag(path=path, modifiers=modifiers):
                start, end, held = path[0], path[-1], _keys(modifiers)
                async with self._holding(held):
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
            case MouseDown(button=button):
                await mouse.down(button=_button(button), request_timeout=_REQUEST_TIMEOUT)
            case MouseUp(button=button):
                await mouse.up(button=_button(button), request_timeout=_REQUEST_TIMEOUT)
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                held = _keys(modifiers)
                x, y = await self._at(x, y)
                await mouse.scroll(x, y, direction, amount, modifiers=held, request_timeout=_REQUEST_TIMEOUT)
            case TypeText(text=text):
                if any(ord(character) > 0xFFFF for character in text):
                    raise WorkspaceUnsupportedActionError(
                        "Daytona's desktop cannot type emoji or other characters beyond U+FFFF."
                    )
                await keyboard.type(text, request_timeout=_TYPE_TIMEOUT)
            case PressKeys(keys=keys, repeat=repeat):
                names = _keys(keys)
                modifiers = [name for name in names if name in _MODIFIERS.values()]
                others = [name for name in names if name not in modifiers]
                for _ in range(repeat):
                    if len(others) == 1:
                        await keyboard.press(others[0], modifiers, request_timeout=_REQUEST_TIMEOUT)
                    else:  # only modifiers, or several other keys: `press` takes one
                        async with self._holding(names):
                            pass
            case HoldKeys(keys=keys, seconds=seconds):
                async with self._holding(_keys(keys)):
                    await anyio.sleep(seconds)
            case _:
                raise WorkspaceUnsupportedActionError(f"Daytona's desktop cannot perform {action!r}.")
