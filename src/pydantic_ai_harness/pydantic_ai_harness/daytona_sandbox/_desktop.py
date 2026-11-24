"""`DaytonaDisplayDriver` controls a Daytona sandbox's screen through Daytona's computer-use API.

Below is how Daytona's default snapshot behaves. Re-run the live display tests in
`tests/harness/daytona_sandbox/test_daytona_live.py` before changing how actions map to Daytona's API:

- `computer_use.start()` returns once the display answers. Starting a desktop that already runs does no harm.
- `VNC_RESOLUTION` on a new sandbox sets the display size, such as 800x600. Without it, the size is 1024x768.
- Screenshots are PNGs at the display's size. A region screenshot is the region at full resolution.
- `mouse.scroll`'s amount counts wheel clicks. `left` and `right` send X buttons 6 and 7.
- Modifiers given to `click`, `scroll` and `press` are held down around the event. `cmd` is `Super_L`.
- `keyboard.down` and `keyboard.up` take letters and named keys, not only modifiers.
- `keyboard.type` types Unicode up to U+FFFF and newlines as Enter, and drops characters beyond U+FFFF, such as most
  emoji.
- `keyboard.press` refuses shifted symbols such as `!`, X key names such as `KP_Enter`, and print screen, pause and
  scroll lock (`unsupported key "printscreen"`). The desktop keeps working after that. `keyboard.down` refuses the
  same keys with a 500, as it does `menu` (live, 2026-11-18).
- The mouse has `left`, `right` and `middle` buttons only (`unsupported mouse button "back"`, a 400).
- Num Lock is `num_lock`.
- After the sandbox stops and starts, `get_status()` says `inactive` until `start()`. Input sent before that start
  breaks the desktop until the sandbox stops and starts again. Until then, input answers 500
  (`connection is shut down`), and `start()` and `get_status()` answer 503.
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
    MouseMove,
    MouseUp,
    NamedKey,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness._workspace_provider import run_to_completion
from pydantic_ai_harness.computer_use import DisplayAction, UnsupportedActionError
from pydantic_ai_harness.computer_use._display import RELEASE_TIMEOUT, png_from_base64
from pydantic_ai_harness.computer_use._drag import DRAG_STEP_PAUSE, drag_steps

if TYPE_CHECKING:
    from daytona import AsyncComputerUse

_T = TypeVar('_T')

DISPLAY = ':0'
"""The X display of every Daytona sandbox's desktop."""

REQUEST_TIMEOUT = 30
"""Maximum time for one Daytona API request."""

CHROMIUM_SETTINGS_DIR = '/etc/chromium.d'

_NO_SANDBOX_FLAG_FILE = f'{CHROMIUM_SETTINGS_DIR}/pydantic-ai-no-sandbox'

ADD_CHROMIUM_NO_SANDBOX_FLAG = (
    f'[ ! -d {CHROMIUM_SETTINGS_DIR} ] || [ -e {_NO_SANDBOX_FLAG_FILE} ] || '
    """echo 'CHROMIUM_FLAGS="$CHROMIUM_FLAGS --no-sandbox"' | """
    f'if [ "$(id -u)" = 0 ]; then tee {_NO_SANDBOX_FLAG_FILE}; '
    f'elif ! unshare -Ur true 2>/dev/null; then sudo -n tee {_NO_SANDBOX_FLAG_FILE}; fi >/dev/null 2>&1'
)
"""Makes the desktop's own browser launchers work. The default snapshot's dock "Web Browser" and its menu's Chromium
both run Debian's `/usr/bin/chromium`, which exits with "No usable sandbox!" as the `daytona` user, since the user
can't make a user namespace there, and refuses to run as root. That script reads every file in `/etc/chromium.d`, so
one file there adds `--no-sandbox`, as root or with the user's passwordless `sudo`. It does nothing where Chromium's
sandbox works, where there's no Debian Chromium, or without root or `sudo`. Daytona runs it with zsh, so it doesn't rely
on sh's word splitting."""

_TYPE_TIMEOUT = 60
"""Maximum time for one typing request, of at most `_MAX_CHARS_PER_CALL` characters."""

_MAX_CHARS_PER_CALL = 2_000
"""Characters typed by one request, well inside `_TYPE_TIMEOUT`: 2,000 take about 11 seconds."""

_MAX_SCROLL_CLICKS_PER_CALL = 500
"""Wheel clicks scrolled by one request, well inside `REQUEST_TIMEOUT`: 1,000 take about 21 seconds."""

_MODIFIERS = frozenset({'ctrl', 'shift', 'alt', 'cmd'})

_RENAMED_KEYS: dict[NamedKey, str] = {'super': 'cmd', 'numlock': 'num_lock'}
"""`ComputerUse` key names that Daytona spells differently."""

_UNSUPPORTED_KEYS: frozenset[NamedKey] = frozenset({'printscreen', 'pause', 'scrolllock', 'menu'})
"""`ComputerUse` key names that Daytona refuses, with an error that doesn't prove nothing was pressed. They are refused
before they're sent, because a sent key counts as possibly held until its release is confirmed."""

_UNSHIFTED_KEY_FOR_SYMBOL = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))
"""Each symbol typed with shift, mapped to the key it is on. Daytona's display has a US keyboard layout."""


async def _send_input(request: Callable[[], Awaitable[_T]]) -> _T:
    """Send one input request to Daytona.

    Daytona carries out a request it accepted, even if we stop waiting for it. So a cancel either stops before the
    request is sent, or waits for it to finish: the next action, perhaps from another run, must not reach the screen
    first. Every input request goes through here. A request whose outcome is unknown ends the run: Daytona still
    carries out a request after a timeout (including a 504), a dropped connection, or the 502 its proxy returns when
    it loses the sandbox after passing the request on.
    """
    await anyio.lowlevel.checkpoint_if_cancelled()
    try:
        return await run_to_completion(request)
    except (DaytonaTimeoutError, DaytonaConnectionError, DaytonaBadGatewayError) as error:
        raise WorkspaceUnavailableError(
            f"Daytona didn't confirm the input, which may still be running on its desktop: {error}"
        ) from error


async def _release_after_failure(
    error: BaseException, held: list[str], release: Callable[[str], Awaitable[object]]
) -> None:
    """After `error`, let go of each key or button in `held`, last first. If any stay held, the error says which.

    Releases run even after a cancel. If Daytona doesn't confirm a release, that error is raised instead of the cancel.
    """
    with anyio.move_on_after(RELEASE_TIMEOUT, shield=True):
        for name in reversed(list(held)):
            try:
                await _send_input(partial(release, name))
                held.remove(name)
            except WorkspaceUnavailableError:
                raise
            except Exception:
                pass
    # A credential failure is raised as is, so the backend can clean its text.
    if (
        held
        and isinstance(error, Exception)
        and not isinstance(error, (WorkspaceUnavailableError, DaytonaAuthenticationError, DaytonaForbiddenError))
    ):
        raise WorkspaceError(
            f'{error}; releasing {" and ".join(held)} failed too, so {"it" if len(held) == 1 else "they"} '
            'may still be held'
        ) from error


def _daytona_key_names(keys: Sequence[str]) -> list[str]:
    """Daytona's names for the keys, in order, with `shift` before a shifted symbol."""
    names: list[str] = []
    for key in keys:
        if key in _UNSUPPORTED_KEYS:
            raise UnsupportedActionError(f"Daytona's desktop cannot press the key {key!r}.")
        if key in _UNSHIFTED_KEY_FOR_SYMBOL:
            names += ['shift', _UNSHIFTED_KEY_FOR_SYMBOL[key]]
        elif key in _RENAMED_KEYS:
            names.append(_RENAMED_KEYS[key])
        else:
            names.append(key)
    return list(dict.fromkeys(names))


class DaytonaDisplayDriver:
    """Controls a Daytona sandbox's desktop, which must already be started, with its computer-use API.

    `get_computer_use` returns the attached sandbox's API. It is read on every call, since the backend attaches again
    with a new client after closing the old one.
    """

    def __init__(self, get_computer_use: Callable[[], AsyncComputerUse]):
        self._get_computer_use = get_computer_use

    async def display_size(self) -> tuple[int, int]:
        displays = (await self._get_computer_use().display.get_info(request_timeout=REQUEST_TIMEOUT)).displays or []
        display = next((display for display in displays if display.is_active), None)
        if display is None or not display.width or not display.height:
            raise WorkspaceError(f'Daytona reported no display size: {displays!r}')
        return display.width, display.height

    async def screenshot(self) -> bytes:
        response = await self._get_computer_use().screenshot.take_full_screen(request_timeout=REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def zoom(self, region: Region) -> bytes:
        area = ScreenshotRegion(
            x=region.left, y=region.top, width=region.right - region.left, height=region.bottom - region.top
        )
        response = await self._get_computer_use().screenshot.take_region(area, request_timeout=REQUEST_TIMEOUT)
        return png_from_base64(response.screenshot, 'Daytona')

    async def cursor_position(self) -> Point:
        position = await self._get_computer_use().mouse.get_position(request_timeout=REQUEST_TIMEOUT)
        if position.x is None or position.y is None:
            raise WorkspaceError(f'Daytona reported no pointer position: {position!r}')
        return Point(x=position.x, y=position.y)

    async def _position_or_pointer(self, x: int | None, y: int | None) -> tuple[int, int]:
        """The given position, or the pointer's if none is given. Daytona's click and scroll always need a position."""
        if x is None or y is None:
            point = await self.cursor_position()
            return point.x, point.y
        return x, y

    @asynccontextmanager
    async def _holding(self, keys: Sequence[str]) -> AsyncGenerator[None]:
        """Hold `keys` down, in order. On any failure or cancel, even while letting go, let go of all of them.

        A failure that leaves keys down says which, so the model doesn't type on with them held.
        """
        keyboard = self._get_computer_use().keyboard
        pressed: list[str] = []
        try:
            for key in keys:
                # Recorded before the request, because the request may press the key whatever Daytona answers. Keys
                # Daytona refuses are refused before this point, so any failure here leaves this key to let go.
                pressed.append(key)
                await _send_input(partial(keyboard.down, key, request_timeout=REQUEST_TIMEOUT))
            yield
            while pressed:
                await _send_input(partial(keyboard.up, pressed[-1], request_timeout=REQUEST_TIMEOUT))
                pressed.pop()  # only once it's up, so a failed release is retried below
        except BaseException as error:
            await _release_after_failure(error, pressed, partial(keyboard.up, request_timeout=REQUEST_TIMEOUT))
            raise

    async def _scroll(self, x: int, y: int, direction: str, amount: int, modifiers: list[str]) -> None:
        """Scroll `amount` wheel clicks, `_MAX_SCROLL_CLICKS_PER_CALL` at a time."""
        for start in range(0, amount, _MAX_SCROLL_CLICKS_PER_CALL):
            clicks = min(_MAX_SCROLL_CLICKS_PER_CALL, amount - start)
            scroll = partial(
                self._get_computer_use().mouse.scroll,
                x,
                y,
                direction,
                clicks,
                modifiers=modifiers,
                request_timeout=REQUEST_TIMEOUT,
            )
            # Scroll is the one call that reports a failure in its result instead of raising.
            if not await _send_input(scroll):
                raise WorkspaceError("Daytona's desktop did not scroll.")

    async def _type(self, text: str) -> None:
        """Type `text`, `_MAX_CHARS_PER_CALL` characters at a time."""
        if any(ord(character) > 0xFFFF for character in text):
            raise UnsupportedActionError("Daytona's desktop cannot type characters beyond U+FFFF, such as most emoji.")
        keyboard = self._get_computer_use().keyboard
        # Daytona does this within each request. Doing it first keeps a `\r\n` split across two requests from being
        # typed as two Enters.
        text = text.replace('\r\n', '\n')
        for start in range(0, len(text), _MAX_CHARS_PER_CALL):
            part = text[start : start + _MAX_CHARS_PER_CALL]
            await _send_input(partial(keyboard.type, part, request_timeout=_TYPE_TIMEOUT))

    async def act(self, action: DisplayAction) -> None:
        mouse = self._get_computer_use().mouse
        match action:
            case Click(x=x, y=y, button=button, count=count, modifiers=modifiers):
                modifier_names = _daytona_key_names(modifiers)
                x, y = await self._position_or_pointer(x, y)
                await _send_input(
                    partial(
                        mouse.click,
                        x,
                        y,
                        button,
                        clicks=count,
                        modifiers=modifier_names,
                        request_timeout=REQUEST_TIMEOUT,
                    )
                )
            case MouseMove(x=x, y=y):
                await _send_input(partial(mouse.move, x, y, request_timeout=REQUEST_TIMEOUT))
            case Drag(path=path, modifiers=modifiers):
                async with self._holding(_daytona_key_names(modifiers)):
                    await self._drag(path)
            case MouseDown(button=button):
                await _send_input(partial(mouse.down, button=button, request_timeout=REQUEST_TIMEOUT))
            case MouseUp(button=button):
                await _send_input(partial(mouse.up, button=button, request_timeout=REQUEST_TIMEOUT))
            case Scroll(x=x, y=y, direction=direction, amount=amount, modifiers=modifiers):
                modifier_names = _daytona_key_names(modifiers)
                await self._scroll(*await self._position_or_pointer(x, y), direction, amount, modifier_names)
            case TypeText(text=text):
                await self._type(text)
            case PressKeys(keys=keys, repeat=repeat):
                await self._press(_daytona_key_names(keys), repeat)
            case HoldKeys(keys=keys, seconds=seconds):
                async with self._holding(_daytona_key_names(keys)):
                    await anyio.sleep(seconds)
            case _:
                raise UnsupportedActionError(f"Daytona's desktop cannot perform {action!r}.")

    async def _drag(self, path: Sequence[Point]) -> None:
        mouse, start, end = self._get_computer_use().mouse, path[0], path[-1]
        try:
            await _send_input(partial(mouse.down, start.x, start.y, request_timeout=REQUEST_TIMEOUT))
            for point in drag_steps(path):
                await anyio.sleep(DRAG_STEP_PAUSE)
                await _send_input(partial(mouse.move, point.x, point.y, request_timeout=REQUEST_TIMEOUT))
            await anyio.sleep(DRAG_STEP_PAUSE)
            await _send_input(partial(mouse.up, end.x, end.y, request_timeout=REQUEST_TIMEOUT))
        except BaseException as error:
            await _release_after_failure(
                error, ['the left mouse button'], lambda _: mouse.up(request_timeout=REQUEST_TIMEOUT)
            )
            raise

    async def _press(self, names: list[str], repeat: int) -> None:
        modifiers = [name for name in names if name in _MODIFIERS]
        non_modifiers = [name for name in names if name not in modifiers]
        for _ in range(repeat):
            if len(non_modifiers) == 1:
                keyboard = self._get_computer_use().keyboard
                await _send_input(partial(keyboard.press, non_modifiers[0], modifiers, request_timeout=REQUEST_TIMEOUT))
            else:  # only modifiers, or several other keys: `press` takes a single key
                async with self._holding(names):
                    pass
