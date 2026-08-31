"""Tests for the Playwright capability.

The Playwright API surface is mocked throughout: no real Chromium is launched.
An in-memory page double (`_FakePage`) backs the toolset, and a fake Playwright
driver chain backs the `wrap_run` lifecycle, so the suite runs in CI with only
the `playwright` Python package installed (no browser binary).
"""

from __future__ import annotations

import asyncio
import inspect
import socket
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Protocol, get_args

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer
from playwright._impl._errors import TargetClosedError
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import StorageState
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import CapabilityOrdering
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, ToolReturn, ToolReturnPart
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

import pydantic_ai_harness.playwright._toolset as toolset_module
from pydantic_ai_harness.playwright import (
    DEFAULT_ACTION_TIMEOUT_MS,
    DEFAULT_ALLOWLIST_REACH,
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    DEFAULT_RESOLVED_KINDS,
    BrowserEvent,
    BrowserUnavailableError,
    BrowserUnavailableWarning,
    EgressPolicy,
    EgressRequest,
    PlaywrightBrowser,
    PlaywrightBrowserSession,
    PlaywrightBrowserToolset,
    RequestKind,
)

pytestmark = pytest.mark.anyio

_STORAGE_STATE: StorageState = {'cookies': [{'name': 'session', 'value': 'abc', 'domain': 'example.com', 'path': '/'}]}

_HISTORY_RESPONSE = object()


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


# --- Doubles for the Playwright API surface ---------------------------------


class _FakeLocator:
    def __init__(self, page: _FakePage, selector: str) -> None:
        self._page = page
        self._selector = selector

    async def press_sequentially(self, text: str, *, delay: float | None = None, timeout: float | None = None) -> None:
        self._page.timeouts['press_sequentially'] = timeout
        self._page.typed.append((self._selector, text))


class _FakeDialog:
    def __init__(self, *, type: str = 'confirm', message: str = 'Delete this?') -> None:
        self.type = type
        self.message = message
        self.accepted_with: str | None = None
        self.answer: str | None = None

    async def accept(self, prompt_text: str | None = None) -> None:
        self.answer = 'accept'
        self.accepted_with = prompt_text

    async def dismiss(self) -> None:
        self.answer = 'dismiss'


class _FakeUnanswerableDialog(_FakeDialog):
    """A dialog whose page has already gone away."""

    async def dismiss(self) -> None:
        raise PlaywrightError('Target page, context or browser has been closed')


class _FakeMouse:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    async def click(self, x: float, y: float) -> None:
        self.calls.append(('click', int(x), int(y)))

    async def move(self, x: float, y: float) -> None:
        self.calls.append(('move', int(x), int(y)))

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        self.calls.append(('wheel', int(delta_x), int(delta_y)))


class _FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    async def press(self, key: str) -> None:
        self.pressed.append(key)


class _FakeFrameContent:
    """A child frame whose text page-level reads cannot see."""

    def __init__(
        self,
        *,
        url: str = 'https://embed.example.com/',
        text: str = 'Embedded schedule',
        inner_text_error: PlaywrightError | None = None,
        wait_for_error: PlaywrightError | None = None,
        wait_delay: float = 0.0,
    ) -> None:
        self.url = url
        self._text = text
        self._inner_text_error = inner_text_error
        self._wait_for_error = wait_for_error
        self._wait_delay = wait_delay
        self.wait_states: list[str | None] = []

    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
        if self._inner_text_error is not None:
            raise self._inner_text_error
        return self._text

    async def wait_for_selector(
        self, selector: str, *, timeout: float | None = None, state: str | None = None
    ) -> object:
        self.wait_states.append(state)
        if self._wait_delay:
            await asyncio.sleep(self._wait_delay)
        if self._wait_for_error is not None:
            raise self._wait_for_error
        return None


class _HangingFrame(_FakeFrameContent):
    """A frame whose text read never returns, to drive the sweep budget."""

    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
        await asyncio.Event().wait()
        return ''  # pragma: no cover -- unreachable; the wait is cancelled


class _FakeConsoleMessage:
    def __init__(self, type: str, text: str) -> None:
        self.type = type
        self.text = text


class _FakeNetworkRequest:
    def __init__(self, *, url: str, method: str = 'GET', failure: str | None = None) -> None:
        self.url = url
        self.method = method
        self.failure = failure


class _FakeResponse:
    def __init__(self, *, url: str, status: int = 200, method: str = 'GET') -> None:
        self.url = url
        self.status = status
        self.request = _FakeNetworkRequest(url=url, method=method)


class _FakeRoute:
    def __init__(self) -> None:
        self.aborted = False
        self.continued = False

    async def abort(self) -> None:
        self.aborted = True

    async def continue_(self) -> None:
        self.continued = True


class _FakeRequestPage:
    def __init__(self) -> None:
        self.main_frame = _FakeFrame(self)


class _FakeFrame:
    def __init__(self, page: _FakeRequestPage) -> None:
        self.page = page


class _FakeRequest:
    def __init__(
        self,
        url: str,
        *,
        navigation: bool,
        frame: _FakeFrame | None = None,
        frame_error: PlaywrightError | None = None,
        resource_type: str | None = None,
        method: str = 'GET',
    ) -> None:
        self.url = url
        self._navigation = navigation
        self._frame = frame
        self._frame_error = frame_error
        self.resource_type = resource_type if resource_type is not None else ('document' if navigation else 'image')
        self.method = method

    def is_navigation_request(self) -> bool:
        return self._navigation

    @property
    def frame(self) -> _FakeFrame:
        if self._frame_error is not None:
            raise self._frame_error
        assert self._frame is not None
        return self._frame


class _FakeRouteHandler(Protocol):
    def __call__(self, route: _FakeRoute, request: _FakeRequest) -> Awaitable[None]: ...  # pragma: no cover


class _FakeWebSocketRoute:
    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self.connected = False

    async def close(self, *, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True

    def connect_to_server(self) -> _FakeWebSocketRoute:
        self.connected = True
        return self


class _FakeBrowserContext:
    def __init__(
        self,
        page: _FakePage,
        *,
        storage_state: StorageState | None = None,
        service_workers: str | None = None,
        accept_downloads: bool | None = None,
    ) -> None:
        self.page = page
        self.opened: list[_FakePage] = []
        self.storage_state = storage_state
        self.service_workers = service_workers
        self.accept_downloads = accept_downloads
        self.routes: list[str] = []
        self.route_handler: _FakeRouteHandler | None = None
        self.websocket_routes: list[str] = []
        self.websocket_handler: Callable[[_FakeWebSocketRoute], Awaitable[None]] | None = None

    async def new_page(self) -> _FakePage:
        if not self.opened:
            self.opened.append(self.page)
            return self.page
        page = _FakePage(url='about:blank', title='')
        page._context = self
        self.opened.append(page)
        return page

    async def route(self, url: str, handler: _FakeRouteHandler) -> None:
        self.routes.append(url)
        self.route_handler = handler

    async def route_web_socket(self, url: str, handler: Callable[[_FakeWebSocketRoute], Awaitable[None]]) -> None:
        self.websocket_routes.append(url)
        self.websocket_handler = handler

    async def dispatch_websocket(self, url: str) -> _FakeWebSocketRoute:
        assert self.websocket_handler is not None
        websocket = _FakeWebSocketRoute(url)
        await self.websocket_handler(websocket)
        return websocket

    async def dispatch(self, request: _FakeRequest) -> _FakeRoute:
        assert self.route_handler is not None
        route = _FakeRoute()
        await self.route_handler(route, request)
        return route


class _FakePage:
    def __init__(
        self,
        *,
        url: str = 'https://example.com/',
        title: str = 'Example',
        body: str = 'Hello body',
        evaluate_result: object = None,
        evaluate_raises: Exception | None = None,
        selector_raises: bool = False,
        element_text: str | None = None,
        redirect_to: str | None = None,
        screenshot_bytes: bytes = b'PNG-BYTES',
        close_error: Exception | None = None,
        goto_error: PlaywrightError | None = None,
        bounce_error: PlaywrightError | None = None,
        click_error: PlaywrightError | None = None,
        fill_error: PlaywrightError | None = None,
        inner_text_error: PlaywrightError | None = None,
        screenshot_error: PlaywrightError | None = None,
        go_back_error: PlaywrightError | None = None,
        go_forward_error: PlaywrightError | None = None,
        go_back_result: object | None = _HISTORY_RESPONSE,
        go_forward_result: object | None = _HISTORY_RESPONSE,
        wait_for_error: PlaywrightError | None = None,
        hover_error: PlaywrightError | None = None,
        press_error: PlaywrightError | None = None,
        select_option_error: PlaywrightError | None = None,
        aria_snapshot_tree: str = '- heading "Example" [ref=e1]\n- button "Go" [ref=e2]',
        aria_snapshot_error: PlaywrightError | None = None,
        title_error: PlaywrightError | None = None,
    ) -> None:
        self._url = url
        self._title = title
        self._body = body
        self._evaluate_result = evaluate_result
        self._evaluate_raises = evaluate_raises
        self._selector_raises = selector_raises
        self._element_text = element_text
        self._redirect_to = redirect_to
        self._screenshot_bytes = screenshot_bytes
        self._close_error = close_error
        self._goto_error = goto_error
        self._bounce_error = bounce_error
        self._click_error = click_error
        self._fill_error = fill_error
        self._inner_text_error = inner_text_error
        self._screenshot_error = screenshot_error
        self._go_back_error = go_back_error
        self._go_forward_error = go_forward_error
        self._go_back_result = go_back_result
        self._go_forward_result = go_forward_result
        self._wait_for_error = wait_for_error
        self._hover_error = hover_error
        self._press_error = press_error
        self._select_option_error = select_option_error
        self._aria_snapshot_tree = aria_snapshot_tree
        self._aria_snapshot_error = aria_snapshot_error
        self._title_error = title_error
        self._context: _FakeBrowserContext | None = None
        self._popup_on_screenshot: _FakePage | None = None
        self.evaluated: list[str] = []
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.frames: list[_FakePage | _FakeFrameContent] = [self]
        self.timeouts: dict[str, float | None] = {}
        self.goto_calls: list[str] = []
        self.load_states: list[str] = []
        self.filled: list[tuple[str, str]] = []
        self.clicked: list[str] = []
        self.pressed: list[tuple[str, str]] = []
        self.hovered: list[str] = []
        self.selected: list[tuple[str, list[str]]] = []
        self.handlers: dict[str, list[Callable[[object], None]]] = {}
        self.popup_events: list[str] = []
        self.popup_handlers: list[Callable[[_FakePage], None]] = []
        self.typed: list[tuple[str, str]] = []
        self.wait_states: list[str | None] = []
        self.wait_queries: list[str] = []
        self.brought_to_front = 0
        self.closed = False

    @property
    def url(self) -> str:
        return self._url

    @property
    def context(self) -> _FakeBrowserContext:
        assert self._context is not None
        return self._context

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self.goto_calls.append(url)
        self.timeouts['goto'] = timeout
        if self._goto_error is not None and url != 'about:blank':
            raise self._goto_error
        if self._bounce_error is not None and url == 'about:blank':
            raise self._bounce_error
        # A configured redirect lands the page on a different host than requested,
        # modelling a 3xx to a disallowed domain (but never for the bounce itself).
        self._url = self._redirect_to if self._redirect_to is not None and url != 'about:blank' else url

    async def wait_for_load_state(self, state: str, *, timeout: float | None = None) -> None:
        self.load_states.append(state)
        self.timeouts['wait_for_load_state'] = timeout

    async def title(self) -> str:
        if self._title_error is not None:
            raise self._title_error
        return self._title

    def locator(self, selector: str) -> _FakeLocator:
        return _FakeLocator(self, selector)

    async def bring_to_front(self) -> None:
        self.brought_to_front += 1

    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
        self.timeouts['inner_text'] = timeout
        if self._inner_text_error is not None:
            raise self._inner_text_error
        if selector == 'body':
            return self._body
        if self._selector_raises:
            raise RuntimeError('element not found')
        return self._element_text if self._element_text is not None else f'text:{selector}'

    async def click(self, selector: str, *, timeout: float | None = None) -> None:
        self.timeouts['click'] = timeout
        if self._click_error is not None:
            raise self._click_error
        self.clicked.append(selector)

    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None:
        self.timeouts['fill'] = timeout
        if self._fill_error is not None:
            raise self._fill_error
        self.filled.append((selector, value))

    async def screenshot(self, *, full_page: bool = False, timeout: float | None = None) -> bytes:
        self.timeouts['screenshot'] = timeout
        if self._screenshot_error is not None:
            raise self._screenshot_error
        if self._popup_on_screenshot is not None:
            for handler in self.popup_handlers:
                handler(self._popup_on_screenshot)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        return self._screenshot_bytes

    async def evaluate(self, expression: str) -> object:
        self.evaluated.append(expression)
        if self._evaluate_raises is not None:
            raise self._evaluate_raises
        return self._evaluate_result

    async def go_back(self, *, timeout: float | None = None) -> object | None:
        self.timeouts['go_back'] = timeout
        if self._go_back_error is not None:
            raise self._go_back_error
        return self._go_back_result

    async def go_forward(self, *, timeout: float | None = None) -> object | None:
        self.timeouts['go_forward'] = timeout
        if self._go_forward_error is not None:
            raise self._go_forward_error
        return self._go_forward_result

    async def wait_for_selector(
        self, selector: str, *, timeout: float | None = None, state: str | None = None
    ) -> object:
        self.timeouts['wait_for_selector'] = timeout
        self.wait_states.append(state)
        self.wait_queries.append(selector)
        if self._wait_for_error is not None:
            raise self._wait_for_error
        return None

    async def aria_snapshot(self, *, mode: str = 'default', timeout: float | None = None) -> str:
        self.timeouts['aria_snapshot'] = timeout
        if self._aria_snapshot_error is not None:
            raise self._aria_snapshot_error
        return self._aria_snapshot_tree

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error
        self.emit('close', self)

    async def hover(self, selector: str, *, timeout: float | None = None) -> None:
        self.timeouts['hover'] = timeout
        if self._hover_error is not None:
            raise self._hover_error
        self.hovered.append(selector)

    async def press(self, selector: str, key: str, *, timeout: float | None = None) -> None:
        self.timeouts['press'] = timeout
        if self._press_error is not None:
            raise self._press_error
        self.pressed.append((selector, key))

    async def select_option(self, selector: str, value: Sequence[str], *, timeout: float | None = None) -> list[str]:
        self.timeouts['select_option'] = timeout
        if self._select_option_error is not None:
            raise self._select_option_error
        self.selected.append((selector, list(value)))
        return list(value)

    def on(self, event: str, handler: Callable[[object], None]) -> None:
        self.handlers.setdefault(event, []).append(handler)
        if event == 'popup':
            self.popup_events.append(event)
            self.popup_handlers.append(handler)

    def emit(self, event: str, payload: object) -> None:
        """Deliver a page event the way Playwright's receive loop would."""
        for handler in self.handlers.get(event, []):
            handler(payload)


class _ControlledNavigationPage(_FakePage):
    def __init__(self) -> None:
        super().__init__()
        self.first_navigation_started = asyncio.Event()
        self.release_first_navigation = asyncio.Event()

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self.goto_calls.append(url)
        self._url = url
        if url == 'https://example.com/first':
            self.first_navigation_started.set()
            await self.release_first_navigation.wait()


class _HangingMousePage(_FakePage):
    """A page whose mouse never settles, to drive the coordinate-scroll deadline."""

    def __init__(self) -> None:
        super().__init__()
        self.mouse = _HangingMouse()


class _HangingMouse(_FakeMouse):
    async def move(self, x: float, y: float) -> None:
        await asyncio.Event().wait()

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        await asyncio.Event().wait()  # pragma: no cover -- unreachable; move never returns


class _HangingScreenshotPage(_FakePage):
    """A page whose `screenshot` blocks until cancelled, to drive mid-tool teardown."""

    def __init__(self) -> None:
        super().__init__()
        self.screenshot_started = asyncio.Event()

    async def screenshot(self, *, full_page: bool = False, timeout: float | None = None) -> bytes:
        self.screenshot_started.set()
        await asyncio.Event().wait()
        return self._screenshot_bytes  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingTitlePage(_FakePage):
    async def title(self) -> str:
        await asyncio.Event().wait()
        return self._title  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingEvaluatePage(_FakePage):
    """A page whose `evaluate` never returns, modelling a never-resolving promise."""

    async def evaluate(self, expression: str) -> object:
        await asyncio.Event().wait()
        return None  # pragma: no cover -- unreachable; the wait is cancelled


class _FakeInstallerProcess:
    def __init__(self, *, returncode: int, output: bytes, hang: bool = False) -> None:
        self.returncode = returncode
        self.output = output
        self.hang = hang
        self.communicate_started = asyncio.Event()
        self.terminated = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, None]:
        self.communicate_started.set()
        if self.hang:
            await asyncio.Event().wait()
        return self.output, None

    def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


def _install_fake_installer_process(monkeypatch: pytest.MonkeyPatch, process: _FakeInstallerProcess) -> None:
    async def _create_subprocess_exec(*args: str, **kwargs: int) -> _FakeInstallerProcess:
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', _create_subprocess_exec)


class _FakePlaywrightBrowser:
    def __init__(self, page: _FakePage, *, close_error: Exception | None = None) -> None:
        self._page = page
        self._close_error = close_error
        self.closed = False
        self.contexts: list[_FakeBrowserContext] = []

    async def new_context(
        self,
        *,
        storage_state: StorageState | None = None,
        service_workers: str | None = None,
        accept_downloads: bool | None = None,
    ) -> _FakeBrowserContext:
        context = _FakeBrowserContext(
            self._page,
            storage_state=storage_state,
            service_workers=service_workers,
            accept_downloads=accept_downloads,
        )
        self._page._context = context
        self.contexts.append(context)
        return context

    async def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _FakeChromium:
    def __init__(
        self,
        page: _FakePage,
        *,
        executable_missing: bool = False,
        launch_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._page = page
        # An existing path so the real `os.path.exists` pre-check takes the launch
        # branch; a bogus path models a missing Chromium binary.
        self._executable_path = '/nonexistent/chromium-binary' if executable_missing else sys.executable
        self._launch_error = launch_error
        self._close_error = close_error
        self.launched: list[bool] = []
        self.sandboxed: list[bool] = []
        self.connected: list[str] = []
        self.launch_timeouts: list[int] = []
        self.browser: _FakePlaywrightBrowser | None = None
        self.browsers: list[_FakePlaywrightBrowser] = []

    @property
    def executable_path(self) -> str:
        return self._executable_path

    async def launch(
        self, *, headless: bool, chromium_sandbox: bool = False, timeout: int | None = None
    ) -> _FakePlaywrightBrowser:
        self.launched.append(headless)
        self.sandboxed.append(chromium_sandbox)
        self.launch_timeouts.append(timeout if timeout is not None else -1)
        if self._launch_error is not None:
            raise self._launch_error
        self.browser = _FakePlaywrightBrowser(self._page, close_error=self._close_error)
        self.browsers.append(self.browser)
        return self.browser

    async def connect_over_cdp(self, endpoint_url: str, *, timeout: int | None = None) -> _FakePlaywrightBrowser:
        self.connected.append(endpoint_url)
        self.launch_timeouts.append(timeout if timeout is not None else -1)
        if self._launch_error is not None:
            raise self._launch_error
        self.browser = _FakePlaywrightBrowser(self._page, close_error=self._close_error)
        self.browsers.append(self.browser)
        return self.browser


class _FakeDriver:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium


class _FakeDriverCM:
    def __init__(self, driver: _FakeDriver) -> None:
        self.driver = driver
        self._driver = driver
        self.entered = False
        self.entries = 0
        self.exited = False

    async def __aenter__(self) -> _FakeDriver:
        self.entered = True
        self.entries += 1
        return self._driver

    async def __aexit__(self, *exc: object) -> bool:
        self.exited = True
        return False


class _ScriptedSession(PlaywrightBrowserSession):
    """A session whose launch is scripted, to drive the locking without a real Chromium."""

    def __init__(self, on_launch: Callable[[_ScriptedSession], Awaitable[None]]) -> None:
        super().__init__()
        self._on_launch = on_launch
        self.launches = 0

    async def _launch(self) -> None:
        self.launches += 1
        await self._on_launch(self)


def _allows(url: str, allowed_domains: list[str] | None, **policy_options: object) -> bool:
    """Whether the egress policy admits `url` as a top-level navigation."""
    policy = toolset_module.EgressPolicy(allowed_domains=allowed_domains, **policy_options)  # pyright: ignore[reportArgumentType]
    return policy.refuse(toolset_module.EgressRequest(url=url, kind='navigation')) is None


def _toolset(
    page: _FakePage,
    *,
    allowed_domains: list[str] | None = None,
    block_private_addresses: bool = True,
    screenshot_on_navigate: bool = False,
    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
    action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    session: PlaywrightBrowserSession | None = None,
) -> PlaywrightBrowserToolset[None]:
    """Build a toolset whose active page is the given double."""
    session = (
        session
        if session is not None
        else PlaywrightBrowserSession(
            policy=toolset_module.EgressPolicy(
                allowed_domains=allowed_domains, block_private_addresses=block_private_addresses
            )
        )
    )
    session.page = page
    if not session.pages:
        session.pages = [page]
    return PlaywrightBrowserToolset[None](
        session=session,
        screenshot_on_navigate=screenshot_on_navigate,
        max_content_tokens=max_content_tokens,
        action_timeout_ms=action_timeout_ms,
        navigation_timeout_ms=navigation_timeout_ms,
    )


def _tool_results(result: AgentRunResult[object]) -> str:
    """Every tool result of a finished run, joined, for asserting what the model was told."""
    return '\n'.join(
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )


def _ctx() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


def _install_fake_driver(
    monkeypatch: pytest.MonkeyPatch,
    page: _FakePage,
    *,
    executable_missing: bool = False,
    launch_error: Exception | None = None,
    close_error: Exception | None = None,
) -> _FakeDriverCM:
    """Point the capability's `async_playwright` at a fake driver chain."""
    chromium = _FakeChromium(
        page, executable_missing=executable_missing, launch_error=launch_error, close_error=close_error
    )
    cm = _FakeDriverCM(_FakeDriver(chromium))
    monkeypatch.setattr(toolset_module, 'async_playwright', lambda: cm)
    return cm


# --- Tool behavior ----------------------------------------------------------


class TestPlaywrightBrowserTools:
    def test_check_allowed_domain_rejects_malformed_url(self) -> None:
        assert _allows('https://[::1', None) is False
        assert _allows('https://[::1', ['::1']) is False

    def test_hostless_navigation_is_rejected_under_open_egress(self) -> None:
        assert _allows('mailto:a@b.com', None) is False

    def test_about_blank_stays_navigable(self) -> None:
        # Where a context starts and where a refused navigation is bounced to, so
        # denying it would refuse every tool call made before the first navigate.
        assert _allows('about:blank', None) is True
        assert _allows('about:blank', ['example.com']) is True

    def test_check_allowed_domain_rejects_trailing_dot_host_for_blank_entries(self) -> None:
        url = 'https://169.254.169.254./'
        assert _allows(url, ['']) is False
        assert _allows(url, [' \t ']) is False

    def test_check_allowed_domain_rejects_userinfo_host_spoof(self) -> None:
        # CVE-2025-47241 class: the real host is `evil.com`; `allowed.com` is only
        # the userinfo, so `.hostname` resolves it correctly and the match fails.
        assert _allows('https://allowed.com:pass@evil.com/', ['allowed.com']) is False

    def test_check_allowed_domain_rejects_backslash_before_url_parsing(self) -> None:
        assert _allows(r'https://evil.com\@example.com/', ['example.com']) is False

    @pytest.mark.parametrize('host', ['evil-example.com', 'example.com.attacker.com'])
    def test_check_allowed_domain_rejects_sibling_domain_tricks(self, host: str) -> None:
        assert _allows(f'https://{host}/', ['example.com']) is False

    def test_check_allowed_domain_matches_idn_against_punycode_allowlist(self) -> None:
        # A Unicode host and its `xn--` spelling get the same verdict against an
        # ASCII allowlist entry, in both directions.
        assert _allows('https://пример.рф/path', ['xn--e1afmkfd.xn--p1ai']) is True
        assert _allows('https://xn--e1afmkfd.xn--p1ai/path', ['пример.рф']) is True

    def test_check_allowed_domain_falls_back_when_entry_not_idna_encodable(self) -> None:
        # An over-long label cannot be IDNA-encoded; the entry falls back to its
        # lowercased form rather than crashing, and simply does not match.
        assert _allows('https://example.com/', ['a' * 64]) is False

    async def test_navigate_returns_url_title_and_text(self) -> None:
        toolset = _toolset(_FakePage(title='Docs', body='Page text here'))
        result = await toolset.navigate('https://example.com/')
        assert result == 'URL: https://example.com/\nTitle: Docs\n\nPage text here'

    async def test_navigate_rejects_domain_outside_allowlist(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://evil.com/')
        assert result == 'Error: domain not in allowed_domains: https://evil.com/'
        assert page.goto_calls == []

    async def test_navigate_allows_exact_subdomain_and_port(self) -> None:
        for url in ('https://example.com/', 'https://docs.example.com/', 'https://example.com:8443/x'):
            page = _FakePage()
            toolset = _toolset(page, allowed_domains=['example.com'])
            result = await toolset.navigate(url)
            assert isinstance(result, str) and result.startswith('URL:')
            assert page.goto_calls == [url]

    async def test_navigate_allows_ipv6_host_in_allowlist(self) -> None:
        # A bracketed IPv6 literal must match its allowlist entry (regression: the
        # old netloc.split(':') turned '[::1]' into '['). Loopback needs the
        # private-address opt-out on top of the allowlist entry.
        page = _FakePage(url='http://[::1]:8080/')
        toolset = _toolset(page, allowed_domains=['::1'], block_private_addresses=False)
        result = await toolset.navigate('http://[::1]:8080/')
        assert isinstance(result, str) and result.startswith('URL:')
        assert page.goto_calls == ['http://[::1]:8080/']

    @pytest.mark.parametrize('url', ['mailto:a@b.com', 'file:///etc/passwd', 'data:text/html,<h1>x'])
    async def test_navigate_rejects_url_without_host(self, url: str) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate(url)
        # The phrase does not name the allowlist: this toolset has none.
        assert result == f'Error: URL with no host: {url}'
        assert page.goto_calls == []

    async def test_fresh_page_is_usable_before_the_first_navigation(self) -> None:
        # A context opens on about:blank, which is also where a disallowed navigation
        # is sent. Denying it refused every tool call made before the first navigate.
        page = _FakePage(url='about:blank')
        toolset = _toolset(page)
        assert await toolset.click('button#go') == "Clicked 'button#go'. URL: about:blank\n\nHello body"
        assert 'no host' not in await toolset.execute_js('1 + 1')

    def test_idna_matching_follows_chromium_not_the_stdlib_codec(self) -> None:
        # The stdlib 'idna' codec maps the deviation characters the IDNA-2003 way
        # ('faß.de' -> 'fass.de'), where Chromium connects to 'xn--fa-hia.de'. An
        # allowlist that disagrees with the browser blocks the host it means to allow.
        assert _allows('https://faß.de/', ['faß.de']) is True
        assert _allows('https://xn--fa-hia.de/', ['faß.de']) is True
        assert _allows('https://faß.de/', ['fass.de']) is False

    def test_undecodable_host_falls_back_to_the_original(self) -> None:
        # An over-long label cannot be encoded; the comparison still has to return a
        # verdict rather than raise, and an unencodable host matches nothing.
        assert _allows(f'https://{"a" * 64}.example.com/', ['example.com']) is True
        assert _allows('https://example.com/', [f'{"a" * 64}.com']) is False

    async def test_navigate_allows_trailing_dot_spelling_of_an_allowlisted_host(self) -> None:
        # The trailing dot names the DNS root, so the absolute spelling is the same host.
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=['example.com'])
        assert isinstance(await toolset.navigate('https://example.com./'), str)
        assert isinstance(await toolset.navigate('https://sub.example.com./'), str)
        assert page.goto_calls == ['https://example.com./', 'https://sub.example.com./']

    async def test_navigate_rejects_trailing_dot_host_for_blank_allowlist_entry(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=[' \t '])
        url = 'https://evil.example./'
        result = await toolset.navigate(url)
        assert result == f'Error: domain not in allowed_domains: {url}'
        assert page.goto_calls == []

    async def test_navigate_rejects_malformed_url(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate('https://[::1')
        assert result == 'Error: URL with no host: https://[::1'
        assert page.goto_calls == []

    async def test_navigate_rejects_backslash_url_without_opening_page(self) -> None:
        page = _FakePage()
        url = r'https://evil.com\@example.com/'
        result = await _toolset(page, allowed_domains=['example.com']).navigate(url)
        assert result == f'Error: URL with no host: {url}'
        assert page.goto_calls == []

    async def test_navigate_attaches_screenshot_when_configured(self) -> None:
        toolset = _toolset(_FakePage(), screenshot_on_navigate=True)
        result = await toolset.navigate('https://example.com/')
        assert isinstance(result, ToolReturn)
        assert isinstance(result.return_value, str) and result.return_value.startswith('URL:')
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES' and image.media_type == 'image/png'

    async def test_navigate_truncates_long_page_text(self) -> None:
        toolset = _toolset(_FakePage(body='X' * 40), max_content_tokens=1)
        result = await toolset.navigate('https://example.com/')
        assert result == 'URL:'

    async def test_navigate_truncates_url_and_title_within_shared_budget(self) -> None:
        url = f'https://example.com/{"u" * 40}'
        toolset = _toolset(_FakePage(title='T' * 40), max_content_tokens=2)
        result = await toolset.navigate(url)
        assert result == 'URL: htt'

    async def test_navigate_bounces_on_redirect_to_disallowed_host(self) -> None:
        page = _FakePage(redirect_to='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://example.com/start')
        assert result == 'Error: navigate reached a domain not in allowed_domains: https://evil.com/landing'
        assert page.goto_calls == ['https://example.com/start', 'about:blank']

    async def test_navigate_empty_allowlist_blocks_every_host(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=[])
        result = await toolset.navigate('https://example.com/')
        assert result == 'Error: domain not in allowed_domains: https://example.com/'
        assert page.goto_calls == []

    async def test_navigate_reports_landing_url_after_allowed_redirect(self) -> None:
        page = _FakePage(redirect_to='https://docs.example.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.navigate('https://example.com/start')
        assert isinstance(result, str)
        assert result.startswith('URL: https://docs.example.com/landing')
        assert page.goto_calls == ['https://example.com/start']

    async def test_navigate_waits_for_domcontentloaded(self) -> None:
        page = _FakePage()
        await _toolset(page).navigate('https://example.com/')
        assert page.load_states == ['domcontentloaded']

    async def test_concurrent_navigations_return_their_own_page_state(self) -> None:
        page = _ControlledNavigationPage()
        toolset = _toolset(page)
        first = asyncio.create_task(toolset.navigate('https://example.com/first'))
        await page.first_navigation_started.wait()

        second = asyncio.create_task(toolset.navigate('https://example.com/second'))
        await asyncio.sleep(0)
        assert page.goto_calls == ['https://example.com/first']

        page.release_first_navigation.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert isinstance(first_result, str)
        assert first_result.startswith('URL: https://example.com/first')
        assert isinstance(second_result, str)
        assert second_result.startswith('URL: https://example.com/second')

    async def test_click_css_selector(self) -> None:
        page = _FakePage(body='after click')
        toolset = _toolset(page)
        result = await toolset.click('button#go')
        assert page.clicked == ['button#go']
        assert result == "Clicked 'button#go'. URL: https://example.com/\n\nafter click"

    async def test_click_pixel_coordinates(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.click('450,300')
        assert page.mouse.calls == [('click', 450, 300)]

    async def test_click_negative_pixel_coordinates(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.click('-10,-20')
        assert page.mouse.calls == [('click', -10, -20)]

    async def test_click_malformed_coordinates_are_treated_as_css(self) -> None:
        page = _FakePage()
        await _toolset(page).click('--1,2')
        assert page.clicked == ['--1,2']
        assert page.mouse.calls == []

    async def test_click_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.click('a.external')
        assert result == 'Error: click reached a domain not in allowed_domains: https://evil.com/landing'
        assert page.goto_calls == ['about:blank']

    async def test_a_blocked_redirect_names_the_refused_url_not_the_error_page(self) -> None:
        session = PlaywrightBrowserSession(
            policy=toolset_module.EgressPolicy(allowed_domains=['example.com'], block_private_addresses=False)
        )

        class _RefusedRedirectPage(_FakePage):
            """A click whose redirect the route guard refused: Chromium keeps its error page."""

            async def click(self, selector: str, *, timeout: float | None = None) -> None:
                await super().click(selector, timeout=timeout)
                session.record(
                    toolset_module.BrowserEvent(
                        kind='request_blocked',
                        level='warning',
                        message='domain not in allowed_domains',
                        url='https://join.example.org/invite',
                    )
                )
                # The refusal is rarely the last thing a page reports before it settles.
                session.record(
                    toolset_module.BrowserEvent(
                        kind='console', level='error', message='error: navigation failed', url=None
                    )
                )

        page = _RefusedRedirectPage(url='chrome-error://chromewebdata/')
        result = await _toolset(page, session=session).click('a.external')
        assert result == (
            'Error: click loaded no page (domain not in allowed_domains: https://join.example.org/invite); '
            'the browser is back at about:blank.'
        )
        assert page.goto_calls == ['about:blank']

    async def test_an_earlier_refusal_is_not_reported_as_this_failures_cause(self) -> None:
        page = _FakePage(url='chrome-error://chromewebdata/')
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        session.record(
            toolset_module.BrowserEvent(
                kind='request_blocked', level='warning', message='domain not in allowed_domains', url='https://old/'
            )
        )
        result = await _toolset(page, session=session).click('a.external')
        assert (
            result
            == 'Error: click loaded no page (the navigation did not complete); the browser is back at about:blank.'
        )

    async def test_type_text_fills_field(self) -> None:
        page = _FakePage(body='typed')
        toolset = _toolset(page)
        result = await toolset.type_text('input#q', 'hello')
        assert page.filled == [('input#q', 'hello')]
        assert result == "Typed into 'input#q'.\n\ntyped"

    async def test_screenshot_returns_binary_content(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/p'))
        result = await toolset.screenshot()
        assert isinstance(result, ToolReturn)
        assert result.return_value == 'Screenshot captured. URL: https://example.com/p'
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES'

    async def test_screenshot_full_page(self) -> None:
        toolset = _toolset(_FakePage())
        result = await toolset.screenshot(full_page=True)
        assert isinstance(result, ToolReturn)

    async def test_screenshot_bounds_text_without_dropping_image(self) -> None:
        page = _FakePage(url=f'https://example.com/{"u" * 40}')
        result = await _toolset(page, max_content_tokens=1).screenshot()
        assert isinstance(result, ToolReturn)
        assert result.return_value == 'Scre'
        assert result.content is not None
        image = result.content[0]
        assert isinstance(image, BinaryContent)
        assert image.data == b'PNG-BYTES'

    async def test_screenshot_over_size_limit_returns_error_not_image(self) -> None:
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        result = await _toolset(_FakePage(screenshot_bytes=png)).screenshot(full_page=True)
        assert isinstance(result, str)
        assert result.startswith(f'Error: screenshot is {len(png)} bytes')
        assert 'full_page=False' in result

    async def test_navigate_omits_oversized_screenshot_attachment(self) -> None:
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        toolset = _toolset(_FakePage(screenshot_bytes=png), screenshot_on_navigate=True)
        result = await toolset.navigate('https://example.com/')
        assert isinstance(result, str)  # no ToolReturn: the image is dropped, the text result survives
        assert result.startswith('URL: https://example.com/')
        assert 'Error: screenshot is' in result

    async def test_navigate_keeps_the_oversized_note_when_page_text_fills_the_budget(self) -> None:
        # The page result is already budgeted, so the note has to be given room
        # rather than appended: re-truncating a full result would drop it and
        # leave ordinary page text with no sign the screenshot went missing.
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        page = _FakePage(body='b' * 5_000, screenshot_bytes=png)
        toolset = _toolset(page, screenshot_on_navigate=True, max_content_tokens=100)
        result = await toolset.navigate('https://example.com/')
        assert isinstance(result, str)
        assert result.startswith('URL: https://example.com/')
        assert 'Error: screenshot is' in result
        assert len(result) <= 100 * 4

    async def test_oversized_note_takes_the_whole_budget_when_it_cannot_fit(self) -> None:
        png = b'x' * (toolset_module._MAX_SCREENSHOT_BYTES + 1)
        toolset = _toolset(_FakePage(screenshot_bytes=png), screenshot_on_navigate=True, max_content_tokens=5)
        result = await toolset.navigate('https://example.com/')
        assert result == 'Error: screenshot is'

    async def test_get_text_with_selector(self) -> None:
        toolset = _toolset(_FakePage())
        assert await toolset.get_text('h1') == 'text:h1'

    async def test_get_text_selector_error(self) -> None:
        toolset = _toolset(_FakePage(selector_raises=True))
        result = await toolset.get_text('#missing')
        assert result.startswith("Error getting text from '#missing':")

    async def test_get_text_surfaces_playwright_error(self) -> None:
        toolset = _toolset(_FakePage(inner_text_error=PlaywrightTimeoutError('inner_text timed out')))
        result = await toolset.get_text('h1')
        assert result == "Error getting text from 'h1': inner_text timed out"

    async def test_get_text_full_page_surfaces_playwright_error(self) -> None:
        toolset = _toolset(_FakePage(inner_text_error=PlaywrightTimeoutError('body timed out')))
        result = await toolset.get_text()
        assert result == (
            'Error: get_text timed out after 5000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again. '
            'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
        )

    async def test_get_text_full_page(self) -> None:
        toolset = _toolset(_FakePage(body='full page text'))
        assert await toolset.get_text() == 'full page text'

    async def test_get_text_selector_truncated(self) -> None:
        toolset = _toolset(_FakePage(element_text='Y' * 40), max_content_tokens=1)
        result = await toolset.get_text('article')
        assert result == 'Y' * 4

    async def test_scroll_window(self) -> None:
        page = _FakePage(body='scrolled', evaluate_result='0|648|2000')
        toolset = _toolset(page)
        result = await toolset.scroll('down')
        assert result == 'Scrolled down. 648 of 2000 px down.\n\nscrolled'
        assert page.mouse.calls == []

    async def test_scroll_localized(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        await toolset.scroll('up', 5, 6)
        assert page.mouse.calls == [('move', 5, 6), ('wheel', 0, -300)]

    async def test_scroll_invalid_direction(self) -> None:
        toolset = _toolset(_FakePage())
        result = await toolset.scroll('sideways')
        assert result == "Error: invalid direction 'sideways'; use up/down/left/right/top/bottom"

    async def test_scroll_moves_about_one_screenful(self) -> None:
        page = _FakePage()
        await _toolset(page).scroll('down')
        assert 'window.scrollBy(0, window.innerHeight * 0.9)' in page.evaluated[0]

    async def test_scroll_bottom_jumps_to_the_end_of_the_page(self) -> None:
        page = _FakePage(body='loaded more', evaluate_result='0|1532|1532')
        result = await _toolset(page).scroll('bottom')
        assert 'window.scrollTo(0, document.body.scrollHeight)' in page.evaluated[0]
        assert result == 'Scrolled bottom. At the bottom of the page.\n\nloaded more'

    async def test_scroll_bottom_ignores_coordinates(self) -> None:
        # `top`/`bottom` name a place in the page, which a wheel event over one
        # element cannot express, so they stay page-level whatever is passed.
        page = _FakePage(evaluate_result='0|0|0')
        await _toolset(page).scroll('bottom', 5, 6)
        assert page.mouse.calls == []
        assert 'window.scrollTo(0, document.body.scrollHeight)' in page.evaluated[0]

    async def test_scroll_reports_a_page_that_cannot_scroll(self) -> None:
        page = _FakePage(body='short page', evaluate_result='0|0|0')
        result = await _toolset(page).scroll('bottom')
        assert result == 'Scrolled bottom. The page has nothing to scroll.\n\nshort page'

    async def test_scroll_reports_reaching_the_top(self) -> None:
        page = _FakePage(body='top of page', evaluate_result='500|0|2000')
        result = await _toolset(page).scroll('top')
        assert result == 'Scrolled top. At the top of the page.\n\ntop of page'

    async def test_scroll_reports_a_position_that_did_not_move(self) -> None:
        # A page whose own handler intercepts the wheel: the model needs to see that
        # repeating the call will not reveal anything new.
        page = _FakePage(body='same view', evaluate_result='300|300|2000')
        result = await _toolset(page).scroll('down')
        assert result == 'Scrolled down. Position unchanged, 300 of 2000 px down.\n\nsame view'

    async def test_go_back(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/prev', body='prev'))
        result = await toolset.go_back()
        assert result == 'Went back. URL: https://example.com/prev\n\nprev'

    async def test_go_back_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.go_back()
        assert result == 'Error: go_back reached a domain not in allowed_domains: https://evil.com/'

    async def test_go_back_reports_empty_history(self) -> None:
        result = await _toolset(_FakePage(go_back_result=None)).go_back()
        assert result == 'No previous page in browser history.'

    async def test_go_forward(self) -> None:
        toolset = _toolset(_FakePage(url='https://example.com/next', body='next'))
        result = await toolset.go_forward()
        assert result == 'Went forward. URL: https://example.com/next\n\nnext'

    async def test_go_forward_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.go_forward()
        assert result == 'Error: go_forward reached a domain not in allowed_domains: https://evil.com/'

    async def test_go_forward_reports_empty_history(self) -> None:
        result = await _toolset(_FakePage(go_forward_result=None)).go_forward()
        assert result == 'No next page in browser history.'

    async def test_execute_js_string_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result='the title'))
        assert await toolset.execute_js('document.title') == 'the title'

    async def test_execute_js_json_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result={'a': 1}))
        assert await toolset.execute_js('({a:1})') == '{"a": 1}'

    @pytest.mark.parametrize(
        ('value', 'prefix'),
        [
            ('S' * 40, 'SSSS'),
            ({'value': 'J' * 40}, '{"va'),
        ],
    )
    async def test_execute_js_truncates_string_and_serialized_results(self, value: object, prefix: str) -> None:
        toolset = _toolset(_FakePage(evaluate_result=value), max_content_tokens=1)
        assert await toolset.execute_js('largeResult') == prefix

    def test_truncation_marker_fits_inside_budget(self) -> None:
        result = toolset_module._truncate('X' * 200, 80)
        assert len(result) == 80
        assert result.endswith('[... tool output truncated at 80 characters]')
        assert toolset_module._truncate('content', 0) == ''

    async def test_execute_js_null_result(self) -> None:
        toolset = _toolset(_FakePage(evaluate_result=None))
        assert await toolset.execute_js('void 0') == 'undefined'

    async def test_execute_js_error(self) -> None:
        toolset = _toolset(_FakePage(evaluate_raises=ValueError('boom')))
        assert await toolset.execute_js('bad(') == 'JS error: boom'

    async def test_execute_js_closed_target_is_a_browser_error_not_a_script_error(self) -> None:
        # `evaluate` raises for script exceptions too, so the closed target is picked
        # out by type; 'JS error: ...' would hide the signal that the browser is gone.
        toolset = _toolset(_FakePage(evaluate_raises=TargetClosedError()))
        assert await toolset.execute_js('document.title') == (
            'Error: execute_js failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )

    async def test_execute_js_bounces_off_disallowed_domain(self) -> None:
        page = _FakePage(url='https://evil.com/', evaluate_result='x')
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.execute_js('location.href="https://evil.com"')
        assert result == 'Error: execute_js reached a domain not in allowed_domains: https://evil.com/'

    async def test_execute_js_bounce_failure_returns_bounded_error(self) -> None:
        # The bounce is a navigation, so its failure is mapped like any other browser
        # error instead of escaping the tool and aborting the run.
        page = _FakePage(url='https://evil.com/', evaluate_result='x', bounce_error=TargetClosedError())
        toolset = _toolset(page, allowed_domains=['example.com'])
        result = await toolset.execute_js('location.href="https://evil.com"')
        assert result == (
            'Error: execute_js failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )


_BLOCKED_ADDRESS_HOSTS = [
    '127.0.0.1',
    '10.0.0.5',
    '172.16.0.1',
    '192.168.1.1',
    '169.254.169.254',
    '0.0.0.0',
    '100.64.0.1',
    '224.0.0.1',
    '240.0.0.1',
    '::1',
    'fe80::1',
    'fd00::1',
    '::ffff:127.0.0.1',
    'localhost',
    'LOCALHOST',
    'app.localhost',
    'localhost.',
    '169.254.169.254.',
]

_PUBLIC_ADDRESS_HOSTS = ['example.com', '8.8.8.8', '2606:4700:4700::1111', 'localhost.example.com', 'my-localhost.dev']


class TestPrivateAddressBlocking:
    @pytest.mark.parametrize('host', _BLOCKED_ADDRESS_HOSTS)
    def test_is_blocked_address_detects_reserved_ranges(self, host: str) -> None:
        assert toolset_module.is_blocked_address(host) is True

    @pytest.mark.parametrize('host', _PUBLIC_ADDRESS_HOSTS)
    def test_is_blocked_address_allows_public_hosts(self, host: str) -> None:
        assert toolset_module.is_blocked_address(host) is False

    def test_refuse_names_the_denying_rule(self) -> None:
        def reason(url: str, allowed: list[str] | None, block: bool) -> str | None:
            policy = toolset_module.EgressPolicy(allowed_domains=allowed, block_private_addresses=block)
            return policy.refuse(toolset_module.EgressRequest(url=url, kind='navigation'))

        assert reason('https://example.com/', None, True) is None
        assert reason('http://127.0.0.1/', None, True) == 'blocked private or link-local address'
        assert reason('http://127.0.0.1/', None, False) is None
        # Deny beats allow whichever way the two rules point.
        assert reason('http://127.0.0.1/', ['example.com'], True) == 'blocked private or link-local address'
        assert reason('http://127.0.0.1/', ['127.0.0.1'], True) == 'blocked private or link-local address'
        assert reason('https://other.com/', ['example.com'], True) == 'domain not in allowed_domains'
        # about:blank is the context's own starting state, not a destination to deny.
        assert reason('about:blank', None, True) is None
        assert reason('file:///etc/passwd', None, True) == 'URL with no host'

    async def test_navigate_blocks_private_address_under_open_egress(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)
        result = await toolset.navigate('http://169.254.169.254/latest/meta-data/')
        assert result == 'Error: blocked private or link-local address: http://169.254.169.254/latest/meta-data/'
        assert page.goto_calls == []

    async def test_navigate_opt_out_reaches_private_address(self) -> None:
        page = _FakePage(url='http://127.0.0.1:8000/')
        toolset = _toolset(page, block_private_addresses=False)
        result = await toolset.navigate('http://127.0.0.1:8000/')
        assert isinstance(result, str) and result.startswith('URL:')
        assert page.goto_calls == ['http://127.0.0.1:8000/']

    async def test_navigate_blocks_allowlisted_private_address_without_opt_out(self) -> None:
        page = _FakePage()
        toolset = _toolset(page, allowed_domains=['127.0.0.1'])
        result = await toolset.navigate('http://127.0.0.1:8000/admin')
        assert result == 'Error: blocked private or link-local address: http://127.0.0.1:8000/admin'
        assert page.goto_calls == []

    async def test_navigate_bounces_on_redirect_to_private_address(self) -> None:
        page = _FakePage(redirect_to='http://169.254.169.254/')
        toolset = _toolset(page)
        result = await toolset.navigate('https://example.com/start')
        assert result == 'Error: navigate reached a blocked private or link-local address: http://169.254.169.254/'
        assert page.goto_calls == ['https://example.com/start', 'about:blank']

    async def test_click_bounces_off_private_address(self) -> None:
        page = _FakePage(url='http://127.0.0.1:8000/admin')
        toolset = _toolset(page)
        result = await toolset.click('a.local')
        assert result == 'Error: click reached a blocked private or link-local address: http://127.0.0.1:8000/admin'
        assert page.goto_calls == ['about:blank']

    async def test_execute_js_bounces_off_private_address(self) -> None:
        page = _FakePage(url='http://169.254.169.254/', evaluate_result='x')
        toolset = _toolset(page)
        result = await toolset.execute_js('location.href="http://169.254.169.254/"')
        assert result == 'Error: execute_js reached a blocked private or link-local address: http://169.254.169.254/'


class TestNameResolutionBlocking:
    """A hostname is not an IP literal, so where it points is only knowable by resolving it."""

    @staticmethod
    def _pointing_at(monkeypatch: pytest.MonkeyPatch, address: str) -> list[str]:
        """Make every name resolve to `address`, recording which names were looked up."""
        looked_up: list[str] = []

        async def resolve(host: str) -> tuple[str, ...]:
            looked_up.append(host)
            return (address,)

        monkeypatch.setattr(toolset_module, '_getaddrinfo', resolve)
        return looked_up

    async def test_navigate_refuses_a_name_that_resolves_private(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The literal check passes this host: wildcard DNS services make the spelling
        # ordinary while the address behind it is the metadata endpoint.
        self._pointing_at(monkeypatch, '169.254.169.254')
        page = _FakePage()
        result = await _toolset(page).navigate('http://169-254-169-254.nip.example/')
        assert result == 'Error: blocked private or link-local address: http://169-254-169-254.nip.example/'
        assert page.goto_calls == []

    async def test_a_name_that_resolves_public_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._pointing_at(monkeypatch, '93.184.216.34')
        result = await _toolset(_FakePage()).navigate('https://example.com/')
        assert isinstance(result, str) and not result.startswith('Error:')

    async def test_opting_out_skips_the_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No lookup means no verdict from one either, so an opted-out session reaches
        # a host whose name would not have resolved.
        looked_up = self._pointing_at(monkeypatch, '169.254.169.254')
        result = await _toolset(_FakePage(), block_private_addresses=False).navigate('http://internal.example/')
        assert isinstance(result, str) and not result.startswith('Error:')
        assert looked_up == []

    async def test_an_address_is_not_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The literal already answers the question; a lookup would be wasted.
        looked_up = self._pointing_at(monkeypatch, '93.184.216.34')
        await _toolset(_FakePage()).navigate('https://93.184.216.34/')
        assert looked_up == []

    async def _guarded_page(self, monkeypatch: pytest.MonkeyPatch) -> _FakePage:
        """Run one agent turn so the session installs its route guard on the context."""
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser[object]()])
        await agent.run('use the browser')
        return page

    async def test_the_route_guard_refuses_a_private_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._pointing_at(monkeypatch, '127.0.0.1')
        page = await self._guarded_page(monkeypatch)
        host_page = _FakeRequestPage()
        route = await page.context.dispatch(
            _FakeRequest('http://data.example/api', navigation=False, resource_type='fetch', frame=host_page.main_frame)
        )
        assert route.aborted is True

    async def test_a_passive_subresource_is_resolved_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The literal check already covers every kind, so a spelling must not decide
        # the verdict: an image pointed at an address and one pointed at a name
        # answering with it are the same request.
        looked_up = self._pointing_at(monkeypatch, '127.0.0.1')
        page = await self._guarded_page(monkeypatch)
        host_page = _FakeRequestPage()
        route = await page.context.dispatch(
            _FakeRequest(
                'http://cdn.example/logo.png', navigation=False, resource_type='image', frame=host_page.main_frame
            )
        )
        assert route.aborted is True
        assert looked_up == ['cdn.example']

    async def test_a_public_subresource_still_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._pointing_at(monkeypatch, '93.184.216.34')
        page = await self._guarded_page(monkeypatch)
        host_page = _FakeRequestPage()
        route = await page.context.dispatch(
            _FakeRequest(
                'http://cdn.example/logo.png', navigation=False, resource_type='image', frame=host_page.main_frame
            )
        )
        assert route.continued is True

    async def test_a_subframe_document_is_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A child frame's text is appended to the page text, so a frame pointed at
        # an internal address is a way back to the model.
        self._pointing_at(monkeypatch, '169.254.169.254')
        page = await self._guarded_page(monkeypatch)
        subframe = _FakeFrame(_FakeRequestPage())  # not `host_page.main_frame`
        route = await page.context.dispatch(_FakeRequest('http://embed.example/', navigation=True, frame=subframe))
        assert route.aborted is True

    async def test_a_failed_lookup_is_a_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Whoever controls the name controls whether the lookup answers, so an
        # unanswered one cannot mean "allowed": it would be the way past the block.
        async def unresolvable(host: str) -> tuple[str, ...]:
            raise socket.gaierror('Name or service not known')

        monkeypatch.setattr(toolset_module, '_getaddrinfo', unresolvable)
        page = _FakePage()
        result = await _toolset(page).navigate('https://nope.invalid/')
        assert result == (
            'Error: host that did not resolve, so the private-address block could not clear it: https://nope.invalid/'
        )
        assert page.goto_calls == []

    async def test_a_host_the_resolver_cannot_encode_is_a_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `getaddrinfo` encodes the name with the stdlib idna codec, which raises a
        # `UnicodeError` for an empty or over-long label. The lookup happens before
        # the operation exists, so an escaping exception would end the run.
        async def unencodable(host: str) -> tuple[str, ...]:
            raise UnicodeError('label empty or too long')

        monkeypatch.setattr(toolset_module, '_getaddrinfo', unencodable)
        page = _FakePage()
        result = await _toolset(page).navigate('http://a..com/')
        assert result == (
            'Error: host that did not resolve, so the private-address block could not clear it: http://a..com/'
        )
        assert page.goto_calls == []


class TestResolutionIsBounded:
    """The lookup runs before an operation's deadline exists, so it carries its own."""

    async def test_a_stalled_resolver_does_not_hold_the_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(toolset_module, '_RESOLUTION_TIMEOUT_SECONDS', 0.01)

        async def never_answers(host: str) -> tuple[str, ...]:
            await asyncio.Event().wait()
            raise AssertionError('unreachable')  # pragma: no cover

        monkeypatch.setattr(toolset_module, '_getaddrinfo', never_answers)
        result = await _toolset(_FakePage()).navigate('https://example.com/')
        # Bounded, and refused rather than allowed: an attacker who controls the name
        # controls how long its lookup takes, so a stall must not be a way through.
        assert result == (
            'Error: host that did not resolve, so the private-address block could not clear it: https://example.com/'
        )


class TestResolutionCaching:
    """The lookup is a duplicate of one Chromium already makes, so it is cached briefly."""

    async def test_a_repeated_host_is_looked_up_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        looked_up = TestNameResolutionBlocking._pointing_at(monkeypatch, '93.184.216.34')
        toolset = _toolset(_FakePage())
        await toolset.navigate('https://example.com/one')
        await toolset.navigate('https://example.com/two')
        assert looked_up == ['example.com']

    async def test_a_full_cache_is_emptied_rather_than_grown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(toolset_module, '_RESOLUTION_CACHE_MAX', 1)
        looked_up = TestNameResolutionBlocking._pointing_at(monkeypatch, '93.184.216.34')
        toolset = _toolset(_FakePage())
        await toolset.navigate('https://one.example/')
        await toolset.navigate('https://two.example/')
        # The first entry was evicted with the rest, so its host is looked up again.
        await toolset.navigate('https://one.example/')
        assert looked_up == ['one.example', 'two.example', 'one.example']


class TestPlaywrightErrorHandling:
    async def test_navigate_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(goto_error=PlaywrightTimeoutError('Timeout 60000ms exceeded.'))
        result = await _toolset(page).navigate('https://example.com/')
        # A navigation failure reports the navigation budget, not the shorter action one.
        assert result == (
            'Error: navigate timed out after 60000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again. '
            'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
        )

    async def test_navigate_preserves_net_error_code(self) -> None:
        page = _FakePage(goto_error=PlaywrightError('page.goto: net::ERR_NAME_NOT_RESOLVED at https://nope.invalid/'))
        result = await _toolset(page).navigate('https://nope.invalid/')
        assert isinstance(result, str)
        assert 'net::ERR_NAME_NOT_RESOLVED' in result

    async def test_click_preserves_strict_mode_match_count(self) -> None:
        page = _FakePage(click_error=PlaywrightError('strict mode violation: locator resolved to 3 elements'))
        result = await _toolset(page).click('button')
        assert result == 'Error: click failed: strict mode violation: locator resolved to 3 elements'

    async def test_click_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(click_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).click('button#missing')
        assert result.startswith('Error: click timed out after 5000ms.')

    async def test_type_text_timeout_returns_bounded_error(self) -> None:
        page = _FakePage(fill_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).type_text('input#missing', 'hi')
        assert result.startswith('Error: type_text timed out after 5000ms.')

    async def test_scroll_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(evaluate_raises=PlaywrightError('scroll blew up'))
        result = await _toolset(page).scroll('down')
        assert result == 'Error: scroll failed: scroll blew up'

    async def test_screenshot_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(screenshot_error=PlaywrightError('screenshot failed'))
        result = await _toolset(page).screenshot()
        assert result == 'Error: screenshot failed: screenshot failed'

    async def test_go_back_target_closed_reports_crash(self) -> None:
        page = _FakePage(go_back_error=TargetClosedError())
        result = await _toolset(page).go_back()
        assert result == (
            'Error: go_back failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )

    async def test_go_forward_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(go_forward_error=PlaywrightError('history unavailable'))
        result = await _toolset(page).go_forward()
        assert result == 'Error: go_forward failed: history unavailable'


_BAD_DEADLINE_AND_BAD_ARGUMENT_CALLS: list[Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]] = [
    lambda t: t.navigate('mailto:a@b.com', timeout_ms=-1),
    lambda t: t.scroll('sideways', timeout_ms=-1),
    lambda t: t.wait_for(timeout_ms=-1),
]

_NON_POSITIVE_TIMEOUT_CALLS: list[Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]] = [
    lambda t: t.navigate('https://example.com/', timeout_ms=-1),
    lambda t: t.click('button', timeout_ms=-1),
    lambda t: t.type_text('input', 'x', timeout_ms=-1),
    lambda t: t.get_text('h1', timeout_ms=-1),
    lambda t: t.screenshot(timeout_ms=-1),
    lambda t: t.go_back(timeout_ms=-1),
    lambda t: t.go_forward(timeout_ms=-1),
    lambda t: t.wait_for(selector='.x', timeout_ms=-1),
    lambda t: t.snapshot(timeout_ms=-1),
    lambda t: t.scroll('down', timeout_ms=-1),
    lambda t: t.execute_js('1 + 1', timeout_ms=-1),
]


class TestPerCallTimeout:
    async def test_navigate_override_reaches_every_playwright_operation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Every stage is handed what is *left* of the budget, counted from a real clock, so a run slow
        # enough to cross a millisecond leaves the trailing stage on 1110. Freeze the clock: the claim
        # here is that the override reaches every stage rather than any stage falling back to the
        # capability default, and `test_override_reaches_each_playwright_call` is where the
        # countdown itself is asserted.
        monkeypatch.setattr('pydantic_ai_harness.playwright._toolset.monotonic', lambda: 0.0)
        page = _FakePage()
        result = await _toolset(page, screenshot_on_navigate=True).navigate('https://example.com/', timeout_ms=1111)
        assert isinstance(result, ToolReturn)
        assert page.timeouts == {
            'goto': 1111,
            'wait_for_load_state': 1111,
            'inner_text': 1111,
            'screenshot': 1111,
        }

    async def test_zero_capability_default_disables_title_deadline(self) -> None:
        result = await _toolset(_FakePage(), action_timeout_ms=0, navigation_timeout_ms=0).navigate(
            'https://example.com/'
        )
        assert isinstance(result, str)
        assert result.startswith('URL:')

    async def test_navigate_title_honors_timeout_override(self) -> None:
        # The reported deadline is the override the call actually ran under, not
        # the capability default.
        result = await _toolset(_HangingTitlePage()).navigate('https://example.com/', timeout_ms=1)
        assert result == (
            'Error: navigate timed out after 1ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again. '
            'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
        )

    async def test_execute_js_bounds_unresolved_promise(self) -> None:
        # `page.evaluate` waits on returned promises indefinitely; the external
        # deadline turns that into a bounded error instead of a held lock.
        result = await _toolset(_HangingEvaluatePage()).execute_js('new Promise(() => {})', timeout_ms=1)
        assert isinstance(result, str)
        assert result.startswith('Error: execute_js timed out after 1ms.')

    async def test_scroll_bounds_hung_evaluate(self) -> None:
        result = await _toolset(_HangingEvaluatePage()).scroll('down', timeout_ms=1)
        assert result.startswith('Error: scroll timed out after 1ms.')

    async def test_override_reaches_each_playwright_call(self) -> None:
        page = _FakePage()
        toolset = _toolset(page)

        def under(stage: str, budget: int) -> None:
            """Assert `stage` ran under `budget`, allowing for what earlier stages spent."""
            spent = page.timeouts[stage]
            # The first Playwright call of an operation gets the whole override and
            # each later one gets what is left of it, so a trailing stage lands just
            # under. The slack is far smaller than the gap between two budgets here,
            # so a value left behind by the previous operation still fails.
            assert spent is not None and budget - 500 < spent <= budget, f'{stage}={spent}, budget={budget}'

        await toolset.navigate('https://example.com/', timeout_ms=1111)
        assert page.timeouts['goto'] == 1111
        # Trailing operations (load wait, page-text read) run under the same
        # override, not the capability default.
        await toolset.click('button#go', timeout_ms=2222)
        assert page.timeouts['click'] == 2222
        under('wait_for_load_state', 2222)
        under('inner_text', 2222)
        await toolset.type_text('input#q', 'hi', timeout_ms=3333)
        assert page.timeouts['fill'] == 3333
        under('inner_text', 3333)
        await toolset.get_text('h1', timeout_ms=4444)
        assert page.timeouts['inner_text'] == 4444
        await toolset.get_text(timeout_ms=4545)
        assert page.timeouts['inner_text'] == 4545
        await toolset.screenshot(timeout_ms=5555)
        assert page.timeouts['screenshot'] == 5555
        await toolset.go_back(timeout_ms=6666)
        assert page.timeouts['go_back'] == 6666
        under('wait_for_load_state', 6666)
        under('inner_text', 6666)
        await toolset.go_forward(timeout_ms=7777)
        assert page.timeouts['go_forward'] == 7777
        under('wait_for_load_state', 7777)
        await toolset.wait_for(selector='.ready', timeout_ms=8888)
        assert page.timeouts['wait_for_selector'] == 8888
        under('inner_text', 8888)
        await toolset.snapshot(timeout_ms=9999)
        assert page.timeouts['aria_snapshot'] == 9999
        await toolset.scroll('down', timeout_ms=1234)
        under('inner_text', 1234)

    async def test_override_bounds_the_disallowed_navigation_bounce(self) -> None:
        # The bounce to about:blank is itself a navigation, so it must run under the
        # caller's deadline rather than Playwright's 30-second default.
        page = _FakePage(url='https://evil.com/landing')
        toolset = _toolset(page, allowed_domains=['example.com'])
        await toolset.click('a.external', timeout_ms=4321)
        assert page.goto_calls == ['about:blank']
        assert page.timeouts['goto'] == 4321

    async def test_none_falls_back_to_capability_default(self) -> None:
        page = _FakePage()
        await _toolset(page).click('button#go')
        assert page.timeouts['click'] == DEFAULT_ACTION_TIMEOUT_MS

    @pytest.mark.parametrize('call', _NON_POSITIVE_TIMEOUT_CALLS)
    async def test_non_positive_override_returns_bounded_error(
        self, call: Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]
    ) -> None:
        result = await call(_toolset(_FakePage()))
        assert result == 'Error: timeout_ms must be greater than 0.'

    @pytest.mark.parametrize('call', _BAD_DEADLINE_AND_BAD_ARGUMENT_CALLS, ids=['navigate', 'scroll', 'wait_for'])
    async def test_bad_deadline_outranks_a_bad_argument(
        self, call: Callable[[PlaywrightBrowserToolset[None]], Awaitable[object]]
    ) -> None:
        # These three refuse their argument before acquiring a page, but the deadline
        # is still checked first, so the reported error does not depend on the order
        # the two problems happen to be noticed in.
        assert await call(_toolset(_FakePage())) == 'Error: timeout_ms must be greater than 0.'

    async def test_zero_override_cannot_disable_the_deadline(self) -> None:
        # 0 means "no deadline" to Playwright, so the model-chosen override refuses it
        # even though the developer-set capability default may use it.
        result = await _toolset(_HangingEvaluatePage()).execute_js('new Promise(() => {})', timeout_ms=0)
        assert result == 'Error: timeout_ms must be greater than 0.'


class TestWaitFor:
    async def test_wait_for_selector_returns_page_text(self) -> None:
        page = _FakePage(body='loaded content')
        result = await _toolset(page).wait_for(selector='.ready')
        assert result == "Found '.ready'.\n\nloaded content"

    async def test_wait_for_text_uses_text_engine(self) -> None:
        page = _FakePage(body='dynamic text')
        result = await _toolset(page).wait_for(text='Submit')
        assert result == "Found 'Submit'.\n\ndynamic text"
        assert page.wait_queries == [':text("Submit")']

    async def test_wait_for_text_is_inert_selector_input(self) -> None:
        # Interpolated after `text=`, a `>>` reads as a selector chain and a quote
        # flips exact matching, so the wait would miss text that is on the page.
        page = _FakePage(body='dynamic text')
        result = await _toolset(page).wait_for(text='Home >> "Products" \\ more')
        assert result.startswith('Found \'Home >> "Products" \\ more\'.')
        assert page.wait_queries == [':text("Home >> \\"Products\\" \\\\ more")']

    async def test_wait_for_requires_exactly_one_argument(self) -> None:
        toolset = _toolset(_FakePage())
        expected = 'Error: wait_for requires exactly one of selector or text.'
        assert await toolset.wait_for() == expected
        assert await toolset.wait_for(selector='.x', text='y') == expected

    async def test_wait_for_page_text_error_returns_mapped_error(self) -> None:
        # The page can close between a successful `wait_for_selector` and the
        # body read; that must stay a tool result, not abort the run.
        page = _FakePage(inner_text_error=TargetClosedError('Target page closed'))
        result = await _toolset(page).wait_for(selector='.ready')
        assert result == (
            'Error: wait_for failed: the browser or page was closed unexpectedly. '
            'Browser tools may be unavailable for the rest of this run.'
        )

    async def test_wait_for_timeout_returns_mapped_error(self) -> None:
        page = _FakePage(wait_for_error=PlaywrightTimeoutError('Timeout 30000ms exceeded.'))
        result = await _toolset(page).wait_for(selector='.never')
        assert result == (
            'Error: wait_for timed out after 5000ms. The element may not exist or the page may be slow; '
            'try a different selector, or navigate again. '
            'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
        )


class TestSnapshot:
    async def test_snapshot_returns_aria_tree(self) -> None:
        result = await _toolset(_FakePage()).snapshot()
        assert result == '- heading "Example" [ref=e1]\n- button "Go" [ref=e2]'

    async def test_snapshot_truncated_to_budget(self) -> None:
        page = _FakePage(aria_snapshot_tree='Z' * 40)
        result = await _toolset(page, max_content_tokens=1).snapshot()
        assert result == 'Z' * 4

    async def test_snapshot_playwright_error_returns_error_string(self) -> None:
        page = _FakePage(aria_snapshot_error=PlaywrightError('snapshot failed'))
        result = await _toolset(page).snapshot()
        assert result == 'Error: snapshot failed: snapshot failed'

    async def test_click_aria_ref_uses_selector_path_not_coordinates(self) -> None:
        page = _FakePage()
        await _toolset(page).click('aria-ref=e2')
        assert page.clicked == ['aria-ref=e2']
        assert page.mouse.calls == []


# --- State / ensure_page ----------------------------------------------------


class TestPlaywrightBrowserSession:
    def test_toolset_validates_max_content_tokens(self) -> None:
        session = PlaywrightBrowserSession()
        with pytest.raises(ValueError, match='^max_content_tokens must be greater than or equal to 0$'):
            PlaywrightBrowserToolset[None](session=session, max_content_tokens=-1)
        PlaywrightBrowserToolset[None](session=session, max_content_tokens=0)

    def test_toolset_validates_timeouts(self) -> None:
        session = PlaywrightBrowserSession()
        with pytest.raises(ValueError, match='^action_timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowserToolset[None](session=session, action_timeout_ms=-1)
        with pytest.raises(ValueError, match='^navigation_timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowserToolset[None](session=session, navigation_timeout_ms=-1)
        # 0 = no deadline, accepted as a developer-set default
        PlaywrightBrowserToolset[None](session=session, action_timeout_ms=0, navigation_timeout_ms=0)

    async def test_tool_raises_when_wrap_run_not_active(self) -> None:
        toolset = PlaywrightBrowserToolset[None](session=PlaywrightBrowserSession())
        with pytest.raises(RuntimeError, match='PlaywrightBrowser is not running'):
            await toolset.screenshot()

    async def test_tool_reports_a_launch_error_instead_of_raising(self) -> None:
        session = PlaywrightBrowserSession()
        session.launch_error = 'Chromium is not installed.'
        toolset = PlaywrightBrowserToolset[None](session=session)
        assert await toolset.screenshot() == 'Chromium is not installed.'
        # Forgotten once reported, so a call after the agent fixes it tries again.
        assert session.launch_error is None

    async def test_concurrent_ensure_page_launches_once(self) -> None:
        # Two tool calls that race before the page exists must launch Chromium once.
        async def _launch(session: _ScriptedSession) -> None:
            await asyncio.sleep(0)  # yield so the second caller blocks on the lock
            session.page = _FakePage()

        async with _ScriptedSession(_launch) as session:
            first, second = await asyncio.gather(session.ensure_page(), session.ensure_page())
        assert session.launches == 1
        assert first is second

    async def test_combined_operation_and_launch_lock_launches_once(self) -> None:
        # Two first tool calls contend both the toolset operation lock and the
        # lazy-launch lock. The operation lock is outer and the launch lock inner,
        # so the launch runs once and both calls observe the same page state.
        async def _launch(session: _ScriptedSession) -> None:
            await asyncio.sleep(0)
            session.page = _FakePage(body='shared body')

        async with _ScriptedSession(_launch) as session:
            toolset = PlaywrightBrowserToolset[None](session=session)
            first, second = await asyncio.gather(toolset.get_text(), toolset.get_text())
        assert session.launches == 1
        assert first == second == 'shared body'

    async def test_concurrent_ensure_page_failed_launch_raises_once(self) -> None:
        async def _launch(session: _ScriptedSession) -> None:
            await asyncio.sleep(0)
            session.launch_error = 'Chromium is not installed.'

        async with _ScriptedSession(_launch) as session:
            results = await asyncio.gather(session.ensure_page(), session.ensure_page(), return_exceptions=True)
        assert session.launches == 1  # second caller sees the error, does not relaunch
        assert all(isinstance(r, RuntimeError) for r in results)


# --- Capability hooks -------------------------------------------------------


class TestPlaywrightBrowserHooks:
    def test_capability_validates_max_content_tokens(self) -> None:
        with pytest.raises(ValueError, match='^max_content_tokens must be greater than or equal to 0$'):
            PlaywrightBrowser[None](max_content_tokens=-1)
        PlaywrightBrowser[None](max_content_tokens=0)

    def test_capability_validates_timeouts(self) -> None:
        with pytest.raises(ValueError, match='^action_timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowser[None](action_timeout_ms=-1)
        with pytest.raises(ValueError, match='^navigation_timeout_ms must be greater than or equal to 0$'):
            PlaywrightBrowser[None](navigation_timeout_ms=-1)
        PlaywrightBrowser[None](action_timeout_ms=0, navigation_timeout_ms=0)

    def test_get_instructions_reports_allowlist(self) -> None:
        instructions = PlaywrightBrowser[None](allowed_domains=['a.com', 'b.com']).get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: a.com, b.com' in text

    def test_get_instructions_reports_all_domains(self) -> None:
        instructions = PlaywrightBrowser[None]().get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: all' in text

    def test_get_instructions_reports_empty_allowlist_as_none(self) -> None:
        # An empty allowlist blocks every domain, so the model must be told 'none',
        # not 'all' (which list-truthiness would collapse it to).
        instructions = PlaywrightBrowser[None](allowed_domains=[]).get_instructions()
        text = instructions(_ctx())
        assert text is not None and 'Allowed domains: none' in text

    def test_get_instructions_notes_private_address_block(self) -> None:
        text = PlaywrightBrowser[None]().get_instructions()(_ctx())
        assert text is not None
        assert 'Allowed domains: all (private/internal addresses blocked)' in text

    def test_get_instructions_omits_private_note_when_opted_out(self) -> None:
        text = PlaywrightBrowser[None](block_private_addresses=False).get_instructions()(_ctx())
        assert text is not None
        assert 'private/internal addresses blocked' not in text
        assert 'Allowed domains: all' in text

    async def test_prepare_tools_preserves_upstream_unapproved_kind(self) -> None:
        browser = PlaywrightBrowser[None]()
        defs = [
            ToolDefinition(
                name='navigate', parameters_json_schema={'type': 'object'}, kind='unapproved', toolset_id='playwright'
            ),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='misc'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        by_name = {td.name: td for td in result}
        assert by_name['navigate'].kind == 'unapproved'
        assert by_name['other'].kind == 'function'

    async def test_the_browser_tools_stay_offered_when_no_browser_started(self) -> None:
        # The model is told what is missing and can install it, which it cannot do
        # if the tools disappear the moment a launch fails.
        browser = PlaywrightBrowser[None]()
        browser._session.launch_error = 'boom'
        defs = [
            ToolDefinition(
                name='navigate', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='playwright'
            ),
            ToolDefinition(name='other', parameters_json_schema={'type': 'object'}, kind='function', toolset_id='misc'),
        ]
        result = await browser.prepare_tools(_ctx(), defs)
        assert [td.name for td in result] == ['navigate', 'other']

    async def test_prepare_tools_ignores_same_named_foreign_tool(self) -> None:
        # A `navigate` from another toolset must be left untouched (not re-approved,
        # not hidden), since matching is by `toolset_id`, not tool name.
        foreign = ToolDefinition(
            name='navigate', parameters_json_schema={'type': 'object'}, kind='unapproved', toolset_id='other_toolset'
        )
        browser = PlaywrightBrowser[None]()
        assert (await browser.prepare_tools(_ctx(), [foreign]))[0].kind == 'unapproved'

    async def test_for_run_isolates_state(self) -> None:
        browser = PlaywrightBrowser[None]()
        first = await browser.for_run(_ctx())
        second = await browser.for_run(_ctx())
        assert first._session is not second._session
        assert first._session is not browser._session

    def test_from_spec_round_trips_fields(self) -> None:
        browser = PlaywrightBrowser[None].from_spec(
            headless=False,
            allowed_domains=['x.com'],
            block_private_addresses=False,
            screenshot_on_navigate=True,
            max_content_tokens=100,
            action_timeout_ms=1500,
            navigation_timeout_ms=5000,
            auto_install_chromium=True,
            cdp_url='http://localhost:9222',
        )
        assert browser.headless is False
        assert browser.allowed_domains == ['x.com']
        assert browser.block_private_addresses is False
        assert browser.screenshot_on_navigate is True
        assert browser.max_content_tokens == 100
        assert browser.action_timeout_ms == 1500
        assert browser.navigation_timeout_ms == 5000
        assert browser.auto_install_chromium is True
        assert browser.cdp_url == 'http://localhost:9222'

    def test_from_spec_defaults_to_open_egress(self) -> None:
        browser = PlaywrightBrowser[None].from_spec()
        assert browser.allowed_domains is None
        assert browser.block_private_addresses is True
        assert browser.auto_install_chromium is False
        assert browser.cdp_url is None
        assert browser.storage_state is None

    def test_repr_omits_credential_bearing_fields(self) -> None:
        # `repr` reaches diagnostics and logs. Session cookies must not ride along,
        # and a managed-browser endpoint can carry an auth token in the URL.
        rendered = repr(
            PlaywrightBrowser[None](
                storage_state=_STORAGE_STATE,
                cdp_url='https://cloud.example.com/?token=s3cret',
                allowed_domains=['example.com'],
            )
        )
        assert 'abc' not in rendered
        assert 'storage_state' not in rendered
        assert 's3cret' not in rendered
        assert 'cdp_url' not in rendered
        assert "allowed_domains=['example.com']" in rendered  # non-secret configuration still shows

    def test_from_spec_refuses_storage_state(self) -> None:
        # Session credentials stay out of a spec: naming it fails loudly rather
        # than moving cookies into whatever stores the spec.
        with pytest.raises(TypeError, match='storage_state'):
            PlaywrightBrowser[None].from_spec(storage_state=_STORAGE_STATE)  # pyright: ignore[reportCallIssue]


class TestDurabilityRejection:
    def test_accepts_innermost_non_durability_capability(self) -> None:
        # `innermost` is not a durability marker: `InputGuard` declares it too,
        # so ordering alone would reject the supported guard-plus-browser pairing.
        class _Innermost(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='innermost')

        Agent(TestModel(), capabilities=[PlaywrightBrowser(), _Innermost()])

    def test_accepts_capability_with_non_durability_ordering(self) -> None:
        class _Outermost(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

        Agent(TestModel(), capabilities=[PlaywrightBrowser(), _Outermost()])

    def test_rejects_temporal_durability_at_construction(self) -> None:
        pytest.importorskip('temporalio')
        from pydantic_ai.durable_exec.temporal import TemporalDurability  # noqa: PLC0415  # needs the temporal extra

        with pytest.raises(UserError, match='does not support durable execution'):
            Agent(TestModel(), capabilities=[PlaywrightBrowser(), TemporalDurability()])


class TestChromiumAutoInstall:
    @pytest.mark.parametrize(
        ('returncode', 'output', 'expected'),
        [
            (0, b'', None),
            (1, b'download failed', 'download failed'),
        ],
    )
    async def test_returns_installer_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        returncode: int,
        output: bytes,
        expected: str | None,
    ) -> None:
        process = _FakeInstallerProcess(returncode=returncode, output=output)
        _install_fake_installer_process(monkeypatch, process)
        assert await toolset_module._auto_install_chromium() == expected
        assert process.terminated is False
        assert process.waited is False

    async def test_cancellation_terminates_and_waits_for_installer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        process = _FakeInstallerProcess(returncode=-15, output=b'', hang=True)
        _install_fake_installer_process(monkeypatch, process)
        task = asyncio.create_task(toolset_module._auto_install_chromium())
        await process.communicate_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.terminated is True
        assert process.waited is True


# --- Lifecycle through Agent + wrap_run -------------------------------------


class TestPlaywrightBrowserLifecycle:
    async def test_lazy_launch_and_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('use the browser')
        chromium = cm._driver.chromium
        assert chromium.launched == [True]
        assert page.popup_events == ['popup']
        assert page.context.routes == ['**/*']  # the default private-address block installs the route guard
        assert chromium.browser is not None and chromium.browser.closed is True
        assert cm.exited is True

    async def test_a_failed_setup_closes_its_browser_before_the_next_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The connect succeeded and the context build raised, so the next tool call
        # starts a second Chromium. Teardown holds only the latest handle, so the
        # first one is closed on the way into the retry or it outlives the session.
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        chromium = cm._driver.chromium
        working_context = _FakePlaywrightBrowser.new_context

        async def failing_context(
            self: _FakePlaywrightBrowser,
            *,
            storage_state: StorageState | None = None,
            service_workers: str | None = None,
            accept_downloads: bool | None = None,
        ) -> _FakeBrowserContext:
            raise PlaywrightError('context creation failed')

        session = PlaywrightBrowserSession()
        async with session:
            monkeypatch.setattr(_FakePlaywrightBrowser, 'new_context', failing_context)
            with pytest.raises(PlaywrightError):
                await session.ensure_page()
            monkeypatch.setattr(_FakePlaywrightBrowser, 'new_context', working_context)
            await session.ensure_page()
            assert len(chromium.browsers) == 2
            assert chromium.browsers[0].closed is True
            assert chromium.browsers[1].closed is False
        assert chromium.browsers[1].closed is True

    async def test_storage_state_reaches_new_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(storage_state=_STORAGE_STATE)],
        )
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [_STORAGE_STATE]

    async def test_cdp_url_attaches_instead_of_launching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222')],
        )
        await agent.run('screenshot the page')
        chromium = cm._driver.chromium
        assert chromium.connected == ['http://localhost:9222']
        assert chromium.launched == []
        assert chromium.browser is not None and chromium.browser.closed is True

    async def test_cdp_url_skips_the_missing_binary_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Attaching needs no local Chromium, so an absent binary must not hide the
        # tools or trigger the install hint.
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, executable_missing=True)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222', auto_install_chromium=True)],
        )
        await agent.run('screenshot the page')
        chromium = cm._driver.chromium
        assert chromium.connected == ['http://localhost:9222']
        assert chromium.launched == []  # no local binary consulted, so no install hint and no download

    async def test_cdp_url_context_still_applies_storage_state_and_guards(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://localhost:9222', storage_state=_STORAGE_STATE)],
        )
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [_STORAGE_STATE]
        assert [c.service_workers for c in browser.contexts] == ['block']
        assert page.context.routes == ['**/*']

    async def test_storage_state_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        assert [c.storage_state for c in browser.contexts] == [None]

    async def test_context_blocks_service_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        # Service-worker traffic bypasses context routes, so workers are blocked
        # to keep the route guard authoritative for all requests.
        assert [c.service_workers for c in browser.contexts] == ['block']

    async def test_context_refuses_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        browser = cm._driver.chromium.browser
        assert browser is not None
        # No tool exposes downloads, so accepting them would only let a page write
        # to the host's temporary storage.
        assert [c.accept_downloads for c in browser.contexts] == [False]

    async def test_no_route_guard_when_open_egress_and_private_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(block_private_addresses=False)],
        )
        await agent.run('screenshot the page')
        assert page.context.routes == []

    async def test_route_guard_blocks_private_navigation_under_open_egress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        request_page = _FakeRequestPage()
        blocked = await page.context.dispatch(
            _FakeRequest('http://169.254.169.254/latest/', navigation=True, frame=request_page.main_frame)
        )
        assert blocked.aborted is True
        assert blocked.continued is False
        allowed = await page.context.dispatch(
            _FakeRequest('https://example.com/', navigation=True, frame=request_page.main_frame)
        )
        assert allowed.aborted is False
        assert allowed.continued is True

    async def test_every_page_event_the_session_acts_on_is_subscribed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        # The dialog listener is what takes Playwright out of its auto-dismiss mode,
        # and the close listener is what keeps the tab list honest when a page closes
        # itself.
        assert set(page.handlers) == {'popup', 'console', 'pageerror', 'response', 'requestfailed', 'dialog', 'close'}

    async def test_cancellation_mid_tool_call_tears_down_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _HangingScreenshotPage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        run = asyncio.create_task(agent.run('screenshot the page'))
        await page.screenshot_started.wait()
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        chromium = cm._driver.chromium
        assert chromium.browser is not None and chromium.browser.closed is True
        assert cm.exited is True

    async def test_allowlist_registers_route_guard_on_browser_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')
        assert page.context.routes == ['**/*']

    async def test_context_route_guard_applies_to_each_pages_top_level_navigation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')

        popup_page = _FakeRequestPage()
        blocked = await page.context.dispatch(
            _FakeRequest('https://evil.com/popup', navigation=True, frame=popup_page.main_frame)
        )
        assert blocked.aborted is True
        assert blocked.continued is False

        allowed = await page.context.dispatch(
            _FakeRequest('https://example.com/popup', navigation=True, frame=popup_page.main_frame)
        )
        assert allowed.aborted is False
        assert allowed.continued is True

        non_navigation = await page.context.dispatch(
            _FakeRequest(
                'https://evil.com/script.js',
                navigation=False,
                frame_error=PlaywrightError('frame must not be accessed'),
            )
        )
        assert non_navigation.aborted is False
        assert non_navigation.continued is True

        popup_before_frame_creation = await page.context.dispatch(
            _FakeRequest(
                'https://evil.com/first-popup-request',
                navigation=True,
                frame_error=PlaywrightError('frame is not available yet'),
            )
        )
        assert popup_before_frame_creation.aborted is True
        assert popup_before_frame_creation.continued is False

    async def test_route_guard_blocks_private_addresses_in_subframes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `snapshot()` reads the ARIA tree of cross-origin child frames (verified
        # against real Chromium), so a private-IP subframe would hand the model
        # the response body the block exists to withhold.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(allowed_domains=['example.com'])]
        )
        await agent.run('screenshot the page')

        host_page = _FakeRequestPage()
        subframe = _FakeFrame(host_page)  # not `host_page.main_frame`

        metadata = await page.context.dispatch(
            _FakeRequest('http://169.254.169.254/latest/meta-data/', navigation=True, frame=subframe)
        )
        assert metadata.aborted is True
        assert metadata.continued is False

        # The allowlist stays top-level: a page's own third-party frames (identity
        # providers, payment steps) still load.
        third_party = await page.context.dispatch(
            _FakeRequest('https://evil.com/embed', navigation=True, frame=subframe)
        )
        assert third_party.aborted is False
        assert third_party.continued is True

    async def test_route_guard_blocks_private_addresses_for_subresources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `execute_js` can run `fetch('http://169.254.169.254/...')` and return the
        # body, so the private-address block has to reach requests that are not
        # navigations. The allowlist deliberately does not: aborting subresources
        # would strip an allowed page of its own assets.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')

        metadata = await page.context.dispatch(
            _FakeRequest(
                'http://169.254.169.254/latest/meta-data/iam/security-credentials/',
                navigation=False,
                frame_error=PlaywrightError('frame must not be accessed'),
            )
        )
        assert metadata.aborted is True
        assert metadata.continued is False

    async def test_route_guard_subframe_block_honors_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(allowed_domains=['example.com'], block_private_addresses=False)],
        )
        await agent.run('screenshot the page')

        host_page = _FakeRequestPage()
        local = await page.context.dispatch(
            _FakeRequest('http://127.0.0.1:8080/panel', navigation=True, frame=_FakeFrame(host_page))
        )
        assert local.aborted is False
        assert local.continued is True

    async def test_popup_is_kept_as_a_tab_without_taking_over_the_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        popup = _FakePage(url='https://example.com/popup')
        page = _FakePage()
        page._popup_on_screenshot = popup
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            await _toolset(page, session=session).screenshot()
            assert popup.closed is False
            assert session.pages == [page, popup]
            # The tab the operation started on is still the active one, and it never moved.
            assert session.page is page
            assert page.goto_calls == []

    async def test_popup_past_the_tab_limit_is_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        refused = _FakePage(url='https://example.com/popup')
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            session.pages.extend(_FakePage() for _ in range(toolset_module._MAX_TABS - 1))
            page.emit('popup', refused)
            await asyncio.gather(*session._event_tasks, return_exceptions=True)
            assert refused.closed is True
            assert len(session.pages) == toolset_module._MAX_TABS
            assert [event.kind for event in session.events] == ['popup_closed']

    async def test_cancelled_event_task_is_discarded(self) -> None:
        browser = PlaywrightBrowser[None]()
        task = asyncio.create_task(asyncio.sleep(1))
        browser._session._event_tasks.add(task)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        browser._session._event_task_done(task)
        assert browser._session._event_tasks == set()

    async def test_run_without_browser_tool_skips_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=[]), capabilities=[PlaywrightBrowser()])
        await agent.run('do nothing with the browser')
        assert cm.entered is False
        assert cm.exited is False

    async def test_missing_binary_returns_the_install_hint_and_the_run_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, executable_missing=True)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        with pytest.warns(toolset_module.BrowserUnavailableWarning, match='playwright install chromium'):
            result = await agent.run('screenshot the page')
        assert cm._driver.chromium.launched == []  # never attempted launch on a missing binary
        assert cm.exited is True  # Playwright driver still cleaned up
        assert 'playwright install chromium' in _tool_results(result)

    async def test_launch_is_bounded_by_the_configured_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Starting or attaching to a browser is the one step no per-call timeout_ms
        # reaches, so it takes the capability's deadline rather than none at all.
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(navigation_timeout_ms=4200)]
        )
        await agent.run('screenshot the page')
        assert cm.driver.chromium.launch_timeouts == [4200]

    async def test_cdp_attach_is_bounded_by_the_configured_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://127.0.0.1:9222', navigation_timeout_ms=4300)],
        )
        await agent.run('screenshot the page')
        assert cm.driver.chromium.launch_timeouts == [4300]

    async def test_browser_error_while_attaching_returns_a_bounded_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dead cdp_url endpoint is a browser failure like any other: the model
        # gets a result it can act on instead of the run being aborted under it.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page, launch_error=PlaywrightError('connect ECONNREFUSED 127.0.0.1:9222'))
        agent = Agent(
            TestModel(call_tools=['screenshot']),
            capabilities=[PlaywrightBrowser(cdp_url='http://127.0.0.1:9222')],
        )
        result = await agent.run('screenshot the page')
        assert 'Error: screenshot failed: connect ECONNREFUSED' in str(result.all_messages())

    async def test_attach_failure_does_not_report_the_endpoint_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Managed-browser providers put a token in the cdp_url, and Playwright quotes
        # the endpoint it tried. That message now reaches the model, so it must not
        # carry the token; the host and port stay, since they name what was unreachable.
        page = _FakePage()
        endpoint = 'http://127.0.0.1:59999/?token=SUPERSECRET'
        _install_fake_driver(
            monkeypatch, page, launch_error=PlaywrightError(f'connect ECONNREFUSED, retrieving url from {endpoint}')
        )
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(cdp_url=endpoint)])
        result = await agent.run('screenshot the page')
        rendered = str(result.all_messages())
        assert 'SUPERSECRET' not in rendered
        assert 'token=' not in rendered
        assert 'http://127.0.0.1:59999' in rendered

    async def test_attach_failure_redacts_an_endpoint_without_a_scheme_or_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        page = _FakePage()
        endpoint = 'browserless.example.com/session/SUPERSECRET'
        _install_fake_driver(monkeypatch, page, launch_error=PlaywrightError(f'cannot reach {endpoint}'))
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(cdp_url=endpoint)])
        result = await agent.run('screenshot the page')
        assert 'SUPERSECRET' not in str(result.all_messages())

    async def test_attach_failure_redacts_an_endpoint_with_an_invalid_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A non-numeric port makes urlparse raise when the port is read, which would
        # otherwise replace the redacted error with an unhandled one.
        page = _FakePage()
        endpoint = 'http://host:bad/session/SUPERSECRET'
        _install_fake_driver(monkeypatch, page, launch_error=PlaywrightError(f'cannot reach {endpoint}'))
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(cdp_url=endpoint)])
        result = await agent.run('screenshot the page')
        rendered = str(result.all_messages())
        assert 'SUPERSECRET' not in rendered
        assert '<cdp_url>' in rendered

    async def test_launch_failure_with_binary_present_surfaces_own_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        installs: list[bool] = []

        async def _spy_install() -> str | None:  # pragma: no cover -- asserted never called
            installs.append(True)
            return None

        monkeypatch.setattr(toolset_module, '_auto_install_chromium', _spy_install)
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, launch_error=RuntimeError('sandbox denied'))
        # auto_install_chromium=True proves a real launch failure does not trigger a download.
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(auto_install_chromium=True)]
        )
        with pytest.raises(RuntimeError, match='sandbox denied'):
            await agent.run('screenshot the page')
        assert installs == []  # binary present -> no install attempt
        assert cm.exited is True  # driver still cleaned up

    async def test_auto_install_retry_when_install_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page, executable_missing=True)

        async def _fake_install() -> str | None:
            return 'download error: HTTP 403 forbidden'

        monkeypatch.setattr(toolset_module, '_auto_install_chromium', _fake_install)
        agent = Agent(
            TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser(auto_install_chromium=True)]
        )
        # A failed auto-install carries the installer output tail so it is diagnosable,
        # not just the generic missing-binary hint.
        with pytest.warns(toolset_module.BrowserUnavailableWarning, match='HTTP 403 forbidden') as warned:
            result = await agent.run('screenshot the page')
        assert 'Chromium is not installed' in str(warned[0].message)
        assert 'HTTP 403 forbidden' in _tool_results(result)

    async def test_close_error_still_exits_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, close_error=RuntimeError('close failed'))
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()])
        with pytest.raises(RuntimeError, match='close failed'):
            await agent.run('screenshot the page')
        assert cm.exited is True  # driver exited despite the close error

    async def test_teardown_error_does_not_mask_run_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, close_error=RuntimeError('close failed'))
        browser = PlaywrightBrowser()

        class _RunFailure(Exception):
            pass

        async def _handler() -> AgentRunResult[None]:
            await browser._session.ensure_page()
            raise _RunFailure('run failed')

        # The run's own exception wins; the close error raised during teardown is dropped.
        with pytest.raises(_RunFailure, match='run failed'):
            await browser.wrap_run(_ctx(), handler=_handler)
        # cm.exited is set in the teardown finally after browser.close() raised, so the
        # driver still tore down and only the masking close error was swallowed.
        assert cm.exited is True

    async def test_pending_event_tasks_cancelled_on_run_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        browser = PlaywrightBrowser()

        async def _same_instance(self: PlaywrightBrowser[None], ctx: RunContext[None]) -> PlaywrightBrowser[None]:
            return self

        monkeypatch.setattr(PlaywrightBrowser, 'for_run', _same_instance)
        pending = asyncio.ensure_future(asyncio.sleep(3600))
        browser._session._event_tasks.add(pending)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[browser])
        await agent.run('screenshot the page')
        assert pending.cancelled()
        assert browser._session._event_tasks == set()


# --- Package surface --------------------------------------------------------


def test_public_exports() -> None:
    assert DEFAULT_MAX_CONTENT_TOKENS == 4000
    assert DEFAULT_ACTION_TIMEOUT_MS == 5_000
    assert DEFAULT_NAVIGATION_TIMEOUT_MS == 60_000
    assert BrowserEvent is not None
    assert issubclass(PlaywrightBrowser, object)
    assert PlaywrightBrowserToolset is not None
    assert PlaywrightBrowserSession is not None
    assert issubclass(BrowserUnavailableError, RuntimeError)
    assert issubclass(BrowserUnavailableWarning, UserWarning)
    assert EgressPolicy().refuse(EgressRequest(url='https://example.com/', kind='navigation')) is None
    assert DEFAULT_ALLOWLIST_REACH == frozenset({'navigation', 'data'})
    assert DEFAULT_RESOLVED_KINDS == frozenset(get_args(RequestKind))
    assert set(get_args(RequestKind)) == {'navigation', 'subframe', 'data', 'subresource'}


# --- Embedded frames --------------------------------------------------------


class TestEmbeddedFrames:
    """Content inside an iframe: page-level reads stop at the frame boundary."""

    async def test_page_text_includes_child_frame_text(self) -> None:
        page = _FakePage(body='Conference')
        page.frames.append(_FakeFrameContent(text='Deep dive on agents'))
        result = await _toolset(page).get_text()
        assert result == 'Conference\n\n[frame https://embed.example.com/]\nDeep dive on agents'

    async def test_navigate_reports_embedded_content(self) -> None:
        page = _FakePage(body='Conference')
        page.frames.append(_FakeFrameContent(text='Deep dive on agents'))
        result = await _toolset(page).navigate('https://example.com/')
        assert isinstance(result, str)
        assert 'Deep dive on agents' in result

    async def test_frame_url_loses_its_credentials(self) -> None:
        # An OAuth or payment step is exactly the kind of thing a page embeds, and its
        # frame URL carries the code the model must not be handed.
        page = _FakePage(body='Checkout')
        page.frames.append(
            _FakeFrameContent(url='https://idp.example.com/callback?code=secret-code', text='Signing you in')
        )
        result = await _toolset(page).get_text()
        assert result == 'Checkout\n\n[frame https://idp.example.com/callback?code=REDACTED]\nSigning you in'

    async def test_blank_frame_is_left_out(self) -> None:
        page = _FakePage(body='Conference')
        page.frames.append(_FakeFrameContent(text='   \n '))
        assert await _toolset(page).get_text() == 'Conference'

    async def test_unreadable_frame_does_not_fail_the_read(self) -> None:
        # A frame can detach mid-read; a missing embed must not turn a successful
        # action into an error.
        page = _FakePage(body='Conference')
        page.frames.append(_FakeFrameContent(inner_text_error=PlaywrightError('frame detached')))
        page.frames.append(_FakeFrameContent(url='https://ok.example.com/', text='visible embed'))
        result = await _toolset(page).get_text()
        assert result == 'Conference\n\n[frame https://ok.example.com/]\nvisible embed'

    async def test_frame_sweep_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # One unresponsive frame must not consume the action's deadline: whatever
        # was collected before the sweep budget ran out is kept.
        monkeypatch.setattr(toolset_module, '_FRAME_TEXT_BUDGET_MS', 10)
        page = _FakePage(body='Conference')
        page.frames.append(_HangingFrame())
        assert await _toolset(page).get_text() == 'Conference'

    async def test_wait_for_matches_inside_a_frame(self) -> None:
        page = _FakePage(body='Conference', wait_for_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'))
        page.frames.append(_FakeFrameContent(text='late content'))
        result = await _toolset(page).wait_for(text='late content')
        assert result.startswith("Found 'late content'.")

    async def test_wait_for_reports_the_timeout_when_no_frame_matches(self) -> None:
        page = _FakePage(wait_for_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'))
        page.frames.append(
            _FakeFrameContent(wait_for_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'), wait_delay=0.01)
        )
        result = await _toolset(page).wait_for(selector='.missing')
        assert result.startswith('Error: wait_for timed out after 5000ms.')

    async def test_slower_waits_are_cancelled_once_one_matches(self) -> None:
        page = _FakePage(body='Conference')
        slow = _FakeFrameContent(wait_delay=5)
        page.frames.append(slow)
        # The main-frame wait resolves immediately; the pending frame wait is
        # cancelled rather than left running past the tool call.
        assert (await _toolset(page).wait_for(selector='.ready')).startswith("Found '.ready'.")


# --- Keyboard, dropdowns, hover ---------------------------------------------


class TestInteractionTools:
    async def test_press_key_without_selector_uses_the_keyboard(self) -> None:
        page = _FakePage(body='results')
        result = await _toolset(page).press_key('Enter')
        assert page.keyboard.pressed == ['Enter']
        assert result == "Pressed 'Enter'.\n\nresults"

    async def test_press_key_with_selector_focuses_the_element(self) -> None:
        page = _FakePage(body='results')
        await _toolset(page).press_key('Enter', selector='input#q')
        assert page.pressed == [('input#q', 'Enter')]

    async def test_press_key_settles_navigation_and_enforces_policy(self) -> None:
        page = _FakePage(url='https://evil.com/landing')
        result = await _toolset(page, allowed_domains=['example.com']).press_key('Enter')
        assert result == 'Error: press_key reached a domain not in allowed_domains: https://evil.com/landing'

    async def test_press_key_error_is_bounded(self) -> None:
        page = _FakePage(press_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'))
        result = await _toolset(page).press_key('Enter', selector='input#q')
        assert result.startswith('Error: press_key timed out after 5000ms.')

    async def test_select_option_reports_what_was_selected(self) -> None:
        page = _FakePage(body='filtered')
        result = await _toolset(page).select_option('select#track', ['ai'])
        assert page.selected == [('select#track', ['ai'])]
        assert result == "Selected ['ai'] in 'select#track'.\n\nfiltered"

    async def test_select_option_enforces_policy_after_the_change(self) -> None:
        page = _FakePage(url='http://169.254.169.254/')
        result = await _toolset(page).select_option('select#track', ['ai'])
        assert result == (
            'Error: select_option reached a blocked private or link-local address: http://169.254.169.254/'
        )

    async def test_select_option_error_is_bounded(self) -> None:
        page = _FakePage(select_option_error=PlaywrightError('strict mode violation'))
        result = await _toolset(page).select_option('select#track', ['ai'])
        assert result == 'Error: select_option failed: strict mode violation'

    async def test_hover_returns_the_revealed_page(self) -> None:
        page = _FakePage(body='menu open')
        result = await _toolset(page).hover('nav .menu')
        assert page.hovered == ['nav .menu']
        assert result == "Hovered 'nav .menu'.\n\nmenu open"

    async def test_hover_error_is_bounded(self) -> None:
        page = _FakePage(hover_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'))
        result = await _toolset(page).hover('nav .menu')
        assert result.startswith('Error: hover timed out after 5000ms.')


# --- Browser events and tracing ---------------------------------------------


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


class TestBrowserEvents:
    """What the page did between tool calls: console, network, refusals, popups."""

    def _session(self, page: _FakePage) -> PlaywrightBrowserSession:
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page]
        return session

    async def test_console_messages_carry_the_page_severity(self) -> None:
        page = _FakePage()
        session = self._session(page)
        toolset = _toolset(page, session=session)
        for message in (
            _FakeConsoleMessage('log', 'starting'),
            _FakeConsoleMessage('warning', 'deprecated call'),
            _FakeConsoleMessage('error', 'boom'),
        ):
            session._on_console(message)
        session._on_page_error(RuntimeError('uncaught TypeError'))
        assert await toolset.console_messages() == (
            '[info] console log: starting\n'
            '[warning] console warning: deprecated call\n'
            '[error] console error: boom\n'
            '[error] page_error uncaught TypeError'
        )
        assert await toolset.console_messages(errors_only=True) == (
            '[error] console error: boom\n[error] page_error uncaught TypeError'
        )

    async def test_network_requests_report_status_and_filter_by_url(self) -> None:
        page = _FakePage()
        session = self._session(page)
        toolset = _toolset(page, session=session)
        session._on_response(_FakeResponse(url='https://example.com/api/sessions', status=200))
        session._on_response(_FakeResponse(url='https://example.com/missing', status=404))
        session._on_request_failed(
            _FakeNetworkRequest(url='https://cdn.example.com/app.js', failure='net::ERR_CONNECTION_REFUSED')
        )
        assert await toolset.network_requests() == (
            '[info] response GET 200 https://example.com/api/sessions\n'
            '[error] response GET 404 https://example.com/missing\n'
            '[error] request_failed GET https://cdn.example.com/app.js net::ERR_CONNECTION_REFUSED'
        )
        assert await toolset.network_requests(url_contains='/api/') == (
            '[info] response GET 200 https://example.com/api/sessions'
        )

    async def test_request_failed_without_a_reason_still_records(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session._on_request_failed(_FakeNetworkRequest(url='https://example.com/x', failure=None))
        assert await _toolset(page, session=session).network_requests() == (
            '[error] request_failed GET https://example.com/x failed'
        )

    async def test_guard_abort_is_recorded_once_with_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        async with session:
            await session.ensure_page()
            request_page = _FakeRequestPage()
            await page.context.dispatch(
                _FakeRequest('https://evil.com/', navigation=True, frame=request_page.main_frame)
            )
            # Chromium reports the abort as a failed request too; the guard's entry
            # already names the reason, so the bare failure is dropped.
            session._on_request_failed(_FakeNetworkRequest(url='https://evil.com/', failure='net::ERR_FAILED'))
        assert [event.describe() for event in session.events] == [
            '[warning] request_blocked https://evil.com/ domain not in allowed_domains'
        ]

    async def test_opened_popup_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            page.emit('popup', _FakePage(url='https://example.com/popup'))
            await asyncio.sleep(0)
        assert [event.describe() for event in session.events] == [
            '[info] popup_opened https://example.com/popup opened by the page'
        ]

    async def test_empty_log_reports_that_rather_than_nothing(self) -> None:
        toolset = _toolset(_FakePage())
        assert await toolset.console_messages() == 'No console messages recorded.'
        assert await toolset.network_requests() == 'No network requests recorded.'

    async def test_operation_span_carries_the_action_and_page_events(self) -> None:
        page = _FakePage()
        session = self._session(page)
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        toolset = _toolset(page, session=session)

        class _EmittingPage(_FakePage):
            async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
                session._on_console(_FakeConsoleMessage('error', 'boom'))
                return await super().inner_text(selector, timeout=timeout)

        session.page = _EmittingPage()
        await toolset.get_text()
        (span,) = exporter.get_finished_spans()
        assert span.name == 'browser get_text'
        assert span.attributes is not None
        assert span.attributes['browser.action'] == 'get_text'
        assert span.attributes['browser.timeout_ms'] == DEFAULT_ACTION_TIMEOUT_MS
        assert span.attributes['browser.outcome'] == 'ok'
        assert span.attributes['url.full'] == 'https://example.com/'
        (event,) = span.events
        assert event.name == 'browser.console'
        assert event.attributes is not None
        assert event.attributes['browser.event.message'] == 'error: boom'

    async def test_network_event_attributes_follow_otel_conventions(self) -> None:
        page = _FakePage()
        session = self._session(page)
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        toolset = _toolset(page, session=session)

        class _RequestingPage(_FakePage):
            async def inner_text(self, selector: str, *, timeout: float | None = None) -> str:
                session._on_response(_FakeResponse(url='https://example.com/api', status=503, method='POST'))
                return await super().inner_text(selector, timeout=timeout)

        session.page = _RequestingPage()
        await toolset.get_text()
        (span,) = exporter.get_finished_spans()
        (event,) = span.events
        assert event.name == 'browser.response'
        assert event.attributes is not None
        assert event.attributes['url.full'] == 'https://example.com/api'
        assert event.attributes['http.request.method'] == 'POST'
        assert event.attributes['http.response.status_code'] == 503
        # A response carries no message, so no empty attribute is emitted for one.
        assert 'browser.event.message' not in event.attributes

    async def test_failed_operation_is_marked_on_the_span(self) -> None:
        page = _FakePage(click_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.'))
        session = self._session(page)
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        await _toolset(page, session=session).click('button#go')
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes['browser.outcome'] == 'error'
        # Recorded on the failure path too: which page the click failed on is what
        # makes the span diagnosable.
        assert span.attributes['url.full'] == 'https://example.com/'

    async def test_events_outside_an_operation_are_still_logged(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session._on_console(_FakeConsoleMessage('log', 'between tool calls'))
        assert [event.kind for event in session.events] == ['console']

    async def test_the_log_is_bounded(self) -> None:
        page = _FakePage()
        session = self._session(page)
        for index in range(toolset_module._EVENT_LOG_LIMIT + 10):
            session._on_console(_FakeConsoleMessage('log', str(index)))
        assert len(session.events) == toolset_module._EVENT_LOG_LIMIT

    async def test_a_run_reports_browser_spans_to_the_agents_instrumentation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The capability adopts the run's tracer, so browser spans land wherever the
        # agent's instrumentation settings send its tool calls.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        model = InstrumentedModel(
            TestModel(call_tools=['screenshot']), InstrumentationSettings(tracer_provider=provider)
        )
        agent = Agent(model, capabilities=[PlaywrightBrowser()])
        await agent.run('screenshot the page')
        assert 'browser screenshot' in [span.name for span in exporter.get_finished_spans()]


# --- Review fixes -----------------------------------------------------------


class TestSessionAndToolsetAgree:
    async def test_the_toolset_uses_the_sessions_policy(self) -> None:
        # A session built to allow a local app must not be second-guessed by a
        # toolset holding a policy of its own.
        page = _FakePage(url='http://127.0.0.1:8000/admin')
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(block_private_addresses=False))
        session.page = page
        toolset = PlaywrightBrowserToolset[None](session=session)
        result = await toolset.navigate('http://127.0.0.1:8000/admin')
        assert isinstance(result, str)
        assert result.startswith('URL: http://127.0.0.1:8000/admin')

    async def test_the_toolset_cannot_hold_a_second_policy(self) -> None:
        # The two layers act at different moments -- the route guard before the
        # request leaves, the tool check after the page settled -- so a toolset
        # with its own stricter policy would let the guard send what the tools
        # would have refused. There is no parameter to create that split.
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        session.page = _FakePage()
        toolset = PlaywrightBrowserToolset[None](session=session)
        assert 'policy' not in inspect.signature(PlaywrightBrowserToolset.__init__).parameters
        assert await toolset.navigate('https://evil.com/') == (
            'Error: domain not in allowed_domains: https://evil.com/'
        )


class TestSessionReuse:
    async def test_a_session_can_be_entered_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The second block never touches the browser, so its driver context manager
        # is never entered; teardown must not exit one that was never started.
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
        assert cm.exited is True
        async with session:
            pass


class TestDeadlinesCoverEveryAwait:
    async def test_frame_sweep_never_outlives_a_shorter_per_call_deadline(self) -> None:
        page = _FakePage()
        page.frames.append(_HangingFrame())
        toolset = _toolset(page)
        # The sweep is capped by the caller's deadline when that is the shorter of
        # the two, and by the sweep budget when it is not.
        assert toolset._frame_budget(50) == 50
        assert toolset._frame_budget(30_000) == toolset_module._FRAME_TEXT_BUDGET_MS
        assert toolset._frame_budget(None) == toolset_module._FRAME_TEXT_BUDGET_MS
        assert toolset._frame_budget(0) == toolset_module._FRAME_TEXT_BUDGET_MS
        assert await toolset.get_text(timeout_ms=50) == 'Hello body'

    async def test_coordinate_scroll_runs_under_the_deadline(self) -> None:
        # `Mouse.move`/`Mouse.wheel` take no timeout, so an unbounded await would
        # hold the operation lock for the rest of the run.
        result = await _toolset(_HangingMousePage()).scroll('down', x=10, y=20, timeout_ms=1)
        assert result.startswith('Error: scroll timed out after 1ms.')


class TestCredentialsStayOutOfTelemetry:
    async def test_recorded_urls_lose_their_userinfo(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session._on_response(_FakeResponse(url='https://user:secret@example.com/api?q=1'))
        assert await _toolset(page, session=session).network_requests() == (
            '[info] response GET 200 https://example.com/api?q=1'
        )

    async def test_span_url_loses_its_userinfo(self) -> None:
        page = _FakePage(url='https://user:secret@example.com/dashboard')
        session = PlaywrightBrowserSession()
        session.page = page
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        await _toolset(page, session=session, block_private_addresses=False).get_text()
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None
        assert span.attributes['url.full'] == 'https://example.com/dashboard'

    async def test_credential_parameters_are_redacted_by_name(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session._on_response(_FakeResponse(url='https://example.com/dl?id=7&token=secret&sig=abc'))
        # The endpoint and the parameter names survive -- they are what makes a
        # recorded request findable -- and only the credential values go.
        assert await _toolset(page, session=session).network_requests() == (
            '[info] response GET 200 https://example.com/dl?id=7&token=REDACTED&sig=REDACTED'
        )

    def test_credentials_go_from_the_fragment_too(self) -> None:
        # An OAuth implicit grant returns its token in the fragment.
        assert toolset_module._without_credentials('https://app.example.com/cb#access_token=abc&state=x') == (
            'https://app.example.com/cb#access_token=REDACTED&state=x'
        )

    def test_the_prefixed_oauth_parameters_are_redacted_too(self) -> None:
        # The pattern anchors a name at `?`, `&` or `#`, so the bare `secret` and
        # `token` entries never match these spellings.
        assert toolset_module._without_credentials('https://id.example.com/t?client_secret=a&oauth_token=b&x=1') == (
            'https://id.example.com/t?client_secret=REDACTED&oauth_token=REDACTED&x=1'
        )

    def test_a_url_urlsplit_rejects_is_still_cleaned(self) -> None:
        # Chromium accepts hosts the stdlib parser raises on, so the strip cannot
        # depend on parsing succeeding.
        assert toolset_module._without_credentials('http://user:pw@[::1/x') == 'http://[::1/x'
        assert toolset_module._without_credentials('https://example.com/a@b') == 'https://example.com/a@b'


class TestErrorMessagesLoseTheirCredentials:
    """Playwright quotes the URL it was working on, and its errors reach the model verbatim."""

    async def test_an_interpolated_playwright_error_is_redacted(self) -> None:
        page = _FakePage(
            goto_error=PlaywrightError('Page.goto: net::ERR_UNSAFE_PORT at https://example.com/cb?code=SECRET')
        )
        result = await _toolset(page).navigate('https://example.com/')
        assert result == (
            'Error: navigate failed: Page.goto: net::ERR_UNSAFE_PORT at https://example.com/cb?code=REDACTED'
        )

    async def test_a_failed_element_read_is_redacted(self) -> None:
        page = _FakePage(inner_text_error=PlaywrightError('waiting for https://example.com/api?token=SECRET'))
        result = await _toolset(page).get_text('h1')
        assert result == "Error getting text from 'h1': waiting for https://example.com/api?token=REDACTED"

    async def test_a_script_error_is_redacted(self) -> None:
        page = _FakePage(evaluate_raises=PlaywrightError('TypeError at https://example.com/app.js?sig=SECRET'))
        result = await _toolset(page).execute_js('boom()')
        assert result == 'JS error: TypeError at https://example.com/app.js?sig=REDACTED'


class TestWebSocketEgress:
    """`context.route` never sees a WebSocket, so the block needs its own guard."""

    async def _context(self, monkeypatch: pytest.MonkeyPatch, browser: PlaywrightBrowser[object]) -> _FakePage:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        agent = Agent(TestModel(call_tools=['screenshot']), capabilities=[browser])
        await agent.run('screenshot the page')
        return page

    async def test_socket_to_a_private_address_is_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = await self._context(monkeypatch, PlaywrightBrowser[object]())
        websocket = await page.context.dispatch_websocket('ws://127.0.0.1:9000/admin')
        assert websocket.closed is True
        assert websocket.connected is False

    async def test_a_public_socket_is_connected_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = await self._context(monkeypatch, PlaywrightBrowser[object]())
        websocket = await page.context.dispatch_websocket('wss://example.com/live')
        assert websocket.connected is True
        assert websocket.closed is False

    async def test_the_allowlist_reaches_sockets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A socket moves data like `fetch` does, so it answers to the same list.
        page = await self._context(monkeypatch, PlaywrightBrowser[object](allowed_domains=['example.com']))
        refused = await page.context.dispatch_websocket('wss://realtime.other.com/live')
        assert refused.connected is False
        permitted = await page.context.dispatch_websocket('wss://realtime.example.com/live')
        assert permitted.connected is True

    async def test_the_allowlist_reaches_sockets_without_the_address_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The two axes are independent: turning the address block off must not take
        # the allowlist off the sockets it bounds.
        page = await self._context(
            monkeypatch,
            PlaywrightBrowser[object](allowed_domains=['example.com'], block_private_addresses=False),
        )
        assert page.context.websocket_routes == ['**/*']
        refused = await page.context.dispatch_websocket('wss://evil.test/live')
        assert refused.connected is False
        assert refused.closed is True
        permitted = await page.context.dispatch_websocket('wss://realtime.example.com/live')
        assert permitted.connected is True

    async def test_no_socket_guard_when_nothing_is_enforced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = await self._context(monkeypatch, PlaywrightBrowser[object](block_private_addresses=False))
        assert page.context.websocket_routes == []

    async def test_a_closed_socket_is_recorded_with_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = PlaywrightBrowserSession()
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        async with session:
            await session.ensure_page()
            await page.context.dispatch_websocket('ws://169.254.169.254/')
        assert [event.describe() for event in session.events] == [
            '[warning] request_blocked ws://169.254.169.254/ blocked private or link-local address'
        ]


class TestMessageRedaction:
    async def test_a_console_message_loses_its_credentials(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session._on_console(_FakeConsoleMessage('error', 'failed: https://api.example.com/v1?access_token=secret'))
        assert await _toolset(page, session=session).console_messages() == (
            '[error] console error: failed: https://api.example.com/v1?access_token=REDACTED'
        )


# --- Tabs, dialogs and the operation deadline -------------------------------


class _HangingClickMouse(_FakeMouse):
    async def click(self, x: float, y: float) -> None:
        await asyncio.Event().wait()


class _HangingClickMousePage(_FakePage):
    """A page whose coordinate click never settles."""

    def __init__(self) -> None:
        super().__init__()
        self.mouse = _HangingClickMouse()


class _Clock:
    """A monotonic clock the test moves by hand, so a slow stage costs no real time."""

    def __init__(self) -> None:
        self.seconds = 1_000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class _SlowLoadPage(_FakePage):
    """A page whose navigation takes six seconds of the fake clock."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self._clock = clock

    async def goto(self, url: str, *, timeout: float | None = None) -> None:
        self._clock.advance(6.0)
        await super().goto(url, timeout=timeout)


class _SlowSettlePage(_FakePage):
    """A page that settles six seconds after the action that navigated it."""

    def __init__(self, clock: _Clock) -> None:
        super().__init__()
        self._clock = clock

    async def wait_for_load_state(self, state: str, *, timeout: float | None = None) -> None:
        self._clock.advance(6.0)
        await super().wait_for_load_state(state, timeout=timeout)


class TestOperationDeadline:
    async def test_each_stage_gets_what_is_left_not_the_whole_budget(self) -> None:
        class _SlowGotoPage(_FakePage):
            async def goto(self, url: str, *, timeout: float | None = None) -> None:
                await asyncio.sleep(0.05)
                await super().goto(url, timeout=timeout)

        page = _SlowGotoPage()
        await _toolset(page).navigate('https://example.com/', timeout_ms=2000)
        # The goto saw the full budget; every later call in the same operation saw
        # less, because the time the goto spent is gone.
        assert page.timeouts['goto'] == 2000
        for stage in ('wait_for_load_state', 'inner_text'):
            remaining = page.timeouts[stage]
            assert remaining is not None and 0 < remaining < 2000

    async def test_a_zero_default_still_means_no_deadline(self) -> None:
        page = _FakePage()
        await _toolset(page, action_timeout_ms=0, navigation_timeout_ms=0).get_text()
        assert page.timeouts['inner_text'] == 0

    async def test_coordinate_click_is_bounded(self) -> None:
        page = _HangingClickMousePage()
        result = await _toolset(page).click('10,20', timeout_ms=20)
        assert result.startswith('Error: click timed out after 20ms.')

    async def test_the_reported_budget_is_the_one_that_was_configured(self) -> None:
        class _SlowClickPage(_FakePage):
            async def click(self, selector: str, *, timeout: float | None = None) -> None:
                await asyncio.sleep(0.03)
                raise PlaywrightTimeoutError('Timeout 1000ms exceeded.')

        result = await _toolset(_SlowClickPage()).click('button#go', timeout_ms=1000)
        assert result.startswith('Error: click timed out after 1000ms.')

    async def test_a_slow_load_does_not_starve_the_reads_that_follow_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Six seconds of loading leaves nothing of the five-second action budget, so
        # a trailing read bounded by that one would fail on a page that loaded fine.
        clock = _Clock()
        monkeypatch.setattr(toolset_module, 'monotonic', clock)
        page = _SlowLoadPage(clock)
        result = await _toolset(page).navigate('https://example.com/')
        assert result == 'URL: https://example.com/\nTitle: Example\n\nHello body'
        body_deadline = page.timeouts['inner_text']
        assert body_deadline is not None and body_deadline > 10_000

    async def test_a_slow_settle_does_not_starve_the_read_after_a_click(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = _Clock()
        monkeypatch.setattr(toolset_module, 'monotonic', clock)
        page = _SlowSettlePage(clock)
        result = await _toolset(page).click('button#go')
        assert result == "Clicked 'button#go'. URL: https://example.com/\n\nHello body"
        body_deadline = page.timeouts['inner_text']
        assert body_deadline is not None and body_deadline > 10_000


class _HangingContext:
    """A browser context whose new page never opens."""

    async def new_page(self) -> _FakePage:
        await asyncio.Event().wait()
        return _FakePage()  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingContextBrowser(_FakePlaywrightBrowser):
    """A browser that connects and then never opens a context."""

    async def new_context(
        self,
        *,
        storage_state: StorageState | None = None,
        service_workers: str | None = None,
        accept_downloads: bool | None = None,
    ) -> _FakeBrowserContext:
        await asyncio.Event().wait()
        return _FakeBrowserContext(self._page)  # pragma: no cover -- unreachable; the wait is cancelled


class _HangingContextChromium(_FakeChromium):
    async def launch(
        self, *, headless: bool, chromium_sandbox: bool = False, timeout: int | None = None
    ) -> _FakePlaywrightBrowser:
        return _HangingContextBrowser(self._page)


class _HangingTabPage(_FakePage):
    """A tab that never comes to the front and never closes."""

    async def bring_to_front(self) -> None:
        await asyncio.Event().wait()

    async def close(self) -> None:
        await asyncio.Event().wait()


class TestLaunchIsBounded:
    """Starting a browser happens inside a tool call, holding the operation lock."""

    async def test_a_context_that_never_opens_becomes_a_bounded_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _FakeDriverCM(_FakeDriver(_HangingContextChromium(page)))
        monkeypatch.setattr(toolset_module, 'async_playwright', lambda: cm)
        session = PlaywrightBrowserSession(launch_timeout_ms=20)
        async with session:
            result = await PlaywrightBrowserToolset[None](session=session).get_text()
        assert result.startswith('Error: get_text timed out')

    async def test_a_zero_launch_timeout_still_means_no_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession(launch_timeout_ms=0)
        async with session:
            assert await session.ensure_page() is page


class TestEgressPolicy:
    """The policy's own decisions, and the guard asking it about real request kinds."""

    def _refuse(self, policy: toolset_module.EgressPolicy, url: str, kind: str, resource_type: str = '') -> str | None:
        request = toolset_module.EgressRequest(
            url=url,
            kind=kind,  # pyright: ignore[reportArgumentType]
            resource_type=resource_type or ('document' if kind == 'navigation' else 'fetch'),
        )
        return policy.refuse(request)

    def test_data_requests_answer_to_the_allowlist_and_assets_do_not(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'])
        assert self._refuse(policy, 'https://evil.test/?stolen=1', 'data') == 'domain not in allowed_domains'
        assert self._refuse(policy, 'https://cdn.other.net/app.js', 'subresource', 'script') is None
        assert self._refuse(policy, 'https://idp.other.net/login', 'subframe', 'document') is None

    def test_a_subdomain_of_an_allowed_domain_is_reachable(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'])
        assert self._refuse(policy, 'https://api.example.com/v1/items', 'data') is None
        assert self._refuse(policy, 'https://example.com/', 'navigation') is None

    def test_subdomains_can_be_turned_off(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'], include_subdomains=False)
        assert self._refuse(policy, 'https://api.example.com/v1', 'data') == 'domain not in allowed_domains'
        assert self._refuse(policy, 'https://example.com/', 'navigation') is None

    def test_the_reach_can_cover_every_kind(self) -> None:
        policy = toolset_module.EgressPolicy(
            allowed_domains=['example.com'], allowlist_reach=frozenset(get_args(toolset_module.RequestKind))
        )
        assert self._refuse(policy, 'https://cdn.other.net/app.js', 'subresource', 'script') is not None
        assert self._refuse(policy, 'https://idp.other.net/login', 'subframe', 'document') is not None

    def test_blocked_domains_win_over_the_allowlist(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'], blocked_domains=['ads.example.com'])
        assert self._refuse(policy, 'https://ads.example.com/beacon', 'data') == 'domain in blocked_domains'
        # And they reach the kinds the allowlist deliberately does not.
        assert self._refuse(policy, 'https://ads.example.com/px.gif', 'subresource', 'image') is not None

    def test_blocked_domains_apply_without_an_allowlist(self) -> None:
        policy = toolset_module.EgressPolicy(blocked_domains=['tracker.test'])
        assert self._refuse(policy, 'https://tracker.test/px', 'data') == 'domain in blocked_domains'
        assert self._refuse(policy, 'https://anything.test/', 'navigation') is None
        assert policy.enforced() is True

    def test_inline_assets_load_while_hostless_navigation_does_not(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'])
        assert self._refuse(policy, 'data:image/png;base64,AAAA', 'subresource', 'image') is None
        assert self._refuse(policy, 'data:text/html,<h1>x', 'navigation') == 'URL with no host'

    def test_a_private_address_is_refused_for_every_kind(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'])
        for kind, resource_type in (('data', 'fetch'), ('subresource', 'image'), ('subframe', 'document')):
            reason = self._refuse(policy, 'http://169.254.169.254/latest/meta-data/', kind, resource_type)
            assert reason == 'blocked private or link-local address'

    def test_a_wildcard_entry_is_refused_at_construction(self) -> None:
        # It would match nothing and read as a configured allowlist.
        with pytest.raises(UserError, match='wildcard'):
            toolset_module.EgressPolicy(allowed_domains=['*.example.com'])
        with pytest.raises(UserError, match='wildcard'):
            toolset_module.EgressPolicy(blocked_domains=['*.ads.example.com'])

    def test_a_leading_dot_entry_is_refused_at_construction(self) -> None:
        # `.example.com` matches neither the host nor the `.example.com` suffix test,
        # so it reads as a configured allowlist while admitting nothing.
        with pytest.raises(UserError, match='starts with a dot'):
            toolset_module.EgressPolicy(allowed_domains=['.example.com'])
        with pytest.raises(UserError, match='starts with a dot'):
            toolset_module.EgressPolicy(blocked_domains=['.ads.example.com'])
        # Checked as `_matches` will read it: surrounding whitespace must not
        # slip a dodged entry past the same validation.
        with pytest.raises(UserError, match='starts with a dot'):
            toolset_module.EgressPolicy(allowed_domains=[' .example.com'])
        with pytest.raises(UserError, match='wildcard'):
            toolset_module.EgressPolicy(blocked_domains=[' *.example.com '])

    def test_a_subclass_decides_what_the_fields_cannot(self) -> None:
        class FontsFromAnywhere(toolset_module.EgressPolicy):
            def refuse(self, request: toolset_module.EgressRequest) -> str | None:
                if request.resource_type == 'font':
                    return None
                return super().refuse(request)

        policy = FontsFromAnywhere(
            allowed_domains=['example.com'], allowlist_reach=frozenset(get_args(toolset_module.RequestKind))
        )
        assert self._refuse(policy, 'https://fonts.other.net/x.woff2', 'subresource', 'font') is None
        assert self._refuse(policy, 'https://fonts.other.net/x.css', 'subresource', 'stylesheet') is not None

    def test_describe_drops_the_subdomain_phrase_when_they_are_off(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'], include_subdomains=False)
        assert 'subdomains' not in policy.describe()

    def test_describe_names_a_reach_that_is_not_the_default(self) -> None:
        # A locked-down or narrowed allowlist reads the same as the default one
        # unless the description says what it bounds, and the instructions then
        # promise the model a reach the guards do not grant.
        assert 'allowlist bounds' not in toolset_module.EgressPolicy(allowed_domains=['example.com']).describe()
        locked = toolset_module.EgressPolicy(
            allowed_domains=['example.com'], allowlist_reach=frozenset(get_args(toolset_module.RequestKind))
        )
        assert 'allowlist bounds data, navigation, subframe, subresource' in locked.describe()
        bounds_nothing = toolset_module.EgressPolicy(allowed_domains=['example.com'], allowlist_reach=frozenset())
        assert 'allowlist bounds nothing' in bounds_nothing.describe()

    def test_describe_names_what_the_model_may_reach(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'], blocked_domains=['ads.example.com'])
        described = policy.describe()
        assert 'example.com (and their subdomains)' in described
        assert 'except ads.example.com' in described
        assert 'private/internal addresses blocked' in described

    async def test_the_guard_refuses_a_page_fetch_to_another_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        async with session:
            await session.ensure_page()
            assert page.context is not None
            frame = _FakeFrame(_FakeRequestPage())
            route = await page.context.dispatch(
                _FakeRequest('https://evil.test/?stolen=1', navigation=False, frame=frame, resource_type='fetch')
            )
            assert route.aborted is True
            assert any(event.kind == 'request_blocked' for event in session.events)

    async def test_the_guard_lets_an_image_from_anywhere_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        async with session:
            await session.ensure_page()
            assert page.context is not None
            frame = _FakeFrame(_FakeRequestPage())
            route = await page.context.dispatch(
                _FakeRequest('https://cdn.other.net/hero.png', navigation=False, frame=frame, resource_type='image')
            )
            assert route.aborted is False
            assert route.continued is True

    async def test_a_beacon_counts_as_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # `navigator.sendBeacon` is a send channel, not an asset the page renders.
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        async with session:
            await session.ensure_page()
            assert page.context is not None
            frame = _FakeFrame(_FakeRequestPage())
            route = await page.context.dispatch(
                _FakeRequest('https://evil.test/collect', navigation=False, frame=frame, resource_type='ping')
            )
            assert route.aborted is True

    def test_the_capability_refuses_a_policy_and_a_shorthand_together(self) -> None:
        with pytest.raises(UserError, match='not both'):
            PlaywrightBrowser[object](
                allowed_domains=['example.com'], policy=toolset_module.EgressPolicy(allowed_domains=['other.com'])
            )
        with pytest.raises(UserError, match='not both'):
            PlaywrightBrowser[object](block_private_addresses=False, policy=toolset_module.EgressPolicy())

    def test_the_capability_uses_the_policy_it_is_given(self) -> None:
        policy = toolset_module.EgressPolicy(allowed_domains=['example.com'], include_subdomains=False)
        browser = PlaywrightBrowser[object](policy=policy)
        assert browser._session.policy is policy  # pyright: ignore[reportPrivateUsage]


class TestSpanOutcome:
    async def test_a_returned_error_marks_the_span_failed(self) -> None:
        page = _FakePage(selector_raises=True)
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page]
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        result = await _toolset(page, session=session).get_text('#missing')
        assert result.startswith('Error getting text')
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None and span.attributes['browser.outcome'] == 'error'

    async def test_an_escaping_exception_marks_the_span_failed(self) -> None:
        # A capability that was never started is a wiring bug: it still ends the run,
        # unlike a browser that could not be launched.
        session = PlaywrightBrowserSession()
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        toolset = PlaywrightBrowserToolset[None](session=session)
        with pytest.raises(RuntimeError, match='PlaywrightBrowser is not running'):
            await toolset.snapshot()
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None and span.attributes['browser.outcome'] == 'error'
        assert 'url.full' not in span.attributes

    async def test_a_refusal_opens_its_own_span(self) -> None:
        # A call the egress policy turns away never reaches a page, and a trace that
        # showed only the calls that did would not show it happened at all.
        page = _FakePage()
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        session.page = page
        session.pages = [page]
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        result = await _toolset(page, session=session).navigate('https://evil.example/')
        assert result == 'Error: domain not in allowed_domains: https://evil.example/'
        (span,) = exporter.get_finished_spans()
        assert span.name == 'browser navigate'
        assert span.attributes is not None and span.attributes['browser.outcome'] == 'error'
        assert 'url.full' not in span.attributes

    async def test_the_span_names_the_tab_the_operation_ended_on(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://example.com/popup')
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page, popup]
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        await _toolset(page, session=session).tabs('select', 1)
        (span,) = exporter.get_finished_spans()
        assert span.attributes is not None and span.attributes['url.full'] == 'https://example.com/popup'

    async def test_the_log_tools_open_their_own_span(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page]
        tracer, exporter = _recording_tracer()
        session.tracer = tracer
        toolset = _toolset(page, session=session)
        await toolset.console_messages()
        await toolset.network_requests()
        await toolset.handle_next_dialog(accept=True)
        assert [span.name for span in exporter.get_finished_spans()] == [
            'browser console_messages',
            'browser network_requests',
            'browser handle_next_dialog',
        ]


class TestEventLogBounds:
    def _session(self, page: _FakePage) -> PlaywrightBrowserSession:
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page]
        return session

    async def test_a_long_console_message_is_clipped_when_recorded(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session._on_console(_FakeConsoleMessage('log', 'x' * 5000))
        (event,) = session.events
        assert len(event.message) == toolset_module._MAX_EVENT_CHARS + len('...')

    async def test_the_oldest_requests_are_dropped_not_the_newest(self) -> None:
        page = _FakePage()
        session = self._session(page)
        for index in range(60):
            session.record(
                toolset_module.BrowserEvent(
                    kind='response', level='info', message='', url=f'https://example.com/{index}', status=200
                )
            )
        result = await _toolset(page, session=session, max_content_tokens=40).network_requests()
        assert result.startswith('[... ')
        assert 'https://example.com/59' in result
        assert 'https://example.com/0 ' not in result

    async def test_one_request_larger_than_the_budget_is_still_returned(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session.record(
            toolset_module.BrowserEvent(
                kind='response', level='info', message='', url='https://example.com/' + 'a' * 80
            )
        )
        result = await _toolset(page, session=session, max_content_tokens=5).network_requests()
        # Nothing fits, so the one entry is kept and cut rather than replaced by a
        # marker saying it was dropped.
        assert result == '[info] response http'
        assert len(result) == 20

    async def test_errors_only_drops_the_successful_requests(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session._on_response(_FakeResponse(url='https://example.com/ok', status=200))
        session._on_response(_FakeResponse(url='https://example.com/gone', status=404))
        result = await _toolset(page, session=session).network_requests(errors_only=True)
        assert 'https://example.com/gone' in result
        assert 'https://example.com/ok' not in result


class TestTypingAndWaiting:
    async def test_sequential_typing_clears_then_presses_each_key(self) -> None:
        page = _FakePage()
        await _toolset(page).type_text('input#q', 'hello', sequential=True)
        assert page.filled == [('input#q', '')]
        assert page.typed == [('input#q', 'hello')]

    async def test_waiting_for_something_to_go_away_asks_for_the_hidden_state(self) -> None:
        page = _FakePage()
        page.frames.append(_FakeFrameContent())
        result = await _toolset(page).wait_for(selector='.spinner', gone=True)
        assert result.startswith("Gone '.spinner'.")
        assert page.wait_states == ['hidden']
        assert page.frames[1].wait_states == ['hidden']

    async def test_a_frame_that_keeps_the_element_fails_the_disappearance_wait(self) -> None:
        page = _FakePage()
        page.frames.append(_FakeFrameContent(wait_for_error=PlaywrightTimeoutError('Timeout 5000ms exceeded.')))
        result = await _toolset(page).wait_for(text='Loading', gone=True)
        assert result.startswith('Error: wait_for timed out')


class TestTabs:
    def _session(self, page: _FakePage) -> PlaywrightBrowserSession:
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = [page]
        return session

    async def test_list_marks_the_active_tab(self) -> None:
        page = _FakePage(url='https://example.com/', title='Example')
        session = self._session(page)
        session.pages.append(_FakePage(url='https://example.com/popup', title='Popup'))
        assert await _toolset(page, session=session).tabs() == (
            '0 (active): Example -- https://example.com/\n1: Popup -- https://example.com/popup'
        )

    async def test_a_tab_whose_title_cannot_be_read_is_still_listed(self) -> None:
        page = _FakePage(title_error=PlaywrightError('page crashed'))
        assert await _toolset(page).tabs() == '0 (active): <title unavailable> -- https://example.com/'

    async def test_select_switches_the_page_every_tool_acts_on(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://example.com/popup', body='Popup body')
        session = self._session(page)
        session.pages.append(popup)
        result = await _toolset(page, session=session).tabs('select', 1)
        assert result == 'Selected tab 1. URL: https://example.com/popup\n\nPopup body'
        assert session.page is popup
        assert popup.brought_to_front == 1

    async def test_selecting_a_tab_outside_the_allowlist_bounces_it(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://evil.example/')
        session = PlaywrightBrowserSession(policy=toolset_module.EgressPolicy(allowed_domains=['example.com']))
        session.page = page
        session.pages = [page, popup]
        result = await _toolset(page, session=session).tabs('select', 1)
        assert result == 'Error: tabs reached a domain not in allowed_domains: https://evil.example/'
        assert popup.goto_calls == ['about:blank']

    async def test_new_opens_a_blank_tab_and_makes_it_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            result = await _toolset(page, session=session).tabs('new')
            assert result == 'Opened blank tab 1 and made it active. Load it with navigate.'
            assert len(session.pages) == 2
            assert session.page is session.pages[1]

    async def test_closing_the_active_tab_moves_to_the_one_that_remains(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://example.com/popup')
        session = self._session(page)
        session.pages.append(popup)
        session.page = popup
        result = await _toolset(popup, session=session).tabs('close')
        assert result == 'Closed tab 1. Active tab is now 0.'
        assert popup.closed is True
        assert session.pages == [page]
        assert session.page is page

    async def test_closing_another_tab_leaves_the_active_one_alone(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://example.com/popup')
        session = self._session(page)
        session.pages.append(popup)
        result = await _toolset(page, session=session).tabs('close', 1)
        assert result == 'Closed tab 1. Active tab is now 0.'
        assert session.pages == [page]

    async def test_a_tab_that_refuses_to_close_reports_it(self) -> None:
        page = _FakePage()
        popup = _FakePage(url='https://example.com/popup', close_error=PlaywrightError('page is busy'))
        session = self._session(page)
        session.pages.append(popup)
        result = await _toolset(page, session=session).tabs('close', 1)
        assert result == 'Error: tabs failed: page is busy'
        assert session.pages == [page, popup]

    async def test_a_tool_on_a_closed_last_tab_names_the_recovery(self) -> None:
        # The active pointer stays on the closed page on purpose, so the tools that
        # act on it have to say what reopens one.
        session = PlaywrightBrowserSession()
        page = _FakePage(inner_text_error=TargetClosedError('Target page closed'))
        toolset = _toolset(page, session=session)
        # What `_on_page_closed` leaves behind when the page that closed was the last.
        session.pages = []
        result = await toolset.get_text()
        assert result == "Error: get_text failed: the active tab has closed. Open one with tabs('new')."

    async def test_the_last_tab_cannot_be_closed(self) -> None:
        page = _FakePage()
        assert await _toolset(page).tabs('close') == 'Error: the last tab cannot be closed.'
        assert page.closed is False

    async def test_an_index_with_no_tab_behind_it_is_refused(self) -> None:
        assert await _toolset(_FakePage()).tabs('select', 4) == 'Error: no tab 4. 1 open; list them with tabs.'

    async def test_an_unknown_action_never_reaches_the_browser(self) -> None:
        session = PlaywrightBrowserSession()
        toolset = PlaywrightBrowserToolset[None](session=session)
        result = await toolset.tabs('reorder')
        assert result == "Error: unknown tabs action 'reorder'; use list, select, close or new."
        assert session.page is None

    async def test_the_tab_limit_applies_to_tabs_the_model_opens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            session.pages.extend(_FakePage() for _ in range(toolset_module._MAX_TABS - 1))
            result = await _toolset(page, session=session).tabs('new')
            assert result == f'Error: the tab limit of {toolset_module._MAX_TABS} is reached. Close one first.'
            assert len(session.pages) == toolset_module._MAX_TABS

    async def test_an_active_tab_with_nothing_behind_it_is_reported(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session.pages = []
        toolset = PlaywrightBrowserToolset[None](session=session)
        assert await toolset.tabs('list') == 'No tabs open.'
        assert await toolset.tabs('close') == "Error: the active tab has closed. Open one with tabs('new')."

    async def test_switching_to_a_tab_that_never_fronts_is_bounded(self) -> None:
        # `bring_to_front`, `new_page` and `close` take no timeout of their own, so
        # without the operation's deadline they hold the lock for the rest of the run.
        page = _FakePage()
        session = self._session(page)
        session.pages.append(_HangingTabPage(url='https://example.com/stuck'))
        result = await _toolset(page, session=session, action_timeout_ms=20).tabs('select', 1)
        assert result.startswith('Error: tabs timed out after 20ms.')

    async def test_closing_a_tab_that_never_closes_is_bounded(self) -> None:
        page = _FakePage()
        session = self._session(page)
        session.pages.append(_HangingTabPage(url='https://example.com/stuck'))
        result = await _toolset(page, session=session, action_timeout_ms=20).tabs('close', 1)
        assert result.startswith('Error: tabs timed out after 20ms.')

    async def test_opening_a_tab_that_never_appears_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        session = self._session(page)
        monkeypatch.setattr(session, '_context', _HangingContext())
        result = await _toolset(page, session=session, action_timeout_ms=20).tabs('new')
        assert result.startswith('Error: tabs timed out after 20ms.')

    async def test_a_tab_that_closes_itself_leaves_the_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        popup = _FakePage(url='https://example.com/popup')
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            page.emit('popup', popup)
            session.page = popup
            await popup.close()
            assert session.pages == [page]
            assert session.page is page


class TestDialogs:
    @asynccontextmanager
    async def _launched(
        self, monkeypatch: pytest.MonkeyPatch, page: _FakePage
    ) -> AsyncGenerator[PlaywrightBrowserSession]:
        """A session whose page is wired the way a launch wires it."""
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            yield session

    async def _open(self, session: PlaywrightBrowserSession, page: _FakePage, dialog: _FakeDialog) -> None:
        page.emit('dialog', dialog)
        await asyncio.gather(*session._event_tasks, return_exceptions=True)

    async def test_an_unarmed_dialog_is_dismissed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        dialog = _FakeDialog()
        async with self._launched(monkeypatch, page) as session:
            await self._open(session, page, dialog)
            assert dialog.answer == 'dismiss'
            assert [event.describe() for event in session.events] == [
                '[warning] dialog confirm dismissed: Delete this?'
            ]

    async def test_an_armed_dialog_is_accepted_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        first, second = _FakeDialog(), _FakeDialog()
        async with self._launched(monkeypatch, page) as session:
            await _toolset(page, session=session).handle_next_dialog(accept=True)
            await self._open(session, page, first)
            await self._open(session, page, second)
        assert first.answer == 'accept'
        assert second.answer == 'dismiss'

    async def test_a_prompt_is_answered_with_the_given_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        dialog = _FakeDialog(type='prompt', message='How many?')
        async with self._launched(monkeypatch, page) as session:
            result = await _toolset(page, session=session).handle_next_dialog(accept=True, prompt_text='42')
            assert result == 'The next dialog will be accepted.'
            await self._open(session, page, dialog)
        assert dialog.accepted_with == '42'

    async def test_an_armed_decision_does_not_survive_into_the_next_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arming says what to do about the dialog this run is expecting; a dialog in
        # a later use of the session is not that one.
        page = _FakePage()
        _install_fake_driver(monkeypatch, page)
        session = PlaywrightBrowserSession()
        async with session:
            await session.ensure_page()
            await _toolset(page, session=session).handle_next_dialog(accept=True)
        dialog = _FakeDialog()
        async with session:
            await session.ensure_page()
            await self._open(session, page, dialog)
        assert dialog.answer == 'dismiss'

    async def test_arming_a_dismissal_reports_it(self) -> None:
        page = _FakePage()
        assert await _toolset(page).handle_next_dialog(accept=False) == 'The next dialog will be dismissed.'

    async def test_a_dialog_whose_page_is_gone_does_not_fail_the_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        async with self._launched(monkeypatch, page) as session:
            await self._open(session, page, _FakeUnanswerableDialog())
            assert session._event_tasks == set()


class TestReviewFollowUps:
    """Fixes for findings raised on the pushed branch."""

    async def test_the_renderer_sandbox_is_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        await Agent(TestModel(call_tools=['screenshot']), capabilities=[PlaywrightBrowser()]).run('shot')
        assert cm._driver.chromium.sandboxed == [True]

    async def test_the_renderer_sandbox_can_be_turned_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page)
        capability = PlaywrightBrowser[object](chromium_sandbox=False)
        await Agent(TestModel(call_tools=['screenshot']), capabilities=[capability]).run('shot')
        assert cm._driver.chromium.sandboxed == [False]

    async def test_a_session_entered_again_retries_a_failed_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        _install_fake_driver(monkeypatch, page, executable_missing=True)
        async with session:
            with (
                pytest.warns(toolset_module.BrowserUnavailableWarning),
                pytest.raises(toolset_module.BrowserUnavailableError, match='Chromium is not installed'),
            ):
                await session.ensure_page()
        # The binary is there the second time: the recorded failure described the
        # previous use and must not refuse this one before it tries.
        _install_fake_driver(monkeypatch, page)
        async with session:
            assert await session.ensure_page() is page

    async def test_userinfo_is_stripped_from_the_middle_of_a_message(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session._on_console(_FakeConsoleMessage('error', 'failed https://user:secret@example.com/api'))
        assert await _toolset(page, session=session).console_messages() == (
            '[error] console error: failed https://example.com/api'
        )

    async def test_an_ordinary_sentence_with_an_at_sign_is_left_alone(self) -> None:
        page = _FakePage()
        session = PlaywrightBrowserSession()
        session.page = page
        session._on_console(_FakeConsoleMessage('log', 'see http://example.com/ then mail me@example.com'))
        assert await _toolset(page, session=session).console_messages() == (
            '[info] console log: see http://example.com/ then mail me@example.com'
        )

    async def test_a_browser_installed_mid_run_is_used_by_the_next_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        chromium = _FakeChromium(page, executable_missing=True)
        cm = _FakeDriverCM(_FakeDriver(chromium))
        monkeypatch.setattr(toolset_module, 'async_playwright', lambda: cm)
        session = PlaywrightBrowserSession()
        async with session:
            toolset = PlaywrightBrowserToolset[None](session=session)
            with pytest.warns(BrowserUnavailableWarning):
                assert 'playwright install chromium' in await toolset.get_text()
            # What an agent with a shell does between the two calls.
            chromium._executable_path = sys.executable
            assert await toolset.get_text() == 'Hello body'

    async def test_retrying_a_failed_launch_reuses_the_one_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = _FakePage()
        cm = _install_fake_driver(monkeypatch, page, executable_missing=True)
        session = PlaywrightBrowserSession()
        async with session:
            toolset = PlaywrightBrowserToolset[None](session=session)
            with pytest.warns(BrowserUnavailableWarning):
                await toolset.get_text()
                await toolset.get_text()
        # Each retry re-launches Chromium; re-entering the driver as well would leave
        # the earlier connection running until teardown.
        assert cm.entries == 1

    async def test_a_callback_url_is_redacted_wherever_a_tool_reports_it(self) -> None:
        landing = 'https://example.com/cb?code=secret&state=xyz'
        page = _FakePage(url=landing)
        toolset = _toolset(page)
        redacted = 'https://example.com/cb?code=REDACTED&state=xyz'
        landed = await toolset.navigate(landing)
        assert isinstance(landed, str)
        assert f'URL: {redacted}' in landed
        assert 'secret' not in landed
        assert f'URL: {redacted}' in await toolset.click('a#next')
        assert f'URL: {redacted}' in await toolset.go_back()
        assert redacted in await toolset.tabs('list')
