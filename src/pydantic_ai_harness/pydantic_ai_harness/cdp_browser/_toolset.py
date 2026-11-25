"""The `browser_cdp` tool, and the per-run Chrome behind it."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import Annotated, Self
from urllib.parse import urlparse

import anyio
from pydantic import Field

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._browser_egress import (
    DEFAULT_ALLOWLIST_REACH,
    EgressPolicy,
    EgressRequest,
    refusal_reason,
    request_kind,
)
from pydantic_ai_harness._browser_proxy import PRIVATE_ADDRESS_BLOCK, socks_proxy
from pydantic_ai_harness._typeguards import is_str_keyed_dict
from pydantic_ai_harness.cdp_browser._client import CDPClient, CDPError, CDPSession, launch_chrome

DEFAULT_ALLOWED_CDP_METHODS: tuple[str, ...] = (
    'Page.navigate',
    'Page.reload',
    'Page.getNavigationHistory',
    'Page.navigateToHistoryEntry',
    'Page.getFrameTree',
    'Page.getLayoutMetrics',
    'Page.captureScreenshot',
    'Runtime.evaluate',
    'Runtime.callFunctionOn',
    'Runtime.getProperties',
    'Runtime.releaseObject',
    'DOM.getDocument',
    'DOM.querySelector',
    'DOM.querySelectorAll',
    'DOM.describeNode',
    'DOM.resolveNode',
    'DOM.getOuterHTML',
    'DOM.getBoxModel',
    'DOM.getContentQuads',
    'DOM.scrollIntoViewIfNeeded',
    'DOM.focus',
    'Input.dispatchMouseEvent',
    'Input.dispatchKeyEvent',
    'Input.insertText',
)
"""The CDP commands the model may send by default: reading and interacting with the page.

Anything else is refused, including `Browser.*` and `Target.*` (which reach beyond the tab), `Fetch.*` and
`Network.*` (which could lift the egress guard), and `DOM.setFileInputFiles` and `Input.dispatchDragEvent` (which
take host file paths).
"""

DEFAULT_TIMEOUT_SECONDS = 30.0
"""How long one `browser_cdp` call waits for its command and its awaited event, unless the model passes `timeout`."""

SETUP_TIMEOUT_SECONDS = 10.0
"""How long Chrome gets to attach its first tab once it is running."""

GUARD_ARGS = ('--disable-features=FedCm',)
"""Chrome flags `CDPBrowser` always launches with."""


def proxy_args(socket_port: int, http_port: int | None) -> list[str]:
    """The Chrome flags that send WebSockets to the proxy on `socket_port`, and `http` and `https` to `http_port`.

    No rule names `ws` or `wss`, so Chrome sends them to the `socks` entry. With no `http_port`, `http` and `https` go
    direct. `<-loopback>` removes Chrome's implicit exemption for loopback and link-local hosts, and Chrome has no
    fallback to a direct connection, so a connection a proxy does not relay fails. `scripts/cdp_browser_smoke.py`
    checks this on a real Chrome.
    """
    http = 'direct://' if http_port is None else f'socks5://127.0.0.1:{http_port}'
    return [
        f'--proxy-server=http={http};https={http};socks=socks5://127.0.0.1:{socket_port}',
        '--proxy-bypass-list=<-loopback>',
    ]


MAX_SCREENSHOT_BYTES = 5_000_000
"""Largest screenshot returned as image content, the strictest per-image limit among mainstream model providers.

A larger one becomes an error string instead of a `BinaryContent` that would fail the next model request.
"""

AUTO_ATTACH: dict[str, object] = {'autoAttach': True, 'waitForDebuggerOnStart': True, 'flatten': True}
"""Attach to every new target, held until its requests are guarded, on the one websocket."""

FETCH_EVERY_REQUEST: dict[str, object] = {'patterns': [{'urlPattern': '*', 'requestStage': 'Request'}]}

PAGE_TARGET_TYPES = frozenset({'page', 'iframe'})
"""Target types that load documents, so they get the dialog handler and the service-worker block."""

BLOCK_SERVICE_WORKER_REGISTRATION = (
    'if (self.ServiceWorkerContainer) ServiceWorkerContainer.prototype.register = async () => {'
    " console.warn('Service Worker registration blocked by CDPBrowser'); };"
)
"""Keeps pages from registering service workers, whose traffic would come from a target of their own.

The same script Playwright installs for `new_context(service_workers='block')`.
"""

INSTRUCTIONS = """\
You control a Chrome tab by sending Chrome DevTools Protocol commands with the `browser_cdp` tool.

Pages are big: extract only what you need with `Runtime.evaluate` (with `returnByValue: true`) rather than
returning the whole HTML. Prefer one JavaScript expression that returns structured data, e.g.
`JSON.stringify([...document.querySelectorAll('table tr')].map(...))`. Some pages render content after load,
so if what you need is missing, wait briefly and evaluate again.

`Page.captureScreenshot` returns the image itself. Page dialogs (`alert`, `confirm`, `prompt`) are dismissed.
{reach}: {hosts}. A refused request fails with `net::ERR_BLOCKED_BY_CLIENT`.\
"""


DEFAULT_REACH = (
    "Hosts that top-level navigation and page scripts' data requests (`fetch`, XHR, EventSource, `sendBeacon`, "
    'WebSocket) may reach'
)
"""What the allowlist bounds under `DEFAULT_ALLOWLIST_REACH`; any other reach is named by `EgressPolicy.describe()`."""


@dataclass(frozen=True)
class AttachedTarget:
    """What the egress guard needs to know about a target it attached to."""

    target_id: str
    type: str


def screenshot_return(result: dict[str, object], params: dict[str, object]) -> ToolReturn[str] | str:
    """Return a `Page.captureScreenshot` result as an image, so a vision model sees the page instead of base64."""
    data = str(result.get('data', ''))
    hint = "capture a smaller `clip`, or pass `format: 'jpeg'` with a lower `quality`."
    # The longest base64 an image within the limit encodes to, so this refuses no valid image.
    if len(data) > 4 * ((MAX_SCREENSHOT_BYTES + 2) // 3):
        return f'Error: screenshot is over the {MAX_SCREENSHOT_BYTES} byte image limit; {hint}'
    image = base64.b64decode(data)
    if len(image) > MAX_SCREENSHOT_BYTES:
        return f'Error: screenshot is {len(image)} bytes, over the {MAX_SCREENSHOT_BYTES} byte image limit; {hint}'
    image_format = params.get('format')
    media_type = f'image/{image_format}' if image_format in ('jpeg', 'webp') else 'image/png'
    return ToolReturn(
        return_value=f'Screenshot attached ({media_type}, {len(image)} bytes).',
        content=[BinaryContent(data=image, media_type=media_type)],
    )


class CDPBrowserToolset(FunctionToolset[AgentDepsT]):
    """A toolset with one `browser_cdp` tool, which sends CDP commands to a Chrome tab.

    Each run gets its own throwaway Chrome, launched when the run starts and closed when it ends. Every request
    any of its targets makes is paused until `policy` allows or refuses it.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        chrome_path: str | None = None,
        chrome_sandbox: bool = True,
        allowed_methods: Sequence[str] = DEFAULT_ALLOWED_CDP_METHODS,
        policy: EgressPolicy | None = None,
        id: str | None = None,
    ):
        self.policy = policy if policy is not None else EgressPolicy()
        # CDP mistakes are common and cheap to correct, so allow a few in a row
        super().__init__(id=id, instructions=self._describe_tool, max_retries=3)
        self.headless = headless
        self.chrome_path = chrome_path
        self.chrome_sandbox = chrome_sandbox
        self.allowed_methods = frozenset(allowed_methods)
        self.add_function(self.browser_cdp, takes_ctx=False, name='browser_cdp')
        self._exit_stack: AsyncExitStack | None = None
        self._entered = 0
        self._client: CDPClient | None = None
        self._tab: CDPSession | None = None
        self._first_page: asyncio.Future[CDPSession] | None = None
        self._targets: dict[str, AttachedTarget] = {}

    def _describe_tool(self) -> str:
        """The tool's instructions, read from the policy on each request so they name the hosts the guard allows."""
        default = self.policy.allowlist_reach == DEFAULT_ALLOWLIST_REACH
        reach = DEFAULT_REACH if default else 'Hosts the browser may reach'
        return INSTRUCTIONS.format(reach=reach, hosts=self.policy.describe())

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> CDPBrowserToolset[AgentDepsT]:
        """Give each run a fresh toolset, so concurrent runs never share a browser."""
        return CDPBrowserToolset(
            headless=self.headless,
            chrome_path=self.chrome_path,
            chrome_sandbox=self.chrome_sandbox,
            allowed_methods=tuple(self.allowed_methods),
            policy=self.policy,
            id=self.id,
        )

    async def __aenter__(self) -> Self:
        if self._entered == 0:
            stack = AsyncExitStack()
            try:
                socket_port = await stack.enter_async_context(socks_proxy(self.policy))
                http_port = (
                    await stack.enter_async_context(socks_proxy(PRIVATE_ADDRESS_BLOCK))
                    if self.policy.block_private_addresses
                    else None
                )
                self._client = await stack.enter_async_context(
                    launch_chrome(
                        extra_args=[*proxy_args(socket_port, http_port), *GUARD_ARGS],
                        headless=self.headless,
                        chrome_path=self.chrome_path,
                        sandbox=self.chrome_sandbox,
                    )
                )
                self._tab = await self._open_tab(self._client)
            except BaseException:
                await stack.aclose()
                raise
            self._exit_stack = stack
        self._entered += 1
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        self._entered -= 1
        if self._entered == 0 and self._exit_stack is not None:
            stack, self._exit_stack, self._client, self._tab = self._exit_stack, None, None, None
            self._targets.clear()
            await stack.aclose()
        return None

    async def _open_tab(self, client: CDPClient) -> CDPSession:
        """Guard every target Chrome has or opens, and return the session of the tab it started with."""
        self._first_page = client.session_future()
        client.on('Target.attachedToTarget', self._attached)
        client.on('Fetch.requestPaused', self._request_paused)
        client.on('Page.javascriptDialogOpening', self._dialog_opening)
        try:
            with anyio.fail_after(SETUP_TIMEOUT_SECONDS):
                await client.send('Target.setAutoAttach', AUTO_ATTACH)
                return await self._first_page
        except TimeoutError:
            raise RuntimeError(f'Chrome did not attach its first tab within {SETUP_TIMEOUT_SECONDS:g}s') from None

    async def _attached(self, _parent_session_id: str | None, params: dict[str, object]) -> None:
        """Pause a new target's requests for the egress guard, then let it run.

        A dedicated worker has no `Fetch` domain of its own: its requests are paused on the session of the page that
        started it. Any other target whose requests cannot be paused is never resumed, so it makes none: the guard
        fails closed.
        """
        session_id = params.get('sessionId')
        info = params.get('targetInfo')
        if self._client is None or not isinstance(session_id, str) or not is_str_keyed_dict(info):
            return
        target = AttachedTarget(target_id=str(info.get('targetId')), type=str(info.get('type')))
        self._targets[session_id] = target
        session = CDPSession(self._client, session_id)
        if target.type != 'worker':
            await session.send('Fetch.enable', FETCH_EVERY_REQUEST)
        if target.type in PAGE_TARGET_TYPES:
            await session.send('Page.enable')
            await session.send('Page.addScriptToEvaluateOnNewDocument', {'source': BLOCK_SERVICE_WORKER_REGISTRATION})
        await session.send('Target.setAutoAttach', AUTO_ATTACH)
        await session.send('Runtime.runIfWaitingForDebugger')
        if target.type == 'page' and self._first_page is not None and not self._first_page.done():
            self._first_page.set_result(session)

    async def _request_paused(self, session_id: str | None, params: dict[str, object]) -> None:
        """Let a paused request through, or fail it, as the egress policy decides."""
        request_id = params.get('requestId')
        request = params.get('request')
        if self._client is None or session_id is None or not isinstance(request_id, str):
            return
        if not is_str_keyed_dict(request):
            return
        resource_type = str(params.get('resourceType', 'other')).lower()
        target = self._targets.get(session_id)
        top_level = target is not None and target.type == 'page' and params.get('frameId') == target.target_id
        egress = EgressRequest(
            url=str(request.get('url')),
            kind=request_kind(resource_type, is_navigation=resource_type == 'document', top_level=top_level),
            resource_type=resource_type,
            method=str(request.get('method', 'GET')),
            top_level=top_level,
        )
        session = CDPSession(self._client, session_id)
        if await refusal_reason(self.policy, egress) is None:
            await session.send('Fetch.continueRequest', {'requestId': request_id})
        else:
            await session.send('Fetch.failRequest', {'requestId': request_id, 'errorReason': 'BlockedByClient'})

    async def _dialog_opening(self, session_id: str | None, params: dict[str, object]) -> None:
        """Dismiss a dialog, which would otherwise block the page until something answers it.

        A `beforeunload` prompt is accepted instead, so the navigation that raised it goes through.
        """
        if self._client is None or session_id is None:
            return
        accept = params.get('type') == 'beforeunload'
        await self._client.send('Page.handleJavaScriptDialog', {'accept': accept}, session_id)

    async def browser_cdp(
        self,
        method: str,
        params: dict[str, object] | None = None,
        wait_for_event: str | None = None,
        timeout: Annotated[float, Field(gt=0)] = DEFAULT_TIMEOUT_SECONDS,
    ) -> object:
        r"""Send one Chrome DevTools Protocol command to the open tab and return its result.

        Only commands for reading and interacting with the page are allowed, and `Page.navigate` only
        opens `http://` and `https://` URLs.

        Examples:
            `browser_cdp(method='Page.navigate', params={'url': url}, wait_for_event='Page.loadEventFired')`
            `browser_cdp(method='Runtime.evaluate', params={'expression': 'document.title', 'returnByValue': True})`
            `browser_cdp(method='Input.dispatchMouseEvent', params={'type': 'mousePressed', 'x': 10, 'y': 20,
            'button': 'left', 'clickCount': 1})`
            `browser_cdp(method='Input.dispatchKeyEvent', params={'type': 'keyDown', 'key': 'Enter', 'text': '\r'})`

        Args:
            method: A CDP command, e.g. `'Runtime.evaluate'`.
            params: The command's parameters.
            wait_for_event: A CDP event, e.g. `'Page.loadEventFired'`, to wait for after sending the command.
                The return value is then `{'result': <command result>, 'event': <event params>}`.
            timeout: Seconds to wait for the command and, when given, `wait_for_event`.
        """
        params = params or {}
        if method not in self.allowed_methods:
            raise ModelRetry(f'{method} is not allowed, allowed commands: {", ".join(sorted(self.allowed_methods))}')
        if method == 'Page.navigate':
            await self._check_navigation(str(params.get('url')))
        tab = self._tab
        if tab is None:
            raise RuntimeError('`CDPBrowserToolset` must be entered (by an agent run) before its tool is called')
        event = tab.wait_for_event(wait_for_event) if wait_for_event is not None else None
        result: dict[str, object] | None = None
        event_params: dict[str, object] | None = None
        try:
            with anyio.fail_after(timeout):
                result = await tab.send(method, params)
                if event is not None:
                    event_params = await event
        except TimeoutError:
            if result is None:
                await self._stop_page_script(tab)
                raise ModelRetry(
                    f'{method} did not answer within {timeout:g}s, so the page script was stopped; '
                    'try a simpler command or a longer timeout'
                ) from None
            raise ModelRetry(
                f'Timed out after {timeout:g}s waiting for {wait_for_event!r}; the command returned {result!r}'
            ) from None
        except CDPError as e:
            raise ModelRetry(str(e)) from None
        finally:
            if event is not None:
                event.cancel()
        if method == 'Page.captureScreenshot':
            return screenshot_return(result, params)
        return result if event is None else {'result': result, 'event': event_params}

    async def _check_navigation(self, url: str) -> None:
        """Refuse a navigation before it is sent, so the model learns why rather than reading a network error."""
        try:
            scheme = urlparse(url).scheme
        except ValueError:
            raise ModelRetry(f'{url!r} is not a valid URL') from None
        if scheme not in ('http', 'https'):
            raise ModelRetry(f'Only http and https URLs can be opened, got {url!r}')
        if (reason := await refusal_reason(self.policy, EgressRequest(url=url, kind='navigation'))) is not None:
            raise ModelRetry(f'{url!r} is refused by the egress policy: {reason}')

    async def _stop_page_script(self, tab: CDPSession) -> None:
        """Stop the page's running script, such as `while (true) {}`, so the tab answers the next command."""
        with anyio.move_on_after(SETUP_TIMEOUT_SECONDS), suppress(CDPError):
            await tab.send('Runtime.terminateExecution')
