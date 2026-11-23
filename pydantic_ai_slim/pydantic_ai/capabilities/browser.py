from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import KW_ONLY, dataclass
from typing import Any, Self
from urllib.parse import urlparse

import anyio

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from .abstract import AbstractCapability

__all__ = ('Browser', 'BrowserToolset', 'DEFAULT_ALLOWED_CDP_METHODS')

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

Anything else is refused, including `Browser.*` and `Target.*` (which reach beyond the tab), and
`DOM.setFileInputFiles` and `Input.dispatchDragEvent` (which take host file paths).
"""

BROWSER_INSTRUCTIONS = """\
You control a Chrome tab by sending Chrome DevTools Protocol commands with the `browser_cdp` tool.

Pages are big: extract only what you need with `Runtime.evaluate` (with `returnByValue: true`) rather than
returning the whole HTML. Prefer one JavaScript expression that returns structured data, e.g.
`JSON.stringify([...document.querySelectorAll('table tr')].map(...))`. Some pages render content after load,
so if what you need is missing, wait briefly and evaluate again.\
"""


class BrowserToolset(FunctionToolset[AgentDepsT]):
    """A toolset with one `browser_cdp` tool, which sends CDP commands to a Chrome tab.

    Each run gets its own throwaway Chrome, launched when the run starts and closed when it ends.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        chrome_path: str | None = None,
        allowed_methods: Sequence[str] = DEFAULT_ALLOWED_CDP_METHODS,
        id: str | None = None,
    ):
        # CDP mistakes are common and cheap to correct, so allow a few in a row
        super().__init__(id=id, instructions=BROWSER_INSTRUCTIONS, max_retries=3)
        self.headless = headless
        self.chrome_path = chrome_path
        self.allowed_methods = frozenset(allowed_methods)
        self.add_function(self.browser_cdp, takes_ctx=False, name='browser_cdp')
        self._exit_stack: AsyncExitStack | None = None
        self._entered = 0
        self._session: Any = None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> BrowserToolset[AgentDepsT]:
        # a fresh toolset, so concurrent runs never share a browser
        return BrowserToolset(
            headless=self.headless,
            chrome_path=self.chrome_path,
            allowed_methods=tuple(self.allowed_methods),
            id=self.id,
        )

    async def __aenter__(self) -> Self:
        if self._entered == 0:
            from pydantic_ai._cdp import launch_chrome

            stack = AsyncExitStack()
            try:
                self._session = await stack.enter_async_context(
                    launch_chrome(headless=self.headless, chrome_path=self.chrome_path)
                )
            except BaseException:
                await stack.aclose()
                raise
            self._exit_stack = stack
        self._entered += 1
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        self._entered -= 1
        if self._entered == 0 and self._exit_stack is not None:
            stack, self._exit_stack, self._session = self._exit_stack, None, None
            await stack.aclose()
        return None

    async def browser_cdp(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        wait_for_event: str | None = None,
        timeout: float = 30,
    ) -> Any:
        """Send one Chrome DevTools Protocol command to the open tab and return its result.

        Only commands for reading and interacting with the page are allowed, and `Page.navigate` only
        opens `http://` and `https://` URLs.

        Examples:
            `browser_cdp(method='Page.navigate', params={'url': url}, wait_for_event='Page.loadEventFired')`
            `browser_cdp(method='Runtime.evaluate', params={'expression': 'document.title', 'returnByValue': True})`
            `browser_cdp(method='Input.dispatchMouseEvent', params={'type': 'mousePressed', 'x': 10, 'y': 20, 'button': 'left', 'clickCount': 1})`

        Args:
            method: A CDP command, e.g. `'Runtime.evaluate'`.
            params: The command's parameters.
            wait_for_event: A CDP event, e.g. `'Page.loadEventFired'`, to wait for after sending the command.
                The return value is then `{'result': <command result>, 'event': <event params>}`.
            timeout: Seconds to wait for `wait_for_event`.
        """
        params = params or {}
        if method not in self.allowed_methods:
            raise ModelRetry(f'{method} is not allowed, allowed commands: {", ".join(sorted(self.allowed_methods))}')
        if method == 'Page.navigate' and urlparse(url := str(params.get('url'))).scheme not in ('http', 'https'):
            raise ModelRetry(f'Only http and https URLs can be opened, got {url!r}')
        session = self._session
        if session is None:
            raise RuntimeError('`BrowserToolset` must be entered (by an agent run) before its tool is called')
        if wait_for_event is None:
            return await session.send(method, params)
        event = session.wait_for_event(wait_for_event)
        result = await session.send(method, params)
        try:
            with anyio.fail_after(timeout):
                event_params = await event
        except TimeoutError:
            raise ModelRetry(
                f'Timed out after {timeout}s waiting for {wait_for_event!r}; the command returned {result!r}'
            )
        return {'result': result, 'event': event_params}


@dataclass
class Browser(AbstractCapability[AgentDepsT]):
    """Lets the agent drive a Chrome tab by sending raw [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/) commands.

    Each run launches its own Chrome with a throwaway profile, and closes it when the run ends.
    Requires Chrome or Chromium to be installed, and the `browser` optional group:

    ```bash
    pip install "pydantic-ai-slim[browser]"
    ```

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Browser

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[Browser()])
    result = agent.run_sync('What is the top story on news.ycombinator.com right now?')
    ```

    Only the commands in `allowed_methods` can be sent; the defaults read and interact with the page,
    but can't reach the host filesystem or other tabs. The page itself can still reach anything Chrome
    can reach on the network, including services on the host.
    """

    _: KW_ONLY

    headless: bool = True
    """Run Chrome without a window."""

    chrome_path: str | None = None
    """The Chrome or Chromium binary. Defaults to `CHROME_PATH`, then the usual install locations."""

    allowed_methods: Sequence[str] = DEFAULT_ALLOWED_CDP_METHODS
    """The CDP commands the model may send. Defaults to `pydantic_ai.capabilities.browser.DEFAULT_ALLOWED_CDP_METHODS`."""

    id: str | None = 'browser'
    """Fixed, so a later `Browser` replaces an earlier one; pass distinct ids (and wrap one in `PrefixTools`) to keep both."""

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        return capabilities[-1]

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return BrowserToolset(
            headless=self.headless, chrome_path=self.chrome_path, allowed_methods=self.allowed_methods, id=self.id
        )
