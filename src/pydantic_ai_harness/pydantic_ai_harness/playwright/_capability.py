"""Playwright capability -- a real, stateful Chromium browser for agents.

The dated external-service assumptions this package relies on (aria snapshots,
selector engines, service-worker blocking, binary detection, timeouts) are
recorded in the `_toolset` module docstring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.playwright._toolset import (
    DEFAULT_ACTION_TIMEOUT_MS,
    DEFAULT_MAX_CONTENT_TOKENS,
    DEFAULT_NAVIGATION_TIMEOUT_MS,
    CdpContextMode,
    EgressPolicy,
    PlaywrightBrowserSession,
    PlaywrightBrowserToolset,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from playwright.async_api import StorageState

    from pydantic_ai.agent import AbstractAgent

_INSTRUCTIONS = """\
You have a real web browser powered by Playwright. Use it for pages the lighter tools cannot handle:
pages behind login or session cookies, JavaScript-heavy SPAs, interactive multi-step flows (clicking,
filling forms), and dynamically loaded content. For looking up information or reading a static, public
URL, prefer web search or web fetch.

Read: `navigate(url)`, `snapshot()`, `get_text(selector?)`, `screenshot(full_page?)`.
Act: `click(selector)` (CSS selector, 'x,y' pixel coordinates, or an `aria-ref=` handle from
`snapshot`), `type_text(selector, text, sequential?)`, `press_key(key, selector?)`,
`select_option(selector, values)`, `hover(selector)`, `scroll(direction)`,
`wait_for(selector?, text?, gone?)`, `go_back()`, `go_forward()`, `execute_js(script)`.
Manage: `tabs(action, index?)`, `handle_next_dialog(accept, prompt_text?)`.
Inspect: `console_messages(errors_only?)`, `network_requests(url_contains?, errors_only?)`.
Every page action takes an optional `timeout_ms` override.

Prefer `snapshot` to read the page structure and obtain `aria-ref` handles, then target elements by
`aria-ref=` for reliable clicks. `type_text` fills a field but does not submit: use `press_key('Enter')`
for a search box or form, and `sequential=True` for an autocomplete or masked field that reacts to
each keystroke. Use `wait_for` for content that loads after an action and `wait_for(gone=True)` to
wait out a spinner or overlay, and `screenshot` only for visual checks (charts, layout). A page that
renders after it loads returns little text at first: read it again with `snapshot` or `get_text`, or
`wait_for` something a snapshot showed you -- waiting for text you have only guessed at spends the
whole timeout. A long list needs `scroll('down')` repeated, collecting what you need from each result as you
go: `scroll` reports where the page now sits, and a feed that renders only the rows near the
viewport drops the earlier ones, so jumping to `bottom` and reading once returns a fraction of it.

Tools act on the active tab. When a link, sign-in popup or payment step opens a new one, `tabs('list')`
shows what is open and `tabs('select', index)` moves there. A page dialog (`alert`, `confirm`,
`prompt`) is dismissed unless you call `handle_next_dialog(accept=True)` before the action that
opens it.

When page text looks empty or is missing what you expect, the content is probably inside an iframe
(embedded schedules, checkout steps, chat widgets). Call `snapshot`: refs inside an embed look like
`f1e4` and work with `click`, `type_text`, `hover` and `get_text`, while a CSS selector does not reach
there. If a page renders from an API, `network_requests` finds the request holding the data, which is
often easier to read than the DOM.

Textual tool results are truncated to roughly {max_content_tokens} tokens; use `get_text` with a
selector to read a specific section of a large page. Allowed domains: {allowed_domains}.
"""


@dataclass
class PlaywrightBrowser(AbstractCapability[AgentDepsT]):
    """A real, stateful Chromium browser for an agent, via async Playwright.

    Adds eighteen tools -- navigate, snapshot, click, type_text, press_key,
    select_option, hover, wait_for, screenshot, get_text, scroll, go_back,
    go_forward, execute_js, tabs, handle_next_dialog, console_messages,
    network_requests -- backed by a Chromium context that persists across tool
    calls within a run. Reach for it when
    the lighter web tools fall short: pages behind login/session cookies,
    JavaScript-rendered SPAs, and interactive multi-step flows. For query-based
    research prefer a web-search tool; for a static URL prefer a web-fetch tool.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.playwright import PlaywrightBrowser

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[PlaywrightBrowser()])
    ```

    Requires the `playwright` optional extra and the Chromium binary:

    ```bash
    pip install 'pydantic-ai-harness[playwright]'
    playwright install chromium
    ```

    Egress: by default the agent can reach any public host; `allowed_domains`
    bounds that, and `block_private_addresses` (on by default) refuses private,
    loopback, link-local and other reserved addresses even under open egress.
    Neither is a general security boundary: an unanswered DNS lookup is refused
    but Chromium resolves the name again before it connects, so rebinding is not
    closed, and a proxy-based enforcement mode is tracked in
    https://github.com/pydantic/pydantic-ai/issues/9204. Set
    `allowed_domains` when the agent may act on untrusted input.

    The first browser-tool call launches Chromium or connects through CDP. At
    run end, a local browser is closed or a CDP connection is disconnected. Runs
    without browser calls start neither. When no browser can be started the tool returns a
    `playwright install chromium` hint instead of ending the run, so an agent that
    can run a shell can install it and carry on, and the process gets a
    `BrowserUnavailableWarning` for the developer who is not reading tool results.
    Set `auto_install_chromium=True` to fetch the binary automatically instead.
    Set `cdp_url` to attach to a Chromium that is already running
    (managed-browser providers, benchmark harnesses) rather than launching one.

    Every browser operation runs inside an OpenTelemetry span, and what the page
    did during it (console output, responses, requests the egress policy refused,
    dialogs it opened, tabs it opened) is attached to that span as span events and
    readable by the agent through `console_messages` and `network_requests`.

    Durable execution (e.g. `TemporalDurability`) is not supported: durability
    replays tool calls as activities and a live Chromium page cannot survive
    replay or worker restart, so combining both on one agent, or wrapping the
    agent in a deprecated wrapper agent such as `TemporalAgent`, raises
    `UserError` at construction.
    """

    _durable_unsupported_reason: ClassVar[str | None] = (
        'a live Chromium browser cannot survive activity replay or worker restart'
    )

    headless: bool = True
    """Run Chromium without a visible window. `True` suits servers and CI."""

    allowed_domains: list[str] | None = None
    """Egress allowlist. `None` (default) allows every public host -- see the egress note above.

    Each entry matches its exact host and any subdomain, so `example.com` reaches
    `api.example.com`. It bounds top-level navigation and the requests a page's
    scripts use to move data (`fetch`, XHR, EventSource, WebSocket, `sendBeacon`);
    passive subresources and sub-frame documents are left alone, so a permitted
    page keeps its CDN assets and its identity-provider frames. Enforced at two
    layers: a network route guard aborts a refused request before it leaves, and
    each tool re-checks the resulting URL and bounces to `about:blank` so
    disallowed content never reaches the model.

    For anything this does not express -- a denylist, apex-only matching, locking
    down every request type, a rule of your own -- pass an `EgressPolicy` as
    `policy` instead.
    """

    policy: EgressPolicy | None = None
    """Full egress policy, for rules the two shorthands above cannot express.

    Mutually exclusive with `allowed_domains` and `block_private_addresses`, which
    build the default policy when this is unset. Subclass `EgressPolicy` and
    override `refuse` for a decision the fields do not cover. Absent from
    `from_spec`, since a policy can carry arbitrary code.
    """

    block_private_addresses: bool = True
    """Refuse addresses that are not globally routable, for every request kind in every frame.

    Covers the cloud metadata endpoint (`169.254.169.254`), loopback
    (`127.0.0.1`, `::1`, `localhost`), and the RFC 1918 ranges, independent of
    `allowed_domains` -- open egress still cannot reach them. A hostname is
    resolved first, so a name pointing at one of those addresses is refused too,
    as is one whose lookup does not answer (see the egress note above). Set
    `False` when the agent should reach a local app or an internal dashboard.
    """

    screenshot_on_navigate: bool = False
    """Attach a screenshot (as image content) to every `navigate` result."""

    max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS
    """Approximate token budget for textual tool results."""

    action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS
    """Default deadline for element actions (click, type, read, wait), in milliseconds.

    Shorter than the navigation budget on purpose. An action that misses is
    normally a selector matching nothing, and a long deadline makes that look
    like a hung agent rather than a fast failure the model can react to. Raise it
    for pages whose elements appear slowly.
    """

    navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS
    """Default deadline for navigation and load settling, and for starting or attaching to the browser, in milliseconds."""

    chromium_sandbox: bool = True
    """Run the launched Chromium with its renderer sandbox.

    On by default, unlike Playwright itself: this capability opens pages nobody
    vetted, and the sandbox is what keeps a renderer that a crafted page
    compromises from reaching the host. Set `False` where the sandbox cannot
    start -- a container without the kernel privileges it needs is the usual case
    -- and accept that a renderer exploit then runs with the agent's own access.
    Ignored when `cdp_url` is set: that browser is already running under its own
    configuration.
    """

    auto_install_chromium: bool = False
    """Fetch the Chromium binary via `playwright install chromium` on the first miss.

    Off by default: a library should not download a browser as a side effect. When
    the binary is missing the browser tool returns a clear install hint, which an
    agent that can run a shell can act on. Set `True` to opt into the automatic
    download instead.
    """

    storage_state: StorageState | None = field(default=None, repr=False)
    """Playwright storage state (cookies, localStorage) loaded into the browser context at launch.

    Obtain it in your own code -- `await context.storage_state()`, or
    `json.loads(Path('auth.json').read_text())` for a file written by
    `playwright codegen --save-storage` -- so the login runs where you control
    it and the credentials never have to reach wherever the agent runs. This is
    session material equivalent to the account: keep it out of source control
    and out of logs, and discard it when the session expires.

    An object rather than a path so the capability never assumes a filesystem it
    can read: agents often run where the local disk is neither durable nor
    writable, so where the state lives stays the caller's decision. For the same
    reason `from_spec` does not accept it.
    """

    cdp_url: str | Callable[[RunContext[AgentDepsT]], str | Awaitable[str]] | None = field(default=None, repr=False)
    """Attach to a Chromium already running at this Chrome DevTools Protocol endpoint, or a function that returns one.

    No local Chromium binary is needed; `auto_install_chromium` does not apply.
    Each run creates an isolated context unless `cdp_context='default'` is set.
    Endpoints can contain authentication tokens; treat them as secrets.

    A function of the run context, sync or async, can create or resume a
    managed browser for each run. The first browser tool call of a run calls it,
    and its endpoint is reused for the rest of that run, including connection
    retries; a run without browser calls never calls it. `navigation_timeout_ms`
    bounds an async call: one that times out is a tool error, and the next
    browser tool call tries again. An exception the function raises ends the
    run. At run end the capability only disconnects, so a browser the function
    created is the app's to delete, for example from a handle it kept on
    `ctx.deps`. A function cannot be supplied through a serializable spec.
    """

    _: KW_ONLY
    cdp_context: CdpContextMode = 'new'
    """Create an isolated context, or use the CDP browser's existing default context.

    `default` requires `cdp_url` and excludes `storage_state`.
    It inherits saved sign-ins and open tabs, so the agent can act as every account
    signed in to that browser: attach it to a profile dedicated to the agent, not
    an everyday browser. Only tabs showing an `http(s)` page, `about:blank` or
    Chromium's error page are adopted: any other tab, such as the browser's own
    start page, is never listed, read or closed, whatever the policy, and neither
    is a tab holding a frame outside the web other than an inline one, such as an
    extension's injected UI. When a tracked tab moves to such a page or gets such
    a frame, the call that finds it returns an error naming the page by its scheme
    and host only, and the agent stops using that tab, which stays open in the
    browser. A frame that appears during a call can still be captured by that
    call's `snapshot` or `screenshot_on_navigate` image. Loaded tab and frame URLs must pass the egress policy
    before handlers are attached and before tools read or act on a page. A
    document that passed is not checked again until its frame navigates, and one
    call spends at most five seconds checking: a call that runs out asks the model
    to retry, and the retry picks up where it stopped. Egress
    guards and native session handlers still apply, and documents loaded while
    attached cannot register service workers. Downloads follow the browser's own
    settings. Requests from workers registered before attachment go through the
    guards too, unless `PLAYWRIGHT_DISABLE_SERVICE_WORKER_NETWORK` turns off
    Playwright's worker routing, and a document loaded before attachment can still
    register one until it navigates. Script that ran before attachment can also
    keep the native `WebSocket` constructor, and a socket it opens through it
    bypasses the guards.
    """

    id: str | None = 'playwright_browser'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    An agent has one set of browser tools, so two `PlaywrightBrowser` sharing this `id` are one browser
    stated twice when they agree, and an error when they differ, since a merge could widen the egress one
    of them was given. To keep two, give each its own `id` and wrap them in `PrefixTools`, since their
    tool names are the same.
    """

    # Built from the fields above in `__post_init__`, so they're left out of equality: two browsers configured
    # the same way compare equal, which is what lets `combine` treat them as one.
    _session: PlaywrightBrowserSession = field(init=False, repr=False, compare=False)
    _toolset: PlaywrightBrowserToolset[AgentDepsT] = field(init=False, repr=False, compare=False)
    _policy: EgressPolicy = field(init=False, repr=False, compare=False)
    _run_ctx: RunContext[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.policy is not None and (self.allowed_domains is not None or not self.block_private_addresses):
            raise UserError(
                'Pass either policy or the allowed_domains/block_private_addresses shorthands, not both: '
                'set those fields on the policy instead, so one object decides.'
            )
        self._policy = self.policy or EgressPolicy(
            allowed_domains=self.allowed_domains, block_private_addresses=self.block_private_addresses
        )
        self._session = PlaywrightBrowserSession(
            policy=self._policy,
            headless=self.headless,
            storage_state=self.storage_state,
            cdp_url=self._call_cdp_url if callable(self.cdp_url) else self.cdp_url,
            cdp_context=self.cdp_context,
            auto_install_chromium=self.auto_install_chromium,
            chromium_sandbox=self.chromium_sandbox,
            launch_timeout_ms=self.navigation_timeout_ms,
        )
        self._toolset = PlaywrightBrowserToolset[AgentDepsT](
            session=self._session,
            screenshot_on_navigate=self.screenshot_on_navigate,
            max_content_tokens=self.max_content_tokens,
            action_timeout_ms=self.action_timeout_ms,
            navigation_timeout_ms=self.navigation_timeout_ms,
        )

    def _call_cdp_url(self) -> str | Awaitable[str]:
        """Call the `cdp_url` function with the context of the run this instance serves."""
        assert callable(self.cdp_url) and self._run_ctx is not None
        return self.cdp_url(self._run_ctx)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one browser stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    def for_agent(self, agent: AbstractAgent[AgentDepsT, object]) -> AbstractCapability[AgentDepsT]:
        """Refuse to bind to a durable-execution agent.

        Durable execution (e.g. `TemporalDurability`) wraps the toolsets captured
        at agent construction and replays tool calls as activities. A live
        Chromium page cannot be checkpointed across activity boundaries or worker
        restarts, so the composition cannot work; without this guard it fails
        deep inside the first browser tool call instead.

        Detection matches `BaseDurabilityCapability`, the shared base of the
        bundled Temporal/DBOS/Prefect integrations. Pydantic AI exposes no public
        marker for the durability tier, and the `innermost` ordering position is
        not one: `InputGuard` also declares `innermost`, so ordering alone would
        reject the supported guard-plus-browser composition. The deprecated
        wrapper agents add no durability capability, so they refuse the browser
        themselves, from `_durable_unsupported_reason`.
        """
        siblings: list[AbstractCapability[AgentDepsT]] = []
        agent.root_capability.apply(siblings.append)
        for sibling in siblings:
            if isinstance(sibling, BaseDurabilityCapability):
                raise UserError(
                    'PlaywrightBrowser does not support durable execution (e.g. TemporalDurability): '
                    f'{self._durable_unsupported_reason}. '
                    'Run the browser agent outside the durable agent, or remove the durability capability.'
                )
        return self

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> PlaywrightBrowser[AgentDepsT]:
        """Create per-run connection state; a default CDP context may share pages with other runs."""
        return replace(self)

    def get_toolset(self) -> PlaywrightBrowserToolset[AgentDepsT]:
        """Provide the eighteen browser tools."""
        return self._toolset

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str | None]:
        """When-to-use guidance for the browser."""

        def _instructions(ctx: RunContext[AgentDepsT]) -> str | None:
            return _INSTRUCTIONS.format(
                max_content_tokens=self.max_content_tokens, allowed_domains=self._policy.describe()
            )

        return _instructions

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Hold the run's browser session open, and release it however the run ends.

        Chromium starts on the first browser-tool call, not here, so a run that
        never browses never launches one. The run's tracer is adopted here so
        browser spans follow the agent's instrumentation settings rather than a
        tracer of this module's choosing, and its context is what a `cdp_url`
        function receives.
        """
        self._session.tracer = ctx.tracer
        self._run_ctx = ctx
        async with self._session:
            return await handler()

    @classmethod
    def from_spec(
        cls,
        *,
        headless: bool = True,
        allowed_domains: list[str] | None = None,
        block_private_addresses: bool = True,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
        auto_install_chromium: bool = False,
        chromium_sandbox: bool = True,
        cdp_url: str | None = None,
        cdp_context: CdpContextMode = 'new',
        id: str | None = 'playwright_browser',
        description: str | None = None,
        defer_loading: bool = False,
    ) -> PlaywrightBrowser[AgentDepsT]:
        """Construct the capability from serializable spec options (all fields are plain scalars/lists).

        A spec carries connection configuration, not session credentials:
        `storage_state` is deliberately absent, so a spec naming it raises rather
        than moving cookies into whatever stores the spec. Set it on the
        constructed capability instead. `cdp_context='default'` is configuration
        too: it selects the browser's signed-in context without putting its
        credentials in the spec.
        """
        return cls(
            headless=headless,
            allowed_domains=list(allowed_domains) if allowed_domains is not None else None,
            block_private_addresses=block_private_addresses,
            screenshot_on_navigate=screenshot_on_navigate,
            max_content_tokens=max_content_tokens,
            action_timeout_ms=action_timeout_ms,
            navigation_timeout_ms=navigation_timeout_ms,
            auto_install_chromium=auto_install_chromium,
            chromium_sandbox=chromium_sandbox,
            cdp_url=cdp_url,
            cdp_context=cdp_context,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )
