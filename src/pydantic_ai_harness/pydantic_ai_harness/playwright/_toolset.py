"""Playwright toolset -- gives an agent a real, stateful Chromium browser.

External-service assumptions (Playwright SDK + bundled Chromium + `playwright` CLI).
These depend on Playwright internals and Chromium packaging that change on the
vendor's schedule and that the mocked tests do not exercise. Re-verify against the
installed package and the linked sources before changing version, selector, or
teardown handling; bump the date when a fact still holds, update code and date
together when it changed. Pinned in the `[playwright]` extra as
`playwright>=1.61.0` (pyproject.toml). Every fact below was verified against
1.61.0, and the signature-checkable ones re-verified against 1.62.0 on 2026-08-18.

- `page.aria_snapshot(mode='ai')` returns an agent-oriented tree whose nodes carry
  `[ref=eN]` handles. `mode` accepts `Literal['ai', 'default'] | None` through
  1.62.0; the `ref` attributes have shipped since Playwright 1.52. Verified
  2026-08-18.
  Source: <https://playwright.dev/python/docs/aria-snapshots>. Re-check:
  `inspect.signature(playwright.async_api.Page.aria_snapshot)` still offers 'ai'.
- The `aria-ref=eN` handles from that snapshot are resolvable by the `aria-ref=`
  selector engine, so they can be passed straight to `page.click` / `page.fill`.
  Verified 2026-08-18 (engine present in the bundled driver `coreBundle.js`).
  Source: <https://playwright.dev/python/docs/other-locators>. Re-check: pass a
  `snapshot` ref back into `click` against a live page, or grep the installed
  driver bundle for `aria-ref`.
- `browser.new_context(service_workers='block')` disables page service workers;
  the option is `Literal['allow', 'block'] | None` through 1.62.0. Verified
  2026-08-18.
  Source: <https://playwright.dev/python/docs/api/class-browsercontext>
  (`serviceWorkers` option). Re-check: inspect the `service_workers` parameter of
  `Browser.new_context`.
- `new_context` accepts downloads by default; `accept_downloads=False` refuses
  them, so a page cannot write attachments to the host's temporary storage.
  Verified 2026-08-18 (parameter documented as "Defaults to `true` where all the
  downloads are accepted"). Source:
  <https://playwright.dev/python/docs/api/class-browser#browser-new-context>.
  Re-check: `inspect.getdoc(Browser.new_context)` still states that default.
- `TargetClosedError` is not re-exported from `playwright.async_api` through
  1.62.0; it lives at `playwright._impl._errors`. A driver-raised instance only
  carries `.name`, so `isinstance` is the reliable discriminator. Verified
  2026-08-18.
  Source: <https://github.com/microsoft/playwright-python> (async_api `__init__`).
  Re-check: `hasattr(playwright.async_api, 'TargetClosedError')` (expect `False`);
  if it becomes `True`, switch to the public import.
- Missing-binary detection uses `chromium.executable_path` plus an on-disk
  `os.path.exists` check; the install command is `python -m playwright install
  chromium`. Verified 2026-08-18. Source:
  <https://playwright.dev/python/docs/browsers#install-browsers>. Re-check:
  confirm `BrowserType.executable_path` exists and `playwright install chromium`
  still fetches the binary.
- Playwright's own default action/navigation timeout is a single 30000ms, and
  `timeout=0` disables the deadline (the toolset treats 0 the same way in
  `_await_with_timeout`). The defaults here deliberately split that number in two:
  a missed element is usually a wrong selector and should fail fast, while a page
  load legitimately takes longer. Verified 2026-08-18
  (`DEFAULT_PLAYWRIGHT_TIMEOUT_IN_MILLISECONDS = 30000` in `_impl/_helper.py`).
  Source: <https://playwright.dev/python/docs/api/class-page#page-set-default-timeout>.
  Re-check: grep the installed package for `DEFAULT_PLAYWRIGHT_TIMEOUT`.
- `aria-ref=fNeM` handles from a `mode='ai'` snapshot resolve inside the child
  frame they came from: the ref carries the frame sequence and the driver jumps to
  that frame before matching (`_jumpToAriaRefFrameIfNeeded` in the bundled
  driver). Plain CSS selectors do not cross frames, so a snapshot ref is the only
  handle that reaches embedded content. Verified 2026-08-18 against a real
  Chromium (read and click inside a cross-origin child frame).
  Source: <https://playwright.dev/python/docs/other-locators>. Re-check: grep the
  driver bundle for `_jumpToAriaRefFrameIfNeeded`.
- A request aborted by a context route reaches `requestfailed` with the failure
  text `net::ERR_FAILED`, which is what lets the guard's own entry (carrying the
  reason) stand alone in the event log. Verified 2026-08-18 against a real
  Chromium: a refused navigation produced exactly one recorded event. Source:
  <https://playwright.dev/python/docs/api/class-route#route-abort> (`errorCode`
  defaults to `failed`). Re-check: the private-address scenario in
  `scripts/playwright_smoke.py` asserts the single entry.
- Attaching any `page.on('dialog')` handler takes Playwright out of its
  auto-dismiss behavior: the dialog then blocks the page until the handler calls
  `accept`/`dismiss`, so every path through the handler has to answer. Verified
  2026-08-18. Source: <https://playwright.dev/python/docs/dialogs>. Re-check: the
  dialog scenario in `scripts/playwright_smoke.py`.
- A page a site opens (`window.open`, `target="_blank"`) arrives on the opener's
  `popup` event and belongs to the same `BrowserContext`, so the context's route
  guard and storage state already cover it. Its `url` at event time is usually
  still `about:blank`. Verified 2026-08-18. Source:
  <https://playwright.dev/python/docs/pages#handling-new-pages>. Re-check: the
  tab scenario in `scripts/playwright_smoke.py`.
- `locator.press_sequentially(text, timeout=...)` types key by key, unlike
  `page.fill`, which sets the value and dispatches no key events. Verified
  2026-08-18 against 1.62.0. Source:
  <https://playwright.dev/python/docs/input#type-characters>. Re-check:
  `inspect.signature(playwright.async_api.Locator.press_sequentially)`.
- `wait_for_selector(state='hidden')` is satisfied by an element that is hidden
  *or* absent, so a frame that never contained it reports success immediately --
  which is why the disappearance wait requires every frame rather than the first.
  Verified 2026-08-18 against 1.62.0. Source:
  <https://playwright.dev/python/docs/api/class-page#page-wait-for-selector>.
  Re-check: the `state` parameter still accepts `'hidden'`.
- `frame.inner_text('body')` reads a child frame the page-level call cannot see;
  `page.wait_for_selector` matches in the main frame only. Verified 2026-08-18
  against a real Chromium. Source:
  <https://playwright.dev/python/docs/frames>. Re-check: the iframe scenario in
  `scripts/playwright_smoke.py`.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import socket
import sys
import warnings
from collections import deque
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, get_args
from urllib.parse import urlparse

import idna
from opentelemetry import trace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import Self

try:
    # Import-time gate (mirrors `pydantic_ai_harness.exa._toolset`): importing the
    # capability fails fast with an install hint when the optional dep is absent.
    # `TargetClosedError` is not re-exported from `playwright.async_api`; the
    # `playwright._impl._errors` module documents its own classes as stable public
    # API, and a driver-raised instance only carries `.name` (so isinstance is the
    # reliable discriminator, not the name attribute).
    from playwright._impl._errors import TargetClosedError as TargetClosedError
    from playwright.async_api import Error as _PlaywrightError
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright as async_playwright

    PlaywrightError = _PlaywrightError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'playwright is required for PlaywrightBrowser. '
        'Install it with: pip install "pydantic-ai-harness[playwright]"\n'
        'Then run: playwright install chromium'
    ) from _import_error

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Sequence

    from opentelemetry.trace import Span, Tracer
    from playwright.async_api import Browser as PlaywrightBrowserHandle
    from playwright.async_api import BrowserContext as PlaywrightBrowserContext
    from playwright.async_api import Page as PlaywrightPage
    from playwright.async_api import Playwright as PlaywrightDriver
    from playwright.async_api import Request as PlaywrightRequest
    from playwright.async_api import Route as PlaywrightRoute
    from playwright.async_api import StorageState
    from playwright.async_api import WebSocketRoute as PlaywrightWebSocketRoute

_T = TypeVar('_T')

DEFAULT_MAX_CONTENT_TOKENS: int = 4000
"""Default token budget for textual tool results injected into the agent context."""

DEFAULT_ACTION_TIMEOUT_MS: int = 5_000
"""Default deadline for element actions (click, type, read, wait), in milliseconds.

Deliberately shorter than the navigation budget: an action that misses is
normally a selector that matches nothing, and a long deadline turns that into a
stall a developer reads as a hung agent rather than a fast, actionable failure.
"""

DEFAULT_NAVIGATION_TIMEOUT_MS: int = 60_000
"""Default deadline for navigation, load settling, and starting or attaching to the browser, in milliseconds."""

_FRAME_TEXT_BUDGET_MS: int = 2_000
"""Total deadline for reading text out of a page's child frames.

One budget for the whole sweep rather than one per frame: a page can carry many
frames, and an unresponsive one must not consume the action's deadline. Whatever
was collected before the budget runs out is kept.
"""

_CHARS_PER_TOKEN = 4
"""Characters-per-token estimate used to turn a token budget into a character cap."""

_PRIVATE_ADDRESS_REASON = 'blocked private or link-local address'
_UNRESOLVED_REASON = 'host that did not resolve, so the private-address block could not clear it'
"""Why the private-address block refused a URL, shared by the pre-check and the route guard."""

_BLANK_PAGE = 'about:blank'
_ERROR_PAGE_SCHEME = 'chrome-error://'
"""The blank page a context starts on, and where a disallowed navigation is sent."""

_MAX_TABS = 8
"""Pages one session keeps open at once.

A tab is a live renderer process, and a page controls how many it opens. Past the
limit a newly opened one is closed and recorded rather than tracked, so a site
that opens windows in a loop costs a bounded amount instead of the host's memory.
"""

_MAX_SCREENSHOT_BYTES = 5_000_000
"""Largest screenshot PNG returned as image content.

Screenshots bypass the textual token budget, and a full-page capture of a long
page can exceed what model providers accept per image (5 MB is the strictest
mainstream limit). An oversized capture becomes a bounded error string instead
of a `BinaryContent` that would fail the next model request and abort the run.
"""


_WaitState = Literal['attached', 'detached', 'hidden', 'visible']
"""The `wait_for_selector` states the toolset asks for."""


class _Locator(Protocol):
    """The subset of `playwright.async_api.Locator` the toolset drives."""

    async def press_sequentially(
        self, text: str, *, delay: float | None = None, timeout: float | None = None
    ) -> None: ...  # pragma: no cover


class _Dialog(Protocol):
    """The subset of `playwright.async_api.Dialog` the session answers."""

    @property
    def type(self) -> str: ...  # pragma: no cover
    @property
    def message(self) -> str: ...  # pragma: no cover
    async def accept(self, prompt_text: str | None = None) -> None: ...  # pragma: no cover
    async def dismiss(self) -> None: ...  # pragma: no cover


class _Mouse(Protocol):
    """The subset of `playwright.async_api.Mouse` the toolset drives."""

    async def click(self, x: float, y: float) -> None: ...  # pragma: no cover
    async def move(self, x: float, y: float) -> None: ...  # pragma: no cover
    async def wheel(self, delta_x: float, delta_y: float) -> None: ...  # pragma: no cover


class _Keyboard(Protocol):
    """The subset of `playwright.async_api.Keyboard` the toolset drives."""

    async def press(self, key: str) -> None: ...  # pragma: no cover


class _Frame(Protocol):
    """The subset of `playwright.async_api.Frame` the toolset reads.

    Child frames are read directly because page-level calls do not cross a frame
    boundary: an embedded schedule, checkout step, or chat widget is invisible to
    `page.inner_text` and to `page.wait_for_selector`.
    """

    @property
    def url(self) -> str: ...  # pragma: no cover
    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str: ...  # pragma: no cover
    async def wait_for_selector(
        self, selector: str, *, timeout: float | None = None, state: _WaitState | None = None
    ) -> object: ...  # pragma: no cover


class _Page(Protocol):
    """The subset of `playwright.async_api.Page` the toolset drives.

    A structural type rather than the concrete `Page`: a real Playwright page
    satisfies it, and tests supply an in-memory double with the same surface
    without launching Chromium. Parameter types are subsets of the real
    signatures and return types supersets, so a real `Page` is assignable here.
    """

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def mouse(self) -> _Mouse: ...  # pragma: no cover
    @property
    def keyboard(self) -> _Keyboard: ...  # pragma: no cover
    @property
    def frames(self) -> Sequence[_Frame]: ...  # pragma: no cover
    async def goto(self, url: str, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def wait_for_load_state(
        self, state: Literal['domcontentloaded'], *, timeout: float | None = None
    ) -> None: ...  # pragma: no cover
    async def title(self) -> str: ...  # pragma: no cover
    async def inner_text(self, selector: str, *, timeout: float | None = None) -> str: ...  # pragma: no cover
    async def click(self, selector: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def fill(self, selector: str, value: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def screenshot(
        self, *, full_page: bool = False, timeout: float | None = None
    ) -> bytes: ...  # pragma: no cover
    async def evaluate(self, expression: str) -> object: ...  # pragma: no cover
    async def go_back(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def go_forward(self, *, timeout: float | None = None) -> object: ...  # pragma: no cover
    async def wait_for_selector(
        self, selector: str, *, timeout: float | None = None, state: _WaitState | None = None
    ) -> object: ...  # pragma: no cover
    def locator(self, selector: str) -> _Locator: ...  # pragma: no cover
    async def bring_to_front(self) -> None: ...  # pragma: no cover
    async def close(self) -> None: ...  # pragma: no cover
    async def aria_snapshot(
        self, *, mode: Literal['ai', 'default'] = 'default', timeout: float | None = None
    ) -> str: ...  # pragma: no cover
    async def hover(self, selector: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def press(self, selector: str, key: str, *, timeout: float | None = None) -> None: ...  # pragma: no cover
    async def select_option(
        self, selector: str, value: Sequence[str], *, timeout: float | None = None
    ) -> list[str]: ...  # pragma: no cover


@dataclass(frozen=True)
class _Deadlines:
    """The budget one operation runs under, as time remaining rather than time allowed.

    A tool that acts on an element and a tool that loads a page fail for
    different reasons and deserve different budgets: a selector that matches
    nothing should fail fast, while a page load legitimately takes longer. A
    per-call `timeout_ms` collapses both to the value the model asked for.

    One tool call makes several Playwright calls in sequence, so `action` and
    `navigation` return what is left of the budget, counted from when the
    operation started. Handing every stage the whole number instead would let one
    call spend its deadline once per stage: `navigate(timeout_ms=2000)` waits on a
    goto, a load state, a title, a body read and a screenshot, and would be
    entitled to 2 seconds each.

    Both count down from that one start, which decides which budget each stage
    takes: a page load may legitimately spend the whole navigation budget, and
    the action budget is long gone by then, so the reads that follow it would get
    1ms and fail on arrival. Every stage after a completed `_settle` -- and in
    `navigate`, everything after the `goto` -- is therefore bounded by
    `navigation`, the budget that allowed for the time already spent. Stages
    before or without a navigation keep `action`.
    """

    action_ms: int
    navigation_ms: int
    started: float

    @property
    def action(self) -> int:
        """Milliseconds left of the element-action budget."""
        return self._remaining(self.action_ms)

    @property
    def navigation(self) -> int:
        """Milliseconds left of the navigation budget."""
        return self._remaining(self.navigation_ms)

    def _remaining(self, budget_ms: int) -> int:
        """Return `budget_ms` less the time already spent, never reaching zero.

        `0` is Playwright's "no deadline", so a configured `0` stays `0` while
        every other budget keeps at least 1ms: counting down to zero would remove
        the deadline at the exact moment it should expire.
        """
        if budget_ms == 0:
            return 0
        return max(1, budget_ms - int((monotonic() - self.started) * 1000))


def _to_idna(host: str) -> str:
    """Return `host` in its ASCII/IDNA form so Unicode and `xn--` spellings compare equal.

    Encoding goes through the `idna` package under UTS46 non-transitional rules,
    which is what Chromium applies. The stdlib `'idna'` codec implements the older
    IDNA-2003 mapping and disagrees on the deviation characters (`ß`, `ς`, ZWJ,
    ZWNJ): it renders `faß.de` as `fass.de` where the browser connects to
    `xn--fa-hia.de`, so an allowlist entry and the request it is meant to permit
    would never match.

    A host that cannot be encoded (over-long or empty labels, IP literals) falls
    back to the input unchanged, so IPv4/IPv6 literals are left alone. The trailing
    dot of a fully-qualified spelling is dropped first: it names the DNS root rather
    than a label, and encoding rejects the empty label it produces, which would
    otherwise deny `example.com.` against an `example.com` allowlist entry.
    """
    host = host.rstrip('.')
    try:
        return idna.encode(host, uts46=True, transitional=False).decode('ascii')
    except (idna.IDNAError, UnicodeError):
        return host


def _url_host(url: str) -> str | None:
    """Extract the host that policy checks run against, or `None` when there is none.

    A URL containing a backslash is treated as hostless: WHATWG parsing (what
    Chromium applies) turns a backslash into `/`, so `urlparse` would report a
    host the browser never connects to. A malformed URL that `urlparse` rejects
    is also treated as hostless, so it fails closed instead of crashing the caller.
    """
    if '\\' in url:
        return None
    try:
        return urlparse(url).hostname
    except ValueError:
        return None


def is_blocked_address(host: str) -> bool:
    """Return whether `host` names an address that is not globally routable.

    Covers private (RFC 1918), loopback, link-local (including the cloud
    metadata endpoint), carrier-grade NAT, reserved, and multicast ranges. This
    classifies a host string, so it sees IP literals and the loopback hostnames
    `localhost` / `*.localhost`; a name pointing at a private address is caught by
    `PlaywrightBrowserSession.decide` resolving it first and passing the answers
    to `refuse`.
    Neither is rebinding-proof, since Chromium resolves the name again before it
    connects (https://github.com/pydantic/pydantic-ai-harness/issues/415).
    A trailing dot is stripped so the fully-qualified spelling gets the same
    verdict, and an IPv4-mapped IPv6 literal is classified by its embedded IPv4
    address. The named category flags are checked alongside `is_global` because
    older stdlib versions classify some of these ranges as global.
    """
    host = host.lower().rstrip('.')
    if host == 'localhost' or host.endswith('.localhost'):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def _is_ip_literal(host: str) -> bool:
    """Whether `host` is already an address, so no name resolution is needed."""
    try:
        ipaddress.ip_address(host.rstrip('.'))
    except ValueError:
        return False
    return True


_RESOLUTION_TTL_SECONDS = 30.0
_RESOLUTION_CACHE_MAX = 256
_RESOLUTION_TIMEOUT_SECONDS = 2.0
_resolution_cache: dict[str, tuple[float, tuple[str, ...]]] = {}


async def _resolve_host(host: str) -> tuple[str, ...] | None:
    """Return the addresses `host` resolves to, or `None` when the lookup did not answer.

    The two are distinguished because they lead to opposite verdicts. A host that
    resolved is classified on its addresses; one that did not cannot be cleared, and
    whoever controls the name controls whether the lookup answers -- stalling this
    one and then handing Chromium a private address would otherwise be a way past
    the block. So an unanswered lookup is a refusal, which also costs little when
    the failure is honest: a name this process cannot resolve is one the browser is
    about to fail on too. A host the resolver cannot even encode counts as
    unanswered for the same reason: `getaddrinfo` raises a `UnicodeError` for an
    empty or over-long label (`a..com`), and that is a verdict, not a crash the
    caller should carry.

    The cache is a duplicate of one Chromium keeps anyway, so it is kept short and
    small. It cannot make the block airtight: Chromium resolves the name a second
    time, and a record that changes between the two lookups is the DNS-rebinding
    case that only a proxy or pinned resolver closes.

    The lookup is bounded because of where it runs: ahead of the operation whose
    deadline it would otherwise escape, and inside the route guard, which holds the
    request until it returns. A stalled resolver therefore costs a bounded wait
    rather than the run.
    """
    now = monotonic()
    cached = _resolution_cache.get(host)
    if cached is not None and cached[0] > now:
        return cached[1]
    try:
        addresses = await asyncio.wait_for(_getaddrinfo(host), _RESOLUTION_TIMEOUT_SECONDS)
    except (OSError, UnicodeError, asyncio.TimeoutError):
        return None
    if len(_resolution_cache) >= _RESOLUTION_CACHE_MAX:
        _resolution_cache.clear()
    _resolution_cache[host] = (now + _RESOLUTION_TTL_SECONDS, addresses)
    return addresses


async def _getaddrinfo(host: str) -> tuple[str, ...]:  # pragma: no cover
    """Ask the system resolver what `host` points at, off the event loop thread.

    The one place the capability performs a real lookup, and therefore the seam the
    test suite replaces so that no test depends on a resolver. What it does is
    checked against live DNS in `scripts/playwright_smoke.py::_check_private_name_block`.
    """
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return tuple(sorted({str(info[4][0]) for info in infos}))


def _scroll_script(move: str) -> str:
    """Wrap a scroll expression so the page reports where it ended up."""
    return (
        '(() => { const before = window.scrollY; ' + move + '; return [before, window.scrollY, '
        'Math.max(0, document.documentElement.scrollHeight - window.innerHeight)].join("|"); })()'
    )


def _scroll_position(reported: object) -> str:
    """Say where the page now sits, so a scroll that moved nothing is visible as one.

    Without this a scroll that hit the end of the page, or a page with nothing to
    scroll, returns the same text as one that revealed new content, and the model
    has no way to tell that repeating it is pointless.
    """
    if not isinstance(reported, str):
        return ''  # pragma: no cover -- `evaluate` returns what the expression built
    parts = reported.split('|')
    if len(parts) != 3 or not all(part.lstrip('-').isdigit() for part in parts):
        return ''  # pragma: no cover -- same
    before, after, furthest = (int(part) for part in parts)
    if furthest == 0:
        return 'The page has nothing to scroll.'
    if after >= furthest:
        return 'At the bottom of the page.'
    if after == 0:
        return 'At the top of the page.'
    if after == before:
        return f'Position unchanged, {after} of {furthest} px down.'
    return f'{after} of {furthest} px down.'


RequestKind = Literal['navigation', 'subframe', 'data', 'subresource']
"""What a request is for, which is what an egress rule is usually about.

`navigation` is the top-level document, `subframe` a document loaded into an
embedded frame, `data` anything a script uses to move data (`fetch`, XHR,
EventSource, WebSocket, `sendBeacon`), and `subresource` the passive assets that
render a page (images, stylesheets, scripts, fonts, media).
"""

DEFAULT_ALLOWLIST_REACH: frozenset[RequestKind] = frozenset({'navigation', 'data'})
"""Which request kinds `allowed_domains` bounds unless the policy says otherwise.

Navigation, so the agent stays on the sites it was given, and data requests, so a
script on one of those pages cannot read from or post to somewhere else. Passive
subresources are left alone because a page whose images, fonts and scripts are
aborted renders as a broken page, and sub-frame documents because a permitted
page's identity-provider and payment steps live in them.
"""

_DATA_RESOURCE_TYPES = frozenset({'fetch', 'xhr', 'eventsource', 'websocket', 'ping', 'preflight'})
"""Playwright resource types that carry data rather than render the page.

`ping` is `navigator.sendBeacon`, which is a send channel like the rest of them.
"""


def _request_kind(request: PlaywrightRequest, top_level: bool) -> RequestKind:
    """Classify a Playwright request for the policy.

    A document is a navigation only in the main frame; the same type inside an
    embedded frame is what loads an identity provider or a payment step, which a
    policy usually treats differently.
    """
    if request.resource_type in _DATA_RESOURCE_TYPES:
        return 'data'
    if request.is_navigation_request():
        return 'navigation' if top_level else 'subframe'
    return 'subresource'


@dataclass(frozen=True)
class EgressRequest:
    """One request the browser is about to make, as the policy sees it.

    Everything the guard knows is passed through, including the raw Playwright
    `resource_type`, so a policy that needs a distinction this module never
    anticipated can make it without waiting for a new field here.
    """

    url: str
    kind: RequestKind
    resource_type: str = 'document'
    """Playwright's own classification, e.g. `document`, `xhr`, `image`, `font`."""
    method: str = 'GET'
    top_level: bool = True
    """Whether this is the main frame's own document rather than something inside the page."""
    resolved_addresses: tuple[str, ...] = ()
    """What the host resolved to, when the caller resolved it. Empty when it did not."""
    resolution_failed: bool = False
    """Whether a lookup the policy asked for did not answer, leaving the host unclassifiable."""


DEFAULT_RESOLVED_KINDS: frozenset[RequestKind] = frozenset(get_args(RequestKind))
"""Which kinds have their host resolved before the private-address block classifies it.

A name is not an IP literal, so `is_blocked_address` cannot see that it points at a
private address; resolving first is what closes that. Every kind is resolved,
because the literal check already covers every kind and a spelling should not
decide the verdict: an `img` pointed at `169.254.169.254` and one pointed at a name
answering with that address are the same request. Passive subresources do not
return content to the model, but a page can still time their loads to map what is
listening inside the network.

The cost is bounded by the resolution cache, which makes this a lookup per distinct
host rather than per request, and a page draws on few hosts.
"""


@dataclass(frozen=True)
class EgressPolicy:
    """Where the browser may go, and what a page loaded there may talk to.

    The axes are independent and deny wins: a blocked address is refused even when
    the allowlist names it, and `blocked_domains` beats `allowed_domains`. Holding
    them in one object keeps enforcement and the description given to the model
    from drifting apart -- the failure mode being instructions that promise a reach
    the guards do not grant.

    The fields cover the rules we expect; `refuse` covers the rest. Subclass and
    override it for anything else -- per-path rules, a CDN allowed only for fonts,
    a method-dependent rule -- calling `super().refuse(request)` for the default
    verdict.
    """

    allowed_domains: list[str] | None = None
    """Hosts the reachable kinds are limited to. `None` allows every public host."""
    blocked_domains: list[str] | None = None
    """Hosts refused whatever else permits them, for every kind of request."""
    block_private_addresses: bool = True
    """Refuse addresses that are not globally routable, in every frame and for every kind."""
    include_subdomains: bool = True
    """Whether an entry also covers its subdomains, so `example.com` reaches `api.example.com`."""
    allowlist_reach: frozenset[RequestKind] = DEFAULT_ALLOWLIST_REACH
    """Which kinds `allowed_domains` bounds. `frozenset(get_args(RequestKind))` locks down everything."""
    resolved_kinds: frozenset[RequestKind] = DEFAULT_RESOLVED_KINDS
    """Which kinds have their host resolved so `block_private_addresses` can see where a name points."""

    def __post_init__(self) -> None:
        for field_name in ('allowed_domains', 'blocked_domains'):
            for entry in getattr(self, field_name) or ():
                # Validated as `_matches` will read it, so an entry that only
                # differs by surrounding whitespace cannot dodge a check.
                entry = entry.strip()
                if '*' in entry:
                    raise UserError(
                        f'{field_name} entry {entry!r} contains a wildcard, which never matches a host. '
                        'Write the bare domain: subdomains are included unless include_subdomains=False.'
                    )
                if entry.startswith('.'):
                    raise UserError(
                        f'{field_name} entry {entry!r} starts with a dot, which never matches a host. '
                        'Write the bare domain: subdomains are included unless include_subdomains=False.'
                    )

    def refuse(self, request: EgressRequest) -> str | None:
        """Why the browser must not make this request, or `None` to allow it.

        `about:blank` is permitted: it is the state a context starts in and the
        target a refused navigation is bounced to, so denying it would refuse every
        tool call made before the first navigation. Other hostless URLs (`data:`,
        `blob:`) are refused as navigation and allowed as page content, where they
        are ordinary inline assets.
        """
        if request.url == _BLANK_PAGE:
            return None
        host = _url_host(request.url)
        if host is not None and self.block_private_addresses and is_blocked_address(host):
            return _PRIVATE_ADDRESS_REASON
        if self.block_private_addresses and request.resolution_failed:
            return _UNRESOLVED_REASON
        if self.block_private_addresses and any(is_blocked_address(a) for a in request.resolved_addresses):
            return _PRIVATE_ADDRESS_REASON
        if host is None:
            return None if request.kind in ('data', 'subresource') else 'URL with no host'
        if self._matches(host, self.blocked_domains):
            return 'domain in blocked_domains'
        if request.kind in self.allowlist_reach and not self._permitted(host):
            return 'domain not in allowed_domains'
        return None

    def _permitted(self, host: str) -> bool:
        """Whether the allowlist admits `host`, with `None` meaning every public host."""
        return self.allowed_domains is None or self._matches(host, self.allowed_domains)

    def _matches(self, host: str, domains: list[str] | None) -> bool:
        """Whether `host` equals an entry, or sits under one when subdomains count."""
        if not domains:
            return False
        host = _to_idna(host)
        for entry in domains:
            domain = _to_idna(entry.strip().lower())
            if domain and (host == domain or (self.include_subdomains and host.endswith('.' + domain))):
                return True
        return False

    def needs_resolution(self, request: EgressRequest) -> bool:
        """Whether the caller must resolve this host before `refuse` can classify it.

        Asked before `refuse` so that `refuse` stays synchronous: an override needs
        no async machinery, and the lookup a name-based private address requires
        happens outside it. A host that is already an address needs none.
        """
        if not self.block_private_addresses or request.kind not in self.resolved_kinds:
            return False
        host = _url_host(request.url)
        return host is not None and not _is_ip_literal(host)

    def enforced(self) -> bool:
        """Whether anything is restricted, i.e. whether a route guard is worth installing."""
        return self.allowed_domains is not None or bool(self.blocked_domains) or self.block_private_addresses

    def describe(self) -> str:
        """The reach, phrased for the model's instructions."""
        if self.allowed_domains is None:
            domains = 'all'
        elif self.allowed_domains:
            domains = ', '.join(self.allowed_domains)
            if self.include_subdomains:
                domains += ' (and their subdomains)'
        else:
            domains = 'none'
        if self.allowed_domains is not None and self.allowlist_reach != DEFAULT_ALLOWLIST_REACH:
            # Named only when it is not the default, so the model is told what the
            # allowlist actually bounds rather than being left with the reach the
            # default sentence implies.
            domains += f' (allowlist bounds {", ".join(sorted(self.allowlist_reach)) or "nothing"})'
        if self.blocked_domains:
            domains += f', except {", ".join(self.blocked_domains)}'
        if self.block_private_addresses:
            domains += ' (private/internal addresses blocked)'
        return domains


_CREDENTIAL_PARAMETERS = (
    'access_token',
    'api_key',
    'apikey',
    'auth',
    'awsaccesskeyid',
    'client_secret',
    'code',
    'id_token',
    'key',
    'oauth_token',
    'password',
    'pwd',
    'refresh_token',
    'secret',
    'session',
    'sig',
    'signature',
    'token',
    'x-amz-signature',
    'x-goog-signature',
)
"""Query and fragment parameters whose values are treated as credentials.

Names rather than whole query strings: which endpoint a page called is what makes
a recorded request useful, and the OTel HTTP conventions redact known-sensitive
parameters rather than dropping the query. The list covers the OAuth grant and
the signed-URL parameters of the major clouds. Prefixed spellings are listed in
full because the pattern anchors a name at `?`, `&` or `#`, so `secret` does not
cover `client_secret`.
"""

_USERINFO = re.compile(r'(?<![\w.+-])([a-zA-Z][\w+.\-]*://)[^/?#\s@]*@')
"""A URL's `user:password@` prefix, wherever the URL sits.

Unanchored, because this also cleans console text, where the URL a page logged is
rarely the first thing on the line. Whitespace ends the authority so an ordinary
sentence with an `@` in it later cannot be swallowed into one match.
"""

_CREDENTIAL_VALUE = re.compile(
    r'(?i)([?&#](?:' + '|'.join(_CREDENTIAL_PARAMETERS) + r')=)[^&#]*',
)


def _without_credentials(url: str) -> str:
    """Return `url` with userinfo removed and credential parameter values redacted.

    Applied to every URL that leaves the browser: the ones tools return, the ones
    events carry, and the one each operation's span records. An OAuth callback or
    a signed link puts a usable credential in the address bar, and a tool result
    is written into the message history like anything else the model reads.

    URLs reach the model and an OpenTelemetry backend, where a bearer
    token, an OAuth code, or a signed-URL signature is replayable by anyone who
    can read them. Regexes rather than a parse-and-rebuild, so a URL Chromium
    accepted but `urlsplit` rejects is still cleaned rather than raising. The
    scheme, host, path, and the names of every parameter survive: they are what
    identifies the request the model is looking for.
    """
    return _CREDENTIAL_VALUE.sub(r'\1REDACTED', _USERINFO.sub(r'\1', url))


def _without_endpoint_credentials(message: str, cdp_url: str) -> str:
    """Replace the CDP endpoint in `message` with a form carrying no credentials.

    A failure to attach reaches the model as a tool result, and Playwright quotes
    the endpoint it tried, call log included. Managed-browser providers routinely
    put a token in that URL's query string or path, so only the scheme, host and
    port survive -- enough to see which endpoint was unreachable, which the driver
    reports separately anyway.
    """
    try:
        parsed = urlparse(cdp_url)
        host = parsed.hostname or ''
        # `.port` parses lazily and raises on a non-numeric port, so it has to be
        # read inside the guard: this runs while an error is already being built.
        if parsed.port is not None:
            host = f'{host}:{parsed.port}'
        scheme = parsed.scheme
    except ValueError:
        return message.replace(cdp_url, '<cdp_url>')
    safe = f'{scheme}://{host}' if scheme else host
    return message.replace(cdp_url, safe)


def _truncate(text: str, max_chars: int) -> str:
    """Cap tool output at `max_chars`, keeping the head where the substance sits."""
    if len(text) <= max_chars:
        return text
    marker = f'\n[... tool output truncated at {max_chars} characters]'
    if len(marker) >= max_chars:
        return text[:max_chars]
    return f'{text[: max_chars - len(marker)]}{marker}'


class BrowserUnavailableError(RuntimeError):
    """No browser could be started for this run.

    Raised by `PlaywrightBrowserSession.ensure_page` and turned into a tool result
    by the toolset, rather than ending the run: installing a browser is something
    an agent with a shell can do once it is told what is missing.
    """


class BrowserUnavailableWarning(UserWarning):
    """A browser could not be started, warned once where the process can see it.

    The model learns this from the tool result and a trace shows the failed
    operation, but a developer watching a terminal sees neither. Filter it with
    `warnings.simplefilter('ignore', BrowserUnavailableWarning)` when the miss is
    expected.
    """


_CHROMIUM_MISSING_MESSAGE = (
    'Chromium is not installed. Run `playwright install chromium` (on a fresh Linux or CI image use '
    '`playwright install --with-deps chromium` to also install the required system libraries) and restart '
    'the agent to enable browser tools.'
)


async def _auto_install_chromium() -> str | None:
    """Run `playwright install chromium` in this interpreter; `None` on success, else the installer output.

    Only invoked when `auto_install_chromium=True` and the binary is missing. It
    shells out to a subprocess and downloads a browser, so it runs outside the
    mocked test surface. On failure the merged stdout/stderr is returned so the
    launch path can surface why the install failed instead of the generic hint.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        '-m',
        'playwright',
        'install',
        'chromium',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await proc.communicate()
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise
    if proc.returncode == 0:
        return None
    return stdout.decode(errors='replace')


_FALLBACK_TRACER = trace.get_tracer('pydantic-ai-harness.playwright')
"""Tracer used when the session is driven outside an agent run.

Inside a run the session takes the run's own tracer, so browser spans land
wherever the agent's instrumentation settings send its tool calls.
"""

_ABORTED_FAILURE = 'net::ERR_FAILED'
"""Failure text Chromium reports for a request the route guard aborted.

The guard records its own refusal with the reason, so the matching
`requestfailed` event is dropped rather than logged a second time without one.
"""

_EVENT_LOG_LIMIT = 500
"""Browser events retained per run. A ring buffer: a chatty page cannot grow it without bound."""

_MAX_EVENT_CHARS = 2_000
"""Longest message or URL kept on one recorded browser event.

The page controls both, and every event is held for the length of the run and
exported as span attributes, so an unbounded one lets the page decide how much
memory and telemetry volume the run spends.
"""


def _clip(text: str) -> str:
    """Cap a page-controlled string at the per-event limit."""
    return text if len(text) <= _MAX_EVENT_CHARS else f'{text[:_MAX_EVENT_CHARS]}...'


@dataclass(frozen=True)
class _DialogDecision:
    """How the next dialog a page opens is answered."""

    accept: bool
    prompt_text: str | None


@dataclass(frozen=True)
class BrowserEvent:
    """Something the browser did that no tool call records.

    Console output, uncaught page errors, responses, requests the egress policy
    refused, dialogs the page opened, and tabs it opened or that the session
    refused to open. The `console_messages` and
    `network_requests` tools read this log, and every entry is also added as an
    OpenTelemetry span event on the browser operation that was running, so a
    trace shows what the page did between tool calls.
    """

    kind: Literal[
        'console',
        'page_error',
        'response',
        'request_failed',
        'request_blocked',
        'popup_opened',
        'popup_closed',
        'dialog',
    ]
    level: Literal['info', 'warning', 'error']
    message: str
    url: str | None = None
    method: str | None = None
    status: int | None = None

    def describe(self) -> str:
        """Render the event as one model-facing line."""
        parts = [f'[{self.level}] {self.kind}']
        if self.method is not None:
            parts.append(self.method)
        if self.status is not None:
            parts.append(str(self.status))
        if self.url is not None:
            parts.append(self.url)
        if self.message:
            parts.append(self.message)
        return ' '.join(parts)

    def attributes(self) -> dict[str, str | int]:
        """Render the event as OpenTelemetry span-event attributes.

        Names follow the OTel HTTP semantic conventions where one applies, so a
        backend that already understands `url.full` or `http.response.status_code`
        reads these without a custom mapping.
        """
        attributes: dict[str, str | int] = {'browser.event.kind': self.kind, 'browser.event.level': self.level}
        if self.message:
            attributes['browser.event.message'] = self.message
        if self.url is not None:
            attributes['url.full'] = self.url
        if self.method is not None:
            attributes['http.request.method'] = self.method
        if self.status is not None:
            attributes['http.response.status_code'] = self.status
        return attributes


class _ConsoleMessage(Protocol):
    """The subset of `playwright.async_api.ConsoleMessage` the session reads."""

    @property
    def type(self) -> str: ...  # pragma: no cover
    @property
    def text(self) -> str: ...  # pragma: no cover


class _Response(Protocol):
    """The subset of `playwright.async_api.Response` the session reads."""

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def status(self) -> int: ...  # pragma: no cover
    @property
    def request(self) -> _FailedRequest: ...  # pragma: no cover


class _FailedRequest(Protocol):
    """The subset of `playwright.async_api.Request` the session reads."""

    @property
    def url(self) -> str: ...  # pragma: no cover
    @property
    def method(self) -> str: ...  # pragma: no cover
    @property
    def failure(self) -> str | None: ...  # pragma: no cover


class PlaywrightBrowserSession:
    """One agent run's Chromium: how a page is obtained, guarded, and released.

    Entering the session arms it; nothing starts until `ensure_page` is first
    awaited, so a run that never calls a browser tool never launches a browser.
    Exiting closes whatever was started, in the order that a half-built session
    still tears down cleanly.

    `PlaywrightBrowser` creates one per run (via `for_run`), so concurrent runs
    never share a page. It can also be driven directly:

    ```python
    async with PlaywrightBrowserSession(policy=EgressPolicy()) as session:
        page = await session.ensure_page()
    ```
    """

    def __init__(
        self,
        *,
        policy: EgressPolicy | None = None,
        headless: bool = True,
        storage_state: StorageState | None = None,
        cdp_url: str | None = None,
        auto_install_chromium: bool = False,
        chromium_sandbox: bool = True,
        launch_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    ) -> None:
        self.policy = policy if policy is not None else EgressPolicy()
        """Egress policy the route guard enforces, and the default for a toolset built on this session."""
        self._headless = headless
        self._storage_state = storage_state
        self._cdp_url = cdp_url
        self._auto_install_chromium = auto_install_chromium
        self._chromium_sandbox = chromium_sandbox
        self._launch_timeout_ms = launch_timeout_ms
        self.page: _Page | None = None
        """Active page: the one every tool acts on, or `None` before the browser is launched."""
        self.pages: list[_Page] = []
        """Every open tab, in the order it opened. The first entry is the page the session started on."""
        self.dialog_decision: _DialogDecision | None = None
        """How the next dialog is answered, or `None` to dismiss it.

        A dialog blocks the page until it is answered and the tool call that
        triggered it is still running, so the decision has to exist before the
        action rather than be asked for while the dialog is open. It is consumed
        by the first dialog that arrives, so an armed `accept` cannot go on
        silently answering every later one.
        """
        self.launch_error: str | None = None
        """Set when a launch attempt failed (e.g. the Chromium binary is missing)."""
        self._driver_cm: AbstractAsyncContextManager[PlaywrightDriver] | None = None
        self._driver: PlaywrightDriver | None = None
        self._driver_entered = False
        self._browser: PlaywrightBrowserHandle | None = None
        self._context: PlaywrightBrowserContext | None = None
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._launch_lock = asyncio.Lock()
        self.tracer: Tracer = _FALLBACK_TRACER
        """Tracer browser operations report to. `PlaywrightBrowser.wrap_run` sets the run's own."""
        self.events: deque[BrowserEvent] = deque(maxlen=_EVENT_LOG_LIMIT)
        """What the browser did outside the tool calls, newest last."""
        self.events_recorded = 0
        """How many events have been recorded, including those the log has dropped.

        The log is bounded, so a position in it is not stable; this count is, which
        is what lets an operation ask only about the events its own action caused.
        """
        self.operation_events_mark = 0
        """Value of `events_recorded` when the running operation started."""
        self.operation_span: Span | None = None
        """Span of the browser operation currently running, when one is.

        Playwright delivers page events from its own receive task, so the
        operation's span cannot be recovered from the OpenTelemetry context there.
        The toolset publishes it here for the length of an operation instead.
        """

    def record(self, event: BrowserEvent) -> None:
        """Log a browser event and attach it to the operation that was running.

        Credentials are removed here, at the one point every event passes
        through, so neither the tool output nor the exported span carries them.
        """
        # The message is sanitised alongside the URL: a page that logs its own
        # request (`console.error('...?access_token=...')`) reaches the model and
        # the exporter through exactly the same two methods.
        event = replace(
            event,
            url=None if event.url is None else _clip(_without_credentials(event.url)),
            message=_clip(_without_credentials(event.message)),
        )
        self.events.append(event)
        self.events_recorded += 1
        span = self.operation_span
        if span is not None and span.is_recording():
            span.add_event(f'browser.{event.kind}', event.attributes())

    def failure_since_operation_start(self) -> BrowserEvent | None:
        """The last refused or failed request caused by the running operation, if any.

        A navigation Chromium could not complete leaves the page on its own error
        page, whose URL (`chrome-error://chromewebdata/`) names neither the
        destination nor what stopped it. The event log holds both, and scoping the
        search to this operation keeps an older refusal from being reported as the
        cause of this one.
        """
        caused_here = self.events_recorded - self.operation_events_mark
        for event in reversed(list(self.events)[-caused_here:] if caused_here > 0 else []):
            if event.kind in ('request_blocked', 'request_failed'):
                return event
        return None

    def _on_console(self, message: _ConsoleMessage) -> None:
        """Record a console message, keeping the page's own severity."""
        level: Literal['info', 'warning', 'error'] = 'info'
        if message.type == 'error':
            level = 'error'
        elif message.type == 'warning':
            level = 'warning'
        self.record(BrowserEvent(kind='console', level=level, message=f'{message.type}: {message.text}'))

    def _on_page_error(self, error: object) -> None:
        """Record an uncaught exception from page scripts."""
        self.record(BrowserEvent(kind='page_error', level='error', message=str(error)))

    def _on_response(self, response: _Response) -> None:
        """Record a response, so the agent can find the request a page made for its data."""
        self.record(
            BrowserEvent(
                kind='response',
                level='error' if response.status >= 400 else 'info',
                message='',
                url=response.url,
                method=response.request.method,
                status=response.status,
            )
        )

    def _on_request_failed(self, request: _FailedRequest) -> None:
        """Record a request the network layer never completed.

        Requests the egress policy aborts arrive here too, but the guard has
        already recorded those with their reason, so they are dropped to keep one
        entry per refusal.
        """
        failure = request.failure or 'failed'
        if failure == _ABORTED_FAILURE:
            return
        self.record(
            BrowserEvent(kind='request_failed', level='error', message=failure, url=request.url, method=request.method)
        )

    async def ensure_page(self) -> _Page:
        """Return the active page, launching Chromium lazily on the first call.

        Tool calls in one model response run concurrently, so the launch is
        serialized: the first caller runs it under the lock and the rest observe
        the populated `page` (or the launch error) instead of launching a second
        Chromium.

        A launch that fails raises `BrowserUnavailableError`, which the tools turn
        into a result rather than an exception. The recorded failure is cleared
        once it has been reported, so a later call tries again: an agent that just
        installed the browser gets one, instead of a session that refuses for the
        rest of the run.
        """
        if self.launch_error is not None:
            raise BrowserUnavailableError(self.launch_error)
        if self.page is None:
            async with self._launch_lock:
                if self.page is None and self.launch_error is None:
                    if self._driver_cm is None:
                        # Not a `BrowserUnavailableError`: nothing the model does can
                        # fix a capability that was never started, so this one ends
                        # the run rather than becoming a tool result.
                        raise RuntimeError(
                            'PlaywrightBrowser is not running: PlaywrightBrowser.wrap_run must be active before any '
                            'browser tool.'
                        )
                    await self._launch()
            if self.launch_error is not None:
                raise BrowserUnavailableError(self.launch_error)
            if self.page is None:
                raise BrowserUnavailableError('Browser failed to launch.')  # pragma: no cover
        return self.page

    async def _launch(self) -> None:
        """Start the driver and Chromium, then open the guarded page.

        The driver is entered once per session rather than once per attempt: a
        launch that failed is retried on the next tool call, and re-entering the
        context manager would start a second driver connection and leave the first
        one running until teardown.

        Each step runs under `launch_timeout_ms` -- the connect through
        Playwright's own `timeout`, the rest through `_bounded` -- because all of
        it happens inside a tool call holding the operation lock: a browser that
        accepts the connection and then never opens a context would stall the run
        with no deadline of its own. The auto-install download inside `_connect`
        is the documented exception.
        """
        assert self._driver_cm is not None
        if self._driver is None:
            self._driver = await self._driver_cm.__aenter__()
            self._driver_entered = True
        # An attempt that connected and then failed to build its context left a browser
        # behind. It is closed before another is started, since teardown holds only the
        # latest handle; a close that fails keeps the handle so teardown can try again.
        if self._browser is not None:
            stale = self._browser
            await self._bounded(stale.close())
            self._browser = None
        browser = await self._connect(self._driver)
        if browser is None:
            return
        # Assigned before the context and page are built, so teardown closes Chromium
        # even when a later setup step raises.
        self._browser = browser
        # Service workers can issue requests that context routes never see, so they
        # are blocked to keep all traffic on the routable path. Downloads are refused
        # because no tool exposes them: accepting them only lets a page write to the
        # host's temporary storage for the length of the run.
        context = await self._bounded(
            browser.new_context(
                storage_state=self._storage_state,
                service_workers='block',
                accept_downloads=False,
            )
        )
        self._context = context
        page = await self._bounded(context.new_page())
        if self.policy.enforced():
            await self._bounded(context.route('**/*', self._route_guard))
            # A network route never sees a WebSocket, so sockets get a guard of their own.
            await self._bounded(context.route_web_socket('**/*', self._websocket_guard))
        self._wire_page(page)
        self.pages.append(page)
        self.page = page

    async def _bounded(self, awaitable: Awaitable[_T]) -> _T:
        """Run a setup step that has no `timeout` parameter under `launch_timeout_ms`.

        A configured `0` keeps Playwright's meaning of "no deadline", the same way
        the per-call deadlines treat it.
        """
        if self._launch_timeout_ms == 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, self._launch_timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            # Raised as a Playwright timeout so the tools map it like any other
            # deadline they already handle.
            raise PlaywrightTimeoutError(f'Timeout {self._launch_timeout_ms}ms exceeded.') from exc

    async def _connect(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Attach to or start a browser, or record why none is available.

        Both paths are bounded by `launch_timeout_ms`: a `cdp_url` pointing at an
        unresponsive endpoint would otherwise hold the operation lock with no
        deadline at all. The auto-install download is deliberately left unbounded,
        since a first-run browser fetch legitimately outlasts an action timeout.

        A launched browser gets Chromium's renderer sandbox, which Playwright
        leaves off by default. This capability exists to open pages nobody vetted,
        and the sandbox is what keeps a compromised renderer inside the browser.
        A browser reached over `cdp_url` is already running, so its own
        configuration decides this.
        """
        if self._cdp_url is not None:
            try:
                return await pw.chromium.connect_over_cdp(self._cdp_url, timeout=self._launch_timeout_ms)
            except PlaywrightError as exc:
                raise PlaywrightError(_without_endpoint_credentials(str(exc), self._cdp_url)) from exc
        if os.path.exists(pw.chromium.executable_path):
            return await pw.chromium.launch(
                headless=self._headless, chromium_sandbox=self._chromium_sandbox, timeout=self._launch_timeout_ms
            )
        # Binary genuinely absent: raise a clear install hint, or fetch it when
        # opted in. A launch failure with the binary present (sandbox, missing
        # system libs, no display) is left to surface as its own error rather than
        # being masked as "Chromium is not installed".
        browser = await self._install_and_retry(pw) if self._auto_install_chromium else None
        if browser is None:  # pragma: no branch
            if self.launch_error is None:
                self.launch_error = _CHROMIUM_MISSING_MESSAGE
            # The model hears about this through the tool result and a trace shows
            # the failed operation. Neither reaches a developer reading a terminal.
            warnings.warn(self.launch_error, BrowserUnavailableWarning, stacklevel=2)
        return browser

    async def _install_and_retry(self, pw: PlaywrightDriver) -> PlaywrightBrowserHandle | None:
        """Fetch Chromium and relaunch once.

        Returns the browser on success. On install failure it records a launch
        error carrying a bounded tail of the installer output (so the failure is
        diagnosable) and returns `None`.
        """
        install_output = await _auto_install_chromium()
        if install_output is None:
            return await pw.chromium.launch(  # pragma: no cover
                headless=self._headless, chromium_sandbox=self._chromium_sandbox, timeout=self._launch_timeout_ms
            )
        self.launch_error = f'{_CHROMIUM_MISSING_MESSAGE}\nAuto-install failed:\n{install_output[-300:]}'
        return None

    async def decide(self, request: EgressRequest) -> str | None:
        """Why the policy refuses `request`, or `None` to allow it, resolving the host first if needed.

        Every enforcement point goes through here rather than calling `refuse`
        directly, so a name that points at a private address is caught wherever it
        is asked about: the route guard, the socket guard, the `navigate`
        pre-check, and the post-action re-check.
        """
        host = _url_host(request.url)
        if host is not None and self.policy.needs_resolution(request):
            addresses = await _resolve_host(host)
            request = replace(request, resolved_addresses=addresses or (), resolution_failed=addresses is None)
        return self.policy.refuse(request)

    async def _route_guard(self, route: PlaywrightRoute, request: PlaywrightRequest) -> None:
        """Network-layer egress policy: abort what the policy refuses, pass the rest.

        Every request reaches here, whatever its type and whichever frame made it,
        so the policy is asked about all of them and decides which distinctions
        matter. The frame lookup can raise for a request whose frame is already
        gone; that request is treated as a top-level navigation, which is the
        strictest reading.
        """
        try:
            frame = request.frame
            top_level = frame == frame.page.main_frame
        except PlaywrightError:
            top_level = True
        kind = _request_kind(request, top_level)
        reason = await self.decide(
            EgressRequest(
                url=request.url,
                kind=kind,
                resource_type=request.resource_type,
                method=request.method,
                top_level=top_level and kind == 'navigation',
            )
        )
        if reason is None:
            await route.continue_()
            return
        await self._abort(route, request.url, reason)

    async def _websocket_guard(self, websocket: PlaywrightWebSocketRoute) -> None:
        """Apply the egress policy to WebSocket connections.

        `context.route` never sees a WebSocket, so without this a page could open
        `ws://127.0.0.1:<port>` and talk to an internal service the HTTP guard
        refuses. A socket is asked about as `data`, the kind it belongs to, so the
        allowlist bounds it exactly as it bounds `fetch`; the guard is therefore
        installed whenever anything is enforced, not only for the address block.

        A socket that is permitted has to be connected explicitly: registering a
        handler at all takes Playwright out of its pass-through mode.
        """
        reason = await self.decide(
            EgressRequest(url=websocket.url, kind='data', resource_type='websocket', top_level=False)
        )
        if reason is not None:
            self.record(BrowserEvent(kind='request_blocked', level='warning', message=reason, url=websocket.url))
            await websocket.close()
            return
        websocket.connect_to_server()

    async def _abort(self, route: PlaywrightRoute, url: str, reason: str) -> None:
        """Refuse a request and record why, so a blocked page is diagnosable."""
        self.record(BrowserEvent(kind='request_blocked', level='warning', message=reason, url=url))
        await route.abort()

    def _wire_page(self, page: PlaywrightPage) -> None:
        """Subscribe to what a page reports, for the session's first page and every tab it opens.

        A tab the session does not listen to is a tab whose console output,
        requests and dialogs never reach the model, so every page goes through
        here rather than only the one the context started with.
        """
        page.on('popup', self._on_popup)
        page.on('console', self._on_console)
        page.on('pageerror', self._on_page_error)
        page.on('response', self._on_response)
        page.on('requestfailed', self._on_request_failed)
        page.on('dialog', self._on_dialog)
        page.on('close', self._on_page_closed)

    def _spawn(self, coro: Coroutine[object, object, None]) -> None:
        """Run work a synchronous page event cannot await, keeping a strong task reference.

        Playwright delivers page events from its own receive task, so answering a
        dialog or closing a refused tab -- both async -- can only be scheduled from
        here. A task nobody holds can be garbage collected mid-flight.
        """
        task = asyncio.create_task(coro)
        self._event_tasks.add(task)
        task.add_done_callback(self._event_task_done)

    def _event_task_done(self, task: asyncio.Task[None]) -> None:
        """Release a finished event task and retrieve any error it carried."""
        self._event_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _on_popup(self, popup: PlaywrightPage) -> None:
        """Track a tab the site opened, or close it when the session is already full.

        The page is kept rather than closed: a `target="_blank"` link, a sign-in
        popup and a payment step all arrive this way, and closing them made those
        flows impossible to complete. It does not become active -- the model
        selects it through the `tabs` tool -- so a popup cannot take the page out
        from under the operation that triggered it.
        """
        if len(self.pages) >= _MAX_TABS:
            self.record(
                BrowserEvent(
                    kind='popup_closed',
                    level='warning',
                    message=f'tab limit of {_MAX_TABS} reached',
                    url=popup.url,
                )
            )
            self._spawn(popup.close())
            return
        self.record(BrowserEvent(kind='popup_opened', level='info', message='opened by the page', url=popup.url))
        self._wire_page(popup)
        self.pages.append(popup)

    def _on_page_closed(self, closed: object) -> None:
        """Drop a tab that has closed, and move the active pointer off it."""
        self.pages = [page for page in self.pages if page is not closed]
        if self.page is closed and self.pages:
            self.page = self.pages[-1]

    def _on_dialog(self, dialog: _Dialog) -> None:
        """Answer a dialog from the armed decision, and record what it said.

        Registering any handler takes Playwright out of its auto-dismiss mode and
        leaves the page blocked until the dialog is answered, so every path here
        answers. The default is to dismiss, which is what Playwright does on its
        own: the cancelling branch of a `confirm`, and an empty `prompt`.
        """
        decision = self.dialog_decision
        self.dialog_decision = None
        accepted = decision is not None and decision.accept
        self.record(
            BrowserEvent(
                kind='dialog',
                level='warning',
                message=f'{dialog.type} {"accepted" if accepted else "dismissed"}: {dialog.message}',
            )
        )
        self._spawn(self._answer_dialog(dialog, decision))

    async def _answer_dialog(self, dialog: _Dialog, decision: _DialogDecision | None) -> None:
        """Accept or dismiss a dialog. Errors are swallowed: the page may already be gone."""
        try:
            if decision is not None and decision.accept:
                await dialog.accept(decision.prompt_text)
            else:
                await dialog.dismiss()
        except PlaywrightError:
            pass

    async def open_tab(self) -> _Page:
        """Open a blank tab in the same context, wire it, and make it active."""
        assert self._context is not None
        page = await self._context.new_page()
        self._wire_page(page)
        self.pages.append(page)
        self.page = page
        return page

    async def activate(self, page: _Page) -> _Page:
        """Make `page` the tab every tool acts on."""
        self.page = page
        await page.bring_to_front()
        return page

    async def close_tab(self, page: _Page) -> None:
        """Close a tab, moving the active pointer off it when it was the active one.

        The removal is repeated here rather than left to the `close` handler:
        whether Playwright has delivered that event yet must not decide which tab
        the next tool call acts on.
        """
        await page.close()
        self._on_page_closed(page)

    async def __aenter__(self) -> Self:
        """Arm the session. Cheap by design: no driver and no browser start here.

        The driver flag is cleared as well as set: a session entered a second time
        gets a fresh, unentered context manager, and a flag left over from the
        first use would have teardown exit a driver that was never started. A
        recorded launch failure is cleared for the same reason: it described the
        previous use, and keeping it would refuse every tool call of this one
        without ever trying to start a browser. So is an armed dialog decision,
        which would otherwise answer the first dialog of this use with an intent
        expressed during the last one.
        """
        self._driver_cm = async_playwright()
        self._driver = None
        self._driver_entered = False
        self.launch_error = None
        self.dialog_decision = None
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Release everything the session started, without masking the run's own failure.

        Page-event tasks are cancelled first so none outlives the browser it acts
        on. A `close()` failure still exits the driver, and when the run is
        already unwinding the teardown error is dropped: the exception the caller
        is carrying is the one worth reporting.
        """
        driver_cm = self._driver_cm
        self._driver_cm = None
        self._driver = None
        if self._event_tasks:
            for task in self._event_tasks:
                task.cancel()
            await asyncio.gather(*self._event_tasks, return_exceptions=True)
            self._event_tasks.clear()
        if driver_cm is None:  # pragma: no cover
            return
        try:
            browser = self._browser
            if browser is not None:
                self.page = None
                self.pages = []
                self._browser = None
                try:
                    await browser.close()
                finally:
                    await driver_cm.__aexit__(None, None, None)
            elif self._driver_entered:
                await driver_cm.__aexit__(None, None, None)
        except Exception:
            if exc_type is None:
                raise


class PlaywrightBrowserToolset(FunctionToolset[AgentDepsT]):
    """Async Playwright-backed browser tools: navigate, interact, extract, screenshot, run JS, inspect.

    The tools read the active page from a shared `PlaywrightBrowserSession`, which
    `PlaywrightBrowser.wrap_run` populates lazily on the first tool call. Use the
    toolset through `PlaywrightBrowser` rather than directly; construct it
    directly (with a `session` whose `page` you set) only to drive tools against a
    page double.

    Page text is extracted with Playwright itself (`inner_text`), across the main
    frame and any child frames, and every textual result is truncated to
    `max_content_tokens`; no HTML-to-Markdown dependency is pulled in. `screenshot`
    returns a `ToolReturn` carrying `BinaryContent` so vision models see the image
    natively instead of a base64 string bloating the text context.

    Each operation runs inside an OpenTelemetry span, and browser events the page
    produced during it (console output, responses, refused requests) are attached
    to that span as events.
    """

    def __init__(
        self,
        *,
        session: PlaywrightBrowserSession,
        screenshot_on_navigate: bool = False,
        max_content_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
        action_timeout_ms: int = DEFAULT_ACTION_TIMEOUT_MS,
        navigation_timeout_ms: int = DEFAULT_NAVIGATION_TIMEOUT_MS,
    ) -> None:
        if max_content_tokens < 0:
            raise ValueError('max_content_tokens must be greater than or equal to 0')
        if action_timeout_ms < 0:
            raise ValueError('action_timeout_ms must be greater than or equal to 0')
        if navigation_timeout_ms < 0:
            raise ValueError('navigation_timeout_ms must be greater than or equal to 0')
        super().__init__(id='playwright')
        self._session = session
        # The session's policy, never a separate one. These checks and the route
        # guard the session installs are two layers of one decision: a toolset
        # holding its own policy could refuse what the guard allows, or -- worse --
        # let the guard send a request the tools would have refused, since the
        # guard acts first and the tool check only bounces the page afterwards.
        # Both layers therefore ask the session, which owns the policy.
        self._screenshot_on_navigate = screenshot_on_navigate
        self._max_content_tokens = max_content_tokens
        self._action_timeout_ms = action_timeout_ms
        self._navigation_timeout_ms = navigation_timeout_ms
        self._operation_lock = asyncio.Lock()
        self.add_function(self.navigate, name='navigate')
        self.add_function(self.click, name='click')
        self.add_function(self.type_text, name='type_text')
        self.add_function(self.press_key, name='press_key')
        self.add_function(self.select_option, name='select_option')
        self.add_function(self.hover, name='hover')
        self.add_function(self.screenshot, name='screenshot')
        self.add_function(self.get_text, name='get_text')
        self.add_function(self.scroll, name='scroll')
        self.add_function(self.go_back, name='go_back')
        self.add_function(self.go_forward, name='go_forward')
        self.add_function(self.execute_js, name='execute_js')
        self.add_function(self.wait_for, name='wait_for')
        self.add_function(self.snapshot, name='snapshot')
        self.add_function(self.tabs, name='tabs')
        self.add_function(self.handle_next_dialog, name='handle_next_dialog')
        self.add_function(self.console_messages, name='console_messages')
        self.add_function(self.network_requests, name='network_requests')

    async def _frame_text(self, page: _Page, budget_ms: int) -> list[str]:
        """Return the text of each child frame that has any, newest layout first.

        Page-level reads stop at the frame boundary, so an embedded schedule,
        checkout step, or chat widget is absent from `inner_text` even though the
        model can see it in a `snapshot`. Reading the frames directly closes that
        gap for every tool that returns page text.

        Failures are skipped rather than raised: a frame can detach mid-read, and
        a missing embed should not turn a successful action into an error.
        """
        texts: list[str] = []

        async def sweep() -> None:
            for frame in page.frames[1:]:
                try:
                    text = await frame.inner_text('body', timeout=budget_ms)
                except PlaywrightError:
                    continue
                if text.strip():
                    texts.append(f'[frame {_without_credentials(frame.url)}]\n{text}')

        try:
            await asyncio.wait_for(sweep(), budget_ms / 1000)
        except asyncio.TimeoutError:
            pass
        return texts

    async def _page_text(self, page: _Page, timeout_ms: int | None = None) -> str:
        """Return the visible text of `page` and its child frames, truncated to the token budget.

        The page is passed in rather than re-acquired: every caller already holds
        the one it just acted on, and re-entering `ensure_page` here would be a
        second path into the launch machinery, raising `RuntimeError` where the
        callers only guard against `PlaywrightError`.
        """
        text = await page.inner_text('body', timeout=timeout_ms)
        frames = await self._frame_text(page, self._frame_budget(timeout_ms))
        return self._truncate_output('\n\n'.join([text, *frames]))

    def _frame_budget(self, timeout_ms: int | None) -> int:
        """Return the deadline for the child-frame sweep.

        The sweep is capped so one unresponsive embed cannot spend the whole
        action budget, and it is capped again by a shorter caller deadline, so a
        tight `timeout_ms` is not overrun by the frames the caller never asked
        about. A caller with no deadline at all still gets the cap.
        """
        if timeout_ms is None or timeout_ms == 0:
            return _FRAME_TEXT_BUDGET_MS
        return min(timeout_ms, _FRAME_TEXT_BUDGET_MS)

    def _truncate_output(self, text: str) -> str:
        """Apply the configured token budget to a model-facing textual result."""
        return _truncate(text, self._max_content_tokens * _CHARS_PER_TOKEN)

    def _error(self, message: str) -> str:
        """Bound an error result and record the failure on the operation's span.

        Tools return their failures as strings the model can act on, so a span
        that took its outcome from whether an exception escaped would report every
        one of them as a success.
        """
        span = self._session.operation_span
        if span is not None and span.is_recording():
            span.set_attribute('browser.outcome', 'error')
        return self._truncate_output(message)

    def _truncate_output_keeping(self, text: str, note: str) -> str:
        """Bound `text` and `note` together, giving `note` its room first.

        Appending to an already-budgeted string and re-truncating drops the
        appended part, which is the opposite of what a note reporting a dropped
        result should do. A note too large for the budget on its own wins the
        whole budget: it carries why the result is missing.
        """
        budget = self._max_content_tokens * _CHARS_PER_TOKEN
        separator = '\n\n'
        room = budget - len(note) - len(separator)
        if room <= 0:
            return _truncate(note, budget)
        return f'{_truncate(text, room)}{separator}{note}'

    def _playwright_error(self, action: str, exc: PlaywrightError, timeout_ms: int) -> str:
        """Map a Playwright error to a bounded, model-actionable string.

        A timeout, strict-mode match count, `net::ERR_*` code, or closed target is
        a routine event when a model drives a browser, so it is returned as a tool
        result the model can react to rather than raised to abort the run.
        `timeout_ms` is the deadline the call actually ran under, so a per-call
        override is reported accurately.

        Playwright quotes the URL it was working on, call log included, so the
        interpolated message is cleaned like any other URL that reaches the model.
        """
        if isinstance(exc, PlaywrightTimeoutError):
            return (
                f'Error: {action} timed out after {timeout_ms}ms. The element may not exist or '
                'the page may be slow; try a different selector, or navigate again. '
                'Content inside an iframe needs an `aria-ref=` handle from `snapshot`, not a CSS selector.'
            )
        if isinstance(exc, TargetClosedError):
            if not self._session.pages:
                # The active pointer deliberately stays on the closed page rather than
                # falling back to a fresh browser, which would drop the session's
                # cookies and history without saying so.
                return self._error(f"Error: {action} failed: the active tab has closed. Open one with tabs('new').")
            return (
                f'Error: {action} failed: the browser or page was closed unexpectedly. '
                'Browser tools may be unavailable for the rest of this run.'
            )
        return f'Error: {action} failed: {_without_credentials(str(exc))}'

    def _deadlines(self, timeout_ms: int | None) -> _Deadlines:
        """Resolve the deadlines an operation runs under.

        A per-call override replaces both: the model asked one call to take at
        most this long, and a navigation triggered by that call is part of it.
        """
        started = monotonic()
        if timeout_ms is not None:
            return _Deadlines(action_ms=timeout_ms, navigation_ms=timeout_ms, started=started)
        return _Deadlines(action_ms=self._action_timeout_ms, navigation_ms=self._navigation_timeout_ms, started=started)

    async def _await_with_timeout(self, awaitable: Awaitable[_T], timeout_ms: int) -> _T:
        """Bound an operation whose Playwright API has no `timeout` parameter."""
        if timeout_ms == 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            # asyncio.wait_for raises asyncio.TimeoutError, which is a distinct class
            # from the builtin TimeoutError on Python 3.10 (aliased only from 3.11).
            raise PlaywrightTimeoutError(f'Timeout {timeout_ms}ms exceeded.') from exc

    def _timeout_error(self, timeout_ms: int | None) -> str | None:
        """Return a bounded error when a per-call timeout override is not positive, else `None`.

        `0` means "no deadline" to both Playwright and `_await_with_timeout`, so it
        stays available to the developer as a capability default but is refused as
        a per-call override: the model chooses that argument, and an injected page
        could otherwise ask for an unbounded call that never returns.
        """
        if timeout_ms is not None and timeout_ms <= 0:
            return self._truncate_output('Error: timeout_ms must be greater than 0.')
        return None

    def _oversized_screenshot_error(self, png: bytes) -> str | None:
        """Return a bounded error when a capture exceeds the image size limit, else `None`."""
        if len(png) <= _MAX_SCREENSHOT_BYTES:
            return None
        return (
            f'Error: screenshot is {len(png)} bytes, over the {_MAX_SCREENSHOT_BYTES} byte image limit; '
            'capture the viewport (full_page=False) or scroll and capture sections instead.'
        )

    async def _enforce_navigation_policy(self, page: _Page, action: str, timeout: int) -> str | None:
        """After an action, bounce to `about:blank` if the page left the permitted set.

        Navigation can happen through clicks, `execute_js` setting
        `location.href`, or history moves, so the current URL is re-checked --
        against both the allowlist and the private-address block -- after each
        such action. When it is disallowed the page is moved to `about:blank`
        and an error string is returned, so disallowed content never reaches the
        model. The network-level route guard installed by
        `PlaywrightBrowser.wrap_run` is the primary boundary; this is the second
        layer.

        The bounce runs under the caller's resolved `timeout` so a short
        per-call `timeout_ms` is not silently replaced by Playwright's default.

        A navigation the browser could not complete lands on Chromium's own error
        page, which is not a destination the policy has an opinion about: reporting
        it as an allowlist miss names `chrome-error://chromewebdata/` as the domain
        the model reached. That case reports the refused or failed request instead.
        """
        if page.url.startswith(_ERROR_PAGE_SCHEME):
            await page.goto(_BLANK_PAGE, timeout=timeout)
            failure = self._session.failure_since_operation_start()
            cause = 'the navigation did not complete' if failure is None else f'{failure.message}: {failure.url}'
            return self._error(f'Error: {action} loaded no page ({cause}); the browser is back at about:blank.')
        reason = await self._session.decide(EgressRequest(url=page.url, kind='navigation'))
        if reason is None:
            return None
        blocked = _without_credentials(page.url)
        await page.goto(_BLANK_PAGE, timeout=timeout)
        return self._error(f'Error: {action} reached a {reason}: {blocked}')

    async def _in_operation(
        self,
        action: str,
        timeout_ms: int | None,
        body: Callable[[_Page, _Deadlines], Awaitable[_T]],
        *,
        governed_by_navigation: bool = False,
    ) -> _T | str:
        """Run `body` as one complete operation on the shared page.

        Every tool needs the same things in the same order: exclusive use of the
        page, a per-call deadline that is either absent or positive, a span the
        page's own events can attach to, the page itself (launching Chromium on
        the first call), the resolved deadlines, and a Playwright failure turned
        into a result the model can read instead of an exception that ends the run.

        Acquiring the page is inside the guarded region: starting or attaching to
        a browser can fail the same way an action can -- an endpoint that is gone,
        a browser that was never installed -- and the model can act on either only
        if it arrives as a result rather than as an exception that ends the run.

        `governed_by_navigation` names which deadline a failure is reported
        against, so a navigation timeout is not described with the action budget.

        An argument check that must not launch a browser belongs in the tool,
        before this call -- see `_refuse`.
        """
        async with self._operation_lock:
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            deadlines = self._deadlines(timeout_ms)
            # The configured budget, not what is left of it: the error copy tells the
            # model how long the call was allowed, which a countdown would understate.
            reported = deadlines.navigation_ms if governed_by_navigation else deadlines.action_ms
            with self._session.tracer.start_as_current_span(
                f'browser {action}',
                # The outcome starts at `ok` and every failure path overwrites it,
                # because a tool failure is a returned string, not an exception.
                attributes={'browser.action': action, 'browser.timeout_ms': reported, 'browser.outcome': 'ok'},
            ) as span:
                self._session.operation_span = span
                self._session.operation_events_mark = self._session.events_recorded
                page: _Page | None = None
                try:
                    page = await self._session.ensure_page()
                    return await body(page, deadlines)
                except PlaywrightError as exc:
                    return self._error(self._playwright_error(action, exc, reported))
                except BrowserUnavailableError as exc:
                    # Forgotten once reported, so the next call launches again rather
                    # than refusing on a failure the agent may already have fixed.
                    self._session.launch_error = None
                    return self._error(str(exc))
                except BaseException:
                    span.set_attribute('browser.outcome', 'error')
                    raise
                finally:
                    # Recorded whether or not the operation succeeded: which page a
                    # failed action was on is the part that makes it diagnosable.
                    # The session's active page rather than the one acquired above,
                    # because `tabs` can have moved it, and the span should name the
                    # tab the operation left rather than the one it started on.
                    ended_on = self._session.page if self._session.page is not None else page
                    if ended_on is not None:
                        span.set_attribute('url.full', _without_credentials(ended_on.url))
                    self._session.operation_span = None

    async def _refuse(self, action: str, timeout_ms: int | None, message: str) -> str:
        """Return a bounded refusal without acquiring a page.

        A rejected argument must not start a browser, so these refusals happen
        before `_in_operation`. The deadline is still validated first, so a call
        that is wrong in both ways reports the same error either way.

        The refusal still opens the operation's span: a call the egress policy or
        an argument check turned away is something the agent did, and a trace
        showing only the calls that reached a page would not show it. There is no
        page, so no `url.full`.
        """
        async with self._operation_lock:
            if (error := self._timeout_error(timeout_ms)) is not None:
                return error
            with self._session.tracer.start_as_current_span(
                f'browser {action}', attributes={'browser.action': action, 'browser.outcome': 'ok'}
            ) as span:
                self._session.operation_span = span
                try:
                    return self._error(message)
                finally:
                    self._session.operation_span = None

    async def _settle(self, page: _Page, action: str, deadlines: _Deadlines) -> str | None:
        """Let the navigation finish, then re-check where it landed.

        The order is the point: the policy has to see the settled URL, because
        reading it before the load completes checks the page the action started
        from. Returns the bounced error, or `None` when the result is permitted.
        """
        await page.wait_for_load_state('domcontentloaded', timeout=deadlines.navigation)
        return await self._enforce_navigation_policy(page, action, deadlines.navigation)

    async def navigate(self, url: str, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Navigate to a URL and return the page's title and visible text.

        Args:
            url: Full URL to navigate to (e.g. `https://example.com`).
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page URL, title, and visible text, including the text of any
            embedded frames. When `screenshot_on_navigate` is set, a screenshot is
            attached as image content for vision models.
        """
        reason = await self._session.decide(EgressRequest(url=url, kind='navigation'))
        if reason is not None:
            # Refused before `_in_operation`, so a disallowed URL never launches Chromium.
            # Not redacted, unlike the URLs below: this one is the argument the model
            # just passed, so echoing it tells it nothing it did not write, and a
            # refusal that hid the userinfo would hide what was refused.
            return await self._refuse('navigate', timeout_ms, f'Error: {reason}: {url}')

        async def _navigate(page: _Page, deadlines: _Deadlines) -> str | ToolReturn[str]:
            await page.goto(url, timeout=deadlines.navigation)
            if (blocked := await self._settle(page, 'navigate', deadlines)) is not None:
                return blocked
            # Everything past the load runs on the navigation budget: the action one
            # is already spent by any page that took more than a moment to arrive.
            title = await self._await_with_timeout(page.title(), deadlines.navigation)
            text = await self._page_text(page, deadlines.navigation)
            result = self._truncate_output(f'URL: {_without_credentials(page.url)}\nTitle: {title}\n\n{text}')
            if not self._screenshot_on_navigate:
                return result
            png = await page.screenshot(timeout=deadlines.navigation)
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._truncate_output_keeping(result, oversized)
            return ToolReturn(result, content=[BinaryContent(data=png, media_type='image/png')])

        return await self._in_operation('navigate', timeout_ms, _navigate, governed_by_navigation=True)

    async def click(self, selector: str, timeout_ms: int | None = None) -> str:
        """Click an element on the current page.

        Args:
            selector: A CSS selector (e.g. `button#submit`), an `aria-ref=` handle
                from `snapshot` (the most reliable way to target an element, and
                the only one that reaches inside an iframe), or pixel coordinates
                as `'x,y'` (e.g. `'450,300'`).
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the click.
        """
        parts = selector.split(',', 1)
        coordinates: tuple[int, int] | None = None
        if len(parts) == 2:
            try:
                coordinates = (int(parts[0]), int(parts[1]))
            except ValueError:
                pass

        async def _click(page: _Page, deadlines: _Deadlines) -> str:
            if coordinates is not None:
                # `Mouse.click` takes no `timeout`, so it is bounded externally like
                # the coordinate branch of `scroll`.
                await self._await_with_timeout(page.mouse.click(*coordinates), deadlines.action)
            else:
                await page.click(selector, timeout=deadlines.action)
            if (blocked := await self._settle(page, 'click', deadlines)) is not None:
                return blocked
            # The read follows a navigation the click may have triggered, so it takes
            # the budget that allowed for it rather than what is left of the action's.
            text = await self._page_text(page, deadlines.navigation)
            return self._truncate_output(f"Clicked '{selector}'. URL: {_without_credentials(page.url)}\n\n{text}")

        return await self._in_operation('click', timeout_ms, _click)

    async def type_text(self, selector: str, text: str, sequential: bool = False, timeout_ms: int | None = None) -> str:
        """Type text into an input field, replacing any existing value.

        This sets the field's value; it does not submit. Use `press_key('Enter')`
        for a form or search box that submits on Enter.

        Args:
            selector: CSS selector for the target input element, or an `aria-ref=`
                handle from `snapshot`.
            text: Text to type into the field.
            sequential: Send the text as individual key presses rather than setting
                the value in one step. Slower, and what fields that react to
                keystrokes need: autocomplete and type-ahead widgets, masked or
                formatted inputs, and editors that ignore a value set directly.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after typing.
        """

        async def _type_text(page: _Page, deadlines: _Deadlines) -> str:
            if sequential:
                # Cleared first: `press_sequentially` types at the cursor, so without
                # this the field would keep whatever it already held and the two
                # modes would not mean the same thing.
                await page.fill(selector, '', timeout=deadlines.action)
                await page.locator(selector).press_sequentially(text, timeout=deadlines.action)
            else:
                await page.fill(selector, text, timeout=deadlines.action)
            return self._truncate_output(f"Typed into '{selector}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('type_text', timeout_ms, _type_text)

    async def press_key(self, key: str, selector: str | None = None, timeout_ms: int | None = None) -> str:
        """Press a keyboard key, optionally focusing an element first.

        Reaches what typing cannot: submitting a search box with `Enter`, closing
        an overlay with `Escape`, moving between fields with `Tab`.

        Args:
            key: A Playwright key name, e.g. `Enter`, `Escape`, `Tab`, `ArrowDown`,
                `Control+a`.
            selector: Element to focus before pressing. Omit to send the key to
                whatever currently has focus.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the key press.
        """

        async def _press_key(page: _Page, deadlines: _Deadlines) -> str:
            if selector is None:
                await self._await_with_timeout(page.keyboard.press(key), deadlines.action)
            else:
                await page.press(selector, key, timeout=deadlines.action)
            # A key press is a common way to trigger navigation (Enter in a search
            # box), so the result is settled and re-checked like a click.
            if (blocked := await self._settle(page, 'press_key', deadlines)) is not None:
                return blocked
            return self._truncate_output(f"Pressed '{key}'.\n\n{await self._page_text(page, deadlines.navigation)}")

        return await self._in_operation('press_key', timeout_ms, _press_key)

    async def select_option(self, selector: str, values: list[str], timeout_ms: int | None = None) -> str:
        """Choose one or more options in a `<select>` dropdown.

        A native dropdown does not open as page content, so clicking it does not
        expose its options.

        Args:
            selector: CSS selector for the `<select>` element, or an `aria-ref=`
                handle from `snapshot`.
            values: Option values (or labels) to select. Pass one value for a
                single-choice dropdown.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after the selection.
        """

        async def _select_option(page: _Page, deadlines: _Deadlines) -> str:
            selected = await page.select_option(selector, values, timeout=deadlines.action)
            if (blocked := await self._settle(page, 'select_option', deadlines)) is not None:
                return blocked
            text = await self._page_text(page, deadlines.navigation)
            return self._truncate_output(f"Selected {selected} in '{selector}'.\n\n{text}")

        return await self._in_operation('select_option', timeout_ms, _select_option)

    async def hover(self, selector: str, timeout_ms: int | None = None) -> str:
        """Hover over an element, revealing menus and tooltips that appear on hover.

        Args:
            selector: CSS selector for the element, or an `aria-ref=` handle from
                `snapshot`.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after hovering.
        """

        async def _hover(page: _Page, deadlines: _Deadlines) -> str:
            await page.hover(selector, timeout=deadlines.action)
            return self._truncate_output(f"Hovered '{selector}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('hover', timeout_ms, _hover)

    async def screenshot(self, full_page: bool = False, timeout_ms: int | None = None) -> str | ToolReturn[str]:
        """Capture a screenshot of the current page.

        Args:
            full_page: Capture the full scrollable page when `True`, else the
                current viewport.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A short note with the page URL, and the PNG as image content so
            vision models can see it.
        """

        async def _screenshot(page: _Page, deadlines: _Deadlines) -> str | ToolReturn[str]:
            png = await page.screenshot(full_page=full_page, timeout=deadlines.action)
            if (oversized := self._oversized_screenshot_error(png)) is not None:
                return self._error(oversized)
            return ToolReturn(
                self._truncate_output(f'Screenshot captured. URL: {_without_credentials(page.url)}'),
                content=[BinaryContent(data=png, media_type='image/png')],
            )

        return await self._in_operation('screenshot', timeout_ms, _screenshot)

    async def get_text(self, selector: str | None = None, timeout_ms: int | None = None) -> str:
        """Extract text from the page or a specific element.

        Args:
            selector: CSS selector to read, or an `aria-ref=` handle from
                `snapshot`. A CSS selector matches the main frame only; reading
                inside an iframe needs the `aria-ref=` handle. Omit for the whole
                page's visible text, which includes embedded frames.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The element's text, or the full page's visible text when no selector
            is given.
        """

        async def _get_text(page: _Page, deadlines: _Deadlines) -> str:
            if not selector:
                return await self._page_text(page, deadlines.action)
            try:
                text = await page.inner_text(selector, timeout=deadlines.action)
            except Exception as exc:
                # Named after the selector rather than the action: which selector failed
                # is the part the model needs to act on.
                return self._error(f"Error getting text from '{selector}': {_without_credentials(str(exc))}")
            return self._truncate_output(text)

        return await self._in_operation('get_text', timeout_ms, _get_text)

    async def scroll(
        self, direction: str, x: int | None = None, y: int | None = None, timeout_ms: int | None = None
    ) -> str:
        """Scroll the page in a direction.

        Args:
            direction: One of `'up'`, `'down'`, `'left'`, `'right'` -- about one
                screenful each -- or `'top'`/`'bottom'` for the ends of the page.
            x: Optional x coordinate to scroll from (paired with `y`), which scrolls
                the element under that point by a fixed step instead of the page.
                Ignored for `'top'` and `'bottom'`, which always move the page.
            y: Optional y coordinate to scroll from (paired with `x`).
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The page's visible text after scrolling.
        """
        # A screenful, less a sliver of overlap, so consecutive scrolls do not skip
        # a line: the previous fixed 300px moved less than half a default viewport,
        # which cost several calls (and a full page read each) per screen.
        moves = {
            'up': 'window.scrollBy(0, -window.innerHeight * 0.9)',
            'down': 'window.scrollBy(0, window.innerHeight * 0.9)',
            'left': 'window.scrollBy(-window.innerWidth * 0.9, 0)',
            'right': 'window.scrollBy(window.innerWidth * 0.9, 0)',
            'top': 'window.scrollTo(0, 0)',
            'bottom': 'window.scrollTo(0, document.body.scrollHeight)',
        }
        deltas: dict[str, tuple[int, int]] = {
            'up': (0, -300),
            'down': (0, 300),
            'left': (-300, 0),
            'right': (300, 0),
        }
        move = moves.get(direction.lower())
        if move is None:
            return await self._refuse(
                'scroll', timeout_ms, f'Error: invalid direction {direction!r}; use up/down/left/right/top/bottom'
            )
        delta = deltas.get(direction.lower())

        async def _scroll(page: _Page, deadlines: _Deadlines) -> str:
            if x is not None and y is not None and delta is not None:
                # `Mouse.move`/`Mouse.wheel` take no `timeout`, so they are bounded
                # externally like `evaluate` below: an unbounded await here would
                # hold the operation lock for the rest of the run.
                await self._await_with_timeout(page.mouse.move(x, y), deadlines.action)
                await self._await_with_timeout(page.mouse.wheel(*delta), deadlines.action)
            else:
                # `evaluate` has no `timeout` parameter and hangs if the page's
                # main thread is blocked, so it is bounded externally.
                reported = await self._await_with_timeout(page.evaluate(_scroll_script(move)), deadlines.action)
                return self._truncate_output(
                    f'Scrolled {direction}. {_scroll_position(reported)}\n\n{await self._page_text(page, deadlines.action)}'
                )
            return self._truncate_output(f'Scrolled {direction}.\n\n{await self._page_text(page, deadlines.action)}')

        return await self._in_operation('scroll', timeout_ms, _scroll)

    async def go_back(self, timeout_ms: int | None = None) -> str:
        """Navigate back in the browser history.

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The previous page's visible text, or a note when there is no history
            entry to go back to.
        """

        async def _go_back(page: _Page, deadlines: _Deadlines) -> str:
            if await page.go_back(timeout=deadlines.navigation) is None:
                return self._truncate_output('No previous page in browser history.')
            if (blocked := await self._settle(page, 'go_back', deadlines)) is not None:
                return blocked
            text = await self._page_text(page, deadlines.navigation)
            return self._truncate_output(f'Went back. URL: {_without_credentials(page.url)}\n\n{text}')

        return await self._in_operation('go_back', timeout_ms, _go_back, governed_by_navigation=True)

    async def go_forward(self, timeout_ms: int | None = None) -> str:
        """Navigate forward in the browser history.

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The next page's visible text, or a note when there is no history entry
            to go forward to.
        """

        async def _go_forward(page: _Page, deadlines: _Deadlines) -> str:
            if await page.go_forward(timeout=deadlines.navigation) is None:
                return self._truncate_output('No next page in browser history.')
            if (blocked := await self._settle(page, 'go_forward', deadlines)) is not None:
                return blocked
            text = await self._page_text(page, deadlines.navigation)
            return self._truncate_output(f'Went forward. URL: {_without_credentials(page.url)}\n\n{text}')

        return await self._in_operation('go_forward', timeout_ms, _go_forward, governed_by_navigation=True)

    async def execute_js(self, script: str, timeout_ms: int | None = None) -> str:
        """Evaluate a JavaScript expression and return its result.

        Args:
            script: JavaScript expression to evaluate, e.g. `document.title`.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A string result as-is, objects/arrays as JSON, `null`/`undefined` as
            `'undefined'`, or `JS error: ...` when evaluation raises.
        """

        async def _execute_js(page: _Page, deadlines: _Deadlines) -> str:
            try:
                # `evaluate` waits for a returned promise and has no `timeout`
                # parameter, so a never-resolving promise (or a blocked main
                # thread) would hold the operation lock forever without the
                # external deadline.
                result = await self._await_with_timeout(page.evaluate(script), deadlines.action)
            except (PlaywrightTimeoutError, TargetClosedError):
                # `evaluate` raises for script exceptions too, so the browser's own
                # failures are picked out by type and left to the operation's mapper.
                raise
            except Exception as exc:
                return self._error(f'JS error: {_without_credentials(str(exc))}')
            try:
                blocked = await self._enforce_navigation_policy(page, 'execute_js', deadlines.navigation)
            except PlaywrightError as exc:
                # Kept out of the try above on purpose: that one ends early so a
                # script exception maps to 'JS error', which a failed bounce is not.
                return self._error(self._playwright_error('execute_js', exc, deadlines.navigation_ms))
            if blocked is not None:
                return blocked
            if result is None:
                return self._truncate_output('undefined')
            if isinstance(result, str):
                return self._truncate_output(result)
            try:
                return self._truncate_output(json.dumps(result, default=str))
            except TypeError:  # pragma: no cover
                return self._truncate_output(str(result))

        return await self._in_operation('execute_js', timeout_ms, _execute_js)

    async def wait_for(
        self,
        selector: str | None = None,
        text: str | None = None,
        gone: bool = False,
        timeout_ms: int | None = None,
    ) -> str:
        """Wait for dynamic content to appear or disappear, then return the page's visible text.

        Pass exactly one of `selector` or `text`. Use this after an action that
        loads content asynchronously, so a following read sees the settled page.
        The wait covers embedded frames as well as the main page.

        Args:
            selector: CSS selector (or an `aria-ref=` handle) to wait for.
            text: Visible text to wait for, matched with Playwright's text engine.
            gone: Wait for the match to disappear instead of appear. This is how a
                loading spinner, a progress bar, or an overlay is waited out when
                what replaces it is not known in advance.
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            A short confirmation followed by the page's visible text, or a bounded
            error string when neither/both arguments are given or the wait times out.
        """
        invalid = 'Error: wait_for requires exactly one of selector or text.'
        if text is not None:
            if selector is not None:
                return await self._refuse('wait_for', timeout_ms, invalid)
            # Quoted inside `:text()` rather than interpolated after `text=`, where
            # the selector grammar would read a `>>` in the text as a chain and a
            # quote as an exact-match toggle. `:text` keeps the substring matching
            # `text=` has, and the quotes make the value inert.
            escaped = text.replace('\\', '\\\\').replace('"', '\\"')
            query = f':text("{escaped}")'
        elif selector is not None:
            query = selector
        else:
            return await self._refuse('wait_for', timeout_ms, invalid)

        # The engine query matches; the caller's own argument is what the result
        # echoes, so the model reads back what it asked for.
        label = selector if selector is not None else text

        async def _wait_for(page: _Page, deadlines: _Deadlines) -> str:
            await self._wait_in_any_frame(page, query, deadlines.action, gone=gone)
            settled = 'Gone' if gone else 'Found'
            return self._truncate_output(f"{settled} '{label}'.\n\n{await self._page_text(page, deadlines.action)}")

        return await self._in_operation('wait_for', timeout_ms, _wait_for)

    async def _wait_in_any_frame(self, page: _Page, query: str, timeout_ms: int, *, gone: bool) -> None:
        """Wait until `query` matches in the main page or in any child frame, or matches nowhere.

        `page.wait_for_selector` matches the main frame only, so content arriving
        inside an embed would never satisfy a wait even though the model can see
        that frame in a snapshot.

        The two directions combine the frames differently. Appearing is a race:
        every frame is watched at once and the first match wins, and when none
        matches the failure of whichever wait finished last is raised, which is the
        timeout the caller is waiting on anyway. Disappearing has to hold
        everywhere at once, so those are awaited together -- a frame that never
        contained the element reports it hidden immediately, and a race would
        settle on that frame before the one that does contain it has let go.
        """
        state: _WaitState = 'hidden' if gone else 'visible'
        waits = [page.wait_for_selector(query, timeout=timeout_ms, state=state)]
        waits.extend(frame.wait_for_selector(query, timeout=timeout_ms, state=state) for frame in page.frames[1:])
        tasks = [asyncio.ensure_future(wait) for wait in waits]
        if gone:
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            return
        failure: BaseException = PlaywrightTimeoutError(f'Timeout {timeout_ms}ms exceeded.')
        try:
            while tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                # Every finished wait is read, including when an earlier one in the
                # same batch already matched: an exception nobody retrieves surfaces
                # later, out of context, as an unhandled task error.
                matched = False
                for task in done:
                    error = task.exception()
                    if error is None:
                        matched = True
                    else:
                        failure = error
                if matched:
                    return
        finally:
            for task in tasks:
                task.cancel()
        raise failure

    async def snapshot(self, timeout_ms: int | None = None) -> str:
        """Return the page's accessibility tree with `aria-ref` handles for targeting.

        The snapshot is the structured, low-cost way to read the page and obtain
        `aria-ref=eN` handles; pass one back to `click`, `type_text`, `hover` or
        `get_text` to target an element reliably. It includes iframe content that
        CSS selectors cannot reach: refs inside an embed look like `f1e4`, and they
        resolve in the frame they came from. Use `screenshot` only for visual
        checks (charts, layout).

        Args:
            timeout_ms: Override the default timeout for this call, in milliseconds.
                Must be greater than 0.

        Returns:
            The accessibility tree (truncated to the token budget), or a bounded
            error string when the snapshot fails.
        """

        async def _snapshot(page: _Page, deadlines: _Deadlines) -> str:
            return self._truncate_output(await page.aria_snapshot(mode='ai', timeout=deadlines.action))

        return await self._in_operation('snapshot', timeout_ms, _snapshot)

    async def tabs(self, action: str = 'list', index: int | None = None) -> str:
        """List the open tabs, or switch to, open, or close one.

        A site opens a second tab for a `target="_blank"` link, a sign-in popup,
        or a payment step. Every other tool acts on the active tab, so a flow that
        continues in a new one needs `select` before it can be read or driven. A
        tab appears a moment after the action that opened it, so a `list` that does
        not show one you expected is worth repeating once.

        Args:
            action: One of `'list'`, `'select'`, `'close'`, `'new'`. A new tab
                starts blank and active; load it with `navigate`.
            index: Which tab `'select'` and `'close'` act on, numbered as `'list'`
                shows them. `'close'` defaults to the active tab.

        Returns:
            The tab list, or a confirmation followed by the active tab's visible
            text, or a bounded error string.
        """
        if action not in ('list', 'select', 'close', 'new'):
            return await self._refuse(
                'tabs', None, f'Error: unknown tabs action {action!r}; use list, select, close or new.'
            )

        async def _tabs(page: _Page, deadlines: _Deadlines) -> str:
            session = self._session
            if action == 'list':
                return self._truncate_output(await self._describe_tabs(session.pages, page, deadlines))
            if action == 'new':
                if len(session.pages) >= _MAX_TABS:
                    return self._error(f'Error: the tab limit of {_MAX_TABS} is reached. Close one first.')
                # The three tab mutations drive Playwright calls that take no
                # `timeout` of their own, so the operation's deadline is applied
                # around them rather than left to the driver.
                await self._await_with_timeout(session.open_tab(), deadlines.action)
                return self._truncate_output(
                    f'Opened blank tab {len(session.pages) - 1} and made it active. Load it with navigate.'
                )
            if page not in session.pages:
                # A page can close itself, and the session moves the active pointer
                # to whatever is left. Nothing was, so there is no tab to act on.
                return self._error("Error: the active tab has closed. Open one with tabs('new').")
            target = session.pages.index(page) if index is None else index
            if not 0 <= target < len(session.pages):
                return self._error(f'Error: no tab {target}. {len(session.pages)} open; list them with tabs.')
            if action == 'close':
                if len(session.pages) == 1:
                    return self._error('Error: the last tab cannot be closed.')
                closing = session.pages[target]
                await self._await_with_timeout(session.close_tab(closing), deadlines.action)
                # The session repoints only when the tab that closed was the active
                # one; otherwise the operation keeps acting on the page it started on.
                active = page if closing is not page else session.pages[-1]
                return self._truncate_output(f'Closed tab {target}. Active tab is now {session.pages.index(active)}.')
            selected = await self._await_with_timeout(session.activate(session.pages[target]), deadlines.action)
            if (blocked := await self._enforce_navigation_policy(selected, 'tabs', deadlines.navigation)) is not None:
                return blocked
            text = await self._page_text(selected, deadlines.action)
            url = _without_credentials(selected.url)
            return self._truncate_output(f'Selected tab {target}. URL: {url}\n\n{text}')

        return await self._in_operation('tabs', None, _tabs)

    async def _describe_tabs(self, pages: list[_Page], active: _Page, deadlines: _Deadlines) -> str:
        """Render one line per open tab, marking the active one.

        A title that cannot be read does not fail the listing: the reason to list
        tabs is often that one of them is misbehaving.
        """
        if not pages:
            return 'No tabs open.'
        lines: list[str] = []
        for position, page in enumerate(pages):
            try:
                title = await self._await_with_timeout(page.title(), deadlines.action)
            except PlaywrightError:
                title = '<title unavailable>'
            marker = ' (active)' if page is active else ''
            lines.append(f'{position}{marker}: {title} -- {_without_credentials(page.url)}')
        return '\n'.join(lines)

    async def handle_next_dialog(self, accept: bool, prompt_text: str | None = None) -> str:
        """Decide how the next `alert`, `confirm` or `prompt` dialog is answered.

        A dialog blocks the page until it is answered, so the decision has to be
        made before the action that opens one: call this, then click the button
        that triggers it. Left alone, a dialog is dismissed -- the cancelling
        branch of a `confirm`, and an empty `prompt` -- so this is how the
        accepting branch of a delete confirmation or a prompt is reached.

        The decision covers one dialog. Anything the page opens after that is
        dismissed again unless this is called once more.

        Args:
            accept: Take the OK branch when `True`, the Cancel branch when `False`.
            prompt_text: Text to answer a `prompt` with. Other dialogs ignore it.

        Returns:
            A confirmation of the armed decision.
        """

        def _arm() -> str:
            self._session.dialog_decision = _DialogDecision(accept=accept, prompt_text=prompt_text)
            answer = 'be accepted' if accept else 'be dismissed'
            return f'The next dialog will {answer}.'

        return await self._in_session('handle_next_dialog', _arm)

    async def console_messages(self, errors_only: bool = False) -> str:
        """Return the console output and uncaught script errors the page produced.

        Reaches what the page never renders: a failed script, a rejected request
        logged by the site's own code, a framework warning explaining why content
        is missing.

        Args:
            errors_only: Return only errors, dropping logs and warnings.

        Returns:
            One line per message, oldest first, or a note when there are none.
        """
        kinds = ('console', 'page_error', 'dialog')
        events = [event for event in self._session.events if event.kind in kinds]
        if errors_only:
            events = [event for event in events if event.level == 'error']
        return await self._in_session('console_messages', lambda: self._render_events(events, 'console messages'))

    async def network_requests(self, url_contains: str | None = None, errors_only: bool = False) -> str:
        """Return the network requests the page made, with their URL, method and status.

        Pages that render from an API often name the data more directly here than
        the DOM does: find the request, then fetch that URL with `navigate` or
        `execute_js`. Response bodies are not recorded, only what identifies the
        request. Requests the egress policy refused are listed with the reason, so
        a page that fails to load is diagnosable.

        Args:
            url_contains: Keep only requests whose URL contains this substring.
            errors_only: Keep only failures -- refused requests, requests the
                network never completed, and responses with a 4xx or 5xx status.

        Returns:
            One line per request, oldest first, or a note when there are none.
        """
        kinds = ('response', 'request_failed', 'request_blocked')
        events = [event for event in self._session.events if event.kind in kinds]
        if url_contains is not None:
            events = [event for event in events if event.url is not None and url_contains in event.url]
        if errors_only:
            events = [event for event in events if event.level != 'info']
        return await self._in_session('network_requests', lambda: self._render_events(events, 'network requests'))

    async def _in_session(self, action: str, body: Callable[[], str]) -> str:
        """Run an operation that reads or arms session state, without acquiring a page.

        These take no deadline and must not start a browser -- they act on what the
        session already holds -- but they are still operations, and a trace showing
        the page tools and not these would misrepresent what the agent did. The
        lock is still taken: what they read or arm is what a concurrent operation
        is producing or about to consume.
        """
        async with self._operation_lock:
            with self._session.tracer.start_as_current_span(
                f'browser {action}', attributes={'browser.action': action, 'browser.outcome': 'ok'}
            ):
                return body()

    def _render_events(self, events: list[BrowserEvent], label: str) -> str:
        """Render browser events as bounded, model-facing lines, keeping the recent ones.

        The budget is spent from the end. What makes this log worth reading is what
        just happened, and truncating from the head would keep the static assets a
        page loaded first and drop the failure that prompted the call.
        """
        if not events:
            return self._truncate_output(f'No {label} recorded.')
        lines = [event.describe() for event in events]
        budget = self._max_content_tokens * _CHARS_PER_TOKEN
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            if kept and used + len(line) + 1 > budget:
                break
            used += len(line) + 1
            kept.append(line)
        if len(kept) == len(lines):
            return self._truncate_output('\n'.join(lines))
        # The marker counts against the budget like any other line, and dropping one
        # more entry to make room for it changes what it says.
        marker = f'[... {len(lines) - len(kept)} older {label} dropped]'
        while len(kept) > 1 and used + len(marker) + 1 > budget:
            used -= len(kept.pop()) + 1
            marker = f'[... {len(lines) - len(kept)} older {label} dropped]'
        return self._truncate_output('\n'.join([marker, *reversed(kept)]))
