"""The egress policy the browser capabilities share: where a browser may go, and what a page there may talk to.

`PlaywrightBrowser` and `CDPBrowser` enforce the same `EgressPolicy` through different transports: a Playwright
route guard, and the Chrome DevTools Protocol's `Fetch` interception. Keeping the policy, the address checks and
the resolver here gives both one vocabulary and one verdict for the same request. A browser either one launches also
connects through the proxy in `_browser_proxy`, which applies the address checks to what it dials.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, replace
from time import monotonic
from typing import Literal, get_args
from urllib.parse import urlparse

import idna

from pydantic_ai._ssrf import is_cloud_metadata_ip, is_private_ip
from pydantic_ai.exceptions import UserError

PRIVATE_ADDRESS_REASON = 'blocked private or link-local address'
UNRESOLVED_REASON = 'host that did not resolve, so the private-address block could not clear it'
"""Why the private-address block refused a URL, shared by the pre-check and the route guard."""

BLANK_PAGE = 'about:blank'
"""The blank page a context starts on, and where a disallowed navigation is sent."""


def to_idna(host: str) -> str:
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


def url_host(url: str) -> str | None:
    """Extract the host that policy checks run against, or `None` when there is none.

    Only the part before the query and fragment is parsed. A backslash in that part
    makes the URL hostless: WHATWG parsing (what Chromium applies) turns it into `/`,
    so `urlparse` would report a host the browser never connects to. A malformed URL
    that `urlparse` rejects is also treated as hostless, so it fails closed instead of
    crashing the caller.
    """
    head = url.partition('?')[0].partition('#')[0]
    if '\\' in head:
        return None
    try:
        return urlparse(head).hostname
    except ValueError:
        return None


def is_blocked_address(host: str) -> bool:
    """Return whether `host` names an address that is not globally routable.

    Covers private (RFC 1918 and IPv6 site-local), loopback, link-local, carrier-grade
    NAT, reserved, and multicast ranges, and every address that URL
    downloads refuse. This
    classifies a host string, so it sees IP literals and the loopback hostnames
    `localhost` / `*.localhost`; a name pointing at a private address is caught by
    the capability's guard resolving it first and passing the answers to
    `refuse`.
    A browser the capabilities launch connects through `_browser_proxy`, which
    applies this check to the address it dials, so a name that changes after the
    guard's lookup cannot reach a private one. A browser reached over `cdp_url`
    resolves the name again itself before it connects, so there it can.
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
    if is_cloud_metadata_ip(host) or is_private_ip(host):
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def is_ip_literal(host: str) -> bool:
    """Whether `host` is already an address, so no name resolution is needed."""
    try:
        ipaddress.ip_address(host.rstrip('.'))
    except ValueError:
        return False
    return True


RESOLUTION_TTL_SECONDS = 30.0
RESOLUTION_CACHE_MAX = 256
RESOLUTION_TIMEOUT_SECONDS = 2.0
resolution_cache: dict[str, tuple[float, tuple[str, ...]]] = {}


async def resolve_host(host: str) -> tuple[str, ...] | None:
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

    The cache is kept short and small. The proxy in `_browser_proxy` resolves
    through it too, so a launched browser connects to the addresses the guard
    checked. A browser reached over `cdp_url` resolves the name a second time
    itself, and a record that changes between the two lookups gets past the
    block there.

    The lookup is bounded because of where it runs: ahead of the operation whose
    deadline it would otherwise escape, and inside the route guard, which holds the
    request until it returns. A stalled resolver therefore costs a bounded wait
    rather than the run.
    """
    now = monotonic()
    cached = resolution_cache.get(host)
    if cached is not None and cached[0] > now:
        return cached[1]
    try:
        addresses = await asyncio.wait_for(getaddrinfo(host), RESOLUTION_TIMEOUT_SECONDS)  # noqa: TID251
    except (TimeoutError, OSError, UnicodeError):
        return None
    if len(resolution_cache) >= RESOLUTION_CACHE_MAX:
        resolution_cache.clear()
    resolution_cache[host] = (now + RESOLUTION_TTL_SECONDS, addresses)
    return addresses


async def getaddrinfo(host: str) -> tuple[str, ...]:  # pragma: no cover
    """Ask the system resolver what `host` points at, off the event loop thread.

    The one place the capability performs a real lookup, and therefore the seam the
    test suite replaces so that no test depends on a resolver. What it does is
    checked against live DNS in `scripts/playwright_smoke.py::_check_private_name_block` and
    `scripts/cdp_browser_smoke.py`.
    """
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return tuple(sorted({str(info[4][0]) for info in infos}))


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

DATA_RESOURCE_TYPES = frozenset({'fetch', 'xhr', 'eventsource', 'websocket', 'ping', 'preflight'})
"""Resource types, lowercased, that carry data rather than render the page.

Playwright's `resource_type` values, which are the Chrome DevTools Protocol's `Network.ResourceType` lowercased.

`ping` is `navigator.sendBeacon`, which is a send channel like the rest of them.
"""


def request_kind(resource_type: str, *, is_navigation: bool, top_level: bool) -> RequestKind:
    """Classify a request for the policy from its lowercased resource type, so both browsers judge it alike.

    A document is a navigation only in the main frame; the same type inside an
    embedded frame is what loads an identity provider or a payment step, which a
    policy usually treats differently.
    """
    if resource_type in DATA_RESOURCE_TYPES:
        return 'data'
    if is_navigation:
        return 'navigation' if top_level else 'subframe'
    return 'subresource'


@dataclass(frozen=True)
class EgressRequest:
    """One request the browser is about to make, as the policy sees it.

    Everything the guard knows is passed through, including the browser's own
    `resource_type`, so a policy that needs a distinction this module never
    anticipated can make it without waiting for a new field here.
    """

    url: str
    kind: RequestKind
    resource_type: str = 'document'
    """The browser's own classification, lowercased, e.g. `document`, `xhr`, `image`, `font`.

    Playwright's `resource_type`, or the Chrome DevTools Protocol's `Network.ResourceType` lowercased.
    """
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
    """Which kinds have their host resolved so `block_private_addresses` can see where a name points.

    This sets what the request guards look up. The proxy a launched browser connects through checks every name it
    connects to, whatever this holds.
    """

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
        if request.url == BLANK_PAGE:
            return None
        host = url_host(request.url)
        if host is not None and self.block_private_addresses and is_blocked_address(host):
            return PRIVATE_ADDRESS_REASON
        if self.block_private_addresses and request.resolution_failed:
            return UNRESOLVED_REASON
        if self.block_private_addresses and any(is_blocked_address(a) for a in request.resolved_addresses):
            return PRIVATE_ADDRESS_REASON
        if host is None:
            return None if request.kind in ('data', 'subresource') else 'URL with no host'
        return self._domain_reason(host, request.kind)

    def _domain_reason(self, host: str, kind: RequestKind) -> str | None:
        """Why the domain lists refuse `host` for this kind of request, or `None` if they do not."""
        if self._matches(host, self.blocked_domains):
            return 'domain in blocked_domains'
        if kind in self.allowlist_reach and not self._permitted(host):
            return 'domain not in allowed_domains'
        return None

    def _permitted(self, host: str) -> bool:
        """Whether the allowlist admits `host`, with `None` meaning every public host."""
        return self.allowed_domains is None or self._matches(host, self.allowed_domains)

    def _matches(self, host: str, domains: list[str] | None) -> bool:
        """Whether `host` equals an entry, or sits under one when subdomains count."""
        if not domains:
            return False
        host = to_idna(host)
        for entry in domains:
            domain = to_idna(entry.strip().lower())
            if domain and (host == domain or (self.include_subdomains and host.endswith('.' + domain))):
                return True
        return False

    def needs_resolution(self, request: EgressRequest) -> bool:
        """Whether the caller must resolve this host before `refuse` can classify it.

        Asked before `refuse` so that `refuse` stays synchronous: an override needs
        no async machinery, and the lookup a name-based private address requires
        happens outside it. A host that is already an address needs none, and
        neither does one the domain lists refuse, since `refuse` refuses it
        whatever the host resolves to. That holds only for this
        class's own `refuse`: an override may permit what the lists refuse, so its
        hosts are still resolved. An override that refuses some hosts by name should
        override this too, returning `False` for them, so they are not looked up.
        """
        if not self.block_private_addresses or request.kind not in self.resolved_kinds:
            return False
        host = url_host(request.url)
        if host is None or is_ip_literal(host):
            return False
        return type(self).refuse is not EgressPolicy.refuse or self._domain_reason(host, request.kind) is None

    def enforced(self) -> bool:
        """Whether anything is restricted, i.e. whether a request guard is worth installing."""
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


async def refusal_reason(policy: EgressPolicy, request: EgressRequest) -> str | None:
    """Why `policy` refuses `request`, or `None` to allow it, resolving the host first when the policy needs it.

    Every enforcement point of every browser capability goes through here rather than calling `refuse`
    directly, so a name that points at a private address is caught wherever it is asked about.
    """
    host = url_host(request.url)
    if host is not None and policy.needs_resolution(request):
        addresses = await resolve_host(host)
        request = replace(request, resolved_addresses=addresses or (), resolution_failed=addresses is None)
    return policy.refuse(request)
