"""`Host` header validation for the web chat UI, to defeat DNS rebinding.

Binding the UI to loopback keeps other machines out, but not other *websites*: a page the developer
visits can make their browser send requests to `http://127.0.0.1:7932`. The chat endpoint's
`application/json` requirement stops that, because a browser can't send a non-safelisted content
type cross-origin without a preflight the server refuses.

DNS rebinding gets around all of it. The attacker points a name they control at `127.0.0.1`, so the
browser believes `http://evil.example:7932` and the web UI are the *same* origin: no preflight, no
CORS, and a CSRF token wouldn't help either, since a same-origin page can simply read it. The one
thing the attacker cannot change is the `Host` header the browser sends, which still carries their
name. Checking it is therefore the control available at this layer — putting an authenticating proxy
in front of the app is the other answer, and the one to reach for outside local development.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from pydantic_ai.exceptions import UserError

ANY_HOST = '*'
"""`allowed_hosts` entry that accepts every `Host` header, turning the protection off."""

LOCALHOST = 'localhost'

# Characters that make `urlsplit` read part of the value as userinfo, a path, a query or a fragment.
# None of them may appear in a `Host` header, and treating them as a parse failure is what keeps
# `evil.example@127.0.0.1` from being read as loopback.
_ILLEGAL_HOST_CHARS = frozenset('@/?#')


def _hostname(host_header: str) -> str | None:
    """Return the lowercased hostname in a `Host` header value, or `None` if it isn't parsable.

    The port is dropped: only one server is listening on the port the request arrived at, so the
    port identifies nothing. Note that it can't be dropped by slicing at the first `:` — which is
    what Starlette's own `TrustedHostMiddleware` does — because that mangles the bracketed IPv6
    form `[::1]:7932` into `[`.

    A single trailing dot is dropped as well. `http://localhost./` names the same host as
    `http://localhost/`, and a browser keeps that dot in the `Host` header, so without this
    `localhost.` would be turned away. Exactly one dot, so `localhost..`, which denotes no host at
    all, stays rejected.
    """
    if _ILLEGAL_HOST_CHARS & set(host_header):
        return None
    try:
        hostname = urlsplit(f'//{host_header}').hostname
    except ValueError:
        return None
    if not hostname:
        # `urlsplit('//')` yields `None`, which is how an absent or empty `Host` lands here.
        return None
    return hostname.removesuffix('.') or None


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def normalized_pattern(pattern: str) -> str:
    """Normalize an `allowed_hosts` entry the way `_hostname` normalizes an incoming header.

    Raises `UserError` on a wildcard naming no domain. `'*.'` is the one that matters: it reads like
    a wildcard for something, but dropping its root dot would leave `'*'` — the sentinel that accepts
    every host — so a typo would silently turn the check off rather than fail. A wildcard that isn't
    `*.`-prefixed (`'*example.com'`) is rejected in the same pass; it can only ever match a literal
    host of that name, which is to say never.
    """
    lowered = pattern.lower()
    if lowered != ANY_HOST and lowered.startswith('*') and not (lowered.startswith('*.') and lowered[2:].strip('.')):
        raise UserError(
            f'Invalid `allowed_hosts` pattern {pattern!r}. '
            f'Use a hostname, `*.example.com` to match its subdomains, or `{ANY_HOST}` to allow any host.'
        )
    return lowered.removesuffix('.')


def _matches(hostname: str, pattern: str) -> bool:
    if pattern.startswith('*.'):
        # Subdomains only, not the apex — `*.example.com` means exactly what it means in Starlette's
        # `TrustedHostMiddleware`, which this parameter is named after. A deployment that serves the
        # apex too lists it separately; over-matching an allowlist is the worse way to be wrong.
        # Note this compares against `.example.com`, so `a.example.com.evil` doesn't slip through on
        # a bare substring.
        return hostname.endswith(pattern[1:])
    return hostname == pattern


def is_allowed_host(host_header: str, allowed_hosts: Sequence[str]) -> bool:
    """Whether a request's `Host` header names a server this app is willing to answer as.

    `allowed_hosts` entries must already be normalized the way `HostValidationMiddleware` normalizes
    them: lowercased, with any trailing dot removed. Hostnames are compared in their ASCII form, so
    an internationalized name has to be given as punycode (`xn--bcher-kva.example`), which is what a
    browser puts in the header anyway.
    """
    if ANY_HOST in allowed_hosts:
        return True

    hostname = _hostname(host_header)
    if hostname is None:
        return False

    # An IP literal can't be the product of DNS rebinding: rebinding works by pointing a *name* the
    # attacker controls at the server's address, and a name is exactly what an IP literal is not. A
    # page that navigates straight to `http://127.0.0.1:7932` is an ordinary cross-origin request,
    # which the chat endpoint's `application/json` requirement already turns away. Accepting every
    # IP literal rather than just the loopback ones is deliberate: reaching a dev server over the
    # LAN or through a forwarded port is a normal thing to do, and none of it is rebindable.
    if _is_ip_address(hostname):
        return True

    # RFC 6761 reserves `localhost` and everything under it for the loopback interface, and browsers
    # resolve those names locally instead of over DNS, so they can't be rebound either.
    if hostname == LOCALHOST or hostname.endswith(f'.{LOCALHOST}'):
        return True

    return any(_matches(hostname, pattern) for pattern in allowed_hosts)


class HostValidationMiddleware:
    """Turn away requests whose `Host` header names a server other than a local one.

    Deliberately reads only the `Host` header, never `X-Forwarded-Host` or `Forwarded`: those are
    just request headers, so a client talking to the app directly can set them to anything. A
    deployment behind a proxy names the public hostname in `allowed_hosts` instead.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str]) -> None:
        self.app = app
        self.allowed_hosts = [normalized_pattern(pattern) for pattern in allowed_hosts]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] not in ('http', 'websocket'):
            await self.app(scope, receive, send)
            return

        host_header = Headers(scope=scope).get('host', '')
        if is_allowed_host(host_header, self.allowed_hosts):
            await self.app(scope, receive, send)
            return

        if scope['type'] == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
            return

        response = PlainTextResponse(
            f'Host {host_header!r} is not allowed.\n\n'
            'The web chat UI only answers requests whose `Host` header is an IP address, '
            '`localhost`, or a name under `.localhost`, so that a website cannot reach it on your '
            'machine by pointing a hostname it controls at you (DNS rebinding).\n\n'
            'To serve the UI under a hostname, pass it in `allowed_hosts`, or run `clai web` with '
            '`--allowed-host`.\n',
            status_code=421,
            # The message quotes the client's own `Host` header back at it. `text/plain` is enough
            # to keep a browser from rendering that as markup, as long as it isn't sniffed.
            headers={'x-content-type-options': 'nosniff'},
        )
        await response(scope, receive, send)
