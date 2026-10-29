"""Whether a page lets the web UI show it in a frame, read from the headers it is served with.

A browser refuses to frame a page whose `Content-Security-Policy` `frame-ancestors` leaves out the
framing page's origin or, without that directive, that sends `X-Frame-Options`; it draws an error box
instead. So the server reads a side panel's headers once and the page frames only what would show.
"""

import re
from collections.abc import Awaitable, Callable
from urllib.parse import SplitResult, urlsplit

import httpx
from anyio import fail_after

FetchHeaders = Callable[[str], Awaitable[httpx.Headers]]
"""Reads the response headers of a page that answers itself; raises `httpx.HTTPError` when it cannot."""
FETCH_TIMEOUT = 5.0
"""Seconds to read a page's headers before showing it in a window of its own instead."""
FRAME_SOURCES = "'self' https: http://127.0.0.1:* http://localhost:*"
"""The pages the web UI's own `Content-Security-Policy` lets it frame: on the web over https, or on this machine."""

_DEFAULT_PORTS = {'http': 80, 'https': 443}
_HOST_SOURCE = re.compile(
    r'^(?:(?P<scheme>[a-z][a-z0-9+.-]*)://)?(?P<host>\*|(?:\*\.)?[^:/*]+)(?::(?P<port>\*|\d+))?(?:/.*)?$'
)


async def fetch_headers(url: str, *, transport: httpx.AsyncBaseTransport | None = None) -> httpx.Headers:
    """The headers `url` answers a `GET` with, without reading its body.

    A redirect or an error raises `httpx.HTTPStatusError`: redirects are not followed, so a page cannot send
    this request on to another address, such as one on this machine, and its own headers cannot be known.
    """
    async with httpx.AsyncClient(transport=transport, timeout=5) as client:
        async with client.stream('GET', url) as response:
            if not response.is_success:
                raise httpx.HTTPStatusError(
                    f'{url} answered {response.status_code}', request=response.request, response=response
                )
            return response.headers


class FrameCheck:
    """Whether each page can be framed, fetching its headers once, whatever its query string."""

    def __init__(self, fetch: FetchHeaders = fetch_headers) -> None:
        self._fetch = fetch
        self._headers: dict[str, httpx.Headers] = {}
        """By the page's address without its query."""

    async def __call__(self, url: str, *, ancestor: str) -> bool:
        """Whether a page on the origin `ancestor`, such as `http://127.0.0.1:8000`, can show `url` in a frame.

        A page outside `FRAME_SOURCES`, or that cannot be read now, is not framed, which would show an error box;
        one that could not be read is read again next time.
        """
        parts = urlsplit(url)
        if not (parts.scheme == 'https' or (parts.scheme == 'http' and parts.hostname in ('127.0.0.1', 'localhost'))):
            return False
        key = parts._replace(query='', fragment='').geturl()
        if key not in self._headers:
            try:
                with fail_after(FETCH_TIMEOUT):
                    self._headers[key] = await self._fetch(url)
            except (httpx.HTTPError, httpx.InvalidURL, TimeoutError):
                return False
        return frameable(self._headers[key], page=url, ancestor=ancestor)


def frameable(headers: httpx.Headers, *, page: str, ancestor: str) -> bool:
    """Whether a browser shows `page`, served with `headers`, in a frame on a page from the origin `ancestor`.

    Every policy with `frame-ancestors` must allow it; only without one does `X-Frame-Options` apply.
    """
    target, framer = urlsplit(page), urlsplit(ancestor)
    policies = [
        sources
        for header in headers.get_list('content-security-policy')
        for policy in header.split(',')
        if (sources := _frame_ancestors(policy)) is not None
    ]
    if policies:
        return all(any(_matches(source, target=target, framer=framer) for source in sources) for sources in policies)
    options = {value.strip().lower() for value in headers.get_list('x-frame-options', split_commas=True)}
    if len(options) > 1 and options & {'deny', 'sameorigin', 'allowall'}:
        return False  # Conflicting values block, as the HTML standard says.
    if 'deny' in options:
        return False
    if 'sameorigin' in options:
        return _origin(target) == _origin(framer)
    return True  # None, `ALLOWALL`, or one browsers ignore, such as `ALLOW-FROM`.


def _frame_ancestors(policy: str) -> list[str] | None:
    """The policy's `frame-ancestors` sources, lowercased; `None` without the directive. `'none'` is an empty list."""
    for directive in policy.split(';'):
        words = [word.lower() for word in directive.split()]
        if words and words[0] == 'frame-ancestors':
            return [source for source in words[1:] if source != "'none'"]
    return None


def _matches(source: str, *, target: SplitResult, framer: SplitResult) -> bool:
    """CSP's source matching for an ancestor, ignoring paths, which no `frame-ancestors` CLAI meets uses."""
    if source == '*':
        return framer.scheme in _DEFAULT_PORTS
    if source == "'self'":
        return _origin(target) == _origin(framer)
    if source.endswith(':') and '/' not in source:
        return _scheme_matches(source[:-1], framer.scheme)
    found = _HOST_SOURCE.match(source)
    if found is None or framer.hostname is None:
        return False
    scheme = found['scheme']
    if not _scheme_matches(scheme or target.scheme, framer.scheme):
        return False
    host = found['host']
    if host.startswith('*.'):
        if not framer.hostname.endswith(host[1:]):
            return False
    elif host != framer.hostname:
        return False
    port = found['port']
    if port == '*':
        return True
    return _port(framer) == (int(port) if port else _DEFAULT_PORTS.get(framer.scheme))


def _scheme_matches(expected: str, actual: str) -> bool:
    """A source's scheme matches itself, and `http` also matches `https`."""
    return actual == expected or (expected == 'http' and actual == 'https')


def _port(url: SplitResult) -> int | None:
    return url.port or _DEFAULT_PORTS.get(url.scheme)


def _origin(url: SplitResult) -> tuple[str, str | None, int | None]:
    return url.scheme, url.hostname, _port(url)
