"""SSRF (Server-Side Request Forgery) protection for URL downloads.

This module provides security measures to prevent SSRF attacks when downloading
content from URLs. It validates protocols, resolves hostnames to IP addresses,
and blocks requests to private/internal networks and cloud metadata endpoints.
"""

from __future__ import annotations

import ipaddress
import socket
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx2

from ._http import create_async_httpx2_client, legacy_httpx as _legacy_httpx
from ._utils import run_in_executor

__all__ = ['safe_download']

_DOWNLOAD_EXCEEDS_TEMPLATE = 'Download exceeds the maximum size of {max_bytes} bytes.'
# Bounded downloads only negotiate encodings we can size-limit while streaming.
# Brotli/Zstandard can expand a few compressed bytes into multi-MiB output in one
# decoder step, and `deflate` has to be buffered whole before its zlib-wrapped vs raw
# framing can be told apart, so all three are excluded from Accept-Encoding and
# rejected if a server still returns them.
_BOUNDED_ACCEPT_ENCODING = 'identity, gzip'

# Private IP ranges that should be blocked by default (i.e. unless allow_local=True).
# IPv6 transition forms (6to4, NAT64, IPv4-mapped/-compatible, ISATAP) are not listed here;
# they are decoded to their embedded IPv4 by `_embedded_ipv4s()` and checked against this table.
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    # IPv4 private ranges
    ipaddress.IPv4Network('0.0.0.0/8'),  # "This" network
    ipaddress.IPv4Network('10.0.0.0/8'),  # Private
    ipaddress.IPv4Network('100.64.0.0/10'),  # CGNAT (RFC 6598), includes Alibaba Cloud metadata
    ipaddress.IPv4Network('127.0.0.0/8'),  # Loopback
    ipaddress.IPv4Network('169.254.0.0/16'),  # Link-local (includes cloud metadata)
    ipaddress.IPv4Network('172.16.0.0/12'),  # Private
    ipaddress.IPv4Network('192.168.0.0/16'),  # Private
    # IPv4 IANA-reserved / special-purpose ranges (not globally routable)
    ipaddress.IPv4Network('192.0.0.0/24'),  # IETF Protocol Assignments (RFC 6890)
    ipaddress.IPv4Network('192.0.2.0/24'),  # TEST-NET-1 (RFC 5737)
    ipaddress.IPv4Network('198.18.0.0/15'),  # Network benchmarking (RFC 2544)
    ipaddress.IPv4Network('198.51.100.0/24'),  # TEST-NET-2 (RFC 5737)
    ipaddress.IPv4Network('203.0.113.0/24'),  # TEST-NET-3 (RFC 5737)
    ipaddress.IPv4Network('224.0.0.0/4'),  # Multicast (RFC 5771)
    ipaddress.IPv4Network('240.0.0.0/4'),  # Reserved + limited broadcast 255.255.255.255 (RFC 1112)
    # IPv6 private ranges
    ipaddress.IPv6Network('::/128'),  # Unspecified address
    ipaddress.IPv6Network('::1/128'),  # Loopback
    ipaddress.IPv6Network('fe80::/10'),  # Link-local
    ipaddress.IPv6Network('fc00::/7'),  # Unique local address
    # IPv6 IANA-reserved / special-purpose ranges
    ipaddress.IPv6Network('100::/64'),  # Discard prefix (RFC 6666)
    ipaddress.IPv6Network('2001::/32'),  # Teredo tunneling (RFC 4380)
    ipaddress.IPv6Network('2001:db8::/32'),  # Documentation (RFC 3849)
    ipaddress.IPv6Network('ff00::/8'),  # Multicast (RFC 4291)
)

# RFC 6052 §2.2: byte offsets (within the 16-byte address) of the embedded IPv4 for each
# standardized NAT64 prefix length, plus the 6to4 (RFC 3056) position. Byte 8 is the
# reserved "u" octet that the IPv4 skips in the shorter NAT64 prefixes.
_NAT64_OFFSETS_BY_PREFIX_LEN: dict[int, tuple[int, int, int, int]] = {
    32: (4, 5, 6, 7),
    40: (5, 6, 7, 9),
    48: (6, 7, 9, 10),
    56: (7, 9, 10, 11),
    64: (9, 10, 11, 12),
    96: (12, 13, 14, 15),
}
_LOW32_OFFSETS = (12, 13, 14, 15)  # IPv4-mapped/-compatible, NAT64 /96, ISATAP, generic
_SIXTOFOUR_OFFSETS = (2, 3, 4, 5)  # 6to4 2002::/16 (bits 16-47)
_ALL_EMBEDDED_OFFSETS: tuple[tuple[int, int, int, int], ...] = (
    *_NAT64_OFFSETS_BY_PREFIX_LEN.values(),
    _SIXTOFOUR_OFFSETS,
)

# NAT64 prefixes paired with the embedding lengths an operator may use within them.
# RFC 6052 well-known prefix is /96-only; the RFC 8215 local-use prefix is a /48 that
# operators may further subnet to /56, /64, or /96.
_NAT64_PREFIXES: tuple[tuple[ipaddress.IPv6Network, tuple[tuple[int, int, int, int], ...]], ...] = (
    (ipaddress.IPv6Network('64:ff9b::/96'), (_NAT64_OFFSETS_BY_PREFIX_LEN[96],)),
    (
        ipaddress.IPv6Network('64:ff9b:1::/48'),
        tuple(_NAT64_OFFSETS_BY_PREFIX_LEN[pl] for pl in (48, 56, 64, 96)),
    ),
)

# ISATAP (RFC 5214) interface identifiers: `::0:5efe:a.b.c.d` and `::200:5efe:a.b.c.d`,
# i.e. bytes 8-11 of the address carry the marker and bytes 12-15 carry the IPv4.
_ISATAP_INTERFACE_IDS = (b'\x00\x00\x5e\xfe', b'\x02\x00\x5e\xfe')

# Teredo (RFC 4380): 2001::/32 carries the client IPv4 in the low 32 bits, XOR'd with
# all-ones (obfuscated). The raw low-32 bytes are meaningless, so it needs its own decode.
_TEREDO_PREFIX = ipaddress.IPv6Network('2001::/32')

# Cloud metadata / credential endpoints - always blocked, even with allow_local=True.
# When allow_local=True we skip the private-IP check, so these must be caught explicitly.
# Most are also covered by the private ranges above, but 168.63.129.16 (Azure) is a public
# IP, so the metadata guard is the only thing that blocks it.
_CLOUD_METADATA_IPV4: frozenset[ipaddress.IPv4Address] = frozenset(
    ipaddress.IPv4Address(ip)
    for ip in (
        '169.254.169.254',  # AWS IMDS, GCP, Azure, OCI, DigitalOcean, Hetzner, IBM, OpenStack, ...
        '169.254.170.2',  # AWS ECS task IAM role credentials
        '169.254.170.23',  # AWS EKS Pod Identity Agent
        '168.63.129.16',  # Azure WireServer / platform channel (public IP)
        '100.100.100.200',  # Alibaba Cloud
        '192.0.0.192',  # Oracle Cloud (Classic)
        '169.254.42.42',  # Scaleway
    )
)
_CLOUD_METADATA_IPV6: frozenset[ipaddress.IPv6Address] = frozenset(
    ipaddress.IPv6Address(ip)
    for ip in (
        'fd00:ec2::254',  # AWS IMDS IPv6
        'fd00:ec2::23',  # AWS EKS Pod Identity Agent IPv6
        'fd20:ce::254',  # GCP IPv6 (IPv6-only instances)
        'fd00:42::42',  # Scaleway IPv6
    )
)

_MAX_REDIRECTS = 10
_DEFAULT_TIMEOUT = 30  # seconds
_SENSITIVE_HEADERS = frozenset(('authorization', 'cookie', 'proxy-authorization'))


# These initialize classes inheriting from both HTTPX families, so they call `Exception.__init__`
# directly: `super().__init__` would walk a diamond MRO spanning two libraries and run only the
# first family's initializer. `_request` is the private backing field of the `request` property in
# both libraries, so an upstream rename of it breaks these silently.
def _compatible_request_error_init(self: Exception, message: str, *, request: httpx2.Request | None = None) -> None:
    Exception.__init__(self, message)
    self.__dict__['_request'] = request


def _compatible_http_status_error_init(
    self: Exception, message: str, *, request: httpx2.Request, response: httpx2.Response
) -> None:
    Exception.__init__(self, message)
    self.__dict__['_request'] = request
    self.__dict__['response'] = response


# TODO(v3): remove the compatibility classes below and raise the plain httpx2 errors; they exist
# only so exception handlers written against legacy `httpx` keep matching during the v2 window.
if _legacy_httpx is not None:
    _CompatibleRequestError = type(
        '_CompatibleRequestError',
        (httpx2.RequestError, _legacy_httpx.RequestError),
        {'__init__': _compatible_request_error_init},
    )
    _CompatibleHTTPStatusError = type(
        '_CompatibleHTTPStatusError',
        (httpx2.HTTPStatusError, _legacy_httpx.HTTPStatusError),
        {'__init__': _compatible_http_status_error_init},
    )
    _CompatibleDecodingError = type(
        '_CompatibleDecodingError',
        (httpx2.DecodingError, _legacy_httpx.DecodingError),
        {'__init__': _compatible_request_error_init},
    )
else:
    _CompatibleRequestError = httpx2.RequestError
    _CompatibleHTTPStatusError = httpx2.HTTPStatusError
    _CompatibleDecodingError = httpx2.DecodingError


def _compatible_request_error(error: httpx2.RequestError) -> Exception:
    return _CompatibleRequestError(str(error), request=error.request)


async def _send_request(client: httpx2.AsyncClient, request: httpx2.Request) -> httpx2.Response:
    try:
        return await client.send(request, follow_redirects=False, stream=True)
    except httpx2.RequestError as e:
        raise _compatible_request_error(e) from e


async def _read_body(response: httpx2.Response) -> None:
    try:
        await response.aread()
    except httpx2.RequestError as e:
        raise _compatible_request_error(e) from e


@dataclass
class ResolvedUrl:
    """Result of URL validation and DNS resolution."""

    resolved_ip: str
    """The resolved IP address to connect to."""

    hostname: str
    """The original hostname (used for Host header)."""

    port: int
    """The port number."""

    is_https: bool
    """Whether to use HTTPS."""

    path: str
    """The path including query string and fragment."""


def _embedded_ipv4s(ip: ipaddress.IPv6Address, *, exhaustive: bool) -> set[ipaddress.IPv4Address]:
    """Return the IPv4 addresses `ip` may route to via an IPv6 transition mechanism.

    An IPv6 literal can carry an IPv4 destination (IPv4-mapped, IPv4-compatible, 6to4,
    NAT64, ISATAP, Teredo, ...) that dual-stack or translating networks deliver to the
    embedded IPv4 endpoint. The blocklist guards must therefore consider that embedded
    IPv4, not just the IPv6 wrapper, or an attacker can smuggle a blocked IPv4 past them
    in IPv6 clothing.

    With `exhaustive=False`, only well-recognized transition contexts are decoded, so a
    real public IPv6 address whose bytes happen to coincide with a private range is never
    misclassified. With `exhaustive=True`, every standardized embedding position is
    decoded unconditionally; this is only used for the cloud-metadata guard, whose target
    set is small enough that a coincidental match is effectively impossible, and it
    additionally covers operator-chosen NAT64 prefixes that we cannot enumerate.
    """
    packed = ip.packed

    def at(offsets: tuple[int, int, int, int]) -> ipaddress.IPv4Address:
        return ipaddress.IPv4Address(bytes(packed[i] for i in offsets))

    candidates: set[ipaddress.IPv4Address] = set()

    if exhaustive:
        candidates.update(at(offsets) for offsets in _ALL_EMBEDDED_OFFSETS)
        if ip in _TEREDO_PREFIX:  # client IPv4 = low 32 bits XOR all-ones (RFC 4380)
            candidates.add(ipaddress.IPv4Address(int.from_bytes(packed[12:16], 'big') ^ 0xFFFFFFFF))
        return candidates

    if ip.ipv4_mapped is not None:  # ::ffff:a.b.c.d (RFC 4291 §2.5.5.2)
        candidates.add(ip.ipv4_mapped)
    if ip.sixtofour is not None:  # 2002::/16 (RFC 3056)
        candidates.add(ip.sixtofour)
    for prefix, offsets_list in _NAT64_PREFIXES:  # 64:ff9b::/96 (RFC 6052), 64:ff9b:1::/48 (RFC 8215)
        if ip in prefix:
            candidates.update(at(offsets) for offsets in offsets_list)
    if int(ip) >> 32 == 0 and not ip.is_loopback and not ip.is_unspecified:  # ::a.b.c.d (deprecated)
        candidates.add(at(_LOW32_OFFSETS))
    if packed[8:12] in _ISATAP_INTERFACE_IDS:  # ...:[0|200]:5efe:a.b.c.d (RFC 5214)
        candidates.add(at(_LOW32_OFFSETS))
    return candidates


def is_cloud_metadata_ip(ip_str: str) -> bool:
    """Check if an IP address is a cloud metadata/credential endpoint.

    These are always blocked for security reasons, even with allow_local=True. IPv6
    transition forms are decoded so a metadata IP cannot be smuggled in as IPv6.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in _CLOUD_METADATA_IPV4
    if ip in _CLOUD_METADATA_IPV6:
        return True
    return any(candidate in _CLOUD_METADATA_IPV4 for candidate in _embedded_ipv4s(ip, exhaustive=True))


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/internal range.

    Handles both IPv4 and IPv6 addresses, including IPv6 transition forms that embed an
    IPv4 address (IPv4-mapped, IPv4-compatible, 6to4, NAT64, ISATAP).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Invalid IP address, treat as potentially dangerous
        return True
    targets: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        targets.extend(_embedded_ipv4s(ip, exhaustive=False))
    return any(target in network for target in targets for network in _PRIVATE_NETWORKS)


async def resolve_hostname(hostname: str) -> list[str]:
    """Resolve a hostname to its IP addresses using DNS.

    Uses run_in_executor to run DNS resolution in a thread pool to avoid blocking.

    Returns:
        List of IP address strings, preserving DNS order with duplicates removed.

    Raises:
        ValueError: If DNS resolution fails.
    """
    try:
        # getaddrinfo returns list of (family, type, proto, canonname, sockaddr)
        # sockaddr is (ip, port) for IPv4 or (ip, port, flowinfo, scope_id) for IPv6
        results = await run_in_executor(socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        # Extract unique IP addresses, preserving order (first IP is typically preferred)
        seen: set[str] = set()
        ips: list[str] = []
        for result in results:
            ip = str(result[4][0])
            if ip not in seen:
                seen.add(ip)
                ips.append(ip)
        if not ips:
            raise ValueError(f'DNS resolution failed for hostname: {hostname}')  # pragma: no cover
        return ips
    except socket.gaierror as e:
        raise ValueError(f'DNS resolution failed for hostname "{hostname}": {e}') from e


def validate_url_protocol(url: str) -> tuple[str, bool]:
    """Validate that the URL uses an allowed protocol (http or https).

    Args:
        url: The URL to validate.

    Returns:
        Tuple of (scheme, is_https).

    Raises:
        ValueError: If the protocol is not http or https.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme not in ('http', 'https'):
        raise ValueError(f'URL protocol "{scheme}" is not allowed. Only http:// and https:// are supported.')

    return scheme, scheme == 'https'


def extract_host_and_port(url: str) -> tuple[str, str, int, bool]:
    """Extract hostname, path, port, and protocol info from a URL.

    Returns:
        Tuple of (hostname, path_with_query, port, is_https)

    Raises:
        ValueError: If the URL is malformed or uses an unsupported protocol.
    """
    # Validate protocol first, before trying to extract hostname
    _, is_https = validate_url_protocol(url)

    parsed = urlparse(url)
    hostname = parsed.hostname

    # Strip the trailing-dot (FQDN root label): DNS treats `host.` and `host` as the same,
    # so leaving it in would bypass exact-match domain allow/blocklists and skip the
    # IP-literal fast path (e.g. `169.254.169.254.`). urlparse already lowercases the host.
    if hostname:
        hostname = hostname.rstrip('.')

    if not hostname:
        raise ValueError(f'Invalid URL: no hostname found in "{url}"')

    default_port = 443 if is_https else 80
    port = parsed.port or default_port

    # Reconstruct path with query string
    path = parsed.path or '/'
    if parsed.query:
        path = f'{path}?{parsed.query}'
    if parsed.fragment:
        path = f'{path}#{parsed.fragment}'

    return hostname, path, port, is_https


def build_url_with_ip(resolved: ResolvedUrl) -> str:
    """Build a URL using a resolved IP address instead of the hostname.

    For IPv6 addresses, wraps them in brackets as required by URL syntax.
    """
    scheme = 'https' if resolved.is_https else 'http'
    default_port = 443 if resolved.is_https else 80

    # IPv6 addresses need brackets in URLs
    try:
        ip_obj = ipaddress.ip_address(resolved.resolved_ip)
        if isinstance(ip_obj, ipaddress.IPv6Address):
            host_part = f'[{resolved.resolved_ip}]'
        else:
            host_part = resolved.resolved_ip
    except ValueError:
        host_part = resolved.resolved_ip

    # Only include port if non-default
    if resolved.port != default_port:
        host_part = f'{host_part}:{resolved.port}'

    return urlunparse((scheme, host_part, resolved.path, '', '', ''))


async def validate_and_resolve_url(url: str, allow_local: bool) -> ResolvedUrl:
    """Validate URL and resolve hostname to IP addresses.

    Performs protocol validation, DNS resolution, and IP validation.

    Args:
        url: The URL to validate.
        allow_local: Whether to allow private/internal IP addresses.

    Returns:
        ResolvedUrl with all the information needed to make the request.

    Raises:
        ValueError: If the URL fails validation.
    """
    hostname, path, port, is_https = extract_host_and_port(url)

    # Check if hostname is already an IP address
    try:
        # Handle IPv6 addresses in brackets
        ip_str = hostname.strip('[]')
        ipaddress.ip_address(ip_str)
        ips = [ip_str]
    except ValueError:
        # It's a hostname, resolve it
        ips = await resolve_hostname(hostname)

    # Validate all resolved IPs
    for ip in ips:
        # Cloud metadata IPs are always blocked
        if is_cloud_metadata_ip(ip):
            raise ValueError(f'Access to cloud metadata service ({ip}) is blocked for security reasons.')

        # Private IPs are blocked unless allow_local is True
        if not allow_local and is_private_ip(ip):
            raise ValueError(
                f'Access to private/internal IP address ({ip}) is blocked. '
                f'Use force_download="allow-local" to allow local network access.'
            )

    # Use the first resolved IP
    return ResolvedUrl(
        resolved_ip=ips[0],
        hostname=hostname,
        port=port,
        is_https=is_https,
        path=path,
    )


def resolve_redirect_url(current_url: str, location: str) -> str:
    """Resolve a redirect location against the current URL.

    Args:
        current_url: The URL that returned the redirect.
        location: The Location header value (absolute or relative).

    Returns:
        The absolute URL to follow.
    """
    parsed_location = urlparse(location)

    # Check if it's an absolute URL (has scheme) or protocol-relative URL (has netloc but no scheme)
    if parsed_location.scheme:
        return location
    if parsed_location.netloc:
        # Protocol-relative URL (e.g., "//example.com/path") - use current scheme
        parsed_current = urlparse(current_url)
        return urlunparse(
            (
                parsed_current.scheme,
                parsed_location.netloc,
                parsed_location.path,
                '',
                parsed_location.query,
                parsed_location.fragment,
            )
        )

    # Relative URL - resolve against current URL
    parsed_current = urlparse(current_url)
    if location.startswith('/'):
        # Absolute path
        return urlunparse((parsed_current.scheme, parsed_current.netloc, location, '', '', ''))
    else:
        # Relative path
        base_path = parsed_current.path.rsplit('/', 1)[0]
        return urlunparse((parsed_current.scheme, parsed_current.netloc, f'{base_path}/{location}', '', '', ''))


def _check_domain(hostname: str, *, allowed_domains: list[str] | None, blocked_domains: list[str] | None) -> None:
    """Validate a hostname against allowed/blocked domain lists.

    Raises:
        ValueError: If the hostname is not allowed or is blocked.
    """
    if allowed_domains is not None and hostname not in allowed_domains:
        raise ValueError(f'Domain {hostname!r} is not in the allowed domains list. Allowed: {allowed_domains}')
    if blocked_domains is not None and hostname in blocked_domains:
        raise ValueError(f'Domain {hostname!r} is blocked.')


def _origin(url: str) -> tuple[str, str, int]:
    """Return the normalized origin (scheme, host, port) of a URL for redirect credential decisions.

    Normalization is delegated to `extract_host_and_port`, so the trailing-dot and
    lowercasing rules are the ones the request itself uses for DNS, `Host` and SNI, and
    the port defaults to 443 for https and 80 for http as in httpx's origin computation.

    Raises:
        ValueError: If the URL is malformed or uses an unsupported protocol, matching
            what `validate_and_resolve_url` would raise for the same URL.
    """
    hostname, _, port, is_https = extract_host_and_port(url)
    return 'https' if is_https else 'http', hostname, port


def _keeps_credentials(from_url: str, to_url: str) -> bool:
    """Whether sensitive headers may be forwarded from `from_url` to `to_url`.

    Credentials are kept on a same-origin redirect (scheme + host + port all
    match) and on an http→https upgrade on the same host (from http:80 to
    https:443); they are stripped on every other redirect, including port
    changes, https→http downgrades, and cross-host hops. This applies the
    origin rule httpx uses for `Authorization`, including its http→https
    upgrade exemption, to every header in `_SENSITIVE_HEADERS`.
    """
    from_scheme, from_host, from_port = _origin(from_url)
    to_scheme, to_host, to_port = _origin(to_url)
    if (from_scheme, from_host, from_port) == (to_scheme, to_host, to_port):
        return True
    return (
        from_scheme == 'http' and from_port == 80 and to_scheme == 'https' and to_port == 443 and from_host == to_host
    )


async def safe_download(
    url: str,
    allow_local: bool = False,
    max_redirects: int = _MAX_REDIRECTS,
    timeout: int = _DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    max_bytes: int | None = None,
) -> httpx2.Response:
    """Download content from a URL with SSRF protection.

    This function:
    1. Validates the URL protocol (only http/https allowed)
    2. Resolves the hostname to IP addresses
    3. Validates that no resolved IP is private (unless allow_local=True)
    4. Always blocks cloud metadata endpoints
    5. Validates the hostname against allowed/blocked domain lists
    6. Makes the request to the resolved IP with the Host header set
    7. Manually follows redirects, validating each hop

    Args:
        url: The URL to download from.
        allow_local: If True, allows requests to private/internal IP addresses.
                    Cloud metadata endpoints are always blocked regardless.
        max_redirects: Maximum number of redirects to follow (default: 10).
        timeout: Request timeout in seconds (default: 30).
        max_bytes: Maximum response-body size in bytes. When set, the response body
            is read as a stream and rejected once either the decoded body or the
            encoded stream it arrives in exceeds this limit.
        headers: Additional HTTP headers to include in the request.
                The `Host` header is always set to the original host, including a
                non-default port, and cannot be overridden. Sensitive headers (`Authorization`,
                `Cookie`, `Proxy-Authorization`) are stripped when a redirect
                crosses origins (scheme + host + port), except for a same-host
                http:80→https:443 upgrade.
        allowed_domains: If set, only these hostnames are permitted (exact match).
                Checked on every hop including redirects.
        blocked_domains: If set, these hostnames are rejected (exact match).
                Checked on every hop including redirects.

    Returns:
        The httpx2.Response object.

    Raises:
        ValueError: If the URL fails SSRF validation, domain validation,
                or too many redirects occur.
        httpx2.HTTPStatusError: If the response has an error status code. When legacy
            `httpx` is installed, this also matches `httpx.HTTPStatusError` handlers.
        httpx2.RequestError: If the request fails or the response body cannot be read.
            Request errors are re-raised at family level, so the specific subclass
            (`ConnectError`, `TimeoutException`, ...) is not preserved and handlers must
            catch `httpx2.RequestError` itself; when legacy `httpx` is installed, they also
            match `httpx.RequestError` handlers.
        httpx2.DecodingError: If a `gzip`-encoded body is malformed. When legacy `httpx` is
            installed, this also matches `httpx.DecodingError` handlers.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError('max_bytes must be non-negative')

    current_url = url
    redirects_followed = 0
    effective_headers: dict[str, str] = dict(headers) if headers else {}

    async with create_async_httpx2_client(timeout=timeout) as client:
        while True:
            # Validate and resolve the current URL
            resolved = await validate_and_resolve_url(current_url, allow_local)

            # Check domain restrictions (on every hop to prevent redirect bypass)
            _check_domain(resolved.hostname, allowed_domains=allowed_domains, blocked_domains=blocked_domains)

            # Build URL with resolved IP
            request_url = build_url_with_ip(resolved)

            # For HTTPS, set sni_hostname so TLS uses the original hostname for SNI
            # and certificate validation, even though we're connecting to the resolved IP.
            extensions: dict[str, str] = {}
            if resolved.is_https:
                extensions['sni_hostname'] = resolved.hostname

            request_headers: dict[str, str] = {k: v for k, v in effective_headers.items() if k.lower() != 'host'}
            default_port = 443 if resolved.is_https else 80
            if resolved.port == default_port:
                request_headers['Host'] = resolved.hostname
            else:
                host = resolved.hostname
                # Bracket an IPv6 literal before appending the port so the `:port` stays
                # unambiguous (RFC 3986 §3.2.2), matching the connect URL from build_url_with_ip.
                try:
                    if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
                        host = f'[{host}]'
                except ValueError:
                    pass
                request_headers['Host'] = f'{host}:{resolved.port}'
            if max_bytes is not None and not any(k.lower() == 'accept-encoding' for k in request_headers):
                request_headers['Accept-Encoding'] = _BOUNDED_ACCEPT_ENCODING

            # Stream the raw response so gzip members can be decoded and validated before
            # httpx2's automatic content decoder discards member boundaries.
            request = client.build_request('GET', request_url, headers=request_headers, extensions=extensions)
            response = await _send_request(client, request)

            # Check if we need to follow a redirect
            if response.is_redirect:
                await response.aclose()
                redirects_followed += 1
                if redirects_followed > max_redirects:
                    raise ValueError(f'Too many redirects ({redirects_followed}). Maximum allowed: {max_redirects}')

                # Get redirect location
                location = response.headers.get('location')
                if not location:
                    raise ValueError('Redirect response missing Location header')

                previous_url = current_url
                current_url = resolve_redirect_url(current_url, location)

                # Drop caller-supplied credentials when the redirect crosses origins, as
                # RFC 9110 section 15.4 advises for headers added by the calling context.
                if not _keeps_credentials(previous_url, current_url):
                    effective_headers = {
                        k: v for k, v in effective_headers.items() if k.lower() not in _SENSITIVE_HEADERS
                    }

                continue

            # Not a redirect, we're done
            try:
                try:
                    response.raise_for_status()
                except httpx2.HTTPStatusError as e:
                    raise _CompatibleHTTPStatusError(str(e), request=e.request, response=e.response) from e
                if max_bytes is not None:
                    content = await _read_capped_body(response, max_bytes)
                    return _response_with_decoded_content(response, content)
                if _content_encodings(response) in (['gzip'], ['x-gzip']):
                    content = await _read_gzip_body(response)
                    return _response_with_decoded_content(response, content)
                await _read_body(response)
                return response
            finally:
                await response.aclose()


def _download_exceeds(max_bytes: int) -> ValueError:
    return ValueError(_DOWNLOAD_EXCEEDS_TEMPLATE.format(max_bytes=max_bytes))


def _response_with_decoded_content(response: httpx2.Response, content: bytes) -> httpx2.Response:
    # Body is already decoded, so the reconstructed response must not carry the content
    # coding, or `httpx2.Response` would run it through the decoder again. `content-length`
    # described the encoded body and no longer applies; httpx2 recomputes it from `content`.
    decoded_headers = [
        (key, value)
        for key, value in response.headers.multi_items()
        if key.lower() not in ('content-encoding', 'content-length')
    ]
    return httpx2.Response(
        response.status_code,
        headers=decoded_headers,
        content=content,
        request=response.request,
        history=response.history,
        extensions=response.extensions,
    )


def _content_encodings(response: httpx2.Response) -> list[str]:
    encodings: list[str] = []
    for value in response.headers.get_list('content-encoding'):
        for part in value.split(','):
            coding = part.strip().lower()
            if coding and coding != 'identity':
                encodings.append(coding)
    return encodings


async def _read_capped_body(response: httpx2.Response, max_bytes: int) -> bytes:
    """Read a streamed response body without buffering more than `max_bytes` of decoded data.

    Streams the *encoded* body via `aiter_raw` so oversized wire traffic is rejected as it
    arrives, and applies gzip with zlib's output `max_length` so a highly compressible payload
    cannot expand past `max_bytes` mid-decode.

    Only `identity` and `gzip`/`x-gzip` are supported on this path. Codings that cannot be
    size-limited while streaming are rejected; callers with `max_bytes` also send
    `Accept-Encoding: identity, gzip` so servers are not invited to use them.

    Some transports (e.g. httpx2 mock responses built from an in-memory `content=` bytes object)
    preload the body and mark the stream consumed; in that case the decoded body is already in
    `response.content` and we only enforce the size cap on it.
    """
    if response.is_stream_consumed:
        data = response.content
        if len(data) > max_bytes:
            raise _download_exceeds(max_bytes)
        return data

    encodings = _content_encodings(response)

    if not encodings:
        return await _read_capped_identity(response, max_bytes)
    if encodings in (['gzip'], ['x-gzip']):
        return await _read_gzip_body(response, max_bytes)
    raise ValueError(
        f'Unsupported content-encoding for bounded download: {encodings}. '
        f'Only identity and gzip can be size-limited while streaming.'
    )


async def _aiter_raw(response: httpx2.Response) -> AsyncIterator[bytes]:
    """Stream the raw response body, re-raising read failures through the dual-family error type.

    The translation lives in the generator rather than around the consuming loop so that errors the
    loop body raises itself (the size cap, malformed gzip) keep their own type.
    """
    try:
        async for raw in response.aiter_raw():
            yield raw
    except httpx2.RequestError as e:
        raise _compatible_request_error(e) from e


async def _read_capped_identity(response: httpx2.Response, max_bytes: int) -> bytes:
    content = bytearray()
    async for raw in _aiter_raw(response):
        if len(content) + len(raw) > max_bytes:
            raise _download_exceeds(max_bytes)
        content.extend(raw)
    return bytes(content)


async def _read_gzip_body(response: httpx2.Response, max_bytes: int | None = None) -> bytes:
    content = bytearray()
    encoded_total = 0
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    member_started = False
    async for raw in _aiter_raw(response):
        encoded_total += len(raw)
        if max_bytes is not None and encoded_total > max_bytes:
            raise _download_exceeds(max_bytes)
        while raw:
            if not member_started:
                member_started = True
            elif decompressor.eof:
                # CPython's gzip reader accepts zero padding between/after members. Preserve
                # that compatibility while treating any other remaining bytes as a new member.
                raw = raw.lstrip(b'\x00')
                if not raw:
                    break
                decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)

            # Decompressing one byte past the cap distinguishes an oversized body from one that
            # exactly fills it, so a gzip CRC/ISIZE trailer arriving in a later chunk is still
            # consumed (it produces no output) instead of being rejected.
            max_length = max_bytes + 1 - len(content) if max_bytes is not None else 0
            try:
                content.extend(decompressor.decompress(raw, max_length=max_length))
            except zlib.error as e:
                raise _CompatibleDecodingError(f'Invalid gzip response body: {e}', request=response.request) from e
            if max_bytes is not None and len(content) > max_bytes:
                raise _download_exceeds(max_bytes)

            raw = decompressor.unconsumed_tail or decompressor.unused_data

    if not member_started:
        return b''
    try:
        content.extend(decompressor.flush())
    except zlib.error as e:
        raise _CompatibleDecodingError(f'Invalid gzip response body: {e}', request=response.request) from e
    if max_bytes is not None and len(content) > max_bytes:
        raise _download_exceeds(max_bytes)
    if not decompressor.eof:
        raise _CompatibleDecodingError('Received an incomplete gzip response body', request=response.request)
    return bytes(content)
