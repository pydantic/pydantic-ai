"""SSRF (Server-Side Request Forgery) protection for URL downloads.

This module provides security measures to prevent SSRF attacks when downloading
content from URLs. It validates protocols, resolves hostnames to IP addresses,
and blocks requests to private/internal networks and cloud metadata endpoints.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import httpx

from ._utils import run_in_executor
from .models import create_async_http_client

__all__ = ['safe_download']

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


async def safe_download(
    url: str,
    allow_local: bool = False,
    max_redirects: int = _MAX_REDIRECTS,
    timeout: int = _DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> httpx.Response:
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
        headers: Additional HTTP headers to include in the request.
                The `Host` header is always set to the original hostname
                and cannot be overridden.
        allowed_domains: If set, only these hostnames are permitted (exact match).
                Checked on every hop including redirects.
        blocked_domains: If set, these hostnames are rejected (exact match).
                Checked on every hop including redirects.

    Returns:
        The httpx.Response object.

    Raises:
        ValueError: If the URL fails SSRF validation, domain validation,
                or too many redirects occur.
        httpx.HTTPStatusError: If the response has an error status code.
    """
    current_url = url
    redirects_followed = 0
    original_hostname = urlparse(url).hostname
    effective_headers: dict[str, str] = dict(headers) if headers else {}

    async with create_async_http_client(timeout=timeout) as client:
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
            request_headers['Host'] = resolved.hostname

            # Make request with Host header set to original hostname
            response = await client.get(
                request_url,
                headers=request_headers,
                extensions=extensions,
                follow_redirects=False,
            )

            # Check if we need to follow a redirect
            if response.is_redirect:
                redirects_followed += 1
                if redirects_followed > max_redirects:
                    raise ValueError(f'Too many redirects ({redirects_followed}). Maximum allowed: {max_redirects}')

                # Get redirect location
                location = response.headers.get('location')
                if not location:
                    raise ValueError('Redirect response missing Location header')

                current_url = resolve_redirect_url(current_url, location)

                # Strip sensitive headers on cross-origin redirects (RFC 7235)
                redirect_hostname = urlparse(current_url).hostname
                if redirect_hostname != original_hostname:
                    effective_headers = {
                        k: v for k, v in effective_headers.items() if k.lower() not in _SENSITIVE_HEADERS
                    }

                continue

            # Not a redirect, we're done
            response.raise_for_status()
            return response
