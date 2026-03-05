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
from .models import cached_async_http_client

__all__ = ['safe_download']

# Private IP ranges that should be blocked by default
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    # IPv4 private ranges
    ipaddress.IPv4Network('127.0.0.0/8'),  # Loopback
    ipaddress.IPv4Network('10.0.0.0/8'),  # Private
    ipaddress.IPv4Network('172.16.0.0/12'),  # Private
    ipaddress.IPv4Network('192.168.0.0/16'),  # Private
    ipaddress.IPv4Network('169.254.0.0/16'),  # Link-local (includes cloud metadata)
    ipaddress.IPv4Network('0.0.0.0/8'),  # "This" network
    ipaddress.IPv4Network('100.64.0.0/10'),  # CGNAT (RFC 6598), includes Alibaba Cloud metadata
    # IPv6 private ranges
    ipaddress.IPv6Network('::1/128'),  # Loopback
    ipaddress.IPv6Network('fe80::/10'),  # Link-local
    ipaddress.IPv6Network('fc00::/7'),  # Unique local address
    ipaddress.IPv6Network('2002::/16'),  # 6to4 (can embed private IPv4 addresses)
)

# Cloud metadata IPs - always blocked, even with allow_local=True
# These need to be checked explicitly because when allow_local=True,
# we skip the private IP check but still need to block metadata endpoints.
_CLOUD_METADATA_IPS: frozenset[str] = frozenset(
    {
        '169.254.169.254',  # AWS, GCP, Azure metadata endpoint
        'fd00:ec2::254',  # AWS EC2 IPv6 metadata endpoint
        '100.100.100.200',  # Alibaba Cloud metadata endpoint
    }
)

_MAX_REDIRECTS = 10
_DEFAULT_TIMEOUT = 30  # seconds


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


def is_cloud_metadata_ip(ip_str: str) -> bool:
    """Check if an IP address is a cloud metadata endpoint.

    These are always blocked for security reasons, even with allow_local=True.
    """
    return ip_str in _CLOUD_METADATA_IPS


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/internal range.

    Handles both IPv4 and IPv6 addresses, including IPv4-mapped IPv6 addresses.
    """
    try:
        ip = ipaddress.ip_address(ip_str)

        # Handle IPv4-mapped IPv6 addresses (e.g., ::ffff:192.168.1.1)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped

        return any(ip in network for network in _PRIVATE_NETWORKS)
    except ValueError:
        # Invalid IP address, treat as potentially dangerous
        return True


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


async def safe_download(
    url: str,
    allow_local: bool = False,
    max_redirects: int = _MAX_REDIRECTS,
    timeout: int = _DEFAULT_TIMEOUT,
) -> httpx.Response:
    """Download content from a URL with SSRF protection.

    This function:
    1. Validates the URL protocol (only http/https allowed)
    2. Resolves the hostname to IP addresses
    3. Validates that no resolved IP is private (unless allow_local=True)
    4. Always blocks cloud metadata endpoints
    5. Makes the request to the resolved IP with the Host header set
    6. Manually follows redirects, validating each hop

    Args:
        url: The URL to download from.
        allow_local: If True, allows requests to private/internal IP addresses.
                    Cloud metadata endpoints are always blocked regardless.
        max_redirects: Maximum number of redirects to follow (default: 10).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        The httpx.Response object.

    Raises:
        ValueError: If the URL fails SSRF validation or too many redirects occur.
        httpx.HTTPStatusError: If the response has an error status code.
    """
    current_url = url
    redirects_followed = 0

    client = cached_async_http_client(timeout=timeout)
    while True:
        # Validate and resolve the current URL
        resolved = await validate_and_resolve_url(current_url, allow_local)

        # Build URL with resolved IP
        request_url = build_url_with_ip(resolved)

        # For HTTPS, set sni_hostname so TLS uses the original hostname for SNI
        # and certificate validation, even though we're connecting to the resolved IP.
        extensions: dict[str, str] = {}
        if resolved.is_https:
            extensions['sni_hostname'] = resolved.hostname

        # Make request with Host header set to original hostname
        response = await client.get(
            request_url,
            headers={'Host': resolved.hostname},
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
            continue

        # Not a redirect, we're done
        response.raise_for_status()
        return response
