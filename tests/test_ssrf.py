"""Tests for SSRF (Server-Side Request Forgery) protection."""

from __future__ import annotations

import gzip
import subprocess
import sys
import textwrap
import zlib
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from pydantic_ai import _ssrf
from pydantic_ai._ssrf import (
    _DEFAULT_TIMEOUT,  # pyright: ignore[reportPrivateUsage]
    _MAX_REDIRECTS,  # pyright: ignore[reportPrivateUsage]
    ResolvedUrl,
    build_url_with_ip,
    extract_host_and_port,
    is_cloud_metadata_ip,
    is_private_ip,
    resolve_hostname,
    resolve_redirect_url,
    safe_download,
    validate_and_resolve_url,
    validate_url_protocol,
)

pytestmark = [pytest.mark.anyio]


@pytest.fixture
def mock_dns(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch DNS resolution in _ssrf to prevent real network calls."""
    mock = AsyncMock()
    monkeypatch.setattr('pydantic_ai._ssrf.run_in_executor', mock)
    return mock


@pytest.fixture
def mock_ssrf_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch HTTP client creation in _ssrf to prevent real network calls.

    The wrapper configures the returned mock as an async context manager that yields
    itself (matching `httpx2.AsyncClient` behavior), so tests work regardless of
    whether `safe_download` uses the client directly or via `async with`.
    """
    mock = MagicMock()

    def factory_wrapper(**kwargs: Any) -> Any:
        client = mock(**kwargs)
        client.__aenter__.return_value = client
        # `safe_download` keeps the resolved-IP jar empty, so the mock needs a real
        # jar like `httpx2.AsyncClient` has.
        client.cookies = httpx2.Cookies()

        # Most tests in this file predate raw streaming and configure `get` directly.
        # Adapt that response setup to the build/send boundary used by `safe_download`.
        def build_request(method: str, url: str, **request_kwargs: Any) -> tuple[str, dict[str, Any]]:
            assert method == 'GET'
            return url, request_kwargs

        async def send(request: tuple[str, dict[str, Any]], *, follow_redirects: bool, stream: bool) -> httpx2.Response:
            assert stream is True
            url, request_kwargs = request
            response = await client.get(url, follow_redirects=follow_redirects, **request_kwargs)
            if not isinstance(response.headers, (dict, httpx2.Headers)):
                response.headers = httpx2.Headers()
            return response

        client.build_request = build_request
        client.send = send
        return client

    monkeypatch.setattr('pydantic_ai._ssrf.create_async_httpx2_client', factory_wrapper)
    return mock


class TestIsPrivateIp:
    """Tests for is_private_ip function."""

    @pytest.mark.parametrize(
        'ip',
        [
            # IPv4 loopback
            '127.0.0.1',
            '127.0.0.2',
            '127.255.255.255',
            # IPv4 private class A
            '10.0.0.1',
            '10.255.255.255',
            # IPv4 private class B
            '172.16.0.1',
            '172.31.255.255',
            # IPv4 private class C
            '192.168.0.1',
            '192.168.255.255',
            # IPv4 link-local
            '169.254.0.1',
            '169.254.255.255',
            # IPv4 "this" network
            '0.0.0.0',
            '0.255.255.255',
            # IPv4 CGNAT (RFC 6598)
            '100.64.0.1',
            '100.127.255.255',
            '100.100.100.200',  # Alibaba Cloud metadata
            # IPv4 IANA-reserved / special-purpose ranges
            '192.0.0.1',  # IETF Protocol Assignments (RFC 6890)
            '192.0.0.170',  # NAT64 well-known address
            '192.0.2.1',  # TEST-NET-1
            '198.18.0.1',  # Network benchmarking (RFC 2544)
            '198.19.255.255',
            '198.51.100.1',  # TEST-NET-2
            '203.0.113.1',  # TEST-NET-3
            '224.0.0.1',  # Multicast (RFC 5771)
            '239.255.255.255',
            '240.0.0.1',  # Reserved for future use
            '255.255.255.255',  # Limited broadcast
            # IPv6 loopback
            '::1',
            # IPv6 link-local
            'fe80::1',
            'fe80::ffff:ffff:ffff:ffff',
            # IPv6 unique local
            'fc00::1',
            'fdff:ffff:ffff:ffff:ffff:ffff:ffff:ffff',
            # IPv6 6to4 (can embed private IPv4)
            '2002::1',
            '2002:c0a8:0101::1',  # Embeds 192.168.1.1
            '2002:0a00:0001::1',  # Embeds 10.0.0.1
            # IPv6 IANA-reserved / special-purpose ranges
            '::',  # Unspecified
            '100::1',  # Discard prefix (RFC 6666)
            '2001::1',  # Teredo tunneling (RFC 4380)
            '2001:db8::1',  # Documentation (RFC 3849)
            'ff02::1',  # Multicast (RFC 4291)
            # NAT64 well-known prefix (RFC 6052) wrapping a private IPv4
            '64:ff9b::192.168.1.1',
            '64:ff9b::a9fe:a9fe',  # Wraps 169.254.169.254
            # NAT64 RFC 8215 local-use prefix 64:ff9b:1::/48 wrapping a private IPv4
            '64:ff9b:1::192.168.1.1',  # /96-style embedding
            '64:ff9b:1:c0a8:1:100::',  # proper RFC 6052 /48 embedding of 192.168.1.1
            # IPv4-compatible IPv6 ::a.b.c.d (deprecated, RFC 4291)
            '::192.168.1.1',
            '::10.0.0.1',
            '::a9fe:a9fe',  # 169.254.169.254
            # ISATAP (RFC 5214) with a public prefix, embedding a private/link-local IPv4
            '2606:4700::5efe:192.168.1.1',
            '2606:4700::200:5efe:169.254.169.254',
        ],
    )
    def test_private_ips_detected(self, ip: str) -> None:
        assert is_private_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            # Public IPv4
            '8.8.8.8',
            '1.1.1.1',
            '93.184.215.14',  # example.com
            '140.82.114.4',  # github.com
            # Public IPv6
            '2001:4860:4860::8888',
            '2606:4700:4700::1111',
        ],
    )
    def test_public_ips_allowed(self, ip: str) -> None:
        assert is_private_ip(ip) is False

    @pytest.mark.parametrize(
        'ip',
        [
            # IPv4-mapped IPv6 private addresses
            '::ffff:127.0.0.1',
            '::ffff:10.0.0.1',
            '::ffff:192.168.1.1',
            '::ffff:172.16.0.1',
        ],
    )
    def test_ipv4_mapped_ipv6_private(self, ip: str) -> None:
        assert is_private_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            # IPv4-mapped IPv6 public addresses
            '::ffff:8.8.8.8',
            '::ffff:1.1.1.1',
        ],
    )
    def test_ipv4_mapped_ipv6_public(self, ip: str) -> None:
        assert is_private_ip(ip) is False

    def test_invalid_ip_treated_as_private(self) -> None:
        """Invalid IP addresses should be treated as potentially dangerous."""
        assert is_private_ip('not-an-ip') is True
        assert is_private_ip('') is True


class TestIsCloudMetadataIp:
    """Tests for is_cloud_metadata_ip function."""

    @pytest.mark.parametrize(
        'ip',
        [
            '169.254.169.254',  # AWS IMDS, GCP, Azure, OCI, DigitalOcean, Hetzner, IBM, OpenStack
            '169.254.170.2',  # AWS ECS task IAM role credentials
            '169.254.170.23',  # AWS EKS Pod Identity Agent
            '168.63.129.16',  # Azure WireServer / platform channel (public IP)
            '100.100.100.200',  # Alibaba Cloud
            '192.0.0.192',  # Oracle Cloud (Classic)
            '169.254.42.42',  # Scaleway
            'fd00:ec2::254',  # AWS EC2 IMDS IPv6
            'fd00:ec2::23',  # AWS EKS Pod Identity Agent IPv6
            'fd20:ce::254',  # GCP IPv6 (IPv6-only instances)
            'fd00:42::42',  # Scaleway IPv6
        ],
    )
    def test_cloud_metadata_ips_detected(self, ip: str) -> None:
        assert is_cloud_metadata_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            '8.8.8.8',
            '127.0.0.1',
            '169.254.169.253',  # Close but not the metadata IP
            '169.254.169.255',
            '169.254.170.1',  # Close but not the ECS creds IP
            '169.254.170.3',
            '168.63.129.15',  # Close but not Azure WireServer
            '168.63.129.17',
            '100.100.100.199',  # Close but not Alibaba metadata
            '100.100.100.201',
        ],
    )
    def test_non_metadata_ips(self, ip: str) -> None:
        assert is_cloud_metadata_ip(ip) is False

    @pytest.mark.parametrize(
        'ip',
        [
            '::ffff:169.254.169.254',  # IPv4-mapped form of AWS/GCP/Azure metadata
            '::ffff:a9fe:a9fe',  # Same IP, hex-encoded last 32 bits
            '::ffff:100.100.100.200',  # IPv4-mapped form of Alibaba metadata
        ],
    )
    def test_ipv4_mapped_ipv6_metadata_detected(self, ip: str) -> None:
        """IPv4-mapped IPv6 forms of metadata IPs must be blocked.

        Dual-stack hosts route `::ffff:a.b.c.d` to the underlying IPv4 address,
        so allowing these would bypass the cloud-metadata blocklist when callers
        opt into `allow_local=True`. Regression test for the incomplete fix of
        GHSA-2jrp-274c-jhv3.
        """
        assert is_cloud_metadata_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            '64:ff9b::169.254.169.254',  # NAT64 wrap of AWS/GCP/Azure metadata
            '64:ff9b::a9fe:a9fe',  # Same, hex form of low 32 bits
            '64:ff9b::100.100.100.200',  # NAT64 wrap of Alibaba metadata
        ],
    )
    def test_nat64_metadata_detected(self, ip: str) -> None:
        """NAT64-wrapped (RFC 6052) metadata IPs must be blocked.

        In NAT64-configured networks, `64:ff9b::a.b.c.d` translates transparently
        to the IPv4 endpoint, so the wrapper must not disguise a metadata IP.
        """
        assert is_cloud_metadata_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            '2002:a9fe:a9fe::',  # 6to4 embedding 169.254.169.254 (AWS/GCP/Azure)
            '2002:6464:64c8::',  # 6to4 embedding 100.100.100.200 (Alibaba)
        ],
    )
    def test_6to4_metadata_detected(self, ip: str) -> None:
        """6to4-encoded (RFC 3056) metadata IPs must be blocked.

        On hosts with 6to4 routing, `2002:WWXX:YYZZ::` translates to the embedded
        IPv4 `W.X.Y.Z`, so the wrapper must not disguise a metadata IP.
        """
        assert is_cloud_metadata_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            # NAT64 RFC 8215 local-use prefix 64:ff9b:1::/48
            '64:ff9b:1::169.254.169.254',  # /96-style embedding
            '64:ff9b:1:a9fe:a9:fe00::',  # proper RFC 6052 /48 embedding
            # IPv4-compatible IPv6 ::a.b.c.d (deprecated)
            '::169.254.169.254',
            '::a9fe:a9fe',
            # ISATAP (RFC 5214) with a public prefix
            '2606:4700::5efe:169.254.169.254',
            '2606:4700::200:5efe:169.254.169.254',
            # Operator-chosen NAT64 prefix we cannot enumerate (caught by exhaustive sweep)
            '2001:db8:64::a9fe:a9fe',
            # Other clouds via transition forms
            '64:ff9b::169.254.170.2',  # AWS ECS creds via NAT64
            # Teredo (RFC 4380): client IPv4 is the low 32 bits XOR all-ones; 169.254.169.254
            '2001::5601:5601',
        ],
    )
    def test_transition_form_metadata_detected(self, ip: str) -> None:
        """Every standardized IPv6 transition encoding of a metadata IP must be blocked.

        Closes the class of bypasses behind CVE-2026-25580 / CVE-2026-46678: an IPv4
        metadata endpoint encoded as IPv4-mapped, IPv4-compatible, 6to4, NAT64 (any
        prefix), or ISATAP must not slip past the always-on cloud-metadata guard.
        """
        assert is_cloud_metadata_ip(ip) is True

    @pytest.mark.parametrize(
        'ip',
        [
            '::8.8.8.8',  # IPv4-compatible embedding public 8.8.8.8
            '2606:4700::5efe:8.8.8.8',  # ISATAP embedding public 8.8.8.8
            '64:ff9b::8.8.8.8',  # NAT64 embedding public 8.8.8.8
            '2606:4700:4700::1111',  # ordinary public IPv6 (low bits must not be misread)
        ],
    )
    def test_transition_form_public_not_metadata(self, ip: str) -> None:
        """Transition forms embedding a non-metadata IPv4 must not be misflagged."""
        assert is_cloud_metadata_ip(ip) is False

    def test_invalid_ip_not_metadata(self) -> None:
        assert is_cloud_metadata_ip('not-an-ip') is False


class TestValidateUrlProtocol:
    """Tests for validate_url_protocol function."""

    @pytest.mark.parametrize(
        'url',
        [
            'http://example.com',
            'https://example.com',
            'HTTP://EXAMPLE.COM',
            'HTTPS://EXAMPLE.COM',
        ],
    )
    def test_allowed_protocols(self, url: str) -> None:
        scheme, is_https = validate_url_protocol(url)
        assert scheme in ('http', 'https')
        assert is_https == (scheme == 'https')

    @pytest.mark.parametrize(
        ('url', 'protocol'),
        [
            ('file:///etc/passwd', 'file'),
            ('ftp://ftp.example.com/file.txt', 'ftp'),
            ('gopher://gopher.example.com', 'gopher'),
            ('gs://bucket/object', 'gs'),
            ('s3://bucket/key', 's3'),
            ('data:text/plain,hello', 'data'),
            ('javascript:alert(1)', 'javascript'),
        ],
    )
    def test_blocked_protocols(self, url: str, protocol: str) -> None:
        with pytest.raises(ValueError, match=f'URL protocol "{protocol}" is not allowed'):
            validate_url_protocol(url)


class TestExtractHostAndPort:
    """Tests for extract_host_and_port function."""

    def test_basic_http_url(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('http://example.com/path')
        assert hostname == 'example.com'
        assert path == '/path'
        assert port == 80
        assert is_https is False

    def test_basic_https_url(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('https://example.com/path')
        assert hostname == 'example.com'
        assert path == '/path'
        assert port == 443
        assert is_https is True

    def test_custom_port(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('http://example.com:8080/path')
        assert hostname == 'example.com'
        assert path == '/path'
        assert port == 8080
        assert is_https is False

    def test_path_with_query_string(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('https://example.com/path?query=value')
        assert hostname == 'example.com'
        assert path == '/path?query=value'
        assert port == 443
        assert is_https is True

    def test_path_with_fragment(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('https://example.com/path#fragment')
        assert hostname == 'example.com'
        assert path == '/path#fragment'
        assert port == 443
        assert is_https is True

    def test_empty_path(self) -> None:
        hostname, path, port, is_https = extract_host_and_port('https://example.com')
        assert hostname == 'example.com'
        assert path == '/'
        assert port == 443
        assert is_https is True

    def test_invalid_url_no_hostname(self) -> None:
        with pytest.raises(ValueError, match='Invalid URL: no hostname found'):
            extract_host_and_port('http://')


class TestBuildUrlWithIp:
    """Tests for build_url_with_ip function."""

    def test_http_default_port(self) -> None:
        resolved = ResolvedUrl(
            resolved_ip='203.0.113.50', hostname='example.com', port=80, is_https=False, path='/path'
        )
        url = build_url_with_ip(resolved)
        assert url == 'http://203.0.113.50/path'

    def test_https_default_port(self) -> None:
        resolved = ResolvedUrl(
            resolved_ip='203.0.113.50', hostname='example.com', port=443, is_https=True, path='/path'
        )
        url = build_url_with_ip(resolved)
        assert url == 'https://203.0.113.50/path'

    def test_custom_port(self) -> None:
        resolved = ResolvedUrl(
            resolved_ip='203.0.113.50', hostname='example.com', port=8080, is_https=False, path='/path'
        )
        url = build_url_with_ip(resolved)
        assert url == 'http://203.0.113.50:8080/path'

    def test_ipv6_address(self) -> None:
        resolved = ResolvedUrl(resolved_ip='2001:db8::1', hostname='example.com', port=443, is_https=True, path='/path')
        url = build_url_with_ip(resolved)
        assert url == 'https://[2001:db8::1]/path'

    def test_ipv6_address_custom_port(self) -> None:
        resolved = ResolvedUrl(
            resolved_ip='2001:db8::1', hostname='example.com', port=8443, is_https=True, path='/path'
        )
        url = build_url_with_ip(resolved)
        assert url == 'https://[2001:db8::1]:8443/path'


class TestResolveRedirectUrl:
    """Tests for resolve_redirect_url function."""

    def test_absolute_url(self) -> None:
        """Test that absolute URLs are returned as-is."""
        result = resolve_redirect_url('https://example.com/path', 'https://other.com/new-path')
        assert result == 'https://other.com/new-path'

    def test_protocol_relative_url(self) -> None:
        """Test that protocol-relative URLs use the current scheme."""
        result = resolve_redirect_url('https://example.com/path', '//other.com/new-path')
        assert result == 'https://other.com/new-path'

        result = resolve_redirect_url('http://example.com/path', '//other.com/new-path')
        assert result == 'http://other.com/new-path'

    def test_absolute_path(self) -> None:
        """Test that absolute paths are resolved against the current URL."""
        result = resolve_redirect_url('https://example.com/old/path', '/new/path')
        assert result == 'https://example.com/new/path'

    def test_relative_path(self) -> None:
        """Test that relative paths are resolved against the current URL."""
        result = resolve_redirect_url('https://example.com/old/path', 'new-file.txt')
        assert result == 'https://example.com/old/new-file.txt'

    def test_protocol_relative_url_preserves_query_and_fragment(self) -> None:
        """Test that protocol-relative URLs preserve query strings and fragments."""
        result = resolve_redirect_url('https://example.com/path', '//cdn.example.com/file.txt?token=abc#section')
        assert result == 'https://cdn.example.com/file.txt?token=abc#section'


class TestResolveHostname:
    """Tests for resolve_hostname function."""

    async def test_resolve_success(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [
            (2, 1, 6, '', ('93.184.215.14', 0)),
            (2, 1, 6, '', ('93.184.215.14', 0)),  # Duplicate should be removed
        ]
        ips = await resolve_hostname('example.com')
        assert ips == ['93.184.215.14']

    async def test_resolve_failure(self, mock_dns: AsyncMock) -> None:
        import socket

        mock_dns.side_effect = socket.gaierror('DNS lookup failed')
        with pytest.raises(ValueError, match='DNS resolution failed for hostname'):
            await resolve_hostname('nonexistent.invalid')


class TestValidateAndResolveUrl:
    """Tests for validate_and_resolve_url function."""

    async def test_public_ip_allowed(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        resolved = await validate_and_resolve_url('https://example.com/path', allow_local=False)
        assert resolved.resolved_ip == '93.184.215.14'
        assert resolved.hostname == 'example.com'
        assert resolved.port == 443
        assert resolved.is_https is True
        assert resolved.path == '/path'

    async def test_private_ip_blocked_by_default(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('192.168.1.1', 0))]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://internal.local/path', allow_local=False)

    async def test_private_ip_allowed_with_allow_local(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('192.168.1.1', 0))]
        resolved = await validate_and_resolve_url('http://internal.local/path', allow_local=True)
        assert resolved.resolved_ip == '192.168.1.1'

    async def test_cloud_metadata_always_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('169.254.169.254', 0))]
        with pytest.raises(ValueError, match='Access to cloud metadata service'):
            await validate_and_resolve_url('http://metadata.google.internal/path', allow_local=True)

    async def test_alibaba_cloud_metadata_always_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('100.100.100.200', 0))]
        with pytest.raises(ValueError, match='Access to cloud metadata service'):
            await validate_and_resolve_url('http://metadata.aliyun.internal/path', allow_local=True)

    @pytest.mark.parametrize(
        'url',
        [
            'http://[::ffff:169.254.169.254]/latest/meta-data/',  # IPv4-mapped IPv6 literal
            'http://[::ffff:a9fe:a9fe]/latest/meta-data/',  # Same address, hex form
            'http://[::ffff:100.100.100.200]/latest/meta-data/',  # Alibaba via IPv4-mapped IPv6
            'http://[64:ff9b::169.254.169.254]/latest/meta-data/',  # NAT64 wrap of metadata IP
            'http://[64:ff9b::a9fe:a9fe]/latest/meta-data/',  # Same, hex form
            'http://[2002:a9fe:a9fe::]/latest/meta-data/',  # 6to4 wrap of metadata IP
            'http://[64:ff9b:1::169.254.169.254]/latest/meta-data/',  # NAT64 RFC 8215 local-use prefix
            'http://[64:ff9b:1:a9fe:a9:fe00::]/latest/meta-data/',  # Same, proper RFC 6052 /48 embedding
            'http://[::169.254.169.254]/latest/meta-data/',  # IPv4-compatible IPv6
            'http://[::a9fe:a9fe]/latest/meta-data/',  # Same, hex form
            'http://[2606:4700::5efe:169.254.169.254]/latest/meta-data/',  # ISATAP, public prefix
            'http://[2001:db8:64::a9fe:a9fe]/latest/meta-data/',  # operator-chosen NAT64 prefix
        ],
    )
    async def test_transition_address_metadata_url_blocked_with_allow_local(self, url: str) -> None:
        """IPv6-encoded transition forms of metadata URLs must be blocked even with `allow_local=True`.

        Regression test for the incomplete-fix chain GHSA-2jrp-274c-jhv3 / CVE-2026-46678:
        an IPv4 metadata endpoint encoded as IPv4-mapped, IPv4-compatible, 6to4, NAT64 (any
        prefix, including RFC 8215 local-use and operator-chosen prefixes), or ISATAP must
        not bypass the always-on cloud-metadata guard, since dual-stack / NAT64 routing
        still delivers the request to the underlying IPv4 metadata endpoint.
        """
        with pytest.raises(ValueError, match='Access to cloud metadata service'):
            await validate_and_resolve_url(url, allow_local=True)

    async def test_ipv4_mapped_ipv6_metadata_dns_blocked_with_allow_local(self, mock_dns: AsyncMock) -> None:
        """A hostname that resolves to the IPv4-mapped IPv6 form of a metadata IP is still blocked."""
        mock_dns.return_value = [(10, 1, 6, '', ('::ffff:169.254.169.254', 0, 0, 0))]
        with pytest.raises(ValueError, match='Access to cloud metadata service'):
            await validate_and_resolve_url('http://attacker.example.com/path', allow_local=True)

    async def test_iana_reserved_ipv4_blocked(self, mock_dns: AsyncMock) -> None:
        """IANA-reserved IPv4 ranges (TEST-NETs, benchmarking, etc.) are blocked by default."""
        mock_dns.return_value = [(2, 1, 6, '', ('198.18.0.1', 0))]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://benchmark.example.com/path', allow_local=False)

    async def test_nat64_private_ipv4_blocked(self) -> None:
        """NAT64-wrapped private IPv4 addresses are blocked by default."""
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://[64:ff9b::192.168.1.1]/path', allow_local=False)

    async def test_literal_ip_address_in_url(self) -> None:
        resolved = await validate_and_resolve_url('http://8.8.8.8/path', allow_local=False)
        assert resolved.resolved_ip == '8.8.8.8'
        assert resolved.hostname == '8.8.8.8'

    async def test_literal_private_ip_blocked(self) -> None:
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://192.168.1.1/path', allow_local=False)

    async def test_any_private_ip_blocks_request(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [
            (2, 1, 6, '', ('93.184.215.14', 0)),
            (2, 1, 6, '', ('192.168.1.1', 0)),
        ]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://example.com/path', allow_local=False)

    async def test_6to4_address_blocked(self) -> None:
        # 2002:c0a8:0101::1 embeds 192.168.1.1
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://[2002:c0a8:0101::1]/path', allow_local=False)

    async def test_cgnat_range_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('100.64.0.1', 0))]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://cgnat-host.internal/path', allow_local=False)


RequestHandler = Callable[[httpx2.Request], httpx2.Response]


def stream_response(body: bytes, *, content_encoding: str | None = None) -> RequestHandler:
    """Builds a handler serving `body` as a streamed response, so it is read through `aiter_raw`."""
    return chunked_stream_response([body], content_encoding=content_encoding)


def chunked_stream_response(chunks: list[bytes], *, content_encoding: str | None = None) -> RequestHandler:
    """Builds a handler serving raw response-body chunks through `aiter_raw`."""
    headers = {'content-encoding': content_encoding} if content_encoding else {}

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        async def stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return httpx2.Response(200, content=stream(), headers=headers, request=request)

    return handle_request


class TestSafeDownload:
    """Tests for safe_download function."""

    async def test_negative_max_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_bytes must be non-negative'):
            await safe_download('https://example.com/file.txt', max_bytes=-1)

    async def test_successful_download(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_response.content = b'test content'

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        response = await safe_download('https://example.com/file.txt')
        assert response.content == b'test content'

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert '93.184.215.14' in call_args[0][0]
        assert call_args[1]['headers']['Host'] == 'example.com'
        assert call_args[1]['extensions'] == {'sni_hostname': 'example.com'}

    @pytest.mark.skipif(httpx is None, reason='legacy httpx is not installed')
    async def test_request_errors_remain_compatible_with_legacy_httpx(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """Public download errors keep matching legacy httpx request handlers when available."""
        assert httpx is not None
        request = httpx2.Request('GET', 'https://93.184.215.14/file.txt')
        client = AsyncMock()
        client.get.side_effect = httpx2.ConnectError('Connection failed', request=request)
        mock_ssrf_client.return_value = client
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        with pytest.raises(httpx.RequestError) as exc_info:
            await safe_download('https://example.com/file.txt')

        assert isinstance(exc_info.value, httpx2.RequestError)
        assert exc_info.value.request is request

    @pytest.mark.skipif(httpx is None, reason='legacy httpx is not installed')
    async def test_status_errors_remain_compatible_with_legacy_httpx(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """Public download errors keep matching legacy httpx status handlers when available."""
        assert httpx is not None
        request = httpx2.Request('GET', 'https://93.184.215.14/file.txt')
        response = httpx2.Response(404, request=request)
        client = AsyncMock()
        client.get.return_value = response
        mock_ssrf_client.return_value = client
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await safe_download('https://example.com/file.txt')

        assert isinstance(exc_info.value, httpx2.HTTPStatusError)
        assert exc_info.value.request is request
        assert exc_info.value.response is response

    @pytest.mark.skipif(httpx is None, reason='legacy httpx is not installed')
    @pytest.mark.parametrize('max_bytes', [None, 64])
    async def test_body_read_errors_remain_compatible_with_legacy_httpx(
        self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None
    ) -> None:
        """A transport failure that only surfaces while the body streams stays legacy-catchable.

        The unbounded path reads through `aread` and the bounded one through `aiter_raw`, so both
        boundaries are exercised.
        """
        assert httpx is not None

        def handle_request(request: httpx2.Request) -> httpx2.Response:
            async def stream() -> AsyncIterator[bytes]:
                yield b'partial'
                raise httpx2.ReadError('connection lost mid-body', request=request)

            return httpx2.Response(200, content=stream(), request=request)

        serve_requests(handle_request)

        with pytest.raises(httpx.RequestError) as exc_info:
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert isinstance(exc_info.value, httpx2.RequestError)

    @pytest.mark.skipif(httpx is None, reason='legacy httpx is not installed')
    @pytest.mark.parametrize('max_bytes', [None, 512])
    async def test_gzip_decoding_errors_remain_compatible_with_legacy_httpx(
        self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None
    ) -> None:
        """A malformed gzip body raises an error both families' decoding handlers match."""
        assert httpx is not None
        encoded = bytearray(gzip.compress(b'corrupt gzip body'))
        encoded[-8] ^= 0xFF
        serve_requests(stream_response(bytes(encoded), content_encoding='gzip'))

        with pytest.raises(httpx.DecodingError) as exc_info:
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert isinstance(exc_info.value, httpx2.DecodingError)
        assert isinstance(exc_info.value, httpx.RequestError)
        assert exc_info.value.request.url.path == '/file.txt'

    def test_safe_download_works_without_legacy_httpx(self) -> None:
        """The public download boundary remains usable when only httpx2 is installed."""
        script = """
            import asyncio
            import sys

            import httpx2


            class BlockHttpx:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == 'httpx' or fullname.startswith('httpx.'):
                        raise ImportError('httpx is not installed')


            sys.meta_path.insert(0, BlockHttpx())

            from pydantic_ai import _ssrf


            class FailingClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

                def build_request(self, method, url, **kwargs):
                    return httpx2.Request(method, url, **kwargs)

                async def send(self, request, **kwargs):
                    raise httpx2.ConnectError('Connection failed', request=request)


            async def validate_and_resolve_url(url, allow_local):
                return _ssrf.ResolvedUrl(
                    resolved_ip='93.184.216.34',
                    hostname='example.com',
                    port=80,
                    is_https=False,
                    path='/',
                )


            def create_async_httpx2_client(*, timeout):
                return FailingClient()


            _ssrf.validate_and_resolve_url = validate_and_resolve_url
            _ssrf.create_async_httpx2_client = create_async_httpx2_client


            async def main():
                try:
                    await _ssrf.safe_download('http://example.com')
                except httpx2.RequestError as error:
                    assert type(error) is httpx2.RequestError
                else:
                    raise AssertionError('safe_download did not raise the failed request')


            asyncio.run(main())
            assert not any(name == 'httpx' or name.startswith('httpx.') for name in sys.modules)
        """
        result = subprocess.run(
            [sys.executable, '-W', 'error', '-c', textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr == ''

    @pytest.mark.parametrize(
        ('url', 'expected_host'),
        [
            ('https://example.com/file.txt', 'example.com'),
            ('https://example.com:8443/file.txt', 'example.com:8443'),
            ('http://example.com/file.txt', 'example.com'),
            ('http://example.com:8080/file.txt', 'example.com:8080'),
            ('https://[2606:4700:4700::1111]:8443/file.txt', '[2606:4700:4700::1111]:8443'),
            ('http://93.184.215.14:8080/file.txt', '93.184.215.14:8080'),
        ],
    )
    async def test_host_header_includes_non_default_port(
        self, url: str, expected_host: str, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """The Host header includes the non-default port, matching the connect URL and RFC 9110 §7.2."""
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download(url)

        call_args = mock_client.get.call_args
        assert call_args[1]['headers']['Host'] == expected_host

    @pytest.fixture
    def serve_requests(self, mock_dns: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> Callable[[RequestHandler], None]:
        """Serves canned responses to `safe_download` through an `httpx2.MockTransport`.

        Also resolves every hostname to a public IP, so the download reaches the handler.
        """

        def serve(handler: RequestHandler) -> None:
            mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
            client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

            def create_http_client(*, timeout: int) -> httpx2.AsyncClient:
                return client

            monkeypatch.setattr('pydantic_ai._ssrf.create_async_httpx2_client', create_http_client)

        return serve

    async def test_max_bytes_reads_streamed_body(self, serve_requests: Callable[[RequestHandler], None]) -> None:
        """A bounded download buffers a streamed response only after enforcing its limit."""
        serve_requests(lambda request: httpx2.Response(200, content=b'streamed content', request=request))

        response = await safe_download('https://example.com/file.txt', max_bytes=16)

        assert response.content == b'streamed content'

    async def test_max_bytes_rejects_oversized_streamed_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """A missing or false content-length header cannot bypass the streamed-body limit."""
        serve_requests(lambda request: httpx2.Response(200, content=b'content longer than the limit', request=request))

        with pytest.raises(ValueError, match='maximum size of 16 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=16)

    async def test_max_bytes_rejects_body_that_decodes_oversized(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """A small compressed body that expands past the limit is rejected on its decoded size."""
        encoded = gzip.compress(bytes(1_000_000))
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        assert len(encoded) < 100_000
        with pytest.raises(ValueError, match='maximum size of 100000 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=100_000)

    async def test_max_bytes_rejects_gzip_bomb_without_materializing_full_decoded_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """A single highly compressible raw chunk is rejected without buffering the full expansion.

        `aiter_bytes` would materialize the entire decoded bomb before any size check; the capped
        path must use `max_length` decompression so peak decoded buffering stays near the limit.
        """
        # ~1 MiB of zeros compresses to a tiny gzip payload delivered as one network chunk.
        encoded = gzip.compress(bytes(1_000_000))
        max_bytes = 10_000
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        assert len(encoded) < max_bytes
        with pytest.raises(ValueError, match=f'maximum size of {max_bytes} bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

    async def test_max_bytes_rejects_oversized_encoded_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """The limit bounds the encoded stream too, so a body that decodes small can't stream on unchecked."""
        payload = bytes(range(256))
        encoded = gzip.compress(payload)
        max_bytes = (len(payload) + len(encoded)) // 2
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        with pytest.raises(ValueError, match=f'maximum size of {max_bytes} bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

    async def test_max_bytes_decodes_compressed_body_once(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """A bounded download of a compressed body decodes it exactly once.

        The client advertises `gzip` on every request, so the streamed path buffers already-decoded
        bytes; carrying `content-encoding` into the reconstructed response would decode them again.
        """
        serve_requests(
            lambda request: httpx2.Response(
                200,
                content=gzip.compress(b'streamed content'),
                headers={'content-encoding': 'gzip'},
                request=request,
            )
        )

        response = await safe_download('https://example.com/file.txt', max_bytes=64)

        assert response.content == b'streamed content'
        assert 'content-encoding' not in response.headers

    async def test_max_bytes_follows_redirect(self, serve_requests: Callable[[RequestHandler], None]) -> None:
        """A bounded download closes each streamed redirect hop before re-requesting."""

        def handle_request(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == '/file.txt':
                return httpx2.Response(302, headers={'location': 'https://example.com/final.txt'}, request=request)
            return stream_response(b'redirected content')(request)

        serve_requests(handle_request)

        response = await safe_download('https://example.com/file.txt', max_bytes=64)

        assert response.content == b'redirected content'

    async def test_max_bytes_reads_streamed_identity_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """Unencoded bodies are read via `aiter_raw` under the size cap."""
        serve_requests(stream_response(b'streamed content'))

        response = await safe_download('https://example.com/file.txt', max_bytes=64)
        assert response.content == b'streamed content'

    async def test_max_bytes_rejects_oversized_streamed_identity_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        serve_requests(stream_response(b'content longer than the limit'))

        with pytest.raises(ValueError, match='maximum size of 16 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=16)

    @pytest.mark.parametrize('max_bytes', [None, 64])
    async def test_x_gzip_alias(self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None) -> None:
        payload = b'x-gzip body'
        serve_requests(stream_response(gzip.compress(payload), content_encoding='x-gzip'))

        response = await safe_download('https://example.com/file.txt', max_bytes=max_bytes)
        assert response.content == payload

    @pytest.mark.parametrize('max_bytes', [None, 512])
    @pytest.mark.parametrize('split_after', [None, 'member', 'header'])
    async def test_decodes_all_gzip_members(
        self,
        serve_requests: Callable[[RequestHandler], None],
        max_bytes: int | None,
        split_after: str | None,
    ) -> None:
        """Every gzip member is decoded, including members split across raw network chunks."""
        first = gzip.compress(b'first member\n')
        second = gzip.compress(b'second member\n')
        encoded = first + second
        if split_after == 'member':
            chunks = [encoded[: len(first)], encoded[len(first) :]]
        elif split_after == 'header':
            chunks = [encoded[: len(first) + 3], encoded[len(first) + 3 :]]
        else:
            chunks = [encoded]
        serve_requests(chunked_stream_response(chunks, content_encoding='gzip'))

        response = await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert response.content == b'first member\nsecond member\n'
        assert 'content-encoding' not in response.headers

    @pytest.mark.parametrize('max_bytes', [None, 512])
    @pytest.mark.parametrize('complete_prefix', [False, True], ids=['first-member', 'later-member'])
    async def test_rejects_truncated_gzip_member(
        self,
        serve_requests: Callable[[RequestHandler], None],
        max_bytes: int | None,
        complete_prefix: bool,
    ) -> None:
        """A missing CRC/ISIZE trailer is reported instead of returning partial content."""
        encoded = gzip.compress(b'x' * 200)[:-8]
        if complete_prefix:
            encoded = gzip.compress(b'complete member') + encoded
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        with pytest.raises(httpx2.DecodingError, match='incomplete gzip') as exc_info:
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert exc_info.value.request.url.path == '/file.txt'

    @pytest.mark.parametrize('max_bytes', [None, 512])
    async def test_rejects_corrupt_gzip_member(
        self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None
    ) -> None:
        """CRC failures consistently use HTTPX's content-decoding error type."""
        encoded = bytearray(gzip.compress(b'corrupt gzip body'))
        encoded[-8] ^= 0xFF
        serve_requests(stream_response(bytes(encoded), content_encoding='gzip'))

        with pytest.raises(httpx2.DecodingError) as exc_info:
            await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert exc_info.value.request.url.path == '/file.txt'

    @pytest.mark.parametrize('max_bytes', [None, 512])
    async def test_accepts_empty_gzip_body(
        self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None
    ) -> None:
        """Preserve the existing behavior for an empty body labelled as gzip."""
        serve_requests(stream_response(b'', content_encoding='gzip'))

        response = await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert response.content == b''

    @pytest.mark.parametrize('max_bytes', [None, 512])
    async def test_accepts_zero_padding_between_and_after_gzip_members(
        self, serve_requests: Callable[[RequestHandler], None], max_bytes: int | None
    ) -> None:
        """Preserve Python gzip compatibility for zero padding around later members."""
        encoded = gzip.compress(b'first') + b'\x00\x00' + gzip.compress(b'second') + b'\x00\x00'
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        response = await safe_download('https://example.com/file.txt', max_bytes=max_bytes)

        assert response.content == b'firstsecond'

    async def test_max_bytes_applies_across_gzip_members(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """The decoded-size limit is cumulative across concatenated gzip members."""
        encoded = gzip.compress(b'a' * 40) + gzip.compress(b'b' * 40)
        assert len(encoded) < 64
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        with pytest.raises(ValueError, match='maximum size of 64 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=64)

    async def test_max_bytes_strips_identity_from_content_encoding(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        payload = b'identity stripped'
        serve_requests(stream_response(gzip.compress(payload), content_encoding='identity, gzip'))

        response = await safe_download('https://example.com/file.txt', max_bytes=64)
        assert response.content == payload

    @pytest.mark.parametrize('coding', ['br', 'zstd', 'deflate', 'gzip, deflate', 'not-a-real-coding'])
    async def test_max_bytes_rejects_unsupported_content_encoding(
        self, serve_requests: Callable[[RequestHandler], None], coding: str
    ) -> None:
        """Brotli/zstd/deflate/stacked/unknown codings are rejected rather than decoded unsafely."""

        def handle_request(request: httpx2.Request) -> httpx2.Response:
            # Body is never read: unsupported content-coding is rejected before streaming.
            async def stream() -> AsyncIterator[bytes]:  # pragma: no cover
                yield b'x'

            return httpx2.Response(200, content=stream(), headers={'content-encoding': coding}, request=request)

        serve_requests(handle_request)

        with pytest.raises(ValueError, match='Unsupported content-encoding for bounded download'):
            await safe_download('https://example.com/file.txt', max_bytes=64)

    async def test_max_bytes_sets_bounded_accept_encoding(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """Bounded downloads negotiate only encodings that can be size-limited while streaming."""
        seen: dict[str, str] = {}

        def handle_request(request: httpx2.Request) -> httpx2.Response:
            seen['accept-encoding'] = request.headers.get('accept-encoding', '')
            return stream_response(b'ok')(request)

        serve_requests(handle_request)

        response = await safe_download('https://example.com/file.txt', max_bytes=64)
        assert response.content == b'ok'
        assert seen['accept-encoding'] == 'identity, gzip'

    async def test_max_bytes_rejects_oversized_preloaded_body(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """Transports that preload `content=` still enforce the decoded size cap."""
        serve_requests(lambda request: httpx2.Response(200, content=b'x' * 2000, request=request))

        with pytest.raises(ValueError, match='maximum size of 1024 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=1024)

    async def test_max_bytes_rejects_oversized_gzip_encoded_stream(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """Encoded gzip wire traffic above the cap is rejected before full decompression."""
        # Incompressible-ish payload so the gzip frame itself exceeds a small cap.
        encoded = gzip.compress(bytes(range(256)) * 8)
        assert len(encoded) > 64
        serve_requests(stream_response(encoded, content_encoding='gzip'))

        with pytest.raises(ValueError, match='maximum size of 64 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=64)

    async def test_max_bytes_accepts_gzip_body_exactly_at_cap_with_split_trailer(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """A valid gzip body whose size equals the cap must not fail when the trailer is a separate chunk."""
        payload = b'x' * 100
        encoded = gzip.compress(payload)

        def handle_request(request: httpx2.Request) -> httpx2.Response:
            async def stream() -> AsyncIterator[bytes]:
                yield encoded[:-8]
                yield encoded[-8:]

            return httpx2.Response(200, content=stream(), headers={'content-encoding': 'gzip'}, request=request)

        serve_requests(handle_request)

        response = await safe_download('https://example.com/file.txt', max_bytes=100)
        assert response.content == payload

    async def test_max_bytes_rejects_gzip_flush_overflow(
        self, serve_requests: Callable[[RequestHandler], None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Streamed gzip rejects when the final flush would push the decoded body past the cap."""

        class _Flushy:
            eof = True
            unused_data = b''

            def decompress(self, data: bytes, max_length: int = 0) -> bytes:
                return b'abcd'

            @property
            def unconsumed_tail(self) -> bytes:
                return b''

            def flush(self) -> bytes:
                return b'e'

        def _make_flushy(wbits: int) -> _Flushy:
            return _Flushy()

        monkeypatch.setattr(_ssrf.zlib, 'decompressobj', _make_flushy)
        serve_requests(stream_response(b'raw', content_encoding='gzip'))

        with pytest.raises(ValueError, match='maximum size of 4 bytes'):
            await safe_download('https://example.com/file.txt', max_bytes=4)

    async def test_gzip_flush_error_uses_httpx_decoding_error(
        self, serve_requests: Callable[[RequestHandler], None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zlib failure during finalization does not leak its implementation-specific type."""

        class _BadFlush:
            eof = False
            unused_data = b''
            unconsumed_tail = b''

            def decompress(self, data: bytes, max_length: int = 0) -> bytes:
                return b''

            def flush(self) -> bytes:
                raise zlib.error('invalid stream')

        def make_bad_flush(wbits: int) -> _BadFlush:
            return _BadFlush()

        monkeypatch.setattr(_ssrf.zlib, 'decompressobj', make_bad_flush)
        serve_requests(stream_response(b'raw', content_encoding='gzip'))

        with pytest.raises(httpx2.DecodingError, match='Invalid gzip response body') as exc_info:
            await safe_download('https://example.com/file.txt')

        assert exc_info.value.request.url.path == '/file.txt'

    async def test_redirect_followed(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://cdn.example.com/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('140.82.114.4', 0))],
        ]

        mock_client = AsyncMock()
        mock_client.get.side_effect = [redirect_response, final_response]
        mock_ssrf_client.return_value = mock_client

        response = await safe_download('https://example.com/file.txt')
        assert response.content == b'final content'
        assert mock_client.get.call_count == 2

    async def test_redirect_to_private_ip_blocked(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'http://internal.local/file.txt'}

        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('192.168.1.1', 0))],
        ]

        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await safe_download('https://example.com/file.txt')

    async def test_max_redirects_exceeded(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://example.com/redirect'}

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match=f'Too many redirects \\({_MAX_REDIRECTS + 1}\\)'):
            await safe_download('https://example.com/file.txt')

    async def test_relative_redirect_resolved(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '/new-path/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.side_effect = [redirect_response, final_response]
        mock_ssrf_client.return_value = mock_client

        response = await safe_download('https://example.com/old-path/file.txt')
        assert response.content == b'final content'

        second_call = mock_client.get.call_args_list[1]
        assert '/new-path/file.txt' in second_call[0][0]

    async def test_missing_location_header(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {}

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match='Redirect response missing Location header'):
            await safe_download('https://example.com/file.txt')

    async def test_protocol_relative_redirect(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '//cdn.example.com/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('140.82.114.4', 0))],
        ]

        mock_client = AsyncMock()
        mock_client.get.side_effect = [redirect_response, final_response]
        mock_ssrf_client.return_value = mock_client

        response = await safe_download('https://example.com/file.txt')
        assert response.content == b'final content'
        assert mock_client.get.call_count == 2

        second_call = mock_client.get.call_args_list[1]
        assert second_call[1]['headers']['Host'] == 'cdn.example.com'

    async def test_protocol_relative_redirect_to_private_blocked(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '//internal.local/file.txt'}

        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('192.168.1.1', 0))],
        ]

        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await safe_download('https://example.com/file.txt')

    async def test_http_no_sni_extension(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download('http://example.com/file.txt')

        call_args = mock_client.get.call_args
        assert call_args[1]['extensions'] == {}

    async def test_server_cookies_are_isolated_from_hosts_sharing_an_ip(
        self, serve_requests: Callable[[RequestHandler], None]
    ) -> None:
        """Each host receives only its own cookies even though every hop uses one IP."""

        sent_cookies: list[str | None] = []
        hop = 0

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal hop
            hop += 1
            assert request.url.host == '93.184.215.14'
            host = request.headers.get('host', '')
            if hop == 1:
                assert host == 'tenant-a.example.com'
                return httpx2.Response(
                    302,
                    headers={'location': 'https://tenant-b.example.com/file', 'set-cookie': 'sid=a; Path=/'},
                    request=request,
                )
            if hop == 2:
                assert host == 'tenant-b.example.com'
                sent_cookies.append(request.headers.get('cookie'))
                return httpx2.Response(
                    302,
                    headers={'location': 'https://tenant-a.example.com/file', 'set-cookie': 'sid=b; Path=/'},
                    request=request,
                )
            assert host == 'tenant-a.example.com'
            sent_cookies.append(request.headers.get('cookie'))
            return httpx2.Response(200, content=b'final', request=request)

        serve_requests(handler)
        await safe_download('https://tenant-a.example.com./file')

        assert sent_cookies == [None, 'sid=a']

    @pytest.mark.parametrize(
        ('start_url', 'redirect_url', 'set_cookie', 'expected_cookie'),
        [
            pytest.param(
                'https://a.example/start',
                'https://a.example/download',
                'sid=secret; Path=/',
                'sid=secret',
                id='same-host',
            ),
            pytest.param(
                'https://app.example.com/start',
                'https://cdn.example.com/download',
                'sid=secret; Domain=example.com; Path=/',
                None,
                id='domain-cookie-not-shared-between-hosts',
            ),
            pytest.param(
                'https://attacker.co.uk/start',
                'https://bank.co.uk/download',
                'sid=secret; Domain=co.uk; Path=/',
                None,
                id='public-suffix-cookie-not-shared-between-hosts',
            ),
            pytest.param(
                'https://93.184.216.34/start',
                'https://52.184.216.34/download',
                'sid=secret; Domain=184.216.34; Path=/',
                None,
                id='ip-suffix-cookie-not-shared-between-hosts',
            ),
            pytest.param(
                'https://evil.example.com/start',
                'https://victim.example.com/download',
                '__Host-sid=secret; Domain=example.com; Path=/; Secure',
                None,
                id='protected-cookie-not-shared-between-hosts',
            ),
            pytest.param(
                'https://a.example/private/start',
                'https://a.example/public/download',
                'sid=secret; Path=/private',
                None,
                id='path-mismatch',
            ),
            pytest.param(
                'https://a.example/start',
                'http://a.example/download',
                'sid=secret; Secure; Path=/',
                None,
                id='secure-cookie-downgrade',
            ),
            pytest.param(
                'https://a.example:8443/start',
                'https://a.example:9443/download',
                'sid=secret; Path=/',
                'sid=secret',
                id='cookies-ignore-port',
            ),
        ],
    )
    async def test_server_cookie_scope(
        self,
        serve_requests: Callable[[RequestHandler], None],
        start_url: str,
        redirect_url: str,
        set_cookie: str,
        expected_cookie: str | None,
    ) -> None:
        sent_cookies: list[str | None] = []
        first_request = True

        def handler(request: httpx2.Request) -> httpx2.Response:
            nonlocal first_request
            if first_request:
                first_request = False
                return httpx2.Response(
                    302,
                    headers={'location': redirect_url, 'set-cookie': set_cookie},
                    request=request,
                )
            sent_cookies.append(request.headers.get('cookie'))
            return httpx2.Response(200, content=b'final', request=request)

        serve_requests(handler)
        await safe_download(start_url)

        assert sent_cookies == [expected_cookie]

    async def test_protocol_validation(self) -> None:
        with pytest.raises(ValueError, match='URL protocol "file" is not allowed'):
            await safe_download('file:///etc/passwd')

        with pytest.raises(ValueError, match='URL protocol "ftp" is not allowed'):
            await safe_download('ftp://ftp.example.com/file.txt')

    async def test_timeout_parameter(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download('https://example.com/file.txt', timeout=60)

        mock_ssrf_client.assert_called_once_with(timeout=60)

    async def test_default_timeout(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download('https://example.com/file.txt')

        mock_ssrf_client.assert_called_once_with(timeout=_DEFAULT_TIMEOUT)

    async def test_safe_download_closes_http_client(self, mock_dns: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
        """`safe_download` closes the HTTP client it creates, even on success.

        Without proper cleanup, each call to `safe_download` leaks an unclosed
        `httpx2.AsyncClient`. After switching from cached_async_http_client (which
        reused a global) to `create_async_httpx2_client` (new client per call),
        the client must be explicitly closed.

        Regression test for PR #4421 auto-review feedback.
        https://github.com/pydantic/pydantic-ai/pull/4421
        """
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        created_clients: list[httpx2.AsyncClient] = []

        def tracking_create(**kwargs: Any) -> httpx2.AsyncClient:
            client = httpx2.AsyncClient(
                transport=httpx2.MockTransport(
                    lambda request: httpx2.Response(200, content=b'test content', request=request)
                )
            )
            created_clients.append(client)
            return client

        monkeypatch.setattr('pydantic_ai._ssrf.create_async_httpx2_client', tracking_create)

        response = await safe_download('https://example.com/file.txt')
        assert response.content == b'test content'
        assert len(created_clients) == 1
        assert created_clients[0].is_closed

    async def test_allowed_domains_permits(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        """Test that allowed domain passes validation."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download('https://example.com/page', allowed_domains=['example.com'])

    async def test_allowed_domains_blocks(self, mock_dns: AsyncMock) -> None:
        """Test that non-allowed domain is rejected."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        with pytest.raises(ValueError, match='not in the allowed domains'):
            await safe_download('https://evil.com/page', allowed_domains=['example.com'])

    @pytest.mark.parametrize(
        'url', ['https://example.com./page', 'https://EXAMPLE.com/page', 'https://Example.Com./page']
    )
    async def test_allowed_domains_normalizes_host(
        self, url: str, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """A trailing FQDN dot or uppercasing must not cause false rejection by the allowed-domains list."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download(url, allowed_domains=['example.com'])

    async def test_blocked_domains_blocks(self, mock_dns: AsyncMock) -> None:
        """Test that blocked domain is rejected."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        with pytest.raises(ValueError, match='is blocked'):
            await safe_download('https://evil.com/page', blocked_domains=['evil.com'])

    @pytest.mark.parametrize('url', ['https://evil.com./page', 'https://EVIL.com/page', 'https://Evil.Com./page'])
    async def test_blocked_domains_normalizes_host(self, url: str, mock_dns: AsyncMock) -> None:
        """A trailing FQDN dot or uppercasing must not bypass the blocked-domains list."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        with pytest.raises(ValueError, match='is blocked'):
            await safe_download(url, blocked_domains=['evil.com'])

    async def test_blocked_domains_permits(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        """Test that non-blocked domain passes validation."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_ssrf_client.return_value = mock_client

        await safe_download('https://example.com/page', blocked_domains=['evil.com'])

    async def test_redirect_to_blocked_domain_rejected(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        """Test that redirects to blocked domains are caught."""
        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('140.82.114.4', 0))],
        ]
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://evil.com/payload'}
        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match='is blocked'):
            await safe_download('https://example.com/page', blocked_domains=['evil.com'])

    async def test_redirect_to_non_allowed_domain_rejected(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """Test that redirects to non-allowed domains are caught."""
        mock_dns.side_effect = [
            [(2, 1, 6, '', ('93.184.215.14', 0))],
            [(2, 1, 6, '', ('140.82.114.4', 0))],
        ]
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://other.com/page'}
        mock_client = AsyncMock()
        mock_client.get.return_value = redirect_response
        mock_ssrf_client.return_value = mock_client

        with pytest.raises(ValueError, match='not in the allowed domains'):
            await safe_download('https://example.com/page', allowed_domains=['example.com'])


class TestSensitiveHeaderStrippingOnRedirects:
    """Tests for sensitive-header (Authorization/Cookie/Proxy-Authorization) stripping on redirects.

    `safe_download` compares full origins (scheme + host + port) against the *previous* hop,
    keeping credentials on same-origin redirects and same-host http:80→https:443 upgrades, and
    stripping them on cross-host hops, port changes, and https→http downgrades
    (RFC 9110 section 15.4). See https://github.com/pydantic/pydantic-ai/issues/6810.

    These patch the client rather than using VCR because no real endpoint deterministically
    issues redirect chains that change scheme, port, and host on demand, and because the
    assertions are about what we send rather than what a server replies.
    """

    _SENSITIVE_VALUES = {
        'Authorization': 'Bearer SECRET',
        'Cookie': 'session=abc',
        'Proxy-Authorization': 'Basic abc',
    }

    @staticmethod
    def _redirect_response(location: str) -> AsyncMock:
        response = AsyncMock()
        response.is_redirect = True
        response.headers = {'location': location}
        return response

    @staticmethod
    def _final_response() -> AsyncMock:
        response = AsyncMock()
        response.is_redirect = False
        response.raise_for_status = lambda: None
        response.content = b'final'
        return response

    @staticmethod
    def _client(*responses: AsyncMock) -> tuple[AsyncMock, list[dict[str, str]]]:
        """A mock client whose `get` snapshots the sent headers at call time.

        `call_args_list` records the headers dict by reference, so a strip that
        happened after the request went out would be invisible to it.
        """
        client = AsyncMock()
        sent_headers: list[dict[str, str]] = []
        responses_iter = iter(responses)

        async def get(url: str, **kwargs: Any) -> AsyncMock:
            # `follow_redirects=False` is load-bearing: `safe_download` must follow
            # redirects itself so that every hop is re-validated.
            assert kwargs['follow_redirects'] is False
            sent_headers.append(dict(kwargs['headers']))
            return next(responses_iter)

        client.get = get
        return client, sent_headers

    @staticmethod
    def _header(headers: dict[str, str], name: str) -> str | None:
        """Case-insensitive lookup in a snapshot of sent headers."""
        return next((v for k, v in headers.items() if k.lower() == name.lower()), None)

    @pytest.mark.parametrize(
        'start_url,location,kept',
        [
            # same origin
            ('https://example.com/file', 'https://example.com/elsewhere', True),
            # same origin with the default port spelled out
            ('https://example.com:443/file', 'https://example.com/elsewhere', True),
            # same origin via a relative Location, resolved before the comparison
            ('https://example.com/file', '/elsewhere', True),
            # http→https upgrade on the same host, matching httpx
            ('http://example.com/file', 'https://example.com/file', True),
            ('http://example.com:80/file', 'https://example.com:443/file', True),
            # Hostnames are case-insensitive.
            ('https://example.com/file', 'https://EXAMPLE.com/elsewhere', True),
            # `example.com.` (FQDN root label) is the same server as `example.com`
            ('https://example.com./file', 'https://example.com/file', True),
            # cross-host
            ('https://example.com/file', 'https://other.com/file', False),
            # protocol-relative Location to another host
            ('https://example.com/file', '//other.com/file', False),
            # same host, different port
            ('https://example.com/file', 'https://example.com:8443/file', False),
            # https→http downgrade on the same host
            ('https://example.com/file', 'http://example.com/file', False),
            # https→http downgrade with the default ports spelled out
            ('https://example.com:443/file', 'http://example.com:80/file', False),
            # http→https from a non-default port is not the upgrade exemption
            ('http://example.com:8080/file', 'https://example.com/file', False),
            # http→https landing on a non-default port is not the upgrade exemption
            ('http://example.com/file', 'https://example.com:8443/file', False),
            # http→https to a different host is not the upgrade exemption
            ('http://example.com/file', 'https://other.com/file', False),
        ],
    )
    async def test_sensitive_headers_across_redirect(
        self,
        mock_dns: AsyncMock,
        mock_ssrf_client: MagicMock,
        start_url: str,
        location: str,
        kept: bool,
    ) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        client, sent = self._client(self._redirect_response(location), self._final_response())
        mock_ssrf_client.return_value = client

        await safe_download(start_url, headers={**self._SENSITIVE_VALUES, 'Accept': 'text/html'})

        for name, value in self._SENSITIVE_VALUES.items():
            assert self._header(sent[1], name.lower()) == (value if kept else None)
        # Non-sensitive headers are always forwarded.
        assert self._header(sent[1], 'accept') == 'text/html'

    async def test_chained_redirect_keeps_headers_stripped(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """Once stripped on a cross-origin hop, headers stay stripped for the rest of the chain.

        The a.com→b.com→a.com return hop is same-host relative to the first URL but
        cross-origin relative to the previous hop; either way the strip is destructive,
        so the credential must not reappear on the third request.
        """
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        client, sent = self._client(
            self._redirect_response('https://b.com/file'),
            self._redirect_response('https://a.com/file'),
            self._final_response(),
        )
        mock_ssrf_client.return_value = client

        await safe_download('https://a.com/file', headers={'Authorization': 'Bearer SECRET'})

        assert self._header(sent[1], 'authorization') is None
        assert self._header(sent[2], 'authorization') is None

    async def test_invalid_redirect_protocol_rejected(self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock) -> None:
        """Unsupported redirect protocols fail before another request is sent."""
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        client, _ = self._client(self._redirect_response('ftp://example.com/file'))
        mock_ssrf_client.return_value = client

        with pytest.raises(ValueError, match='URL protocol "ftp" is not allowed'):
            await safe_download('https://example.com/file', headers={'Authorization': 'Bearer SECRET'})

    async def test_upgrade_then_downgrade_compares_previous_hop(
        self, mock_dns: AsyncMock, mock_ssrf_client: MagicMock
    ) -> None:
        """http→https→http on one host strips on the downgrade hop.

        This pins the comparison to the *previous* hop: measured against the first URL,
        the final hop would count as same-origin (both plain http on the same host) and
        the credential would leak back onto cleartext.
        """
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        client, sent = self._client(
            self._redirect_response('https://example.com/file'),
            self._redirect_response('http://example.com/file'),
            self._final_response(),
        )
        mock_ssrf_client.return_value = client

        await safe_download('http://example.com/file', headers={'Authorization': 'Bearer SECRET'})

        # http→https upgrade keeps the credential, https→http downgrade then strips it.
        assert self._header(sent[1], 'authorization') == 'Bearer SECRET'
        assert self._header(sent[2], 'authorization') is None


class TestDnsRebindingPrevention:
    """Tests specifically for DNS rebinding attack prevention."""

    async def test_hostname_resolving_to_private_ip_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('127.0.0.1', 0))]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://attacker.com/path', allow_local=False)

    async def test_hostname_resolving_to_cloud_metadata_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [(2, 1, 6, '', ('169.254.169.254', 0))]
        with pytest.raises(ValueError, match='Access to cloud metadata service'):
            await validate_and_resolve_url('http://attacker.com/path', allow_local=True)

    async def test_multiple_ips_with_any_private_blocked(self, mock_dns: AsyncMock) -> None:
        mock_dns.return_value = [
            (2, 1, 6, '', ('8.8.8.8', 0)),  # Public
            (10, 1, 6, '', ('::1', 0)),  # Private IPv6 loopback
        ]
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://attacker.com/path', allow_local=False)
