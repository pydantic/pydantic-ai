"""Tests for SSRF (Server-Side Request Forgery) protection."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2 as httpx
import pytest

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
    itself (matching `httpx.AsyncClient` behavior), so tests work regardless of
    whether `safe_download` uses the client directly or via `async with`.
    """
    mock = MagicMock()

    def factory_wrapper(**kwargs: Any) -> Any:
        client = mock(**kwargs)
        client.__aenter__.return_value = client
        return client

    monkeypatch.setattr('pydantic_ai._ssrf._create_client', factory_wrapper)
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


class TestSafeDownload:
    """Tests for safe_download function."""

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
        `httpx.AsyncClient` (one client per call, no shared cache).

        Regression test for PR #4421 auto-review feedback.
        https://github.com/pydantic/pydantic-ai/pull/4421
        """
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_response.content = b'test content'

        mock_dns.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

        created_clients: list[httpx.AsyncClient] = []

        def tracking_create(**kwargs: Any) -> httpx.AsyncClient:
            client = httpx.AsyncClient()
            client.get = AsyncMock(return_value=mock_response)
            created_clients.append(client)
            return client

        monkeypatch.setattr('pydantic_ai._ssrf._create_client', tracking_create)

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
