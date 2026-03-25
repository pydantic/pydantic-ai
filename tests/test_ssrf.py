"""Tests for SSRF (Server-Side Request Forgery) protection."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
            '203.0.113.50',
            '198.51.100.1',
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
            '169.254.169.254',  # AWS, GCP, Azure
            'fd00:ec2::254',  # AWS EC2 IPv6
            '100.100.100.200',  # Alibaba Cloud
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
            '100.100.100.199',  # Close but not Alibaba metadata
            '100.100.100.201',
        ],
    )
    def test_non_metadata_ips(self, ip: str) -> None:
        assert is_cloud_metadata_ip(ip) is False


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

    async def test_resolve_success(self) -> None:
        """Test that hostname resolution returns IP addresses."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [
                (2, 1, 6, '', ('93.184.215.14', 0)),
                (2, 1, 6, '', ('93.184.215.14', 0)),  # Duplicate should be removed
            ]
            ips = await resolve_hostname('example.com')
            assert ips == ['93.184.215.14']

    async def test_resolve_failure(self) -> None:
        """Test that DNS resolution failure raises ValueError."""
        import socket

        with patch('pydantic_ai._ssrf.run_in_executor', side_effect=socket.gaierror('DNS lookup failed')):
            with pytest.raises(ValueError, match='DNS resolution failed for hostname'):
                await resolve_hostname('nonexistent.invalid')


class TestValidateAndResolveUrl:
    """Tests for validate_and_resolve_url function."""

    async def test_public_ip_allowed(self) -> None:
        """Test that public IPs are allowed."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]
            resolved = await validate_and_resolve_url('https://example.com/path', allow_local=False)
            assert resolved.resolved_ip == '93.184.215.14'
            assert resolved.hostname == 'example.com'
            assert resolved.port == 443
            assert resolved.is_https is True
            assert resolved.path == '/path'

    async def test_private_ip_blocked_by_default(self) -> None:
        """Test that private IPs are blocked by default."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('192.168.1.1', 0))]
            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await validate_and_resolve_url('http://internal.local/path', allow_local=False)

    async def test_private_ip_allowed_with_allow_local(self) -> None:
        """Test that private IPs are allowed with allow_local=True."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('192.168.1.1', 0))]
            resolved = await validate_and_resolve_url('http://internal.local/path', allow_local=True)
            assert resolved.resolved_ip == '192.168.1.1'

    async def test_cloud_metadata_always_blocked(self) -> None:
        """Test that cloud metadata IPs are always blocked, even with allow_local=True."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('169.254.169.254', 0))]
            with pytest.raises(ValueError, match='Access to cloud metadata service'):
                await validate_and_resolve_url('http://metadata.google.internal/path', allow_local=True)

    async def test_alibaba_cloud_metadata_always_blocked(self) -> None:
        """Test that Alibaba Cloud metadata IP is always blocked, even with allow_local=True."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('100.100.100.200', 0))]
            with pytest.raises(ValueError, match='Access to cloud metadata service'):
                await validate_and_resolve_url('http://metadata.aliyun.internal/path', allow_local=True)

    async def test_literal_ip_address_in_url(self) -> None:
        """Test handling of literal IP addresses in URLs."""
        # Public IP - should work
        resolved = await validate_and_resolve_url('http://8.8.8.8/path', allow_local=False)
        assert resolved.resolved_ip == '8.8.8.8'
        assert resolved.hostname == '8.8.8.8'

    async def test_literal_private_ip_blocked(self) -> None:
        """Test that literal private IPs in URLs are blocked."""
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://192.168.1.1/path', allow_local=False)

    async def test_any_private_ip_blocks_request(self) -> None:
        """Test that if any resolved IP is private, the request is blocked."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            # Return both public and private IPs
            mock_executor.return_value = [
                (2, 1, 6, '', ('93.184.215.14', 0)),
                (2, 1, 6, '', ('192.168.1.1', 0)),
            ]
            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await validate_and_resolve_url('http://example.com/path', allow_local=False)

    async def test_6to4_address_blocked(self) -> None:
        """Test that 6to4 addresses (which can embed private IPv4) are blocked."""
        # 2002:c0a8:0101::1 embeds 192.168.1.1
        with pytest.raises(ValueError, match='Access to private/internal IP address'):
            await validate_and_resolve_url('http://[2002:c0a8:0101::1]/path', allow_local=False)

    async def test_cgnat_range_blocked(self) -> None:
        """Test that CGNAT range (100.64.0.0/10) is blocked."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            mock_executor.return_value = [(2, 1, 6, '', ('100.64.0.1', 0))]
            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await validate_and_resolve_url('http://cgnat-host.internal/path', allow_local=False)


class TestSafeDownload:
    """Tests for safe_download function."""

    async def test_successful_download(self) -> None:
        """Test successful download of a public URL."""
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None
        mock_response.content = b'test content'

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_fn.return_value = mock_client

            response = await safe_download('https://example.com/file.txt')
            assert response.content == b'test content'

            # Verify the request was made to the resolved IP with Host header and SNI
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert '93.184.215.14' in call_args[0][0]
            assert call_args[1]['headers']['Host'] == 'example.com'
            assert call_args[1]['extensions'] == {'sni_hostname': 'example.com'}

    async def test_redirect_followed(self) -> None:
        """Test that redirects are followed with validation."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://cdn.example.com/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            # First call for example.com, second for cdn.example.com
            mock_executor.side_effect = [
                [(2, 1, 6, '', ('93.184.215.14', 0))],
                [(2, 1, 6, '', ('203.0.113.50', 0))],
            ]

            mock_client = AsyncMock()
            mock_client.get.side_effect = [redirect_response, final_response]
            mock_client_fn.return_value = mock_client

            response = await safe_download('https://example.com/file.txt')
            assert response.content == b'final content'
            assert mock_client.get.call_count == 2

    async def test_redirect_to_private_ip_blocked(self) -> None:
        """Test that redirects to private IPs are blocked."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'http://internal.local/file.txt'}

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            # First call for example.com (public), second for internal.local (private)
            mock_executor.side_effect = [
                [(2, 1, 6, '', ('93.184.215.14', 0))],
                [(2, 1, 6, '', ('192.168.1.1', 0))],
            ]

            mock_client = AsyncMock()
            mock_client.get.return_value = redirect_response
            mock_client_fn.return_value = mock_client

            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await safe_download('https://example.com/file.txt')

    async def test_max_redirects_exceeded(self) -> None:
        """Test that too many redirects raises an error."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': 'https://example.com/redirect'}

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = redirect_response
            mock_client_fn.return_value = mock_client

            with pytest.raises(ValueError, match=f'Too many redirects \\({_MAX_REDIRECTS + 1}\\)'):
                await safe_download('https://example.com/file.txt')

    async def test_relative_redirect_resolved(self) -> None:
        """Test that relative redirect URLs are resolved correctly."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '/new-path/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.side_effect = [redirect_response, final_response]
            mock_client_fn.return_value = mock_client

            response = await safe_download('https://example.com/old-path/file.txt')
            assert response.content == b'final content'

            # Check that the second request was to the correct path
            second_call = mock_client.get.call_args_list[1]
            assert '/new-path/file.txt' in second_call[0][0]

    async def test_missing_location_header(self) -> None:
        """Test that redirect without Location header raises error."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {}

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = redirect_response
            mock_client_fn.return_value = mock_client

            with pytest.raises(ValueError, match='Redirect response missing Location header'):
                await safe_download('https://example.com/file.txt')

    async def test_protocol_relative_redirect(self) -> None:
        """Test that protocol-relative redirects are handled correctly."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '//cdn.example.com/file.txt'}

        final_response = AsyncMock()
        final_response.is_redirect = False
        final_response.raise_for_status = lambda: None
        final_response.content = b'final content'

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            # First call for example.com, second for cdn.example.com
            mock_executor.side_effect = [
                [(2, 1, 6, '', ('93.184.215.14', 0))],
                [(2, 1, 6, '', ('203.0.113.50', 0))],
            ]

            mock_client = AsyncMock()
            mock_client.get.side_effect = [redirect_response, final_response]
            mock_client_fn.return_value = mock_client

            response = await safe_download('https://example.com/file.txt')
            assert response.content == b'final content'
            assert mock_client.get.call_count == 2

            # Verify second request was to cdn.example.com with https
            second_call = mock_client.get.call_args_list[1]
            assert second_call[1]['headers']['Host'] == 'cdn.example.com'

    async def test_protocol_relative_redirect_to_private_blocked(self) -> None:
        """Test that protocol-relative redirects to private IPs are blocked."""
        redirect_response = AsyncMock()
        redirect_response.is_redirect = True
        redirect_response.headers = {'location': '//internal.local/file.txt'}

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.side_effect = [
                [(2, 1, 6, '', ('93.184.215.14', 0))],
                [(2, 1, 6, '', ('192.168.1.1', 0))],
            ]

            mock_client = AsyncMock()
            mock_client.get.return_value = redirect_response
            mock_client_fn.return_value = mock_client

            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await safe_download('https://example.com/file.txt')

    async def test_http_no_sni_extension(self) -> None:
        """Test that sni_hostname extension is not set for HTTP requests."""
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_fn.return_value = mock_client

            await safe_download('http://example.com/file.txt')

            call_args = mock_client.get.call_args
            assert call_args[1]['extensions'] == {}

    async def test_protocol_validation(self) -> None:
        """Test that non-http(s) protocols are rejected."""
        with pytest.raises(ValueError, match='URL protocol "file" is not allowed'):
            await safe_download('file:///etc/passwd')

        with pytest.raises(ValueError, match='URL protocol "ftp" is not allowed'):
            await safe_download('ftp://ftp.example.com/file.txt')

    async def test_timeout_parameter(self) -> None:
        """Test that timeout parameter is passed to client."""
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_fn.return_value = mock_client

            await safe_download('https://example.com/file.txt', timeout=60)

            mock_client_fn.assert_called_once_with(timeout=60)

    async def test_default_timeout(self) -> None:
        """Test that default timeout is used."""
        mock_response = AsyncMock()
        mock_response.is_redirect = False
        mock_response.raise_for_status = lambda: None

        with (
            patch('pydantic_ai._ssrf.run_in_executor') as mock_executor,
            patch('pydantic_ai._ssrf.cached_async_http_client') as mock_client_fn,
        ):
            mock_executor.return_value = [(2, 1, 6, '', ('93.184.215.14', 0))]

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client_fn.return_value = mock_client

            await safe_download('https://example.com/file.txt')

            mock_client_fn.assert_called_once_with(timeout=_DEFAULT_TIMEOUT)


class TestDnsRebindingPrevention:
    """Tests specifically for DNS rebinding attack prevention."""

    async def test_hostname_resolving_to_private_ip_blocked(self) -> None:
        """Test that a hostname resolving to a private IP is blocked."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            # Attacker's DNS returns private IP
            mock_executor.return_value = [(2, 1, 6, '', ('127.0.0.1', 0))]
            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await validate_and_resolve_url('http://attacker.com/path', allow_local=False)

    async def test_hostname_resolving_to_cloud_metadata_blocked(self) -> None:
        """Test that a hostname resolving to cloud metadata IP is blocked."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            # Attacker's DNS returns cloud metadata IP
            mock_executor.return_value = [(2, 1, 6, '', ('169.254.169.254', 0))]
            with pytest.raises(ValueError, match='Access to cloud metadata service'):
                await validate_and_resolve_url('http://attacker.com/path', allow_local=True)

    async def test_multiple_ips_with_any_private_blocked(self) -> None:
        """Test that if any IP in the resolution is private, request is blocked."""
        with patch('pydantic_ai._ssrf.run_in_executor') as mock_executor:
            # DNS returns multiple IPs, one of which is private
            mock_executor.return_value = [
                (2, 1, 6, '', ('8.8.8.8', 0)),  # Public
                (10, 1, 6, '', ('::1', 0)),  # Private IPv6 loopback
            ]
            with pytest.raises(ValueError, match='Access to private/internal IP address'):
                await validate_and_resolve_url('http://attacker.com/path', allow_local=False)
