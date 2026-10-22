"""Tests for the web fetch common tool."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from threading import Event, Lock
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import httpx2
import pytest
from markdownify import MarkdownConverter, markdownify

from pydantic_ai._utils import run_in_executor, using_thread_executor
from pydantic_ai.common_tools.web_fetch import (
    _HTML_CONVERSION_LIMITER,  # pyright: ignore[reportPrivateUsage]
    _MAX_HTML_CONVERSION_COST,  # pyright: ignore[reportPrivateUsage]
    _MAX_HTML_TEXT_SCAN_COST,  # pyright: ignore[reportPrivateUsage]
    WebFetchLocalTool,
    _convert_html,  # pyright: ignore[reportPrivateUsage]
    web_fetch_tool,
)
from pydantic_ai.exceptions import ModelRetry

READINESS_WAIT_TIMEOUT = 10


def _html_response(html: str, *, content_type: str = 'text/html; charset=utf-8') -> httpx2.Response:
    """Helper to create a mock HTML response."""
    return httpx2.Response(
        200,
        text=html,
        headers={'content-type': content_type},
        request=httpx2.Request('GET', 'https://example.com'),
    )


class TestWebFetchLocalTool:
    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_html_conversion_error_releases_slot(self, anyio_backend: str):
        """A conversion that raises frees its slot in the HTML conversion limiter."""
        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response('<p>x</p>'),
            ),
            patch('pydantic_ai.common_tools.web_fetch._convert_html', side_effect=ModelRetry('too complex')),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            with pytest.raises(ModelRetry, match=r'Failed to convert .*too complex'):
                await tool('https://example.com')
        assert _HTML_CONVERSION_LIMITER.get().value == 4

    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_html_conversion_default_cancellation_waits_for_completion(self, anyio_backend: str):
        """A cancelled fetch keeps its slot until the worker thread finishes converting."""
        started = anyio.Event()
        release = Event()
        finished = anyio.Event()
        scopes: list[anyio.CancelScope] = []

        def convert(html: str) -> tuple[str, str]:
            anyio.from_thread.run_sync(started.set)
            release.wait()
            return '', html

        async def fetch(tool: WebFetchLocalTool):
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await tool('https://example.com')
            finished.set()

        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response('<p>x</p>'),
            ),
            patch('pydantic_ai.common_tools.web_fetch._convert_html', side_effect=convert),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            async with anyio.create_task_group() as group:
                try:
                    group.start_soon(fetch, tool)
                    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                        await started.wait()
                    scopes[0].cancel()
                    await anyio.wait_all_tasks_blocked()
                    assert not finished.is_set()
                    assert _HTML_CONVERSION_LIMITER.get().value == 3
                finally:
                    release.set()
        assert finished.is_set()
        assert _HTML_CONVERSION_LIMITER.get().value == 4

    async def test_html_conversion_keeps_default_workers_available(self):
        """Concurrent HTML conversions are capped, so unrelated worker-thread calls still run."""
        loop = asyncio.get_running_loop()
        # With five worker tokens, eight uncapped conversions would leave none for the unrelated call.
        thread_limiter = anyio.to_thread.current_default_thread_limiter()
        total_tokens = thread_limiter.total_tokens
        thread_limiter.total_tokens = 5
        started = asyncio.Event()
        release = Event()
        lock = Lock()
        running = 0

        def convert(html: str) -> tuple[str, str]:
            nonlocal running
            with lock:
                running += 1
                if running == 4:
                    loop.call_soon_threadsafe(started.set)
            release.wait()
            return '', html

        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response('<p>x</p>'),
            ),
            patch('pydantic_ai.common_tools.web_fetch._convert_html', side_effect=convert),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            tasks = [asyncio.create_task(tool('https://example.com')) for _ in range(8)]
            try:
                await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
                assert running == 4
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    assert await run_in_executor(lambda: 'available') == 'available'
            finally:
                release.set()
                await asyncio.gather(*tasks)
                thread_limiter.total_tokens = total_tokens

    @pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
    async def test_fetch_html(self, anyio_backend: str):
        """Fetches HTML and converts to markdown."""
        html = '<html><head><title>Test Page</title></head><body><h1>Hello</h1><p>World</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['url'] == 'https://example.com'
        assert result['title'] == 'Test Page'
        assert 'Hello' in result['content']
        assert 'World' in result['content']

    async def test_fetch_html_title_with_whitespace(self):
        """Title whitespace is stripped."""
        html = '<html><head><title>  Hello  </title></head><body><p>Content</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'Hello'

    async def test_fetch_html_no_title(self):
        """HTML without title returns empty string."""
        html = '<html><head></head><body><p>Content</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == ''
        assert 'Content' in result['content']

    async def test_fetch_html_empty_title(self):
        """Empty title tag returns empty string."""
        html = '<html><head><title></title></head><body><p>Content</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == ''

    async def test_fetch_html_collapses_excessive_newlines(self):
        """Excessive newlines in converted content are collapsed."""
        html = '<html><body><p>A</p><br><br><br><br><p>B</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert 'A' in result['content']
        assert 'B' in result['content']
        assert '\n\n\n' not in result['content']

    async def test_fetch_json(self):
        """Fetches JSON and returns formatted."""
        mock_response = httpx2.Response(
            200,
            text='{"key": "value"}',
            headers={'content-type': 'application/json'},
            request=httpx2.Request('GET', 'https://api.example.com/data'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://api.example.com/data')

        assert isinstance(result, dict)
        assert result['title'] == ''
        assert '```json' in result['content']
        assert '"key": "value"' in result['content']

    async def test_fetch_invalid_json(self):
        """Invalid JSON is returned as-is."""
        mock_response = httpx2.Response(
            200,
            text='{invalid json',
            headers={'content-type': 'application/json'},
            request=httpx2.Request('GET', 'https://api.example.com/data'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://api.example.com/data')

        assert isinstance(result, dict)
        assert result['content'] == '{invalid json'

    async def test_fetch_plain_text(self):
        """Fetches plain text and returns as-is."""
        mock_response = httpx2.Response(
            200,
            text='Hello, plain text!',
            headers={'content-type': 'text/plain'},
            request=httpx2.Request('GET', 'https://example.com/file.txt'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com/file.txt')

        assert isinstance(result, dict)
        assert result['content'] == 'Hello, plain text!'

    async def test_fetch_no_content_type(self):
        """Missing content-type is treated as HTML."""
        html = '<html><head><title>No CT</title></head><body><p>Test</p></body></html>'
        mock_response = httpx2.Response(
            200,
            content=html.encode(),
            headers={},
            request=httpx2.Request('GET', 'https://example.com'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'No CT'
        assert 'Test' in result['content']

    async def test_content_truncation(self):
        """Content exceeding max_content_length is truncated."""
        html = '<html><body><p>' + 'x' * 200 + '</p></body></html>'
        mock_response = _html_response(html)

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=50, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['content'].endswith('[Content truncated]')

    async def test_no_truncation_when_none(self):
        """No truncation when max_content_length is None."""
        long_text = 'x' * 100_000
        mock_response = httpx2.Response(
            200,
            text=long_text,
            headers={'content-type': 'text/plain'},
            request=httpx2.Request('GET', 'https://example.com'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert len(result['content']) == 100_000

    async def test_fetch_xml(self):
        """XML content types are treated as text."""
        xml = '<?xml version="1.0"?><root><item>Hello</item></root>'
        mock_response = httpx2.Response(
            200,
            text=xml,
            headers={'content-type': 'application/xml'},
            request=httpx2.Request('GET', 'https://example.com/feed.xml'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com/feed.xml')

        assert isinstance(result, dict)
        assert '<root>' in result['content']
        assert 'Hello' in result['content']

    async def test_fetch_xhtml(self):
        """XHTML content is converted to markdown like HTML."""
        xhtml = '<html><head><title>XHTML Page</title></head><body><h1>Hello</h1><p>World</p></body></html>'
        mock_response = httpx2.Response(
            200,
            text=xhtml,
            headers={'content-type': 'application/xhtml+xml'},
            request=httpx2.Request('GET', 'https://example.com'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'XHTML Page'
        assert 'Hello' in result['content']
        assert '<h1>' not in result['content']

    async def test_binary_content_type(self):
        """Binary content types return BinaryContent."""
        from pydantic_ai.messages import BinaryContent

        pdf_bytes = b'%PDF-1.4 fake content'
        mock_response = httpx2.Response(
            200,
            content=pdf_bytes,
            headers={'content-type': 'application/pdf'},
            request=httpx2.Request('GET', 'https://example.com/doc.pdf'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com/doc.pdf')

        assert isinstance(result, BinaryContent)
        assert result.data == pdf_bytes
        assert result.media_type == 'application/pdf'

    async def test_passes_allow_local(self):
        """allow_local_urls is passed to safe_download."""
        html = '<html><body>ok</body></html>'
        mock_response = httpx2.Response(
            200,
            text=html,
            headers={'content-type': 'text/html'},
            request=httpx2.Request('GET', 'http://localhost:8080'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=True, timeout=60)
            await tool('http://localhost:8080')

        mock_dl.assert_called_once_with(
            'http://localhost:8080',
            allow_local=True,
            timeout=60,
            headers={'Accept': 'text/markdown, text/html;q=0.9, */*;q=0.8'},
            allowed_domains=None,
            blocked_domains=None,
            max_bytes=50 * 1024 * 1024,
        )

    async def test_safe_download_error_raises_model_retry(self):
        """Errors from safe_download are converted to ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            side_effect=ValueError('DNS resolution failed'),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            with pytest.raises(ModelRetry, match='Failed to fetch'):
                await tool('https://nonexistent.invalid')

    async def test_http_error_raises_model_retry(self):
        """HTTP errors are converted to ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        request = httpx2.Request('GET', 'https://example.com')
        response = httpx2.Response(404, request=request)
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            side_effect=httpx2.HTTPStatusError('Not Found', request=request, response=response),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            with pytest.raises(ModelRetry, match='Failed to fetch'):
                await tool('https://example.com/missing')

    async def test_invalid_url_raises_model_retry(self):
        """URL without valid protocol raises ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
        with pytest.raises(ModelRetry, match='Failed to fetch'):
            await tool('not-a-url')

    async def test_idna_invalid_hostname_raises_model_retry(self):
        """A hostname httpx2's IDNA parser rejects raises ModelRetry, not `httpx2.InvalidURL`.

        The URL is rejected before any name resolution, so this never reaches the network.
        """
        # A fullwidth "e" (U+FF45), spelled as an escape so the hostname survives any normalization
        # a tool might apply to this file, and so a reader can tell it apart from a plain "e".
        url = 'https://\uff45xample.com/'

        tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
        with pytest.raises(ModelRetry, match='Failed to fetch') as exc_info:
            await tool(url)

        assert isinstance(exc_info.value.__cause__, httpx2.InvalidURL)

    async def test_allowed_domains_permits(self):
        """Allowed domain passes validation and is forwarded to safe_download."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, allowed_domains=['example.com']
            )
            result = await tool('https://example.com/page')

        assert isinstance(result, dict)
        assert result['url'] == 'https://example.com/page'
        assert mock_dl.call_args[1]['allowed_domains'] == ['example.com']

    async def test_allowed_domains_blocks(self):
        """Non-allowed domain raises ModelRetry (domain check enforced by safe_download)."""
        from pydantic_ai.exceptions import ModelRetry

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            side_effect=ValueError("Domain 'evil.com' is not in the allowed domains list."),
        ):
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, allowed_domains=['example.com']
            )
            with pytest.raises(ModelRetry, match='Failed to fetch'):
                await tool('https://evil.com/page')

    async def test_blocked_domains_blocks(self):
        """Blocked domain raises ModelRetry (domain check enforced by safe_download)."""
        from pydantic_ai.exceptions import ModelRetry

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            side_effect=ValueError("Domain 'evil.com' is blocked."),
        ):
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, blocked_domains=['evil.com']
            )
            with pytest.raises(ModelRetry, match='Failed to fetch'):
                await tool('https://evil.com/page')

    async def test_blocked_domains_permits(self):
        """Non-blocked domain passes validation and is forwarded to safe_download."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, blocked_domains=['evil.com']
            )
            result = await tool('https://example.com/page')

        assert isinstance(result, dict)
        assert result['url'] == 'https://example.com/page'
        assert mock_dl.call_args[1]['blocked_domains'] == ['evil.com']

    async def test_fetch_markdown_response(self):
        """Server returning text/markdown is used as-is without markdownify conversion."""
        markdown_content = '# Hello\n\nThis is **markdown** from the server.'
        mock_response = httpx2.Response(
            200,
            text=markdown_content,
            headers={'content-type': 'text/markdown; charset=utf-8'},
            request=httpx2.Request('GET', 'https://example.com/page'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com/page')

        assert isinstance(result, dict)
        assert result['content'] == markdown_content
        assert result['title'] == ''

    async def test_fetch_x_markdown_response(self):
        """Server returning text/x-markdown is used as-is."""
        markdown_content = '## Test'
        mock_response = httpx2.Response(
            200,
            text=markdown_content,
            headers={'content-type': 'text/x-markdown'},
            request=httpx2.Request('GET', 'https://example.com'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['content'] == '## Test'

    async def test_default_accept_header(self):
        """Default Accept header requests text/markdown."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            await tool('https://example.com')

        call_headers = mock_dl.call_args[1]['headers']
        assert 'text/markdown' in call_headers['Accept']

    async def test_custom_headers(self):
        """Custom headers are passed through to safe_download."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(
                max_content_length=None,
                allow_local_urls=False,
                timeout=30,
                headers={'Authorization': 'Bearer token123'},
            )
            await tool('https://example.com')

        call_headers = mock_dl.call_args[1]['headers']
        assert call_headers['Authorization'] == 'Bearer token123'
        assert 'text/markdown' in call_headers['Accept']

    async def test_custom_accept_header_overrides_default(self):
        """User-provided Accept header overrides the default."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(
                max_content_length=None,
                allow_local_urls=False,
                timeout=30,
                headers={'Accept': 'text/html'},
            )
            await tool('https://example.com')

        call_headers = mock_dl.call_args[1]['headers']
        assert call_headers['Accept'] == 'text/html'

    @pytest.fixture
    def serve_response(self, monkeypatch: pytest.MonkeyPatch) -> Callable[[httpx2.Response], None]:
        """Serves a canned response through the real `safe_download` so its download bound applies.

        The tests using it request an IP-literal URL, so no DNS resolution is involved.
        """

        def serve(response: httpx2.Response) -> None:
            client = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: response))

            def create_http_client(*, timeout: int) -> httpx2.AsyncClient:
                return client

            monkeypatch.setattr('pydantic_ai._ssrf.create_async_httpx2_client', create_http_client)

        return serve

    @pytest.mark.parametrize('content_type', ['text/plain', 'application/pdf'])
    async def test_download_over_max_download_bytes_raises_model_retry(
        self, serve_response: Callable[[httpx2.Response], None], content_type: str
    ):
        """A response body larger than `max_download_bytes` is rejected before it is buffered."""
        request = httpx2.Request('GET', 'https://93.184.215.14/doc')
        serve_response(
            httpx2.Response(200, content=b'x' * 2000, headers={'content-type': content_type}, request=request)
        )

        tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30, max_download_bytes=1024)
        with pytest.raises(ModelRetry, match='maximum size of 1024 bytes'):
            await tool('https://93.184.215.14/doc')

    async def test_no_download_limit_when_none(self, serve_response: Callable[[httpx2.Response], None]):
        """`max_download_bytes=None` keeps reading the whole body, however large."""
        request = httpx2.Request('GET', 'https://93.184.215.14/big.txt')
        serve_response(
            httpx2.Response(200, text='x' * 200_000, headers={'content-type': 'text/plain'}, request=request)
        )

        tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30, max_download_bytes=None)
        result = await tool('https://93.184.215.14/big.txt')

        assert isinstance(result, dict)
        assert len(result['content']) == 200_000

    async def test_fetch_html_title_is_raw_and_case_insensitive(self):
        """The title is the raw text between the tags, matched case-insensitively, with attributes ignored."""
        html = '<html><head><TITLE lang="en">Fish &amp; Chips</TITLE></head><body><p>Content</p></body></html>'

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'Fish &amp; Chips'

    async def test_fetch_html_title_after_case_expanding_character(self):
        """Characters whose lowercase form is longer (`İ` becomes two code points) don't shift the title's offsets."""
        html = '<html><head><meta name="x" content="İ"><title>İstanbul</title></head><body>İ</body></html>'

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'İstanbul'

    async def test_html_decoding_and_conversion_run_in_worker_thread(self):
        """Decoding the body and converting the HTML run through the sync-function executor, not on the event loop.

        Both costs scale with the server-controlled body, and the charset the server picks can make
        decoding far worse than linear, so neither may stall every other coroutine in the process.
        `using_thread_executor` makes the offload observable: the decode and the conversion are the
        only sync work the tool submits.
        """

        class RecordingExecutor(ThreadPoolExecutor):
            def __init__(self):
                super().__init__()
                self.submitted: list[Future[Any]] = []

            def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
                future = super().submit(fn, *args, **kwargs)
                self.submitted.append(future)
                return future

        html = '<html><head><title>Threaded</title></head><body><p>Content</p></body></html>'
        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response(html),
            ),
            RecordingExecutor() as executor,
            using_thread_executor(executor),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == 'Threaded'
        assert [future.result() for future in executor.submitted] == [html, ('Threaded', 'Threaded\n\nContent')]

    async def test_fetch_html_repeated_unclosed_title_tags(self):
        """A body made of `<title` fragments with no closing `>` converts in seconds, not minutes.

        Each fragment is a candidate title start with no end in reach, which previously made title
        extraction quadratic in the body size: a body of this size took minutes, during which the
        event loop was blocked. The bound is generous; the point is that it isn't minutes.
        """
        html = '<title' * 300_000

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            start = time.perf_counter()
            result = await tool('https://example.com')
            elapsed = time.perf_counter() - start

        assert isinstance(result, dict)
        assert result['title'] == ''
        assert result['content'] == ''
        assert elapsed < 60

    @pytest.mark.parametrize('html', ['<title>never closed', '<title never opened'])
    async def test_fetch_html_unterminated_title_is_empty(self, html: str):
        """A `<title>` that is never closed, or never even opened, yields no title."""
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['title'] == ''

    async def test_fetch_html_nested_too_deeply_raises_model_retry(self):
        """A page nested deeper than the recursion limit can't be converted, so the model is told to move on."""
        html = '<div>' * 2000 + 'Content' + '</div>' * 2000

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            with pytest.raises(ModelRetry, match='nested too deeply'):
                await tool('https://example.com')

    async def test_fetch_html_conversion_budget_returns_model_retry(self):
        """The conversion budget surfaces a URL-qualified retry to the agent."""
        html = '<dd>' * 15 + 'x\n' * 300_000 + '</dd>' * 15
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response(html),
        ):
            tool = WebFetchLocalTool(max_content_length=50_000, allow_local_urls=False, timeout=30)
            with pytest.raises(
                ModelRetry, match=r'Failed to convert https://example\.com: the document is too complex'
            ):
                await tool('https://example.com')

    async def test_nested_html_is_rejected_before_conversion(self):
        """A deeply nested definition list is rejected by the up-front estimate, before `markdownify` walks it.

        Converting this page means rescanning and re-indenting its text at each of the 120 levels, which
        took over 30 seconds of worker time before the estimate existed, so the estimate has to reject the
        page before that walk starts rather than midway through it.
        """
        html = '<dd>' * 120 + 'line\n' * 150_000 + '</dd>' * 120
        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response(html),
            ),
            patch.object(
                MarkdownConverter, 'convert_soup', autospec=True, side_effect=MarkdownConverter.convert_soup
            ) as convert_soup,
        ):
            tool = WebFetchLocalTool(max_content_length=50_000, allow_local_urls=False, timeout=30)
            with pytest.raises(
                ModelRetry, match=r'Failed to convert https://example\.com: the document is too complex'
            ):
                await tool('https://example.com')

        convert_soup.assert_not_called()

    async def test_fetch_html_shallow_markup_budget_returns_model_retry(self):
        """A page over the markup budget asks the model to try something else."""
        html = '<ol>' + '<li>x</li>' * 100 + '</ol>'
        with (
            patch(
                'pydantic_ai.common_tools.web_fetch.safe_download',
                new_callable=AsyncMock,
                return_value=_html_response(html),
            ),
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20),
        ):
            tool = WebFetchLocalTool(max_content_length=50_000, allow_local_urls=False, timeout=30)
            with pytest.raises(
                ModelRetry, match=r'Failed to convert https://example\.com: the document is too complex'
            ):
                await tool('https://example.com')

    @pytest.mark.parametrize('charset', ['idna', 'rot_13', 'base64_codec'])
    async def test_undecodable_charset_raises_model_retry(self, charset: str):
        """A charset the server picks that can't decode a document is reported as a failed fetch.

        `idna` is a registered codec that rejects the replacement error handler; `rot_13` and
        `base64_codec` are registered codecs that aren't text encodings at all. An unknown label,
        by contrast, falls back to UTF-8 and never gets here.
        """
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response('<p>Content</p>', content_type=f'text/html; charset={charset}'),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            with pytest.raises(ModelRetry, match='Failed to decode'):
                await tool('https://example.com')

    async def test_declared_charset_is_honored(self):
        """The body is decoded with the charset the server declares, with undecodable bytes replaced."""
        response = httpx2.Response(
            200,
            headers={'content-type': 'text/plain; charset=latin-1'},
            content='caf\xe9'.encode('latin-1'),
        )
        with patch('pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=response):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['content'] == 'caf\xe9'

    async def test_fetch_json_nested_too_deeply_returns_raw_text(self, monkeypatch: pytest.MonkeyPatch):
        """A JSON document nested deeper than the recursion limit is returned as-is, like one that doesn't parse.

        The depth at which `json.loads` gives up differs between interpreters, and past it some
        overflow the stack instead of raising, so the parser is stood in for rather than fed a
        real document.
        """

        def loads(text: str) -> Any:
            raise RecursionError('maximum recursion depth exceeded')

        monkeypatch.setattr(json, 'loads', loads)
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download',
            new_callable=AsyncMock,
            return_value=_html_response('[[[[]]]]', content_type='application/json'),
        ):
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=False, timeout=30)
            result = await tool('https://example.com')

        assert isinstance(result, dict)
        assert result['content'] == '[[[[]]]]'


_CONVERTER_PARITY_CASES = [
    pytest.param(
        '<h1>Title</h1>\n<p>Some   text\twith  \n\n  mixed \r\n whitespace &amp; <b>bold</b> <code> x  y </code></p>',
        id='whitespace',
    ),
    pytest.param(
        '<ol start="3"><li>three</li><li>four\nsecond line</li><li></li><li><p>five</p><ul><li>a</li><li>b</li></ul></li></ol>'
        '<ul><li>one</li><li><ol><li>nested</li><li>again</li></ol></li></ul><ol>\n  <li>a</li>\n  <li>b</li>\n</ol>',
        id='lists',
    ),
    pytest.param(
        '<pre>\n\n  code\n    more\n\n</pre><pre>   \n x \n   </pre><pre>x  </pre><pre>  x</pre><pre>\n</pre><pre></pre>'
        '<pre><code class="language-py">print( 1 )\n\n</code></pre>',
        id='pre',
    ),
    pytest.param(
        '<div><p>a</p>   <p> b </p></div><table><tr><th>h</th></tr><tr><td> c  d </td></tr></table>'
        '<blockquote>\n q\n</blockquote><a href="/x">  link  </a><!-- comment  with   spaces -->',
        id='blocks',
    ),
    pytest.param(
        '<blockquote><blockquote>quote\nsecond</blockquote></blockquote>'
        '<dl><dd><dd>definition\nnext</dd></dd></dl>'
        '<ul><li><ul><li>one<br>two</li></ul></li></ul>',
        id='nested-indentation',
    ),
    pytest.param(
        '<p>a<![CDATA[ x   y \n z ]]>b<?php  echo  1 ?>c</p>',
        id='cdata-and-pi',
    ),
]

# Budget tests that would otherwise need tens of megabytes of HTML to cross the real budgets
# shrink the budget they target by this factor, and their input with it. Every charge is linear in
# the input's length, so the same shape crosses (or stays under) the scaled budget by the same margin.
_BUDGET_SCALE = 100


@pytest.fixture
def scaled_conversion_budget():
    with patch(
        'pydantic_ai.common_tools.web_fetch._MAX_HTML_CONVERSION_COST', _MAX_HTML_CONVERSION_COST // _BUDGET_SCALE
    ):
        yield


@pytest.fixture
def scaled_text_scan_budget():
    with patch(
        'pydantic_ai.common_tools.web_fetch._MAX_HTML_TEXT_SCAN_COST', _MAX_HTML_TEXT_SCAN_COST // _BUDGET_SCALE
    ):
        yield


class TestMarkdownConverter:
    def test_large_well_formed_reference_list_passes_markup_budgets(self):
        """A page with more markup than a 2.9 MB Wikipedia article (~101k parser events) reaches parsing."""

        class Parsed(Exception):
            pass

        html = '<ul>' + '<li class="r"><a href="/x">Ref</a> &ndash; note</li>' * 17_000 + '</ul>'
        with patch('pydantic_ai.common_tools.web_fetch.BeautifulSoup', side_effect=Parsed):
            with pytest.raises(Parsed):
                _convert_html(html)

    @pytest.mark.parametrize(
        'html',
        [
            pytest.param('<p>' + '< ' * 21 + '</p>', id='less-than'),
            pytest.param('</>' * 21, id='empty-end-tag'),
            pytest.param('<p>' + '& ' * 21 + '</p>', id='ampersand'),
        ],
    )
    def test_markup_delimiters_are_counted_before_parsing(self, html: str):
        """`<` and `&` that produce no parser event still count, before any parser runs."""
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_DELIMITERS', 20),
            patch('pydantic_ai.common_tools.web_fetch._HTMLMarkupBudget', side_effect=AssertionError('parsed')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.parametrize('tag', ['script', 'style'])
    def test_raw_text_markup_does_not_spend_event_budget(self, tag: str):
        """Tag-like text inside `<script>` or `<style>` does not spend the event budget."""
        html = f'<{tag} data-kind="text">' + '<div></div>' * 21 + f'</{tag}><p>ok</p>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            assert 'ok' in _convert_html(html)[1]

    def test_self_closing_raw_text_tag_does_not_hide_markup(self):
        """A self-closing `<script/>` does not start a raw-text section that hides later markup."""
        html = '<script/>' + '<div></div>' * 21
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_raw_text_string_in_attribute_does_not_hide_markup(self):
        """A `<script>` string inside an attribute value does not hide later markup."""
        html = '<div data-text="<script>">' + '<li>x</li>' * 21 + '</div></script>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_unquoted_attribute_quote_does_not_hide_markup(self):
        """A quote inside an unquoted attribute value does not hide later markup."""
        html = "<div data=x' >" + '<li>x</li>' * 21 + "</div><span data=y' >"
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_markup_string_in_comment_does_not_spend_event_budget(self):
        """Tag-like text inside a comment does not spend the event budget."""
        html = '<!--' + '<div></div>' * 21 + '--><p>ok</p>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            assert 'ok' in _convert_html(html)[1]

    @pytest.mark.parametrize(
        ('html', 'whole_tag_match'),
        [
            pytest.param('<div' + ' ' * 100_001 + '>ok</div>', 'check_for_whole_start_tag', id='start-tag'),
            pytest.param('<div a=1' + ' \n' * 50_001 + '>ok</div>', 'check_for_whole_start_tag', id='after-attribute'),
            pytest.param(
                '<div' + ' a=1' * 1_000 + ' ' * 99_100 + '>ok</div>',
                'check_for_whole_start_tag',
                id='across-attributes',
            ),
            pytest.param('</div' + '/' * 100_001 + '>', 'parse_endtag', id='end-tag'),
        ],
    )
    def test_tag_separators_are_counted_before_matching(self, html: str, whole_tag_match: str):
        """More than 100,000 whitespace or `/` characters inside one tag are rejected before the parser matches it."""
        with patch(f'html.parser.HTMLParser.{whole_tag_match}', side_effect=AssertionError('whole-tag match')):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_separator_run_at_the_limit_converts(self):
        """A tag with 100,000 whitespace characters inside it is accepted."""
        assert _convert_html('<p' + ' ' * 100_000 + '>ok</p>')[1] == 'ok'

    @pytest.mark.parametrize('html', ['a</>b', 'ok</p'])
    def test_end_tag_check_skips_what_the_parser_does_not_match(self, html: str):
        """`</>` and an end tag with no closing `>` are not matched as tags, so they are not walked."""
        with patch(
            'pydantic_ai.common_tools.web_fetch._HTMLMarkupBudget._check_tag', side_effect=AssertionError('walked')
        ):
            _convert_html(html)

    def test_end_tag_attributes_are_counted(self):
        """Attribute-like text in an end tag is counted before the parser matches the tag."""
        html = '</div' + ' x=1' * 21 + '>'
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_TAG_ATTRIBUTES', 20),
            patch('html.parser.HTMLParser.parse_endtag', side_effect=AssertionError('whole-tag match')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_end_tag_attributes_spend_the_event_budget(self):
        """Attribute-like text in end tags counts toward the event budget."""
        html = '</div' + ' x' * 21 + '>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.parametrize('prefix', ['<div ', '<div data-x=">" ', "<div data=x' "])
    def test_many_attributes_are_rejected_before_whole_tag_scan(self, prefix: str):
        """A tag's attributes are counted before the parser matches the whole tag."""
        html = prefix + ' x=1' * 21 + '>ok</div>'
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_TAG_ATTRIBUTES', 20),
            patch('html.parser.HTMLParser.check_for_whole_start_tag', side_effect=AssertionError('whole-tag scan')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.parametrize('separator', ['', '/'])
    def test_adjacent_attributes_are_counted(self, separator: str):
        """Attributes with no whitespace between them are still counted."""
        html = '<div a0="x"' + (separator + 'a1="x"') * 21 + '>ok</div>'
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_TAG_ATTRIBUTES', 20),
            patch('html.parser.HTMLParser.check_for_whole_start_tag', side_effect=AssertionError('whole-tag scan')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_slashes_in_unquoted_attribute_value_are_not_attributes(self):
        """Slashes inside an unquoted attribute value do not count as attribute separators."""
        html = '<a href=https://example.com/' + 'path/' * 21 + '>ok</a>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_TAG_ATTRIBUTES', 20):
            assert 'ok' in _convert_html(html)[1]

    @pytest.mark.parametrize('entity', ['&amp;', '&#65;'])
    def test_many_entities_are_rejected_before_parsing(self, entity: str):
        """Entity and character references are counted before the document is parsed."""
        html = '<p>' + entity * 21 + '</p>'
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20),
            patch('pydantic_ai.common_tools.web_fetch.BeautifulSoup', side_effect=AssertionError('parsed')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_attributes_share_markup_budget(self):
        """Each attribute counts toward the same budget as its tag."""
        html = '<div a=1 b=2 c=3><span d=4>ok</span></div>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 5):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_whitespace_closed_comment_does_not_hide_markup(self):
        """Markup after a comment ending in `-- >` is either counted toward the budget or not rendered."""
        html = '<!--comment-- >' + '<li>x</li>' * 21
        # Python versions disagree on whether '-- >' closes the comment. The page must
        # either be rejected for complexity or produce no visible content.
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20), suppress(ModelRetry):
            assert not _convert_html(html)[1]

    @pytest.mark.parametrize(
        ('prefix', 'suffix'),
        [
            ('<!--comment--!>', ''),
            ('<?pi a="', '"?>'),
            ('<!DOCTYPE html x="', '" >'),
            ('<![CDATA[x="]]>', '" ]]>'),
        ],
        ids=['malformed-comment', 'processing-instruction', 'doctype', 'cdata'],
    )
    def test_markup_after_special_section_is_counted(self, prefix: str, suffix: str):
        """Markup after a comment, processing instruction, doctype or CDATA section is still counted."""
        html = prefix + '<li>x</li>' * 21 + suffix
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    def test_cdata_markup_does_not_spend_event_budget(self):
        """Tag-like text inside a CDATA section does not spend the event budget."""
        html = '<![CDATA[' + '<div></div>' * 21 + ']]><p>ok</p>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            assert 'ok' in _convert_html(html)[1]

    def test_text_less_than_signs_do_not_spend_event_budget(self):
        """A `<` in plain text does not spend the event budget."""
        html = '<pre>' + 'if (a < b) x++;\n' * 21 + '</pre>'
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20):
            assert _convert_html(html)[1]

    @pytest.mark.parametrize(
        'html',
        [
            pytest.param('<ol>' + '<li>x</li>' * 100 + '</ol>', id='list'),
            pytest.param('<dl>' + '<dd>x</dd>' * 100 + '</dl>', id='definition-list'),
            pytest.param('<table><tr>' + '<td>x</td>' * 100 + '</tr></table>', id='table-cells'),
            pytest.param('<div>' + '<p>x</p>' * 100 + '</div>', id='paragraphs'),
            pytest.param('<div>' + '<a href="/x">x</a>' * 100 + '</div>', id='links'),
            pytest.param('<div>' + '<code>x</code>' * 100 + '</div>', id='code'),
            pytest.param('<div>' + '<h2>x</h2>' * 100 + '</div>', id='headings'),
        ],
    )
    def test_shallow_markup_is_rejected_before_conversion(self, html: str):
        """The markup budget is enforced before conversion."""
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20),
            patch.object(MarkdownConverter, 'convert_soup', autospec=True) as convert_soup,
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)
        convert_soup.assert_not_called()

    def test_markup_count_is_checked_before_parsing(self):
        """The parser itself must not spend unbounded time constructing a large tree."""
        html = '<ol>' + '<li>x</li>' * 100 + '</ol>'
        with (
            patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_MARKUP_EVENTS', 20),
            patch('pydantic_ai.common_tools.web_fetch.BeautifulSoup', side_effect=AssertionError('parsed')),
        ):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.parametrize('html', _CONVERTER_PARITY_CASES)
    def test_matches_upstream(self, html: str):
        """The linear-time replacements produce exactly what `markdownify`'s own steps produce."""
        _, content = _convert_html(html)
        assert content == markdownify(html, strip=['img', 'script', 'style'])

    def test_non_decimal_list_start_is_ignored(self):
        """A `start` made of digits `int()` rejects, like `²`, numbers the list from 1 instead of raising.

        `markdownify` checks `isnumeric()` and then calls `int()`, which raises on such digits.
        """
        _, content = _convert_html('<ol start="²"><li>one</li><li>two</li></ol>')
        assert content == '1. one\n2. two'

    @pytest.mark.parametrize('tag', ['blockquote', 'dd', 'li'])
    def test_deeply_nested_indentation_is_bounded(self, tag: str):
        """The converter rejects repeated indentation before intermediate Markdown expands."""
        html = f'<{tag}>' * 120 + 'line\n' * 150_000 + f'</{tag}>' * 120
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    def test_deeply_nested_empty_tags_are_bounded(self):
        """Generated line breaks must count towards work even without descendant text."""
        html = '<q>' * 120 + '<br>' * 30_000 + '</q>' * 120
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    def test_shallow_nested_indentation_is_bounded(self):
        """Indented lines also count when the document is fewer than 16 levels deep."""
        html = '<dd>' * 15 + 'x\n' * 300_000 + '</dd>' * 15
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    @pytest.mark.parametrize(
        ('tag', 'depth', 'prefix'),
        [
            pytest.param('dd', 15, ':   ' * 15, id='definition-items'),
            pytest.param('div', 30, '', id='ordinary-tags'),
        ],
    )
    def test_large_single_line_text_is_not_overcharged(self, tag: str, depth: int, prefix: str):
        """Nested text that converts quickly without output growth must remain available."""
        text = 'x' * 1_500_000
        html = f'<{tag}>' * depth + text + f'</{tag}>' * depth
        _, content = _convert_html(html)
        assert content == prefix + text

    def test_ignored_comment_is_not_overcharged(self):
        """Comments and doctypes never reach the Markdown converter."""
        html = '<!doctype html>' + '<dd>' * 15 + '<!--' + 'x\n' * 300_000 + '-->' + '</dd>' * 15
        assert _convert_html(html)[1] == ''

    @pytest.mark.parametrize(('tag', 'attribute'), [('img', 'alt'), ('div', 'data-big')])
    def test_ignored_attribute_is_not_overcharged(self, tag: str, attribute: str):
        """Attributes absent from Markdown do not add to the deep text scan budget."""
        value = 'x' * 1_500_000
        html = '<div>' * 30 + f'<{tag} {attribute}="{value}"></{tag}>' + '</div>' * 30
        assert _convert_html(html)[1] == ''

    @pytest.mark.parametrize(
        ('element', 'expected'),
        [
            ('<source src="{value}">', ''),
            ('<a href="{value}"></a>', ''),
            ('<a title="{value}">text</a>', 'text'),
            ('<a href="{value}"><!--ignored--></a>', ''),
            ('<a href="{value}"><img alt="ignored"></a>', ''),
        ],
    )
    def test_unused_output_attribute_is_not_overcharged(self, element: str, expected: str):
        """Attributes that the converter omits do not add deep scan work."""
        html = '<div>' * 300 + element.format(value='x' * 18_000_000) + '</div>' * 300
        assert _convert_html(html)[1] == expected

    def test_collapsed_whitespace_is_not_overcharged(self):
        """A long whitespace run becomes one character before ancestor scans."""
        html = '<div>' * 300 + 'x' + ' ' * 18_000_000 + 'x' + '</div>' * 300
        assert _convert_html(html)[1] == 'x x'

    def test_small_nested_link_converts(self):
        """A link's URL counts towards deep scans without rejecting a small link."""
        html = '<div>' * 30 + '<a href="/x">link</a>' + '</div>' * 30
        assert _convert_html(html)[1] == '[link](/x)'

    @pytest.mark.parametrize('content', ['link<!--ignored-->', '<em>link</em>'])
    def test_nested_link_child_parity(self, content: str):
        """Ignored comments and formatted children keep their normal Markdown output."""
        html = '<div>' * 30 + f'<a href="/x">{content}</a>' + '</div>' * 30
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_deep_autolink_is_not_overcharged(self):
        """Autolink syntax replaces its text with the URL rather than appending a copy."""
        value = 'x' * 90_000
        html = '<div>' * 300 + f'<a href="{value}">{value}</a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_wrapped_deep_autolink_is_not_overcharged(self):
        """Transparent descendants preserve the converter's autolink shortcut."""
        value = 'x' * 90_000
        html = '<div>' * 300 + f'<a href="{value}"><span>{value}</span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_wrapped_autolink_with_surrounding_spaces_is_not_overcharged(self):
        """The autolink shortcut strips surrounding whitespace before comparing the URL."""
        value = 'x' * 90_000
        html = '<div>' * 300 + f'<a href="{value}"><span> {value} </span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_wrapped_autolink_with_ignored_comment_is_not_overcharged(self):
        """Ignored comments do not interrupt autolink text in transparent descendants."""
        value = 'x' * 90_000
        content = value[:45_000] + '<!--ignored-->' + value[45_000:]
        html = '<div>' * 300 + f'<a href="{value}"><span>{content}</span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_wrapped_autolink_collapses_newlines_across_ignored_comment(self):
        """Autolink detection matches markdownify's newline merging between child strings."""
        left = 'x' * 45_000
        right = 'x' * 44_999
        href = left + '\n' + right
        content = left + '\n<!--ignored-->\n' + right
        html = '<div>' * 300 + f'<a href="{href}"><span>{content}</span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{href}>'

    def test_wrapped_autolink_collapses_newline_only_child(self):
        """A child containing only newlines is collapsed at its own tag boundary."""
        href = 'x\ny'
        html = '<div>' * 17 + f'<a href="{href}"><span>x<!--ignored-->\n\n<!--ignored-->y</span></a>'
        html += '</div>' * 17
        assert _convert_html(html)[1] == f'<{href}>'

    def test_many_autolink_fragments_keep_conversion_bounded(self):
        """Transparent ancestors do not duplicate every descendant text fragment."""
        value = 'x' * 20_000
        content = 'x<!--ignored-->' * 20_000
        html = f'<a href="{value}">' + '<span>' * 300 + content + '</span>' * 300 + '</a>'
        assert _convert_html(html)[1] == f'<{value}>'

    def test_autolink_probe_node_budget(self):
        """Many link descendants are charged even when they render as one autolink."""
        value = 'x' * 200
        html = '<div>' * 17 + f'<a href="{value}">' + 'x<!--ignored-->' * 200 + '</a>' + '</div>' * 17
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_CONVERSION_COST', 1500):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.parametrize('budget', [50, 150, 250])
    def test_autolink_probe_text_budget(self, budget: int):
        """Rendered link text is bounded while inline context skips the leaf scan estimate."""
        value = 'x' * 100
        html = '<div>' * 16 + f'<h3><a href="{value}">' + 'x<!--ignored-->' * 100
        html += '</a></h3>' + '</div>' * 16
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_TEXT_SCAN_COST', budget):
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html(html)

    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_pre_padding_is_not_overcharged(self):
        """Preformatted whitespace is stripped before enclosing blocks scan it."""
        html = '<div>' * 300 + '<pre>' + ' ' * 180_000 + '</pre>' + '</div>' * 300
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    @pytest.mark.parametrize('container', ['<table><tr><td>{content}</td></tr></table>', '<h3>{content}</h3>'])
    @pytest.mark.usefixtures('scaled_conversion_budget')
    def test_collapsed_newlines_are_not_overcharged(self, container: str):
        """Cells and headings collapse newlines before outer definition items see them."""
        html = '<dd>' * 15 + container.format(content='x\n' * 3_000) + '</dd>' * 15
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    def test_inline_video_does_not_use_src(self):
        """A video in a table cell keeps its text and ignores its source URL."""
        html = (
            '<div>' * 300
            + '<table><tr><td><video src="'
            + 'x' * 18_000_000
            + '"></video></td></tr></table>'
            + '</div>' * 300
        )
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    @pytest.mark.parametrize('container', ['<table><tr><td>{content}</td></tr></table>', '<h2>{content}</h2>'])
    def test_inline_indentation_is_not_overcharged(self, container: str):
        """Blockquotes inside inline cells and headings do not indent their lines."""
        content = '<blockquote>' * 15 + 'x\n' * 300_000 + '</blockquote>' * 15
        html = container.format(content=content)
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    def test_shallow_table_colspan_is_bounded(self):
        """A small table can generate millions of cell and header separators."""
        html = '<table><tr>' + '<td colspan="1000">x</td>' * 3000 + '</tr></table>'
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_repeated_table_row_search_is_bounded(self):
        """Rows under thead must not each rescan all of their siblings."""
        html = '<table><thead>' + '<tr><td>x</td></tr>' * 5000 + '</thead></table>'
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    def test_repeated_tbody_table_search_is_bounded(self):
        """The first row of each tbody must not rescan the whole table."""
        html = '<table>' + '<tbody><tr><td>x</td></tr></tbody>' * 2000 + '</table>'
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    def test_small_table_colspan_converts(self):
        """Small decimal colspans retain the converter's output."""
        html = '<table><tr><td colspan="002">x</td></tr></table>'
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    def test_nested_video_source_search_is_bounded(self):
        """Repeated source searches through ignored descendants must be counted before conversion."""
        comments = '<!---->' * 100
        with patch('pydantic_ai.common_tools.web_fetch._MAX_HTML_CONVERSION_COST', 1_000):
            assert _convert_html(f'<video>{comments}</video>')[1] == ''
            with pytest.raises(ModelRetry, match='too complex'):
                _convert_html('<video>' * 3 + comments + '</video>' * 3)

    @pytest.mark.parametrize(('tag', 'character'), [('code', '`'), ('h1', 'x')])
    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_generated_text_growth_is_bounded(self, tag: str, character: str):
        """Code delimiters and underlined headings multiply long child text."""
        html = '<div>' * 300 + f'<{tag}>' + character * 160_000 + f'</{tag}>' + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_heading_generated_link_growth_is_bounded(self):
        """An underlined heading duplicates its rendered link, including the URL."""
        value = 'x' * 17_000_000
        html = '<div>' * 300 + f'<h1><a href="{value}">link</a></h1>' + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    @pytest.mark.usefixtures('scaled_conversion_budget')
    def test_generated_link_newlines_are_bounded(self):
        """Link URLs can create lines that nested definition items must indent."""
        html = '<dd>' * 15 + '<a href="' + 'x\n' * 6_000 + '">z</a>' + '</dd>' * 15
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    @pytest.mark.parametrize('element', ['<pre><code>x</code></pre>', '<table><tr><td><h1>x</h1></td></tr></table>'])
    def test_nested_formatted_text_is_not_overcharged(self, element: str):
        """Code in pre and headings in table cells do not add format growth."""
        html = '<div>' * 20 + element + '</div>' * 20
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    def test_escaped_link_title_is_bounded(self):
        """Every quote in a link title adds an escape character."""
        html = '<div>' * 300 + "<a href='/x' title='" + '"' * 17_000_000 + "'>link</a>" + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_anchor_in_code_does_not_use_href(self):
        """A link inside code keeps only its text, even when deeply nested."""
        html = '<div>' * 300 + '<code><a href="' + 'x' * 18_000_000 + '">link</a></code>' + '</div>' * 300
        assert _convert_html(html)[1] == '`link`'

    @pytest.mark.parametrize(
        ('template', 'character', 'length'),
        [
            pytest.param('{value}', 'x', 180_000, id='text'),
            pytest.param('<a href="{value}">link</a>', 'x', 180_000, id='link'),
            pytest.param('<video src="{value}"></video>', 'x', 180_000, id='video'),
            pytest.param('<video><source src="{value}"></video>', 'x', 180_000, id='video-source'),
            pytest.param('{value}', '*', 160_000, id='escaped-asterisks'),
        ],
    )
    @pytest.mark.usefixtures('scaled_text_scan_budget')
    def test_deep_text_scan_is_bounded(self, template: str, character: str, length: int):
        """Large output text copied through hundreds of ancestors has a separate work bound."""
        content = template.format(value=character * length)
        html = '<div>' * 300 + content + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    @pytest.mark.usefixtures('scaled_conversion_budget')
    def test_shallow_generated_line_breaks_are_bounded(self):
        """Tags that create line breaks count even without newline text nodes."""
        html = '<dl>' + '<dd>' * 13 + 'x<br>' * 3_200 + '</dd>' * 13 + '</dl>'
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_wide_ordered_list_marker_is_bounded(self):
        """A large list start must count towards indentation on every continuation line."""
        html = '<ol start="' + '9' * 4300 + '"><li>' + 'x\n' * 10_000 + '</li></ol>'
        started = time.perf_counter()
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)
        assert time.perf_counter() - started < 60

    def test_generated_list_lines_with_wide_marker_are_bounded(self):
        """A URL can generate the lines an ordered-list marker would indent."""
        html = '<ol start="' + '9' * 1000 + '"><li><a href="' + 'x\n' * 30_000 + '">z</a></li></ol>'
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    @pytest.mark.parametrize(
        'html',
        [
            pytest.param('<p>x' + ' ' * 300_000 + 'x</p>', id='spaces-in-paragraph'),
            pytest.param('<p><![CDATA[x' + ' ' * 300_000 + 'x]]></p>', id='spaces-in-cdata'),
            pytest.param('<pre>' + ' ' * 300_000 + 'x</pre>', id='spaces-in-pre'),
            pytest.param('<ol>' + '<li>x</li>' * 50_000 + '</ol>', id='long-ordered-list'),
            pytest.param('<div>x' * 20_000, id='deep-nesting'),
            pytest.param('x <i></i>' * 50_000, id='many-sibling-text-nodes'),
        ],
    )
    def test_converts_pathological_runs_quickly(self, html: str):
        """Whitespace runs, `<pre>` padding, ordered lists, deep nesting, and wide trees are handled in linear time.

        `markdownify` on its own takes minutes on the whitespace and list shapes: a run of spaces
        restarts its whitespace regexes at every character, and each `<li>` recounts its previous
        siblings. The nested page can't be converted at all, but finding that out must not take
        long either, and neither may normalizing text among tens of thousands of siblings. The
        bound is generous; the point is that it isn't minutes.
        """
        start = time.perf_counter()
        try:
            _convert_html(html)
        except (RecursionError, ModelRetry):
            assert html.startswith('<div>x<div>')
        assert time.perf_counter() - start < 60


class TestWebFetchToolFactory:
    def test_creates_tool(self):
        """web_fetch_tool() returns a Tool with correct name."""
        tool = web_fetch_tool()
        assert tool.name == 'web_fetch'

    def test_custom_parameters(self):
        """web_fetch_tool() accepts custom parameters."""
        tool = web_fetch_tool(
            max_content_length=10_000, timeout=60, allow_local_urls=True, max_download_bytes=1_000_000
        )
        assert tool.name == 'web_fetch'
