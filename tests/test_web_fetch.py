"""Tests for the web fetch common tool."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, patch

import anyio
import httpx2
import pytest
from markdownify import markdownify

from pydantic_ai._utils import using_thread_executor
from pydantic_ai.common_tools.web_fetch import (
    WebFetchLocalTool,
    _convert_html,  # pyright: ignore[reportPrivateUsage]
    web_fetch_tool,
)
from pydantic_ai.exceptions import ModelRetry


def _html_response(html: str, *, content_type: str = 'text/html; charset=utf-8') -> httpx2.Response:
    """Helper to create a mock HTML response."""
    return httpx2.Response(
        200,
        text=html,
        headers={'content-type': content_type},
        request=httpx2.Request('GET', 'https://example.com'),
    )


class TestWebFetchLocalTool:
    async def test_fetch_html(self):
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

    async def test_nested_html_conversion_keeps_event_loop_responsive(self):
        """A deeply nested definition list cannot occupy a worker and delay other coroutines for seconds."""
        html = '<dd>' * 120 + 'line\n' * 150_000 + '</dd>' * 120
        finished = anyio.Event()
        heartbeat_delays: list[float] = []

        async def heartbeat() -> None:
            previous = time.perf_counter()
            while not finished.is_set():
                await anyio.sleep(0.01)
                now = time.perf_counter()
                heartbeat_delays.append(now - previous)
                previous = now

        started = time.perf_counter()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(heartbeat)
            await anyio.sleep(0)
            try:
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
            finally:
                finished.set()

        assert time.perf_counter() - started < 60
        assert heartbeat_delays and max(heartbeat_delays) < 0.5

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


class TestMarkdownConverter:
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

    def test_deep_autolink_is_not_overcharged(self):
        """Autolink syntax replaces its text with the URL rather than appending a copy."""
        value = 'x' * 9_000_000
        html = '<div>' * 300 + f'<a href="{value}">{value}</a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    def test_wrapped_deep_autolink_is_not_overcharged(self):
        """Transparent descendants preserve the converter's autolink shortcut."""
        value = 'x' * 9_000_000
        html = '<div>' * 300 + f'<a href="{value}"><span>{value}</span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    def test_wrapped_autolink_with_surrounding_spaces_is_not_overcharged(self):
        """The autolink shortcut strips surrounding whitespace before comparing the URL."""
        value = 'x' * 9_000_000
        html = '<div>' * 300 + f'<a href="{value}"><span> {value} </span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    def test_wrapped_autolink_with_ignored_comment_is_not_overcharged(self):
        """Ignored comments do not interrupt autolink text in transparent descendants."""
        value = 'x' * 9_000_000
        content = value[:4_500_000] + '<!--ignored-->' + value[4_500_000:]
        html = '<div>' * 300 + f'<a href="{value}"><span>{content}</span></a>' + '</div>' * 300
        assert _convert_html(html)[1] == f'<{value}>'

    def test_wrapped_autolink_collapses_newlines_across_ignored_comment(self):
        """Autolink detection matches markdownify's newline merging between child strings."""
        left = 'x' * 4_500_000
        right = 'x' * 4_499_999
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

    def test_pre_padding_is_not_overcharged(self):
        """Preformatted whitespace is stripped before enclosing blocks scan it."""
        html = '<div>' * 300 + '<pre>' + ' ' * 18_000_000 + '</pre>' + '</div>' * 300
        assert _convert_html(html)[1] == markdownify(html, strip=['img', 'script', 'style'])

    @pytest.mark.parametrize('container', ['<table><tr><td>{content}</td></tr></table>', '<h3>{content}</h3>'])
    def test_collapsed_newlines_are_not_overcharged(self, container: str):
        """Cells and headings collapse newlines before outer definition items see them."""
        html = '<dd>' * 15 + container.format(content='x\n' * 300_000) + '</dd>' * 15
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
    def test_generated_text_growth_is_bounded(self, tag: str, character: str):
        """Code delimiters and underlined headings multiply long child text."""
        html = '<div>' * 300 + f'<{tag}>' + character * 16_000_000 + f'</{tag}>' + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_heading_generated_link_growth_is_bounded(self):
        """An underlined heading duplicates its rendered link, including the URL."""
        value = 'x' * 17_000_000
        html = '<div>' * 300 + f'<h1><a href="{value}">link</a></h1>' + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_generated_link_newlines_are_bounded(self):
        """Link URLs can create lines that nested definition items must indent."""
        html = '<dd>' * 15 + '<a href="' + 'x\n' * 600_000 + '">z</a>' + '</dd>' * 15
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
            pytest.param('{value}', 'x', 18_000_000, id='text'),
            pytest.param('<a href="{value}">link</a>', 'x', 18_000_000, id='link'),
            pytest.param('<video src="{value}"></video>', 'x', 18_000_000, id='video'),
            pytest.param('<video><source src="{value}"></video>', 'x', 18_000_000, id='video-source'),
            pytest.param('{value}', '*', 16_000_000, id='escaped-asterisks'),
        ],
    )
    def test_deep_text_scan_is_bounded(self, template: str, character: str, length: int):
        """Large output text copied through hundreds of ancestors has a separate work bound."""
        content = template.format(value=character * length)
        html = '<div>' * 300 + content + '</div>' * 300
        with pytest.raises(ModelRetry, match='too complex'):
            _convert_html(html)

    def test_shallow_generated_line_breaks_are_bounded(self):
        """Tags that create line breaks count even without newline text nodes."""
        html = '<dl>' + '<dd>' * 13 + 'x<br>' * 320_000 + '</dd>' * 13 + '</dl>'
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
