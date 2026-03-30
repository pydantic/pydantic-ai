"""Tests for the web fetch common tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pydantic_ai.common_tools.web_fetch import (
    WebFetchLocalTool,
    web_fetch_tool,
)

pytestmark = [pytest.mark.anyio]


def _html_response(html: str, *, content_type: str = 'text/html; charset=utf-8') -> httpx.Response:
    """Helper to create a mock HTML response."""
    return httpx.Response(
        200,
        text=html,
        headers={'content-type': content_type},
        request=httpx.Request('GET', 'https://example.com'),
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
        mock_response = httpx.Response(
            200,
            text='{"key": "value"}',
            headers={'content-type': 'application/json'},
            request=httpx.Request('GET', 'https://api.example.com/data'),
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
        mock_response = httpx.Response(
            200,
            text='{invalid json',
            headers={'content-type': 'application/json'},
            request=httpx.Request('GET', 'https://api.example.com/data'),
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
        mock_response = httpx.Response(
            200,
            text='Hello, plain text!',
            headers={'content-type': 'text/plain'},
            request=httpx.Request('GET', 'https://example.com/file.txt'),
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
        mock_response = httpx.Response(
            200,
            content=html.encode(),
            headers={},
            request=httpx.Request('GET', 'https://example.com'),
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
        mock_response = httpx.Response(
            200,
            text=long_text,
            headers={'content-type': 'text/plain'},
            request=httpx.Request('GET', 'https://example.com'),
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
        mock_response = httpx.Response(
            200,
            text=xml,
            headers={'content-type': 'application/xml'},
            request=httpx.Request('GET', 'https://example.com/feed.xml'),
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
        mock_response = httpx.Response(
            200,
            text=xhtml,
            headers={'content-type': 'application/xhtml+xml'},
            request=httpx.Request('GET', 'https://example.com'),
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
        mock_response = httpx.Response(
            200,
            content=pdf_bytes,
            headers={'content-type': 'application/pdf'},
            request=httpx.Request('GET', 'https://example.com/doc.pdf'),
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
        mock_response = httpx.Response(
            200,
            text=html,
            headers={'content-type': 'text/html'},
            request=httpx.Request('GET', 'http://localhost:8080'),
        )

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ) as mock_dl:
            tool = WebFetchLocalTool(max_content_length=None, allow_local_urls=True, timeout=60)
            await tool('http://localhost:8080')

        mock_dl.assert_called_once_with('http://localhost:8080', allow_local=True, timeout=60)

    async def test_invalid_url_no_hostname(self):
        """URL without hostname raises ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        tool = WebFetchLocalTool(
            max_content_length=None, allow_local_urls=False, timeout=30, allowed_domains=['example.com']
        )
        with pytest.raises(ModelRetry, match='no hostname found'):
            await tool('not-a-url')

    async def test_allowed_domains_permits(self):
        """Allowed domain passes validation."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, allowed_domains=['example.com']
            )
            result = await tool('https://example.com/page')

        assert isinstance(result, dict)
        assert result['url'] == 'https://example.com/page'

    async def test_allowed_domains_blocks(self):
        """Non-allowed domain raises ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        tool = WebFetchLocalTool(
            max_content_length=None, allow_local_urls=False, timeout=30, allowed_domains=['example.com']
        )
        with pytest.raises(ModelRetry, match='not in the allowed domains'):
            await tool('https://evil.com/page')

    async def test_blocked_domains_blocks(self):
        """Blocked domain raises ModelRetry."""
        from pydantic_ai.exceptions import ModelRetry

        tool = WebFetchLocalTool(
            max_content_length=None, allow_local_urls=False, timeout=30, blocked_domains=['evil.com']
        )
        with pytest.raises(ModelRetry, match='is blocked'):
            await tool('https://evil.com/page')

    async def test_blocked_domains_permits(self):
        """Non-blocked domain passes validation."""
        mock_response = _html_response('<html><body>ok</body></html>')

        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            tool = WebFetchLocalTool(
                max_content_length=None, allow_local_urls=False, timeout=30, blocked_domains=['evil.com']
            )
            result = await tool('https://example.com/page')

        assert isinstance(result, dict)
        assert result['url'] == 'https://example.com/page'


class TestWebFetchToolFactory:
    def test_creates_tool(self):
        """web_fetch_tool() returns a Tool with correct name."""
        tool = web_fetch_tool()
        assert tool.name == 'web_fetch'

    def test_custom_parameters(self):
        """web_fetch_tool() accepts custom parameters."""
        tool = web_fetch_tool(max_content_length=10_000, timeout=60, allow_local_urls=True)
        assert tool.name == 'web_fetch'
