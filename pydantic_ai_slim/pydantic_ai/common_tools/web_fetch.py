"""Web fetch tool for Pydantic AI agents.

Fetches web pages and converts their content to markdown using SSRF-protected
HTTP requests and the ``markdownify`` library for HTML-to-markdown conversion.
"""

from __future__ import annotations

import json
import re
from dataclasses import KW_ONLY, dataclass

from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from markdownify import markdownify as md
except ImportError as _import_error:
    raise ImportError(
        'Please install `markdownify` to use the web fetch tool, '
        'you can use the `web-fetch` optional group — `pip install "pydantic-ai-slim[web-fetch]"`'
    ) from _import_error

__all__ = ('web_fetch_tool',)


class WebFetchResult(TypedDict):
    """Result of fetching a web page."""

    url: str
    """The URL that was fetched."""
    title: str
    """The page title, or empty string if not found."""
    content: str
    """The page content converted to markdown."""


@dataclass
class WebFetchLocalTool:
    """Fetches a URL and converts the response to markdown."""

    _: KW_ONLY

    max_content_length: int | None
    """Maximum character length of returned content. None for no limit."""

    allow_local_urls: bool
    """Whether to allow fetching from private/local IP addresses."""

    timeout: int
    """Request timeout in seconds."""

    async def __call__(self, url: str) -> WebFetchResult:
        """Fetches the content of a web page at the given URL and returns it as markdown.

        Args:
            url: The URL to fetch.

        Returns:
            The fetched page content.
        """
        from pydantic_ai._ssrf import safe_download

        response = await safe_download(
            url,
            allow_local=self.allow_local_urls,
            timeout=self.timeout,
        )

        content_type = response.headers.get('content-type', '')
        content_type = content_type.split(';')[0].strip().lower()

        text = response.text
        title = ''

        if not content_type or content_type == 'text/html':
            title = _extract_title(text)
            content = md(text, strip=['img', 'script', 'style'])
        elif content_type == 'application/json':
            try:
                parsed = json.loads(text)
                content = f'```json\n{json.dumps(parsed, indent=2)}\n```'
            except (json.JSONDecodeError, ValueError):
                content = text
        elif content_type.startswith('text/'):
            content = text
        else:
            content = f'[Binary content of type {content_type} cannot be displayed as text]'

        content = _clean_whitespace(content)

        if self.max_content_length is not None and len(content) > self.max_content_length:
            content = content[: self.max_content_length] + '\n\n[Content truncated]'

        return WebFetchResult(url=url, title=title, content=content)


def _extract_title(html: str) -> str:
    """Extract the <title> from HTML."""
    lower = html.lower()
    start = lower.find('<title')
    if start == -1:
        return ''
    start = lower.find('>', start)
    if start == -1:
        return ''
    end = lower.find('</title>', start)
    if end == -1:
        return ''
    return html[start + 1 : end].strip()


def _clean_whitespace(text: str) -> str:
    """Collapse runs of 3+ newlines into 2 newlines."""
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
) -> Tool[Any]:
    """Creates a web fetch tool that fetches URLs and converts content to markdown.

    This tool uses SSRF protection via ``pydantic_ai._ssrf.safe_download``.

    Args:
        max_content_length: Maximum character length of returned content.
            Defaults to 50,000 (~12,500 tokens). Use ``None`` for no limit.
        allow_local_urls: Whether to allow fetching from private/local IP addresses.
            Defaults to ``False``.
        timeout: Request timeout in seconds. Defaults to 30.
    """
    return Tool[Any](
        WebFetchLocalTool(
            max_content_length=max_content_length,
            allow_local_urls=allow_local_urls,
            timeout=timeout,
        ).__call__,
        name='web_fetch',
        description='Fetches the content of a web page at the given URL and returns it as markdown.',
    )
