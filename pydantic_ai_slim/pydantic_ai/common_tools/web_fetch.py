"""Web fetch tool for Pydantic AI agents.

Fetches web pages and converts their content to markdown using SSRF-protected
HTTP requests and the `markdownify` library for HTML-to-markdown conversion.
"""

from __future__ import annotations

import json
import re
from dataclasses import KW_ONLY, dataclass, field
from urllib.parse import urlparse

from typing_extensions import Any, TypedDict

from pydantic_ai._ssrf import safe_download
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import Tool

try:
    from markdownify import markdownify as md
except ImportError as _import_error:
    raise ImportError(
        'Please install `markdownify` to use the web fetch tool, '
        'you can use the `web-fetch` optional group — `pip install "pydantic-ai-slim[web-fetch]"`'
    ) from _import_error

__all__ = ('WebFetchResult', 'web_fetch_tool')

_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')


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

    allowed_domains: list[str] | None = field(default=None)
    """Only fetch from these domains. Raises `ModelRetry` on violation."""

    blocked_domains: list[str] | None = field(default=None)
    """Never fetch from these domains. Raises `ModelRetry` on violation."""

    async def __call__(self, url: str) -> WebFetchResult | BinaryContent:
        """Fetches the content of a web page at the given URL and returns it as markdown.

        For textual content (HTML, JSON, plain text), returns a
        [`WebFetchResult`][pydantic_ai.common_tools.web_fetch.WebFetchResult].
        For binary content (PDF, images, etc.), returns a
        [`BinaryContent`][pydantic_ai.messages.BinaryContent] so the model can
        process it natively.

        Args:
            url: The URL to fetch.

        Returns:
            The fetched page content.
        """
        _check_domain(url, self.allowed_domains, self.blocked_domains)

        response = await safe_download(
            url,
            allow_local=self.allow_local_urls,
            timeout=self.timeout,
        )

        media_type = response.headers.get('content-type', '')
        media_type = media_type.split(';')[0].strip().lower()

        title = ''

        if _is_text_like_media_type(media_type):
            text = response.text

            if not media_type or media_type in ('text/html', 'application/xhtml+xml'):
                title = _extract_title(text)
                content = md(text, strip=['img', 'script', 'style'])
            elif media_type == 'application/json':
                try:
                    parsed = json.loads(text)
                    content = f'```json\n{json.dumps(parsed, indent=2)}\n```'
                except (json.JSONDecodeError, ValueError):
                    content = text
            else:
                content = text
        else:
            return BinaryContent(data=response.content, media_type=media_type or 'application/octet-stream')

        content = _clean_whitespace(content)

        if self.max_content_length is not None and len(content) > self.max_content_length:
            content = content[: self.max_content_length] + '\n\n[Content truncated]'

        return WebFetchResult(url=url, title=title, content=content)


def _check_domain(url: str, allowed_domains: list[str] | None, blocked_domains: list[str] | None) -> None:
    """Validate URL domain against allowed/blocked lists. Raises `ModelRetry` on violation."""
    hostname = urlparse(url).hostname
    if not hostname:
        raise ModelRetry('Invalid URL: no hostname found.')
    if allowed_domains is not None and hostname not in allowed_domains:
        raise ModelRetry(f'Domain {hostname!r} is not in the allowed domains list. Allowed: {allowed_domains}')
    if blocked_domains is not None and hostname in blocked_domains:
        raise ModelRetry(f'Domain {hostname!r} is blocked. Try a different URL.')


def _is_text_like_media_type(media_type: str) -> bool:
    """Check if a media type represents text-like content."""
    return (
        not media_type
        or media_type.startswith('text/')
        or media_type == 'application/json'
        or media_type.endswith('+json')
        or media_type == 'application/xml'
        or media_type.endswith('+xml')
        or media_type in ('application/x-yaml', 'application/yaml')
    )


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
    return _EXCESSIVE_NEWLINES_RE.sub('\n\n', text).strip()


def web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> Tool[Any]:
    """Creates a web fetch tool that fetches URLs and converts content to markdown.

    This tool uses SSRF protection via `pydantic_ai._ssrf.safe_download`.

    Args:
        max_content_length: Maximum character length of returned content.
            Defaults to 50,000 (~12,500 tokens). Use `None` for no limit.
        allow_local_urls: Whether to allow fetching from private/local IP addresses.
            Defaults to `False`.
        timeout: Request timeout in seconds. Defaults to 30.
        allowed_domains: Only fetch from these domains. Raises `ModelRetry` on violation.
        blocked_domains: Never fetch from these domains. Raises `ModelRetry` on violation.
    """
    return Tool[Any](
        WebFetchLocalTool(
            max_content_length=max_content_length,
            allow_local_urls=allow_local_urls,
            timeout=timeout,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        ).__call__,
        name='web_fetch',
        description='Fetches the content of a web page at the given URL and returns it as markdown.',
    )
