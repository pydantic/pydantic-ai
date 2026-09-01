"""Web fetch tool for Pydantic AI agents.

Fetches web pages and converts their content to markdown using SSRF-protected
HTTP requests and the `markdownify` library for HTML-to-markdown conversion.
"""

from __future__ import annotations

import json
import re
from dataclasses import KW_ONLY, dataclass, field

import httpx2
from typing_extensions import Any, TypedDict

from pydantic_ai._ssrf import safe_download
from pydantic_ai._utils import is_text_like_media_type, run_in_executor
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import Tool

try:
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter
except ImportError as _import_error:
    raise ImportError(
        'Please install `markdownify` to use the web fetch tool, '
        'you can use the `web-fetch` optional group — `pip install "pydantic-ai-slim[web-fetch]"`'
    ) from _import_error

__all__ = ('WebFetchResult', 'web_fetch_tool')

_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


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

    max_download_bytes: int | None = field(default=_MAX_DOWNLOAD_BYTES)
    """Maximum size in bytes of the response body to download. None for no limit."""

    allowed_domains: list[str] | None = field(default=None)
    """Only fetch from these domains (exact hostname match). Raises `ModelRetry` on violation."""

    blocked_domains: list[str] | None = field(default=None)
    """Never fetch from these domains (exact hostname match). Raises `ModelRetry` on violation."""

    headers: dict[str, str] | None = field(default=None)
    """Additional HTTP headers to include in the request.

    The model controls the URL, so use `allowed_domains` when these include credentials.
    """

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
        request_headers = {'Accept': 'text/markdown, text/html;q=0.9, */*;q=0.8'}
        if self.headers:
            request_headers.update(self.headers)

        try:
            response = await safe_download(
                url,
                allow_local=self.allow_local_urls,
                timeout=self.timeout,
                headers=request_headers,
                allowed_domains=self.allowed_domains,
                blocked_domains=self.blocked_domains,
                max_bytes=self.max_download_bytes,
            )
        except (ValueError, httpx2.HTTPStatusError, httpx2.RequestError) as e:
            raise ModelRetry(f'Failed to fetch {url}: {e}') from e

        media_type = response.headers.get('content-type', '')
        media_type = media_type.split(';')[0].strip().lower()

        title = ''

        if not media_type or is_text_like_media_type(media_type):
            text = response.text

            if media_type in ('text/markdown', 'text/x-markdown'):
                content = text
            elif not media_type or media_type in ('text/html', 'application/xhtml+xml'):
                # Parsing and converting is CPU-bound and scales with the (server-controlled) body
                # size, so run it in a worker thread rather than on the event loop.
                try:
                    title, content = await run_in_executor(_convert_html, text)
                except RecursionError as e:
                    # `markdownify` walks the document recursively, so a page nested deeper than the
                    # interpreter's recursion limit can't be converted; let the model try elsewhere.
                    raise ModelRetry(f'Failed to convert {url}: the HTML is nested too deeply') from e
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


def _convert_html(html: str) -> tuple[str, str]:
    """Return the `<title>` text (empty if there is none) and the markdown conversion of the HTML.

    The document is parsed once and the tree is shared between both, using the same `html.parser`
    backend `markdownify` uses when handed a string.
    """
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.get_text().strip() if soup.title is not None else ''
    content = MarkdownConverter(strip=['img', 'script', 'style']).convert_soup(soup)
    return title, content


def _clean_whitespace(text: str) -> str:
    """Collapse runs of 3+ newlines into 2 newlines."""
    return _EXCESSIVE_NEWLINES_RE.sub('\n\n', text).strip()


def web_fetch_tool(
    *,
    max_content_length: int | None = 50_000,
    allow_local_urls: bool = False,
    timeout: int = 30,
    max_download_bytes: int | None = _MAX_DOWNLOAD_BYTES,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> Tool[Any]:
    """Creates a web fetch tool that fetches URLs and converts content to markdown.

    This tool uses SSRF protection via `pydantic_ai._ssrf.safe_download`.

    By default, sends `Accept: text/markdown` to request markdown directly from
    servers that support it (e.g. Cloudflare, Vercel, Mintlify). This reduces
    token usage and improves content quality. Falls back to HTML-to-markdown
    conversion when the server doesn't support markdown responses.

    Args:
        max_content_length: Maximum character length of returned content.
            Defaults to 50,000 (~12,500 tokens). Use `None` for no limit.
        allow_local_urls: Whether to allow fetching from private/local IP addresses.
            Defaults to `False`.
        timeout: Request timeout in seconds. Defaults to 30.
        max_download_bytes: Maximum size in bytes of the response body to download, applied
            before the body is buffered. Defaults to 50 MiB. Use `None` for no limit, which
            lets a response of any size be read into memory.
        allowed_domains: Only fetch from these domains (exact hostname match). Raises `ModelRetry` on violation.
        blocked_domains: Never fetch from these domains (exact hostname match). Raises `ModelRetry` on violation.
        headers: Additional HTTP headers to include in requests.
            Overrides the default `Accept: text/markdown` header if `Accept` is provided.
            The URL is controlled by the model, so a credential configured here (e.g.
            `Authorization`) can be sent to any URL the model requests that passes the
            domain filters, which match the hostname only, not scheme or port. On
            redirects, configured sensitive headers (`Authorization`, `Cookie`,
            `Proxy-Authorization`) are only forwarded to the same origin (scheme,
            host, and port) or a same-host http→https upgrade on the default ports.
    """
    return Tool[Any](
        WebFetchLocalTool(
            max_content_length=max_content_length,
            allow_local_urls=allow_local_urls,
            timeout=timeout,
            max_download_bytes=max_download_bytes,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            headers=headers,
        ).__call__,
        name='web_fetch',
        description='Fetches the content of a web page at the given URL and returns it as markdown or binary content.',
    )
