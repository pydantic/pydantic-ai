"""Web fetch tool for Pydantic AI agents.

Fetches web pages and converts their content to markdown using SSRF-protected
HTTP requests and the `markdownify` library for HTML-to-markdown conversion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import httpx2
from typing_extensions import Any, TypedDict

from pydantic_ai._ssrf import safe_download
from pydantic_ai._utils import is_text_like_media_type, run_in_executor
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import Tool

try:
    from bs4 import BeautifulSoup, Tag
    from bs4.element import NavigableString
    from markdownify import MarkdownConverter
except ImportError as _import_error:
    raise ImportError(
        'Please install `markdownify` to use the web fetch tool, '
        'you can use the `web-fetch` optional group — `pip install "pydantic-ai-slim[web-fetch]"`'
    ) from _import_error

__all__ = ('WebFetchResult', 'web_fetch_tool')

_EXCESSIVE_NEWLINES_RE = re.compile(r'\n{3,}')
_WHITESPACE_RUN_RE = re.compile(r'[\t \r\n]+')
_LINE_WITH_CONTENT_RE = re.compile(r'^(.*)', flags=re.MULTILINE)
# `markdownify`'s stub doesn't declare its per-tag `convert_<tag>` methods, which it looks up by name.
_upstream_convert_li: Callable[[MarkdownConverter, Tag, str, set[str]], str] = getattr(MarkdownConverter, 'convert_li')
_upstream_process_text: Callable[[MarkdownConverter, NavigableString, set[str] | None], str] = getattr(
    MarkdownConverter, 'process_text'
)
_TITLE_OPEN_RE = re.compile(r'<title', re.IGNORECASE)
_TITLE_CLOSE_RE = re.compile(r'</title>', re.IGNORECASE)
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
    """Only fetch from these domains (exact hostname match, ignoring case, a trailing dot, and IDNA spelling).

    Raises `ModelRetry` on violation.
    """

    blocked_domains: list[str] | None = field(default=None)
    """Never fetch from these domains (exact hostname match, ignoring case, a trailing dot, and IDNA spelling).

    Raises `ModelRetry` on violation.
    """

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
        # The model picks the URL, so every way one can be bad has to come back as something it can
        # act on. `InvalidURL` (an IDNA hostname httpx2 rejects, say) subclasses neither `ValueError`
        # nor `HTTPError`, so it needs naming separately.
        except (ValueError, httpx2.InvalidURL, httpx2.HTTPStatusError, httpx2.RequestError) as e:
            raise ModelRetry(f'Failed to fetch {url}: {e}') from e

        media_type = response.headers.get('content-type', '')
        media_type = media_type.split(';')[0].strip().lower()

        title = ''

        if not media_type or is_text_like_media_type(media_type):
            # The server picks the charset, and decoding costs time proportional to the body, or
            # worse for some codecs, so run it in a worker thread rather than on the event loop.
            try:
                text = await run_in_executor(_decode_text, response)
            except (UnicodeError, LookupError) as e:
                # Not every registered codec can decode a document (`idna`, say), and some aren't
                # text encodings at all (`rot_13`), so don't let a bad label take the run down.
                raise ModelRetry(f'Failed to decode {url}: {e}') from e

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
                except (json.JSONDecodeError, ValueError, RecursionError):
                    # A document nested deeper than the interpreter's recursion limit is returned
                    # as-is, like any other body that doesn't parse.
                    content = text
            else:
                content = text
        else:
            return BinaryContent(data=response.content, media_type=media_type or 'application/octet-stream')

        content = _clean_whitespace(content)

        if self.max_content_length is not None and len(content) > self.max_content_length:
            content = content[: self.max_content_length] + '\n\n[Content truncated]'

        return WebFetchResult(url=url, title=title, content=content)


def _decode_text(response: httpx2.Response) -> str:
    """Decode the body with the charset the server declared, or UTF-8 if it declared none.

    This goes through `bytes.decode` rather than `response.text` so that a label naming a
    registered codec that isn't a text encoding (`rot_13`, `base64_codec`) raises `LookupError`
    instead of failing inside the codec with whatever it happens to raise.
    """
    return response.content.decode(response.encoding or 'utf-8', errors='replace')


def _convert_html(html: str) -> tuple[str, str]:
    """Return the raw `<title>` text (empty if there is none) and the markdown conversion of the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    return _extract_title(html), _MarkdownConverter(strip=['img', 'script', 'style']).convert_soup(soup)


class _MarkdownConverter(MarkdownConverter):
    r"""`markdownify`'s converter with linear-time replacements for its super-linear steps.

    Three of its steps take time quadratic in a run of server-controlled input, and the regex
    ones hold the GIL while they run, so a worker thread doesn't shield the event loop from them:

    - text outside `<pre>` is normalized with `[\t \r\n]*[\r\n][\t \r\n]*`, which restarts at
      every character of a long run of spaces that has no newline;
    - `<pre>` blocks are stripped with `[ \n]*$`, which does the same on a run of spaces that
      isn't at the very end;
    - each `<li>` in an `<ol>` is numbered by counting all of its previous siblings.

    Each override produces exactly what the upstream step produces.
    """

    def __init__(self, **options: Any):
        super().__init__(**options)
        self._ol_indexes: dict[int, int] = {}

    def process_text(self, el: NavigableString, parent_tags: set[str] | None = None) -> str:
        # Collapse whitespace runs ahead of time, the way upstream's regexes would, so they only
        # ever see runs of one character. Upstream reads the node's text and its neighbours, so
        # hand it a detached stand-in carrying the same links rather than editing the tree:
        # `replace_with` has to find the node among its siblings, which is linear per node.
        if 'pre' not in (parent_tags or ()):
            normalized = _WHITESPACE_RUN_RE.sub(_collapse_whitespace_run, el)
            if normalized != el:
                stand_in = type(el)(normalized)
                stand_in.parent = el.parent
                stand_in.previous_sibling = el.previous_sibling
                stand_in.next_sibling = el.next_sibling
                el = stand_in
        return _upstream_process_text(self, el, parent_tags)

    def convert_pre(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        # Mirrors upstream with its default `strip_pre='strip'` applied linearly; the code language
        # options upstream consults here are never set on this converter.
        if not text:
            return ''
        return f'\n\n```\n{_strip_pre(text)}\n```\n\n'

    def convert_li(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        parent = el.parent
        if parent is None or parent.name != 'ol':
            return _upstream_convert_li(self, el, text, parent_tags)
        # The rest mirrors upstream's ordered-list branch, with the sibling count replaced by
        # a per-list index.
        text = (text or '').strip()
        if not text:
            return '\n'
        if id(el) not in self._ol_indexes:
            # Upstream counts each item's previous siblings, which is quadratic per list; index
            # the list once instead.
            index = 0
            for child in parent.children:
                if isinstance(child, Tag) and child.name == 'li':
                    self._ol_indexes[id(child)] = index
                    index += 1
        start_attr = parent.get('start')
        # Upstream checks `isnumeric()` before `int()`, which raises on digits like `²`; treat
        # those as no start rather than letting the page abort the run.
        start = int(start_attr) if isinstance(start_attr, str) and start_attr.isdecimal() else 1
        bullet = f'{start + self._ol_indexes[id(el)]}. '
        bullet_indent = ' ' * len(bullet)

        def indent_line(match: re.Match[str]) -> str:
            line = match.group(1)
            return bullet_indent + line if line else ''

        text = _LINE_WITH_CONTENT_RE.sub(indent_line, text)
        return f'{bullet}{text[len(bullet) :]}\n'


def _collapse_whitespace_run(match: re.Match[str]) -> str:
    run = match.group()
    return '\n' if '\n' in run or '\r' in run else ' '


def _strip_pre(text: str) -> str:
    r"""Strip all leading and trailing newlines from a `<pre>` string, like `markdownify.strip_pre`.

    Upstream removes `^[ \n]*\n` and `[ \n]*$`; this walks each end once instead.
    """
    leading = len(text) - len(text.lstrip(' \n'))
    last_newline = text.rfind('\n', 0, leading)
    if last_newline != -1:
        text = text[last_newline + 1 :]
    return text.rstrip(' \n')


def _extract_title(html: str) -> str:
    """Extract the raw text of the first `<title>` element.

    A single forward scan: the first `<title` start and the `</title>` end are matched
    case-insensitively, the `>` closing the start tag literally. Each step either finds its
    marker or settles the result, and the patterns are plain literals with nothing to backtrack
    over, so the cost is linear in the size of the document regardless of how malformed it is.
    """
    opening = _TITLE_OPEN_RE.search(html)
    if opening is None:
        return ''
    open_end = html.find('>', opening.end())
    if open_end == -1:
        return ''
    closing = _TITLE_CLOSE_RE.search(html, open_end + 1)
    if closing is None:
        return ''
    return html[open_end + 1 : closing.start()].strip()


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
        allowed_domains: Only fetch from these domains (exact hostname match, ignoring case and a
            trailing dot). Raises `ModelRetry` on violation.
        blocked_domains: Never fetch from these domains (exact hostname match, ignoring case and a
            trailing dot). Raises `ModelRetry` on violation.
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
