"""Web fetch tool for Pydantic AI agents.

Fetches web pages and converts their content to markdown using SSRF-protected
HTTP requests and the `markdownify` library for HTML-to-markdown conversion.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field

import httpx
from typing_extensions import Any, TypedDict

from pydantic_ai._ssrf import safe_download
from pydantic_ai._utils import is_text_like_media_type, run_in_executor
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent
from pydantic_ai.tools import Tool

try:
    from bs4 import BeautifulSoup, Tag
    from bs4.element import Comment, Doctype, NavigableString, PageElement
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
_upstream_get_conv_fn: Callable[[MarkdownConverter, str], Callable[[Tag, str, set[str]], str] | None] = getattr(
    MarkdownConverter, 'get_conv_fn'
)
_TITLE_OPEN_RE = re.compile(r'<title', re.IGNORECASE)
_TITLE_CLOSE_RE = re.compile(r'</title>', re.IGNORECASE)
_HTML_HEADING_RE = re.compile(r'h\d+')
_MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
# This rejects 15 nested `<dd>` tags with 300k short lines before 600 KB of HTML expands
# to ~19 MB of Markdown (0.7 s in a local conversion benchmark).
_MAX_HTML_CONVERSION_COST = 20_000_000
# Unchanged text is copied much faster: 10 MB inside 300 `<div>` tags took ~0.85 s,
# despite ~2.8 billion estimated character copies. Budget those separately.
_MAX_HTML_TEXT_SCAN_COST = 5_000_000_000


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
        except (ValueError, httpx.HTTPStatusError, httpx.RequestError) as e:
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
                except ModelRetry as e:
                    raise ModelRetry(f'Failed to convert {url}: {e}') from e
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


def _decode_text(response: httpx.Response) -> str:
    """Decode the body with the charset the server declared, or UTF-8 if it declared none.

    This goes through `bytes.decode` rather than `response.text` so that a label naming a
    registered codec that isn't a text encoding (`rot_13`, `base64_codec`) raises `LookupError`
    instead of failing inside the codec with whatever it happens to raise.
    """
    return response.content.decode(response.encoding or 'utf-8', errors='replace')


def _convert_html(html: str) -> tuple[str, str]:  # noqa: C901
    """Return the raw `<title>` text (empty if there is none) and the markdown conversion of the HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    converter = _MarkdownConverter(strip=['img', 'script', 'style'])
    # `markdownify` repeatedly scans each descendant's converted text as it walks back up the
    # tree. Blockquotes, definition items, and list items also indent every line at each level.
    # Estimate scans beyond 16 levels and indentation at any depth before conversion so a small,
    # deeply nested page cannot produce a huge intermediate string or hold the GIL for seconds.
    cost = 0
    text_scan_cost = 0
    nodes: list[PageElement] = []
    contentful: set[int] = set()
    direct_link_text: dict[int, str] = {}
    anchor_text_lengths: dict[int, int] = {}
    anchors: list[tuple[Tag, int, bool, bool]] = []
    videos: list[tuple[Tag, int, bool]] = []
    rows: list[Tag] = []
    pending: list[tuple[PageElement, int, int, int, bool, bool, bool, int | None]] = [
        (soup, 0, 0, 0, False, False, False, None)
    ]
    while pending:
        node, depth, indent_depth, indent_width, inline, noformat, in_pre, anchor_id = pending.pop()
        nodes.append(node)
        if node.next_sibling is not None:
            pending.append((node.next_sibling, depth, indent_depth, indent_width, inline, noformat, in_pre, anchor_id))
        if isinstance(node, (Comment, Doctype)):
            continue
        if isinstance(node, Tag):
            depth += 1
            if node.name == 'li' or (node.name in ('blockquote', 'dd') and not inline):
                indent_depth += 1
                if node.name == 'li' and isinstance(node.parent, Tag) and node.parent.name == 'ol':
                    parent_id = id(node.parent)
                    if parent_id not in converter.ordered_list_starts:
                        start_attr = node.parent.get('start')
                        start = 1
                        if isinstance(start_attr, str) and start_attr.isdecimal():
                            try:
                                start = int(start_attr)
                            except ValueError:
                                # Python limits decimal integer conversion length independently of HTML size.
                                pass
                        converter.ordered_list_starts[parent_id] = start
                    # `convert_li` bounds the actual marker before indenting lines.
                indent_width += 4
            # A tag can emit line breaks even without any text children (for example, `<br>`).
            cost += indent_width + indent_depth
            work = 8 * (indent_depth + 1)
            if node.name in ('td', 'th'):
                colspan_attr = node.get('colspan')
                digits = colspan_attr.lstrip('0') if isinstance(colspan_attr, str) and colspan_attr.isdecimal() else ''
                colspan = min(1000, int(digits[:4] or '1'))
                # A first-row cell can also generate two full-width header lines.
                work += 8 * colspan
                cost += 8 * colspan
            if node.name == 'a' and depth > 16 and anchor_id is None:
                anchors.append((node, depth, noformat, inline))
            elif node.name == 'video':
                videos.append((node, depth, inline))
            if node.name == 'tr':
                rows.append(node)
            if node.contents:
                child_inline = inline or node.name in ('td', 'th') or _HTML_HEADING_RE.match(node.name) is not None
                child_noformat = noformat or node.name in ('pre', 'code', 'kbd', 'samp')
                child_in_pre = in_pre or node.name == 'pre'
                child_anchor_id = (
                    anchor_id if anchor_id is not None else id(node) if node.name == 'a' and depth > 16 else None
                )
                pending.append(
                    (
                        node.contents[0],
                        depth,
                        indent_depth,
                        indent_width,
                        child_inline,
                        child_noformat,
                        child_in_pre,
                        child_anchor_id,
                    )
                )
            cost += max(depth - 16, 0) * work
        else:
            assert isinstance(node, NavigableString)
            parent_tags: set[str] = {'pre', '_noformat'} if in_pre else {'_noformat'} if noformat else set()
            converted_text = converter.process_text(node, parent_tags)
            if anchor_id is not None:
                direct_link_text[id(node)] = converted_text
            if converted_text.strip():
                contentful.add(id(node))
            # Inline and preformatted containers can discard these lines before an outer tag
            # sees them; the converter's output meter charges what actually survives.
            if not inline and not in_pre:
                cost += (indent_depth + indent_width) * converted_text.count('\n')
                # An anchor can discard escaped text before ancestors scan it. Charge the
                # rendered anchor after probing, or let the output meter handle inner tags.
                if anchor_id is None:
                    text_scan_cost += max(depth - 16, 0) * len(converted_text)
                else:
                    anchor_text_lengths[anchor_id] = anchor_text_lengths.get(anchor_id, 0) + len(converted_text)
        if cost > _MAX_HTML_CONVERSION_COST or text_scan_cost > _MAX_HTML_TEXT_SCAN_COST:
            raise ModelRetry('the document is too complex')

    subtree_sizes: dict[int, int] = {}
    first_sources: dict[int, str] = {}
    descendant_td: set[int] = set()
    colspan_scan_lengths: dict[int, int] = {}
    video_inline = {id(node): inline for node, _, inline in videos}
    for node in reversed(nodes):
        if isinstance(node, Tag):
            node_id = id(node)
            subtree_sizes[node_id] = 1
            colspan_scan_length = 0
            if node.name == 'td':
                descendant_td.add(node_id)
            if node.name in ('td', 'th'):
                colspan_attr = node.get('colspan')
                if isinstance(colspan_attr, str):
                    colspan_scan_length = len(colspan_attr)
            if node.name == 'source' and node.has_attr('src'):
                first_sources[node_id] = str(node.get('src') or '')
            for child in node.contents:
                child_id = id(child)
                subtree_sizes[node_id] += subtree_sizes.get(child_id, 1)
                colspan_scan_length += colspan_scan_lengths.get(child_id, 0)
                if child_id in contentful:
                    contentful.add(node_id)
                if child_id in descendant_td:
                    descendant_td.add(node_id)
                if node_id not in first_sources and child_id in first_sources:
                    first_sources[node_id] = first_sources[child_id]
            colspan_scan_lengths[node_id] = colspan_scan_length
            if node.name in ('hr', 'q', 'td', 'th', 'tr') or (
                node.name == 'video'
                and not video_inline[node_id]
                and (node.get('src') or node.get('poster') or first_sources.get(node_id))
            ):
                contentful.add(node_id)

    link_probe_cost = 0
    # These conversions can leave the child text unchanged after `convert_a` strips its edges.
    link_probe_tags = {
        'a',
        'article',
        'b',
        'blockquote',
        'br',
        'caption',
        'dd',
        'del',
        'div',
        'dl',
        'dt',
        'em',
        'figcaption',
        'hr',
        'i',
        'li',
        'ol',
        'p',
        'q',
        's',
        'section',
        'strong',
        'sub',
        'sup',
        'table',
        'ul',
    }
    for node, depth, noformat, inline in anchors:
        href = node.get('href')
        has_content = any(id(child) in contentful for child in node.contents)
        rendered_length = anchor_text_lengths.get(id(node), 0) if has_content else 0
        if href and not noformat and has_content:
            title = str(node.get('title') or '')
            rendered_length += len(str(href)) + len(title) + title.count('"')
            if not title:
                # Keep only direct child references. Flattening text fragments at every
                # transparent ancestor would retain depth * fragment-count references.
                link_nodes: list[PageElement] = [node]
                for child in node.descendants:
                    if isinstance(child, (Comment, Doctype)):
                        continue
                    cost += 8
                    if cost > _MAX_HTML_CONVERSION_COST:
                        raise ModelRetry('the document is too complex')
                    if isinstance(child, Tag):
                        if child.name == 'video':
                            if not inline and (child.get('src') or child.get('poster') or first_sources.get(id(child))):
                                break
                        elif (
                            child.name not in link_probe_tags
                            and not (inline and _HTML_HEADING_RE.match(child.name))
                            and _upstream_get_conv_fn(converter, child.name) is not None
                        ):
                            break
                    link_nodes.append(child)
                else:
                    rendered: dict[int, str] = {}
                    for child in reversed(link_nodes):
                        if isinstance(child, Tag):
                            child_strings: list[str] = []
                            for grandchild in child.contents:
                                child_string = rendered.pop(id(grandchild), '')
                                if child_string:
                                    child_strings.append(child_string)
                            if len(child_strings) > 1:
                                # Match markdownify's collapse at *each* tag boundary. A flat
                                # leaf join is different when a nested child is all newlines.
                                parts = ['']
                                for child_string in child_strings:
                                    link_probe_cost += len(child_string)
                                    if link_probe_cost > _MAX_HTML_TEXT_SCAN_COST:
                                        raise ModelRetry('the document is too complex')
                                    leading_count = len(child_string) - len(child_string.lstrip('\n'))
                                    trailing_count = len(child_string) - len(child_string.rstrip('\n'))
                                    if leading_count == len(child_string):
                                        trailing_count = 0
                                    leading = child_string[:leading_count]
                                    middle = child_string[leading_count : len(child_string) - trailing_count]
                                    trailing = (
                                        child_string[len(child_string) - trailing_count :] if trailing_count else ''
                                    )
                                    if parts[-1] and leading:
                                        previous = parts.pop()
                                        leading = '\n' * min(2, max(len(previous), len(leading)))
                                    parts.extend((leading, middle, trailing))
                                text = ''.join(parts)
                                link_probe_cost += len(text)
                                if link_probe_cost > _MAX_HTML_TEXT_SCAN_COST:
                                    raise ModelRetry('the document is too complex')
                            else:
                                text = child_strings[0] if child_strings else ''
                            if child is not node and (
                                child.name in link_probe_tags or (inline and _HTML_HEADING_RE.match(child.name))
                            ):
                                conversion = _upstream_get_conv_fn(converter, child.name)
                                assert conversion is not None
                                conversion_tags: set[str] = {'_inline'} if inline else set()
                                if child.name in ('ul', 'ol') and child.find_parent('li') is not None:
                                    conversion_tags.add('li')
                                text = conversion(child, text, conversion_tags)
                                link_probe_cost += len(text)
                                if link_probe_cost > _MAX_HTML_TEXT_SCAN_COST:
                                    raise ModelRetry('the document is too complex')
                            rendered[id(child)] = text
                        else:
                            rendered[id(child)] = direct_link_text[id(child)]
                    candidate = rendered[id(node)]
                    link_probe_cost += len(candidate)
                    if link_probe_cost > _MAX_HTML_TEXT_SCAN_COST:
                        raise ModelRetry('the document is too complex')
                    conversion = _upstream_get_conv_fn(converter, 'a')
                    assert conversion is not None
                    rendered_length = len(conversion(node, candidate, {'_inline'} if inline else set()))
                    link_probe_cost += rendered_length
                    if link_probe_cost > _MAX_HTML_TEXT_SCAN_COST:
                        raise ModelRetry('the document is too complex')
        text_scan_cost += (depth - 16) * rendered_length
        if text_scan_cost > _MAX_HTML_TEXT_SCAN_COST:
            raise ModelRetry('the document is too complex')

    for row in rows:
        cost += 8 * subtree_sizes[id(row)]
        text_scan_cost += colspan_scan_lengths[id(row)]
        parent = row.parent
        if isinstance(parent, Tag) and parent.name == 'thead' and id(row) in descendant_td:
            cost += 8 * subtree_sizes[id(parent)]
        elif (
            isinstance(parent, Tag)
            and parent.name == 'tbody'
            and row.find_previous_sibling() is None
            and isinstance(parent.parent, Tag)
        ):
            cost += 8 * subtree_sizes[id(parent.parent)]
        if cost > _MAX_HTML_CONVERSION_COST or text_scan_cost > _MAX_HTML_TEXT_SCAN_COST:
            raise ModelRetry('the document is too complex')

    for node, depth, inline in videos:
        if not inline:
            src = node.get('src')
            if not src:
                # `markdownify` searches descendants for the first source without its own URL.
                cost += 8 * subtree_sizes[id(node)]
                src = first_sources.get(id(node))
            if depth > 16:
                text_scan_cost += (depth - 16) * (len(str(src or '')) + len(str(node.get('poster') or '')))
            if cost > _MAX_HTML_CONVERSION_COST or text_scan_cost > _MAX_HTML_TEXT_SCAN_COST:
                raise ModelRetry('the document is too complex')
    del nodes, contentful, direct_link_text, anchor_text_lengths, anchors
    del videos, rows, subtree_sizes, first_sources, descendant_td, colspan_scan_lengths, video_inline
    return _extract_title(html), converter.convert_soup(soup)


class _MarkdownConverter(MarkdownConverter):
    r"""`markdownify`'s converter with linear-time replacements for its super-linear steps.

    Three of its steps take time quadratic in a run of server-controlled input, and the regex
    ones hold the GIL while they run, so a worker thread doesn't shield the event loop from them:

    - text outside `<pre>` is normalized with `[\t \r\n]*[\r\n][\t \r\n]*`, which restarts at
      every character of a long run of spaces that has no newline;
    - `<pre>` blocks are stripped with `[ \n]*$`, which does the same on a run of spaces that
      isn't at the very end;
    - each `<li>` in an `<ol>` is numbered by counting all of its previous siblings.

    Each formatting override produces exactly what the upstream step produces. The conversion
    hook stops once generated output or repeated scans exceed the measured work budgets.
    """

    def __init__(self, **options: Any):
        super().__init__(**options)
        self._ol_indexes: dict[int, int] = {}
        self.ordered_list_starts: dict[int, int] = {}
        self._conversion_cost = 0
        self._text_scan_cost = 0

    def get_conv_fn(self, tag_name: str) -> Callable[[Tag, str, set[str]], str]:
        upstream = _upstream_get_conv_fn(self, tag_name)

        def convert(node: Tag, text: str, parent_tags: set[str]) -> str:
            result = upstream(node, text, parent_tags) if upstream is not None else text
            self._conversion_cost += max(len(result) - len(text), 0)
            if node.name == 'li' or (node.name in ('blockquote', 'dd') and '_inline' not in parent_tags):
                self._conversion_cost += result.count('\n')
            self._text_scan_cost += len(result)
            if self._conversion_cost > _MAX_HTML_CONVERSION_COST or self._text_scan_cost > _MAX_HTML_TEXT_SCAN_COST:
                raise ModelRetry('the document is too complex')
            return result

        return convert

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
        start = self.ordered_list_starts[id(parent)]
        try:
            bullet = f'{start + self._ol_indexes[id(el)]}. '
        except ValueError:
            bullet = f'{1 + self._ol_indexes[id(el)]}. '
        bullet_indent = ' ' * len(bullet)
        if len(bullet_indent) * text.count('\n') > _MAX_HTML_CONVERSION_COST:
            raise ModelRetry('the document is too complex')

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
        allowed_domains: Only fetch from these domains (exact hostname match, ignoring case, a
            trailing dot, and IDNA spelling). Raises `ModelRetry` on violation.
        blocked_domains: Never fetch from these domains (exact hostname match, ignoring case, a
            trailing dot, and IDNA spelling). Raises `ModelRetry` on violation.
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
