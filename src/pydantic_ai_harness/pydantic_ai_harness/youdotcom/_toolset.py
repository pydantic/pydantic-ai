"""You.com toolset -- web search and page retrieval backed by the You.com API."""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Concatenate, Literal, ParamSpec, Protocol, TypedDict, TypeVar

import httpx
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness._output import truncate_head

try:
    from youdotcom import You, models
    from youdotcom.errors import NoResponseError, YouError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'youdotcom is required for the You.com capabilities. '
        'Install it with: pip install "pydantic-ai-harness[youdotcom]"'
    ) from _import_error

YOU_MAX_NUM_RESULTS = 20
"""Largest `num_results` requested per `web_search` call."""

DEFAULT_SEARCH_TIMEOUT_MS = 60_000
"""Default per-request timeout for the search and contents endpoints, in milliseconds.

The SDK inherits httpx's 5 second default otherwise, which is short for a
crawl-backed search. Search and contents themselves are fast; the headroom
covers full-page extraction.
"""

_FRESHNESS_KEYWORDS = frozenset({'day', 'week', 'month', 'year'})
# ASCII digits only, and `fullmatch` (below) anchors the whole string so a trailing
# newline or extra characters do not slip through.
_FRESHNESS_RANGE = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2})to([0-9]{4}-[0-9]{2}-[0-9]{2})')

# 401/402/403 are authentication, billing, or authorization states the model
# cannot correct, so they propagate and abort the run. Everything else the SDK
# raises (422, 429, 5xx, transport errors) is something a model can recover from
# by rephrasing or retrying, so it becomes a `ModelRetry`.
_PROPAGATE_STATUS = frozenset({401, 402, 403})

ExtractionModeName = Literal['highlights', 'full_page']
"""How `web_search` attaches page content: query-relevant excerpts or full markdown."""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')


class YouSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def source_list(sources: Mapping[str, str | None]) -> list[YouSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{'url': url, 'title': title} for url, title in sources.items()]


def _is_valid_freshness_range(freshness: str) -> bool:
    """True when `freshness` is a well-formed, correctly ordered `YYYY-MM-DDtoYYYY-MM-DD` range."""
    match = _FRESHNESS_RANGE.fullmatch(freshness)
    if match is None:
        return False
    try:
        start = datetime.strptime(match.group(1), '%Y-%m-%d')
        end = datetime.strptime(match.group(2), '%Y-%m-%d')
    except ValueError:
        return False
    return start <= end


def validate_freshness(freshness: str | None) -> None:
    """Reject a freshness value the You.com API would not accept.

    Accepts `None`, one of `day`/`week`/`month`/`year`, or a
    `YYYY-MM-DDtoYYYY-MM-DD` date range whose endpoints are real dates in
    order. Malformed ranges (non-ASCII digits, impossible or reversed dates,
    trailing characters) are rejected here rather than forwarded to You.com.
    """
    if freshness is None or freshness in _FRESHNESS_KEYWORDS or _is_valid_freshness_range(freshness):
        return
    raise ValueError(
        f'freshness must be one of {sorted(_FRESHNESS_KEYWORDS)} or a YYYY-MM-DDtoYYYY-MM-DD range, got {freshness!r}'
    )


class YouClient(Protocol):
    """The subset of the `youdotcom.You` async API that the You.com toolsets call.

    Any object with these methods can back the toolsets. Pass one via the
    capability's `client` field to configure authentication explicitly, point
    at a different host, or substitute a fake in tests. The signatures mirror
    `You`'s own keyword-only methods, so a real `You` instance satisfies the
    protocol as-is.
    """

    async def search_async(
        self,
        *,
        query: str,
        count: int | None = None,
        freshness: str | None = None,
        country: str | None = None,
        extraction: models.Extraction | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.SearchResponse:
        """Search the web and news, optionally attaching page content to each result."""
        ...  # pragma: no cover

    async def contents_async(
        self,
        *,
        urls: Sequence[str] | None = None,
        formats: Sequence[models.ContentsFormats] | None = None,
    ) -> list[models.ContentsResponse]:
        """Retrieve clean HTML or Markdown for a list of URLs."""
        ...  # pragma: no cover

    async def answer_async(
        self,
        *,
        query: str,
        freshness: str | None = None,
        country: str | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.AnswerResponse:
        """Return a synthesized answer with citations, grounded in live web results."""
        ...  # pragma: no cover

    async def research_async(
        self,
        *,
        input: str,
        research_effort: models.ResearchEffort | None = None,
        background: bool | None = None,
        source_control: models.SourceControl | None = None,
        output_schema: Mapping[str, object] | None = None,
    ) -> models.ResearchResult:
        """Run multi-step research and return a cited, synthesized answer."""
        ...  # pragma: no cover

    async def finance_research_async(
        self,
        *,
        input: str,
        research_effort: models.FinanceResearchEffort | None = None,
    ) -> models.FinanceResearchResponse:
        """Run finance-tuned research and return a cited, synthesized answer."""
        ...  # pragma: no cover


def default_client(timeout_ms: int) -> YouClient:
    """Build a `youdotcom.You` client from the `YDC_API_KEY` environment variable."""
    if not (os.environ.get('YDC_API_KEY') or os.environ.get('YOU_API_KEY_AUTH')):
        raise UserError(
            'The You.com capabilities need an API key: set the YDC_API_KEY environment variable, '
            'or pass a configured client, e.g. YouSearch(client=You(api_key_auth=...)).'
        )
    return You(api_key_auth=None, timeout_ms=timeout_ms)


def recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient You.com API failures into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the run. The SDK raises `YouError`
    subclasses that carry a `status_code`, plus `NoResponseError` and
    `httpx.HTTPError` for transport failures. Rate limits, transient 5xx, and
    rejected parameters are recoverable, so they become retries; 401/402/403
    (auth, billing, authorization) are configuration states that propagate,
    whether they surface as a `YouError` or a raw `httpx.HTTPStatusError`.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except httpx.HTTPStatusError as error:
            # A response-bearing status error may carry an auth/billing/authorization status,
            # so inspect it before the broad `httpx.HTTPError` handler wraps it as a retry.
            if error.response.status_code in _PROPAGATE_STATUS:
                raise
            raise ModelRetry(f'You.com request failed: {error}') from error
        except (httpx.HTTPError, NoResponseError) as error:
            raise ModelRetry(f'You.com request failed: {error}') from error
        except YouError as error:
            status = getattr(error, 'status_code', None)
            if isinstance(status, int) and status in _PROPAGATE_STATUS:
                raise
            raise ModelRetry(f'You.com request failed: {error}') from error

    return wrapper


def _format_result(title: str | None, url: str, page_age: datetime | None, body: str | None) -> str:
    """Render one result as labelled metadata lines followed by its body text."""
    lines = [f'Title: {title or "(untitled)"}', f'URL: {url}']
    if page_age is not None:
        lines.append(f'Published: {page_age.date().isoformat()}')
    if body:
        lines.extend(['', body])
    return '\n'.join(lines)


def _web_result_body(result: models.WebResult, max_text_chars: int, extraction_mode: ExtractionModeName) -> str | None:
    """Pick the most useful body for a web result.

    The configured extraction mode picks the preferred body -- full markdown in
    `'full_page'` mode, query-relevant highlights otherwise -- and the other
    field serves as the fallback when a response carries both. Results without
    contents fall back to snippets, then the description.
    """
    contents = result.contents
    if contents is not None:
        prefer_markdown = extraction_mode == 'full_page'
        if contents.markdown and (prefer_markdown or not contents.highlights):
            return truncate_head(contents.markdown, max_text_chars)
        if contents.highlights:
            return '\n'.join(f'- {highlight}' for highlight in contents.highlights)
    if result.snippets:
        return '\n'.join(f'- {snippet}' for snippet in result.snippets)
    return result.description or None


class YouSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the You.com Search and Contents APIs.

    `web_search` surveys the web and returns results with query-relevant
    excerpts (or full page markdown, with `extraction_mode='full_page'`), and
    `get_page` retrieves the markdown of one specific URL.

    Each tool returns a `ToolReturn` whose `return_value` is the text the model
    sees and whose `metadata['sources']` lists the result URLs and titles
    (`YouSource` dicts), so applications can render citations from
    `ToolReturnPart.metadata` without parsing the text. `web_search` also
    carries the response `search_uuid` and `latency` in metadata for tracing.

    `get_page` and full-page `web_search` text are capped at `max_text_chars`
    characters. Bounds are validated by `YouSearch` at construction.
    """

    def __init__(
        self,
        *,
        client: YouClient | None,
        num_results: int,
        extraction_mode: ExtractionModeName,
        max_text_chars: int,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        boost_domains: list[str] | None = None,
        freshness: str | None = None,
        country: str | None = None,
        timeout_ms: int = DEFAULT_SEARCH_TIMEOUT_MS,
    ) -> None:
        super().__init__()
        self._client = client if client is not None else default_client(timeout_ms)
        self._num_results = num_results
        self._extraction_mode: ExtractionModeName = extraction_mode
        self._max_text_chars = max_text_chars
        self._include_domains = list(include_domains) if include_domains else None
        self._exclude_domains = list(exclude_domains) if exclude_domains else None
        self._boost_domains = list(boost_domains) if boost_domains else None
        self._freshness = freshness
        self._country = country
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')

    def _extraction(self) -> models.Extraction:
        """Build the extraction payload for the configured mode."""
        if self._extraction_mode == 'full_page':
            return models.Extraction(
                extraction_mode=models.ExtractionMode.FULL_PAGE,
                full_page=models.ExtractionFullPage(extraction_formats=[models.ExtractionFormat.MARKDOWN]),
            )
        return models.Extraction(extraction_mode=models.ExtractionMode.HIGHLIGHTS)

    @recoverable
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with its most relevant excerpts.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and excerpts.
        """
        response = await self._client.search_async(
            query=query,
            count=self._num_results,
            freshness=self._freshness,
            country=self._country,
            extraction=self._extraction(),
            include_domains=self._include_domains,
            exclude_domains=self._exclude_domains,
            boost_domains=self._boost_domains,
        )
        web = response.results.web if response.results is not None else None
        rows: list[tuple[str, str | None, datetime | None, str | None]] = []
        for result in web or []:
            if not result.url:
                continue
            body = _web_result_body(result, self._max_text_chars, self._extraction_mode)
            rows.append((result.url, result.title, result.page_age, body))
            if len(rows) >= self._num_results:
                break
        metadata = self._search_metadata(response)
        if not rows:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': [], **metadata})
        sources = source_list({url: title for url, title, _, _ in rows})
        sections = [_format_result(title, url, page_age, body) for url, title, page_age, body in rows]
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        found = f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}'
        return ToolReturn(found, metadata={'sources': sources, **metadata})

    @staticmethod
    def _search_metadata(response: models.SearchResponse) -> dict[str, str | float | None]:
        """Response `search_uuid` and `latency` for tracing, never shown to the model."""
        meta = response.metadata
        if meta is None:
            return {'search_uuid': None, 'latency': None}
        return {'search_uuid': meta.search_uuid, 'latency': meta.latency}

    @recoverable
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve the markdown of a specific URL.

        Use it to read a promising URL from `web_search` results in full, or a
        URL the user provided.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's title, URL, and markdown content.
        """
        responses = await self._client.contents_async(
            urls=[url],
            formats=[models.ContentsFormats.MARKDOWN, models.ContentsFormats.METADATA],
        )
        first = responses[0] if responses else None
        markdown = first.markdown if first is not None and isinstance(first.markdown, str) else None
        if first is None or not markdown:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        page_url = first.url or url
        title = first.title
        metadata = first.metadata
        if not title and isinstance(metadata, models.ContentsMetadata) and isinstance(metadata.site_name, str):
            title = metadata.site_name
        return ToolReturn(
            _format_result(title, page_url, None, truncate_head(markdown, self._max_text_chars)),
            metadata={'sources': source_list({page_url: title})},
        )
