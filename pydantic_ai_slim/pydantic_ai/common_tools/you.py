"""You.com search tool for Pydantic AI agents.

Provides web search capabilities using the You.com API, a search engine that
delivers unified results from web and news sources.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import httpx
from pydantic import BaseModel
from typing_extensions import Any, NotRequired, TypedDict

from pydantic_ai.models import cached_async_http_client
from pydantic_ai.tools import Tool

__all__ = ('you_search_tool',)

_YOU_SEARCH_URL = 'https://ydc-index.io/v1/search'

_Country = Literal[
    'AR',
    'AU',
    'AT',
    'BE',
    'BR',
    'CA',
    'CL',
    'DK',
    'FI',
    'FR',
    'DE',
    'HK',
    'IN',
    'ID',
    'IT',
    'JP',
    'KR',
    'MY',
    'MX',
    'NL',
    'NZ',
    'NO',
    'CN',
    'PL',
    'PT',
    'PH',
    'RU',
    'SA',
    'ZA',
    'ES',
    'SE',
    'CH',
    'TW',
    'TR',
    'GB',
    'US',
]

_Language = Literal[
    'AR',
    'EU',
    'BN',
    'BG',
    'CA',
    'ZH-HANS',
    'ZH-HANT',
    'HR',
    'CS',
    'DA',
    'NL',
    'EN',
    'EN-GB',
    'ET',
    'FI',
    'FR',
    'GL',
    'DE',
    'EL',
    'GU',
    'HE',
    'HI',
    'HU',
    'IS',
    'IT',
    'JP',
    'KN',
    'KO',
    'LV',
    'LT',
    'MS',
    'ML',
    'MR',
    'NB',
    'PL',
    'PT-BR',
    'PT-PT',
    'PA',
    'RO',
    'RU',
    'SR',
    'SK',
    'SL',
    'ES',
    'SV',
    'TA',
    'TE',
    'TH',
    'TR',
    'UK',
    'VI',
]

Freshness = Literal['day', 'week', 'month', 'year'] | str
SafeSearch = Literal['off', 'moderate', 'strict']
LiveCrawl = Literal['web', 'news', 'all']
LiveCrawlFormats = Literal['html', 'markdown']


class YouSearchContents(TypedDict, total=False):
    """Contents of a page when livecrawl is enabled."""

    html: str
    """The HTML content of the page."""
    markdown: str
    """The Markdown content of the page."""


class YouSearchResult(TypedDict):
    """A You.com search result.

    See [You.com Search API documentation](https://docs.you.com/api-reference/search/v1-search)
    for more information.

    `title` and `url` are always present. All other fields are optional.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    description: NotRequired[str]
    """A description or snippet of the search result."""
    snippets: NotRequired[list[str]]
    """Text snippets from the search result, providing a preview of the content."""
    thumbnail_url: NotRequired[str]
    """URL of the thumbnail image for the search result."""
    page_age: NotRequired[datetime]
    """The age/publication date of the search result (ISO 8601 format)."""
    favicon_url: NotRequired[str]
    """The URL of the favicon of the search result's domain."""
    contents: NotRequired[YouSearchContents]
    """Contents of the page if livecrawl was enabled."""
    authors: NotRequired[list[str]]
    """An array of authors of the search result."""


class _RawContents(BaseModel):
    html: str | None = None
    markdown: str | None = None

    def to_contents(self) -> YouSearchContents | None:
        if not self.html and not self.markdown:
            return None
        result: YouSearchContents = {}
        if self.html:
            result['html'] = self.html
        if self.markdown:
            result['markdown'] = self.markdown
        return result


class _RawResult(BaseModel):
    """Shared fields present in both web and news results."""

    title: str
    url: str
    description: str | None = None
    thumbnail_url: str | None = None
    page_age: datetime | None = None
    contents: _RawContents | None = None

    def to_result(self) -> YouSearchResult:
        result: dict[str, Any] = {'title': self.title, 'url': self.url}
        if self.description:
            result['description'] = self.description
        if self.thumbnail_url:
            result['thumbnail_url'] = self.thumbnail_url
        if self.page_age is not None:
            result['page_age'] = self.page_age
        if self.contents is not None:
            contents = self.contents.to_contents()
            if contents:
                result['contents'] = contents
        return cast(YouSearchResult, result)


class _RawWebResult(_RawResult):
    """Web result with additional fields not present in news results."""

    snippets: list[str] | None = None
    favicon_url: str | None = None
    authors: list[str] | None = None

    def to_result(self) -> YouSearchResult:
        result = super().to_result()
        if self.snippets:
            result['snippets'] = self.snippets
        if self.favicon_url:
            result['favicon_url'] = self.favicon_url
        if self.authors:
            result['authors'] = self.authors
        return result


class _RawSearchResponse(BaseModel):
    web: list[_RawWebResult] | None = None
    news: list[_RawResult] | None = None


@dataclass
class YouSearchTool:
    """The You.com search tool."""

    api_key: str
    """The You.com API key."""

    http_client: httpx.AsyncClient | None = None
    """HTTP client for API requests. If `None`, a shared cached client is used."""

    count: int | None = None
    """Default maximum number of results per section (web/news). Range: 1-100. API default is 10."""

    offset: int | None = None
    """Pagination offset (0-9). Calculated in multiples of count. Not controllable by LLM."""

    freshness: Freshness | None = None
    """Default freshness of results: 'day', 'week', 'month', 'year', or date range 'YYYY-MM-DDtoYYYY-MM-DD'."""

    country: _Country | None = None
    """Default country code for geographic focus of results."""

    language: _Language | None = None
    """Default language of results (BCP 47 format). Default: 'EN'."""

    safesearch: SafeSearch | None = None
    """Default safe search filter: 'off', 'moderate', or 'strict'. Defaults to 'moderate'."""

    livecrawl: LiveCrawl | None = None
    """Default sections to livecrawl: 'web', 'news', or 'all'."""

    livecrawl_formats: LiveCrawlFormats | None = None
    """Default format for livecrawled content: 'html' or 'markdown'."""

    async def __call__(
        self,
        query: str,
        count: int | None = None,
        freshness: Freshness | None = None,
        country: str | None = None,
        language: str | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
    ) -> list[YouSearchResult]:
        """Searches You.com for the given query and returns the results.

        Args:
            query: The search query to execute with You.com.
            count: Maximum number of results per section (1-100). Only used if not configured at tool creation.
            freshness: Result freshness: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'.
                Only used if not configured at tool creation.
            country: ISO 3166-1 alpha-2 country code for geographic focus (e.g. 'US', 'GB', 'DE').
                Only used if not configured at tool creation.
            language: BCP 47 language code for results (e.g. 'EN', 'FR', 'ZH-HANS').
                Only used if not configured at tool creation.
            safesearch: Content moderation filter: 'off', 'moderate', or 'strict'.
                Only used if not configured at tool creation.
            livecrawl: Sections to livecrawl for full page content: 'web', 'news', or 'all'.
                Only used if not configured at tool creation.
            livecrawl_formats: Format for livecrawled content: 'html' or 'markdown'.
                Only used if not configured at tool creation.

        Returns:
            A list of search results from You.com.
        """
        params = self._build_params(
            query=query,
            count=count,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
        )
        payload = await self._request(params)
        return self._parse_results(payload)

    def _build_params(
        self,
        *,
        query: str,
        count: int | None,
        freshness: Freshness | None,
        country: _Country | str | None,
        language: _Language | str | None,
        safesearch: SafeSearch | None,
        livecrawl: LiveCrawl | None,
        livecrawl_formats: LiveCrawlFormats | None,
    ) -> dict[str, str | int]:
        # Use configured defaults when set, otherwise allow LLM override.
        params: dict[str, str | int] = {'query': query}

        effective_count = self.count if self.count is not None else count
        if effective_count is not None:
            params['count'] = effective_count
        if self.offset is not None:
            params['offset'] = self.offset

        effective_values: tuple[tuple[str, object | None], ...] = (
            ('freshness', self.freshness if self.freshness is not None else freshness),
            ('country', self.country if self.country is not None else country),
            ('language', self.language if self.language is not None else language),
            ('safesearch', self.safesearch if self.safesearch is not None else safesearch),
            ('livecrawl', self.livecrawl if self.livecrawl is not None else livecrawl),
            (
                'livecrawl_formats',
                self.livecrawl_formats if self.livecrawl_formats is not None else livecrawl_formats,
            ),
        )
        for key, value in effective_values:
            normalized = self._normalize_param(value)
            if normalized is not None:
                params[key] = normalized
        return params

    async def _request(self, params: dict[str, str | int]) -> dict[str, Any]:
        client = self.http_client or cached_async_http_client(provider='you')
        response = await client.get(
            _YOU_SEARCH_URL,
            params=params,
            headers={'X-API-Key': self.api_key},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        return {}

    def _parse_results(self, payload: dict[str, Any]) -> list[YouSearchResult]:
        response = _RawSearchResponse.model_validate(payload.get('results', {}))
        results: list[YouSearchResult] = []
        for item in response.web or []:
            results.append(item.to_result())
        for item in response.news or []:
            results.append(item.to_result())
        return results

    @staticmethod
    def _normalize_param(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)


def you_search_tool(
    api_key: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    count: int | None = None,
    offset: int | None = None,
    freshness: Freshness | None = None,
    country: _Country | None = None,
    language: _Language | None = None,
    safesearch: SafeSearch | None = None,
    livecrawl: LiveCrawl | None = None,
    livecrawl_formats: LiveCrawlFormats | None = None,
) -> Tool[Any]:
    """Creates a You.com search tool.

    Args:
        api_key: You.com API key. [Get one here](https://you.com/platform/api-keys).
        http_client: An existing `httpx.AsyncClient` to use.
            If not provided, a shared cached client is used.
        count: Default maximum number of results per section (1-100). Default: 10.
        offset: Pagination offset (0-9). Not controllable by LLM.
        freshness: Default result freshness: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'.
        country: Default country code for geographic focus of results.
        language: Default language of results (BCP 47 format). Default: 'EN'.
        safesearch: Default safe search filter: 'off', 'moderate', or 'strict'.
        livecrawl: Default sections to livecrawl: 'web', 'news', or 'all'.
        livecrawl_formats: Default format for livecrawled content: 'html' or 'markdown'.

    Returns:
        A [`Tool`][pydantic_ai.tools.Tool] that searches the web and news using You.com and can be used with Pydantic AI agents.
    """
    return Tool[Any](
        YouSearchTool(
            api_key=api_key,
            http_client=http_client,
            count=count,
            offset=offset,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
        ).__call__,
        name='you_search',
        description='Searches the web and news using the You.com Search API and returns the results.',
    )
