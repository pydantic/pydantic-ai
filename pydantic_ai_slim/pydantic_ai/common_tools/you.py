"""You.com search tool for Pydantic AI agents.

Provides web search capabilities using the You.com API, a search engine that
delivers unified results from web and news sources.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import httpx
from typing_extensions import Any, TypedDict

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


class YouSearchResult(TypedDict, total=False):
    """A You.com search result.

    See [You.com Search API documentation](https://docs.you.com/api-reference/search/v1-search)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    description: str
    """A description or snippet of the search result."""
    snippets: list[str]
    """Text snippets from the search result, providing a preview of the content."""
    thumbnail_url: str
    """URL of the thumbnail image for the search result."""
    page_age: datetime
    """The age/publication date of the search result (ISO 8601 format)."""
    favicon_url: str
    """The URL of the favicon of the search result's domain."""
    contents: YouSearchContents
    """Contents of the page if livecrawl was enabled."""
    authors: list[str]
    """An array of authors of the search result."""


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

    country: _Country | str | None = None
    """Default country code for geographic focus of results."""

    language: _Language | str | None = None
    """Default language of results (BCP 47 format). Default: 'EN'."""

    safesearch: SafeSearch | str | None = None
    """Default safe search filter: 'off', 'moderate', or 'strict'. Defaults to 'moderate'."""

    livecrawl: LiveCrawl | str | None = None
    """Default sections to livecrawl: 'web', 'news', or 'all'."""

    livecrawl_formats: LiveCrawlFormats | str | None = None
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
        safesearch: SafeSearch | str | None,
        livecrawl: LiveCrawl | str | None,
        livecrawl_formats: LiveCrawlFormats | str | None,
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
        response_results_any = payload.get('results')
        if not isinstance(response_results_any, dict):
            return []
        response_results = cast(dict[str, Any], response_results_any)

        results: list[YouSearchResult] = []
        results.extend(self._parse_web_results(response_results.get('web')))
        results.extend(self._parse_news_results(response_results.get('news')))
        return results

    def _parse_web_results(self, web_results: Any) -> list[YouSearchResult]:
        if not isinstance(web_results, list):
            return []
        web_results_list = cast(list[object], web_results)

        parsed_results: list[YouSearchResult] = []
        for result in web_results_list:
            if not isinstance(result, dict):
                continue
            result_dict = cast(dict[str, Any], result)
            search_result = self._build_result(result_dict)
            snippets = self._string_list(result_dict.get('snippets'))
            if snippets:
                search_result['snippets'] = snippets
            favicon_url = result_dict.get('favicon_url')
            if isinstance(favicon_url, str) and favicon_url:
                search_result['favicon_url'] = favicon_url
            contents = result_dict.get('contents')
            if isinstance(contents, dict):
                built_contents = self._build_contents(cast(dict[str, Any], contents))
                if built_contents:
                    search_result['contents'] = built_contents
            authors = self._string_list(result_dict.get('authors'))
            if authors:
                search_result['authors'] = authors
            parsed_results.append(search_result)
        return parsed_results

    def _parse_news_results(self, news_results: Any) -> list[YouSearchResult]:
        if not isinstance(news_results, list):
            return []
        news_results_list = cast(list[object], news_results)

        parsed_results: list[YouSearchResult] = []
        for result in news_results_list:
            if not isinstance(result, dict):
                continue
            result_dict = cast(dict[str, Any], result)
            search_result = self._build_result(result_dict)
            contents = result_dict.get('contents')
            if isinstance(contents, dict):
                built_contents = self._build_contents(cast(dict[str, Any], contents))
                if built_contents:
                    search_result['contents'] = built_contents
            parsed_results.append(search_result)
        return parsed_results

    @staticmethod
    def _normalize_param(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        enum_value = getattr(value, 'value', None)
        if isinstance(enum_value, str):
            return enum_value
        return str(value)

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        normalized = value.replace('Z', '+00:00')
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in cast(list[object], value) if isinstance(item, str)]

    def _build_result(self, result: dict[str, Any]) -> YouSearchResult:
        """Build a YouSearchResult, including only fields with values."""
        search_result: YouSearchResult = {}
        title = result.get('title')
        if isinstance(title, str) and title:
            search_result['title'] = title
        url = result.get('url')
        if isinstance(url, str) and url:
            search_result['url'] = url
        description = result.get('description')
        if isinstance(description, str) and description:
            search_result['description'] = description
        thumbnail_url = result.get('thumbnail_url')
        if isinstance(thumbnail_url, str) and thumbnail_url:
            search_result['thumbnail_url'] = thumbnail_url
        page_age = result.get('page_age')
        if isinstance(page_age, str):
            parsed_page_age = self._parse_datetime(page_age)
            if parsed_page_age is not None:
                search_result['page_age'] = parsed_page_age
        return search_result

    def _build_contents(self, contents: dict[str, Any]) -> YouSearchContents:
        """Build YouSearchContents, including only fields with values."""
        result: YouSearchContents = {}
        html = contents.get('html')
        if isinstance(html, str) and html:
            result['html'] = html
        markdown = contents.get('markdown')
        if isinstance(markdown, str) and markdown:
            result['markdown'] = markdown
        return result


def you_search_tool(
    api_key: str,
    *,
    http_client: httpx.AsyncClient | None = None,
    count: int | None = None,
    offset: int | None = None,
    freshness: Freshness | None = None,
    country: _Country | str | None = None,
    language: _Language | str | None = None,
    safesearch: SafeSearch | str | None = None,
    livecrawl: LiveCrawl | str | None = None,
    livecrawl_formats: LiveCrawlFormats | str | None = None,
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
