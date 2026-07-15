from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal

import httpx
from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import Tool

__all__ = (
    'BraveImageSearchResponse',
    'BraveLLMContextResponse',
    'BraveLocalDescriptionsResponse',
    'BraveLocalPOIsResponse',
    'BraveNewsSearchResponse',
    'BravePlaceSearchResponse',
    'BraveRichSearchResponse',
    'BraveSearchToolset',
    'BraveVideoSearchResponse',
    'BraveWebSearchResponse',
    'brave_image_search_tool',
    'brave_llm_context_tool',
    'brave_local_descriptions_tool',
    'brave_local_pois_tool',
    'brave_news_search_tool',
    'brave_place_search_tool',
    'brave_rich_search_tool',
    'brave_video_search_tool',
    'brave_web_search_tool',
)

_BASE_URL = 'https://api.search.brave.com/res/v1'
_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from `None` in factory kwargs."""

_HttpMethod = Literal['GET', 'POST']
_SafeSearch = Literal['off', 'moderate', 'strict']
_ImageSafeSearch = Literal['off', 'strict']
_Units = Literal['metric', 'imperial']
_ContextThresholdMode = Literal['disabled', 'strict', 'balanced', 'lenient']
_ParamValue = str | int | float | bool | list[str] | None
_CleanParamValue = str | int | float | bool | list[str]


class BraveQuery(TypedDict, total=False):
    """Brave query metadata."""

    original: str
    altered: str
    cleaned: str
    spellcheck_off: bool
    more_results_available: bool
    show_strict_warning: bool
    search_operators: BraveSearchOperators


class BraveSearchOperators(TypedDict, total=False):
    """Search operators applied to a query."""

    applied: bool
    cleaned_query: str
    sites: list[str]


class BraveMetaURL(TypedDict, total=False):
    """Metadata for a result URL."""

    scheme: str
    netloc: str
    hostname: str
    favicon: str
    path: str


class BraveThumbnail(TypedDict, total=False):
    """Thumbnail metadata."""

    src: str
    original: str
    width: int
    height: int
    logo: bool


class BraveProfile(TypedDict, total=False):
    """Publisher or author profile metadata."""

    name: str
    url: str
    long_name: str
    img: str


class BraveMixedResultReference(TypedDict, total=False):
    """Reference to a result in a mixed Brave result set."""

    type: str
    index: int
    all: bool


class BraveMixedResults(TypedDict, total=False):
    """Preferred display order for mixed Brave results."""

    type: str
    main: list[BraveMixedResultReference]
    top: list[BraveMixedResultReference]
    side: list[BraveMixedResultReference]


class BraveWebResult(TypedDict, total=False):
    """A Brave web search result."""

    type: str
    title: str
    url: str
    description: str
    age: str
    language: str
    meta_url: BraveMetaURL
    thumbnail: BraveThumbnail
    profile: BraveProfile
    page_age: str
    extra_snippets: list[str]
    fetched_content_timestamp: int
    deep_results: dict[str, Any]
    schemas: list[Any]
    product: dict[str, Any]
    recipe: dict[str, Any]
    article: dict[str, Any]
    book: dict[str, Any]
    software: dict[str, Any]
    rating: dict[str, Any]
    faq: dict[str, Any]
    movie: dict[str, Any]
    video: dict[str, Any]
    location: dict[str, Any]
    qa: dict[str, Any]
    creative_work: dict[str, Any]
    music_recording: dict[str, Any]
    organization: dict[str, Any]
    review: dict[str, Any]
    content_type: str


class BraveResultGroup(TypedDict, total=False):
    """A grouped set of Brave results."""

    type: str
    results: list[dict[str, Any]]
    mutated_by_goggles: bool
    family_friendly: bool


class BraveWebResultGroup(TypedDict, total=False):
    """The `web` result group in a Brave web search response."""

    type: str
    results: list[BraveWebResult]
    mutated_by_goggles: bool
    family_friendly: bool


class BraveRichHint(TypedDict, total=False):
    """Hint for retrieving rich real-time data."""

    vertical: str
    callback_key: str


class BraveRichHintContainer(TypedDict, total=False):
    """Container for a rich result hint."""

    type: str
    hint: BraveRichHint


class BraveWebSearchResponse(TypedDict, total=False):
    """Response from Brave Web Search.

    See [Brave Web Search API documentation](https://api-dashboard.search.brave.com/api-reference/web/search/get)
    for more information.
    """

    type: Literal['search']
    query: BraveQuery
    web: BraveWebResultGroup
    mixed: BraveMixedResults
    discussions: BraveResultGroup
    faq: BraveResultGroup
    infobox: BraveResultGroup
    locations: BraveResultGroup
    news: BraveResultGroup
    videos: BraveResultGroup
    rich: BraveRichHintContainer


class BraveNewsResult(TypedDict, total=False):
    """A Brave news search result."""

    type: str
    title: str
    url: str
    description: str
    age: str
    page_age: str
    page_fetched: str
    fetched_content_timestamp: int
    meta_url: BraveMetaURL
    thumbnail: BraveThumbnail
    extra_snippets: list[str]


class BraveNewsSearchResponse(TypedDict, total=False):
    """Response from Brave News Search."""

    type: Literal['news']
    query: BraveQuery
    results: list[BraveNewsResult]


class BraveImageProperties(TypedDict, total=False):
    """Image properties returned by Brave Image Search."""

    url: str
    placeholder: str
    width: int
    height: int


class BraveImageResult(TypedDict, total=False):
    """A Brave image search result."""

    type: str
    title: str
    url: str
    source: str
    page_fetched: str
    thumbnail: BraveThumbnail
    properties: BraveImageProperties
    meta_url: BraveMetaURL
    confidence: str


class BraveOffensiveResultsMetadata(TypedDict, total=False):
    """Metadata about whether results may be offensive."""

    might_be_offensive: bool


class BraveImageSearchResponse(TypedDict, total=False):
    """Response from Brave Image Search."""

    type: Literal['images']
    query: BraveQuery
    results: list[BraveImageResult]
    extra: BraveOffensiveResultsMetadata


class BraveVideoAuthor(TypedDict, total=False):
    """Video author metadata."""

    name: str
    url: str
    long_name: str
    img: str


class BraveVideoMetadata(TypedDict, total=False):
    """Video metadata returned by Brave Video Search."""

    duration: str
    views: int
    creator: str
    publisher: str
    requires_subscription: bool
    tags: list[str]
    author: BraveVideoAuthor


class BraveVideoResult(TypedDict, total=False):
    """A Brave video search result."""

    type: str
    title: str
    url: str
    description: str
    age: str
    page_age: str
    page_fetched: str
    fetched_content_timestamp: int
    thumbnail: BraveThumbnail
    video: BraveVideoMetadata
    meta_url: BraveMetaURL


class BraveVideoSearchResponse(TypedDict, total=False):
    """Response from Brave Video Search."""

    type: Literal['videos']
    query: BraveQuery
    extra: BraveOffensiveResultsMetadata
    results: list[BraveVideoResult]


class BraveGroundingItem(TypedDict, total=False):
    """A grounding item returned by Brave LLM Context."""

    name: str
    url: str
    title: str
    snippets: list[str]


class BraveLLMContextGrounding(TypedDict, total=False):
    """Grounding content returned by Brave LLM Context."""

    generic: list[BraveGroundingItem]
    poi: BraveGroundingItem | None
    map: list[BraveGroundingItem]


class BraveLLMContextSource(TypedDict, total=False):
    """Source metadata returned by Brave LLM Context."""

    title: str
    hostname: str
    age: list[str] | None
    site_name: str
    favicon: str
    thumbnail: BraveThumbnail


class BraveLLMContextResponse(TypedDict, total=False):
    """Response from Brave LLM Context."""

    grounding: BraveLLMContextGrounding
    sources: dict[str, BraveLLMContextSource]


class BravePostalAddress(TypedDict, total=False):
    """Postal address for a Brave location result."""

    type: str
    displayAddress: str
    streetAddress: str
    addressLocality: str
    addressRegion: str
    postalCode: str
    country: str


class BraveContact(TypedDict, total=False):
    """Contact details for a Brave location result."""

    telephone: str
    email: str


class BraveRating(TypedDict, total=False):
    """Rating details for a Brave location result."""

    ratingValue: float
    bestRating: float
    reviewCount: int
    profile: BraveProfile


class BraveOpeningHoursPeriod(TypedDict, total=False):
    """Opening-hours period for a Brave location result."""

    abbr_name: str
    full_name: str
    opens: str
    closes: str


class BraveOpeningHours(TypedDict, total=False):
    """Opening hours for a Brave location result."""

    current_day: list[BraveOpeningHoursPeriod]
    days: list[list[BraveOpeningHoursPeriod]]


class BraveDistance(TypedDict, total=False):
    """Distance metadata for a Brave location result."""

    value: float
    units: str


class BraveLocationReviews(TypedDict, total=False):
    """Review metadata for a Brave location result."""

    reviews_in_foreign_language: bool


class BraveLocationPictures(TypedDict, total=False):
    """Picture metadata for a Brave location result."""

    results: list[dict[str, Any]]


class BraveLocationAction(TypedDict, total=False):
    """Action metadata for a Brave location result."""

    type: str
    url: str


class BraveLocationResult(TypedDict, total=False):
    """A Brave location result."""

    type: str
    title: str
    url: str
    provider_url: str
    id: str
    description: str
    postal_address: BravePostalAddress
    contact: BraveContact
    rating: BraveRating
    opening_hours: BraveOpeningHours
    coordinates: tuple[float, float] | list[float]
    distance: BraveDistance
    categories: list[str]
    price_range: str
    serves_cuisine: list[str]
    thumbnail: BraveThumbnail
    profiles: list[BraveProfile]
    reviews: BraveLocationReviews
    pictures: BraveLocationPictures
    action: BraveLocationAction
    results: list[dict[str, Any]]
    timezone: str
    timezone_offset: int


class BravePlaceSearchResponse(TypedDict, total=False):
    """Response from Brave Place Search."""

    type: Literal['locations']
    query: BraveQuery
    results: list[BraveLocationResult]
    cities: list[dict[str, Any]]
    addresses: list[dict[str, Any]]
    streets: list[dict[str, Any]]
    mixed: list[dict[str, Any]]
    location: dict[str, Any]


class BraveLocalPOIsResponse(TypedDict, total=False):
    """Response from Brave Local POIs."""

    type: Literal['local_pois']
    results: list[BraveLocationResult]


class BraveLocalDescription(TypedDict, total=False):
    """A Brave local POI description."""

    type: str
    id: str
    description: str | None


class BraveLocalDescriptionsResponse(TypedDict, total=False):
    """Response from Brave Local Descriptions."""

    type: Literal['local_descriptions']
    results: list[BraveLocalDescription | None]


class BraveRichSearchResponse(TypedDict, total=False):
    """Response from Brave Rich Search."""

    type: Literal['rich']
    results: list[dict[str, Any]]
    response_callback_info: dict[str, Any]


brave_web_search_ta = TypeAdapter(BraveWebSearchResponse)
brave_news_search_ta = TypeAdapter(BraveNewsSearchResponse)
brave_image_search_ta = TypeAdapter(BraveImageSearchResponse)
brave_video_search_ta = TypeAdapter(BraveVideoSearchResponse)
brave_llm_context_ta = TypeAdapter(BraveLLMContextResponse)
brave_place_search_ta = TypeAdapter(BravePlaceSearchResponse)
brave_local_pois_ta = TypeAdapter(BraveLocalPOIsResponse)
brave_local_descriptions_ta = TypeAdapter(BraveLocalDescriptionsResponse)
brave_rich_search_ta = TypeAdapter(BraveRichSearchResponse)


@dataclass
class _BraveAPIClient:
    client: httpx.AsyncClient
    api_key: str

    _: KW_ONLY

    base_url: str = _BASE_URL

    async def request(
        self,
        endpoint: str,
        *,
        method: _HttpMethod,
        params: Mapping[str, _ParamValue],
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        request_headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.api_key,
        }
        if headers:
            request_headers.update(headers)

        request_params = _clean_params(params)
        url = f'{self.base_url.rstrip("/")}/{endpoint.lstrip("/")}'
        if method == 'GET':
            response = await self.client.get(
                url, params=httpx.QueryParams(_query_params(request_params)), headers=request_headers
            )
        else:
            response = await self.client.post(url, json=request_params, headers=request_headers)
        response.raise_for_status()
        return response.json()


@dataclass
class BraveWebSearchTool:
    """The Brave Web Search tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    _: KW_ONLY

    method: _HttpMethod = 'GET'
    """The HTTP method to use for Brave Web Search."""

    async def __call__(
        self,
        query: str,
        country: str = 'US',
        search_lang: str = 'en',
        ui_lang: str = 'en-US',
        count: int = 20,
        offset: int = 0,
        safesearch: _SafeSearch = 'moderate',
        freshness: str | None = None,
        text_decorations: bool = True,
        spellcheck: bool = True,
        result_filter: str | None = None,
        goggles: str | list[str] | None = None,
        extra_snippets: bool | None = None,
        summary: bool | None = None,
        enable_rich_callback: bool = False,
        include_fetch_metadata: bool = False,
        operators: bool = True,
        units: _Units | None = None,
        loc_lat: float | None = None,
        loc_long: float | None = None,
        loc_timezone: str | None = None,
        loc_city: str | None = None,
        loc_state: str | None = None,
        loc_state_name: str | None = None,
        loc_country: str | None = None,
        loc_postal_code: str | None = None,
    ) -> BraveWebSearchResponse:
        """Searches Brave Web Search for the given query.

        Args:
            query: The search query to execute with Brave.
            country: The country code where search results come from.
            search_lang: The language code for search results.
            ui_lang: The UI language preferred in the response.
            count: The number of web results to return.
            offset: The page offset for pagination.
            safesearch: The adult-content filter.
            freshness: Page-age filter such as `pd`, `pw`, `pm`, `py`, or a date range.
            text_decorations: Whether to include decoration markers in result text.
            spellcheck: Whether Brave should spell-check the query.
            result_filter: Comma-separated result types to include.
            goggles: Goggle URL or inline definition for custom ranking.
            extra_snippets: Whether to include extra snippets for each result.
            summary: Whether to enable summary key generation.
            enable_rich_callback: Whether to include a rich-search callback key when relevant.
            include_fetch_metadata: Whether to include fetch metadata.
            operators: Whether to apply Brave search operators.
            units: Measurement units for rich/local results.
            loc_lat: Latitude for location-aware results.
            loc_long: Longitude for location-aware results.
            loc_timezone: IANA timezone for location-aware results.
            loc_city: City for location-aware results.
            loc_state: State or region code for location-aware results.
            loc_state_name: State or region name for location-aware results.
            loc_country: Country code for location-aware results.
            loc_postal_code: Postal code for location-aware results.

        Returns:
            The Brave Web Search response.
        """
        response = await self.client.request(
            'web/search',
            method=self.method,
            params={
                'q': query,
                'country': country,
                'search_lang': search_lang,
                'ui_lang': ui_lang,
                'count': count,
                'offset': offset,
                'safesearch': safesearch,
                'freshness': freshness,
                'text_decorations': text_decorations,
                'spellcheck': spellcheck,
                'result_filter': result_filter,
                'goggles': goggles,
                'extra_snippets': extra_snippets,
                'summary': summary,
                'enable_rich_callback': enable_rich_callback,
                'include_fetch_metadata': include_fetch_metadata,
                'operators': operators,
                'units': units,
            },
            headers=_location_headers(
                loc_lat=loc_lat,
                loc_long=loc_long,
                loc_timezone=loc_timezone,
                loc_city=loc_city,
                loc_state=loc_state,
                loc_state_name=loc_state_name,
                loc_country=loc_country,
                loc_postal_code=loc_postal_code,
            ),
        )
        return brave_web_search_ta.validate_python(response)


@dataclass
class BraveNewsSearchTool:
    """The Brave News Search tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    _: KW_ONLY

    method: _HttpMethod = 'GET'
    """The HTTP method to use for Brave News Search."""

    async def __call__(
        self,
        query: str,
        country: str = 'US',
        search_lang: str = 'en',
        ui_lang: str = 'en-US',
        count: int = 20,
        offset: int = 0,
        safesearch: _SafeSearch = 'strict',
        freshness: str | None = None,
        spellcheck: bool = True,
        extra_snippets: bool | None = None,
        goggles: str | list[str] | None = None,
        operators: bool = True,
        include_fetch_metadata: bool = False,
    ) -> BraveNewsSearchResponse:
        """Searches Brave News Search for the given query.

        Args:
            query: The news search query to execute with Brave.
            country: The country code where news results come from.
            search_lang: The language code for news results.
            ui_lang: The UI language preferred in the response.
            count: The number of news results to return.
            offset: The page offset for pagination.
            safesearch: The adult-content filter.
            freshness: Page-age filter such as `pd`, `pw`, `pm`, `py`, or a date range.
            spellcheck: Whether Brave should spell-check the query.
            extra_snippets: Whether to include extra snippets for each result.
            goggles: Goggle URL or inline definition for custom ranking.
            operators: Whether to apply Brave search operators.
            include_fetch_metadata: Whether to include fetch metadata.

        Returns:
            The Brave News Search response.
        """
        response = await self.client.request(
            'news/search',
            method=self.method,
            params={
                'q': query,
                'country': country,
                'search_lang': search_lang,
                'ui_lang': ui_lang,
                'count': count,
                'offset': offset,
                'safesearch': safesearch,
                'freshness': freshness,
                'spellcheck': spellcheck,
                'extra_snippets': extra_snippets,
                'goggles': goggles,
                'operators': operators,
                'include_fetch_metadata': include_fetch_metadata,
            },
        )
        return brave_news_search_ta.validate_python(response)


@dataclass
class BraveImageSearchTool:
    """The Brave Image Search tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    async def __call__(
        self,
        query: str,
        country: str = 'US',
        search_lang: str = 'en',
        count: int = 50,
        safesearch: _ImageSafeSearch = 'strict',
        spellcheck: bool = True,
    ) -> BraveImageSearchResponse:
        """Searches Brave Image Search for the given query.

        Args:
            query: The image search query to execute with Brave.
            country: The country code where image results come from.
            search_lang: The language code for image results.
            count: The number of image results to return.
            safesearch: The adult-content filter.
            spellcheck: Whether Brave should spell-check the query.

        Returns:
            The Brave Image Search response.
        """
        response = await self.client.request(
            'images/search',
            method='GET',
            params={
                'q': query,
                'country': country,
                'search_lang': search_lang,
                'count': count,
                'safesearch': safesearch,
                'spellcheck': spellcheck,
            },
        )
        return brave_image_search_ta.validate_python(response)


@dataclass
class BraveVideoSearchTool:
    """The Brave Video Search tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    _: KW_ONLY

    method: _HttpMethod = 'GET'
    """The HTTP method to use for Brave Video Search."""

    async def __call__(
        self,
        query: str,
        country: str = 'US',
        search_lang: str = 'en',
        ui_lang: str = 'en-US',
        count: int = 20,
        offset: int = 0,
        safesearch: _SafeSearch = 'moderate',
        freshness: str | None = None,
        spellcheck: bool = True,
        operators: bool = True,
        include_fetch_metadata: bool = False,
    ) -> BraveVideoSearchResponse:
        """Searches Brave Video Search for the given query.

        Args:
            query: The video search query to execute with Brave.
            country: The country code where video results come from.
            search_lang: The language code for video results.
            ui_lang: The UI language preferred in the response.
            count: The number of video results to return.
            offset: The page offset for pagination.
            safesearch: The adult-content filter.
            freshness: Page-age filter such as `pd`, `pw`, `pm`, `py`, or a date range.
            spellcheck: Whether Brave should spell-check the query.
            operators: Whether to apply Brave search operators.
            include_fetch_metadata: Whether to include fetch metadata.

        Returns:
            The Brave Video Search response.
        """
        response = await self.client.request(
            'videos/search',
            method=self.method,
            params={
                'q': query,
                'country': country,
                'search_lang': search_lang,
                'ui_lang': ui_lang,
                'count': count,
                'offset': offset,
                'safesearch': safesearch,
                'freshness': freshness,
                'spellcheck': spellcheck,
                'operators': operators,
                'include_fetch_metadata': include_fetch_metadata,
            },
        )
        return brave_video_search_ta.validate_python(response)


@dataclass
class BraveLLMContextTool:
    """The Brave LLM Context tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    _: KW_ONLY

    method: _HttpMethod = 'GET'
    """The HTTP method to use for Brave LLM Context."""

    async def __call__(
        self,
        query: str,
        country: str = 'US',
        search_lang: str = 'en',
        count: int = 20,
        spellcheck: bool = True,
        maximum_number_of_urls: int = 20,
        maximum_number_of_tokens: int = 8192,
        maximum_number_of_snippets: int = 50,
        context_threshold_mode: _ContextThresholdMode = 'balanced',
        maximum_number_of_tokens_per_url: int = 4096,
        maximum_number_of_snippets_per_url: int = 50,
        goggles: str | list[str] | None = None,
        freshness: str | None = None,
        enable_local: bool | None = None,
        enable_source_metadata: bool = False,
        loc_lat: float | None = None,
        loc_long: float | None = None,
        loc_city: str | None = None,
        loc_state: str | None = None,
        loc_state_name: str | None = None,
        loc_country: str | None = None,
        loc_postal_code: str | None = None,
    ) -> BraveLLMContextResponse:
        """Gets Brave LLM Context for the given query.

        Args:
            query: The search query to ground with Brave.
            country: The country code where search results come from.
            search_lang: The language code for search results.
            count: The maximum number of search results considered for context.
            spellcheck: Whether Brave should spell-check the query.
            maximum_number_of_urls: Maximum number of URLs to include.
            maximum_number_of_tokens: Approximate maximum tokens in the context.
            maximum_number_of_snippets: Maximum number of snippets in the context.
            context_threshold_mode: Relevance threshold mode for context inclusion.
            maximum_number_of_tokens_per_url: Maximum tokens per URL.
            maximum_number_of_snippets_per_url: Maximum snippets per URL.
            goggles: Goggle URL or inline definition for custom ranking.
            freshness: Page-age filter such as `pd`, `pw`, `pm`, `py`, or a date range.
            enable_local: Whether to enable local recall.
            enable_source_metadata: Whether to enrich source metadata.
            loc_lat: Latitude for local recall.
            loc_long: Longitude for local recall.
            loc_city: City for local recall.
            loc_state: State or region code for local recall.
            loc_state_name: State or region name for local recall.
            loc_country: Country code for local recall.
            loc_postal_code: Postal code for local recall.

        Returns:
            The Brave LLM Context response.
        """
        response = await self.client.request(
            'llm/context',
            method=self.method,
            params={
                'q': query,
                'country': country,
                'search_lang': search_lang,
                'count': count,
                'spellcheck': spellcheck,
                'maximum_number_of_urls': maximum_number_of_urls,
                'maximum_number_of_tokens': maximum_number_of_tokens,
                'maximum_number_of_snippets': maximum_number_of_snippets,
                'context_threshold_mode': context_threshold_mode,
                'maximum_number_of_tokens_per_url': maximum_number_of_tokens_per_url,
                'maximum_number_of_snippets_per_url': maximum_number_of_snippets_per_url,
                'goggles': goggles,
                'freshness': freshness,
                'enable_local': enable_local,
                'enable_source_metadata': enable_source_metadata,
            },
            headers=_location_headers(
                loc_lat=loc_lat,
                loc_long=loc_long,
                loc_city=loc_city,
                loc_state=loc_state,
                loc_state_name=loc_state_name,
                loc_country=loc_country,
                loc_postal_code=loc_postal_code,
            ),
        )
        return brave_llm_context_ta.validate_python(response)


@dataclass
class BravePlaceSearchTool:
    """The Brave Place Search tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    async def __call__(
        self,
        query: str = '',
        radius: float | None = None,
        count: int = 20,
        country: str = 'US',
        search_lang: str = 'en',
        ui_lang: str = 'en-US',
        units: _Units = 'metric',
        safesearch: _SafeSearch = 'strict',
        spellcheck: bool = True,
        geoloc: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        location: str | None = None,
    ) -> BravePlaceSearchResponse:
        """Searches Brave Place Search for points of interest.

        Args:
            query: The POI query. Empty string returns general POIs in the area.
            radius: Radius bias around the given coordinates, in meters.
            count: The number of place results to return.
            country: The country code to scope the search.
            search_lang: The language code for results.
            ui_lang: The UI language preferred in the response.
            units: Measurement units for distance values.
            safesearch: The adult-content filter.
            spellcheck: Whether Brave should spell-check the query.
            geoloc: User geolocation in `<latitude>x<longitude>` format.
            latitude: Latitude of the geographical coordinates.
            longitude: Longitude of the geographical coordinates.
            location: Location string to search around.

        Returns:
            The Brave Place Search response.
        """
        response = await self.client.request(
            'local/place_search',
            method='GET',
            params={
                'q': query,
                'radius': radius,
                'count': count,
                'country': country,
                'search_lang': search_lang,
                'ui_lang': ui_lang,
                'units': units,
                'safesearch': safesearch,
                'spellcheck': spellcheck,
                'geoloc': geoloc,
                'latitude': latitude,
                'longitude': longitude,
                'location': location,
            },
        )
        return brave_place_search_ta.validate_python(response)


@dataclass
class BraveLocalPOIsTool:
    """The Brave Local POIs tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    async def __call__(
        self,
        ids: list[str],
        search_lang: str = 'en',
        ui_lang: str = 'en-US',
        units: _Units | None = None,
        loc_lat: float | None = None,
        loc_long: float | None = None,
    ) -> BraveLocalPOIsResponse:
        """Gets full POI details for Brave location IDs.

        Args:
            ids: POI IDs from Brave Web Search or Place Search results.
            search_lang: The language code for results.
            ui_lang: The UI language preferred in the response.
            units: Measurement units for distance values.
            loc_lat: Latitude for distance calculations.
            loc_long: Longitude for distance calculations.

        Returns:
            The Brave Local POIs response.
        """
        response = await self.client.request(
            'local/pois',
            method='GET',
            params={'ids': ids, 'search_lang': search_lang, 'ui_lang': ui_lang, 'units': units},
            headers=_location_headers(loc_lat=loc_lat, loc_long=loc_long),
        )
        return brave_local_pois_ta.validate_python(response)


@dataclass
class BraveLocalDescriptionsTool:
    """The Brave Local Descriptions tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    async def __call__(self, ids: list[str]) -> BraveLocalDescriptionsResponse:
        """Gets AI-generated descriptions for Brave location IDs.

        Args:
            ids: POI IDs from Brave Web Search or Place Search results.

        Returns:
            The Brave Local Descriptions response.
        """
        response = await self.client.request('local/descriptions', method='GET', params={'ids': ids})
        return brave_local_descriptions_ta.validate_python(response)


@dataclass
class BraveRichSearchTool:
    """The Brave Rich Search callback tool."""

    client: _BraveAPIClient
    """The Brave API client."""

    async def __call__(self, callback_key: str) -> BraveRichSearchResponse:
        """Gets a rich real-time result from a Brave rich callback key.

        Args:
            callback_key: Callback key returned by Web Search with `enable_rich_callback`.

        Returns:
            The Brave Rich Search response.
        """
        response = await self.client.request('web/rich', method='GET', params={'callback_key': callback_key})
        return brave_rich_search_ta.validate_python(response)


def brave_web_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    method: _HttpMethod = 'GET',
    query: str = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    ui_lang: str = _UNSET,
    count: int = _UNSET,
    offset: int = _UNSET,
    safesearch: _SafeSearch = _UNSET,
    freshness: str | None = _UNSET,
    text_decorations: bool = _UNSET,
    spellcheck: bool = _UNSET,
    result_filter: str | None = _UNSET,
    goggles: str | list[str] | None = _UNSET,
    extra_snippets: bool | None = _UNSET,
    summary: bool | None = _UNSET,
    enable_rich_callback: bool = _UNSET,
    include_fetch_metadata: bool = _UNSET,
    operators: bool = _UNSET,
    units: _Units | None = _UNSET,
    loc_lat: float | None = _UNSET,
    loc_long: float | None = _UNSET,
    loc_timezone: str | None = _UNSET,
    loc_city: str | None = _UNSET,
    loc_state: str | None = _UNSET,
    loc_state_name: str | None = _UNSET,
    loc_country: str | None = _UNSET,
    loc_postal_code: str | None = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Web Search tool.

    Args:
        api_key: The Brave Search API key.
        client: An existing `httpx.AsyncClient`.
        method: HTTP method to use. Defaults to `GET`.
        query: Fixed search query.
        country: Fixed country code.
        search_lang: Fixed search language code.
        ui_lang: Fixed UI language.
        count: Fixed result count.
        offset: Fixed page offset.
        safesearch: Fixed adult-content filter.
        freshness: Fixed page-age filter.
        text_decorations: Fixed text decorations setting.
        spellcheck: Fixed spellcheck setting.
        result_filter: Fixed result type filter.
        goggles: Fixed Goggles custom ranking.
        extra_snippets: Fixed extra snippets setting.
        summary: Fixed summary setting.
        enable_rich_callback: Fixed rich callback setting.
        include_fetch_metadata: Fixed fetch metadata setting.
        operators: Fixed search operators setting.
        units: Fixed measurement units.
        loc_lat: Fixed latitude header.
        loc_long: Fixed longitude header.
        loc_timezone: Fixed timezone header.
        loc_city: Fixed city header.
        loc_state: Fixed state header.
        loc_state_name: Fixed state name header.
        loc_country: Fixed country header.
        loc_postal_code: Fixed postal code header.
    """
    return _make_tool(
        BraveWebSearchTool(client=_make_client(api_key, client), method=method).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client', 'method'}),
        name='brave_web_search',
        description='Searches Brave Web Search and returns web, news, video, discussion, location, and rich result metadata.',
    )


def brave_news_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    method: _HttpMethod = 'GET',
    query: str = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    ui_lang: str = _UNSET,
    count: int = _UNSET,
    offset: int = _UNSET,
    safesearch: _SafeSearch = _UNSET,
    freshness: str | None = _UNSET,
    spellcheck: bool = _UNSET,
    extra_snippets: bool | None = _UNSET,
    goggles: str | list[str] | None = _UNSET,
    operators: bool = _UNSET,
    include_fetch_metadata: bool = _UNSET,
) -> Tool[Any]:
    """Creates a Brave News Search tool."""
    return _make_tool(
        BraveNewsSearchTool(client=_make_client(api_key, client), method=method).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client', 'method'}),
        name='brave_news_search',
        description='Searches Brave News Search and returns news articles with source metadata.',
    )


def brave_image_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    query: str = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    count: int = _UNSET,
    safesearch: _ImageSafeSearch = _UNSET,
    spellcheck: bool = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Image Search tool."""
    return _make_tool(
        BraveImageSearchTool(client=_make_client(api_key, client)).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client'}),
        name='brave_image_search',
        description='Searches Brave Image Search and returns image results with source and thumbnail metadata.',
    )


def brave_video_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    method: _HttpMethod = 'GET',
    query: str = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    ui_lang: str = _UNSET,
    count: int = _UNSET,
    offset: int = _UNSET,
    safesearch: _SafeSearch = _UNSET,
    freshness: str | None = _UNSET,
    spellcheck: bool = _UNSET,
    operators: bool = _UNSET,
    include_fetch_metadata: bool = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Video Search tool."""
    return _make_tool(
        BraveVideoSearchTool(client=_make_client(api_key, client), method=method).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client', 'method'}),
        name='brave_video_search',
        description='Searches Brave Video Search and returns video results with duration, creator, and thumbnail metadata.',
    )


def brave_llm_context_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    method: _HttpMethod = 'GET',
    query: str = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    count: int = _UNSET,
    spellcheck: bool = _UNSET,
    maximum_number_of_urls: int = _UNSET,
    maximum_number_of_tokens: int = _UNSET,
    maximum_number_of_snippets: int = _UNSET,
    context_threshold_mode: _ContextThresholdMode = _UNSET,
    maximum_number_of_tokens_per_url: int = _UNSET,
    maximum_number_of_snippets_per_url: int = _UNSET,
    goggles: str | list[str] | None = _UNSET,
    freshness: str | None = _UNSET,
    enable_local: bool | None = _UNSET,
    enable_source_metadata: bool = _UNSET,
    loc_lat: float | None = _UNSET,
    loc_long: float | None = _UNSET,
    loc_city: str | None = _UNSET,
    loc_state: str | None = _UNSET,
    loc_state_name: str | None = _UNSET,
    loc_country: str | None = _UNSET,
    loc_postal_code: str | None = _UNSET,
) -> Tool[Any]:
    """Creates a Brave LLM Context tool."""
    return _make_tool(
        BraveLLMContextTool(client=_make_client(api_key, client), method=method).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client', 'method'}),
        name='brave_llm_context',
        description='Gets Brave LLM Context with extracted snippets and source metadata for grounding an agent response.',
    )


def brave_place_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    query: str = _UNSET,
    radius: float | None = _UNSET,
    count: int = _UNSET,
    country: str = _UNSET,
    search_lang: str = _UNSET,
    ui_lang: str = _UNSET,
    units: _Units = _UNSET,
    safesearch: _SafeSearch = _UNSET,
    spellcheck: bool = _UNSET,
    geoloc: str | None = _UNSET,
    latitude: float | None = _UNSET,
    longitude: float | None = _UNSET,
    location: str | None = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Place Search tool."""
    return _make_tool(
        BravePlaceSearchTool(client=_make_client(api_key, client)).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client'}),
        name='brave_place_search',
        description='Searches Brave Place Search for points of interest and local place metadata.',
    )


def brave_local_pois_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    ids: list[str] = _UNSET,
    search_lang: str = _UNSET,
    ui_lang: str = _UNSET,
    units: _Units | None = _UNSET,
    loc_lat: float | None = _UNSET,
    loc_long: float | None = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Local POIs tool."""
    return _make_tool(
        BraveLocalPOIsTool(client=_make_client(api_key, client)).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client'}),
        name='brave_local_pois',
        description='Gets full Brave local POI details for location IDs returned by Brave search.',
    )


def brave_local_descriptions_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    ids: list[str] = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Local Descriptions tool."""
    return _make_tool(
        BraveLocalDescriptionsTool(client=_make_client(api_key, client)).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client'}),
        name='brave_local_descriptions',
        description='Gets AI-generated markdown descriptions for Brave local POI IDs.',
    )


def brave_rich_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    callback_key: str = _UNSET,
) -> Tool[Any]:
    """Creates a Brave Rich Search callback tool."""
    return _make_tool(
        BraveRichSearchTool(client=_make_client(api_key, client)).__call__,
        _fixed_kwargs(locals(), exclude={'api_key', 'client'}),
        name='brave_rich_search',
        description='Gets Brave rich real-time results using a callback key from Web Search.',
    )


class BraveSearchToolset(FunctionToolset):
    """A toolset that provides Brave Search API tools with a shared HTTP client.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.common_tools.brave import BraveSearchToolset

    toolset = BraveSearchToolset(api_key='your-api-key')
    agent = Agent('openai:gpt-5.2', toolsets=[toolset])
    ```
    """

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = _BASE_URL,
        include_web_search: bool = True,
        include_news_search: bool = True,
        include_image_search: bool = True,
        include_video_search: bool = True,
        include_llm_context: bool = True,
        include_place_search: bool = True,
        include_local_pois: bool = True,
        include_local_descriptions: bool = True,
        include_rich_search: bool = True,
        id: str | None = None,
    ):
        """Creates a Brave Search toolset.

        Args:
            api_key: The Brave Search API key.
            client: An existing `httpx.AsyncClient`.
            base_url: The Brave Search API base URL.
            include_web_search: Whether to include the web search tool.
            include_news_search: Whether to include the news search tool.
            include_image_search: Whether to include the image search tool.
            include_video_search: Whether to include the video search tool.
            include_llm_context: Whether to include the LLM context tool.
            include_place_search: Whether to include the place search tool.
            include_local_pois: Whether to include the local POIs tool.
            include_local_descriptions: Whether to include the local descriptions tool.
            include_rich_search: Whether to include the rich search tool.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        api_client = _BraveAPIClient(client=client or httpx.AsyncClient(), api_key=api_key, base_url=base_url)
        tools: list[Tool[Any]] = []

        if include_web_search:
            tools.append(
                _make_tool(
                    BraveWebSearchTool(client=api_client).__call__,
                    {},
                    name='brave_web_search',
                    description='Searches Brave Web Search and returns web, news, video, discussion, location, and rich result metadata.',
                )
            )
        if include_news_search:
            tools.append(
                _make_tool(
                    BraveNewsSearchTool(client=api_client).__call__,
                    {},
                    name='brave_news_search',
                    description='Searches Brave News Search and returns news articles with source metadata.',
                )
            )
        if include_image_search:
            tools.append(
                _make_tool(
                    BraveImageSearchTool(client=api_client).__call__,
                    {},
                    name='brave_image_search',
                    description='Searches Brave Image Search and returns image results with source and thumbnail metadata.',
                )
            )
        if include_video_search:
            tools.append(
                _make_tool(
                    BraveVideoSearchTool(client=api_client).__call__,
                    {},
                    name='brave_video_search',
                    description='Searches Brave Video Search and returns video results with duration, creator, and thumbnail metadata.',
                )
            )
        if include_llm_context:
            tools.append(
                _make_tool(
                    BraveLLMContextTool(client=api_client).__call__,
                    {},
                    name='brave_llm_context',
                    description='Gets Brave LLM Context with extracted snippets and source metadata for grounding an agent response.',
                )
            )
        if include_place_search:
            tools.append(
                _make_tool(
                    BravePlaceSearchTool(client=api_client).__call__,
                    {},
                    name='brave_place_search',
                    description='Searches Brave Place Search for points of interest and local place metadata.',
                )
            )
        if include_local_pois:
            tools.append(
                _make_tool(
                    BraveLocalPOIsTool(client=api_client).__call__,
                    {},
                    name='brave_local_pois',
                    description='Gets full Brave local POI details for location IDs returned by Brave search.',
                )
            )
        if include_local_descriptions:
            tools.append(
                _make_tool(
                    BraveLocalDescriptionsTool(client=api_client).__call__,
                    {},
                    name='brave_local_descriptions',
                    description='Gets AI-generated markdown descriptions for Brave local POI IDs.',
                )
            )
        if include_rich_search:
            tools.append(
                _make_tool(
                    BraveRichSearchTool(client=api_client).__call__,
                    {},
                    name='brave_rich_search',
                    description='Gets Brave rich real-time results using a callback key from Web Search.',
                )
            )

        super().__init__(tools, id=id)


def _make_client(api_key: str | None, client: httpx.AsyncClient | None) -> _BraveAPIClient:
    if api_key is None:
        raise ValueError('api_key must be provided')
    return _BraveAPIClient(client=client or httpx.AsyncClient(), api_key=api_key)


def _make_tool(func: Any, kwargs: dict[str, Any], *, name: str, description: str) -> Tool[Any]:
    if kwargs:
        original = func
        func = partial(func, **kwargs)
        func.__name__ = original.__name__
        func.__qualname__ = original.__qualname__
        orig_sig = signature(original)
        func.__signature__ = orig_sig.replace(
            parameters=[p for param_name, p in orig_sig.parameters.items() if param_name not in kwargs]
        )

    return Tool[Any](
        func,
        name=name,
        description=description,
    )


def _fixed_kwargs(values: Mapping[str, Any], *, exclude: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key not in exclude and value is not _UNSET}


def _clean_params(params: Mapping[str, _ParamValue]) -> dict[str, _CleanParamValue]:
    return {key: value for key, value in params.items() if value is not None}


def _query_params(params: Mapping[str, _CleanParamValue]) -> dict[str, str | list[str]]:
    result: dict[str, str | list[str]] = {}
    for key, value in params.items():
        if isinstance(value, list):
            result[key] = [_stringify_param(item) for item in value]
        else:
            result[key] = _stringify_param(value)
    return result


def _stringify_param(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return str(value)


def _location_headers(
    *,
    loc_lat: float | None = None,
    loc_long: float | None = None,
    loc_timezone: str | None = None,
    loc_city: str | None = None,
    loc_state: str | None = None,
    loc_state_name: str | None = None,
    loc_country: str | None = None,
    loc_postal_code: str | None = None,
) -> dict[str, str]:
    values = {
        'X-Loc-Lat': loc_lat,
        'X-Loc-Long': loc_long,
        'X-Loc-Timezone': loc_timezone,
        'X-Loc-City': loc_city,
        'X-Loc-State': loc_state,
        'X-Loc-State-Name': loc_state_name,
        'X-Loc-Country': loc_country,
        'X-Loc-Postal-Code': loc_postal_code,
    }
    return {key: _stringify_param(value) for key, value in values.items() if value is not None}
