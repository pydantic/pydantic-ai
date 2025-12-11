import functools
from dataclasses import KW_ONLY, dataclass

import anyio.to_thread
from pydantic import TypeAdapter
from typing_extensions import Any, Literal, TypedDict

from pydantic_ai.tools import Tool

try:
    try:
        from ddgs.ddgs import DDGS
    except ImportError:  # Fallback for older versions of ddgs
        from duckduckgo_search import DDGS
except ImportError as _import_error:
    raise ImportError(
        'Please install `ddgs` to use the DuckDuckGo search tool, '
        'you can use the `duckduckgo` optional group — `pip install "pydantic-ai-slim[duckduckgo]"`'
    ) from _import_error

__all__ = ('duckduckgo_search_tool', 'duckduckgo_images_search_tool', 'duckduckgo_videos_search_tool', 'duckduckgo_news_search_tool')


class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""


class DuckDuckGoImageResult(TypedDict, total=False):
    """A DuckDuckGo image search result."""

    title: str
    """The title of the image."""
    image: str
    """The image URL."""
    thumbnail: str
    """The thumbnail URL."""
    url: str
    """The source page URL."""
    height: int
    """The image height."""
    width: int
    """The image width."""
    source: str
    """The image source."""


class DuckDuckGoVideoResult(TypedDict, total=False):
    """A DuckDuckGo video search result."""

    content: str
    """The video content URL."""
    description: str
    """The video description."""
    duration: str
    """The video duration."""
    embed_html: str
    """The embed HTML."""
    embed_url: str
    """The embed URL."""
    image_token: str
    """The image token."""
    images: dict[str, str]
    """The video images (large, medium, motion, small)."""
    provider: str
    """The video provider."""
    published: str
    """The publication date."""
    publisher: str
    """The video publisher."""
    statistics: dict[str, int]
    """The video statistics (e.g., viewCount)."""
    title: str
    """The video title."""
    uploader: str
    """The video uploader."""


class DuckDuckGoNewsResult(TypedDict, total=False):
    """A DuckDuckGo news search result."""

    date: str
    """The publication date."""
    title: str
    """The news title."""
    body: str
    """The news body."""
    url: str
    """The news URL."""
    image: str
    """The news image URL."""
    source: str
    """The news source."""


# TypeAdapters for validation
duckduckgo_ta = TypeAdapter(list[DuckDuckGoResult])
images_ta = TypeAdapter(list[DuckDuckGoImageResult])
videos_ta = TypeAdapter(list[DuckDuckGoVideoResult])
news_ta = TypeAdapter(list[DuckDuckGoNewsResult])


@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    _: KW_ONLY

    max_results: int | None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(
        self,
        query: str,
        region: str = 'us-en',
        safesearch: Literal['on', 'moderate', 'off'] = 'moderate',
    ) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.
            region: The region to search in (e.g., 'us-en', 'uk-en').
            safesearch: The safe search setting ('on', 'moderate', or 'off').

        Returns:
            The search results.
        """
        search = functools.partial(
            self.client.text, region=region, safesearch=safesearch, max_results=self.max_results
        )
        results = await anyio.to_thread.run_sync(search, query)
        return duckduckgo_ta.validate_python(results)


def duckduckgo_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo text search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool[Any](
        DuckDuckGoSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_search',
        description='Searches DuckDuckGo for the given query and returns text results.',
    )


@dataclass
class DuckDuckGoImagesSearchTool:
    """The DuckDuckGo images search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    _: KW_ONLY

    max_results: int | None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(
        self,
        query: str,
        region: str = 'us-en',
        safesearch: Literal['on', 'moderate', 'off'] = 'moderate',
        size: Literal['Small', 'Medium', 'Large', 'Wallpaper'] | None = None,
        color: Literal[
            'color',
            'Monochrome',
            'Red',
            'Orange',
            'Yellow',
            'Green',
            'Blue',
            'Purple',
            'Pink',
            'Brown',
            'Black',
            'Gray',
            'Teal',
            'White',
        ]
        | None = None,
        type_image: Literal['photo', 'clipart', 'gif', 'transparent', 'line'] | None = None,
        layout: Literal['Square', 'Tall', 'Wide'] | None = None,
        license_image: Literal['any', 'Public', 'Share', 'ShareCommercially', 'Modify', 'ModifyCommercially']
        | None = None,
    ) -> list[DuckDuckGoImageResult]:
        """Searches DuckDuckGo for images matching the given query.

        Args:
            query: The query to search for.
            region: The region to search in (e.g., 'us-en', 'uk-en').
            safesearch: The safe search setting ('on', 'moderate', or 'off').
            size: Filter by image size.
            color: Filter by image color.
            type_image: Filter by image type.
            layout: Filter by image layout.
            license_image: Filter by image license.

        Returns:
            The image search results.
        """
        kwargs = {'region': region, 'safesearch': safesearch, 'max_results': self.max_results}
        if size is not None:
            kwargs['size'] = size
        if color is not None:
            kwargs['color'] = color
        if type_image is not None:
            kwargs['type_image'] = type_image
        if layout is not None:
            kwargs['layout'] = layout
        if license_image is not None:
            kwargs['license_image'] = license_image
        
        search = functools.partial(self.client.images, **kwargs)
        results = await anyio.to_thread.run_sync(search, query)
        return images_ta.validate_python(results)


def duckduckgo_images_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo images search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool[Any](
        DuckDuckGoImagesSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_images',
        description='Searches DuckDuckGo for images matching the given query.',
    )


@dataclass
class DuckDuckGoVideosSearchTool:
    """The DuckDuckGo videos search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    _: KW_ONLY

    max_results: int | None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(
        self,
        query: str,
        region: str = 'us-en',
        safesearch: Literal['on', 'moderate', 'off'] = 'moderate',
        resolution: str | None = None,
        duration: Literal['short', 'medium', 'long'] | None = None,
        license_videos: Literal['creativeCommon', 'youtube'] | None = None,
    ) -> list[DuckDuckGoVideoResult]:
        """Searches DuckDuckGo for videos matching the given query.

        Args:
            query: The query to search for.
            region: The region to search in (e.g., 'us-en', 'uk-en').
            safesearch: The safe search setting ('on', 'moderate', or 'off').
            resolution: Filter by video resolution.
            duration: Filter by video duration ('short', 'medium', or 'long').
            license_videos: Filter by video license.

        Returns:
            The video search results.
        """
        kwargs = {'region': region, 'safesearch': safesearch, 'max_results': self.max_results}
        if resolution is not None:
            kwargs['resolution'] = resolution
        if duration is not None:
            kwargs['duration'] = duration
        if license_videos is not None:
            kwargs['license_videos'] = license_videos

        search = functools.partial(self.client.videos, **kwargs)
        results = await anyio.to_thread.run_sync(search, query)
        return videos_ta.validate_python(results)


def duckduckgo_videos_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo videos search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool[Any](
        DuckDuckGoVideosSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_videos',
        description='Searches DuckDuckGo for videos matching the given query.',
    )


@dataclass
class DuckDuckGoNewsSearchTool:
    """The DuckDuckGo news search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    _: KW_ONLY

    max_results: int | None
    """The maximum number of results. If None, returns results only from the first response."""

    async def __call__(
        self,
        query: str,
        region: str = 'us-en',
        safesearch: Literal['on', 'moderate', 'off'] = 'moderate',
    ) -> list[DuckDuckGoNewsResult]:
        """Searches DuckDuckGo for news articles matching the given query.

        Args:
            query: The query to search for.
            region: The region to search in (e.g., 'us-en', 'uk-en').
            safesearch: The safe search setting ('on', 'moderate', or 'off').

        Returns:
            The news search results.
        """
        kwargs = {'region': region, 'safesearch': safesearch, 'max_results': self.max_results}

        search = functools.partial(self.client.news, **kwargs)
        results = await anyio.to_thread.run_sync(search, query)
        return news_ta.validate_python(results)


def duckduckgo_news_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
    """Creates a DuckDuckGo news search tool.

    Args:
        duckduckgo_client: The DuckDuckGo search client.
        max_results: The maximum number of results. If None, returns results only from the first response.
    """
    return Tool[Any](
        DuckDuckGoNewsSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
        name='duckduckgo_news',
        description='Searches DuckDuckGo for news articles matching the given query.',
    )
