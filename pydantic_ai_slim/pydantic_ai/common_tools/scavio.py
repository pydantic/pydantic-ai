from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal, overload

from pydantic import TypeAdapter
from typing_extensions import Any, NotRequired, TypedDict

from pydantic_ai.tools import Tool

try:
    from scavio import AsyncScavioClient
except ImportError as _import_error:
    raise ImportError(
        'Please install `scavio` to use the Scavio search tool, '
        'you can use the `scavio` optional group — `pip install "pydantic-ai-slim[scavio]"`'
    ) from _import_error

__all__ = ('scavio_search_tool',)

_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from None in factory kwargs."""


class ScavioSearchResult(TypedDict):
    """A Scavio Google search result.

    See [Scavio documentation](https://scavio.dev/docs/search-api) for more information.
    """

    position: int
    """The rank of the result on the page."""
    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    domain: str
    """The domain of the search result."""
    content: str
    """A short description (snippet) of the search result."""
    date: NotRequired[str | None]
    """The publication date of the result, if available."""


scavio_search_ta = TypeAdapter(list[ScavioSearchResult])


@dataclass
class ScavioSearchTool:
    """The Scavio search tool."""

    client: AsyncScavioClient
    """The Scavio search client."""

    _: KW_ONLY

    light_request: bool = True
    """Use the cheaper, lighter response (1 credit instead of 2). Developer-controlled."""

    async def __call__(
        self,
        query: str,
        country_code: str | None = None,
        language: str | None = None,
        search_type: Literal['classic', 'news', 'maps', 'images', 'lens'] = 'classic',
    ) -> list[ScavioSearchResult]:
        """Searches Google through Scavio for the given query and returns the results.

        Args:
            query: The search query to execute with Scavio.
            country_code: Two-letter country code to localize results (for example `us`).
            language: Two-letter language code for results (for example `en`).
            search_type: The Google vertical to search.

        Returns:
            A list of organic search results from Scavio.
        """
        results: dict[str, Any] = await self.client.google.search(
            query,
            country_code=country_code,
            language=language,
            search_type=search_type,
            light_request=self.light_request,
        )
        return scavio_search_ta.validate_python(results.get('results', []))


@overload
def scavio_search_tool(
    api_key: str,
    *,
    light_request: bool = True,
    country_code: str | None = _UNSET,
    language: str | None = _UNSET,
    search_type: Literal['classic', 'news', 'maps', 'images', 'lens'] = _UNSET,
) -> Tool[Any]: ...


@overload
def scavio_search_tool(
    *,
    client: AsyncScavioClient,
    light_request: bool = True,
    country_code: str | None = _UNSET,
    language: str | None = _UNSET,
    search_type: Literal['classic', 'news', 'maps', 'images', 'lens'] = _UNSET,
) -> Tool[Any]: ...


def scavio_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncScavioClient | None = None,
    light_request: bool = True,
    country_code: str | None = _UNSET,
    language: str | None = _UNSET,
    search_type: Literal['classic', 'news', 'maps', 'images', 'lens'] = _UNSET,
) -> Tool[Any]:
    """Creates a Scavio search tool.

    [Scavio](https://scavio.dev) is a real-time search API for AI agents. This tool runs a
    Google web search and returns the organic results.

    `light_request` is always developer-controlled and does not appear in the LLM tool schema.
    Other parameters, when provided, are fixed for all searches and hidden from the LLM's tool
    schema. Parameters left unset remain available for the LLM to set per-call.

    Args:
        api_key: The Scavio API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.scavio.dev](https://dashboard.scavio.dev).
        client: An existing AsyncScavioClient. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tool instances.
        light_request: Use the cheaper, lighter response (1 credit instead of 2).
        country_code: Two-letter country code to localize results (for example `us`).
        language: Two-letter language code for results (for example `en`).
        search_type: The Google vertical to search.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncScavioClient(api_key=api_key)
    func = ScavioSearchTool(client=client, light_request=light_request).__call__

    kwargs: dict[str, Any] = {}
    if country_code is not _UNSET:
        kwargs['country_code'] = country_code
    if language is not _UNSET:
        kwargs['language'] = language
    if search_type is not _UNSET:
        kwargs['search_type'] = search_type

    if kwargs:
        original = func
        func = partial(func, **kwargs)
        func.__name__ = original.__name__  # type: ignore[union-attr]
        func.__qualname__ = original.__qualname__
        # partial with keyword args only updates defaults, not removes params.
        # Set __signature__ explicitly to exclude bound params from the tool schema.
        orig_sig = signature(original)
        func.__signature__ = orig_sig.replace(  # type: ignore[attr-defined]
            parameters=[p for name, p in orig_sig.parameters.items() if name not in kwargs]
        )

    return Tool[Any](
        func,  # pyright: ignore[reportArgumentType]
        name='scavio_search',
        description='Searches Google through Scavio for the given query and returns the results.',
    )
