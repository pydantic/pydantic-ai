from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal, overload

from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from nimble_python import AsyncNimble
except ImportError as _import_error:
    raise ImportError(
        'Please install `nimble_python` to use the Nimble search tool, '
        'you can use the `nimble` optional group — `pip install "pydantic-ai-slim[nimble]"`'
    ) from _import_error

__all__ = ('nimble_search_tool',)

_CLIENT_SOURCE = 'pydantic-ai'
"""Stable attribution slug sent as `X-Client-Source` on every Nimble request."""

_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from None in factory kwargs."""


class NimbleSearchResult(TypedDict):
    """A Nimble search result.

    See [Nimble Search API documentation](https://docs.nimbleway.com/)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    content: str
    """The result content, or the description when content is empty (e.g. `lite` depth)."""


nimble_search_ta = TypeAdapter(list[NimbleSearchResult])


@dataclass
class NimbleSearchTool:
    """The Nimble search tool."""

    client: AsyncNimble
    """The Nimble async client."""

    _: KW_ONLY

    max_results: int | None = None
    """The maximum number of results. If None, the Nimble default is used."""

    async def __call__(
        self,
        query: str,
        search_depth: Literal['lite', 'fast', 'deep'] = 'lite',
        time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[NimbleSearchResult]:
        """Searches Nimble for the given query and returns the results.

        Args:
            query: The search query to execute with Nimble.
            search_depth: Controls content richness and latency of search results.
            time_range: The time range back from the current date to filter results.
            include_domains: List of domains to specifically include in the search results.
            exclude_domains: List of domains to specifically exclude from the search results.

        Returns:
            A list of search results from Nimble.
        """
        search_kwargs: dict[str, Any] = {
            'query': query,
            'search_depth': search_depth,
            'time_range': time_range,
            'include_domains': include_domains,
            'exclude_domains': exclude_domains,
        }
        if self.max_results is not None:
            search_kwargs['max_results'] = self.max_results
        response = await self.client.search(**search_kwargs)
        projected = [
            {
                'title': result.title,
                'url': result.url,
                'content': result.content or result.description,
            }
            for result in response.results
        ]
        return nimble_search_ta.validate_python(projected)


@overload
def nimble_search_tool(
    api_key: str,
    *,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


@overload
def nimble_search_tool(
    *,
    client: AsyncNimble,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


def nimble_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]:
    """Creates a Nimble search tool.

    `max_results` is always developer-controlled and does not appear in the LLM tool schema.
    Other parameters, when provided, are fixed for all searches and hidden from the LLM's
    tool schema. Parameters left unset remain available for the LLM to set per-call.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.

            You can get one from [Nimble's dashboard](https://online.nimbleway.com/account-settings/api-keys).
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tool instances.
        max_results: The maximum number of results. If None, the Nimble default is used.
        search_depth: Controls content richness and latency of search results.
        time_range: The time range back from the current date to filter results.
        include_domains: List of domains to specifically include in the search results.
        exclude_domains: List of domains to specifically exclude from the search results.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncNimble(api_key=api_key, client_source=_CLIENT_SOURCE)
    func = NimbleSearchTool(client=client, max_results=max_results).__call__

    kwargs: dict[str, Any] = {}
    if search_depth is not _UNSET:
        kwargs['search_depth'] = search_depth
    if time_range is not _UNSET:
        kwargs['time_range'] = time_range
    if include_domains is not _UNSET:
        kwargs['include_domains'] = include_domains
    if exclude_domains is not _UNSET:
        kwargs['exclude_domains'] = exclude_domains

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
        name='nimble_search',
        description='Searches Nimble for the given query and returns the results.',
    )
