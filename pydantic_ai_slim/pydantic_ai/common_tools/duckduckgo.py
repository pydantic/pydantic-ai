from dataclasses import dataclass
from typing import TypedDict

import anyio
import anyio.to_thread
from pydantic import TypeAdapter

from pydantic_ai.tools import Tool

try:
    from duckduckgo_search import DDGS
except ImportError as _import_error:
    raise ImportError(
        'Please install `duckduckgo-search` to use the DuckDuckGo search tool, '
        "you can use the `duckduckgo` optional group — `pip install 'pydantic-ai-slim[duckduckgo]'`"
    ) from _import_error

__all__ = ('duckduckgo_search_tool',)


class DuckDuckGoResult(TypedDict):
    """A DuckDuckGo search result."""

    title: str
    """The title of the search result."""
    href: str
    """The URL of the search result."""
    body: str
    """The body of the search result."""


duckduckgo_ta = TypeAdapter(list[DuckDuckGoResult])


@dataclass
class DuckDuckGoSearchTool:
    """The DuckDuckGo search tool."""

    client: DDGS
    """The DuckDuckGo search client."""

    async def __call__(self, query: str) -> list[DuckDuckGoResult]:
        """Searches DuckDuckGo for the given query and returns the results.

        Args:
            query: The query to search for.

        Returns:
            The search results.
        """
        results = await anyio.to_thread.run_sync(self.client.text, query)
        if len(results) == 0:
            raise RuntimeError('No search results found.')
        return duckduckgo_ta.validate_python(results)


def duckduckgo_search_tool(duckduckgo_client: DDGS | None = None):
    """Creates a DuckDuckGo search tool."""
    return Tool(
        DuckDuckGoSearchTool(client=duckduckgo_client or DDGS()).__call__,
        name='duckduckgo_search',
        description='Searches DuckDuckGo for the given query and returns the results.',
    )
