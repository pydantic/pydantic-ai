"""Tests for the search tools in `pydantic_ai.common_tools`."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai.common_tools.duckduckgo import DDGSException, DuckDuckGoSearchTool


class _StubClient:
    """A stand-in for the `ddgs` client that returns canned results or raises."""

    def __init__(self, results: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self._results = results
        self._error = error
        self.queries: list[str] = []

    def text(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return self._results or []


async def test_search_returns_results():
    client = _StubClient(results=[{'title': 't', 'href': 'https://example.com', 'body': 'b'}])
    tool = DuckDuckGoSearchTool(client=client, max_results=5)  # type: ignore[arg-type]

    assert await tool('pydantic-ai') == [{'title': 't', 'href': 'https://example.com', 'body': 'b'}]
    assert client.queries == ['pydantic-ai']


async def test_search_returns_empty_list_when_no_results():
    """`ddgs` raises `DDGSException('No results found.')` for zero hits; the tool returns `[]` instead."""
    client = _StubClient(error=DDGSException('No results found.'))
    tool = DuckDuckGoSearchTool(client=client, max_results=5)  # type: ignore[arg-type]

    assert await tool('tael of gold to tael of silver ratio China exchange rate') == []


async def test_search_propagates_other_client_errors():
    """Backend failures and rate limits carry their own message and are not swallowed."""
    client = _StubClient(error=DDGSException('backend exploded'))
    tool = DuckDuckGoSearchTool(client=client, max_results=5)  # type: ignore[arg-type]

    with pytest.raises(DDGSException, match='backend exploded'):
        await tool('pydantic-ai')
