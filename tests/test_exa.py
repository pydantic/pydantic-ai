"""Tests for the Exa common tools.

These tests avoid hitting the network by injecting a fake `AsyncExa` client that
records the keyword arguments passed to `search` / `find_similar`, so we can assert
the developer-configured domain filters are forwarded to the Exa SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai.common_tools.exa import (
    ExaFindSimilarTool,
    ExaSearchTool,
    ExaToolset,
)

pytest.importorskip('exa_py', reason='exa extra not installed')

pytestmark = pytest.mark.anyio


@dataclass
class _FakeResult:
    title: str | None = 'Title'
    url: str = 'https://example.com'
    published_date: str | None = None
    author: str | None = None
    text: str | None = 'body'


@dataclass
class _FakeResponse:
    results: list[_FakeResult] = field(default_factory=lambda: [_FakeResult()])


class _FakeExaClient:
    """Records the kwargs passed to `search` / `find_similar`."""

    def __init__(self) -> None:
        self.search_kwargs: dict[str, Any] = {}
        self.find_similar_kwargs: dict[str, Any] = {}

    async def search(self, query: str, **kwargs: Any) -> _FakeResponse:
        self.search_kwargs = {'query': query, **kwargs}
        return _FakeResponse()

    async def find_similar(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.find_similar_kwargs = {'url': url, **kwargs}
        return _FakeResponse()


async def test_search_forwards_domain_filters() -> None:
    client = _FakeExaClient()
    tool = ExaSearchTool(
        client=client,  # type: ignore[arg-type]
        num_results=3,
        max_characters=None,
        include_domains=['arxiv.org'],
        exclude_domains=['spam.example'],
    )

    results = await tool('quantum computing')

    assert results == [
        {
            'title': 'Title',
            'url': 'https://example.com',
            'published_date': None,
            'author': None,
            'text': 'body',
        }
    ]
    assert client.search_kwargs['include_domains'] == ['arxiv.org']
    assert client.search_kwargs['exclude_domains'] == ['spam.example']


async def test_search_defaults_domain_filters_to_none() -> None:
    client = _FakeExaClient()
    tool = ExaSearchTool(client=client, num_results=5, max_characters=None)  # type: ignore[arg-type]

    await tool('hello')

    assert client.search_kwargs['include_domains'] is None
    assert client.search_kwargs['exclude_domains'] is None


async def test_find_similar_forwards_domain_filters() -> None:
    client = _FakeExaClient()
    tool = ExaFindSimilarTool(
        client=client,  # type: ignore[arg-type]
        num_results=2,
        include_domains=['arxiv.org'],
        exclude_domains=['spam.example'],
    )

    await tool('https://example.com/article')

    assert client.find_similar_kwargs['include_domains'] == ['arxiv.org']
    assert client.find_similar_kwargs['exclude_domains'] == ['spam.example']
    assert client.find_similar_kwargs['exclude_source_domain'] is True


def test_toolset_threads_domain_filters_to_search_and_find_similar() -> None:
    toolset = ExaToolset(
        api_key='test-key',
        include_domains=['arxiv.org'],
        exclude_domains=['spam.example'],
        include_get_contents=False,
        include_answer=False,
    )

    search_impl = toolset.tools['exa_search'].function.__self__  # type: ignore[attr-defined]
    find_similar_impl = toolset.tools['exa_find_similar'].function.__self__  # type: ignore[attr-defined]

    assert isinstance(search_impl, ExaSearchTool)
    assert search_impl.include_domains == ['arxiv.org']
    assert search_impl.exclude_domains == ['spam.example']

    assert isinstance(find_similar_impl, ExaFindSimilarTool)
    assert find_similar_impl.include_domains == ['arxiv.org']
    assert find_similar_impl.exclude_domains == ['spam.example']
