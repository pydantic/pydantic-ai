"""Tests for the Tavily search tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.common_tools.tavily import TavilySearchTool, tavily_search_tool

pytestmark = pytest.mark.skipif(not imports_successful(), reason='tavily-python not installed')


@pytest.fixture
def mock_tavily_response() -> dict[str, Any]:
    """Sample Tavily API response."""
    return {
        'results': [
            {
                'title': 'Test Result 1',
                'url': 'https://example.com/1',
                'content': 'This is test content 1',
                'score': 0.95,
            },
            {
                'title': 'Test Result 2',
                'url': 'https://example.com/2',
                'content': 'This is test content 2',
                'score': 0.85,
            },
        ]
    }


@pytest.fixture
def mock_async_tavily_client(mock_tavily_response: dict[str, Any]) -> AsyncMock:
    """Create a mock AsyncTavilyClient."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=mock_tavily_response)
    return mock_client


class TestTavilySearchTool:
    """Tests for TavilySearchTool."""

    async def test_basic_search(self, mock_async_tavily_client: AsyncMock):
        """Test basic search with default parameters."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        results = await tool('test query')

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            max_results=None,
            include_domains=None,
            exclude_domains=None,
        )
        assert len(results) == 2
        assert results[0]['title'] == 'Test Result 1'

    async def test_search_with_include_domains(self, mock_async_tavily_client: AsyncMock):
        """Test search with include_domains specified."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool('test query', include_domains=['arxiv.org', 'github.com'])

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            max_results=None,
            include_domains=['arxiv.org', 'github.com'],
            exclude_domains=None,
        )

    async def test_search_with_exclude_domains(self, mock_async_tavily_client: AsyncMock):
        """Test search with exclude_domains specified."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool('test query', exclude_domains=['medium.com'])

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            max_results=None,
            include_domains=None,
            exclude_domains=['medium.com'],
        )

    async def test_search_with_max_results(self, mock_async_tavily_client: AsyncMock):
        """Test search with max_results specified."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool('test query', max_results=5)

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            max_results=5,
            include_domains=None,
            exclude_domains=None,
        )

    async def test_search_with_all_parameters(self, mock_async_tavily_client: AsyncMock):
        """Test search with all parameters specified."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool(
            'test query',
            search_deep='advanced',
            topic='news',
            time_range='week',
            max_results=10,
            include_domains=['news.com'],
            exclude_domains=['spam.com'],
        )

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='advanced',
            topic='news',
            time_range='week',
            max_results=10,
            include_domains=['news.com'],
            exclude_domains=['spam.com'],
        )


class TestTavilySearchToolFactory:
    """Tests for tavily_search_tool factory function."""

    def test_creates_tool_with_api_key(self):
        """Test that tavily_search_tool creates a Tool with the given API key."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient') as mock_client_class:
            tool = tavily_search_tool('test-api-key')

            mock_client_class.assert_called_once_with('test-api-key')
            assert tool.name == 'tavily_search'

    def test_no_params_bound_exposes_all_in_schema(self):
        """Test that with no factory params, all parameters appear in the tool schema."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient'):
            tool = tavily_search_tool('test-api-key')

            schema_props = tool.function_schema.json_schema['properties']
            assert 'max_results' in schema_props
            assert 'include_domains' in schema_props
            assert 'exclude_domains' in schema_props

    def test_bound_params_hidden_from_schema(self):
        """Test that factory-provided params are excluded from the tool schema via partial."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient'):
            tool = tavily_search_tool(
                'test-api-key',
                max_results=5,
                include_domains=['arxiv.org'],
                exclude_domains=['medium.com'],
            )

            schema_props = tool.function_schema.json_schema['properties']
            assert 'max_results' not in schema_props
            assert 'include_domains' not in schema_props
            assert 'exclude_domains' not in schema_props
            # query should always be visible
            assert 'query' in schema_props
