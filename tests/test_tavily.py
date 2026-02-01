"""Tests for the Tavily search tool."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

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
        """Test basic search without domain filtering."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        results = await tool('test query')

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=None,
            exclude_domains=None,
        )
        assert len(results) == 2
        assert results[0]['title'] == 'Test Result 1'

    async def test_search_with_include_domains_at_call_time(self, mock_async_tavily_client: AsyncMock):
        """Test search with include_domains specified at call time."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool('test query', include_domains=['arxiv.org', 'github.com'])

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=['arxiv.org', 'github.com'],
            exclude_domains=None,
        )

    async def test_search_with_exclude_domains_at_call_time(self, mock_async_tavily_client: AsyncMock):
        """Test search with exclude_domains specified at call time."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool('test query', exclude_domains=['medium.com'])

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=None,
            exclude_domains=['medium.com'],
        )

    async def test_search_with_default_domains(self, mock_async_tavily_client: AsyncMock):
        """Test search with default domains set at initialization."""
        tool = TavilySearchTool(
            client=mock_async_tavily_client,
            include_domains=['docs.python.org'],
            exclude_domains=['stackoverflow.com'],
        )
        await tool('test query')

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=['docs.python.org'],
            exclude_domains=['stackoverflow.com'],
        )

    async def test_call_time_domains_override_defaults(self, mock_async_tavily_client: AsyncMock):
        """Test that call-time domain parameters override defaults."""
        tool = TavilySearchTool(
            client=mock_async_tavily_client,
            include_domains=['default.com'],
            exclude_domains=['default-exclude.com'],
        )
        await tool(
            'test query',
            include_domains=['override.com'],
            exclude_domains=['override-exclude.com'],
        )

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=['override.com'],
            exclude_domains=['override-exclude.com'],
        )

    async def test_search_with_all_parameters(self, mock_async_tavily_client: AsyncMock):
        """Test search with all parameters specified."""
        tool = TavilySearchTool(client=mock_async_tavily_client)
        await tool(
            'test query',
            search_deep='advanced',
            topic='news',
            time_range='week',
            include_domains=['news.com'],
            exclude_domains=['spam.com'],
        )

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='advanced',
            topic='news',
            time_range='week',
            include_domains=['news.com'],
            exclude_domains=['spam.com'],
        )

    async def test_empty_list_clears_default_domains(self, mock_async_tavily_client: AsyncMock):
        """Test that passing empty list clears the default domain filtering.

        This is distinct from passing None (which uses the default).
        Empty list means "explicitly no domain filtering for this call".
        """
        tool = TavilySearchTool(
            client=mock_async_tavily_client,
            include_domains=['default.com'],
            exclude_domains=['default-exclude.com'],
        )
        # Pass empty lists to explicitly clear domain filtering for this call
        await tool('test query', include_domains=[], exclude_domains=[])

        mock_async_tavily_client.search.assert_called_once_with(
            'test query',
            search_depth='basic',
            topic='general',
            time_range=None,
            include_domains=[],
            exclude_domains=[],
        )


class TestTavilySearchToolFactory:
    """Tests for tavily_search_tool factory function."""

    def test_creates_tool_with_api_key(self):
        """Test that tavily_search_tool creates a Tool with the given API key."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient') as mock_client_class:
            tool = tavily_search_tool('test-api-key')

            mock_client_class.assert_called_once_with('test-api-key')
            assert tool.name == 'tavily_search'

    def test_creates_tool_with_default_domains(self):
        """Test that tavily_search_tool accepts domain parameters."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient') as mock_client_class:
            tool = tavily_search_tool(
                'test-api-key',
                include_domains=['arxiv.org'],
                exclude_domains=['medium.com'],
            )

            mock_client_class.assert_called_once_with('test-api-key')
            assert tool.name == 'tavily_search'


class TestTavilySearchToolIntegration:
    """Integration tests for Tavily tool with Agent."""

    def test_tool_with_agent(self, mock_async_tavily_client: AsyncMock):
        """Test that the tool integrates correctly with an Agent."""
        with patch('pydantic_ai.common_tools.tavily.AsyncTavilyClient', return_value=mock_async_tavily_client):
            tool = tavily_search_tool('test-api-key', include_domains=['example.com'])

            agent = Agent(
                TestModel(),
                tools=[tool],
            )

            # Verify the tool is properly registered
            assert 'tavily_search' in agent._function_toolset.tools
