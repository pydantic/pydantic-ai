"""Tests for Anthropic ToolSearchTool and defer_loading features.

These features enable dynamic tool discovery without loading all definitions upfront.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.tools import ToolDefinition

from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.builtin_tools import ToolSearchTool
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


class TestToolDefinitionDeferLoading:
    """Tests for ToolDefinition.defer_loading field."""

    def test_defer_loading_defaults_to_false(self):
        """Test that defer_loading defaults to False."""
        tool_def = ToolDefinition(name='test_tool')
        assert tool_def.defer_loading is False

    def test_defer_loading_can_be_set(self):
        """Test setting defer_loading."""
        tool_def = ToolDefinition(name='test_tool', defer_loading=True)
        assert tool_def.defer_loading is True


class TestToolDeferLoading:
    """Tests for Tool class with defer_loading."""

    def test_tool_defer_loading_default(self):
        """Test that Tool.defer_loading defaults to False."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool)
        assert tool.defer_loading is False
        assert tool.tool_def.defer_loading is False

    def test_tool_defer_loading_set(self):
        """Test setting Tool.defer_loading."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool, defer_loading=True)
        assert tool.defer_loading is True
        assert tool.tool_def.defer_loading is True


class TestToolSearchTool:
    """Tests for ToolSearchTool builtin tool."""

    def test_tool_search_tool_defaults(self):
        """Test ToolSearchTool defaults."""
        tool = ToolSearchTool()
        assert tool.kind == 'tool_search'
        assert tool.search_type is None

    def test_tool_search_tool_with_regex(self):
        """Test ToolSearchTool with regex search type."""
        tool = ToolSearchTool(search_type='regex')
        assert tool.search_type == 'regex'

    def test_tool_search_tool_with_bm25(self):
        """Test ToolSearchTool with bm25 search type."""
        tool = ToolSearchTool(search_type='bm25')
        assert tool.search_type == 'bm25'


class TestAnthropicMapToolDefinitionDeferLoading:
    """Tests for AnthropicModel._map_tool_definition with defer_loading."""

    def test_map_tool_definition_without_defer_loading(self):
        """Test tool definition mapping without defer_loading."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
        )
        c = completion_message(
            [BetaTextBlock(text='Hello', type='text')],
            BetaUsage(input_tokens=5, output_tokens=10),
        )
        mock_client = MockAnthropic.create_mock(c)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
        result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert 'defer_loading' not in result_dict

    def test_map_tool_definition_with_defer_loading(self):
        """Test tool definition mapping with defer_loading."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            defer_loading=True,
        )
        c = completion_message(
            [BetaTextBlock(text='Hello', type='text')],
            BetaUsage(input_tokens=5, output_tokens=10),
        )
        mock_client = MockAnthropic.create_mock(c)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
        result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['defer_loading'] is True


class TestAgentWithDeferLoading:
    """Tests for Agent with defer_loading on tools."""

    def test_agent_with_defer_loading_tool(self):
        """Test creating an agent with a tool that has defer_loading."""

        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        agent = Agent(
            'test',
            tools=[Tool(my_tool, defer_loading=True)],
        )

        # Verify the tool was registered with defer_loading
        tool = agent._function_toolset.tools.get('my_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.defer_loading is True

    def test_agent_tool_plain_decorator_with_defer_loading(self):
        """Test the @agent.tool_plain decorator with defer_loading."""
        agent: Agent[None, str] = Agent('test')

        @agent.tool_plain(defer_loading=True)
        def my_deferred_tool(x: int) -> str:
            """A tool with defer_loading."""
            return str(x)

        tool = agent._function_toolset.tools.get('my_deferred_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.defer_loading is True


class TestAgentWithToolSearchTool:
    """Tests for Agent with ToolSearchTool."""

    def test_agent_with_tool_search_tool(self):
        """Test creating an agent with ToolSearchTool."""
        agent = Agent(
            'test',
            builtin_tools=[ToolSearchTool()],
        )

        assert any(isinstance(t, ToolSearchTool) for t in agent._builtin_tools)  # pyright: ignore[reportPrivateUsage]

    def test_agent_with_tool_search_tool_bm25(self):
        """Test creating an agent with ToolSearchTool using bm25."""
        agent = Agent(
            'test',
            builtin_tools=[ToolSearchTool(search_type='bm25')],
        )

        tool_search = next(t for t in agent._builtin_tools if isinstance(t, ToolSearchTool))  # pyright: ignore[reportPrivateUsage]
        assert tool_search.search_type == 'bm25'
