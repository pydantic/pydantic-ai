"""Tests for Anthropic input_examples feature on ToolDefinition.

This feature allows providing example inputs to help the model understand
correct tool usage patterns.
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

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


class TestToolDefinitionInputExamples:
    """Tests for ToolDefinition.input_examples field."""

    def test_input_examples_defaults_to_none(self):
        """Test that input_examples defaults to None."""
        tool_def = ToolDefinition(name='test_tool')
        assert tool_def.input_examples is None

    def test_input_examples_can_be_set(self):
        """Test setting input_examples."""
        examples = [
            {'param1': 'value1', 'param2': 123},
            {'param1': 'value2'},
        ]
        tool_def = ToolDefinition(name='test_tool', input_examples=examples)
        assert tool_def.input_examples == examples


class TestToolInputExamples:
    """Tests for Tool class with input_examples."""

    def test_tool_input_examples_default(self):
        """Test that Tool.input_examples defaults to None."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool)
        assert tool.input_examples is None
        assert tool.tool_def.input_examples is None

    def test_tool_input_examples_set(self):
        """Test setting Tool.input_examples."""

        def my_tool(x: int) -> str:
            return str(x)

        examples = [{'x': 1}, {'x': 42}]
        tool = Tool(my_tool, input_examples=examples)
        assert tool.input_examples == examples
        assert tool.tool_def.input_examples == examples


class TestAnthropicMapToolDefinition:
    """Tests for AnthropicModel._map_tool_definition with input_examples."""

    def test_map_tool_definition_without_input_examples(self):
        """Test tool definition mapping without input_examples."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
        )
        c = completion_message(
            [BetaTextBlock(text='Hello', type='text')],
            BetaUsage(input_tokens=5, output_tokens=10),
        )
        mock_client = MockAnthropic.create_mock(c)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
        result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['name'] == 'test_tool'
        assert result_dict['description'] == 'A test tool'
        assert 'input_examples' not in result_dict

    def test_map_tool_definition_with_input_examples(self):
        """Test tool definition mapping with input_examples."""
        examples = [{'x': 1}, {'x': 2}]
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            input_examples=examples,
        )
        c = completion_message(
            [BetaTextBlock(text='Hello', type='text')],
            BetaUsage(input_tokens=5, output_tokens=10),
        )
        mock_client = MockAnthropic.create_mock(c)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
        result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['input_examples'] == examples


class TestAgentWithInputExamples:
    """Tests for Agent with input_examples on tools."""

    def test_agent_with_input_examples_tool(self):
        """Test creating an agent with a tool that has input_examples."""

        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        examples = [{'x': 1}, {'x': 42}]
        agent = Agent(
            'test',
            tools=[Tool(my_tool, input_examples=examples)],
        )

        # Verify the tool was registered with input_examples
        tool = agent._function_toolset.tools.get('my_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.input_examples == examples

    def test_agent_tool_plain_decorator_with_input_examples(self):
        """Test the @agent.tool_plain decorator with input_examples."""
        agent: Agent[None, str] = Agent('test')

        examples = [{'x': 1}, {'x': 42}]

        @agent.tool_plain(input_examples=examples)
        def my_example_tool(x: int) -> str:
            """A tool with examples."""
            return str(x)

        tool = agent._function_toolset.tools.get('my_example_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.input_examples == examples
