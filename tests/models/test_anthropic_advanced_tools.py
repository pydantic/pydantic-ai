"""Tests for Anthropic advanced tool use features.

Tests for:
- defer_loading on ToolDefinition
- allowed_callers for programmatic tool calling
- input_examples for tool use examples
- ToolSearchTool built-in tool
- ProgrammaticCodeExecutionTool built-in tool
"""

from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.builtin_tools import ProgrammaticCodeExecutionTool, ToolSearchTool
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, AsyncAnthropic
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


class TestToolDefinitionAdvancedFields:
    """Tests for ToolDefinition advanced fields."""

    def test_tool_definition_defer_loading_default(self):
        """Test that defer_loading defaults to False."""
        tool_def = ToolDefinition(name='test_tool')
        assert tool_def.defer_loading is False

    def test_tool_definition_defer_loading_true(self):
        """Test setting defer_loading to True."""
        tool_def = ToolDefinition(name='test_tool', defer_loading=True)
        assert tool_def.defer_loading is True

    def test_tool_definition_allowed_callers_default(self):
        """Test that allowed_callers defaults to None."""
        tool_def = ToolDefinition(name='test_tool')
        assert tool_def.allowed_callers is None

    def test_tool_definition_allowed_callers_set(self):
        """Test setting allowed_callers."""
        tool_def = ToolDefinition(
            name='test_tool',
            allowed_callers=['code_execution_20250825'],
        )
        assert tool_def.allowed_callers == ['code_execution_20250825']

    def test_tool_definition_input_examples_default(self):
        """Test that input_examples defaults to None."""
        tool_def = ToolDefinition(name='test_tool')
        assert tool_def.input_examples is None

    def test_tool_definition_input_examples_set(self):
        """Test setting input_examples."""
        examples = [
            {'param1': 'value1', 'param2': 123},
            {'param1': 'value2'},
        ]
        tool_def = ToolDefinition(name='test_tool', input_examples=examples)
        assert tool_def.input_examples == examples


class TestToolAdvancedFields:
    """Tests for Tool class with advanced fields."""

    def test_tool_defer_loading_default(self):
        """Test that Tool.defer_loading defaults to False."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool)
        assert tool.defer_loading is False
        assert tool.tool_def.defer_loading is False

    def test_tool_defer_loading_true(self):
        """Test setting Tool.defer_loading to True."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool, defer_loading=True)
        assert tool.defer_loading is True
        assert tool.tool_def.defer_loading is True

    def test_tool_allowed_callers(self):
        """Test setting Tool.allowed_callers."""

        def my_tool(x: int) -> str:
            return str(x)

        tool = Tool(my_tool, allowed_callers=['code_execution_20250825'])
        assert tool.allowed_callers == ['code_execution_20250825']
        assert tool.tool_def.allowed_callers == ['code_execution_20250825']

    def test_tool_input_examples(self):
        """Test setting Tool.input_examples."""

        def my_tool(x: int) -> str:
            return str(x)

        examples = [{'x': 1}, {'x': 42}]
        tool = Tool(my_tool, input_examples=examples)
        assert tool.input_examples == examples
        assert tool.tool_def.input_examples == examples


class TestBuiltinTools:
    """Tests for new builtin tools."""

    def test_tool_search_tool_regex(self):
        """Test ToolSearchTool with regex search type."""
        tool = ToolSearchTool(search_type='regex')
        assert tool.search_type == 'regex'
        assert tool.kind == 'tool_search'

    def test_tool_search_tool_bm25(self):
        """Test ToolSearchTool with BM25 search type."""
        tool = ToolSearchTool(search_type='bm25')
        assert tool.search_type == 'bm25'
        assert tool.kind == 'tool_search'

    def test_tool_search_tool_default(self):
        """Test ToolSearchTool default search type."""
        tool = ToolSearchTool()
        assert tool.search_type == 'regex'

    def test_programmatic_code_execution_tool(self):
        """Test ProgrammaticCodeExecutionTool."""
        tool = ProgrammaticCodeExecutionTool()
        assert tool.kind == 'programmatic_code_execution'


class MockAnthropic:
    """Mock Anthropic client for testing."""

    def __init__(self):
        self.chat_completion_kwargs: list[dict[str, Any]] = []
        self.base_url = 'https://api.anthropic.com'

    @property
    def beta(self):
        return self

    @property
    def messages(self):
        return self

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})
        return BetaMessage(
            id='123',
            content=[BetaTextBlock(text='Hello', type='text')],
            model='claude-sonnet-4-5-20250929',
            role='assistant',
            stop_reason='end_turn',
            type='message',
            usage=BetaUsage(input_tokens=10, output_tokens=5),
        )


class TestAnthropicModelAdvancedToolUse:
    """Tests for AnthropicModel with advanced tool use features."""

    def test_map_tool_definition_basic(self):
        """Test basic tool definition mapping."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}},
        )
        result = AnthropicModel._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['name'] == 'test_tool'
        assert result_dict['description'] == 'A test tool'
        assert 'defer_loading' not in result_dict
        assert 'allowed_callers' not in result_dict
        assert 'input_examples' not in result_dict

    def test_map_tool_definition_with_defer_loading(self):
        """Test tool definition mapping with defer_loading."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            defer_loading=True,
        )
        result = AnthropicModel._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['defer_loading'] is True

    def test_map_tool_definition_with_allowed_callers(self):
        """Test tool definition mapping with allowed_callers."""
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            allowed_callers=['code_execution_20250825'],
        )
        result = AnthropicModel._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['allowed_callers'] == ['code_execution_20250825']

    def test_map_tool_definition_with_input_examples(self):
        """Test tool definition mapping with input_examples."""
        examples = [{'x': 1}, {'x': 2}]
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            input_examples=examples,
        )
        result = AnthropicModel._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['input_examples'] == examples

    def test_map_tool_definition_all_advanced_fields(self):
        """Test tool definition mapping with all advanced fields."""
        examples = [{'x': 1}]
        tool_def = ToolDefinition(
            name='test_tool',
            description='A test tool',
            defer_loading=True,
            allowed_callers=['code_execution_20250825'],
            input_examples=examples,
        )
        result = AnthropicModel._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]
        result_dict = cast(dict[str, Any], result)
        assert result_dict['defer_loading'] is True
        assert result_dict['allowed_callers'] == ['code_execution_20250825']
        assert result_dict['input_examples'] == examples

    async def test_add_builtin_tools_tool_search_regex(self):
        """Test adding ToolSearchTool with regex type."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        model_request_params = ModelRequestParameters(
            function_tools=[],
            builtin_tools=[ToolSearchTool(search_type='regex')],
        )

        tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert len(tools) == 1
        tool_dict = cast(dict[str, Any], tools[0])
        assert tool_dict['type'] == 'tool_search_tool_regex_20251119'
        assert tool_dict['name'] == 'tool_search_tool_regex'
        assert 'advanced-tool-use-2025-11-20' in beta_features

    async def test_add_builtin_tools_tool_search_bm25(self):
        """Test adding ToolSearchTool with BM25 type."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        model_request_params = ModelRequestParameters(
            function_tools=[],
            builtin_tools=[ToolSearchTool(search_type='bm25')],
        )

        tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert len(tools) == 1
        tool_dict = cast(dict[str, Any], tools[0])
        assert tool_dict['type'] == 'tool_search_tool_bm25_20251119'
        assert tool_dict['name'] == 'tool_search_tool_bm25'
        assert 'advanced-tool-use-2025-11-20' in beta_features

    async def test_add_builtin_tools_programmatic_code_execution(self):
        """Test adding ProgrammaticCodeExecutionTool."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        model_request_params = ModelRequestParameters(
            function_tools=[],
            builtin_tools=[ProgrammaticCodeExecutionTool()],
        )

        tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert len(tools) == 1
        tool_dict = cast(dict[str, Any], tools[0])
        assert tool_dict['type'] == 'code_execution_20250825'
        assert tool_dict['name'] == 'code_execution'
        assert 'advanced-tool-use-2025-11-20' in beta_features

    async def test_beta_header_added_for_defer_loading(self):
        """Test that beta header is added when tool uses defer_loading."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        tool_def = ToolDefinition(
            name='deferred_tool',
            description='A deferred tool',
            defer_loading=True,
        )
        model_request_params = ModelRequestParameters(
            function_tools=[tool_def],
            builtin_tools=[],
        )

        _tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert 'advanced-tool-use-2025-11-20' in beta_features

    async def test_beta_header_added_for_allowed_callers(self):
        """Test that beta header is added when tool uses allowed_callers."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        tool_def = ToolDefinition(
            name='callable_tool',
            description='A callable tool',
            allowed_callers=['code_execution_20250825'],
        )
        model_request_params = ModelRequestParameters(
            function_tools=[tool_def],
            builtin_tools=[],
        )

        _tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert 'advanced-tool-use-2025-11-20' in beta_features

    async def test_beta_header_added_for_input_examples(self):
        """Test that beta header is added when tool uses input_examples."""
        mock_client = cast(AsyncAnthropic, MockAnthropic())
        model = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(anthropic_client=mock_client))

        tool_def = ToolDefinition(
            name='example_tool',
            description='A tool with examples',
            input_examples=[{'x': 1}],
        )
        model_request_params = ModelRequestParameters(
            function_tools=[tool_def],
            builtin_tools=[],
        )

        _tools, _mcp_servers, beta_features = model._add_builtin_tools([], model_request_params)  # pyright: ignore[reportPrivateUsage]

        assert 'advanced-tool-use-2025-11-20' in beta_features


class TestAgentWithAdvancedToolUse:
    """Tests for Agent with advanced tool use features."""

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

    def test_agent_with_allowed_callers_tool(self):
        """Test creating an agent with a tool that has allowed_callers."""

        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        agent = Agent(
            'test',
            tools=[Tool(my_tool, allowed_callers=['code_execution_20250825'])],
        )

        # Verify the tool was registered with allowed_callers
        tool = agent._function_toolset.tools.get('my_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.allowed_callers == ['code_execution_20250825']

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

    def test_agent_tool_plain_decorator_with_defer_loading(self):
        """Test the @agent.tool_plain decorator with defer_loading."""
        agent: Agent[None, str] = Agent('test')

        @agent.tool_plain(defer_loading=True)
        def my_deferred_tool(x: int) -> str:
            """A deferred tool."""
            return str(x)

        tool = agent._function_toolset.tools.get('my_deferred_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.defer_loading is True

    def test_agent_tool_plain_decorator_with_allowed_callers(self):
        """Test the @agent.tool_plain decorator with allowed_callers."""
        agent: Agent[None, str] = Agent('test')

        @agent.tool_plain(allowed_callers=['code_execution_20250825'])
        def my_callable_tool(x: int) -> str:
            """A callable tool."""
            return str(x)

        tool = agent._function_toolset.tools.get('my_callable_tool')  # pyright: ignore[reportPrivateUsage]
        assert tool is not None
        assert tool.allowed_callers == ['code_execution_20250825']

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
