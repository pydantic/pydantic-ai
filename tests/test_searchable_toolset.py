"""Unit tests for SearchableToolset."""

from __future__ import annotations

from typing import TypeVar

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets._searchable import SEARCH_TOOLS_NAME, SearchableToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0, messages: list[ModelMessage] | None = None) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=messages or [],
        run_step=run_step,
    )


def create_function_toolset() -> FunctionToolset[None]:
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool
    def get_time(timezone: str) -> str:
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    @toolset.tool(defer_loading=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:
        """Calculate monthly mortgage payment for a loan."""
        return 'Mortgage calculated'

    @toolset.tool(defer_loading=True)
    def stock_price(symbol: str) -> str:
        """Get the current stock price for a symbol."""
        return f'Stock price for {symbol}'

    @toolset.tool(defer_loading=True)
    def crypto_price(coin: str) -> str:
        """Get the current cryptocurrency price."""
        return f'Crypto price for {coin}'

    return toolset


async def test_searchable_toolset_filters_deferred_tools():
    """Test that deferred tools are not exposed initially."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['search_tools', 'get_weather', 'get_time'])
    assert 'calculate_mortgage' not in tool_names
    assert 'stock_price' not in tool_names
    assert 'crypto_price' not in tool_names


async def test_searchable_toolset_search_returns_matching_tools():
    """Test that search_tools returns matching deferred tools."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'mortgage'}, ctx, search_tool)
    assert result == snapshot(
        {
            'message': "Found 1 tool(s) matching 'mortgage'",
            'tools': [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}],
        }
    )


async def test_searchable_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'STOCK'}, ctx, search_tool)
    assert len(result['tools']) == 1
    assert result['tools'][0]['name'] == 'stock_price'


async def test_searchable_toolset_search_matches_description():
    """Test that search matches tool descriptions."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'cryptocurrency'}, ctx, search_tool)
    assert len(result['tools']) == 1
    assert result['tools'][0]['name'] == 'crypto_price'


async def test_searchable_toolset_search_returns_no_matches():
    """Test that search returns empty list when no matches."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'nonexistent'}, ctx, search_tool)
    assert result == snapshot({'message': "No tools found matching 'nonexistent'", 'tools': []})


async def test_searchable_toolset_search_empty_query():
    """Test that search with empty query returns helpful message."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': ''}, ctx, search_tool)
    assert result == snapshot({'message': 'Please provide a search query.', 'tools': []})


async def test_searchable_toolset_max_results():
    """Test that max_results limits the number of results."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset, max_results=1)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'price'}, ctx, search_tool)
    assert len(result['tools']) == 1


async def test_searchable_toolset_discovered_tools_available():
    """Test that discovered tools become available after search."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    search_result = {
        'message': "Found 1 tool(s) matching 'mortgage'",
        'tools': [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}],
    }
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name=SEARCH_TOOLS_NAME, content=search_result),
            ]
        )
    ]
    ctx = build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert 'calculate_mortgage' in tool_names
    assert 'stock_price' not in tool_names


async def test_searchable_toolset_reserved_name_collision():
    """Test that UserError is raised if a tool is named 'search_tools'."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def search_tools(query: str) -> str:
        """Search for tools."""
        return 'search result'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    with pytest.raises(UserError, match="Tool name 'search_tools' is reserved"):
        await searchable.get_tools(ctx)


async def test_searchable_toolset_no_deferred_tools_returns_all():
    """Test that when there are no deferred tools, all tools are returned without search_tools."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool
    def get_time(timezone: str) -> str:
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['get_weather', 'get_time'])
    assert SEARCH_TOOLS_NAME not in tool_names


async def test_searchable_toolset_has_deferred_tools():
    """Test the has_deferred_tools method."""
    toolset_with_deferred = create_function_toolset()
    searchable_with_deferred = SearchableToolset(wrapped=toolset_with_deferred)
    assert searchable_with_deferred.has_deferred_tools() is True

    toolset_without_deferred: FunctionToolset[None] = FunctionToolset()

    @toolset_without_deferred.tool
    def normal_tool() -> str:
        return 'normal'

    searchable_without_deferred = SearchableToolset(wrapped=toolset_without_deferred)
    assert searchable_without_deferred.has_deferred_tools() is False


async def test_agent_auto_injects_searchable_toolset():
    """Test that agent automatically wraps toolset in SearchableToolset when there are deferred tools."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @agent.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float) -> str:
        """Calculate mortgage payment."""
        return 'Calculated'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, SearchableToolset)


async def test_agent_does_not_inject_searchable_without_deferred():
    """Test that agent does not wrap with SearchableToolset when no deferred tools."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Weather in {city}'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert not isinstance(toolset, SearchableToolset)


async def test_tool_manager_with_searchable_toolset():
    """Test that ToolManager works correctly with SearchableToolset."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tool_manager = ToolManager[None](searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)

    tool_names = [t.name for t in run_step_toolset.tool_defs]
    assert 'search_tools' in tool_names
    assert 'get_weather' in tool_names
    assert 'calculate_mortgage' not in tool_names

    result = await run_step_toolset.handle_call(ToolCallPart(tool_name='search_tools', args={'query': 'mortgage'}))
    assert 'calculate_mortgage' in str(result)


async def test_searchable_toolset_tool_with_none_description():
    """Test that tools with None description are handled correctly in search."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(defer_loading=True)
    def no_desc_tool() -> str:
        return 'no description'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'no_desc'}, ctx, search_tool)
    assert result == snapshot(
        {'message': "Found 1 tool(s) matching 'no_desc'", 'tools': [{'name': 'no_desc_tool', 'description': None}]}
    )


async def test_searchable_toolset_multiple_searches_accumulate():
    """Test that tools discovered in multiple searches accumulate correctly."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment.'}],
                    },
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'stock'",
                        'tools': [{'name': 'stock_price', 'description': 'Get stock price.'}],
                    },
                ),
            ]
        ),
    ]
    ctx = build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert 'calculate_mortgage' in tool_names
    assert 'stock_price' in tool_names
    assert 'crypto_price' not in tool_names


async def test_tool_search_config_max_results_propagates():
    """Test that ToolSearchTool max_results config propagates to SearchableToolset."""
    from pydantic_ai.builtin_tools import ToolSearchTool

    agent: Agent[None, str] = Agent('test', builtin_tools=[ToolSearchTool(max_results=3)])

    @agent.tool_plain(defer_loading=True)
    def tool1() -> str:
        return 'tool1'

    @agent.tool_plain(defer_loading=True)
    def tool2() -> str:
        return 'tool2'

    @agent.tool_plain(defer_loading=True)
    def tool3() -> str:
        return 'tool3'

    @agent.tool_plain(defer_loading=True)
    def tool4() -> str:
        return 'tool4'

    tool_search_config = agent._get_tool_search_config()  # pyright: ignore[reportPrivateUsage]
    assert tool_search_config is not None
    assert tool_search_config.max_results == 3


async def test_function_toolset_all_deferred():
    """Test FunctionToolset with all tools having defer_loading=True."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(defer_loading=True)
    def deferred_tool1() -> str:
        """First deferred tool."""
        return 'result1'

    @toolset.tool(defer_loading=True)
    def deferred_tool2() -> str:
        """Second deferred tool."""
        return 'result2'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['search_tools'])
    assert 'deferred_tool1' not in tool_names
    assert 'deferred_tool2' not in tool_names
