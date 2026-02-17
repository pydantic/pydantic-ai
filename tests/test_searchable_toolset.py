"""Unit tests for SearchableToolset."""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturn, ToolReturnPart
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
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool
    def get_time(timezone: str) -> str:  # pragma: no cover
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    @toolset.tool(defer_loading=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:  # pragma: no cover
        """Calculate monthly mortgage payment for a loan."""
        return 'Mortgage calculated'

    @toolset.tool(defer_loading=True)
    def stock_price(symbol: str) -> str:  # pragma: no cover
        """Get the current stock price for a symbol."""
        return f'Stock price for {symbol}'

    @toolset.tool(defer_loading=True)
    def crypto_price(coin: str) -> str:  # pragma: no cover
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

    # Need to call get_tools first to populate _deferred_tools
    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'mortgage'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot(
        {
            'message': "Found 1 tool(s) matching 'mortgage'",
            'tools': [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}],
        }
    )
    assert result.metadata == snapshot(['calculate_mortgage'])


async def test_searchable_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'STOCK'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: dict[str, Any] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv['tools']) == 1
    assert rv['tools'][0]['name'] == 'stock_price'


async def test_searchable_toolset_search_matches_description():
    """Test that search matches tool descriptions."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'cryptocurrency'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: dict[str, Any] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv['tools']) == 1
    assert rv['tools'][0]['name'] == 'crypto_price'


async def test_searchable_toolset_search_returns_no_matches():
    """Test that search returns empty list when no matches."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'nonexistent'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot({'message': "No tools found matching 'nonexistent'", 'tools': []})
    assert result.metadata == snapshot([])


async def test_searchable_toolset_search_empty_query():
    """Test that search with empty query raises ModelRetry."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match='Please provide a search query.'):
        await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': ''}, ctx, search_tool)


async def test_searchable_toolset_max_results():
    """Test that max_results limits the number of results."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset, max_results=1)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'price'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: dict[str, Any] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv['tools']) == 1


async def test_searchable_toolset_discovered_tools_available():
    """Test that discovered tools become available after search."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata=['calculate_mortgage'],
                ),
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
    def search_tools(query: str) -> str:  # pragma: no cover
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
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool
    def get_time(timezone: str) -> str:  # pragma: no cover
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['get_weather', 'get_time'])
    assert SEARCH_TOOLS_NAME not in tool_names


async def test_agent_always_wraps_in_searchable_toolset():
    """Test that agent always wraps toolset in SearchableToolset."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, SearchableToolset)


async def test_agent_wraps_in_searchable_with_deferred():
    """Test that agent wraps with SearchableToolset when there are deferred tools."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @agent.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float) -> str:  # pragma: no cover
        """Calculate mortgage payment."""
        return 'Calculated'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, SearchableToolset)


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
    def no_desc_tool() -> str:  # pragma: no cover
        return 'no description'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(SEARCH_TOOLS_NAME, {'query': 'no_desc'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot(
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
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata=['calculate_mortgage'],
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=SEARCH_TOOLS_NAME,
                    content={'message': "Found 1 tool(s) matching 'stock'", 'tools': [{'name': 'stock_price'}]},
                    metadata=['stock_price'],
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


async def test_function_toolset_all_deferred():
    """Test FunctionToolset with all tools having defer_loading=True."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(defer_loading=True)
    def deferred_tool1() -> str:  # pragma: no cover
        """First deferred tool."""
        return 'result1'

    @toolset.tool(defer_loading=True)
    def deferred_tool2() -> str:  # pragma: no cover
        """Second deferred tool."""
        return 'result2'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['search_tools'])
    assert 'deferred_tool1' not in tool_names
    assert 'deferred_tool2' not in tool_names


async def test_searchable_toolset_search_no_deferred_tools():
    """Test search when no deferred tools exist."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def normal_tool() -> str:  # pragma: no cover
        """A normal non-deferred tool."""
        return 'normal'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    # Call get_tools first to populate _deferred_tools
    await searchable.get_tools(ctx)
    result = await searchable._search_tools({'query': 'anything'}, ctx)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot({'message': 'No searchable tools available.', 'tools': []})


async def test_searchable_toolset_ignores_non_metadata_history():
    """Test that discovery only reads metadata, ignoring malformed content."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        # metadata is None (no discovered tools)
        ModelRequest(parts=[ToolReturnPart(tool_name=SEARCH_TOOLS_NAME, content={'message': 'hi'})]),
        # metadata is not a list
        ModelRequest(
            parts=[ToolReturnPart(tool_name=SEARCH_TOOLS_NAME, content={'tools': 'not a list'}, metadata='not a list')]
        ),
        # metadata contains non-string items
        ModelRequest(parts=[ToolReturnPart(tool_name=SEARCH_TOOLS_NAME, content={'tools': []}, metadata=[123, None])]),
        # valid metadata
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=SEARCH_TOOLS_NAME,
                    content={'message': 'found', 'tools': [{'name': 'calculate_mortgage'}]},
                    metadata=['calculate_mortgage'],
                ),
            ]
        ),
    ]
    ctx = build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert 'calculate_mortgage' in tools
    assert 'stock_price' not in tools
    assert 'crypto_price' not in tools
