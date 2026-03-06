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
from pydantic_ai.toolsets._searchable import (
    _DISCOVERED_TOOLS_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    _SEARCH_TOOLS_NAME,  # pyright: ignore[reportPrivateUsage]
    SearchableToolset,
)
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

    @toolset.tool(lazy=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:  # pragma: no cover
        """Calculate monthly mortgage payment for a loan."""
        return 'Mortgage calculated'

    @toolset.tool(lazy=True)
    def stock_price(symbol: str) -> str:  # pragma: no cover
        """Get the current stock price for a symbol."""
        return f'Stock price for {symbol}'

    @toolset.tool(lazy=True)
    def crypto_price(coin: str) -> str:  # pragma: no cover
        """Get the current cryptocurrency price."""
        return f'Crypto price for {coin}'

    return toolset


async def test_searchable_toolset_filters_lazy_tools():
    """Test that lazy tools are not exposed initially."""
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
    """Test that search_tools returns matching lazy tools."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'mortgage'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot(
        {
            'message': "Found 1 tool(s) matching 'mortgage'",
            'tools': [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}],
        }
    )
    assert result.metadata == snapshot({'discovered_tools': ['calculate_mortgage']})


async def test_searchable_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'STOCK'}, ctx, search_tool)
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
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'cryptocurrency'}, ctx, search_tool)
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
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'nonexistent'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot({'message': "No tools found matching 'nonexistent'", 'tools': []})
    assert result.metadata == snapshot({'discovered_tools': []})


async def test_searchable_toolset_search_empty_query():
    """Test that search with empty query raises ModelRetry."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match='Please provide a search query.'):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': ''}, ctx, search_tool)


async def test_searchable_toolset_max_results():
    """Test that results are capped at `_MAX_SEARCH_RESULTS`."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'price'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: dict[str, Any] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv['tools']) == 2


async def test_searchable_toolset_discovered_tools_available():
    """Test that discovered tools become available after search."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
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
    """Test that `UserError` is raised if a tool is named 'search_tools' and lazy tools exist."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool
    def search_tools(query: str) -> str:  # pragma: no cover
        """Search for tools."""
        return 'search result'

    @toolset.tool(lazy=True)
    def lazy_tool() -> str:  # pragma: no cover
        """A lazy tool to trigger search injection."""
        return 'lazy'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    with pytest.raises(UserError, match="Tool name 'search_tools' is reserved"):
        await searchable.get_tools(ctx)


async def test_searchable_toolset_no_lazy_tools_returns_all():
    """Test that when there are no lazy tools, all tools are returned without search_tools."""
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
    assert _SEARCH_TOOLS_NAME not in tool_names


async def test_agent_always_wraps_in_searchable_toolset():
    """Test that agent always wraps toolset in SearchableToolset."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, SearchableToolset)


async def test_agent_wraps_in_searchable_with_lazy():
    """Test that agent wraps with SearchableToolset when there are lazy tools."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @agent.tool_plain(lazy=True)
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

    @toolset.tool(lazy=True)
    def no_desc_tool() -> str:  # pragma: no cover
        return 'no description'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'no_desc'}, ctx, search_tool)
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
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'message': "Found 1 tool(s) matching 'stock'", 'tools': [{'name': 'stock_price'}]},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['stock_price']},
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


async def test_function_toolset_all_lazy():
    """Test FunctionToolset with all tools having lazy=True."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool(lazy=True)
    def lazy_tool1() -> str:  # pragma: no cover
        """First lazy tool."""
        return 'result1'

    @toolset.tool(lazy=True)
    def lazy_tool2() -> str:  # pragma: no cover
        """Second lazy tool."""
        return 'result2'

    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['search_tools'])
    assert 'lazy_tool1' not in tool_names
    assert 'lazy_tool2' not in tool_names


async def test_searchable_toolset_ignores_non_metadata_history():
    """Test that discovery only reads metadata, ignoring malformed content."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        # metadata is None (no discovered tools)
        ModelRequest(parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'message': 'hi'})]),
        # metadata is not a dict
        ModelRequest(
            parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'tools': 'not a list'}, metadata='not a dict')]
        ),
        # metadata is a dict but value is not a list
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': []},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: 'not a list'},
                )
            ]
        ),
        # metadata contains non-string items in the list
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': []},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: [123, None]},
                )
            ]
        ),
        # valid metadata
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'message': 'found', 'tools': [{'name': 'calculate_mortgage'}]},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
                ),
            ]
        ),
    ]
    ctx = build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert 'calculate_mortgage' in tools
    assert 'stock_price' not in tools
    assert 'crypto_price' not in tools


async def test_call_tool_returns_tool_return_with_metadata():
    """Test that call_tool for search_tools returns a ToolReturn with metadata listing matched tools."""
    toolset = create_function_toolset()
    searchable = SearchableToolset(wrapped=toolset)
    ctx = build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'query': 'mortgage'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value={
                'message': "Found 1 tool(s) matching 'mortgage'",
                'tools': [
                    {'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}
                ],
            },
            metadata={'discovered_tools': ['calculate_mortgage']},
        )
    )
