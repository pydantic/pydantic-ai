"""Tests for Mem0Toolset integration."""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai import Agent, Mem0Toolset
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.mem0 import _import_error
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context(deps: Any, run_step: int = 0) -> RunContext[Any]:
    """Helper to build a RunContext for testing."""
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


def test_mem0_import_error():
    """Test that Mem0Toolset raises ImportError if mem0 is not installed."""
    if _import_error is None:
        pytest.skip('mem0 is installed, skipping import error test')

    with pytest.raises(ImportError, match='mem0 is not installed'):
        Mem0Toolset(api_key='test-key')


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_initialization():
    """Test Mem0Toolset initialization."""
    # Create mock client with async search method
    mock_client = AsyncMock()
    mock_client.search = AsyncMock()

    # Initialize toolset with mock client
    toolset = Mem0Toolset(client=mock_client)

    assert toolset.client is mock_client
    assert toolset._is_async is True


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_tool_registration():
    """Test that Mem0Toolset registers the expected tools."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    # Get tools from the toolset
    context = build_run_context('user_123')
    tools = await toolset.get_tools(context)

    # Should have two tools
    assert len(tools) == 2
    assert '_search_memory_impl' in tools
    assert '_save_memory_impl' in tools

    # Check tool definitions
    search_tool = tools['_search_memory_impl']
    save_tool = tools['_save_memory_impl']

    assert isinstance(search_tool.tool_def, ToolDefinition)
    assert isinstance(save_tool.tool_def, ToolDefinition)


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_string():
    """Test extracting user_id from string deps."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    user_id = toolset._extract_user_id('user_123')
    assert user_id == 'user_123'


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_attribute():
    """Test extracting user_id from object with user_id attribute."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    @dataclass
    class UserDeps:
        user_id: str

    deps = UserDeps(user_id='user_456')
    user_id = toolset._extract_user_id(deps)
    assert user_id == 'user_456'


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_method():
    """Test extracting user_id from object with get_user_id method."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    class UserDeps:
        def get_user_id(self) -> str:
            return 'user_789'

    deps = UserDeps()
    user_id = toolset._extract_user_id(deps)
    assert user_id == 'user_789'


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_method_wrong_type():
    """Test that get_user_id() must return a string."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    class UserDeps:
        def get_user_id(self) -> int:
            return 123

    deps = UserDeps()
    with pytest.raises(ValueError, match='deps.get_user_id\\(\\) must return a string'):
        toolset._extract_user_id(deps)


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_invalid():
    """Test that extracting user_id from invalid deps raises ValueError."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    with pytest.raises(ValueError, match='Cannot extract user_id from deps'):
        toolset._extract_user_id(123)


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_extract_user_id_wrong_type():
    """Test that user_id must be a string."""
    mock_client = AsyncMock()
    toolset = Mem0Toolset(client=mock_client)

    @dataclass
    class UserDeps:
        user_id: int

    deps = UserDeps(user_id=123)
    with pytest.raises(ValueError, match='deps.user_id must be a string'):
        toolset._extract_user_id(deps)


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_search_memory():
    """Test the search_memory tool."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(
        return_value={
            'results': [
                {'memory': 'User likes Python', 'score': 0.95},
                {'memory': 'User prefers dark mode', 'score': 0.88},
            ]
        }
    )

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._search_memory_impl(context, 'preferences')

    # Check that the search was called correctly
    mock_client.search.assert_called_once_with(query='preferences', user_id='user_123', limit=5)

    # Check the formatted output
    assert 'Found relevant memories:' in result
    assert 'User likes Python (relevance: 0.95)' in result
    assert 'User prefers dark mode (relevance: 0.88)' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_search_memory_no_results():
    """Test search_memory when no memories are found."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={'results': []})

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._search_memory_impl(context, 'nonexistent')

    assert result == 'No relevant memories found.'


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_search_memory_error():
    """Test search_memory error handling."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=Exception('API error'))

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._search_memory_impl(context, 'query')

    assert 'Error searching memories: API error' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_search_memory_list_response():
    """Test search_memory with list response format."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(
        return_value=[
            {'memory': 'Memory 1', 'score': 0.9},
            {'memory': 'Memory 2', 'score': 0.8},
        ]
    )

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._search_memory_impl(context, 'test')

    assert 'Found relevant memories:' in result
    assert 'Memory 1 (relevance: 0.90)' in result
    assert 'Memory 2 (relevance: 0.80)' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_search_memory_unexpected_response():
    """Test search_memory with unexpected response format."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value='unexpected string response')

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._search_memory_impl(context, 'test')

    assert 'Error: Unexpected response format from Mem0' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_save_memory():
    """Test the save_memory tool."""
    mock_client = AsyncMock()
    mock_client.add = AsyncMock(return_value=None)

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._save_memory_impl(context, 'User loves Python')

    # Check that add was called correctly
    mock_client.add.assert_called_once_with(
        messages=[{'role': 'user', 'content': 'User loves Python'}],
        user_id='user_123',
    )

    assert result == 'Successfully saved to memory: User loves Python'


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_save_memory_error():
    """Test save_memory error handling."""
    mock_client = AsyncMock()
    mock_client.add = AsyncMock(side_effect=Exception('Storage error'))

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    result = await toolset._save_memory_impl(context, 'test content')

    assert 'Error saving to memory: Storage error' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_with_agent():
    """Test Mem0Toolset integration with an Agent."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={'results': []})
    mock_client.add = AsyncMock(return_value=None)

    toolset: Mem0Toolset[str] = Mem0Toolset(client=mock_client)
    agent = Agent('test', deps_type=str, toolsets=[toolset])

    # Run agent and verify toolset is registered
    result = await agent.run('test', deps='user_123')

    # The agent should have access to the memory tools
    assert result is not None


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_custom_limit():
    """Test Mem0Toolset with custom limit parameter."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={'results': []})

    toolset = Mem0Toolset(client=mock_client, limit=10)
    context = build_run_context('user_123')

    await toolset._search_memory_impl(context, 'test')

    # Should use custom limit
    mock_client.search.assert_called_once_with(query='test', user_id='user_123', limit=10)


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_sync_client():
    """Test Mem0Toolset with synchronous client."""
    mock_client = MagicMock()
    mock_client.search.return_value = {'results': []}
    mock_client.add.return_value = None

    toolset = Mem0Toolset(client=mock_client)
    assert toolset._is_async is False

    context = build_run_context('user_123')

    # Search should still work with sync client
    result = await toolset._search_memory_impl(context, 'test')
    assert 'No relevant memories found.' in result

    # Save should still work with sync client
    result = await toolset._save_memory_impl(context, 'test content')
    assert 'Successfully saved to memory' in result


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_with_dataclass_deps():
    """Test Mem0Toolset with dataclass deps containing user_id."""

    @dataclass
    class UserSession:
        user_id: str
        session_id: str

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={'results': []})
    mock_client.add = AsyncMock(return_value=None)

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context(UserSession(user_id='alice', session_id='session_1'))

    # Test search
    await toolset._search_memory_impl(context, 'test')
    mock_client.search.assert_called_once_with(query='test', user_id='alice', limit=5)

    # Test save
    await toolset._save_memory_impl(context, 'content')
    mock_client.add.assert_called_once_with(messages=[{'role': 'user', 'content': 'content'}], user_id='alice')


@pytest.mark.skipif(_import_error is not None, reason='mem0 is not installed')
async def test_mem0_toolset_tool_manager_integration():
    """Test that Mem0Toolset works correctly with ToolManager."""
    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value={'results': [{'memory': 'Test memory', 'score': 0.9}]})

    toolset = Mem0Toolset(client=mock_client)
    context = build_run_context('user_123')

    tool_manager = await ToolManager(toolset).for_run_step(context)

    # Verify tools are available
    assert len(tool_manager.tool_defs) == 2

    # Get tool names
    tool_names = [td.name for td in tool_manager.tool_defs]
    assert '_search_memory_impl' in tool_names
    assert '_save_memory_impl' in tool_names
