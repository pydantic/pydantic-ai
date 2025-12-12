"""Tests for Anthropic programmatically_callable (Programmatic Tool Calling) feature."""

from __future__ import annotations as _annotations

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai import Agent
    from pydantic_ai.builtin_tools import CodeExecutionTool
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from ..test_anthropic import MockAnthropic, completion_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


async def test_programmatically_callable_true(allow_model_requests: None):
    """Test that programmatically_callable=True maps to allowed_callers with both direct and code_execution."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    agent = Agent(model)

    @agent.tool_plain(programmatically_callable=True)
    def my_tool(x: int) -> int:
        """A tool that can be called from code execution."""
        return x * 2

    await agent.run('test')

    # Check that the tool was configured with allowed_callers
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    # Find the my_tool definition
    my_tool_def = next((t for t in tools if t.get('name') == 'my_tool'), None)
    assert my_tool_def is not None
    assert my_tool_def.get('allowed_callers') == ['direct', 'code_execution_20250825']


async def test_programmatically_callable_only(allow_model_requests: None):
    """Test that programmatically_callable='only' maps to allowed_callers with only code_execution."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    agent = Agent(model)

    @agent.tool_plain(programmatically_callable='only')
    def my_tool(x: int) -> int:
        """A tool that can only be called from code execution."""
        return x * 2

    await agent.run('test')

    # Check that the tool was configured with allowed_callers
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    # Find the my_tool definition
    my_tool_def = next((t for t in tools if t.get('name') == 'my_tool'), None)
    assert my_tool_def is not None
    assert my_tool_def.get('allowed_callers') == ['code_execution_20250825']


async def test_programmatically_callable_false(allow_model_requests: None):
    """Test that programmatically_callable=False (default) doesn't add allowed_callers."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    agent = Agent(model)

    @agent.tool_plain
    def my_tool(x: int) -> int:
        """A regular tool."""
        return x * 2

    await agent.run('test')

    # Check that the tool was NOT configured with allowed_callers
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    # Find the my_tool definition
    my_tool_def = next((t for t in tools if t.get('name') == 'my_tool'), None)
    assert my_tool_def is not None
    assert 'allowed_callers' not in my_tool_def


async def test_auto_adds_code_execution_tool(allow_model_requests: None):
    """Test that CodeExecutionTool is auto-added when programmatically_callable is used."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # No explicit CodeExecutionTool in builtin_tools
    agent = Agent(model)

    @agent.tool_plain(programmatically_callable=True)
    def my_tool(x: int) -> int:
        """A tool that can be called from code execution."""
        return x * 2

    await agent.run('test')

    # Check that code_execution tool was auto-added
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    # Should have my_tool and code_execution
    tool_names = [t.get('name') for t in tools]
    assert 'my_tool' in tool_names
    assert 'code_execution' in tool_names

    # Check that the newer code execution type is used
    code_exec_tool = next((t for t in tools if t.get('name') == 'code_execution'), None)
    assert code_exec_tool is not None
    assert code_exec_tool.get('type') == 'code_execution_20250825'


async def test_uses_newer_code_execution_with_ptc(allow_model_requests: None):
    """Test that the newer code execution tool is used when PTC is enabled."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Explicitly add CodeExecutionTool
    agent = Agent(model, builtin_tools=[CodeExecutionTool()])

    @agent.tool_plain(programmatically_callable=True)
    def my_tool(x: int) -> int:
        """A tool that can be called from code execution."""
        return x * 2

    await agent.run('test')

    # Check that the newer code execution type is used
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    code_exec_tool = next((t for t in tools if t.get('name') == 'code_execution'), None)
    assert code_exec_tool is not None
    assert code_exec_tool.get('type') == 'code_execution_20250825'

    # Also check that the newer beta is used
    betas = mock_client.chat_completion_kwargs[0].get('betas', [])
    assert 'code-execution-2025-08-25' in betas


async def test_uses_older_code_execution_without_ptc(allow_model_requests: None):
    """Test that the older code execution tool is used when PTC is not enabled."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Add CodeExecutionTool but no programmatically_callable tools
    agent = Agent(model, builtin_tools=[CodeExecutionTool()])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        """A regular tool."""
        return x * 2

    await agent.run('test')

    # Check that the older code execution type is used
    assert len(mock_client.chat_completion_kwargs) == 1
    tools = mock_client.chat_completion_kwargs[0]['tools']

    code_exec_tool = next((t for t in tools if t.get('name') == 'code_execution'), None)
    assert code_exec_tool is not None
    assert code_exec_tool.get('type') == 'code_execution_20250522'

    # Also check that the older beta is used
    betas = mock_client.chat_completion_kwargs[0].get('betas', [])
    assert 'code-execution-2025-05-22' in betas
