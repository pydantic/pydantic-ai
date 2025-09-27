import pytest

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool, merge_builtin_tools
from pydantic_ai.models.test import TestModel


def test_merge_builtin_tools_basic():
    """Test that merge_builtin_tools function works correctly."""
    # Test merging with different tool types
    base_tools = [WebSearchTool(allowed_domains=['base.com'])]
    runtime_tools = [CodeExecutionTool()]

    merged = merge_builtin_tools(base_tools, runtime_tools)
    assert len(merged) == 2

    # Test merging with same tool types (runtime should override)
    base_tools = [WebSearchTool(allowed_domains=['base.com'])]
    runtime_tools = [WebSearchTool(allowed_domains=['runtime.com'])]

    merged = merge_builtin_tools(base_tools, runtime_tools)
    assert len(merged) == 1
    # Check that we got the runtime tool (need to check specific attributes)
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is not None
    assert web_tool.allowed_domains == ['runtime.com']

    base_tools = [CodeExecutionTool()]
    runtime_tools = [CodeExecutionTool()]

    merged = merge_builtin_tools(base_tools, runtime_tools)
    assert len(merged) == 1
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is None


def test_merge_builtin_tools_none_handling():
    """Test that merge_builtin_tools handles None correctly."""
    base_tools = [WebSearchTool(allowed_domains=['base.com'])]
    runtime_tools = [WebSearchTool(allowed_domains=['runtime.com'])]

    # Test with None runtime tools
    merged = merge_builtin_tools(base_tools, None)
    assert len(merged) == 1
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is not None
    assert web_tool.allowed_domains == ['base.com']

    # Test with None base tools
    merged = merge_builtin_tools(None, runtime_tools)
    assert len(merged) == 1
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is not None
    assert web_tool.allowed_domains == ['runtime.com']

    # Test with both None
    merged = merge_builtin_tools(None, None)
    assert len(merged) == 0

    base_tools = [CodeExecutionTool()]
    merged = merge_builtin_tools(base_tools, None)
    assert len(merged) == 1
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is None

    runtime_tools = [CodeExecutionTool()]
    merged = merge_builtin_tools(None, runtime_tools)
    assert len(merged) == 1
    web_tool = None
    for tool in merged:
        if isinstance(tool, WebSearchTool):
            web_tool = tool
            break
    else:
        web_tool = None
    assert web_tool is None


def test_agent_builtin_tools_runtime_parameter():
    """Test that Agent.run_sync accepts builtin_tools parameter."""
    agent = Agent(model=TestModel(), builtin_tools=[])

    # Should work with empty builtin_tools
    result = agent.run_sync('Hello', builtin_tools=[])
    assert result.output == 'success (no tool calls)'


async def test_agent_builtin_tools_runtime_parameter_async():
    """Test that Agent.run and Agent.run_stream accept builtin_tools parameter."""
    agent = Agent(model=TestModel(), builtin_tools=[])

    # Test async run
    result = await agent.run('Hello', builtin_tools=[])
    assert result.output == 'success (no tool calls)'

    # Test run_stream
    async with agent.run_stream('Hello', builtin_tools=[]) as stream:
        output = await stream.get_output()
        assert output == 'success (no tool calls)'


def test_agent_builtin_tools_testmodel_rejection():
    """Test that TestModel rejects builtin tools as expected."""
    agent = Agent(model=TestModel(), builtin_tools=[])

    # Should raise error when builtin_tools contains actual tools
    with pytest.raises(Exception, match='TestModel does not support built-in tools'):
        agent.run_sync('Hello', builtin_tools=[WebSearchTool()])
