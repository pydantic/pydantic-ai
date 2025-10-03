import pytest

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.test import TestModel


def test_agent_builtin_tools_runtime_parameter():
    """Test that Agent.run_sync accepts builtin_tools parameter."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[])

    # Should work with empty builtin_tools
    result = agent.run_sync('Hello', builtin_tools=[])
    assert result.output == 'success (no tool calls)'

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == []


async def test_agent_builtin_tools_runtime_parameter_async():
    """Test that Agent.run and Agent.run_stream accept builtin_tools parameter."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[])

    # Test async run
    result = await agent.run('Hello', builtin_tools=[])
    assert result.output == 'success (no tool calls)'

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == []

    # Test run_stream
    async with agent.run_stream('Hello', builtin_tools=[]) as stream:
        output = await stream.get_output()
        assert output == 'success (no tool calls)'

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == []


def test_agent_builtin_tools_testmodel_rejection():
    """Test that TestModel rejects builtin tools as expected."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[])

    # Should raise error when builtin_tools contains actual tools
    web_search_tool = WebSearchTool()
    with pytest.raises(Exception, match='TestModel does not support built-in tools'):
        agent.run_sync('Hello', builtin_tools=[web_search_tool])

    assert model.last_model_request_parameters is not None
    assert len(model.last_model_request_parameters.builtin_tools) == 1
    assert model.last_model_request_parameters.builtin_tools[0] == web_search_tool


def test_agent_builtin_tools_runtime_vs_agent_level():
    """Test that runtime builtin_tools parameter is merged with agent-level builtin_tools."""
    model = TestModel()
    web_search_tool = WebSearchTool()

    # Agent has builtin tools, and we provide same type at runtime
    agent = Agent(model=model, builtin_tools=[web_search_tool])

    # Runtime tool of same type should override agent-level tool
    different_web_search = WebSearchTool(search_context_size='high')
    with pytest.raises(Exception, match='TestModel does not support built-in tools'):
        agent.run_sync('Hello', builtin_tools=[different_web_search])

    assert model.last_model_request_parameters is not None
    assert len(model.last_model_request_parameters.builtin_tools) == 1
    runtime_tool = model.last_model_request_parameters.builtin_tools[0]
    assert isinstance(runtime_tool, WebSearchTool)
    assert runtime_tool.search_context_size == 'high'


def test_agent_builtin_tools_runtime_additional():
    """Test that runtime builtin_tools can add to agent-level builtin_tools when different types."""
    model = TestModel()
    web_search_tool = WebSearchTool()

    agent = Agent(model=model, builtin_tools=[])

    with pytest.raises(Exception, match='TestModel does not support built-in tools'):
        agent.run_sync('Hello', builtin_tools=[web_search_tool])

    assert model.last_model_request_parameters is not None
    assert len(model.last_model_request_parameters.builtin_tools) == 1
    assert model.last_model_request_parameters.builtin_tools[0] == web_search_tool
