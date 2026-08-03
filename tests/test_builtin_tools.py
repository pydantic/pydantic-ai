from __future__ import annotations

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    WebSearchTool,
)


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'cohere', 'huggingface', 'test'), indirect=True)
async def test_builtin_tools_not_supported_web_search(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'huggingface'), indirect=True)
async def test_builtin_tools_not_supported_web_search_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover


@pytest.mark.parametrize('model', ('groq', 'openai'), indirect=True)
async def test_builtin_tools_not_supported_code_execution(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(CodeExecutionTool())])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('groq', 'openai'), indirect=True)
async def test_builtin_tools_not_supported_code_execution_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(CodeExecutionTool())])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover


@pytest.mark.parametrize(
    'model', ('bedrock', 'mistral', 'cohere', 'huggingface', 'groq', 'anthropic', 'test'), indirect=True
)
async def test_builtin_tools_not_supported_file_search(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(FileSearchTool(file_store_ids=['test-id']))])

    with pytest.raises(UserError):
        await agent.run('Search my files')


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'huggingface', 'groq', 'anthropic'), indirect=True)
async def test_builtin_tools_not_supported_file_search_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, capabilities=[NativeTool(FileSearchTool(file_store_ids=['test-id']))])

    with pytest.raises(UserError):
        async with agent.run_stream('Search my files'):
            ...  # pragma: no cover


# --- unless_native swap tests ---


def test_unless_native_model_supports_builtin():
    """When model supports the builtin, the fallback function tool is removed."""
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.tools import ToolDefinition

    model = FunctionModel(lambda m, i: None)  # pyright: ignore[reportArgumentType]  # supports all builtins
    fallback_tool = ToolDefinition(name='my_search', description='Search', unless_native='web_search')
    params = ModelRequestParameters(
        function_tools=[fallback_tool],
        native_tools=[WebSearchTool()],
    )
    _, result = model.prepare_request(None, params)
    # Builtin is supported → fallback removed, builtin kept
    assert len(result.native_tools) == 1
    assert isinstance(result.native_tools[0], WebSearchTool)
    assert len(result.function_tools) == 0


def test_unless_native_model_does_not_support():
    """When model doesn't support the builtin, the builtin is removed and fallback stays."""
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.profiles import ModelProfile
    from pydantic_ai.tools import ToolDefinition

    model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
    fallback_tool = ToolDefinition(name='my_search', description='Search', unless_native='web_search')
    params = ModelRequestParameters(
        function_tools=[fallback_tool],
        native_tools=[WebSearchTool()],
    )
    _, result = model.prepare_request(None, params)
    # Builtin not supported → builtin removed, fallback kept
    assert len(result.native_tools) == 0
    assert len(result.function_tools) == 1
    assert result.function_tools[0].name == 'my_search'


def test_unless_native_no_fallback_raises_error():
    """Unsupported builtin without fallback still raises UserError."""
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.profiles import ModelProfile

    model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
    params = ModelRequestParameters(native_tools=[WebSearchTool()])
    with pytest.raises(UserError, match='not supported by this model'):
        model.prepare_request(None, params)


def test_unless_native_multiple_fallbacks_for_same_builtin():
    """Multiple fallback tools for the same builtin are all removed when builtin is supported."""
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.tools import ToolDefinition

    model = FunctionModel(lambda m, i: None)  # pyright: ignore[reportArgumentType]  # supports all builtins
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='search_a', description='A', unless_native='web_search'),
            ToolDefinition(name='search_b', description='B', unless_native='web_search'),
            ToolDefinition(name='regular_tool', description='C'),
        ],
        native_tools=[WebSearchTool()],
    )
    _, result = model.prepare_request(None, params)
    assert len(result.native_tools) == 1
    # Both fallbacks removed, regular tool kept
    assert [t.name for t in result.function_tools] == ['regular_tool']


def test_unless_native_mixed_support():
    """Multiple builtins with mixed support — each resolved independently."""
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.profiles import ModelProfile
    from pydantic_ai.tools import ToolDefinition

    # Only supports web search, not code execution
    model = FunctionModel(
        lambda m, i: None,  # pyright: ignore[reportArgumentType]
        profile=ModelProfile(supported_native_tools=frozenset({WebSearchTool})),
    )
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='local_search', description='Search', unless_native='web_search'),
            ToolDefinition(name='local_code', description='Code', unless_native='code_execution'),
        ],
        native_tools=[WebSearchTool(), CodeExecutionTool()],
    )
    _, result = model.prepare_request(None, params)
    # WebSearch: builtin supported → fallback removed, builtin kept
    # CodeExecution: builtin not supported → builtin removed, fallback kept
    assert len(result.native_tools) == 1
    assert isinstance(result.native_tools[0], WebSearchTool)
    assert [t.name for t in result.function_tools] == ['local_code']
