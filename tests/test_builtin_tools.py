from __future__ import annotations

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'cohere', 'huggingface', 'test', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_web_search(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('bedrock', 'mistral', 'huggingface', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_web_search_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover


@pytest.mark.parametrize('model', ('groq', 'openai', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_code_execution(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()])

    with pytest.raises(UserError):
        await agent.run('What day is tomorrow?')


@pytest.mark.parametrize('model', ('groq', 'openai', 'outlines'), indirect=True)
async def test_builtin_tools_not_supported_code_execution_stream(model: Model, allow_model_requests: None):
    agent = Agent(model=model, builtin_tools=[CodeExecutionTool()])

    with pytest.raises(UserError):
        async with agent.run_stream('What day is tomorrow?'):
            ...  # pragma: no cover
