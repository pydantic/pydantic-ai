from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelStatusError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..conftest import IsNow

pytestmark = pytest.mark.anyio


def success_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('success')])


def failure_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise ModelStatusError(status_code=500, model_name='test-function-model', body={'error': 'test error'})


success_model = FunctionModel(success_response)
failure_model = FunctionModel(failure_response)


def test_fallback_model_init() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    assert fallback_model.model_name == snapshot(
        'FallBackModel[function:failure_response:, function:success_response:]'
    )


def test_fallback_model_first_successful() -> None:
    fallback_model = FallbackModel(success_model, failure_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.data == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                model_name='function:success_response',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_fallback_model_first_failed() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.data == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                model_name='function:success_response',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_fallback_model_all_failed() -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model)
    with pytest.raises(RuntimeError) as exc_info:
        agent.run_sync('hello')
    assert 'All fallback models failed' in exc_info.value.args[0]
