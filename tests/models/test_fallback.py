from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from datetime import timezone

import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..conftest import IsNow, try_import

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


pytestmark = pytest.mark.anyio


def success_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('success')])


def failure_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})


success_model = FunctionModel(success_response)
failure_model = FunctionModel(failure_response)


def test_init() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    assert fallback_model.model_name == snapshot(
        'FallBackModel[function:failure_response:, function:success_response:]'
    )
    assert fallback_model.system is None
    assert fallback_model.base_url is None


def test_first_successful() -> None:
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
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_first_failed() -> None:
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
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
def test_first_failed_instrumented(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model, instrument=True)
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
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'preparing model request params',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'run_step': 1,
                    'logfire.span_type': 'span',
                    'logfire.msg': 'preparing model request params',
                },
            },
            {
                'name': 'chat function:success_response:',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 5000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat FallBackModel[function:failure_response:, function:success_response:]',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function:success_response:',
                    'gen_ai.response.model': 'function:success_response:',
                    'events': IsJson(
                        [
                            {
                                'content': 'hello',
                                'role': 'user',
                                'gen_ai.system': 'function',
                                'gen_ai.message.index': 0,
                                'event.name': 'gen_ai.user.message',
                            },
                            {
                                'index': 0,
                                'message': {'role': 'assistant', 'content': 'success'},
                                'gen_ai.system': 'function',
                                'event.name': 'gen_ai.choice',
                            },
                        ]
                    ),
                    'logfire.json_schema': '{"type": "object", "properties": {"events": {"type": "array"}}}',
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'model_name': 'FallBackModel[function:failure_response:, function:success_response:]',
                    'agent_name': 'agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'all_messages_events': '[{"content": "hello", "role": "user", "gen_ai.message.index": 0, "event.name": "gen_ai.user.message"}, {"role": "assistant", "content": "success", "gen_ai.message.index": 1, "event.name": "gen_ai.assistant.message"}]',
                    'final_result': 'success',
                    'logfire.json_schema': '{"type": "object", "properties": {"all_messages_events": {"type": "array"}, "final_result": {"type": "object"}}}',
                },
            },
        ]
    )


def test_all_failed() -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model)
    with pytest.raises(ExceptionGroup) as exc_info:
        agent.run_sync('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


async def success_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '
    yield 'world'


async def failure_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    # Note: today we can only handle errors that are raised before the streaming begins
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})
    yield 'uh oh... '


success_model_stream = FunctionModel(stream_function=success_response_stream)
failure_model_stream = FunctionModel(stream_function=failure_response_stream)


async def test_first_success_streaming() -> None:
    fallback_model = FallbackModel(success_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_structured(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.is_complete


async def test_first_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, success_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_structured(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.is_complete


async def test_all_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    with pytest.raises(ExceptionGroup) as exc_info:
        async with agent.run_stream('hello') as result:
            [c async for c, _is_last in result.stream_structured(debounce_by=None)]
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


async def test_fallback_condition_override() -> None:
    def should_fallback(exc: Exception) -> bool:
        return False

    fallback_model = FallbackModel(failure_model, success_model, fallback_on=should_fallback)
    agent = Agent(model=fallback_model)
    with pytest.raises(ModelHTTPError):
        await agent.run('hello')


class PotatoException(Exception): ...


def potato_exception_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise PotatoException()


async def test_fallback_condition_tuple() -> None:
    potato_model = FunctionModel(potato_exception_response)
    fallback_model = FallbackModel(potato_model, success_model, fallback_on=(PotatoException, ModelHTTPError))
    agent = Agent(model=fallback_model)

    response = await agent.run('hello')
    assert response.data == 'success'
