from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Literal, cast

import pytest
from dirty_equals import IsJson
from pydantic import BaseModel
from pydantic_core import to_json
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart, InstructionPart, ModelResponseState

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel, ResponseRejected
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

requires_openai = pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


pytestmark = pytest.mark.anyio


def success_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('success')])


def failure_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})


success_model = FunctionModel(success_response)
failure_model = FunctionModel(failure_response)


def _assert_fallback_all_failed(exc_info: pytest.ExceptionInfo[FallbackExceptionGroup]) -> None:
    """Assert that a FallbackExceptionGroup contains 2 ModelHTTPError 500s from test-function-model."""
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


def test_init() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    assert fallback_model.model_name == snapshot('fallback:function:failure_response:,function:success_response:')
    assert fallback_model.model_id == snapshot(
        'fallback:function:function:failure_response:,function:function:success_response:'
    )
    assert fallback_model.system == 'fallback:function,function'
    assert fallback_model.base_url is None


def test_first_successful() -> None:
    fallback_model = FallbackModel(success_model, failure_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_first_failed() -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
def test_first_failed_instrumented(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model, success_model)
    agent = Agent(model=fallback_model, instrument=True)
    result = agent.run_sync('hello')
    assert result.output == snapshot('success')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )
    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat function:success_response:',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'prompted_output_template': None,
                        'allow_text_output': True,
                        'allow_image_output': False,
                        'instruction_parts': None,
                        'thinking': None,
                    },
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.provider.name': 'function',
                    'logfire.msg': 'chat fallback:function:failure_response:,function:success_response:',
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function:success_response:',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]}],
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success'}]}
                    ],
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'gen_ai.response.model': 'function:success_response:',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'fallback:function:failure_response:,function:success_response:',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 1,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success'}]},
                    ],
                    'final_result': 'success',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_first_failed_instrumented_stream(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model_stream, success_model_stream)
    agent = Agent(model=fallback_model, instrument=True)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.is_complete

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat function::success_response_stream',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'prompted_output_template': None,
                        'allow_text_output': True,
                        'allow_image_output': False,
                        'instruction_parts': None,
                        'thinking': None,
                    },
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.provider.name': 'function',
                    'logfire.msg': 'chat fallback:function::failure_response_stream,function::success_response_stream',
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function::success_response_stream',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'input'}]}],
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello world'}]}
                    ],
                    'gen_ai.usage.input_tokens': 50,
                    'gen_ai.usage.output_tokens': 2,
                    'gen_ai.response.model': 'function::success_response_stream',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'fallback:function::failure_response_stream,function::success_response_stream',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'final_result': 'hello world',
                    'gen_ai.usage.input_tokens': 50,
                    'gen_ai.usage.output_tokens': 2,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'input'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'hello world'}]},
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                },
            },
        ]
    )


def test_all_failed() -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model)
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        agent.run_sync('hello')
    _assert_fallback_all_failed(exc_info)


def add_missing_response_model(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for span in spans:
        attrs = span.setdefault('attributes', {})
        if 'gen_ai.request.model' in attrs:
            attrs.setdefault('gen_ai.response.model', attrs['gen_ai.request.model'])
    return spans


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
def test_all_failed_instrumented(capfire: CaptureLogfire) -> None:
    fallback_model = FallbackModel(failure_model, failure_model)
    agent = Agent(model=fallback_model, instrument=True)
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        agent.run_sync('hello')
    _assert_fallback_all_failed(exc_info)
    assert add_missing_response_model(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'chat fallback:function:failure_response:,function:failure_response:',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 4000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'fallback:function,function',
                    'gen_ai.system': 'fallback:function,function',
                    'gen_ai.request.model': 'fallback:function:failure_response:,function:failure_response:',
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'text',
                        'output_object': None,
                        'output_tools': [],
                        'prompted_output_template': None,
                        'allow_text_output': True,
                        'allow_image_output': False,
                        'instruction_parts': None,
                        'thinking': None,
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'model_request_parameters': {'type': 'object'}},
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat fallback:function:failure_response:,function:failure_response:',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'logfire.level_num': 17,
                    'gen_ai.response.model': 'fallback:function:failure_response:,function:failure_response:',
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 3000000000,
                        'attributes': {
                            'exception.type': 'pydantic_ai.exceptions.FallbackExceptionGroup',
                            'exception.message': 'All models from FallbackModel failed (2 sub-exceptions)',
                            'exception.stacktrace': '+------------------------------------',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'model_name': 'fallback:function:failure_response:,function:failure_response:',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'logfire.exception.fingerprint': '0000000000000000000000000000000000000000000000000000000000000000',
                    'pydantic_ai.all_messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]}],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                    'logfire.level_num': 17,
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 5000000000,
                        'attributes': {
                            'exception.type': 'pydantic_ai.exceptions.FallbackExceptionGroup',
                            'exception.message': 'All models from FallbackModel failed (2 sub-exceptions)',
                            'exception.stacktrace': '+------------------------------------',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
        ]
    )


async def success_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '
    yield 'world'


async def failure_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    # Note: exception-based fallback for streaming only catches errors during stream initialization
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})
    yield 'uh oh... '


success_model_stream = FunctionModel(stream_function=success_response_stream)
failure_model_stream = FunctionModel(stream_function=failure_response_stream)


async def test_first_success_streaming() -> None:
    fallback_model = FallbackModel(success_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.is_complete


async def test_first_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, success_model_stream)
    agent = Agent(model=fallback_model)
    async with agent.run_stream('input') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='hello ')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::success_response_stream',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.is_complete


async def test_all_failed_streaming() -> None:
    fallback_model = FallbackModel(failure_model_stream, failure_model_stream)
    agent = Agent(model=fallback_model)
    with pytest.raises(FallbackExceptionGroup) as exc_info:
        async with agent.run_stream('hello') as result:
            [c async for c, _is_last in result.stream_responses(debounce_by=None)]  # pragma: lax no cover
    _assert_fallback_all_failed(exc_info)


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
    assert response.output == 'success'
    assert response.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_fallback_connection_error() -> None:
    def connection_error_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        raise ModelAPIError(model_name='test-connection-model', message='Connection timed out')

    connection_error_model = FunctionModel(connection_error_response)
    fallback_model = FallbackModel(connection_error_model, success_model)
    agent = Agent(model=fallback_model)

    response = await agent.run('hello')
    assert response.output == 'success'
    assert response.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:success_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_fallback_model_settings_merge():
    """Test that FallbackModel properly merges model settings from wrapped model and runtime settings."""

    def return_settings(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(to_json(info.model_settings).decode())])

    base_model = FunctionModel(return_settings, settings=ModelSettings(temperature=0.1, max_tokens=1024))
    fallback_model = FallbackModel(base_model)

    # Test that base model settings are preserved when no additional settings are provided
    agent = Agent(fallback_model)
    result = await agent.run('Hello')
    assert result.output == IsJson({'max_tokens': 1024, 'temperature': 0.1})

    # Test that runtime model_settings are merged with base settings
    agent_with_settings = Agent(fallback_model, model_settings=ModelSettings(temperature=0.5, parallel_tool_calls=True))
    result = await agent_with_settings.run('Hello')
    expected = {'max_tokens': 1024, 'temperature': 0.5, 'parallel_tool_calls': True}
    assert result.output == IsJson(expected)

    # Test that run-time model_settings override both base and agent settings
    result = await agent_with_settings.run(
        'Hello', model_settings=ModelSettings(temperature=0.9, extra_headers={'runtime_setting': 'runtime_value'})
    )
    expected = {
        'max_tokens': 1024,
        'temperature': 0.9,
        'parallel_tool_calls': True,
        'extra_headers': {
            'runtime_setting': 'runtime_value',
        },
    }
    assert result.output == IsJson(expected)


async def test_fallback_model_settings_merge_streaming():
    """Test that FallbackModel properly merges model settings in streaming mode."""

    async def return_settings_stream(_: list[ModelMessage], info: AgentInfo):
        # Yield the merged settings as JSON to verify they were properly combined
        yield to_json(info.model_settings).decode()

    base_model = FunctionModel(
        stream_function=return_settings_stream,
        settings=ModelSettings(temperature=0.1, extra_headers={'anthropic-beta': 'context-1m-2025-08-07'}),
    )
    fallback_model = FallbackModel(base_model)

    # Test that base model settings are preserved in streaming mode
    agent = Agent(fallback_model)
    async with agent.run_stream('Hello') as result:
        output = await result.get_output()

    assert json.loads(output) == {'extra_headers': {'anthropic-beta': 'context-1m-2025-08-07'}, 'temperature': 0.1}

    # Test that runtime model_settings are merged with base settings in streaming mode
    agent_with_settings = Agent(fallback_model, model_settings=ModelSettings(temperature=0.5))
    async with agent_with_settings.run_stream('Hello') as result:
        output = await result.get_output()

    expected = {'extra_headers': {'anthropic-beta': 'context-1m-2025-08-07'}, 'temperature': 0.5}
    assert json.loads(output) == expected


async def test_fallback_model_structured_output():
    class Foo(BaseModel):
        bar: str

    def tool_output_func(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal enabled_model
        if enabled_model != 'tool':
            raise ModelHTTPError(status_code=500, model_name='tool-model', body=None)

        assert info.model_request_parameters == snapshot(
            ModelRequestParameters(
                output_mode='tool',
                output_tools=[
                    ToolDefinition(
                        name='final_result',
                        parameters_json_schema={
                            'properties': {'bar': {'type': 'string'}},
                            'required': ['bar'],
                            'title': 'Foo',
                            'type': 'object',
                        },
                        description='The final response which ends this conversation',
                        kind='output',
                    )
                ],
                allow_text_output=False,
            )
        )

        args = Foo(bar='baz').model_dump()
        assert info.output_tools
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    def native_output_func(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal enabled_model
        if enabled_model != 'native':
            raise ModelHTTPError(status_code=500, model_name='native-model', body=None)

        assert info.model_request_parameters == snapshot(
            ModelRequestParameters(
                output_mode='native',
                output_object=OutputObjectDefinition(
                    json_schema={
                        'properties': {'bar': {'type': 'string'}},
                        'required': ['bar'],
                        'title': 'Foo',
                        'type': 'object',
                    },
                    name='Foo',
                ),
            )
        )

        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    def prompted_output_func(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal enabled_model
        if enabled_model != 'prompted':
            raise ModelHTTPError(status_code=500, model_name='prompted-model', body=None)  # pragma: lax no cover

        assert info.model_request_parameters == snapshot(
            ModelRequestParameters(
                output_mode='prompted',
                output_object=OutputObjectDefinition(
                    json_schema={
                        'properties': {'bar': {'type': 'string'}},
                        'required': ['bar'],
                        'title': 'Foo',
                        'type': 'object',
                    },
                    name='Foo',
                ),
                prompted_output_template="""\

Always respond with a JSON object that's compatible with this schema:

{schema}

Don't include any text or Markdown fencing before or after.
""",
                instruction_parts=[
                    InstructionPart(
                        content="""\

Always respond with a JSON object that's compatible with this schema:

{"properties": {"bar": {"type": "string"}}, "required": ["bar"], "title": "Foo", "type": "object"}

Don't include any text or Markdown fencing before or after.
"""
                    )
                ],
            )
        )

        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    tool_model = FunctionModel(
        tool_output_func, profile=ModelProfile(default_structured_output_mode='tool', supports_tools=True)
    )
    native_model = FunctionModel(
        native_output_func,
        profile=ModelProfile(default_structured_output_mode='native', supports_json_schema_output=True),
    )
    prompted_model = FunctionModel(
        prompted_output_func, profile=ModelProfile(default_structured_output_mode='prompted')
    )

    fallback_model = FallbackModel(tool_model, native_model, prompted_model)
    agent = Agent(fallback_model, output_type=Foo)

    enabled_model: Literal['tool', 'native', 'prompted'] = 'tool'
    tool_result = await agent.run('hello')
    assert tool_result.output == snapshot(Foo(bar='baz'))

    enabled_model = 'native'
    tool_result = await agent.run('hello')
    assert tool_result.output == snapshot(Foo(bar='baz'))

    enabled_model = 'prompted'
    tool_result = await agent.run('hello')
    assert tool_result.output == snapshot(Foo(bar='baz'))


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_fallback_model_structured_output_instrumented(capfire: CaptureLogfire) -> None:
    class Foo(BaseModel):
        bar: str

    def tool_output_func(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name='tool-model', body=None)

    def prompted_output_func(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters == snapshot(
            ModelRequestParameters(
                output_mode='prompted',
                output_object=OutputObjectDefinition(
                    json_schema={
                        'properties': {'bar': {'type': 'string'}},
                        'required': ['bar'],
                        'title': 'Foo',
                        'type': 'object',
                    },
                    name='Foo',
                ),
                prompted_output_template="""\

Always respond with a JSON object that's compatible with this schema:

{schema}

Don't include any text or Markdown fencing before or after.
""",
                instruction_parts=[
                    InstructionPart(content='Be kind'),
                    InstructionPart(
                        content="""\

Always respond with a JSON object that's compatible with this schema:

{"properties": {"bar": {"type": "string"}}, "required": ["bar"], "title": "Foo", "type": "object"}

Don't include any text or Markdown fencing before or after.
"""
                    ),
                ],
            )
        )

        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    tool_model = FunctionModel(
        tool_output_func, profile=ModelProfile(default_structured_output_mode='tool', supports_tools=True)
    )
    prompted_model = FunctionModel(
        prompted_output_func, profile=ModelProfile(default_structured_output_mode='prompted')
    )
    fallback_model = FallbackModel(tool_model, prompted_model)
    agent = Agent(model=fallback_model, instrument=True, output_type=Foo, instructions='Be kind')
    result = await agent.run('hello')
    assert result.output == snapshot(Foo(bar='baz'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be kind',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"bar":"baz"}')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:prompted_output_func:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )
    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat function:prompted_output_func:',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.tool.definitions': [
                        {
                            'type': 'function',
                            'name': 'final_result',
                            'description': 'The final response which ends this conversation',
                            'parameters': {
                                'properties': {'bar': {'type': 'string'}},
                                'required': ['bar'],
                                'title': 'Foo',
                                'type': 'object',
                            },
                        }
                    ],
                    'model_request_parameters': {
                        'function_tools': [],
                        'builtin_tools': [],
                        'output_mode': 'prompted',
                        'output_object': {
                            'json_schema': {
                                'properties': {'bar': {'type': 'string'}},
                                'required': ['bar'],
                                'title': 'Foo',
                                'type': 'object',
                            },
                            'name': 'Foo',
                            'description': None,
                            'strict': None,
                        },
                        'output_tools': [],
                        'prompted_output_template': """\

Always respond with a JSON object that's compatible with this schema:

{schema}

Don't include any text or Markdown fencing before or after.
""",
                        'allow_text_output': True,
                        'allow_image_output': False,
                        'instruction_parts': [
                            {'content': 'Be kind', 'dynamic': False, 'part_kind': 'instruction'},
                            {
                                'content': """\

Always respond with a JSON object that's compatible with this schema:

{"properties": {"bar": {"type": "string"}}, "required": ["bar"], "title": "Foo", "type": "object"}

Don't include any text or Markdown fencing before or after.
""",
                                'dynamic': False,
                                'part_kind': 'instruction',
                            },
                        ],
                        'thinking': None,
                    },
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.provider.name': 'function',
                    'logfire.msg': 'chat fallback:function:tool_output_func:,function:prompted_output_func:',
                    'gen_ai.system': 'function',
                    'gen_ai.request.model': 'function:prompted_output_func:',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]}],
                    'gen_ai.output.messages': [
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': '{"bar":"baz"}'}]}
                    ],
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be kind'}],
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 4,
                    'gen_ai.response.model': 'function:prompted_output_func:',
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'fallback:function:tool_output_func:,function:prompted_output_func:',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 4,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': '{"bar":"baz"}'}]},
                    ],
                    'final_result': {'bar': 'baz'},
                    'gen_ai.system_instructions': [{'type': 'text', 'content': 'Be kind'}],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    },
                },
            },
        ]
    )


def primary_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('primary response')])


def fallback_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('fallback response')])


primary_model = FunctionModel(primary_response)
fallback_model_impl = FunctionModel(fallback_response)


async def test_response_handler_triggered() -> None:
    """Test that a response handler can trigger fallback based on response content."""

    def should_fallback_on_primary(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    # Auto-detected as response handler via type hint
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=should_fallback_on_primary,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')
    assert result.output == snapshot('fallback response')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='fallback response')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:fallback_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_response_handler_not_triggered() -> None:
    """Test that response handler returning False allows the response through."""

    def never_fallback(response: ModelResponse) -> bool:
        return False

    # Auto-detected as response handler via type hint
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=never_fallback,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')
    assert result.output == snapshot('primary response')


async def test_response_handler_all_fail() -> None:
    """Test that when all models are rejected by response handler, an error is raised."""

    def always_fallback(response: ModelResponse) -> bool:
        return True

    # Auto-detected as response handler via type hint
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=always_fallback,
    )
    agent = Agent(model=fallback)

    with pytest.raises(ExceptionGroup) as exc_info:
        await agent.run('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ResponseRejected)
    assert 'rejected by fallback_on' in str(exc_info.value.exceptions[0])


async def test_mixed_exception_and_response_handlers() -> None:
    """Test combining exception types and response handlers in a list."""
    call_order: list[str] = []

    def first_fails_with_exception(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('first')
        raise ModelHTTPError(status_code=500, model_name='first', body=None)

    def second_fails_response_check(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('second')
        return ModelResponse(parts=[TextPart('bad response')])

    def third_succeeds(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('third')
        return ModelResponse(parts=[TextPart('good response')])

    def reject_bad_response(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(first_fails_with_exception)
    second_model = FunctionModel(second_fails_response_check)
    third_model = FunctionModel(third_succeeds)

    # Use a list to combine exception type and response handler (auto-detected via type hint)
    fallback = FallbackModel(
        first_model,
        second_model,
        third_model,
        fallback_on=[ModelHTTPError, reject_bad_response],
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')

    assert result.output == snapshot('good response')
    assert call_order == snapshot(['first', 'second', 'third'])


async def test_mixed_failures_all_fail() -> None:
    """Test error reporting when both exceptions and response rejections occur."""
    call_order: list[str] = []

    def first_fails_with_exception(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('first')
        raise ModelHTTPError(status_code=500, model_name='first', body=None)

    def second_fails_response_check(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('second')
        return ModelResponse(parts=[TextPart('bad response')])

    def reject_bad_response(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(first_fails_with_exception)
    second_model = FunctionModel(second_fails_response_check)

    # Auto-detected via type hint
    fallback = FallbackModel(
        first_model,
        second_model,
        fallback_on=[ModelHTTPError, reject_bad_response],
    )
    agent = Agent(model=fallback)

    with pytest.raises(ExceptionGroup) as exc_info:
        await agent.run('hello')

    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 2
    assert isinstance(exc_info.value.exceptions[0], ModelHTTPError)
    assert isinstance(exc_info.value.exceptions[1], ResponseRejected)
    assert 'rejected by fallback_on' in str(exc_info.value.exceptions[1])
    assert call_order == ['first', 'second']


async def test_web_fetch_scenario() -> None:
    """Test real-world scenario: fallback when web_fetch builtin tool fails.

    This matches the actual Google SDK structure where content is a list of
    UrlMetadata dicts with 'retrieved_url' and 'url_retrieval_status' fields.
    """

    def google_web_fetch_fails(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        # Content is a list of UrlMetadata dicts, matching google.genai.types.UrlMetadata.model_dump()
        # Include multiple items to cover loop iteration branch
        return ModelResponse(
            parts=[
                BuiltinToolCallPart(tool_name='web_fetch', args={'urls': ['https://example.com']}, tool_call_id='1'),
                BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    tool_call_id='1',
                    content=[
                        {'retrieved_url': 'https://ok.com', 'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS'},
                        {'retrieved_url': 'https://example.com', 'url_retrieval_status': 'URL_RETRIEVAL_STATUS_FAILED'},
                    ],
                ),
                TextPart('Could not fetch URL'),
            ]
        )

    def anthropic_succeeds(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Successfully fetched and summarized the content')])

    class UrlMetadataDict(TypedDict):
        retrieved_url: str
        url_retrieval_status: str

    def web_fetch_failed(response: ModelResponse) -> bool:
        for call, result in response.builtin_tool_calls:  # pragma: no branch
            if call.tool_name != 'web_fetch':
                continue  # pragma: lax no cover
            if not isinstance(result.content, list):
                continue  # pragma: lax no cover
            # Cast needed because result.content is typed as Any
            items = cast(list[UrlMetadataDict], result.content)  # pyright: ignore[reportUnknownMemberType]
            for item in items:  # pragma: no branch
                if item['url_retrieval_status'] != 'URL_RETRIEVAL_STATUS_SUCCESS':
                    return True
        return False

    google_model = FunctionModel(google_web_fetch_fails)
    anthropic_model = FunctionModel(anthropic_succeeds)

    # Auto-detected via type hint
    fallback = FallbackModel(
        google_model,
        anthropic_model,
        fallback_on=web_fetch_failed,
    )
    agent = Agent(model=fallback)

    result = await agent.run('Summarize https://example.com')
    assert result.output == 'Successfully fetched and summarized the content'


def test_response_handler_sync() -> None:
    """Test response handler with synchronous run."""

    def should_fallback(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    # Auto-detected via type hint
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=should_fallback,
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'fallback response'


def test_fallback_on_list_of_exception_types() -> None:
    """Test fallback_on with a list containing individual exception types."""

    class CustomError(Exception):
        pass

    def raises_custom_error(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise CustomError('custom error')

    custom_error_model = FunctionModel(raises_custom_error)

    # List with individual exception types (not a tuple)
    fallback = FallbackModel(
        custom_error_model,
        success_model,
        fallback_on=[CustomError, ModelHTTPError],
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'success'


def test_fallback_on_single_response_handler() -> None:
    """Test fallback_on with a single response handler (auto-detected via type hint)."""

    def reject_primary(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    # Auto-detected as response handler via type hint
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=reject_primary,
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'fallback response'


def test_fallback_on_single_exception_handler() -> None:
    """Test fallback_on with a single exception handler (auto-detected by type hint)."""

    def custom_exception_handler(exc: Exception) -> bool:
        return isinstance(exc, ModelHTTPError) and exc.status_code == 500

    # Auto-detected as exception handler via type hint (first param is Exception, not ModelResponse)
    fallback = FallbackModel(
        failure_model,
        success_model,
        fallback_on=custom_exception_handler,
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'success'


def test_fallback_on_mixed_list() -> None:
    """Test fallback_on with a mixed list of exception types, exception handlers, and response handlers."""

    class CustomError(Exception):
        pass

    def custom_exception_handler(exc: Exception) -> bool:  # pragma: no cover
        return isinstance(exc, ModelHTTPError) and exc.status_code == 503

    def reject_bad_response(response: ModelResponse) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    def bad_response_model(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('bad response')])

    bad_model = FunctionModel(bad_response_model)

    # Mix of exception type, exception handler, and response handler (auto-detected via type hints)
    fallback = FallbackModel(
        bad_model,
        fallback_model_impl,
        fallback_on=[CustomError, custom_exception_handler, reject_bad_response],
    )
    agent = Agent(model=fallback)

    # Should fallback because response contains 'bad'
    result = agent.run_sync('hello')
    assert result.output == 'fallback response'


def test_fallback_on_lambda_exception_handler() -> None:
    """Test that lambdas with 1 param are detected as exception handlers."""
    fallback = FallbackModel(
        failure_model,
        success_model,
        fallback_on=lambda e: isinstance(e, ModelHTTPError),
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'success'


async def test_async_exception_handler() -> None:
    """Test that async exception handlers work correctly."""

    async def async_exc_handler(exc: Exception) -> bool:
        return isinstance(exc, ModelHTTPError)

    fallback = FallbackModel(
        failure_model,
        success_model,
        fallback_on=async_exc_handler,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')
    assert result.output == 'success'


async def test_async_response_handler() -> None:
    """Test that async response handlers work correctly."""

    async def async_response_handler(response: ModelResponse) -> bool:
        # Reject if 'primary' in response
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=async_response_handler,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')
    assert result.output == 'fallback response'


def test_fallback_on_invalid_type() -> None:
    """Test that invalid fallback_on types raise AssertionError via assert_never."""
    with pytest.raises(AssertionError, match='Expected code to be unreachable'):
        FallbackModel(success_model, failure_model, fallback_on='invalid')  # type: ignore


def test_fallback_on_invalid_list_item() -> None:
    """Test that invalid items in fallback_on list raise AssertionError via assert_never."""
    with pytest.raises(AssertionError, match='Expected code to be unreachable'):
        FallbackModel(success_model, failure_model, fallback_on=['invalid'])  # type: ignore


def test_response_handler_only_exception_propagates() -> None:
    """Test that exceptions propagate when only response handlers are configured.

    This documents the expected behavior: if you only configure response handlers
    (no exception types or exception handlers), exceptions are not caught and will
    propagate to the caller.
    """

    def response_check(response: ModelResponse) -> bool:  # pragma: no cover
        return False  # Never reject based on response

    # Auto-detected as response handler via type hint - only a response handler, no exception handling
    fallback = FallbackModel(
        failure_model,  # This will raise ModelHTTPError
        success_model,
        fallback_on=response_check,
    )
    agent = Agent(model=fallback)

    # Exception should propagate since no exception handler is configured
    with pytest.raises(ModelHTTPError):
        agent.run_sync('hello')


def test_callable_class_response_handler() -> None:
    """Test that callable classes with __call__(ModelResponse) trigger response-based fallback."""

    class RejectPrimary:
        def __call__(self, response: ModelResponse) -> bool:
            part = response.parts[0] if response.parts else None
            return isinstance(part, TextPart) and 'primary' in part.content

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=RejectPrimary(),
    )
    agent = Agent(model=fallback)
    result = agent.run_sync('hello')
    assert result.output == 'fallback response'


def test_callable_class_exception_handler() -> None:
    """Test that callable classes with __call__(Exception) trigger exception-based fallback."""

    class HandleHTTPError:
        def __call__(self, exc: Exception) -> bool:
            return isinstance(exc, ModelHTTPError)

    fallback = FallbackModel(
        failure_model,
        success_model,
        fallback_on=HandleHTTPError(),
    )
    agent = Agent(model=fallback)
    result = agent.run_sync('hello')
    assert result.output == 'success'


def test_unresolvable_forward_ref_treated_as_exception_handler() -> None:
    """A handler with unresolvable forward refs is treated as an exception handler."""
    # Create a function whose type hints can't be resolved (triggers except branch in get_first_param_type)
    exec_globals: dict[str, object] = {}
    exec(  # nosec - test-only dynamic function creation for unresolvable forward ref
        """
def handler(x: "NonExistentType") -> bool:
    return isinstance(x, Exception)
""",
        exec_globals,
    )
    handler = exec_globals['handler']

    # Classified as exception handler (forward ref can't resolve), so responses pass through
    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=handler,  # type: ignore[arg-type]
    )
    agent = Agent(model=fallback)
    result = agent.run_sync('hello')
    assert result.output == 'primary response'


def test_fallback_on_single_exception_type_direct() -> None:
    """Test fallback_on with a single exception type (not in tuple/list)."""

    def raises_api_error(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise ModelAPIError('test-model', 'test error')

    fallback = FallbackModel(
        FunctionModel(raises_api_error),
        success_model,
        fallback_on=ModelAPIError,  # Single type, not tuple
    )
    agent = Agent(model=fallback)
    result = agent.run_sync('hello')
    assert result.output == 'success'


def test_empty_fallback_on_list_error() -> None:
    """Test that empty fallback_on list raises UserError."""
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match='empty fallback_on'):
        FallbackModel(
            primary_model,
            fallback_model_impl,
            fallback_on=[],
        )


def test_empty_fallback_on_tuple_error() -> None:
    """Test that empty fallback_on tuple raises UserError."""
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match='empty fallback_on'):
        FallbackModel(
            primary_model,
            fallback_model_impl,
            fallback_on=(),
        )


async def test_response_rejection_error_message() -> None:
    """Test that error message describes response rejections."""

    def always_reject(response: ModelResponse) -> bool:
        return True

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on=always_reject,
    )
    agent = Agent(model=fallback)

    with pytest.raises(ExceptionGroup) as exc_info:
        await agent.run('hello')

    # Find the ResponseRejected in the exception group
    rejection_errors = [e for e in exc_info.value.exceptions if isinstance(e, ResponseRejected)]
    assert len(rejection_errors) == 1

    error_msg = str(rejection_errors[0])
    assert 'rejected by fallback_on handler' in error_msg


@requires_openai
async def test_fallback_model_lifecycle_closes_sub_model_clients():
    """FallbackModel propagates __aenter__/__aexit__ to all sub-models' providers.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider1 = OpenAIProvider(api_key='test-key-1')
    provider2 = OpenAIProvider(api_key='test-key-2')
    model1 = OpenAIChatModel('gpt-4o', provider=provider1)
    model2 = OpenAIChatModel('gpt-4o', provider=provider2)

    fallback = FallbackModel(model1, model2)

    async with fallback:
        assert provider1._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert provider2._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert not provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
        assert not provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]


@requires_openai
async def test_fallback_model_lifecycle_via_agent():
    """Agent context manager propagates lifecycle through FallbackModel to sub-models' providers.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider1 = OpenAIProvider(api_key='test-key-1')
    provider2 = OpenAIProvider(api_key='test-key-2')
    model1 = OpenAIChatModel('gpt-4o', provider=provider1)
    model2 = OpenAIChatModel('gpt-4o', provider=provider2)

    fallback = FallbackModel(model1, model2)
    agent = Agent(model=fallback)

    async with agent:
        assert provider1._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert provider2._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert not provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
        assert not provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]


@requires_openai
async def test_fallback_model_reentrant_lifecycle():
    """Reentrant FallbackModel lifecycle keeps sub-models' clients open until outermost exit.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider1 = OpenAIProvider(api_key='test-key-1')
    provider2 = OpenAIProvider(api_key='test-key-2')
    model1 = OpenAIChatModel('gpt-4o', provider=provider1)
    model2 = OpenAIChatModel('gpt-4o', provider=provider2)

    fallback = FallbackModel(model1, model2)

    async with fallback:
        http1 = provider1._own_http_client  # pyright: ignore[reportPrivateUsage]
        http2 = provider2._own_http_client  # pyright: ignore[reportPrivateUsage]
        assert http1 is not None
        assert http2 is not None
        async with fallback:
            assert not http1.is_closed
            assert not http2.is_closed
        assert not http1.is_closed
        assert not http2.is_closed
    assert http1.is_closed
    assert http2.is_closed


@requires_openai
async def test_fallback_model_instrumented_lifecycle():
    """InstrumentedModel wrapping FallbackModel propagates lifecycle to sub-models.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider1 = OpenAIProvider(api_key='test-key-1')
    provider2 = OpenAIProvider(api_key='test-key-2')
    model1 = OpenAIChatModel('gpt-4o', provider=provider1)
    model2 = OpenAIChatModel('gpt-4o', provider=provider2)

    fallback = FallbackModel(model1, model2)
    instrumented = InstrumentedModel(fallback, InstrumentationSettings())

    async with instrumented:
        assert provider1._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert provider2._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert not provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
        assert not provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]


@requires_openai
async def test_fallback_model_concurrent_entry():
    """Concurrent entry to FallbackModel doesn't race on _entered_count / _exit_stack.

    Without a lock, two coroutines can both see _entered_count == 0 when the first
    yields during sub-model entry, causing one exit stack to be overwritten and leaked.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    import asyncio

    from pydantic_ai.models.wrapper import WrapperModel

    class SlowEnterModel(WrapperModel):
        """Wrapper that yields during __aenter__ to widen the race window."""

        async def __aenter__(self) -> SlowEnterModel:
            await asyncio.sleep(0)
            await self.wrapped.__aenter__()
            return self

    provider1 = OpenAIProvider(api_key='test-key-1')
    provider2 = OpenAIProvider(api_key='test-key-2')
    model1 = SlowEnterModel(OpenAIChatModel('gpt-4o', provider=provider1))
    model2 = SlowEnterModel(OpenAIChatModel('gpt-4o', provider=provider2))

    fallback = FallbackModel(model1, model2)

    async def enter_and_hold(event: asyncio.Event) -> None:
        async with fallback:
            event.set()
            await asyncio.sleep(0.1)

    event1 = asyncio.Event()
    event2 = asyncio.Event()
    task1 = asyncio.create_task(enter_and_hold(event1))
    task2 = asyncio.create_task(enter_and_hold(event2))

    await event1.wait()
    await event2.wait()
    assert provider1._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
    assert provider2._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
    assert not provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert not provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]

    await task1
    await task2
    assert provider1._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider2._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]


# --- Continuation pinning tests ---


def test_fallback_primary_continuation_then_succeeds() -> None:
    """Primary returns state='suspended', gets pinned, then returns normally. Fallback never called."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[TextPart('paused')], state='suspended')
        return ModelResponse(parts=[TextPart('done')])

    def fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('Fallback should not be called')  # pragma: no cover

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'pauseddone'
    assert call_count == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='paused'), TextPart(content='done')],
                usage=RequestUsage(input_tokens=102, output_tokens=3),
                model_name='primary',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_fallback_primary_continuation_multiple_pauses() -> None:
    """Primary returns state='suspended' twice (stays pinned), then finishes. Fallback never called."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return ModelResponse(parts=[TextPart('paused')], state='suspended')
        return ModelResponse(parts=[TextPart('done')])

    def fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('Fallback should not be called')  # pragma: no cover

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'pausedpauseddone'
    assert call_count == 3
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='paused'), TextPart(content='paused'), TextPart(content='done')],
                usage=RequestUsage(input_tokens=153, output_tokens=6),
                model_name='primary',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_fallback_secondary_continuation_back_to_primary() -> None:
    """Primary fails, fallback returns state, gets pinned, finishes with tool call,
    then tool executes. On new request: primary succeeds (pin cleared)."""
    primary_calls = 0
    fallback_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            raise ModelHTTPError(status_code=500, model_name='primary', body='error')
        return ModelResponse(parts=[TextPart('final answer')])

    def fallback_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal fallback_calls
        fallback_calls += 1
        if fallback_calls == 1:
            return ModelResponse(parts=[TextPart('working...')], state='suspended')
        return ModelResponse(
            parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call_1')],
        )

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback_fn, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)

    agent = Agent(model=model)

    @agent.tool_plain
    def my_tool() -> str:
        return 'tool result'

    result = agent.run_sync('test')
    # After fallback finishes continuation (no more state), pin is cleared.
    # Next request (after tool execution) goes through normal fallback chain → primary succeeds.
    assert result.output == 'final answer'
    assert primary_calls == 2  # first call failed, second succeeded
    assert fallback_calls == 2  # first returned continuation, second returned tool call
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='working...'),
                    ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call_1'),
                ],
                usage=RequestUsage(input_tokens=102, output_tokens=6),
                model_name='fallback',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content='tool result',
                        tool_call_id='call_1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final answer')],
                usage=RequestUsage(input_tokens=53, output_tokens=6),
                model_name='primary',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_fallback_primary_continuation_fails() -> None:
    """Primary returns state='suspended', gets pinned, then primary raises a fallback-eligible error.
    Messages are rewound and the fallback chain is tried — fallback succeeds."""
    primary_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            return ModelResponse(parts=[TextPart('paused')], state='suspended')
        raise ModelHTTPError(status_code=500, model_name='primary', body='continuation failed')

    fallback_calls = 0

    def fallback_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal fallback_calls
        fallback_calls += 1
        return ModelResponse(parts=[TextPart('fallback success')])

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback_fn, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'fallback success'
    assert primary_calls == 3  # 1st: suspended, 2nd: continuation fails, 3rd: retried in chain (fails again)
    assert fallback_calls == 1  # called once via fallback chain after rewind


def test_fallback_secondary_continuation_fails() -> None:
    """Primary fails, fallback returns state='suspended', gets pinned, then fallback raises error.
    Messages are rewound and the normal chain is retried — primary succeeds."""
    primary_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            raise ModelHTTPError(status_code=500, model_name='primary', body='error')
        return ModelResponse(parts=[TextPart('primary recovered')])

    fallback_calls = 0

    def fallback_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal fallback_calls
        fallback_calls += 1
        if fallback_calls == 1:
            return ModelResponse(parts=[TextPart('working...')], state='suspended')
        raise ModelHTTPError(status_code=500, model_name='fallback', body='continuation failed')

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback_fn, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'primary recovered'
    assert primary_calls == 2  # 1st: failed, 2nd: succeeded after rewind
    assert fallback_calls == 2  # 1st: suspended, 2nd: continuation failed


def test_fallback_continuation_non_fallback_error_propagates() -> None:
    """Primary returns state='suspended', then raises a non-fallback-eligible error.
    Error propagates directly without trying the fallback chain."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[TextPart('paused')], state='suspended')
        raise PotatoException('not a fallback error')

    def fallback_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('Fallback should not be called')  # pragma: no cover

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback_fn, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    with pytest.raises(PotatoException, match='not a fallback error'):
        agent.run_sync('test')
    assert call_count == 2


def test_fallback_continuation_recovery_replaces_response_parts() -> None:
    """When primary suspends then fails and fallback recovers, the final merged response
    contains only the fallback model's parts — not accumulated parts from the suspended primary."""
    primary_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            return ModelResponse(parts=[TextPart('primary partial')], state='suspended')
        raise ModelHTTPError(status_code=500, model_name='primary', body='fail')

    def fallback_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('fallback complete')])

    primary_model = FunctionModel(primary, model_name='primary')
    fallback_model_instance = FunctionModel(fallback_fn, model_name='fallback')
    model = FallbackModel(primary_model, fallback_model_instance)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'fallback complete'
    # The response should contain only fallback's parts, not accumulated 'primary partial' + 'fallback complete'
    response_msg = result.all_messages()[1]
    assert isinstance(response_msg, ModelResponse)
    assert len(response_msg.parts) == 1
    assert response_msg.parts[0].content == 'fallback complete'  # type: ignore[union-attr]
    assert response_msg.model_name == 'fallback'


# --- Streaming continuation pinning tests ---


@dataclass
class _ContinuationModel(Model):
    """Test model that wraps FunctionModel and supports state in streaming."""

    _inner: FunctionModel
    _stream_state: list[ModelResponseState] = field(default_factory=list[ModelResponseState])
    _stream_call_index: int = field(default=0)

    async def request(  # pragma: no cover
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self._inner.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self._inner.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as streamed_response:
            if self._stream_call_index < len(self._stream_state):  # pragma: no branch
                streamed_response.state = self._stream_state[self._stream_call_index]
            self._stream_call_index += 1
            yield streamed_response

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def system(self) -> str:  # pragma: no cover
        return self._inner.system


async def test_fallback_streaming_continuation_pinning() -> None:
    """Primary fails in streaming, fallback streams with state='suspended' (pinned),
    then second call to fallback goes through pinned continuation path and finishes."""

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError(status_code=500, model_name='primary', body='error')
        yield ''  # pragma: no cover

    fallback_calls = 0

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal fallback_calls
        fallback_calls += 1
        yield f'fallback response {fallback_calls}'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    primary_model = _ContinuationModel(_inner=primary_inner)
    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    # First stream call: state='suspended' (pin); second: False (clear pin)
    fallback_model = _ContinuationModel(_inner=fallback_inner, _stream_state=['suspended', 'complete'])
    model = FallbackModel(primary_model, fallback_model)

    run_id = 'test-run-1'
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')], run_id=run_id)]
    params = ModelRequestParameters()

    # First call: primary fails, fallback streams with state='suspended' → pinned
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'suspended'
    assert fallback_calls == 1

    resp1 = streamed_response.get()
    messages.append(resp1)
    messages.append(ModelRequest(parts=[], run_id=run_id))

    # Second call: goes through pinned continuation path (fallback.py lines 126-136)
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'complete'
    assert fallback_calls == 2

    resp2 = streamed_response.get()
    assert resp2.parts[0].content == 'fallback response 2'  # type: ignore[union-attr]


async def test_fallback_streaming_pinned_continuation_still_continuing() -> None:
    """Streaming: pinned model returns state='suspended' again — pin stays (87->89 branch)."""
    call_count = 0

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        yield f'response {call_count}'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    # First two calls return state='suspended', third returns False
    primary_model = _ContinuationModel(_inner=primary_inner, _stream_state=['suspended', 'suspended', 'complete'])

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise AssertionError('Fallback should not be called')  # pragma: no cover
        yield ''  # pragma: no cover

    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    fallback_model = _ContinuationModel(_inner=fallback_inner)
    model = FallbackModel(primary_model, fallback_model)

    run_id = 'test-run-2'
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')], run_id=run_id)]
    params = ModelRequestParameters()

    # First call: primary streams with state='suspended' → pinned
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'suspended'
    assert call_count == 1

    resp1 = streamed_response.get()
    messages.append(resp1)
    messages.append(ModelRequest(parts=[], run_id=run_id))

    # Second call: pinned, still state='suspended' → pin stays (87->89 branch)
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'suspended'
    assert call_count == 2

    resp2 = streamed_response.get()
    messages.append(resp2)
    messages.append(ModelRequest(parts=[], run_id=run_id))

    # Third call: pinned, state=False → pin cleared
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'complete'
    assert call_count == 3


async def test_fallback_streaming_pinned_continuation_fails_falls_back() -> None:
    """Streaming: pinned model fails to open stream → messages are rewound,
    fallback chain is tried, and the fallback model succeeds."""
    primary_call_count = 0

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal primary_call_count
        primary_call_count += 1
        if primary_call_count == 1:
            yield 'partial'
        else:
            raise ModelHTTPError(status_code=500, model_name='primary', body='continuation error')
            yield ''  # pragma: no cover

    fallback_calls = 0

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal fallback_calls
        fallback_calls += 1
        yield f'fallback response {fallback_calls}'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    primary_model = _ContinuationModel(_inner=primary_inner, _stream_state=['suspended'])
    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    fallback_model = _ContinuationModel(_inner=fallback_inner)
    model = FallbackModel(primary_model, fallback_model)

    run_id = 'test-stream-fail'
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')], run_id=run_id)]
    params = ModelRequestParameters()

    # First call: primary succeeds with state='suspended' → pinned
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert streamed_response.state == 'suspended'
    assert primary_call_count == 1

    resp1 = streamed_response.get()
    messages.append(resp1)
    messages.append(ModelRequest(parts=[], run_id=run_id))

    # Second call: pinned primary fails to open stream → rewind → fallback chain
    # (primary retried in chain and fails again, then fallback succeeds)
    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass
    assert primary_call_count == 3  # 1st: suspended, 2nd: pinned fail, 3rd: retried in chain (fails)
    assert fallback_calls == 1  # fallback succeeded
    resp2 = streamed_response.get()
    assert resp2.parts[0].content == 'fallback response 1'  # type: ignore[union-attr]


async def test_fallback_streaming_pinned_continuation_non_fallback_error_propagates() -> None:
    """Streaming: pinned model raises a non-fallback exception while opening the stream.
    The error propagates without trying fallback models."""
    primary_call_count = 0
    fallback_calls = 0

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal primary_call_count
        primary_call_count += 1
        if primary_call_count == 1:
            yield 'partial'
            return
        raise PotatoException('not a fallback error')
        yield ''  # pragma: no cover

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # pragma: no cover
        nonlocal fallback_calls
        fallback_calls += 1
        yield 'fallback response'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    primary_model = _ContinuationModel(_inner=primary_inner, _stream_state=['suspended'])
    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    fallback_model = _ContinuationModel(_inner=fallback_inner)
    model = FallbackModel(primary_model, fallback_model)

    run_id = 'test-stream-non-fallback'
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')], run_id=run_id)]
    params = ModelRequestParameters()

    async with model.request_stream(messages, None, params) as streamed_response:
        async for _ in streamed_response:
            pass

    messages.append(streamed_response.get())
    messages.append(ModelRequest(parts=[], run_id=run_id))

    with pytest.raises(PotatoException, match='not a fallback error'):
        async with model.request_stream(messages, None, params) as streamed_response:
            async for _ in streamed_response:
                pass

    assert primary_call_count == 2
    assert fallback_calls == 0


async def test_fallback_streaming_rewind_without_trailing_request() -> None:
    """Pinned fallback rewind works when history ends with a suspended response (no continuation request yet)."""
    primary_calls = 0
    fallback_calls = 0

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal primary_calls
        primary_calls += 1
        raise ModelHTTPError(status_code=500, model_name='primary', body='continuation error')
        yield ''  # pragma: no cover

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal fallback_calls
        fallback_calls += 1
        yield 'fallback response'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    primary_model = _ContinuationModel(_inner=primary_inner)
    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    fallback_model = _ContinuationModel(_inner=fallback_inner)
    model = FallbackModel(primary_model, fallback_model)

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='test')]),
        ModelResponse(
            parts=[TextPart('paused')],
            state='suspended',
            metadata={'__pydantic_ai__': {'fallback_model_name': 'primary'}},
        ),
    ]

    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        async for _ in streamed_response:
            pass

    assert primary_calls == 2
    assert fallback_calls == 1
    response = streamed_response.get()
    assert response.parts[0].content == 'fallback response'  # type: ignore[union-attr]


async def test_fallback_streaming_rewind_with_extra_trailing_request() -> None:
    """Pinned fallback rewind handles malformed history with multiple trailing continuation requests."""
    primary_calls = 0
    fallback_calls = 0

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal primary_calls
        primary_calls += 1
        raise ModelHTTPError(status_code=500, model_name='primary', body='continuation error')
        yield ''  # pragma: no cover

    async def fallback_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal fallback_calls
        fallback_calls += 1
        yield 'fallback response'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    primary_model = _ContinuationModel(_inner=primary_inner)
    fallback_inner = FunctionModel(stream_function=fallback_stream, model_name='fallback')
    fallback_model = _ContinuationModel(_inner=fallback_inner)
    model = FallbackModel(primary_model, fallback_model)

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='test')]),
        ModelResponse(
            parts=[TextPart('paused')],
            state='suspended',
            metadata={'__pydantic_ai__': {'fallback_model_name': 'primary'}},
        ),
        ModelRequest(parts=[]),
        ModelRequest(parts=[]),
    ]

    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        async for _ in streamed_response:
            pass

    assert primary_calls == 2
    assert fallback_calls == 1
    response = streamed_response.get()
    assert response.parts[0].content == 'fallback response'  # type: ignore[union-attr]


async def test_fallback_continuation_without_stamp_falls_through() -> None:
    """When state='suspended' but metadata lacks fallback_model_name,
    normal fallback chain is used (no pinning)."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart('primary response')])

    primary_model = FunctionModel(primary, model_name='primary')
    model = FallbackModel(primary_model)

    # Manually construct messages with state but no fallback_model_name in metadata
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='test')]),
        ModelResponse(parts=[TextPart('incomplete')], state='suspended'),
        ModelRequest(parts=[]),
    ]

    result = await model.request(messages, None, ModelRequestParameters())
    assert call_count == 1
    assert result.parts[0].content == 'primary response'  # type: ignore[union-attr]


async def test_fallback_continuation_with_unknown_model_falls_through() -> None:
    """When fallback_model_name doesn't match any model in the FallbackModel,
    normal fallback chain is used (no pinning)."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart('primary response')])

    primary_model = FunctionModel(primary, model_name='primary')
    model = FallbackModel(primary_model)

    # Construct messages with a fallback_model_name that doesn't match any model
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='test')]),
        ModelResponse(
            parts=[TextPart('incomplete')],
            state='suspended',
            metadata={'__pydantic_ai__': {'fallback_model_name': 'nonexistent-model'}},
        ),
        ModelRequest(parts=[]),
    ]

    result = await model.request(messages, None, ModelRequestParameters())
    assert call_count == 1
    assert result.parts[0].content == 'primary response'  # type: ignore[union-attr]


def test_fallback_stamp_with_existing_metadata() -> None:
    """When model response already has provider_details, the stamp is stored in metadata, not provider_details."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[TextPart('paused')],
                state='suspended',
                provider_details={'existing_key': 'existing_value'},
            )
        return ModelResponse(parts=[TextPart('done')])

    primary_model = FunctionModel(primary, model_name='primary')
    model = FallbackModel(primary_model)
    agent = Agent(model=model)

    result = agent.run_sync('test')
    assert result.output == 'pauseddone'
    assert call_count == 2
    # The merged response uses fields from the final (non continuation) response.
    # provider_details should not contain fallback routing state.
    continuation_msg = result.all_messages()[1]
    assert isinstance(continuation_msg, ModelResponse)
    assert continuation_msg.provider_details is None


async def test_fallback_stream_stamp_with_existing_metadata() -> None:
    """When streamed response already has provider_details, the stamp goes into metadata, not provider_details."""

    async def primary_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    primary_inner = FunctionModel(stream_function=primary_stream, model_name='primary')
    # Single call: state='suspended'
    primary_model = _ContinuationModel(_inner=primary_inner, _stream_state=['suspended'])
    model = FallbackModel(primary_model)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')])]
    params = ModelRequestParameters()

    async with model.request_stream(messages, None, params) as streamed_response:
        # Set provider_details before FallbackModel stamps it (simulates a provider that sets details during streaming)
        streamed_response.provider_details = {'existing_key': 'existing_value'}
        async for _ in streamed_response:
            pass

    assert streamed_response.state == 'suspended'
    # Fallback routing info goes in metadata, not provider_details
    assert streamed_response.provider_details == snapshot({'existing_key': 'existing_value'})
    assert streamed_response.metadata == snapshot({'__pydantic_ai__': {'fallback_model_name': 'primary'}})


async def test_fallback_stamp_continuation_with_existing_metadata() -> None:
    """When the model response already has metadata, _stamp_continuation merges into it."""
    call_count = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[TextPart('paused')],
                state='suspended',
                metadata={'provider_key': 'provider_value'},
            )
        return ModelResponse(parts=[TextPart('done')])  # pragma: no cover

    primary_model = FunctionModel(primary, model_name='primary')
    model = FallbackModel(primary_model)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='test')])]
    params = ModelRequestParameters()

    resp = await model.request(messages, None, params)
    assert resp.state == 'suspended'
    assert resp.metadata == snapshot(
        {'provider_key': 'provider_value', '__pydantic_ai__': {'fallback_model_name': 'primary'}}
    )
