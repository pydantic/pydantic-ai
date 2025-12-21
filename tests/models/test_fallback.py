from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from datetime import timezone
from typing import Any, Literal, cast

import pytest
from _pytest.python_api import RaisesContext
from dirty_equals import IsJson
from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic_core import to_json

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    UserPromptPart,
)
from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsNow, IsStr, try_import

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

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
    assert fallback_model.model_name == snapshot('fallback:function:failure_response:,function:success_response:')
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
                    },
                    'logfire.span_type': 'span',
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
                    },
                    'logfire.span_type': 'span',
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
    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        agent.run_sync('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}


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
    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        agent.run_sync('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ModelHTTPError)
    assert exceptions[0].status_code == 500
    assert exceptions[0].model_name == 'test-function-model'
    assert exceptions[0].body == {'error': 'test error'}
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
                    },
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {'model_request_parameters': {'type': 'object'}},
                    },
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat fallback:function:failure_response:,function:failure_response:',
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
    # Note: today we can only handle errors that are raised before the streaming begins
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
    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        async with agent.run_stream('hello') as result:
            [c async for c, _is_last in result.stream_responses(debounce_by=None)]  # pragma: lax no cover
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
    assert response.output == 'success'


async def test_fallback_connection_error() -> None:
    def connection_error_response(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        raise ModelAPIError(model_name='test-connection-model', message='Connection timed out')

    connection_error_model = FunctionModel(connection_error_response)
    fallback_model = FallbackModel(connection_error_model, success_model)
    agent = Agent(model=fallback_model)

    response = await agent.run('hello')
    assert response.output == 'success'


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
            raise ModelHTTPError(status_code=500, model_name='prompted-model', body=None)  # pragma: no cover

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
                    },
                    'logfire.span_type': 'span',
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


async def test_fallback_on_response_triggered() -> None:
    def should_fallback_on_primary(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on_response=should_fallback_on_primary,
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


async def test_fallback_on_response_not_triggered() -> None:
    def never_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        return False

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on_response=never_fallback,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')
    assert result.output == snapshot('primary response')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='primary response')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:primary_response:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_fallback_on_response_all_fail() -> None:
    def always_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        return True

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on_response=always_fallback,
    )
    agent = Agent(model=fallback)

    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        await agent.run('hello')
    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], RuntimeError)
    assert 'rejected by fallback_on_response' in str(exc_info.value.exceptions[0])


async def test_fallback_on_response_with_message_inspection() -> None:
    inspected_messages: list[list[ModelMessage]] = []

    def inspect_messages(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        inspected_messages.append(messages)
        return False

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on_response=inspect_messages,
    )
    agent = Agent(model=fallback)

    await agent.run('hello')

    assert len(inspected_messages) == 1
    assert len(inspected_messages[0]) == 1


async def test_fallback_on_response_combined_with_exception_fallback() -> None:
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

    def reject_bad_response(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(first_fails_with_exception)
    second_model = FunctionModel(second_fails_response_check)
    third_model = FunctionModel(third_succeeds)

    fallback = FallbackModel(
        first_model,
        second_model,
        third_model,
        fallback_on_response=reject_bad_response,
    )
    agent = Agent(model=fallback)

    result = await agent.run('hello')

    assert result.output == snapshot('good response')
    assert call_order == snapshot(['first', 'second', 'third'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='good response')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:third_succeeds:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_fallback_on_response_mixed_failures_all_fail() -> None:
    call_order: list[str] = []

    def first_fails_with_exception(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('first')
        raise ModelHTTPError(status_code=500, model_name='first', body=None)

    def second_fails_response_check(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        call_order.append('second')
        return ModelResponse(parts=[TextPart('bad response')])

    def reject_bad_response(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(first_fails_with_exception)
    second_model = FunctionModel(second_fails_response_check)

    fallback = FallbackModel(
        first_model,
        second_model,
        fallback_on_response=reject_bad_response,
    )
    agent = Agent(model=fallback)

    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        await agent.run('hello')

    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 2
    assert isinstance(exc_info.value.exceptions[0], ModelHTTPError)
    assert isinstance(exc_info.value.exceptions[1], RuntimeError)
    assert 'rejected by fallback_on_response' in str(exc_info.value.exceptions[1])
    assert call_order == ['first', 'second']


async def test_fallback_on_response_web_fetch_scenario() -> None:
    def google_web_fetch_fails(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                BuiltinToolCallPart(tool_name='web_fetch', args={'url': 'https://example.com'}, tool_call_id='1'),
                BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    tool_call_id='1',
                    content={'uri': 'https://example.com', 'url_retrieval_status': 'URL_RETRIEVAL_STATUS_FAILED'},
                ),
                TextPart('Could not fetch URL'),
            ]
        )

    def anthropic_succeeds(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Successfully fetched and summarized the content')])

    def web_fetch_failed(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        for call, result in response.builtin_tool_calls:
            if call.tool_name != 'web_fetch':
                continue  # pragma: no cover
            content = result.content
            if not isinstance(content, dict):
                continue  # pragma: no cover
            content_dict = cast(dict[str, Any], content)
            status = content_dict.get('url_retrieval_status', '')
            if status and status != 'URL_RETRIEVAL_STATUS_SUCCESS':  # pragma: no branch
                return True
        return False

    google_model = FunctionModel(google_web_fetch_fails)
    anthropic_model = FunctionModel(anthropic_succeeds)

    fallback = FallbackModel(
        google_model,
        anthropic_model,
        fallback_on_response=web_fetch_failed,
    )
    agent = Agent(model=fallback)

    result = await agent.run('Summarize https://example.com')
    assert result.output == 'Successfully fetched and summarized the content'


def test_fallback_on_response_sync() -> None:
    def should_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    fallback = FallbackModel(
        primary_model,
        fallback_model_impl,
        fallback_on_response=should_fallback,
    )
    agent = Agent(model=fallback)

    result = agent.run_sync('hello')
    assert result.output == 'fallback response'


async def primary_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'primary '
    yield 'response'


async def fallback_response_stream(_model_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'fallback '
    yield 'response'


primary_model_stream = FunctionModel(stream_function=primary_response_stream)
fallback_model_stream_impl = FunctionModel(stream_function=fallback_response_stream)


async def test_fallback_on_response_streaming_triggered() -> None:
    def should_fallback_on_primary(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'primary' in part.content

    fallback = FallbackModel(
        primary_model_stream,
        fallback_model_stream_impl,
        fallback_on_response=should_fallback_on_primary,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'fallback response'


async def test_fallback_on_response_streaming_not_triggered() -> None:
    def never_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        return False

    fallback = FallbackModel(
        primary_model_stream,
        fallback_model_stream_impl,
        fallback_on_response=never_fallback,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'primary response'


async def test_fallback_on_response_streaming_all_fail() -> None:
    def always_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        return True

    fallback = FallbackModel(
        primary_model_stream,
        fallback_model_stream_impl,
        fallback_on_response=always_fallback,
    )
    agent = Agent(model=fallback)

    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()  # pragma: no cover

    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], RuntimeError)
    assert 'rejected by fallback_on_response' in str(exc_info.value.exceptions[0])


async def test_fallback_on_response_streaming_combined_with_exception() -> None:
    call_order: list[str] = []

    async def first_fails_with_exception(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('first')
        raise ModelHTTPError(status_code=500, model_name='first', body=None)
        yield 'never'  # pragma: no cover

    async def second_fails_response_check(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('second')
        yield 'bad '
        yield 'response'

    async def third_succeeds(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('third')
        yield 'good '
        yield 'response'

    def reject_bad_response(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(stream_function=first_fails_with_exception)
    second_model = FunctionModel(stream_function=second_fails_response_check)
    third_model = FunctionModel(stream_function=third_succeeds)

    fallback = FallbackModel(
        first_model,
        second_model,
        third_model,
        fallback_on_response=reject_bad_response,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'good response'
    assert call_order == ['first', 'second', 'third']


async def test_fallback_on_response_streaming_replays_events() -> None:
    def never_fallback(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        return False

    fallback = FallbackModel(
        primary_model_stream,
        fallback_model_stream_impl,
        fallback_on_response=never_fallback,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        responses = [c async for c, _is_last in result.stream_responses(debounce_by=None)]

    assert len(responses) >= 2
    part = responses[-1].parts[0]
    assert isinstance(part, TextPart)
    assert part.content == 'primary response'


async def test_fallback_on_part_streaming_triggered() -> None:
    models_tried: list[str] = []

    async def bad_response_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        models_tried.append('bad_model')
        yield 'bad content'

    async def good_response_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        models_tried.append('good_model')
        yield 'good content'

    def reject_bad_part(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return isinstance(part, TextPart) and 'bad' in part.content

    bad_model = FunctionModel(stream_function=bad_response_stream)
    good_model = FunctionModel(stream_function=good_response_stream)

    fallback = FallbackModel(
        bad_model,
        good_model,
        fallback_on_part=reject_bad_part,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'good content'
    assert models_tried == ['bad_model', 'good_model']


async def test_fallback_on_part_streaming_not_triggered() -> None:
    async def ok_response_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        yield 'ok content'

    def never_reject(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return False

    ok_model = FunctionModel(stream_function=ok_response_stream)
    fallback_model_not_used = FunctionModel(stream_function=ok_response_stream)

    fallback = FallbackModel(
        ok_model,
        fallback_model_not_used,
        fallback_on_part=never_reject,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'ok content'


async def test_fallback_on_part_streaming_all_fail() -> None:
    async def bad_response_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        yield 'bad content'

    def always_reject(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return True

    bad_model1 = FunctionModel(stream_function=bad_response_stream)
    bad_model2 = FunctionModel(stream_function=bad_response_stream)

    fallback = FallbackModel(
        bad_model1,
        bad_model2,
        fallback_on_part=always_reject,
    )
    agent = Agent(model=fallback)

    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()  # pragma: no cover

    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], RuntimeError)
    assert 'rejected by fallback_on_part' in str(exc_info.value.exceptions[0])


async def test_fallback_on_part_streaming_combined_with_fallback_on_response() -> None:
    call_order: list[str] = []

    async def part_rejected_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('first_part_rejected')
        yield 'reject_part'

    async def response_rejected_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('second_response_rejected')
        yield 'reject_response'

    async def success_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('third_success')
        yield 'success'

    def reject_part_with_keyword(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return isinstance(part, TextPart) and 'reject_part' in part.content

    def reject_response_with_keyword(response: ModelResponse, messages: list[ModelMessage]) -> bool:
        part = response.parts[0] if response.parts else None
        return isinstance(part, TextPart) and 'reject_response' in part.content

    first_model = FunctionModel(stream_function=part_rejected_stream)
    second_model = FunctionModel(stream_function=response_rejected_stream)
    third_model = FunctionModel(stream_function=success_stream)

    fallback = FallbackModel(
        first_model,
        second_model,
        third_model,
        fallback_on_part=reject_part_with_keyword,
        fallback_on_response=reject_response_with_keyword,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'success'
    assert call_order == ['first_part_rejected', 'second_response_rejected', 'third_success']


async def test_fallback_on_part_streaming_with_exception_fallback() -> None:
    call_order: list[str] = []

    async def first_exception_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('first_exception')
        raise ModelHTTPError(status_code=500, model_name='first', body=None)
        yield 'never'  # pragma: no cover

    async def second_part_rejected(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('second_part_rejected')
        yield 'bad_part'

    async def third_success(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        call_order.append('third_success')
        yield 'good'

    def reject_bad_part(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return isinstance(part, TextPart) and 'bad' in part.content

    first_model = FunctionModel(stream_function=first_exception_stream)
    second_model = FunctionModel(stream_function=second_part_rejected)
    third_model = FunctionModel(stream_function=third_success)

    fallback = FallbackModel(
        first_model,
        second_model,
        third_model,
        fallback_on_part=reject_bad_part,
    )
    agent = Agent(model=fallback)

    async with agent.run_stream('hello') as result:
        output = await result.get_output()

    assert output == 'good'
    assert call_order == ['first_exception', 'second_part_rejected', 'third_success']


async def test_fallback_on_part_streaming_mixed_failures_all_fail() -> None:
    async def exception_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError(status_code=500, model_name='exception', body=None)
        yield 'never'  # pragma: no cover

    async def bad_part_stream(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        yield 'bad_part'

    def always_reject(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
        return True

    first_model = FunctionModel(stream_function=exception_stream)
    second_model = FunctionModel(stream_function=bad_part_stream)

    fallback = FallbackModel(
        first_model,
        second_model,
        fallback_on_part=always_reject,
    )
    agent = Agent(model=fallback)

    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        async with agent.run_stream('hello') as result:
            await result.get_output()  # pragma: no cover

    assert 'All models from FallbackModel failed' in exc_info.value.args[0]
    assert len(exc_info.value.exceptions) == 2
    assert isinstance(exc_info.value.exceptions[0], ModelHTTPError)
    assert isinstance(exc_info.value.exceptions[1], RuntimeError)
    assert 'rejected by fallback_on_part' in str(exc_info.value.exceptions[1])
