from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot
from logfire_api import DEFAULT_LOGFIRE_INSTANCE
from opentelemetry._events import NoOpEventLoggerProvider
from opentelemetry.trace import NoOpTracerProvider

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage

from ..conftest import try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='logfire not installed'),
    pytest.mark.anyio,
]

requires_logfire_events = pytest.mark.skipif(
    not hasattr(DEFAULT_LOGFIRE_INSTANCE.config, 'get_event_logger_provider'),
    reason='old logfire without events/logs support',
)


class MyModel(Model):
    @property
    def system(self) -> str:
        return 'my_system'

    @property
    def model_name(self) -> str:
        return 'my_model'

    @property
    def base_url(self) -> str:
        return 'https://example.com:8000/foo'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        return (
            ModelResponse(
                parts=[
                    TextPart('text1'),
                    ToolCallPart('tool1', 'args1', 'tool_call_1'),
                    ToolCallPart('tool2', {'args2': 3}, 'tool_call_2'),
                    TextPart('text2'),
                    {},  # test unexpected parts  # type: ignore
                ],
                model_name='my_model_123',
            ),
            Usage(request_tokens=100, response_tokens=200),
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        yield MyResponseStream()


class MyResponseStream(StreamedResponse):
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = Usage(request_tokens=300, response_tokens=400)
        yield self._parts_manager.handle_text_delta(vendor_part_id=0, content='text1')
        yield self._parts_manager.handle_text_delta(vendor_part_id=0, content='text2')

    @property
    def model_name(self) -> str:
        return 'my_model_123'

    @property
    def timestamp(self) -> datetime:
        return datetime(2022, 1, 1)


@requires_logfire_events
async def test_instrumented_model(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(event_mode='logs'))
    assert model.system == 'my_system'
    assert model.model_name == 'my_model'

    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart('system_prompt'),
                UserPromptPart('user_prompt'),
                ToolReturnPart('tool3', 'tool_return_content', 'tool_call_3'),
                RetryPromptPart('retry_prompt1', tool_name='tool4', tool_call_id='tool_call_4'),
                RetryPromptPart('retry_prompt2'),
                {},  # test unexpected parts  # type: ignore
            ]
        ),
        ModelResponse(
            parts=[
                TextPart('text3'),
            ]
        ),
    ]
    await model.request(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        ),
    )

    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'chat my_model',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 18000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'my_system',
                    'gen_ai.request.model': 'my_model',
                    'server.address': 'example.com',
                    'server.port': 8000,
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat my_model',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'my_model_123',
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 200,
                },
            },
        ]
    )

    assert capfire.log_exporter.exported_logs_as_dicts() == snapshot(
        [
            {
                'body': {'content': 'system_prompt', 'role': 'system'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.system.message',
                },
                'timestamp': 2000000000,
                'observed_timestamp': 3000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'content': 'user_prompt', 'role': 'user'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.user.message',
                },
                'timestamp': 4000000000,
                'observed_timestamp': 5000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'content': 'tool_return_content', 'role': 'tool', 'id': 'tool_call_3', 'name': 'tool3'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.tool.message',
                },
                'timestamp': 6000000000,
                'observed_timestamp': 7000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {
                    'content': """\
retry_prompt1

Fix the errors and try again.\
""",
                    'role': 'tool',
                    'id': 'tool_call_4',
                    'name': 'tool4',
                },
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.tool.message',
                },
                'timestamp': 8000000000,
                'observed_timestamp': 9000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {
                    'content': """\
retry_prompt2

Fix the errors and try again.\
""",
                    'role': 'user',
                },
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.user.message',
                },
                'timestamp': 10000000000,
                'observed_timestamp': 11000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'role': 'assistant', 'content': 'text3'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 1,
                    'event.name': 'gen_ai.assistant.message',
                },
                'timestamp': 12000000000,
                'observed_timestamp': 13000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': 'text1',
                        'tool_calls': [
                            {
                                'id': 'tool_call_1',
                                'type': 'function',
                                'function': {'name': 'tool1', 'arguments': 'args1'},
                            },
                            {
                                'id': 'tool_call_2',
                                'type': 'function',
                                'function': {'name': 'tool2', 'arguments': {'args2': 3}},
                            },
                        ],
                    },
                },
                'severity_number': 9,
                'severity_text': None,
                'attributes': {'gen_ai.system': 'my_system', 'event.name': 'gen_ai.choice'},
                'timestamp': 14000000000,
                'observed_timestamp': 15000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'index': 0, 'message': {'role': 'assistant', 'content': 'text2'}},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {'gen_ai.system': 'my_system', 'event.name': 'gen_ai.choice'},
                'timestamp': 16000000000,
                'observed_timestamp': 17000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
        ]
    )


async def test_instrumented_model_not_recording():
    model = InstrumentedModel(
        MyModel(),
        InstrumentationSettings(tracer_provider=NoOpTracerProvider(), event_logger_provider=NoOpEventLoggerProvider()),
    )

    messages: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('system_prompt')])]
    await model.request(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        ),
    )


@requires_logfire_events
async def test_instrumented_model_stream(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(event_mode='logs'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user_prompt'),
            ]
        ),
    ]
    async with model.request_stream(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        ),
    ) as response_stream:
        assert [event async for event in response_stream] == snapshot(
            [
                PartStartEvent(index=0, part=TextPart(content='text1')),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='text2')),
            ]
        )

    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'chat my_model',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'my_system',
                    'gen_ai.request.model': 'my_model',
                    'server.address': 'example.com',
                    'server.port': 8000,
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat my_model',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'my_model_123',
                    'gen_ai.usage.input_tokens': 300,
                    'gen_ai.usage.output_tokens': 400,
                },
            },
        ]
    )

    assert capfire.log_exporter.exported_logs_as_dicts() == snapshot(
        [
            {
                'body': {'content': 'user_prompt', 'role': 'user'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.user.message',
                },
                'timestamp': 2000000000,
                'observed_timestamp': 3000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'index': 0, 'message': {'role': 'assistant', 'content': 'text1text2'}},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {'gen_ai.system': 'my_system', 'event.name': 'gen_ai.choice'},
                'timestamp': 4000000000,
                'observed_timestamp': 5000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
        ]
    )


@requires_logfire_events
async def test_instrumented_model_stream_break(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(event_mode='logs'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user_prompt'),
            ]
        ),
    ]

    with pytest.raises(RuntimeError):
        async with model.request_stream(
            messages,
            model_settings=ModelSettings(temperature=1),
            model_request_parameters=ModelRequestParameters(
                function_tools=[],
                allow_text_result=True,
                result_tools=[],
            ),
        ) as response_stream:
            async for event in response_stream:
                assert event == PartStartEvent(index=0, part=TextPart(content='text1'))
                raise RuntimeError

    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'chat my_model',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 7000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'my_system',
                    'gen_ai.request.model': 'my_model',
                    'server.address': 'example.com',
                    'server.port': 8000,
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat my_model',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'my_model_123',
                    'gen_ai.usage.input_tokens': 300,
                    'gen_ai.usage.output_tokens': 400,
                    'logfire.level_num': 17,
                },
                'events': [
                    {
                        'name': 'exception',
                        'timestamp': 6000000000,
                        'attributes': {
                            'exception.type': 'RuntimeError',
                            'exception.message': '',
                            'exception.stacktrace': 'RuntimeError',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            },
        ]
    )

    assert capfire.log_exporter.exported_logs_as_dicts() == snapshot(
        [
            {
                'body': {'content': 'user_prompt', 'role': 'user'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'my_system',
                    'gen_ai.message.index': 0,
                    'event.name': 'gen_ai.user.message',
                },
                'timestamp': 2000000000,
                'observed_timestamp': 3000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
            {
                'body': {'index': 0, 'message': {'role': 'assistant', 'content': 'text1'}},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {'gen_ai.system': 'my_system', 'event.name': 'gen_ai.choice'},
                'timestamp': 4000000000,
                'observed_timestamp': 5000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
        ]
    )


async def test_instrumented_model_attributes_mode(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(event_mode='attributes'))
    assert model.system == 'my_system'
    assert model.model_name == 'my_model'

    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart('system_prompt'),
                UserPromptPart('user_prompt'),
                ToolReturnPart('tool3', 'tool_return_content', 'tool_call_3'),
                RetryPromptPart('retry_prompt1', tool_name='tool4', tool_call_id='tool_call_4'),
                RetryPromptPart('retry_prompt2'),
                {},  # test unexpected parts  # type: ignore
            ]
        ),
        ModelResponse(
            parts=[
                TextPart('text3'),
            ]
        ),
    ]
    await model.request(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        ),
    )

    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'chat my_model',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'my_system',
                    'gen_ai.request.model': 'my_model',
                    'server.address': 'example.com',
                    'server.port': 8000,
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat my_model',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'my_model_123',
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 200,
                    'events': IsJson(
                        snapshot(
                            [
                                {
                                    'event.name': 'gen_ai.system.message',
                                    'content': 'system_prompt',
                                    'role': 'system',
                                    'gen_ai.message.index': 0,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.user.message',
                                    'content': 'user_prompt',
                                    'role': 'user',
                                    'gen_ai.message.index': 0,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.tool.message',
                                    'content': 'tool_return_content',
                                    'role': 'tool',
                                    'name': 'tool3',
                                    'id': 'tool_call_3',
                                    'gen_ai.message.index': 0,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.tool.message',
                                    'content': """\
retry_prompt1

Fix the errors and try again.\
""",
                                    'role': 'tool',
                                    'name': 'tool4',
                                    'id': 'tool_call_4',
                                    'gen_ai.message.index': 0,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.user.message',
                                    'content': """\
retry_prompt2

Fix the errors and try again.\
""",
                                    'role': 'user',
                                    'gen_ai.message.index': 0,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.assistant.message',
                                    'role': 'assistant',
                                    'content': 'text3',
                                    'gen_ai.message.index': 1,
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.choice',
                                    'index': 0,
                                    'message': {
                                        'role': 'assistant',
                                        'content': 'text1',
                                        'tool_calls': [
                                            {
                                                'id': 'tool_call_1',
                                                'type': 'function',
                                                'function': {'name': 'tool1', 'arguments': 'args1'},
                                            },
                                            {
                                                'id': 'tool_call_2',
                                                'type': 'function',
                                                'function': {'name': 'tool2', 'arguments': {'args2': 3}},
                                            },
                                        ],
                                    },
                                    'gen_ai.system': 'my_system',
                                },
                                {
                                    'event.name': 'gen_ai.choice',
                                    'index': 0,
                                    'message': {'role': 'assistant', 'content': 'text2'},
                                    'gen_ai.system': 'my_system',
                                },
                            ]
                        )
                    ),
                    'logfire.json_schema': '{"type": "object", "properties": {"events": {"type": "array"}}}',
                },
            },
        ]
    )


def test_messages_to_otel_events_serialization_errors():
    class Foo:
        def __repr__(self):
            return 'Foo()'

    class Bar:
        def __repr__(self):
            raise ValueError('error!')

    messages = [
        ModelResponse(parts=[ToolCallPart('tool', {'arg': Foo()})]),
        ModelRequest(parts=[ToolReturnPart('tool', Bar())]),
    ]

    assert [
        InstrumentedModel.event_to_dict(e) for e in InstrumentedModel.messages_to_otel_events(messages)
    ] == snapshot(
        [
            {
                'body': "{'role': 'assistant', 'tool_calls': [{'id': None, 'type': 'function', 'function': {'name': 'tool', 'arguments': {'arg': Foo()}}}]}",
                'gen_ai.message.index': 0,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'body': 'Unable to serialize: error!',
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.tool.message',
            },
        ]
    )
