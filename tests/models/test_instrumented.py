from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import pytest
from inline_snapshot import snapshot
from inline_snapshot.extra import warns
from logfire_api import DEFAULT_LOGFIRE_INSTANCE
from opentelemetry._events import NoOpEventLoggerProvider
from opentelemetry.trace import NoOpTracerProvider

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    FinalResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsInt, IsStr, try_import

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
    # Use a system and model name that have a known price
    @property
    def system(self) -> str:
        return 'openai'

    @property
    def model_name(self) -> str:
        return 'gpt-4o'

    @property
    def base_url(self) -> str:
        return 'https://example.com:8000/foo'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart('text1'),
                ToolCallPart('tool1', 'args1', 'tool_call_1'),
                ToolCallPart('tool2', {'args2': 3}, 'tool_call_2'),
                TextPart('text2'),
                {},  # test unexpected parts  # type: ignore
            ],
            usage=RequestUsage(
                input_tokens=100,
                output_tokens=200,
                cache_write_tokens=10,
                cache_read_tokens=20,
                input_audio_tokens=10,
                cache_audio_read_tokens=5,
                output_audio_tokens=30,
                details={'reasoning_tokens': 30},
            ),
            model_name='gpt-4o-2024-11-20',
            provider_details=dict(finish_reason='stop', foo='bar'),
            provider_response_id='response_id',
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        yield MyResponseStream(model_request_parameters=model_request_parameters)


class MyResponseStream(StreamedResponse):
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = RequestUsage(input_tokens=300, output_tokens=400)
        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=0, content='text1')
        if maybe_event is not None:  # pragma: no branch
            yield maybe_event
        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=0, content='text2')
        if maybe_event is not None:  # pragma: no branch
            yield maybe_event

    @property
    def model_name(self) -> str:
        return 'gpt-4o-2024-11-20'

    @property
    def provider_name(self) -> str:
        return 'openai'

    @property
    def timestamp(self) -> datetime:
        return datetime(2022, 1, 1)


@requires_logfire_events
async def test_instrumented_model(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(version=1, event_mode='logs'))
    assert model.system == 'openai'
    assert model.model_name == 'gpt-4o'

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
        ModelResponse(parts=[TextPart('text3')]),
    ]
    await model.request(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
            output_mode='text',
            output_object=None,
        ),
    )

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat gpt-4o',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 16000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-4o',
                    'server.address': 'example.com',
                    'server.port': 8000,
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
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat gpt-4o',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4o-2024-11-20',
                    'gen_ai.response.id': 'response_id',
                    'gen_ai.usage.details.reasoning_tokens': 30,
                    'gen_ai.usage.details.cache_write_tokens': 10,
                    'gen_ai.usage.details.cache_read_tokens': 20,
                    'gen_ai.usage.details.input_audio_tokens': 10,
                    'gen_ai.usage.details.cache_audio_read_tokens': 5,
                    'gen_ai.usage.details.output_audio_tokens': 30,
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 200,
                    'operation.cost': 0.00188125,
                },
            },
        ]
    )

    assert capfire.log_exporter.exported_logs_as_dicts() == snapshot(
        [
            {
                'body': {'role': 'system', 'content': 'system_prompt'},
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'openai',
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
                    'gen_ai.system': 'openai',
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
                    'gen_ai.system': 'openai',
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
                    'gen_ai.system': 'openai',
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
Validation feedback:
retry_prompt2

Fix the errors and try again.\
""",
                    'role': 'user',
                },
                'severity_number': 9,
                'severity_text': None,
                'attributes': {
                    'gen_ai.system': 'openai',
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
                    'gen_ai.system': 'openai',
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
                        'content': [{'kind': 'text', 'text': 'text1'}, {'kind': 'text', 'text': 'text2'}],
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
                'attributes': {'gen_ai.system': 'openai', 'event.name': 'gen_ai.choice'},
                'timestamp': 14000000000,
                'observed_timestamp': 15000000000,
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
            allow_text_output=True,
            output_tools=[],
            output_mode='text',
            output_object=None,
        ),
    )


@requires_logfire_events
async def test_instrumented_model_stream(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel(), InstrumentationSettings(version=1, event_mode='logs'))

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
            allow_text_output=True,
            output_tools=[],
            output_mode='text',
            output_object=None,
        ),
    ) as response_stream:
        assert [event async for event in response_stream] == snapshot(
            [
                PartStartEvent(index=0, part=TextPart(content='text1')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='text2')),
                PartEndEvent(index=0, part=TextPart(content='text1text2')),
            ]
        )

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat gpt-4o',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 6000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-4o',
                    'server.address': 'example.com',
                    'server.port': 8000,
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
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat gpt-4o',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4o-2024-11-20',
                    'gen_ai.usage.input_tokens': 300,
                    'gen_ai.usage.output_tokens': 400,
                    'operation.cost': 0.00475,
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
                    'gen_ai.system': 'openai',
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
                'attributes': {'gen_ai.system': 'openai', 'event.name': 'gen_ai.choice'},
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
    model = InstrumentedModel(MyModel(), InstrumentationSettings(version=1, event_mode='logs'))

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
                allow_text_output=True,
                output_tools=[],
                output_mode='text',
                output_object=None,
            ),
        ) as response_stream:
            async for event in response_stream:  # pragma: no branch
                assert event == PartStartEvent(index=0, part=TextPart(content='text1'))
                raise RuntimeError

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat gpt-4o',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 7000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-4o',
                    'server.address': 'example.com',
                    'server.port': 8000,
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
                    'gen_ai.request.temperature': 1,
                    'logfire.msg': 'chat gpt-4o',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'gpt-4o-2024-11-20',
                    'gen_ai.usage.input_tokens': 300,
                    'gen_ai.usage.output_tokens': 400,
                    'operation.cost': 0.00475,
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
                    'gen_ai.system': 'openai',
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
                'attributes': {'gen_ai.system': 'openai', 'event.name': 'gen_ai.choice'},
                'timestamp': 4000000000,
                'observed_timestamp': 5000000000,
                'trace_id': 1,
                'span_id': 1,
                'trace_flags': 1,
            },
        ]
    )


@pytest.mark.parametrize('instrumentation_version', [1, 2])
async def test_instrumented_model_attributes_mode(capfire: CaptureLogfire, instrumentation_version: Literal[1, 2]):
    model = InstrumentedModel(
        MyModel(), InstrumentationSettings(event_mode='attributes', version=instrumentation_version)
    )
    assert model.system == 'openai'
    assert model.model_name == 'gpt-4o'

    messages = [
        ModelRequest(
            instructions='instructions',
            parts=[
                SystemPromptPart('system_prompt'),
                UserPromptPart('user_prompt'),
                ToolReturnPart('tool3', 'tool_return_content', 'tool_call_3'),
                RetryPromptPart('retry_prompt1', tool_name='tool4', tool_call_id='tool_call_4'),
                RetryPromptPart('retry_prompt2'),
                {},  # test unexpected parts  # type: ignore
            ],
        ),
        ModelResponse(parts=[TextPart('text3')]),
    ]
    await model.request(
        messages,
        model_settings=ModelSettings(temperature=1),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
            output_mode='text',
            output_object=None,
        ),
    )

    if instrumentation_version == 1:
        assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
            [
                {
                    'name': 'chat gpt-4o',
                    'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                    'parent': None,
                    'start_time': 1000000000,
                    'end_time': 2000000000,
                    'attributes': {
                        'gen_ai.operation.name': 'chat',
                        'gen_ai.system': 'openai',
                        'gen_ai.request.model': 'gpt-4o',
                        'server.address': 'example.com',
                        'server.port': 8000,
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
                        'gen_ai.request.temperature': 1,
                        'logfire.msg': 'chat gpt-4o',
                        'logfire.span_type': 'span',
                        'gen_ai.response.model': 'gpt-4o-2024-11-20',
                        'gen_ai.usage.input_tokens': 100,
                        'gen_ai.usage.output_tokens': 200,
                        'events': [
                            {
                                'content': 'instructions',
                                'role': 'system',
                                'gen_ai.system': 'openai',
                                'event.name': 'gen_ai.system.message',
                            },
                            {
                                'event.name': 'gen_ai.system.message',
                                'content': 'system_prompt',
                                'role': 'system',
                                'gen_ai.message.index': 0,
                                'gen_ai.system': 'openai',
                            },
                            {
                                'event.name': 'gen_ai.user.message',
                                'content': 'user_prompt',
                                'role': 'user',
                                'gen_ai.message.index': 0,
                                'gen_ai.system': 'openai',
                            },
                            {
                                'event.name': 'gen_ai.tool.message',
                                'content': 'tool_return_content',
                                'role': 'tool',
                                'name': 'tool3',
                                'id': 'tool_call_3',
                                'gen_ai.message.index': 0,
                                'gen_ai.system': 'openai',
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
                                'gen_ai.system': 'openai',
                            },
                            {
                                'event.name': 'gen_ai.user.message',
                                'content': """\
Validation feedback:
retry_prompt2

Fix the errors and try again.\
""",
                                'role': 'user',
                                'gen_ai.message.index': 0,
                                'gen_ai.system': 'openai',
                            },
                            {
                                'event.name': 'gen_ai.assistant.message',
                                'role': 'assistant',
                                'content': 'text3',
                                'gen_ai.message.index': 1,
                                'gen_ai.system': 'openai',
                            },
                            {
                                'index': 0,
                                'message': {
                                    'role': 'assistant',
                                    'content': [
                                        {'kind': 'text', 'text': 'text1'},
                                        {'kind': 'text', 'text': 'text2'},
                                    ],
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
                                'gen_ai.system': 'openai',
                                'event.name': 'gen_ai.choice',
                            },
                        ],
                        'gen_ai.usage.details.reasoning_tokens': 30,
                        'gen_ai.usage.details.cache_write_tokens': 10,
                        'gen_ai.usage.details.cache_read_tokens': 20,
                        'gen_ai.usage.details.input_audio_tokens': 10,
                        'gen_ai.usage.details.cache_audio_read_tokens': 5,
                        'gen_ai.usage.details.output_audio_tokens': 30,
                        'logfire.json_schema': {
                            'type': 'object',
                            'properties': {'events': {'type': 'array'}, 'model_request_parameters': {'type': 'object'}},
                        },
                        'operation.cost': 0.00188125,
                        'gen_ai.response.id': 'response_id',
                    },
                },
            ]
        )
    else:
        assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
            [
                {
                    'name': 'chat gpt-4o',
                    'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                    'parent': None,
                    'start_time': 1000000000,
                    'end_time': 2000000000,
                    'attributes': {
                        'gen_ai.operation.name': 'chat',
                        'gen_ai.system': 'openai',
                        'gen_ai.request.model': 'gpt-4o',
                        'server.address': 'example.com',
                        'server.port': 8000,
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
                        'gen_ai.request.temperature': 1,
                        'logfire.msg': 'chat gpt-4o',
                        'logfire.span_type': 'span',
                        'gen_ai.input.messages': [
                            {
                                'role': 'system',
                                'parts': [
                                    {'type': 'text', 'content': 'system_prompt'},
                                ],
                            },
                            {
                                'role': 'user',
                                'parts': [
                                    {'type': 'text', 'content': 'user_prompt'},
                                    {
                                        'type': 'tool_call_response',
                                        'id': 'tool_call_3',
                                        'name': 'tool3',
                                        'result': 'tool_return_content',
                                    },
                                    {
                                        'type': 'tool_call_response',
                                        'id': 'tool_call_4',
                                        'name': 'tool4',
                                        'result': """\
retry_prompt1

Fix the errors and try again.\
""",
                                    },
                                    {
                                        'type': 'text',
                                        'content': """\
Validation feedback:
retry_prompt2

Fix the errors and try again.\
""",
                                    },
                                ],
                            },
                            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'text3'}]},
                        ],
                        'gen_ai.output.messages': [
                            {
                                'role': 'assistant',
                                'parts': [
                                    {'type': 'text', 'content': 'text1'},
                                    {'type': 'tool_call', 'id': 'tool_call_1', 'name': 'tool1', 'arguments': 'args1'},
                                    {
                                        'type': 'tool_call',
                                        'id': 'tool_call_2',
                                        'name': 'tool2',
                                        'arguments': {'args2': 3},
                                    },
                                    {'type': 'text', 'content': 'text2'},
                                ],
                            }
                        ],
                        'gen_ai.response.model': 'gpt-4o-2024-11-20',
                        'gen_ai.system_instructions': [{'type': 'text', 'content': 'instructions'}],
                        'gen_ai.usage.input_tokens': 100,
                        'gen_ai.usage.output_tokens': 200,
                        'gen_ai.usage.details.reasoning_tokens': 30,
                        'gen_ai.usage.details.cache_write_tokens': 10,
                        'gen_ai.usage.details.cache_read_tokens': 20,
                        'gen_ai.usage.details.input_audio_tokens': 10,
                        'gen_ai.usage.details.cache_audio_read_tokens': 5,
                        'gen_ai.usage.details.output_audio_tokens': 30,
                        'logfire.json_schema': {
                            'type': 'object',
                            'properties': {
                                'gen_ai.input.messages': {'type': 'array'},
                                'gen_ai.output.messages': {'type': 'array'},
                                'gen_ai.system_instructions': {'type': 'array'},
                                'model_request_parameters': {'type': 'object'},
                            },
                        },
                        'operation.cost': 0.00188125,
                        'gen_ai.response.id': 'response_id',
                    },
                },
            ]
        )

    assert capfire.get_collected_metrics() == snapshot(
        [
            {
                'name': 'gen_ai.client.token.usage',
                'description': 'Measures number of input and output tokens used',
                'unit': '{token}',
                'data': {
                    'data_points': [
                        {
                            'attributes': {
                                'gen_ai.system': 'openai',
                                'gen_ai.operation.name': 'chat',
                                'gen_ai.request.model': 'gpt-4o',
                                'gen_ai.response.model': 'gpt-4o-2024-11-20',
                                'gen_ai.token.type': 'input',
                            },
                            'start_time_unix_nano': IsInt(),
                            'time_unix_nano': IsInt(),
                            'count': 1,
                            'sum': 100,
                            'scale': 20,
                            'zero_count': 0,
                            'positive': {'offset': 6966588, 'bucket_counts': [1]},
                            'negative': {'offset': 0, 'bucket_counts': [0]},
                            'flags': 0,
                            'min': 100,
                            'max': 100,
                            'exemplars': [],
                        },
                        {
                            'attributes': {
                                'gen_ai.system': 'openai',
                                'gen_ai.operation.name': 'chat',
                                'gen_ai.request.model': 'gpt-4o',
                                'gen_ai.response.model': 'gpt-4o-2024-11-20',
                                'gen_ai.token.type': 'output',
                            },
                            'start_time_unix_nano': IsInt(),
                            'time_unix_nano': IsInt(),
                            'count': 1,
                            'sum': 200,
                            'scale': 20,
                            'zero_count': 0,
                            'positive': {'offset': 8015164, 'bucket_counts': [1]},
                            'negative': {'offset': 0, 'bucket_counts': [0]},
                            'flags': 0,
                            'min': 200,
                            'max': 200,
                            'exemplars': [],
                        },
                    ],
                    'aggregation_temporality': 1,
                },
            },
            {
                'name': 'operation.cost',
                'description': 'Monetary cost',
                'unit': '{USD}',
                'data': {
                    'data_points': [
                        {
                            'attributes': {
                                'gen_ai.system': 'openai',
                                'gen_ai.operation.name': 'chat',
                                'gen_ai.request.model': 'gpt-4o',
                                'gen_ai.response.model': 'gpt-4o-2024-11-20',
                                'gen_ai.token.type': 'input',
                            },
                            'start_time_unix_nano': IsInt(),
                            'time_unix_nano': IsInt(),
                            'count': 1,
                            'sum': 0.00018125,
                            'scale': 20,
                            'zero_count': 0,
                            'positive': {'offset': -13033519, 'bucket_counts': [1]},
                            'negative': {'offset': 0, 'bucket_counts': [0]},
                            'flags': 0,
                            'min': 0.00018125,
                            'max': 0.00018125,
                            'exemplars': [],
                        },
                        {
                            'attributes': {
                                'gen_ai.system': 'openai',
                                'gen_ai.operation.name': 'chat',
                                'gen_ai.request.model': 'gpt-4o',
                                'gen_ai.response.model': 'gpt-4o-2024-11-20',
                                'gen_ai.token.type': 'output',
                            },
                            'start_time_unix_nano': IsInt(),
                            'time_unix_nano': IsInt(),
                            'count': 1,
                            'sum': 0.0017,
                            'scale': 20,
                            'zero_count': 0,
                            'positive': {'offset': -9647161, 'bucket_counts': [1]},
                            'negative': {'offset': 0, 'bucket_counts': [0]},
                            'flags': 0,
                            'min': 0.0017,
                            'max': 0.0017,
                            'exemplars': [],
                        },
                    ],
                    'aggregation_temporality': 1,
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
        ModelResponse(parts=[ToolCallPart('tool', {'arg': Foo()}, tool_call_id='tool_call_id')]),
        ModelRequest(parts=[ToolReturnPart('tool', Bar(), tool_call_id='return_tool_call_id')]),
    ]

    settings = InstrumentationSettings()
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == [
        {
            'body': "{'role': 'assistant', 'tool_calls': [{'id': 'tool_call_id', 'type': 'function', 'function': {'name': 'tool', 'arguments': {'arg': Foo()}}}]}",
            'gen_ai.message.index': 0,
            'event.name': 'gen_ai.assistant.message',
        },
        {
            'body': 'Unable to serialize: error!',
            'gen_ai.message.index': 1,
            'event.name': 'gen_ai.tool.message',
        },
    ]
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'assistant',
                'parts': [{'type': 'tool_call', 'id': 'tool_call_id', 'name': 'tool', 'arguments': {'arg': 'Foo()'}}],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'type': 'tool_call_response',
                        'id': 'return_tool_call_id',
                        'name': 'tool',
                        'result': 'Unable to serialize: error!',
                    }
                ],
            },
        ]
    )


def test_messages_to_otel_events_instructions():
    messages = [
        ModelRequest(instructions='instructions', parts=[UserPromptPart('user_prompt')]),
        ModelResponse(parts=[TextPart('text1')]),
    ]
    settings = InstrumentationSettings()
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {'content': 'instructions', 'role': 'system', 'event.name': 'gen_ai.system.message'},
            {'content': 'user_prompt', 'role': 'user', 'gen_ai.message.index': 0, 'event.name': 'gen_ai.user.message'},
            {
                'role': 'assistant',
                'content': 'text1',
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.assistant.message',
            },
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'user_prompt'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'text1'}]},
        ]
    )


def test_messages_to_otel_events_instructions_multiple_messages():
    messages = [
        ModelRequest(instructions='instructions', parts=[UserPromptPart('user_prompt')]),
        ModelResponse(parts=[TextPart('text1')]),
        ModelRequest(instructions='instructions2', parts=[UserPromptPart('user_prompt2')]),
    ]
    settings = InstrumentationSettings()
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {'content': 'instructions2', 'role': 'system', 'event.name': 'gen_ai.system.message'},
            {'content': 'user_prompt', 'role': 'user', 'gen_ai.message.index': 0, 'event.name': 'gen_ai.user.message'},
            {
                'role': 'assistant',
                'content': 'text1',
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.assistant.message',
            },
            {'content': 'user_prompt2', 'role': 'user', 'gen_ai.message.index': 2, 'event.name': 'gen_ai.user.message'},
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'user_prompt'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'text1'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'user_prompt2'}]},
        ]
    )


def test_messages_to_otel_events_image_url(document_content: BinaryContent):
    messages = [
        ModelRequest(parts=[UserPromptPart(content=['user_prompt', ImageUrl('https://example.com/image.png')])]),
        ModelRequest(parts=[UserPromptPart(content=['user_prompt2', AudioUrl('https://example.com/audio.mp3')])]),
        ModelRequest(parts=[UserPromptPart(content=['user_prompt3', DocumentUrl('https://example.com/document.pdf')])]),
        ModelRequest(parts=[UserPromptPart(content=['user_prompt4', VideoUrl('https://example.com/video.mp4')])]),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'user_prompt5',
                        ImageUrl('https://example.com/image2.png'),
                        AudioUrl('https://example.com/audio2.mp3'),
                        DocumentUrl('https://example.com/document2.pdf'),
                        VideoUrl('https://example.com/video2.mp4'),
                    ]
                )
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content=['user_prompt6', document_content])]),
        ModelResponse(parts=[TextPart('text1')]),
        ModelResponse(parts=[FilePart(content=document_content)]),
    ]
    settings = InstrumentationSettings()
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {
                'content': ['user_prompt', {'kind': 'image-url', 'url': 'https://example.com/image.png'}],
                'role': 'user',
                'gen_ai.message.index': 0,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': ['user_prompt2', {'kind': 'audio-url', 'url': 'https://example.com/audio.mp3'}],
                'role': 'user',
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': ['user_prompt3', {'kind': 'document-url', 'url': 'https://example.com/document.pdf'}],
                'role': 'user',
                'gen_ai.message.index': 2,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': ['user_prompt4', {'kind': 'video-url', 'url': 'https://example.com/video.mp4'}],
                'role': 'user',
                'gen_ai.message.index': 3,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': [
                    'user_prompt5',
                    {'kind': 'image-url', 'url': 'https://example.com/image2.png'},
                    {'kind': 'audio-url', 'url': 'https://example.com/audio2.mp3'},
                    {'kind': 'document-url', 'url': 'https://example.com/document2.pdf'},
                    {'kind': 'video-url', 'url': 'https://example.com/video2.mp4'},
                ],
                'role': 'user',
                'gen_ai.message.index': 4,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': [
                    'user_prompt6',
                    {'kind': 'binary', 'binary_content': IsStr(), 'media_type': 'application/pdf'},
                ],
                'role': 'user',
                'gen_ai.message.index': 5,
                'event.name': 'gen_ai.user.message',
            },
            {
                'role': 'assistant',
                'content': 'text1',
                'gen_ai.message.index': 6,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'kind': 'binary',
                        'media_type': 'application/pdf',
                        'binary_content': IsStr(),
                    }
                ],
                'gen_ai.message.index': 7,
                'event.name': 'gen_ai.assistant.message',
            },
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt'},
                    {'type': 'image-url', 'url': 'https://example.com/image.png'},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt2'},
                    {'type': 'audio-url', 'url': 'https://example.com/audio.mp3'},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt3'},
                    {'type': 'document-url', 'url': 'https://example.com/document.pdf'},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt4'},
                    {'type': 'video-url', 'url': 'https://example.com/video.mp4'},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt5'},
                    {'type': 'image-url', 'url': 'https://example.com/image2.png'},
                    {'type': 'audio-url', 'url': 'https://example.com/audio2.mp3'},
                    {'type': 'document-url', 'url': 'https://example.com/document2.pdf'},
                    {'type': 'video-url', 'url': 'https://example.com/video2.mp4'},
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt6'},
                    {
                        'type': 'binary',
                        'media_type': 'application/pdf',
                        'content': IsStr(),
                    },
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'text1'}]},
            {
                'role': 'assistant',
                'parts': [
                    {
                        'type': 'binary',
                        'media_type': 'application/pdf',
                        'content': IsStr(),
                    }
                ],
            },
        ]
    )


def test_messages_to_otel_events_without_binary_content(document_content: BinaryContent):
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['user_prompt6', document_content])]),
    ]
    settings = InstrumentationSettings(include_binary_content=False)
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {
                'content': ['user_prompt6', {'kind': 'binary', 'media_type': 'application/pdf'}],
                'role': 'user',
                'gen_ai.message.index': 0,
                'event.name': 'gen_ai.user.message',
            }
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'user_prompt6'},
                    {'type': 'binary', 'media_type': 'application/pdf'},
                ],
            }
        ]
    )


def test_messages_without_content(document_content: BinaryContent):
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('system_prompt')]),
        ModelResponse(parts=[TextPart('text1')]),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'user_prompt1',
                        VideoUrl('https://example.com/video.mp4'),
                        ImageUrl('https://example.com/image.png'),
                        AudioUrl('https://example.com/audio.mp3'),
                        DocumentUrl('https://example.com/document.pdf'),
                        document_content,
                    ]
                )
            ]
        ),
        ModelResponse(parts=[TextPart('text2'), ToolCallPart(tool_name='my_tool', args={'a': 13, 'b': 4})]),
        ModelRequest(parts=[ToolReturnPart('tool', 'tool_return_content', 'tool_call_1')]),
        ModelRequest(parts=[RetryPromptPart('retry_prompt', tool_name='tool', tool_call_id='tool_call_2')]),
        ModelRequest(parts=[UserPromptPart(content=['user_prompt2', document_content])]),
        ModelRequest(parts=[UserPromptPart('simple text prompt')]),
        ModelResponse(parts=[FilePart(content=document_content)]),
    ]
    settings = InstrumentationSettings(include_content=False)
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {
                'role': 'system',
                'gen_ai.message.index': 0,
                'event.name': 'gen_ai.system.message',
            },
            {
                'role': 'assistant',
                'content': [{'kind': 'text'}],
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'content': [
                    {'kind': 'text'},
                    {'kind': 'video-url'},
                    {'kind': 'image-url'},
                    {'kind': 'audio-url'},
                    {'kind': 'document-url'},
                    {'kind': 'binary', 'media_type': 'application/pdf'},
                ],
                'role': 'user',
                'gen_ai.message.index': 2,
                'event.name': 'gen_ai.user.message',
            },
            {
                'role': 'assistant',
                'content': [{'kind': 'text'}],
                'tool_calls': [
                    {
                        'id': IsStr(),
                        'type': 'function',
                        'function': {'name': 'my_tool'},
                    }
                ],
                'gen_ai.message.index': 3,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'role': 'tool',
                'id': 'tool_call_1',
                'name': 'tool',
                'gen_ai.message.index': 4,
                'event.name': 'gen_ai.tool.message',
            },
            {
                'role': 'tool',
                'id': 'tool_call_2',
                'name': 'tool',
                'gen_ai.message.index': 5,
                'event.name': 'gen_ai.tool.message',
            },
            {
                'content': [{'kind': 'text'}, {'kind': 'binary', 'media_type': 'application/pdf'}],
                'role': 'user',
                'gen_ai.message.index': 6,
                'event.name': 'gen_ai.user.message',
            },
            {
                'content': {'kind': 'text'},
                'role': 'user',
                'gen_ai.message.index': 7,
                'event.name': 'gen_ai.user.message',
            },
            {
                'role': 'assistant',
                'content': [{'kind': 'binary', 'media_type': 'application/pdf'}],
                'gen_ai.message.index': 8,
                'event.name': 'gen_ai.assistant.message',
            },
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {'role': 'system', 'parts': [{'type': 'text'}]},
            {'role': 'assistant', 'parts': [{'type': 'text'}]},
            {
                'role': 'user',
                'parts': [
                    {'type': 'text'},
                    {'type': 'video-url'},
                    {'type': 'image-url'},
                    {'type': 'audio-url'},
                    {'type': 'document-url'},
                    {'type': 'binary', 'media_type': 'application/pdf'},
                ],
            },
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text'},
                    {'type': 'tool_call', 'id': IsStr(), 'name': 'my_tool'},
                ],
            },
            {'role': 'user', 'parts': [{'type': 'tool_call_response', 'id': 'tool_call_1', 'name': 'tool'}]},
            {'role': 'user', 'parts': [{'type': 'tool_call_response', 'id': 'tool_call_2', 'name': 'tool'}]},
            {'role': 'user', 'parts': [{'type': 'text'}, {'type': 'binary', 'media_type': 'application/pdf'}]},
            {'role': 'user', 'parts': [{'type': 'text'}]},
            {'role': 'assistant', 'parts': [{'type': 'binary', 'media_type': 'application/pdf'}]},
        ]
    )


def test_message_with_thinking_parts():
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart('text1'), ThinkingPart('thinking1'), TextPart('text2')]),
        ModelResponse(parts=[ThinkingPart('thinking2')]),
        ModelResponse(parts=[ThinkingPart('thinking3'), TextPart('text3')]),
    ]
    settings = InstrumentationSettings()
    assert [InstrumentedModel.event_to_dict(e) for e in settings.messages_to_otel_events(messages)] == snapshot(
        [
            {
                'role': 'assistant',
                'content': [
                    {'kind': 'text', 'text': 'text1'},
                    {'kind': 'thinking', 'text': 'thinking1'},
                    {'kind': 'text', 'text': 'text2'},
                ],
                'gen_ai.message.index': 0,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'role': 'assistant',
                'content': [{'kind': 'thinking', 'text': 'thinking2'}],
                'gen_ai.message.index': 1,
                'event.name': 'gen_ai.assistant.message',
            },
            {
                'role': 'assistant',
                'content': [{'kind': 'thinking', 'text': 'thinking3'}, {'kind': 'text', 'text': 'text3'}],
                'gen_ai.message.index': 2,
                'event.name': 'gen_ai.assistant.message',
            },
        ]
    )
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'content': 'text1'},
                    {'type': 'thinking', 'content': 'thinking1'},
                    {'type': 'text', 'content': 'text2'},
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'thinking', 'content': 'thinking2'}]},
            {
                'role': 'assistant',
                'parts': [{'type': 'thinking', 'content': 'thinking3'}, {'type': 'text', 'content': 'text3'}],
            },
        ]
    )


def test_deprecated_event_mode_warning():
    with pytest.warns(
        UserWarning,
        match='event_mode is only relevant for version=1 which is deprecated and will be removed in a future release',
    ):
        settings = InstrumentationSettings(event_mode='logs')
    assert settings.event_mode == 'logs'
    assert settings.version == 1
    assert InstrumentationSettings().version == 2


async def test_response_cost_error(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    model = InstrumentedModel(MyModel())

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('user_prompt')])]
    monkeypatch.setattr(ModelResponse, 'cost', None)

    with warns(
        snapshot(
            [
                "CostCalculationFailedWarning: Failed to get cost from response: TypeError: 'NoneType' object is not callable"
            ]
        )
    ):
        await model.request(messages, model_settings=ModelSettings(), model_request_parameters=ModelRequestParameters())

    assert capfire.exporter.exported_spans_as_dict(parse_json_attributes=True) == snapshot(
        [
            {
                'name': 'chat gpt-4o',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 2000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'openai',
                    'gen_ai.request.model': 'gpt-4o',
                    'server.address': 'example.com',
                    'server.port': 8000,
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
                    'logfire.msg': 'chat gpt-4o',
                    'gen_ai.input.messages': [{'role': 'user', 'parts': [{'type': 'text', 'content': 'user_prompt'}]}],
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'text', 'content': 'text1'},
                                {'type': 'tool_call', 'id': 'tool_call_1', 'name': 'tool1', 'arguments': 'args1'},
                                {'type': 'tool_call', 'id': 'tool_call_2', 'name': 'tool2', 'arguments': {'args2': 3}},
                                {'type': 'text', 'content': 'text2'},
                            ],
                        }
                    ],
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    },
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 200,
                    'gen_ai.usage.details.reasoning_tokens': 30,
                    'gen_ai.usage.details.cache_write_tokens': 10,
                    'gen_ai.usage.details.cache_read_tokens': 20,
                    'gen_ai.usage.details.input_audio_tokens': 10,
                    'gen_ai.usage.details.cache_audio_read_tokens': 5,
                    'gen_ai.usage.details.output_audio_tokens': 30,
                    'gen_ai.response.model': 'gpt-4o-2024-11-20',
                    'gen_ai.response.id': 'response_id',
                },
            }
        ]
    )


def test_message_with_builtin_tool_calls():
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                TextPart('text1'),
                BuiltinToolCallPart('code_execution', {'code': '2 * 2'}, tool_call_id='tool_call_1'),
                BuiltinToolReturnPart('code_execution', {'output': '4'}, tool_call_id='tool_call_1'),
                TextPart('text2'),
                BuiltinToolCallPart(
                    'web_search',
                    '{"query": "weather: San Francisco, CA", "type": "search"}',
                    tool_call_id='tool_call_2',
                ),
                BuiltinToolReturnPart(
                    'web_search',
                    [
                        {
                            'url': 'https://www.weather.com/weather/today/l/USCA0987:1:US',
                            'title': 'Weather in San Francisco',
                        }
                    ],
                    tool_call_id='tool_call_2',
                ),
                TextPart('text3'),
            ]
        ),
    ]
    settings = InstrumentationSettings()
    # Built-in tool calls are only included in v2-style messages, not v1-style events,
    # as the spec does not yet allow tool results coming from the assistant,
    # and Logfire has special handling for the `type='tool_call_response', 'builtin=True'` messages, but not events.
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'content': 'text1'},
                    {
                        'type': 'tool_call',
                        'id': 'tool_call_1',
                        'name': 'code_execution',
                        'builtin': True,
                        'arguments': {'code': '2 * 2'},
                    },
                    {
                        'type': 'tool_call_response',
                        'id': 'tool_call_1',
                        'name': 'code_execution',
                        'builtin': True,
                        'result': {'output': '4'},
                    },
                    {'type': 'text', 'content': 'text2'},
                    {
                        'type': 'tool_call',
                        'id': 'tool_call_2',
                        'name': 'web_search',
                        'builtin': True,
                        'arguments': '{"query": "weather: San Francisco, CA", "type": "search"}',
                    },
                    {
                        'type': 'tool_call_response',
                        'id': 'tool_call_2',
                        'name': 'web_search',
                        'builtin': True,
                        'result': [
                            {
                                'url': 'https://www.weather.com/weather/today/l/USCA0987:1:US',
                                'title': 'Weather in San Francisco',
                            }
                        ],
                    },
                    {'type': 'text', 'content': 'text3'},
                ],
            }
        ]
    )


def test_cache_point_in_user_prompt():
    """Test that CachePoint is correctly skipped in OpenTelemetry conversion.

    CachePoint is a marker for prompt caching and should not be included in the
    OpenTelemetry message parts output.
    """
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['text before', CachePoint(), 'text after'])]),
    ]
    settings = InstrumentationSettings()

    # Test otel_message_parts - CachePoint should be skipped
    assert settings.messages_to_otel_messages(messages) == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'text before'},
                    {'type': 'text', 'content': 'text after'},
                ],
            }
        ]
    )

    # Test with multiple CachePoints
    messages_multi: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content=['first', CachePoint(), 'second', CachePoint(), 'third']),
            ]
        ),
    ]
    assert settings.messages_to_otel_messages(messages_multi) == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'first'},
                    {'type': 'text', 'content': 'second'},
                    {'type': 'text', 'content': 'third'},
                ],
            }
        ]
    )

    # Test with CachePoint mixed with other content types
    messages_mixed: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'context',
                        CachePoint(),
                        ImageUrl('https://example.com/image.jpg'),
                        CachePoint(),
                        'question',
                    ]
                ),
            ]
        ),
    ]
    assert settings.messages_to_otel_messages(messages_mixed) == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'context'},
                    {'type': 'image-url', 'url': 'https://example.com/image.jpg'},
                    {'type': 'text', 'content': 'question'},
                ],
            }
        ]
    )
