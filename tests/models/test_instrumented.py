from __future__ import annotations

import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage

from ..conftest import try_import

with try_import() as imports_successful:
    import logfire_api
    from logfire.testing import CaptureLogfire


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='logfire not installed'),
    pytest.mark.anyio,
]


class MyModel(Model):
    @property
    def system(self) -> str:
        return 'my_system'

    @property
    def model_name(self) -> str:
        return 'my_model'

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


@pytest.mark.anyio
async def test_instrumented_model(capfire: CaptureLogfire):
    model = InstrumentedModel(MyModel())
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
                'name': 'gen_ai.system.message',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 2000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.system.message',
                    'logfire.msg': 'gen_ai.system.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': 'system_prompt',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{}}}',
                },
            },
            {
                'name': 'gen_ai.user.message',
                'context': {'trace_id': 1, 'span_id': 4, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 3000000000,
                'end_time': 3000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.user.message',
                    'logfire.msg': 'gen_ai.user.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': 'user_prompt',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{}}}',
                },
            },
            {
                'name': 'gen_ai.tool.message',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 4000000000,
                'end_time': 4000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.tool.message',
                    'logfire.msg': 'gen_ai.tool.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': 'tool_return_content',
                    'id': 'tool_call_3',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{},"id":{}}}',
                },
            },
            {
                'name': 'gen_ai.tool.message',
                'context': {'trace_id': 1, 'span_id': 6, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.tool.message',
                    'logfire.msg': 'gen_ai.tool.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': """\
retry_prompt1

Fix the errors and try again.\
""",
                    'id': 'tool_call_4',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{},"id":{}}}',
                },
            },
            {
                'name': 'gen_ai.user.message',
                'context': {'trace_id': 1, 'span_id': 7, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 6000000000,
                'end_time': 6000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.user.message',
                    'logfire.msg': 'gen_ai.user.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': """\
retry_prompt2

Fix the errors and try again.\
""",
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{}}}',
                },
            },
            {
                'name': 'gen_ai.assistant.message',
                'context': {'trace_id': 1, 'span_id': 8, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 7000000000,
                'end_time': 7000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.assistant.message',
                    'logfire.msg': 'gen_ai.assistant.message',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'content': 'text3',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"content":{}}}',
                },
            },
            {
                'name': 'gen_ai.choice',
                'context': {'trace_id': 1, 'span_id': 9, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 8000000000,
                'end_time': 8000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.choice',
                    'logfire.msg': 'gen_ai.choice',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'index': 0,
                    'message': IsJson(
                        {
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
                        }
                    ),
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"index":{},"message":{"type":"object"}}}',
                },
            },
            {
                'name': 'gen_ai.choice',
                'context': {'trace_id': 1, 'span_id': 10, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 9000000000,
                'end_time': 9000000000,
                'attributes': {
                    'logfire.span_type': 'log',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'gen_ai.choice',
                    'logfire.msg': 'gen_ai.choice',
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.system': 'my_system',
                    'index': 0,
                    'message': '{"content":"text2"}',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.system":{},"index":{},"message":{"type":"object"}}}',
                },
            },
            {
                'name': 'chat my_model',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 10000000000,
                'attributes': {
                    'code.filepath': 'test_instrumented.py',
                    'code.function': 'test_instrumented_model',
                    'code.lineno': 123,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'my_system',
                    'gen_ai.request.model': 'my_model',
                    'gen_ai.request.temperature': 1,
                    'logfire.msg_template': 'chat my_model',
                    'logfire.msg': 'chat my_model',
                    'logfire.span_type': 'span',
                    'gen_ai.response.model': 'my_model_123',
                    'gen_ai.usage.input_tokens': 100,
                    'gen_ai.usage.output_tokens': 200,
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.operation.name":{},"gen_ai.system":{},"gen_ai.request.model":{},"gen_ai.request.temperature":{},"gen_ai.response.model":{},"gen_ai.usage.input_tokens":{},"gen_ai.usage.output_tokens":{}}}',
                },
            },
        ]
    )


@pytest.mark.anyio
async def test_instrumented_model_not_recording(capfire: CaptureLogfire):
    logfire_instance = logfire_api.DEFAULT_LOGFIRE_INSTANCE.with_trace_sample_rate(0)
    model = InstrumentedModel(MyModel(), logfire_instance)

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

    assert capfire.exporter.exported_spans_as_dict() == snapshot([])
