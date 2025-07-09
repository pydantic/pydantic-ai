from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
from dirty_equals import IsInt, IsJson, IsList
from inline_snapshot import snapshot
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent
from pydantic_ai._utils import get_traceparent
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel

from .conftest import IsStr

try:
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    logfire_installed = False
else:
    logfire_installed = True


class SpanSummary(TypedDict):
    id: int
    message: str
    children: NotRequired[list[SpanSummary]]


@dataclass(init=False)
class LogfireSummary:
    traces: list[SpanSummary]
    attributes: dict[int, dict[str, Any]]

    def __init__(self, capfire: CaptureLogfire):
        spans = capfire.exporter.exported_spans_as_dict()
        spans.sort(key=lambda s: s['start_time'])
        self.traces = []
        span_lookup: dict[tuple[str, str], SpanSummary] = {}
        self.attributes = {}
        id_counter = 0
        for span in spans:
            tid = span['context']['trace_id'], span['context']['span_id']
            span_lookup[tid] = span_summary = SpanSummary(id=id_counter, message=span['attributes']['logfire.msg'])
            self.attributes[id_counter] = span['attributes']
            id_counter += 1
            if parent := span['parent']:
                parent_span = span_lookup[(parent['trace_id'], parent['span_id'])]
                parent_span.setdefault('children', []).append(span_summary)
            else:
                self.traces.append(span_summary)


@pytest.fixture
def get_logfire_summary(capfire: CaptureLogfire) -> Callable[[], LogfireSummary]:
    def get_summary() -> LogfireSummary:
        return LogfireSummary(capfire)

    return get_summary


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [
        True,
        False,
        InstrumentationSettings(event_mode='attributes'),
        InstrumentationSettings(event_mode='logs'),
    ],
)
def test_logfire(
    get_logfire_summary: Callable[[], LogfireSummary],
    instrument: InstrumentationSettings | bool,
    capfire: CaptureLogfire,
) -> None:
    my_agent = Agent(model=TestModel(), instrument=instrument)

    @my_agent.tool_plain
    async def my_ret(x: int) -> str:
        return str(x + 1)

    result = my_agent.run_sync('Hello')
    assert result.output == snapshot('{"my_ret":"1"}')

    summary = get_logfire_summary()
    if instrument is False:
        assert summary.traces == []
        return

    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'message': 'my_agent run',
                'children': [
                    {'id': 1, 'message': 'chat test'},
                    {
                        'id': 2,
                        'message': 'running 1 tool',
                        'children': [
                            {'id': 3, 'message': 'running tool: my_ret'},
                        ],
                    },
                    {'id': 4, 'message': 'chat test'},
                ],
            }
        ]
    )
    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'gen_ai.usage.input_tokens': 103,
            'gen_ai.usage.output_tokens': 12,
            'all_messages_events': IsJson(
                snapshot(
                    [
                        {
                            'content': 'Hello',
                            'role': 'user',
                            'gen_ai.message.index': 0,
                            'event.name': 'gen_ai.user.message',
                        },
                        {
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': IsStr(),
                                    'type': 'function',
                                    'function': {
                                        'name': 'my_ret',
                                        'arguments': {'x': 0},
                                    },
                                }
                            ],
                            'gen_ai.message.index': 1,
                            'event.name': 'gen_ai.assistant.message',
                        },
                        {
                            'content': '1',
                            'role': 'tool',
                            'id': IsStr(),
                            'name': 'my_ret',
                            'gen_ai.message.index': 2,
                            'event.name': 'gen_ai.tool.message',
                        },
                        {
                            'role': 'assistant',
                            'content': '{"my_ret":"1"}',
                            'gen_ai.message.index': 3,
                            'event.name': 'gen_ai.assistant.message',
                        },
                    ]
                )
            ),
            'final_result': '{"my_ret":"1"}',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {'all_messages_events': {'type': 'array'}, 'final_result': {'type': 'object'}},
                    }
                )
            ),
        }
    )
    chat_span_attributes = summary.attributes[1]
    if instrument is True or instrument.event_mode == 'attributes':
        if hasattr(capfire, 'get_collected_metrics'):  # pragma: no branch
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
                                        'gen_ai.system': 'test',
                                        'gen_ai.operation.name': 'chat',
                                        'gen_ai.request.model': 'test',
                                        'gen_ai.response.model': 'test',
                                        'gen_ai.token.type': 'input',
                                    },
                                    'start_time_unix_nano': IsInt(),
                                    'time_unix_nano': IsInt(),
                                    'count': 2,
                                    'sum': 103,
                                    'scale': 12,
                                    'zero_count': 0,
                                    'positive': {
                                        'offset': 23234,
                                        'bucket_counts': IsList(length=...),  # type: ignore
                                    },
                                    'negative': {'offset': 0, 'bucket_counts': [0]},
                                    'flags': 0,
                                    'min': 51,
                                    'max': 52,
                                    'exemplars': IsList(length=...),  # type: ignore
                                },
                                {
                                    'attributes': {
                                        'gen_ai.system': 'test',
                                        'gen_ai.operation.name': 'chat',
                                        'gen_ai.request.model': 'test',
                                        'gen_ai.response.model': 'test',
                                        'gen_ai.token.type': 'output',
                                    },
                                    'start_time_unix_nano': IsInt(),
                                    'time_unix_nano': IsInt(),
                                    'count': 2,
                                    'sum': 12,
                                    'scale': 7,
                                    'zero_count': 0,
                                    'positive': {
                                        'offset': 255,
                                        'bucket_counts': IsList(length=...),  # type: ignore
                                    },
                                    'negative': {'offset': 0, 'bucket_counts': [0]},
                                    'flags': 0,
                                    'min': 4,
                                    'max': 8,
                                    'exemplars': IsList(length=...),  # type: ignore
                                },
                            ],
                            'aggregation_temporality': 1,
                        },
                    }
                ]
            )

        attribute_mode_attributes = {k: chat_span_attributes.pop(k) for k in ['events']}
        assert attribute_mode_attributes == snapshot(
            {
                'events': IsJson(
                    snapshot(
                        [
                            {
                                'event.name': 'gen_ai.user.message',
                                'content': 'Hello',
                                'role': 'user',
                                'gen_ai.message.index': 0,
                                'gen_ai.system': 'test',
                            },
                            {
                                'event.name': 'gen_ai.choice',
                                'index': 0,
                                'message': {
                                    'role': 'assistant',
                                    'tool_calls': [
                                        {
                                            'id': IsStr(),
                                            'type': 'function',
                                            'function': {'name': 'my_ret', 'arguments': {'x': 0}},
                                        }
                                    ],
                                },
                                'gen_ai.system': 'test',
                            },
                        ]
                    )
                ),
            }
        )

    assert chat_span_attributes == snapshot(
        {
            'gen_ai.operation.name': 'chat',
            'gen_ai.system': 'test',
            'gen_ai.request.model': 'test',
            'model_request_parameters': IsJson(
                snapshot(
                    {
                        'function_tools': [
                            {
                                'name': 'my_ret',
                                'description': None,
                                'parameters_json_schema': {
                                    'additionalProperties': False,
                                    'properties': {'x': {'type': 'integer'}},
                                    'required': ['x'],
                                    'type': 'object',
                                },
                                'outer_typed_dict_key': None,
                                'strict': None,
                            }
                        ],
                        'output_mode': 'text',
                        'output_tools': [],
                        'output_object': None,
                        'allow_text_output': True,
                    }
                )
            ),
            'logfire.json_schema': IsJson(),
            'logfire.span_type': 'span',
            'logfire.msg': 'chat test',
            'gen_ai.response.model': 'test',
            'gen_ai.usage.input_tokens': 51,
            'gen_ai.usage.output_tokens': 4,
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instructions_with_structured_output(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(model=TestModel(), instructions='Here are some instructions', instrument=True)

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == snapshot(MyOutput(content='a'))

    summary = get_logfire_summary()
    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'gen_ai.usage.input_tokens': 51,
            'gen_ai.usage.output_tokens': 5,
            'all_messages_events': IsJson(
                snapshot(
                    [
                        {
                            'content': 'Here are some instructions',
                            'role': 'system',
                            'event.name': 'gen_ai.system.message',
                        },
                        {
                            'content': 'Hello',
                            'role': 'user',
                            'gen_ai.message.index': 0,
                            'event.name': 'gen_ai.user.message',
                        },
                        {
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': IsStr(),
                                    'type': 'function',
                                    'function': {'name': 'final_result', 'arguments': {'content': 'a'}},
                                }
                            ],
                            'gen_ai.message.index': 1,
                            'event.name': 'gen_ai.assistant.message',
                        },
                        {
                            'content': 'Final result processed.',
                            'role': 'tool',
                            'id': IsStr(),
                            'name': 'final_result',
                            'gen_ai.message.index': 2,
                            'event.name': 'gen_ai.tool.message',
                        },
                    ]
                )
            ),
            'final_result': '{"content": "a"}',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {'all_messages_events': {'type': 'array'}, 'final_result': {'type': 'object'}},
                    }
                )
            ),
        }
    )
    chat_span_attributes = summary.attributes[1]
    assert chat_span_attributes['events'] == snapshot(
        IsJson(
            snapshot(
                [
                    {
                        'content': 'Here are some instructions',
                        'role': 'system',
                        'gen_ai.system': 'test',
                        'event.name': 'gen_ai.system.message',
                    },
                    {
                        'event.name': 'gen_ai.user.message',
                        'content': 'Hello',
                        'role': 'user',
                        'gen_ai.message.index': 0,
                        'gen_ai.system': 'test',
                    },
                    {
                        'event.name': 'gen_ai.choice',
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'tool_calls': [
                                {
                                    'id': IsStr(),
                                    'type': 'function',
                                    'function': {'name': 'final_result', 'arguments': {'content': 'a'}},
                                }
                            ],
                        },
                        'gen_ai.system': 'test',
                    },
                ]
            )
        )
    )


def test_instrument_all():
    model = TestModel()
    agent = Agent()

    def get_model():
        return agent._get_model(model)  # type: ignore

    Agent.instrument_all(False)
    assert get_model() is model

    Agent.instrument_all()
    m = get_model()
    assert isinstance(m, InstrumentedModel)
    assert m.wrapped is model
    assert m.instrumentation_settings.event_mode == InstrumentationSettings().event_mode

    options = InstrumentationSettings(event_mode='logs')
    Agent.instrument_all(options)
    m = get_model()
    assert isinstance(m, InstrumentedModel)
    assert m.wrapped is model
    assert m.instrumentation_settings is options

    Agent.instrument_all(False)
    assert get_model() is model


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_feedback(capfire: CaptureLogfire) -> None:
    try:
        from logfire.experimental.annotations import record_feedback
    except ImportError:  # pragma: lax no cover
        pytest.skip('Requires recent version of logfire')

    my_agent = Agent(model=TestModel(), instrument=True)

    async with my_agent.iter('Hello') as agent_run:
        async for _ in agent_run:
            pass
        result = agent_run.result
        assert result
        traceparent = get_traceparent(result)
        assert traceparent == get_traceparent(agent_run)
    assert traceparent == snapshot('00-00000000000000000000000000000001-0000000000000001-01')
    record_feedback(traceparent, 'factuality', 0.1, comment='the agent lied', extra={'foo': 'bar'})

    assert capfire.exporter.exported_spans_as_dict() == snapshot(
        [
            {
                'name': 'chat test',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.system': 'test',
                    'gen_ai.request.model': 'test',
                    'model_request_parameters': '{"function_tools": [], "output_mode": "text", "output_object": null, "output_tools": [], "allow_text_output": true}',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat test',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 4,
                    'gen_ai.response.model': 'test',
                    'events': '[{"content": "Hello", "role": "user", "gen_ai.system": "test", "gen_ai.message.index": 0, "event.name": "gen_ai.user.message"}, {"index": 0, "message": {"role": "assistant", "content": "success (no tool calls)"}, "gen_ai.system": "test", "event.name": "gen_ai.choice"}]',
                    'logfire.json_schema': '{"type": "object", "properties": {"events": {"type": "array"}, "model_request_parameters": {"type": "object"}}}',
                },
            },
            {
                'name': 'agent run',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'test',
                    'agent_name': 'agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'gen_ai.usage.input_tokens': 51,
                    'all_messages_events': '[{"content": "Hello", "role": "user", "gen_ai.message.index": 0, "event.name": "gen_ai.user.message"}, {"role": "assistant", "content": "success (no tool calls)", "gen_ai.message.index": 1, "event.name": "gen_ai.assistant.message"}]',
                    'gen_ai.usage.output_tokens': 4,
                    'final_result': 'success (no tool calls)',
                    'logfire.json_schema': '{"type": "object", "properties": {"all_messages_events": {"type": "array"}, "final_result": {"type": "object"}}}',
                },
            },
            {
                'name': 'feedback: factuality',
                'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': True},
                'start_time': 5000000000,
                'end_time': 5000000000,
                'attributes': {
                    'logfire.span_type': 'annotation',
                    'logfire.level_num': 9,
                    'logfire.msg_template': 'feedback: factuality',
                    'logfire.msg': 'feedback: factuality = 0.1',
                    'code.filepath': 'test_logfire.py',
                    'code.function': 'test_feedback',
                    'code.lineno': 123,
                    'logfire.feedback.name': 'factuality',
                    'factuality': 0.1,
                    'foo': 'bar',
                    'logfire.feedback.comment': 'the agent lied',
                    'logfire.json_schema': '{"type":"object","properties":{"logfire.feedback.name":{},"factuality":{},"foo":{},"logfire.feedback.comment":{},"logfire.span_type":{}}}',
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_include_tool_args_span_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    """Test that tool arguments are included/excluded in span attributes based on instrumentation settings."""

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    test_model = TestModel(seed=42)
    my_agent = Agent(model=test_model, instrument=instrumentation_settings)

    @my_agent.tool_plain
    async def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    result = my_agent.run_sync('Add 42 and 42')
    assert result.output == snapshot('{"add_numbers":84}')

    summary = get_logfire_summary()

    [tool_attributes] = [
        attributes for attributes in summary.attributes.values() if attributes.get('gen_ai.tool.name') == 'add_numbers'
    ]

    if include_content:
        assert tool_attributes == snapshot(
            {
                'gen_ai.tool.name': 'add_numbers',
                'gen_ai.tool.call.id': IsStr(),
                'tool_arguments': '{"x":42,"y":42}',
                'tool_response': '84',
                'logfire.msg': 'running tool: add_numbers',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'tool_arguments': {'type': 'object'},
                                'tool_response': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'logfire.span_type': 'span',
            }
        )
    else:
        assert tool_attributes == snapshot(
            {
                'gen_ai.tool.name': 'add_numbers',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running tool: add_numbers',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'logfire.span_type': 'span',
            }
        )
