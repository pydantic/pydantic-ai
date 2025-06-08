from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
from dirty_equals import IsJson
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
def test_logfire(get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings | bool) -> None:
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
                                'description': '',
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
                        'allow_text_output': True,
                        'output_tools': [],
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
    assert m.settings.event_mode == InstrumentationSettings().event_mode

    options = InstrumentationSettings(event_mode='logs')
    Agent.instrument_all(options)
    m = get_model()
    assert isinstance(m, InstrumentedModel)
    assert m.wrapped is model
    assert m.settings is options

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
                    'model_request_parameters': '{"function_tools": [], "allow_text_output": true, "output_tools": []}',
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
                    'gen_ai.usage.output_tokens': 4,
                    'all_messages_events': '[{"content": "Hello", "role": "user", "gen_ai.message.index": 0, "event.name": "gen_ai.user.message"}, {"role": "assistant", "content": "success (no tool calls)", "gen_ai.message.index": 1, "event.name": "gen_ai.assistant.message"}]',
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
                    'logfire.disable_console_log': True,
                    'logfire.json_schema': '{"type":"object","properties":{"logfire.feedback.name":{},"factuality":{},"foo":{},"logfire.feedback.comment":{},"logfire.span_type":{},"logfire.disable_console_log":{}}}',
                },
            },
        ]
    )
