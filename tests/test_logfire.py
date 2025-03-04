from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

try:
    from logfire.testing import CaptureLogfire
except ImportError:
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
def test_logfire(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    my_agent = Agent(model=TestModel(), instrument=True)

    @my_agent.tool_plain
    async def my_ret(x: int) -> str:
        return str(x + 1)

    result = my_agent.run_sync('Hello')
    assert result.data == snapshot('{"my_ret":"1"}')

    summary = get_logfire_summary()
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'message': 'my_agent run',
                'children': [
                    {'id': 1, 'message': 'preparing model request params'},
                    {'id': 2, 'message': 'chat test'},
                    {'id': 3, 'message': 'running tools: my_ret'},
                    {'id': 4, 'message': 'preparing model request params'},
                    {'id': 5, 'message': 'chat test'},
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
                                    'id': None,
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
                            'id': None,
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
    assert summary.attributes[1] == snapshot(
        {
            'run_step': 1,
            'logfire.span_type': 'span',
            'logfire.msg': 'preparing model request params',
        }
    )
    assert summary.attributes[2] == snapshot(
        {
            'gen_ai.operation.name': 'chat',
            'gen_ai.system': 'test',
            'gen_ai.request.model': 'test',
            'logfire.span_type': 'span',
            'logfire.msg': 'chat test',
            'gen_ai.response.model': 'test',
            'gen_ai.usage.input_tokens': 51,
            'gen_ai.usage.output_tokens': 4,
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
                                        'id': None,
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
            'logfire.json_schema': '{"type": "object", "properties": {"events": {"type": "array"}}}',
        }
    )
