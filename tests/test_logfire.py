from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
from dirty_equals import IsInt, IsJson
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
    my_agent = Agent(model=TestModel())

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
                'message': 'my_agent run prompt=Hello',
                'children': [
                    {'id': 1, 'message': 'preparing model request params run_step=1'},
                    {'id': 2, 'message': 'chat test'},
                    {
                        'id': 3,
                        'message': 'handle model response -> tool-return',
                        'children': [{'id': 4, 'message': "running tools=['my_ret']"}],
                    },
                    {'id': 5, 'message': 'preparing model request params run_step=2'},
                    {'id': 6, 'message': 'chat test'},
                    {'id': 7, 'message': 'handle model response -> final result'},
                ],
            }
        ]
    )
    assert summary.attributes[0] == snapshot(
        {
            'code.filepath': 'test_logfire.py',
            'code.function': 'test_logfire',
            'code.lineno': 123,
            'prompt': 'Hello',
            'agent': IsJson(
                {
                    'model': {
                        'call_tools': 'all',
                        'custom_result_text': None,
                        'custom_result_args': None,
                        'seed': 0,
                        'last_model_request_parameters': None,
                    },
                    'name': 'my_agent',
                    'end_strategy': 'early',
                    'model_settings': None,
                }
            ),
            'model_name': 'test',
            'agent_name': 'my_agent',
            'logfire.msg_template': '{agent_name} run {prompt=}',
            'logfire.msg': 'my_agent run prompt=Hello',
            'logfire.span_type': 'span',
            'all_messages_events': IsJson(
                snapshot(
                    [
                        {
                            'content': 'Hello',
                            'role': 'user',
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
                            'event.name': 'gen_ai.assistant.message',
                        },
                        {
                            'content': '1',
                            'role': 'tool',
                            'id': None,
                            'event.name': 'gen_ai.tool.message',
                        },
                        {
                            'role': 'assistant',
                            'content': '{"my_ret":"1"}',
                            'event.name': 'gen_ai.assistant.message',
                        },
                    ]
                )
            ),
            'usage': IsJson(
                {'requests': 2, 'request_tokens': 103, 'response_tokens': 12, 'total_tokens': 115, 'details': None}
            ),
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'prompt': {},
                            'agent': {
                                'type': 'object',
                                'title': 'Agent',
                                'x-python-datatype': 'dataclass',
                                'properties': {
                                    'model': {'type': 'object', 'title': 'TestModel', 'x-python-datatype': 'dataclass'}
                                },
                            },
                            'model_name': {},
                            'agent_name': {},
                            'usage': {'type': 'object', 'title': 'Usage', 'x-python-datatype': 'dataclass'},
                            'all_messages_events': {'type': 'array'},
                        },
                    }
                )
            ),
        }
    )
    assert summary.attributes[1] == snapshot(
        {
            'code.filepath': 'test_logfire.py',
            'code.function': 'test_logfire',
            'code.lineno': IsInt(),
            'run_step': 1,
            'logfire.msg_template': 'preparing model request params {run_step=}',
            'logfire.span_type': 'span',
            'logfire.msg': 'preparing model request params run_step=1',
            'logfire.json_schema': '{"type":"object","properties":{"run_step":{}}}',
        }
    )
