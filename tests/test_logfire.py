from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Any, Callable

import pytest
from dirty_equals import IsInt, IsJson, IsNow, IsStr
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
    agent = Agent(model=TestModel())

    @agent.retriever_plain
    async def my_ret(x: int) -> str:
        return str(x + 1)

    result = agent.run_sync('Hello')
    assert result.data == snapshot('{"my_ret":"1"}')

    summary = get_logfire_summary()
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'message': 'agent run prompt=Hello',
                'children': [
                    {'id': 1, 'message': 'model request -> model-structured-response'},
                    {
                        'id': 2,
                        'message': 'handle model response -> tool-return',
                        'children': [{'id': 3, 'message': "running tools=['my_ret']"}],
                    },
                    {'id': 4, 'message': 'model request -> model-text-response'},
                    {'id': 5, 'message': 'handle model response -> final result'},
                ],
            }
        ]
    )
    assert summary.attributes[1] == snapshot(
        {
            'code.filepath': 'agent.py',
            'code.function': 'run',
            'code.lineno': IsInt(),
            'run_step': 1,
            'logfire.msg_template': 'model request {run_step=}',
            'logfire.span_type': 'span',
            'response': IsJson(
                {
                    'calls': [{'tool_name': 'my_ret', 'args': {'args_object': {'x': 0}}, 'tool_id': None}],
                    'timestamp': IsStr() & IsNow(iso_string=True, tz=timezone.utc),
                    'role': 'model-structured-response',
                }
            ),
            'cost': IsJson({'request_tokens': None, 'response_tokens': None, 'total_tokens': None, 'details': None}),
            'logfire.msg': 'model request -> model-structured-response',
            'logfire.json_schema': IsJson(
                {
                    'type': 'object',
                    'properties': {
                        'run_step': {},
                        'response': {
                            'type': 'object',
                            'title': 'ModelStructuredResponse',
                            'x-python-datatype': 'dataclass',
                            'properties': {
                                'calls': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'title': 'ToolCall',
                                        'x-python-datatype': 'dataclass',
                                        'properties': {
                                            'args': {
                                                'type': 'object',
                                                'title': 'ArgsObject',
                                                'x-python-datatype': 'dataclass',
                                            }
                                        },
                                    },
                                },
                                'timestamp': {'type': 'string', 'format': 'date-time'},
                            },
                        },
                        'cost': {'type': 'object', 'title': 'Cost', 'x-python-datatype': 'dataclass'},
                    },
                }
            ),
        }
    )
