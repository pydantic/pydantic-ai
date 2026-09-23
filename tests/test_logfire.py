from __future__ import annotations as _annotations

import asyncio
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from dirty_equals import IsJson, IsList

# `StatusCode` lives in `opentelemetry-api`, a core dependency, so it needs no guard.
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import NotRequired, Self, TypedDict

from pydantic_ai import (
    Agent,
    MessageHistoryMutatedWarning,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai._utils import get_traceparent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelHTTPError,
    ModelRetry,
    ToolFailed,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import PromptedOutput, TextOutput
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsStr, strip_logfire_metrics, try_import

with try_import():
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the
    # `pydantic-ai-slim` / `pydantic-evals` install groups either.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

try:
    import logfire
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    logfire_installed = False
else:
    logfire_installed = True


class SpanSummary(TypedDict):
    id: int
    name: str
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
            span_lookup[tid] = span_summary = SpanSummary(
                id=id_counter, name=span['name'], message=span['attributes']['logfire.msg']
            )
            # `logfire.metrics` is a logfire-version-dependent span decoration (added in 4.3x): newer
            # logfire attaches the aggregated `gen_ai.client.token.usage` metric to spans, older does not.
            # Strip it so these assertions hold across the supported logfire range; the token usage itself
            # is still covered by the stable `gen_ai.usage.*` attributes and `get_collected_metrics()`.
            self.attributes[id_counter] = {k: v for k, v in span['attributes'].items() if k != 'logfire.metrics'}
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


def deprecated_instrumentation_settings(version: Literal[2, 3, 4], **kwargs: Any) -> InstrumentationSettings:
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'Instrumentation format versions 2, 3, and 4 are deprecated',
    ):
        return InstrumentationSettings(version=version, **kwargs)


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [
        True,
        False,
        deprecated_instrumentation_settings(version=2),
        deprecated_instrumentation_settings(version=3),
    ],
)
def test_logfire(
    get_logfire_summary: Callable[[], LogfireSummary],
    instrument: InstrumentationSettings | bool,
    capfire: CaptureLogfire,
) -> None:
    class InstrumentedToolset(WrapperToolset):
        async def __aenter__(self) -> Self:
            with logfire.span('toolset_enter'):  # pyright: ignore[reportPossiblyUnboundVariable]
                await super().__aenter__()
                return self

        async def __aexit__(self, *args: Any) -> bool | None:
            with logfire.span('toolset_exit'):  # pyright: ignore[reportPossiblyUnboundVariable]
                return await super().__aexit__(*args)

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
        ) -> Any:
            with logfire.span('toolset_call_tool {name}', name=name):  # pyright: ignore[reportPossiblyUnboundVariable]
                return await super().call_tool(name, tool_args, ctx, tool)

    toolset = FunctionToolset()

    @toolset.tool_plain
    async def my_ret(x: int) -> str:
        return str(x + 1)

    if instrument:
        toolset = InstrumentedToolset(toolset)

    capabilities = (
        [
            Instrumentation(
                settings=instrument if isinstance(instrument, InstrumentationSettings) else InstrumentationSettings()
            )
        ]
        if instrument
        else []
    )
    my_agent = Agent(
        model=TestModel(),
        toolsets=[toolset],
        capabilities=capabilities,
        metadata={'env': 'test'},
    )

    result = my_agent.run_sync('Hello')
    assert result.output == snapshot('{"my_ret":"1"}')

    summary = get_logfire_summary()
    if instrument is False:
        assert summary.traces == []
        return

    if instrument is True or (isinstance(instrument, InstrumentationSettings) and instrument.version >= 3):
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'toolset_enter', 'message': 'toolset_enter'},
                        {'id': 2, 'name': 'chat test', 'message': 'chat test'},
                        {
                            'id': 3,
                            'name': 'execute_tool my_ret',
                            'message': 'running tool: my_ret',
                            'children': [
                                {'id': 4, 'name': 'toolset_call_tool {name}', 'message': 'toolset_call_tool my_ret'}
                            ],
                        },
                        {'id': 5, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 6, 'name': 'toolset_exit', 'message': 'toolset_exit'},
                    ],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'toolset_enter', 'message': 'toolset_enter'},
                        {'id': 2, 'name': 'chat test', 'message': 'chat test'},
                        {
                            'id': 3,
                            'name': 'running tool',
                            'message': 'running tool: my_ret',
                            'children': [
                                {'id': 4, 'name': 'toolset_call_tool {name}', 'message': 'toolset_call_tool my_ret'}
                            ],
                        },
                        {'id': 5, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 6, 'name': 'toolset_exit', 'message': 'toolset_exit'},
                    ],
                }
            ]
        )

    if isinstance(instrument, InstrumentationSettings) and instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'toolset_enter', 'message': 'toolset_enter'},
                        {'id': 2, 'name': 'chat test', 'message': 'chat test'},
                        {
                            'id': 3,
                            'name': 'running tool',
                            'message': 'running tool: my_ret',
                            'children': [
                                {'id': 4, 'name': 'toolset_call_tool {name}', 'message': 'toolset_call_tool my_ret'}
                            ],
                        },
                        {'id': 5, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 6, 'name': 'toolset_exit', 'message': 'toolset_exit'},
                    ],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'toolset_enter', 'message': 'toolset_enter'},
                        {'id': 2, 'name': 'chat test', 'message': 'chat test'},
                        {
                            'id': 3,
                            'name': 'execute_tool my_ret',
                            'message': 'running tool: my_ret',
                            'children': [
                                {'id': 4, 'name': 'toolset_call_tool {name}', 'message': 'toolset_call_tool my_ret'}
                            ],
                        },
                        {'id': 5, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 6, 'name': 'toolset_exit', 'message': 'toolset_exit'},
                    ],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': '{"my_ret":"1"}',
            'gen_ai.aggregated_usage.input_tokens': 103,
            'gen_ai.aggregated_usage.output_tokens': 12,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'my_ret',
                                    'arguments': {'x': 0},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'my_ret',
                                    'result': '1',
                                }
                            ],
                        },
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': '{"my_ret":"1"}'}]},
                    ]
                )
            ),
            'metadata': '{"env":"test"}',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'metadata': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )
    chat_span_attributes = next(
        attrs for attrs in summary.attributes.values() if attrs.get('gen_ai.operation.name', None) == 'chat'
    )
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
                                    'gen_ai.provider.name': 'test',
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
                                    'bucket_counts': IsList(length=...),
                                },
                                'negative': {'offset': 0, 'bucket_counts': [0]},
                                'flags': 0,
                                'min': 51,
                                'max': 52,
                                'exemplars': IsList(length=...),
                            },
                            {
                                'attributes': {
                                    'gen_ai.provider.name': 'test',
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
                                    'bucket_counts': IsList(length=...),
                                },
                                'negative': {'offset': 0, 'bucket_counts': [0]},
                                'flags': 0,
                                'min': 4,
                                'max': 8,
                                'exemplars': IsList(length=...),
                            },
                        ],
                        'aggregation_temporality': 1,
                    },
                }
            ]
        )

    messages_attributes = {
        k: chat_span_attributes.pop(k)
        for k in ['events', 'gen_ai.input.messages', 'gen_ai.output.messages']
        if k in chat_span_attributes
    }
    assert messages_attributes == snapshot(
        {
            'gen_ai.input.messages': IsJson(
                snapshot([{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}])
            ),
            'gen_ai.output.messages': IsJson(
                snapshot(
                    [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'my_ret',
                                    'arguments': {'x': 0},
                                }
                            ],
                        }
                    ]
                )
            ),
        }
    )

    assert chat_span_attributes == snapshot(
        {
            'gen_ai.operation.name': 'chat',
            'gen_ai.provider.name': 'test',
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
                                'sequential': False,
                                'kind': 'function',
                                'metadata': None,
                                'timeout': None,
                                'defer_loading': False,
                                'toolset_id': None,
                                'unless_native': None,
                                'with_native': None,
                                'tool_kind': None,
                                'return_schema': None,
                                'include_return_schema': None,
                                'capability_id': None,
                            }
                        ],
                        'native_tools': [],
                        'tool_visibility': {'my_ret': 'visible'},
                        'revealed_tool_names': [],
                        'deferred_capability_ids': [],
                        'output_mode': 'text',
                        'output_tools': [],
                        'output_object': None,
                        'prompted_output_template': None,
                        'allow_text_output': True,
                        'allow_image_output': False,
                        'instruction_parts': None,
                        'thinking': None,
                    }
                )
            ),
            'gen_ai.tool.definitions': '[{"type":"function","name":"my_ret","parameters":{"additionalProperties":false,"properties":{"x":{"type":"integer"}},"required":["x"],"type":"object"}}]',
            'logfire.json_schema': IsJson(),
            'gen_ai.conversation.id': IsStr(),
            'logfire.span_type': 'span',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'logfire.msg': 'chat test',
            'gen_ai.response.model': 'test',
            'gen_ai.usage.input_tokens': 51,
            'gen_ai.usage.output_tokens': 4,
        }
    )


def _test_logfire_metadata_values_callable_dict(ctx: RunContext[Any]) -> dict[str, str]:
    return {'model_name': ctx.model.model_name}


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_logfire_explicit_run_id(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    """An explicit `run_id=` is emitted as `gen_ai.agent.call.id` on the agent run span."""
    agent = Agent(
        model=TestModel(custom_output_text='ok'),
        name='run_id_agent',
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
    )
    result = agent.run_sync('Hello', run_id='run-from-api-42')
    assert result.run_id == 'run-from-api-42'

    summary = get_logfire_summary()
    agent_run_attrs = next(
        attrs for attrs in summary.attributes.values() if attrs.get('gen_ai.operation.name') == 'invoke_agent'
    )
    assert agent_run_attrs['gen_ai.agent.call.id'] == 'run-from-api-42'


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        pytest.param({'env': 'test'}, '{"env":"test"}', id='dict'),
        pytest.param(_test_logfire_metadata_values_callable_dict, '{"model_name":"test"}', id='callable-dict'),
    ],
)
def test_logfire_metadata_values(
    get_logfire_summary: Callable[[], LogfireSummary],
    metadata: dict[str, Any] | Callable[[RunContext[Any]], dict[str, Any]],
    expected: dict[str, Any],
) -> None:
    agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=deprecated_instrumentation_settings(version=2))],
        metadata=metadata,
    )
    agent.run_sync('Hello')

    summary = get_logfire_summary()
    assert summary.attributes[0]['metadata'] == expected


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_logfire_metadata_override(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=deprecated_instrumentation_settings(version=2))],
        metadata={'env': 'base'},
    )
    with agent.override(metadata={'env': 'override'}):
        agent.run_sync('Hello')

    summary = get_logfire_summary()
    assert summary.attributes[0]['metadata'] == '{"env":"override"}'


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_logfire_streaming_records_time_to_first_chunk(capfire: CaptureLogfire) -> None:
    """A streaming agent run records `gen_ai.client.operation.time_to_first_chunk` on the
    model-request span and as a histogram metric (value is non-deterministic, so assert shape)."""
    agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
    )
    async with agent.run_stream('Hello') as result:
        async for _ in result.stream_text(delta=True):
            pass

    chat_spans = [
        s for s in capfire.exporter.exported_spans_as_dict() if s['attributes'].get('gen_ai.operation.name') == 'chat'
    ]
    assert chat_spans
    for span in chat_spans:
        ttft = span['attributes'].get('gen_ai.client.operation.time_to_first_chunk')
        assert isinstance(ttft, float)

    # Pin the histogram emission through the agent-flow path (capability handler -> req_ctx ->
    # finish), not just the metric name, so a regression that drops the value between the handler
    # and `finish` can't slip through.
    ttft_metrics = [
        m for m in capfire.get_collected_metrics() if m['name'] == 'gen_ai.client.operation.time_to_first_chunk'
    ]
    assert len(ttft_metrics) == 1
    assert ttft_metrics[0]['unit'] == 's'
    assert len(ttft_metrics[0]['data']['data_points']) == 1


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_instructions_with_structured_output(
    get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(
        model=TestModel(),
        instructions='Here are some instructions',
        capabilities=[Instrumentation(settings=instrument)],
    )

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')

    summary = get_logfire_summary()
    chat_span_attributes = summary.attributes[1]
    if instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': '{"content":"a"}',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 5,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'arguments': {'content': 'a'},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'result': 'Final result processed.',
                                }
                            ],
                        },
                    ]
                )
            ),
            'gen_ai.system_instructions': '[{"type":"text","content":"Here are some instructions"}]',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )

    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot([{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}])
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call',
                            'id': IsStr(),
                            'name': 'final_result',
                            'arguments': {'content': 'a'},
                        }
                    ],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instructions_with_structured_output_exclude_content(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    @dataclass
    class MyOutput:
        content: str

    settings: InstrumentationSettings = InstrumentationSettings(include_content=False)

    my_agent = Agent(
        model=TestModel(), instructions='Here are some instructions', capabilities=[Instrumentation(settings=settings)]
    )

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == snapshot(MyOutput(content='a'))

    summary = get_logfire_summary()
    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 5,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                }
                            ],
                        },
                    ]
                )
            ),
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )
    chat_span_attributes = summary.attributes[1]
    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot(
            [
                {'role': 'user', 'parts': [{'type': 'text'}]},
            ]
        )
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call',
                            'id': IsStr(),
                            'name': 'final_result',
                        }
                    ],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'settings',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_prompted_output_schema_instructions_do_not_set_variable_instructions(
    get_logfire_summary: Callable[[], LogfireSummary],
    settings: InstrumentationSettings,
) -> None:
    class City(BaseModel):
        name: str
        population: int

    my_agent = Agent(
        model=TestModel(custom_output_text='{"name":"Paris","population":2148000}'),
        instructions='Be helpful',
        capabilities=[Instrumentation(settings=settings)],
        output_type=PromptedOutput(City, template='Return JSON matching this schema:\n{schema}'),
    )

    result = my_agent.run_sync('Tell me about Paris')
    assert result.output == City(name='Paris', population=2148000)

    summary = get_logfire_summary()
    agent_run_attrs = summary.attributes[0]
    assert 'Be helpful' in agent_run_attrs['gen_ai.system_instructions']
    assert 'pydantic_ai.variable_instructions' not in agent_run_attrs


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'settings',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_stable_instructions_across_tool_calls_do_not_set_variable_instructions(
    get_logfire_summary: Callable[[], LogfireSummary],
    settings: InstrumentationSettings,
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(
        model=TestModel(),
        instructions='Be helpful',
        capabilities=[Instrumentation(settings=settings)],
    )
    instruction_calls = 0

    @my_agent.tool_plain
    def my_tool() -> str:
        nonlocal instruction_calls
        instruction_calls += 1
        return 'tool result'

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')
    # Ensure multi-step execution occurred so instructions were compared across requests
    assert instruction_calls >= 1

    summary = get_logfire_summary()
    assert 'pydantic_ai.variable_instructions' not in summary.attributes[0]


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'settings',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_dynamic_instructions_toggling_from_none_sets_variable_instructions(
    get_logfire_summary: Callable[[], LogfireSummary],
    settings: InstrumentationSettings,
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=settings)],
    )
    instruction_calls = 0

    @my_agent.instructions
    def instructions(_: RunContext[object]) -> str | None:
        nonlocal instruction_calls
        instruction_calls += 1
        if instruction_calls == 1:
            return None
        return 'This is a later step'

    @my_agent.tool_plain
    def my_tool() -> str:
        return 'This is a tool call'

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')
    # Ensure multi-step execution occurred so instructions actually toggled
    assert instruction_calls >= 2

    summary = get_logfire_summary()
    assert summary.attributes[0]['pydantic_ai.variable_instructions'] is True


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('version', [2, 3])
def test_instructions_with_structured_output_exclude_content_v2_v3(
    get_logfire_summary: Callable[[], LogfireSummary],
    version: Literal[2, 3],
) -> None:
    @dataclass
    class MyOutput:
        content: str

    settings: InstrumentationSettings = deprecated_instrumentation_settings(include_content=False, version=version)

    my_agent = Agent(
        model=TestModel(), instructions='Here are some instructions', capabilities=[Instrumentation(settings=settings)]
    )

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')

    summary = get_logfire_summary()

    if version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )

    # Version 2 and 3 have identical snapshots for this test case
    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 5,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                }
                            ],
                        },
                    ]
                )
            ),
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )
    chat_span_attributes = summary.attributes[1]
    assert chat_span_attributes == snapshot(
        {
            'gen_ai.operation.name': 'chat',
            'gen_ai.provider.name': 'test',
            'gen_ai.system': 'test',
            'gen_ai.request.model': 'test',
            'model_request_parameters': IsJson(
                snapshot(
                    {
                        'function_tools': [],
                        'native_tools': [],
                        'tool_visibility': {},
                        'revealed_tool_names': [],
                        'deferred_capability_ids': [],
                        'output_mode': 'tool',
                        'output_object': None,
                        'output_tools': [
                            {
                                'name': 'final_result',
                                'parameters_json_schema': {
                                    'properties': {'content': {'type': 'string'}},
                                    'required': ['content'],
                                    'title': 'MyOutput',
                                    'type': 'object',
                                },
                                'description': 'The final response which ends this conversation',
                                'outer_typed_dict_key': None,
                                'strict': None,
                                'sequential': False,
                                'kind': 'output',
                                'metadata': None,
                                'timeout': None,
                                'defer_loading': False,
                                'toolset_id': '<output>',
                                'unless_native': None,
                                'with_native': None,
                                'tool_kind': None,
                                'return_schema': None,
                                'include_return_schema': None,
                                'capability_id': None,
                            }
                        ],
                        'prompted_output_template': None,
                        'allow_text_output': False,
                        'allow_image_output': False,
                        'instruction_parts': [
                            {
                                'dynamic': False,
                                'name': None,
                                'id': 'agent',
                                'part_kind': 'instruction',
                            }
                        ],
                        'thinking': None,
                    }
                )
            ),
            'gen_ai.tool.definitions': '[{"type":"function","name":"final_result","description":"The final response which ends this conversation","parameters":{"properties":{"content":{"type":"string"}},"required":["content"],"title":"MyOutput","type":"object"}}]',
            'logfire.span_type': 'span',
            'gen_ai.conversation.id': IsStr(),
            'logfire.msg': 'chat test',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.input.messages': IsJson(snapshot([{'role': 'user', 'parts': [{'type': 'text'}]}])),
            'gen_ai.output.messages': IsJson(
                snapshot(
                    [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                }
                            ],
                        }
                    ]
                )
            ),
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'gen_ai.input.messages': {'type': 'array'},
                            'gen_ai.output.messages': {'type': 'array'},
                            'model_request_parameters': {'type': 'object'},
                        },
                    }
                )
            ),
            'gen_ai.usage.input_tokens': 51,
            'gen_ai.usage.output_tokens': 5,
            'gen_ai.response.model': 'test',
        }
    )


def test_instrument_all():
    agent = Agent()

    def resolve():
        return agent._resolve_instrumentation_settings()  # type: ignore[reportPrivateUsage]

    Agent.instrument_all(False)
    assert resolve() is None

    Agent.instrument_all()
    settings = resolve()
    assert settings is not None
    assert settings.version == InstrumentationSettings().version

    options = InstrumentationSettings(version=5)
    Agent.instrument_all(options)
    assert resolve() is options

    Agent.instrument_all(False)
    assert resolve() is None


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_aggregated_usage_attribute_names_default(capfire: CaptureLogfire) -> None:
    """Agent run spans use aggregated usage attribute names by default."""

    def model_function(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        # Return a response with usage that includes extra details (cache tokens)
        # to test that all gen_ai.usage.* attributes are translated
        return ModelResponse(
            parts=[TextPart('Hello!')],
            usage=RequestUsage(input_tokens=10, output_tokens=5, cache_read_tokens=2),
        )

    settings = InstrumentationSettings()
    agent = Agent(model=FunctionModel(model_function), capabilities=[Instrumentation(settings=settings)])

    await agent.run('Hello')

    spans = strip_logfire_metrics(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True))

    # Verify that agent run span uses aggregated_usage attribute names
    agent_run_span = next(s for s in spans if s['name'] == 'invoke_agent agent')
    assert agent_run_span['attributes'] == snapshot(
        {
            'model_name': 'function:model_function:',
            'agent_name': 'agent',
            'gen_ai.agent.name': 'agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'agent run',
            'logfire.span_type': 'span',
            'final_result': 'Hello!',
            'gen_ai.aggregated_usage.input_tokens': 10,
            'gen_ai.aggregated_usage.output_tokens': 5,
            'gen_ai.aggregated_usage.cache_read.input_tokens': 2,
            'gen_ai.aggregated_usage.details.cache_read_tokens': 2,
            'pydantic_ai.all_messages': [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello!'}]},
            ],
            'logfire.json_schema': {
                'type': 'object',
                'properties': {'pydantic_ai.all_messages': {'type': 'array'}, 'final_result': {'type': 'object'}},
            },
        }
    )

    # Verify that model/chat span still uses standard attribute names
    chat_span = next(s for s in spans if 'chat' in s['name'])
    assert chat_span['attributes']['gen_ai.usage.input_tokens'] == 10
    assert chat_span['attributes']['gen_ai.usage.output_tokens'] == 5


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_aggregated_usage_attribute_names_can_be_disabled(capfire: CaptureLogfire) -> None:
    def model_function(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')], usage=RequestUsage(input_tokens=10, output_tokens=5))

    settings = InstrumentationSettings(use_aggregated_usage_attribute_names=False)
    agent = Agent(model=FunctionModel(model_function), capabilities=[Instrumentation(settings=settings)])

    await agent.run('Hello')

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    agent_run_span = next(s for s in spans if s['name'] == 'invoke_agent agent')
    assert agent_run_span['attributes']['gen_ai.usage.input_tokens'] == 10
    assert agent_run_span['attributes']['gen_ai.usage.output_tokens'] == 5
    assert 'gen_ai.aggregated_usage.input_tokens' not in agent_run_span['attributes']


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_in_place_history_mutation_warns_and_leaves_stale_request_spans(capfire: CaptureLogfire) -> None:
    """Mutating a message already in the history in place mid-run is unsupported.

    The per-run fragment cache serves the pre-mutation bytes, so the second request span records the
    original prompt even though the mutated one was sent to the model, while the run-level
    `pydantic_ai.all_messages` (always serialized fresh) reflects the mutation — and the run warns at
    its end. This pins the documented caveat: if the cache ever becomes mutation-proof, update the
    message-history docs along with these snapshots.
    """

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('corrupt_history', {}, tool_call_id='call_1')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        model=FunctionModel(model_function), capabilities=[Instrumentation(settings=InstrumentationSettings())]
    )

    @agent.tool
    async def corrupt_history(ctx: RunContext) -> str:
        first_part = ctx.messages[0].parts[0]
        assert isinstance(first_part, UserPromptPart)
        first_part.content = 'mutated prompt'
        return 'ok'

    with pytest.warns(MessageHistoryMutatedWarning, match='In-place mutation of messages'):
        result = await agent.run('original prompt')
    assert result.output == 'done'

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    chat_inputs = [s['attributes']['gen_ai.input.messages'] for s in spans if s['name'].startswith('chat ')]
    assert chat_inputs == snapshot(
        [
            [{'role': 'user', 'parts': [{'type': 'text', 'content': 'original prompt'}]}],
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'original prompt'}]},
                {
                    'role': 'assistant',
                    'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'corrupt_history', 'arguments': {}}],
                },
                {
                    'role': 'user',
                    'parts': [
                        {'type': 'tool_call_response', 'id': 'call_1', 'name': 'corrupt_history', 'result': 'ok'}
                    ],
                },
            ],
        ]
    )
    agent_run_span = next(s for s in spans if s['name'] == 'invoke_agent agent')
    assert agent_run_span['attributes']['pydantic_ai.all_messages'] == snapshot(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'mutated prompt'}]},
            {
                'role': 'assistant',
                'parts': [{'type': 'tool_call', 'id': 'call_1', 'name': 'corrupt_history', 'arguments': {}}],
            },
            {
                'role': 'user',
                'parts': [{'type': 'tool_call_response', 'id': 'call_1', 'name': 'corrupt_history', 'result': 'ok'}],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'done'}]},
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_history_mutation_in_errored_run_does_not_displace_the_run_error(capfire: CaptureLogfire) -> None:
    """A run that errors after an in-place history mutation surfaces the run's own exception.

    The run-end staleness check is skipped on the error path: this suite configures warnings as
    errors, so a `MessageHistoryMutatedWarning` raised in the run's `finally` would otherwise
    displace the propagating exception.
    """

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('corrupt_history', {}, tool_call_id='call_1')])
        raise RuntimeError('model failure')

    agent = Agent(
        model=FunctionModel(model_function), capabilities=[Instrumentation(settings=InstrumentationSettings())]
    )

    @agent.tool
    async def corrupt_history(ctx: RunContext) -> str:
        first_part = ctx.messages[0].parts[0]
        assert isinstance(first_part, UserPromptPart)
        first_part.content = 'mutated prompt'
        return 'ok'

    with pytest.raises(RuntimeError, match='model failure'):
        await agent.run('original prompt')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_feedback(capfire: CaptureLogfire) -> None:
    from logfire.experimental.annotations import record_feedback

    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=InstrumentationSettings())])

    async with my_agent.iter('Hello') as agent_run:
        async for _ in agent_run:
            pass
        result = agent_run.result
        assert result
        traceparent = get_traceparent(result)
        assert traceparent == get_traceparent(agent_run)
    assert traceparent == snapshot('00-00000000000000000000000000000001-0000000000000001-01')
    record_feedback(traceparent, 'factuality', 0.1, comment='the agent lied', extra={'foo': 'bar'})

    assert strip_logfire_metrics(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)) == snapshot(
        [
            {
                'name': 'chat test',
                'context': {'trace_id': 1, 'span_id': 3, 'is_remote': False},
                'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'start_time': 2000000000,
                'end_time': 3000000000,
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'test',
                    'gen_ai.system': 'test',
                    'gen_ai.request.model': 'test',
                    'model_request_parameters': {
                        'function_tools': [],
                        'native_tools': [],
                        'tool_visibility': {},
                        'revealed_tool_names': [],
                        'deferred_capability_ids': [],
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
                    'logfire.msg': 'chat test',
                    'gen_ai.conversation.id': IsStr(),
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 4,
                    'gen_ai.response.model': 'test',
                    'gen_ai.input.messages': [
                        {
                            'parts': [
                                {
                                    'type': 'text',
                                    'content': 'Hello',
                                },
                            ],
                            'role': 'user',
                        },
                    ],
                    'gen_ai.output.messages': [
                        {
                            'role': 'assistant',
                            'parts': [{'type': 'text', 'content': 'success (no tool calls)'}],
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
                },
            },
            {
                'name': 'invoke_agent agent',
                'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
                'parent': None,
                'start_time': 1000000000,
                'end_time': 4000000000,
                'attributes': {
                    'model_name': 'test',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'final_result': 'success (no tool calls)',
                    'gen_ai.aggregated_usage.input_tokens': 51,
                    'gen_ai.aggregated_usage.output_tokens': 4,
                    'pydantic_ai.all_messages': [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success (no tool calls)'}]},
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
                    'logfire.json_schema': {
                        'type': 'object',
                        'properties': {
                            'logfire.feedback.name': {},
                            'factuality': {},
                            'foo': {},
                            'logfire.feedback.comment': {},
                            'logfire.span_type': {},
                        },
                    },
                },
            },
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content,tool_error', [(True, False), (True, True), (False, False), (False, True)])
def test_include_tool_args_span_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
    tool_error: bool,
) -> None:
    """Test that tool arguments are included/excluded in span attributes based on instrumentation settings."""

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    test_model = TestModel(seed=42)
    my_agent = Agent(model=test_model, capabilities=[Instrumentation(settings=instrumentation_settings)])

    @my_agent.tool_plain
    async def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        if tool_error:
            raise ModelRetry('Tool error')
        return x + y

    try:
        result = my_agent.run_sync('Add 42 and 42')
        assert result.output == snapshot('{"add_numbers":84}')
    except UnexpectedModelBehavior:
        if not tool_error:
            raise  # pragma: no cover

    summary = get_logfire_summary()

    tool_attributes = next(
        attributes for attributes in summary.attributes.values() if attributes.get('gen_ai.tool.name') == 'add_numbers'
    )

    if include_content:
        if tool_error:
            assert tool_attributes == snapshot(
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'add_numbers',
                    'gen_ai.tool.call.id': IsStr(),
                    'gen_ai.tool.call.arguments': '{"x":42,"y":42}',
                    'logfire.msg': 'running tool: add_numbers',
                    'logfire.json_schema': IsJson(
                        snapshot(
                            {
                                'type': 'object',
                                'properties': {
                                    'gen_ai.tool.call.arguments': {'type': 'object'},
                                    'gen_ai.tool.call.result': {'type': 'object'},
                                    'gen_ai.tool.name': {},
                                    'gen_ai.tool.call.id': {},
                                },
                            }
                        )
                    ),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.tool.call.result': """\
Tool error

Fix the errors and try again.\
""",
                    'gen_ai.agent.call.id': IsStr(),
                    'logfire.level_num': 17,
                }
            )
        else:
            assert tool_attributes == snapshot(
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'add_numbers',
                    'gen_ai.tool.call.id': IsStr(),
                    'gen_ai.tool.call.arguments': '{"x":42,"y":42}',
                    'logfire.msg': 'running tool: add_numbers',
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.json_schema': IsJson(
                        snapshot(
                            {
                                'type': 'object',
                                'properties': {
                                    'gen_ai.tool.call.arguments': {'type': 'object'},
                                    'gen_ai.tool.call.result': {'type': 'object'},
                                    'gen_ai.tool.name': {},
                                    'gen_ai.tool.call.id': {},
                                },
                            }
                        )
                    ),
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'logfire.span_type': 'span',
                    'gen_ai.tool.call.result': '84',
                }
            )
    else:
        if tool_error:
            assert tool_attributes == snapshot(
                {
                    'gen_ai.operation.name': 'execute_tool',
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
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'logfire.level_num': 17,
                }
            )
        else:
            assert tool_attributes == snapshot(
                {
                    'gen_ai.operation.name': 'execute_tool',
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
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.span_type': 'span',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                }
            )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('failure_source', ['schema', 'validator'])
def test_tool_argument_validation_failure_emits_span(
    get_logfire_summary: Callable[[], LogfireSummary],
    capfire: CaptureLogfire,
    failure_source: Literal['schema', 'validator'],
    include_content: bool,
) -> None:
    """A rejected tool call emits one error span without duplicating the successful retry."""
    model_calls = 0

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            x: Any = 'private-invalid-value' if failure_source == 'schema' else 2
            return ModelResponse(parts=[ToolCallPart('double', {'x': x})])
        if model_calls == 2:
            return ModelResponse(parts=[ToolCallPart('double', {'x': 2})])
        return ModelResponse(parts=[TextPart('done')])

    def validate_args(ctx: RunContext[Any], x: int) -> None:
        if failure_source == 'validator' and ctx.retry == 0:
            raise ModelRetry(f'reject {x}')

    agent = Agent(
        FunctionModel(call_tool),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=include_content))],
    )
    calls: list[int] = []

    @agent.tool_plain(args_validator=validate_args)
    def double(x: int) -> int:
        calls.append(x)
        return x * 2

    agent.run_sync('Use the tool')

    tool_spans = [
        attributes
        for attributes in get_logfire_summary().attributes.values()
        if attributes.get('gen_ai.operation.name') == 'execute_tool'
    ]
    assert calls == [2]
    assert len(tool_spans) == 2
    assert tool_spans[0]['logfire.level_num'] == 17
    assert tool_spans[0]['logfire.msg'] == 'invalid tool call: double'
    assert tool_spans[0]['pydantic_ai.tool.failure_stage'] == 'validation'
    assert 'logfire.level_num' not in tool_spans[1]
    assert tool_spans[1]['logfire.msg'] == 'running tool: double'
    assert 'pydantic_ai.tool.failure_stage' not in tool_spans[1]
    if include_content:
        expected_error = 'int_parsing' if failure_source == 'schema' else 'reject 2'
        assert expected_error in tool_spans[0]['gen_ai.tool.call.result']
        assert tool_spans[1]['gen_ai.tool.call.result'] == '4'
    else:
        assert all('gen_ai.tool.call.result' not in span for span in tool_spans)
        assert 'private-invalid-value' not in str(capfire.exporter.exported_spans_as_dict())


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('instrumentation_first', [True, False])
def test_recovered_tool_validation_emits_no_error_span(
    get_logfire_summary: Callable[[], LogfireSummary],
    instrumentation_first: bool,
) -> None:
    """Instrumentation observes only errors that earlier capabilities did not recover."""

    @dataclass
    class RecoverValidation(AbstractCapability[Any]):
        async def on_tool_validate_error(self, ctx: RunContext[Any], **kwargs: Any) -> Any:
            return {'x': 2}

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('double', {'x': 'invalid'})])
        return ModelResponse(parts=[TextPart('done')])

    instrumentation = Instrumentation(settings=InstrumentationSettings())
    recovery = RecoverValidation()
    capabilities = [instrumentation, recovery] if instrumentation_first else [recovery, instrumentation]
    agent = Agent(FunctionModel(call_tool), capabilities=capabilities)

    @agent.tool_plain
    def double(x: int) -> int:
        return x * 2

    agent.run_sync('Use the tool')

    tool_spans = [
        attributes
        for attributes in get_logfire_summary().attributes.values()
        if attributes.get('gen_ai.operation.name') == 'execute_tool'
    ]
    assert len(tool_spans) == 1
    assert 'logfire.level_num' not in tool_spans[0]
    assert 'pydantic_ai.tool.failure_stage' not in tool_spans[0]
    assert tool_spans[0]['gen_ai.tool.call.result'] == '4'


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_raw_mode_validation_failure_emits_span(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """Sandboxed dispatch (`handle_call(wrap_validation_errors=False)`) still emits the failure span.

    The raw exception surfaces to the dispatching code rather than the model: no
    `RetryPromptPart` enters message history, so the span's recorded retry prompt is a
    rendered description of the failure, not a message the model received.
    """

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('dispatch', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(call_tool),
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
    )

    @agent.tool_plain
    def double(x: int) -> int:
        return x * 2  # pragma: no cover — only ever dispatched with invalid args

    @agent.tool
    async def dispatch(ctx: RunContext[Any]) -> str:
        assert ctx.tool_manager is not None
        inner = ToolCallPart('double', {'x': 'invalid'}, tool_call_id='inner-call')
        with pytest.raises(ValidationError):
            await ctx.tool_manager.handle_call(inner, wrap_validation_errors=False)
        return 'validation error handled by dispatching code'

    result = agent.run_sync('go')

    tool_spans = [
        attributes
        for attributes in get_logfire_summary().attributes.values()
        if attributes.get('gen_ai.operation.name') == 'execute_tool'
    ]
    failure_spans = [span for span in tool_spans if span.get('pydantic_ai.tool.failure_stage') == 'validation']
    assert len(failure_spans) == 1
    assert failure_spans[0]['logfire.level_num'] == 17
    assert failure_spans[0]['logfire.msg'] == 'invalid tool call: double'
    assert not any(isinstance(part, RetryPromptPart) for message in result.all_messages() for part in message.parts)


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_tool_failed_span_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    """A tool raising `ToolFailed` records the failure as the tool result on the span, like `ModelRetry`.

    Parallel to the retry case in `test_include_tool_args_span_attributes`: the model-visible failure
    message is recorded as `gen_ai.tool.call.result` when content is included (with no "Fix the errors
    and try again." suffix, since it's a failure rather than a retry), and excluded when content is off.
    """
    my_agent = Agent(
        model=TestModel(seed=42),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=include_content))],
    )

    @my_agent.tool_plain
    async def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        raise ToolFailed('numbers service unavailable')

    my_agent.run_sync('Add 42 and 42')

    summary = get_logfire_summary()
    tool_attributes = next(
        attributes for attributes in summary.attributes.values() if attributes.get('gen_ai.tool.name') == 'add_numbers'
    )

    # Guards the documented "traced as an error in telemetry" behavior for `ToolFailed`
    # (level 17 = error); this goes through a different branch than `ModelRetry`'s, so it
    # isn't covered by the retry test. `level_num` is fine here — this file inspects the
    # Logfire-captured view of the spans.
    assert tool_attributes['logfire.level_num'] == 17
    if include_content:
        assert tool_attributes.get('gen_ai.tool.call.result') == 'numbers service unavailable'
    else:
        assert 'gen_ai.tool.call.result' not in tool_attributes


class WeatherInfo(BaseModel):
    temperature: float
    description: str


def get_weather_info(city: str) -> WeatherInfo:
    return WeatherInfo(temperature=28.7, description='sunny')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [
        True,
        False,
        deprecated_instrumentation_settings(version=2),
        deprecated_instrumentation_settings(version=3),
    ],
)
def test_logfire_output_function_v2_v3(
    get_logfire_summary: Callable[[], LogfireSummary],
    instrument: InstrumentationSettings | bool,
) -> None:
    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    capabilities = (
        [
            Instrumentation(
                settings=instrument if isinstance(instrument, InstrumentationSettings) else InstrumentationSettings()
            )
        ]
        if instrument
        else []
    )
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=capabilities)
    result = my_agent.run_sync('Mexico City', output_type=get_weather_info)
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    if isinstance(instrument, InstrumentationSettings) and instrument.version == 2:
        [output_function_attributes] = [
            attributes
            for attributes in summary.attributes.values()
            if attributes.get('gen_ai.tool.name') == 'final_result'
            and 'output function' in attributes.get('logfire.msg', '')
        ]
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'chat function:call_tool:', 'message': 'chat function:call_tool:'},
                        {
                            'id': 2,
                            'name': 'running output function',
                            'message': 'running output function: final_result',
                        },
                    ],
                }
            ]
        )
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'logfire.msg': 'running output function: final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
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
                'tool_arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'tool_response': '{"temperature":28.7,"description":"sunny"}',
            }
        )

    elif instrument is True or isinstance(instrument, InstrumentationSettings) and instrument.version == 3:
        [output_function_attributes] = [
            attributes
            for attributes in summary.attributes.values()
            if attributes.get('gen_ai.tool.name') == 'final_result'
            and 'output function' in attributes.get('logfire.msg', '')
        ]
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'chat function:call_tool:', 'message': 'chat function:call_tool:'},
                        {
                            'id': 2,
                            'name': 'execute_tool final_result',
                            'message': 'running output function: final_result',
                        },
                    ],
                }
            ]
        )
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'logfire.msg': 'running output function: final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert summary.traces == snapshot([])
        assert summary.attributes == snapshot({})


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_function_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=get_weather_info)
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_function_with_run_context_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    def get_weather_with_ctx(ctx: RunContext, city: str) -> WeatherInfo:
        assert ctx is not None
        return WeatherInfo(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=get_weather_with_ctx)
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_function_with_retry_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    def get_weather_with_retry(city: str) -> WeatherInfo:
        if city != 'Mexico City':
            from pydantic_ai import ModelRetry

            raise ModelRetry('City not found, I only know Mexico City')
        return WeatherInfo(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            args_json = '{"city": "New York City"}'
        else:
            args_json = '{"city": "Mexico City"}'

        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('New York City', output_type=get_weather_with_retry)
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    output_function_attributes = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            [
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'final_result',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: final_result',
                    'gen_ai.tool.call.id': IsStr(),
                    'gen_ai.tool.call.arguments': '"New York City"',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                },
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'final_result',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: final_result',
                    'gen_ai.tool.call.id': IsStr(),
                    'gen_ai.tool.call.arguments': '"Mexico City"',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                    'logfire.span_type': 'span',
                    'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
                },
            ]
        )
    else:
        assert output_function_attributes == snapshot(
            [
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'final_result',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: final_result',
                    'gen_ai.tool.call.id': IsStr(),
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                },
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'final_result',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: final_result',
                    'gen_ai.tool.call.id': IsStr(),
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                    'logfire.span_type': 'span',
                },
            ]
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_function_with_custom_tool_name_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    from pydantic_ai.output import ToolOutput

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=ToolOutput(get_weather_info, name='get_weather'))
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes with custom tool name
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'get_weather'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'get_weather',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: get_weather',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'get_weather',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: get_weather',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}, 'gen_ai.tool.call.id': {}}})
                ),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_bound_instance_method_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, city: str):
            return self

    weather = Weather(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=weather.get_weather)
    assert result.output == Weather(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}, 'gen_ai.tool.call.id': {}}})
                ),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_bound_instance_method_with_run_context_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, ctx: RunContext, city: str):
            assert ctx is not None
            return self

    weather = Weather(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=weather.get_weather)
    assert result.output == Weather(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}, 'gen_ai.tool.call.id': {}}})
                ),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_async_function_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    """Test logfire attributes for async output function types."""

    async def get_weather_async(city: str) -> WeatherInfo:
        return WeatherInfo(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('Mexico City', output_type=get_weather_async)
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('gen_ai.tool.name') == 'final_result'
        and 'output function' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'gen_ai.conversation.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                                'gen_ai.tool.call.id': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"Mexico City"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'final_result',
                'gen_ai.tool.call.id': IsStr(),
                'logfire.msg': 'running output function: final_result',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}, 'gen_ai.tool.call.id': {}}})
                ),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


def upcase_text(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_text_output_function_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    """Test logfire attributes for TextOutput functions (TextOutputProcessor)."""

    def call_text_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Return a plain text response (not a tool call)
        from pydantic_ai import TextPart

        return ModelResponse(parts=[TextPart(content='hello world')])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(
        model=FunctionModel(call_text_response), capabilities=[Instrumentation(settings=instrumentation_settings)]
    )

    result = my_agent.run_sync('Say hello', output_type=TextOutput(upcase_text))
    assert result.output == 'HELLO WORLD'

    summary = get_logfire_summary()

    # Find the text output function span attributes
    [text_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if 'running output function: upcase_text' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert text_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'upcase_text',
                'logfire.msg': 'running output function: upcase_text',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"hello world"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': 'HELLO WORLD',
            }
        )
    else:
        assert text_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'upcase_text',
                'logfire.msg': 'running output function: upcase_text',
                'logfire.json_schema': IsJson(snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}}})),
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'my_agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_prompted_output_function_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    """Test that spans are created for PromptedOutput functions with appropriate attributes."""

    def upcase_text(text: str) -> str:
        return text.upper()

    call_count = 0

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        # Simulate the model returning JSON that will be parsed and used to call the function
        return ModelResponse(parts=[TextPart(content='{"text": "hello world"}')])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    agent = Agent(
        model=FunctionModel(call_tool),
        capabilities=[Instrumentation(settings=instrumentation_settings)],
        output_type=PromptedOutput(upcase_text),
    )

    result = agent.run_sync('test')

    # Check that the function was called and returned the expected result
    assert result.output == 'HELLO WORLD'
    assert call_count == 1

    summary = get_logfire_summary()

    # Find the output function span attributes
    [output_function_attributes] = [
        attributes
        for attributes in summary.attributes.values()
        if attributes.get('logfire.msg', '').startswith('running output function: upcase_text')
    ]

    if include_content:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'upcase_text',
                'logfire.msg': 'running output function: upcase_text',
                'gen_ai.conversation.id': IsStr(),
                'logfire.json_schema': IsJson(
                    snapshot(
                        {
                            'type': 'object',
                            'properties': {
                                'gen_ai.tool.call.arguments': {'type': 'object'},
                                'gen_ai.tool.call.result': {'type': 'object'},
                                'gen_ai.tool.name': {},
                            },
                        }
                    )
                ),
                'gen_ai.tool.call.arguments': '"hello world"',
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'gen_ai.tool.call.result': 'HELLO WORLD',
            }
        )
    else:
        assert output_function_attributes == snapshot(
            {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'upcase_text',
                'logfire.msg': 'running output function: upcase_text',
                'logfire.json_schema': IsJson(snapshot({'type': 'object', 'properties': {'gen_ai.tool.name': {}}})),
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
            }
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_output_type_text_output_function_with_retry_logfire_attributes(
    get_logfire_summary: Callable[[], LogfireSummary],
    include_content: bool,
) -> None:
    def get_weather_with_retry(ctx: RunContext, city: str) -> WeatherInfo:
        assert ctx is not None
        if city != 'Mexico City':
            from pydantic_ai import ModelRetry

            raise ModelRetry('City not found, I only know Mexico City')
        return WeatherInfo(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            city = 'New York City'
        else:
            city = 'Mexico City'

        return ModelResponse(parts=[TextPart(content=city)])

    instrumentation_settings = InstrumentationSettings(include_content=include_content)
    my_agent = Agent(model=FunctionModel(call_tool), capabilities=[Instrumentation(settings=instrumentation_settings)])

    result = my_agent.run_sync('New York City', output_type=TextOutput(get_weather_with_retry))
    assert result.output == WeatherInfo(temperature=28.7, description='sunny')

    summary = get_logfire_summary()

    text_function_attributes = [
        attributes
        for attributes in summary.attributes.values()
        if 'running output function: get_weather_with_retry' in attributes.get('logfire.msg', '')
    ]

    if include_content:
        assert text_function_attributes == snapshot(
            [
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'get_weather_with_retry',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: get_weather_with_retry',
                    'gen_ai.tool.call.arguments': '"New York City"',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{}}}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                },
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'get_weather_with_retry',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: get_weather_with_retry',
                    'gen_ai.tool.call.arguments': '"Mexico City"',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{}}}',
                    'logfire.span_type': 'span',
                    'gen_ai.tool.call.result': '{"temperature":28.7,"description":"sunny"}',
                },
            ]
        )
    else:
        assert text_function_attributes == snapshot(
            [
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'get_weather_with_retry',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: get_weather_with_retry',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{}}}',
                    'logfire.span_type': 'span',
                    'logfire.level_num': 17,
                },
                {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'get_weather_with_retry',
                    'gen_ai.agent.name': 'my_agent',
                    'gen_ai.agent.call.id': IsStr(),
                    'gen_ai.conversation.id': IsStr(),
                    'logfire.msg': 'running output function: get_weather_with_retry',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.name":{}}}',
                    'logfire.span_type': 'span',
                },
            ]
        )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_static_function_instructions_in_agent_run_span(
    get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=instrument)])

    @my_agent.instructions
    def instructions():
        return 'Here are some instructions'

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')

    summary = get_logfire_summary()
    chat_span_attributes = summary.attributes[1]
    if instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': '{"content":"a"}',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 5,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'arguments': {'content': 'a'},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'result': 'Final result processed.',
                                }
                            ],
                        },
                    ]
                )
            ),
            'gen_ai.system_instructions': '[{"type":"text","content":"Here are some instructions"}]',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )

    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot([{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}])
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call',
                            'id': IsStr(),
                            'name': 'final_result',
                            'arguments': {'content': 'a'},
                        }
                    ],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instructions_from_history_when_model_request_fails_before_instrumentation(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    class FailBeforeModelRequest(AbstractCapability[Any]):
        async def before_model_request(self, ctx: RunContext[Any], request_context: Any) -> Any:
            raise RuntimeError('boom')

    my_agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings()), FailBeforeModelRequest()],
    )

    with pytest.raises(RuntimeError, match='boom'):
        my_agent.run_sync(
            'Hello',
            message_history=[
                ModelRequest(
                    parts=[UserPromptPart(content='Hi')],
                    instructions='Instructions from history',
                    timestamp=IsDatetime(),
                ),
                ModelResponse(parts=[TextPart(content='Hello')]),
            ],
        )

    summary = get_logfire_summary()
    assert summary.attributes[0]['gen_ai.system_instructions'] == snapshot(
        '[{"type":"text","content":"Instructions from history"}]'
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_dynamic_function_instructions_in_agent_run_span(
    get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=instrument)])

    @my_agent.instructions
    def instructions(ctx: RunContext):
        return f'This is step {ctx.run_step}'

    @my_agent.tool_plain
    def my_tool() -> str:
        return 'This is a tool call'

    result = my_agent.run_sync('Hello', output_type=MyOutput)
    assert result.output == MyOutput(content='a')

    summary = get_logfire_summary()
    chat_span_attributes = summary.attributes[1]
    if instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 2, 'name': 'running tool', 'message': 'running tool: my_tool'},
                        {'id': 3, 'name': 'chat test', 'message': 'chat test'},
                    ],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [
                        {'id': 1, 'name': 'chat test', 'message': 'chat test'},
                        {'id': 2, 'name': 'execute_tool my_tool', 'message': 'running tool: my_tool'},
                        {'id': 3, 'name': 'chat test', 'message': 'chat test'},
                    ],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': '{"content":"a"}',
            'gen_ai.aggregated_usage.input_tokens': 107,
            'gen_ai.aggregated_usage.output_tokens': 9,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': 'pyd_ai_tool_call_id__my_tool',
                                    'name': 'my_tool',
                                    'arguments': {},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': 'pyd_ai_tool_call_id__my_tool',
                                    'name': 'my_tool',
                                    'result': 'This is a tool call',
                                }
                            ],
                        },
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'arguments': {'content': 'a'},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'result': 'Final result processed.',
                                }
                            ],
                        },
                    ]
                )
            ),
            'gen_ai.system_instructions': '[{"type":"text","content":"This is step 2"}]',
            'pydantic_ai.variable_instructions': True,
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'pydantic_ai.variable_instructions': {},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )

    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot([{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}])
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call',
                            'id': IsStr(),
                            'name': 'my_tool',
                            'arguments': {},
                        }
                    ],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
def test_function_instructions_with_history_in_agent_run_span(
    get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings
) -> None:
    @dataclass
    class MyOutput:
        content: str

    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=instrument)])

    @my_agent.instructions
    def instructions(ctx: RunContext):
        return 'Instructions for the current agent run'

    result = my_agent.run_sync(
        'Hello',
        message_history=[
            ModelRequest(
                parts=[UserPromptPart(content='Hi')],
                instructions='Instructions from a previous agent run',
                timestamp=IsDatetime(),
            ),
            ModelResponse(parts=[TextPart(content='Hello')]),
        ],
        output_type=MyOutput,
    )
    assert result.output == MyOutput(content='a')

    summary = get_logfire_summary()
    chat_span_attributes = summary.attributes[1]
    if instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': '{"content":"a"}',
            'gen_ai.aggregated_usage.input_tokens': 52,
            'gen_ai.aggregated_usage.output_tokens': 6,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'arguments': {'content': 'a'},
                                }
                            ],
                        },
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': IsStr(),
                                    'name': 'final_result',
                                    'result': 'Final result processed.',
                                }
                            ],
                        },
                    ]
                )
            ),
            'pydantic_ai.new_message_index': 2,
            'gen_ai.system_instructions': '[{"type":"text","content":"Instructions for the current agent run"}]',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'pydantic_ai.new_message_index': {},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )

    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot(
            [
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
            ]
        )
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool_call',
                            'id': IsStr(),
                            'name': 'final_result',
                            'arguments': {'content': 'a'},
                        }
                    ],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize(
    'instrument',
    [deprecated_instrumentation_settings(version=2), deprecated_instrumentation_settings(version=3)],
)
async def test_run_stream(
    get_logfire_summary: Callable[[], LogfireSummary], instrument: InstrumentationSettings
) -> None:
    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=instrument)])

    @my_agent.instructions
    def instructions(ctx: RunContext):
        return 'Instructions for the current agent run'

    async with my_agent.run_stream('Hello') as stream:
        async for _ in stream.stream_output():
            pass

    summary = get_logfire_summary()
    chat_span_attributes = summary.attributes[1]
    if instrument.version == 2:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'agent run',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    else:
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent my_agent',
                    'message': 'my_agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )

    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': 'success (no tool calls)',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 4,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success (no tool calls)'}]},
                    ]
                )
            ),
            'gen_ai.system_instructions': '[{"type":"text","content":"Instructions for the current agent run"}]',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )

    assert chat_span_attributes['gen_ai.input.messages'] == IsJson(
        snapshot([{'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]}])
    )
    assert chat_span_attributes['gen_ai.output.messages'] == IsJson(
        snapshot(
            [
                {
                    'role': 'assistant',
                    'parts': [{'type': 'text', 'content': 'success (no tool calls)'}],
                }
            ]
        )
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_run_stream_sync(get_logfire_summary: Callable[[], LogfireSummary]) -> None:
    my_agent = Agent(model=TestModel(), capabilities=[Instrumentation()])

    @my_agent.instructions
    def instructions(ctx: RunContext):
        return 'Instructions for the current agent run'

    with my_agent.run_stream_sync('Hello') as stream:
        for _ in stream.stream_output():
            pass

    summary = get_logfire_summary()
    # The `chat` span is correctly nested under the agent run span (not orphaned).
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'name': 'invoke_agent my_agent',
                'message': 'my_agent run',
                'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
            }
        ]
    )
    # The agent run span's end attributes (all_messages, final_result, usage) are all populated.
    assert summary.attributes[0] == snapshot(
        {
            'model_name': 'test',
            'agent_name': 'my_agent',
            'gen_ai.agent.name': 'my_agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.operation.name': 'invoke_agent',
            'logfire.msg': 'my_agent run',
            'logfire.span_type': 'span',
            'final_result': 'success (no tool calls)',
            'gen_ai.aggregated_usage.input_tokens': 51,
            'gen_ai.aggregated_usage.output_tokens': 4,
            'pydantic_ai.all_messages': IsJson(
                snapshot(
                    [
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hello'}]},
                        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'success (no tool calls)'}]},
                    ]
                )
            ),
            'gen_ai.system_instructions': '[{"type":"text","content":"Instructions for the current agent run"}]',
            'logfire.json_schema': IsJson(
                snapshot(
                    {
                        'type': 'object',
                        'properties': {
                            'pydantic_ai.all_messages': {'type': 'array'},
                            'gen_ai.system_instructions': {'type': 'array'},
                            'final_result': {'type': 'object'},
                        },
                    }
                )
            ),
        }
    )


def _get_tool_span(capfire: CaptureLogfire) -> dict[str, Any]:
    """Get the completed tool span from exported spans."""
    spans = strip_logfire_metrics(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True))
    tool_span = next(
        s for s in spans if s['attributes'].get('logfire.span_type') == 'span' and 'tool' in s['name'].lower()
    )
    return tool_span


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_call_deferred_v2(capfire: CaptureLogfire) -> None:
    """Test that CallDeferred on v2 marks span as ERROR with deferral attributes."""
    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=deprecated_instrumentation_settings(version=2))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred(metadata={'task_id': 'task-123'})

    agent.run_sync('Hello')

    assert _get_tool_span(capfire) == snapshot(
        {
            'name': 'running tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 6000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'tool_arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'tool_arguments': {'type': 'object'},
                        'tool_response': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'CallDeferred',
                'pydantic_ai.tool.deferral.metadata': {'task_id': 'task-123'},
                'logfire.level_num': 17,
            },
            'events': [
                {
                    'name': 'exception',
                    'timestamp': 5000000000,
                    'attributes': {
                        'exception.type': 'pydantic_ai.exceptions.CallDeferred',
                        'exception.message': '',
                        'exception.stacktrace': 'pydantic_ai.exceptions.CallDeferred',
                        'exception.escaped': 'True',
                    },
                }
            ],
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_approval_required_v2(capfire: CaptureLogfire) -> None:
    """Test that ApprovalRequired on v2 marks span as ERROR with deferral attributes."""
    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=deprecated_instrumentation_settings(version=2))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise ApprovalRequired(metadata={'task_id': 'task-123'})

    agent.run_sync('Hello')

    assert _get_tool_span(capfire) == snapshot(
        {
            'name': 'running tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 6000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'tool_arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'tool_arguments': {'type': 'object'},
                        'tool_response': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'ApprovalRequired',
                'pydantic_ai.tool.deferral.metadata': {'task_id': 'task-123'},
                'logfire.level_num': 17,
            },
            'events': [
                {
                    'name': 'exception',
                    'timestamp': 5000000000,
                    'attributes': {
                        'exception.type': 'pydantic_ai.exceptions.ApprovalRequired',
                        'exception.message': '',
                        'exception.stacktrace': 'pydantic_ai.exceptions.ApprovalRequired',
                        'exception.escaped': 'True',
                    },
                }
            ],
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_call_deferred_v5(capfire: CaptureLogfire) -> None:
    """Test that CallDeferred on v5 leaves span as UNSET with deferral attributes."""
    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred(metadata={'task_id': 'task-123'})

    agent.run_sync('Hello')

    assert _get_tool_span(capfire) == snapshot(
        {
            'name': 'execute_tool my_tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 5000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'gen_ai.tool.call.arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'gen_ai.tool.call.arguments': {'type': 'object'},
                        'gen_ai.tool.call.result': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'CallDeferred',
                'pydantic_ai.tool.deferral.metadata': {'task_id': 'task-123'},
            },
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_approval_required_v5(capfire: CaptureLogfire) -> None:
    """Test that ApprovalRequired on v5 leaves span as UNSET with deferral attributes."""
    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise ApprovalRequired(metadata={'task_id': 'task-123'})

    agent.run_sync('Hello')

    assert _get_tool_span(capfire) == snapshot(
        {
            'name': 'execute_tool my_tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 5000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'gen_ai.tool.call.arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'gen_ai.tool.call.arguments': {'type': 'object'},
                        'gen_ai.tool.call.result': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'ApprovalRequired',
                'pydantic_ai.tool.deferral.metadata': {'task_id': 'task-123'},
            },
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_no_metadata(capfire: CaptureLogfire) -> None:
    """Test that deferral without metadata doesn't set the metadata attribute."""
    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred()

    agent.run_sync('Hello')

    tool_span = _get_tool_span(capfire)

    assert tool_span == snapshot(
        {
            'name': 'execute_tool my_tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 5000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'gen_ai.tool.call.arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'gen_ai.tool.call.arguments': {'type': 'object'},
                        'gen_ai.tool.call.result': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'CallDeferred',
            },
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_non_serializable_metadata(capfire: CaptureLogfire) -> None:
    """Test that non-JSON-serializable metadata falls back to repr() representation."""

    class CustomObj:
        def __repr__(self) -> str:
            return '<CustomObj>'

    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred(metadata={'obj': CustomObj()})

    agent.run_sync('Hello')

    tool_span = _get_tool_span(capfire)

    assert tool_span == snapshot(
        {
            'name': 'execute_tool my_tool',
            'context': {'trace_id': 1, 'span_id': 5, 'is_remote': False},
            'parent': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'start_time': 4000000000,
            'end_time': 5000000000,
            'attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'my_tool',
                'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__my_tool',
                'gen_ai.tool.call.arguments': {'x': 0},
                'logfire.msg': 'running tool: my_tool',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'gen_ai.tool.call.arguments': {'type': 'object'},
                        'gen_ai.tool.call.result': {'type': 'object'},
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                },
                'gen_ai.conversation.id': IsStr(),
                'logfire.span_type': 'span',
                'gen_ai.agent.name': 'agent',
                'gen_ai.agent.call.id': IsStr(),
                'pydantic_ai.tool.deferral.name': 'CallDeferred',
                'pydantic_ai.tool.deferral.metadata': "{'obj': <CustomObj>}",
            },
        }
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_model_retry_still_errors_v5(capfire: CaptureLogfire) -> None:
    """Test that ModelRetry on v5 still records the span as an error.

    The deferral fix (CallDeferred/ApprovalRequired → UNSET on v5) must not affect
    ModelRetry, which wraps as ToolRetryError and should always be an error span.
    """
    agent = Agent(
        TestModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        raise ModelRetry('please try again with different input')

    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync('Hello')

    tool_span = _get_tool_span(capfire)

    # ToolRetryError should still be recorded as an error on v5 — only deferrals get UNSET
    assert tool_span['attributes'].get('logfire.level_num') == 17
    # No deferral attributes should be set — this is a retry, not a deferral
    assert 'pydantic_ai.tool.deferral.name' not in tool_span['attributes']
    assert 'pydantic_ai.tool.deferral.metadata' not in tool_span['attributes']


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_deferral_unexpected_exception_still_errors_v5(capfire: CaptureLogfire) -> None:
    """Test that unexpected exceptions on v5 still record the span as an error.

    The deferral fix must not affect general exception handling — only
    CallDeferred and ApprovalRequired get UNSET status on v5.
    """
    agent = Agent(
        TestModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings(version=5))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        raise ValueError('something went wrong')

    with pytest.raises(ValueError, match='something went wrong'):
        agent.run_sync('Hello')

    tool_span = _get_tool_span(capfire)

    # ValueError path should still record error regardless of instrumentation version
    assert tool_span['attributes'].get('logfire.level_num') == 17
    # No deferral attributes should be set
    assert 'pydantic_ai.tool.deferral.name' not in tool_span['attributes']
    assert 'pydantic_ai.tool.deferral.metadata' not in tool_span['attributes']


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_agent_description(capfire: CaptureLogfire) -> None:
    agent = Agent(
        model=TestModel(),
        name='my_agent',
        description='An agent that greets users',
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
    )
    assert agent.description == 'An agent that greets users'

    await agent.run('Hello')

    spans = capfire.exporter.exported_spans_as_dict()
    agent_run_span = next(s for s in spans if s['name'] == 'invoke_agent my_agent')
    assert agent_run_span['attributes']['gen_ai.agent.description'] == 'An agent that greets users'

    agent.description = 'Updated description'
    assert agent.description == 'Updated description'


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_agent_description_absent_when_none(capfire: CaptureLogfire) -> None:
    agent = Agent(
        model=TestModel(), name='my_agent', capabilities=[Instrumentation(settings=InstrumentationSettings())]
    )
    assert agent.description is None

    await agent.run('Hello')

    spans = capfire.exporter.exported_spans_as_dict()
    agent_run_span = next(s for s in spans if s['name'] == 'invoke_agent my_agent')
    assert 'gen_ai.agent.description' not in agent_run_span['attributes']


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instrumentation_capability_with_model_settings(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """Test that Instrumentation capability correctly records model settings like temperature."""
    agent = Agent(
        model=TestModel(),
        model_settings={'temperature': 0.5, 'max_tokens': 100},
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
    )
    agent.run_sync('Hello')

    summary = get_logfire_summary()
    chat_attrs = summary.attributes[1]
    assert chat_attrs['gen_ai.request.temperature'] == 0.5
    assert chat_attrs['gen_ai.request.max_tokens'] == 100


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instrumentation_capability_serialization() -> None:
    """`Instrumentation` is constructible from a YAML/JSON spec via the serializable subset of
    `InstrumentationSettings` kwargs; non-serializable OTel providers fall back to the globals."""
    from pydantic_ai.capabilities.instrumentation import Instrumentation
    from pydantic_ai.models.instrumented import InstrumentationSettings

    assert Instrumentation.get_serialization_name() == 'Instrumentation'

    with pytest.warns(
        PydanticAIDeprecationWarning, match=r'Instrumentation format versions 2, 3, and 4 are deprecated'
    ):
        cap = Instrumentation.from_spec(version=2, include_content=False)
    assert isinstance(cap, Instrumentation)
    assert isinstance(cap.settings, InstrumentationSettings)
    assert cap.settings.version == 2
    assert cap.settings.include_content is False

    # Empty kwargs form: `Instrumentation: {}` in YAML, which takes the class-level default id.
    cap_default = Instrumentation.from_spec()
    assert cap_default.settings.version == InstrumentationSettings().version
    assert cap_default.id == 'instrumentation'

    # The spec deliberately cannot name the capability: an agent has one instrumentation
    # configuration, and two under different ids would stop resolving to one.
    with pytest.raises(TypeError, match='id'):
        Instrumentation.from_spec(id='monitoring')  # pyright: ignore[reportCallIssue]

    # Every serializable setting is expressible, including the one the typed signature first missed.
    assert (
        Instrumentation.from_spec(include_model_request_parameters=False).settings.include_model_request_parameters
        is False
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instrumentation_capability_explicit(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """Test using Instrumentation as an explicit capability (not via instrument=True)."""
    from pydantic_ai.capabilities.instrumentation import Instrumentation

    instrumentation = Instrumentation(settings=InstrumentationSettings())
    agent = Agent(model=TestModel(), capabilities=[instrumentation])

    result = agent.run_sync('Hello')
    assert result.output == snapshot('success (no tool calls)')

    summary = get_logfire_summary()
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'name': 'invoke_agent agent',
                'message': 'agent run',
                'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
            }
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instrument_all_skipped_when_capability_already_present(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """`Agent.instrument_all(...)` must not stack a second `Instrumentation` capability when the
    user already added one via `capabilities=[...]` — otherwise spans would be emitted twice.

    Guarded by the `has_capability_type(..., InstrumentationCap)` check in `Agent.iter`.
    """
    from pydantic_ai.capabilities.instrumentation import Instrumentation

    Agent.instrument_all(True)
    try:
        agent = Agent(model=TestModel(), capabilities=[Instrumentation(settings=InstrumentationSettings())])
        result = agent.run_sync('Hello')
        assert result.output == snapshot('success (no tool calls)')

        summary = get_logfire_summary()
        # One `agent run` span, not two.
        assert summary.traces == snapshot(
            [
                {
                    'id': 0,
                    'name': 'invoke_agent agent',
                    'message': 'agent run',
                    'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
                }
            ]
        )
    finally:
        Agent.instrument_all(False)


@dataclass
class _SeenInstrumentation:
    tracer: Any
    include_content: bool
    version: int


def _seen(ctx: RunContext[Any]) -> _SeenInstrumentation:
    return _SeenInstrumentation(ctx.tracer, ctx.trace_include_content, ctx.instrumentation_version)


@dataclass
class _ObserveForRun(AbstractCapability[Any]):
    """Records what a `for_run` hook sees of the run's instrumentation."""

    seen: list[_SeenInstrumentation]

    async def for_run(self, ctx: RunContext[Any]) -> _ObserveForRun:
        self.seen.append(_seen(ctx))
        return self


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('explicit_on', [None, 'agent', 'run'])
def test_run_context_instrumentation_follows_the_capability_that_instruments_the_run(
    explicit_on: Literal['agent', 'run'] | None,
) -> None:
    """An explicit `Instrumentation` capability replaces the `instrument_all()` one, so `RunContext` describes it.

    Its `tracer`, `trace_include_content` and `instrumentation_version` are what the run's spans use, both
    in `for_run` hooks and in the context a tool gets later in the run. Without one, `instrument_all()`'s
    settings still apply.
    """
    implicit = InstrumentationSettings(include_content=True, tracer_provider=TracerProvider())
    explicit = InstrumentationSettings(include_content=False, version=6, tracer_provider=TracerProvider())
    expected = implicit if explicit_on is None else explicit

    for_run_seen: list[_SeenInstrumentation] = []
    agent_capabilities: list[AbstractCapability[Any]] = [_ObserveForRun(for_run_seen)]
    run_capabilities: list[AbstractCapability[Any]] = []
    if explicit_on == 'agent':
        agent_capabilities.append(Instrumentation(settings=explicit))
    elif explicit_on == 'run':
        run_capabilities.append(Instrumentation(settings=explicit))

    Agent.instrument_all(implicit)
    try:
        agent = Agent(TestModel(), capabilities=agent_capabilities)
        tool_seen: list[_SeenInstrumentation] = []

        @agent.tool
        def peek(ctx: RunContext[Any]) -> str:
            tool_seen.append(_seen(ctx))
            return 'ok'

        agent.run_sync('hi', capabilities=run_capabilities)
    finally:
        Agent.instrument_all(False)

    assert implicit.tracer is not explicit.tracer
    expected_seen = _SeenInstrumentation(expected.tracer, expected.include_content, expected.version)
    assert for_run_seen == [expected_seen]
    assert tool_seen == [expected_seen]


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_run_context_instrumentation_follows_a_capability_function_instrumentation() -> None:
    """An `Instrumentation` only a capability function's `for_run` returns still instruments the run, so the
    context the run's tools get describes it too, even though it couldn't be known before `for_run`."""
    implicit_provider = TracerProvider()
    implicit_exporter = InMemorySpanExporter()
    implicit_provider.add_span_processor(SimpleSpanProcessor(implicit_exporter))
    dynamic_provider = TracerProvider()
    dynamic_exporter = InMemorySpanExporter()
    dynamic_provider.add_span_processor(SimpleSpanProcessor(dynamic_exporter))
    dynamic = InstrumentationSettings(include_content=False, version=6, tracer_provider=dynamic_provider)

    Agent.instrument_all(InstrumentationSettings(include_content=True, tracer_provider=implicit_provider))
    try:
        agent = Agent(TestModel())
        tool_seen: list[_SeenInstrumentation] = []

        @agent.tool
        def peek(ctx: RunContext[Any]) -> str:
            tool_seen.append(_seen(ctx))
            return 'ok'

        agent.run_sync('hi', capabilities=[lambda ctx: Instrumentation(settings=dynamic)])
    finally:
        Agent.instrument_all(False)

    assert tool_seen == [_SeenInstrumentation(dynamic.tracer, False, 6)]
    # The capability function's `Instrumentation` is the one that opened the run's spans.
    assert implicit_exporter.get_finished_spans() == ()
    assert [span.name for span in dynamic_exporter.get_finished_spans()] == snapshot(
        ['chat test', 'execute_tool peek', 'chat test', 'invoke_agent agent']
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_agent_with_user_provided_instrumented_model(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """If a user explicitly hands an `InstrumentedModel` to `Agent(model=...)`, its
    settings drive the agent's instrumentation — the model is unwrapped and the
    `Instrumentation` capability emits the spans.
    """
    from pydantic_ai.models.instrumented import InstrumentedModel

    settings = InstrumentationSettings()
    agent = Agent(model=InstrumentedModel(TestModel(), settings))

    result = agent.run_sync('Hello')
    assert result.output == snapshot('success (no tool calls)')

    summary = get_logfire_summary()
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'name': 'invoke_agent agent',
                'message': 'agent run',
                'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
            }
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_agent_instrument_setter(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """`agent.instrument = settings` configures instrumentation post-construction.

    This is the path `logfire.instrument_pydantic_ai(agent)` uses on its `Agent` branch.
    """
    agent = Agent(model=TestModel())
    assert agent.instrument is None
    settings = InstrumentationSettings()
    agent.instrument = settings
    assert agent.instrument is settings

    result = agent.run_sync('Hello')
    assert result.output == snapshot('success (no tool calls)')

    summary = get_logfire_summary()
    assert summary.traces == snapshot(
        [
            {
                'id': 0,
                'name': 'invoke_agent agent',
                'message': 'agent run',
                'children': [{'id': 1, 'name': 'chat test', 'message': 'chat test'}],
            }
        ]
    )


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_instrumentation_capability_template_description(
    capfire: CaptureLogfire,
) -> None:
    """Test that TemplateStr descriptions are rendered in agent run spans."""
    # `TemplateStr` rendering requires the `[spec]` extra (pydantic-handlebars); not in default v2 deps.
    pytest.importorskip('pydantic_handlebars')
    from dataclasses import dataclass

    from pydantic_ai._template import TemplateStr
    from pydantic_ai.capabilities.instrumentation import Instrumentation

    @dataclass
    class MyDeps:
        name: str

    instrumentation = Instrumentation(settings=InstrumentationSettings())
    agent = Agent(
        model=TestModel(),
        capabilities=[instrumentation],
        description=TemplateStr('Agent for {{name}}'),
        deps_type=MyDeps,
    )

    result = agent.run_sync('Hello', deps=MyDeps(name='testing'))
    assert result.output == snapshot('success (no tool calls)')

    spans = strip_logfire_metrics(capfire.exporter.exported_spans_as_dict(parse_json_attributes=True))
    agent_span = spans[-1]  # outermost span is the agent run
    assert agent_span['attributes']['gen_ai.agent.description'] == snapshot('Agent for testing')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
async def test_instrumentation_capability_with_noop_tracer() -> None:
    """When the configured tracer provider is a no-op, model-request spans skip
    attribute population entirely. Regression coverage for the non-recording
    branch in `Instrumentation.wrap_model_request`."""
    from opentelemetry.trace import NoOpTracerProvider

    agent = Agent(
        model=TestModel(),
        capabilities=[Instrumentation(settings=InstrumentationSettings(tracer_provider=NoOpTracerProvider()))],
    )
    result = await agent.run('hello')
    assert result.output == snapshot('success (no tool calls)')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
async def test_instrument_combines_with_outermost_and_innermost_capabilities() -> None:
    """Auto-prepending `Instrumentation` must not wrap a `CombinedCapability` that
    already contains both an outermost and innermost cap — the wrap would force
    `_effective_ordering` to merge those positions and raise "Conflicting positions".
    """
    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering

    class OutermostCap(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

        async def before_run(self, ctx: RunContext[Any]) -> None:
            pass

    class InnermostCap(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

        async def before_run(self, ctx: RunContext[Any]) -> None:
            pass

    agent = Agent(
        model=TestModel(),
        capabilities=[OutermostCap(), InnermostCap(), Instrumentation(settings=InstrumentationSettings())],
    )
    result = await agent.run('hello')
    assert result.output == snapshot('success (no tool calls)')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_output_function_call_deferred_recorded_as_error(
    get_logfire_summary: Callable[[], LogfireSummary],
) -> None:
    """An output function raising `CallDeferred` is recorded as a regular error on the
    `wrap_output_process` span — that hook reserves the deferral-attribute path for
    real tool executions (`wrap_tool_execute`)."""

    def defer_text(text: str) -> str:
        from pydantic_ai.exceptions import CallDeferred

        raise CallDeferred()

    def call_text_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='hi')])

    from pydantic_ai.exceptions import CallDeferred

    my_agent = Agent(
        model=FunctionModel(call_text_response), capabilities=[Instrumentation(settings=InstrumentationSettings())]
    )
    with pytest.raises(CallDeferred):
        my_agent.run_sync('anything', output_type=TextOutput(defer_text))

    summary = get_logfire_summary()
    [span_attrs] = [attrs for attrs in summary.attributes.values() if attrs.get('gen_ai.tool.name') == 'defer_text']
    # The span was recorded with ERROR status — the standard exception path,
    # not the deferral-attribute path that `wrap_tool_execute` uses.
    assert span_attrs.get('logfire.level_num', 0) >= 17  # error level
    assert 'pydantic_ai.tool.deferral.name' not in span_attrs


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
async def test_a_settled_stream_keeps_its_traceparent_after_the_stream_is_gone(capfire: CaptureLogfire) -> None:
    """The trace context has to be captured at completion, not read when `result` is accessed.

    A settled [`AgentRunResult`][pydantic_ai.run.AgentRunResult] is for handing a run to code that
    outlives the stream, and by then the agent run span has closed and there is no ambient context
    left to read.
    """
    agent = Agent(TestModel(custom_output_text='streamed'), capabilities=[Instrumentation()])

    async with agent.run_stream('Stream this') as streamed:
        await streamed.get_output()
        inside = streamed.result

    outside = streamed.result

    assert outside._traceparent(required=False) == inside._traceparent(required=False)  # pyright: ignore[reportPrivateUsage]
    assert outside._traceparent(required=False) is not None  # pyright: ignore[reportPrivateUsage]

    # And it survives the round-trip the serialized shape exists for.
    adapter = TypeAdapter(AgentRunResult[str])
    reloaded = adapter.validate_json(adapter.dump_json(outside))
    assert reloaded._traceparent(required=False) == outside._traceparent(required=False)  # pyright: ignore[reportPrivateUsage]


# Name, `exception.type` and `exception.escaped` of each event, in order. Tool spans record their own
# exceptions as escaped; the run span stands in for the OTel SDK's `use_span`, which does not.
_EXPECTED_EXCEPTION_EVENTS: dict[str, list[tuple[str, str, str]]] = {
    'retry': [
        ('execute_tool my_tool', 'pydantic_ai.exceptions.ToolRetryError', 'True'),
        ('execute_tool my_tool', 'pydantic_ai.exceptions.UnexpectedModelBehavior', 'True'),
        ('invoke_agent agent', 'pydantic_ai.exceptions.UnexpectedModelBehavior', 'False'),
    ],
    'failed': [
        ('execute_tool my_tool', 'pydantic_ai.exceptions.ToolFailedError', 'True'),
        ('execute_tool my_tool', 'pydantic_ai.exceptions.ToolFailedError', 'True'),
    ],
    'exception': [
        ('execute_tool my_tool', 'ValueError', 'True'),
        ('invoke_agent agent', 'ValueError', 'False'),
    ],
}


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('failure', ['retry', 'failed', 'exception'])
def test_exception_events_honor_include_content(
    capfire: CaptureLogfire, include_content: bool, failure: Literal['retry', 'failed', 'exception']
) -> None:
    """Exception events on tool and agent run spans carry a message and stack trace only when content is included.

    A `ToolRetryError` or `ToolFailedError` message is the retry prompt or failed result the model sees.
    When a tool runs out of retries, the `UnexpectedModelBehavior` that ends the run is chained from the
    last retry, and the OTel SDK's stack trace formatting repeats the cause's text on the second tool span
    and on the run span (the test exporter trims stack traces, so only the message is checked here). A
    plain exception from tool code is user-authored but may quote arguments, so it is treated the same
    way. With `include_content=False`, every event keeps only the exception type.
    """
    model_calls = 0

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls <= 2:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 1})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(call_tool),
        retries=1,
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=include_content))],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        if failure == 'retry':
            raise ModelRetry('retry-secret')
        elif failure == 'failed':
            raise ToolFailed('failed-secret')
        else:
            raise ValueError('exception-secret')

    if failure == 'failed':
        agent.run_sync('Use the tool')
    else:
        with pytest.raises(UnexpectedModelBehavior if failure == 'retry' else ValueError):
            agent.run_sync('Use the tool')

    spans = capfire.exporter.exported_spans_as_dict()
    events = [
        (span['name'], event['attributes'])
        for span in spans
        for event in span.get('events', [])
        if event['name'] == 'exception'
    ]
    assert [
        (name, attributes['exception.type'], attributes['exception.escaped']) for name, attributes in events
    ] == _EXPECTED_EXCEPTION_EVENTS[failure]
    if include_content:
        assert all({'exception.message', 'exception.stacktrace'} <= set(attributes) for _, attributes in events)
        assert 'secret' in events[0][1]['exception.message']
    else:
        assert all(set(attributes) == {'exception.type', 'exception.escaped'} for _, attributes in events)
        assert 'secret' not in str(spans)

    # The run span's status description repeats the exception message, and is not part of the
    # exported span dicts above, so it is checked separately.
    errored = [span for span in capfire.exporter.exported_spans if span.status.status_code is StatusCode.ERROR]
    assert [span.name for span in errored] == [name for name, _, _ in _EXPECTED_EXCEPTION_EVENTS[failure]]
    # Only the run span carries a description: `use_span` set one, the tool spans never did.
    assert [span.status.description for span in errored] == [
        (IsStr(regex=r'\w+: [\s\S]+') if include_content else None) if name.startswith('invoke_agent') else None
        for name, _, _ in _EXPECTED_EXCEPTION_EVENTS[failure]
    ]
    if include_content and failure == 'exception':
        # Pinned exactly, because the promise is to reproduce the SDK's `f'{type(e).__name__}: {e}'`.
        assert errored[-1].status.description == 'ValueError: exception-secret'


class _Interrupted(BaseException):
    """Stands in for a cancellation: raised by tool code, not an `Exception`."""


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_run_span_leaves_base_exceptions_unrecorded(capfire: CaptureLogfire) -> None:
    """A `BaseException` escaping the run stays off the run span.

    Recording the run span's exceptions in the run body rather than leaving it to `use_span` has to
    keep that function's rule that only `Exception` is recorded: a cancellation is control flow, not
    an error, and should not start showing up as a recorded exception on every interrupted run.
    """

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {})])

    agent = Agent(FunctionModel(call_tool), capabilities=[Instrumentation()])

    @agent.tool_plain
    def my_tool() -> str:
        raise _Interrupted

    with pytest.raises(_Interrupted):
        agent.run_sync('Use the tool')

    spans = capfire.exporter.exported_spans_as_dict()
    assert [
        span['name'] for span in spans for event in span.get('events', []) if event['name'] == 'exception'
    ] == snapshot(['execute_tool my_tool'])


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_run_span_reports_the_runs_own_usage_not_the_conversations(capfire: CaptureLogfire) -> None:
    """Carrying `usage=` across a conversation must not inflate each run's span.

    `RunUsage` is accumulated into in place, so a run handed the previous run's object would
    otherwise report the conversation's running total, and anything summing agent-run spans would
    count the earlier turns again. The caller's own total is unaffected: that is what they asked
    for by passing it.
    """
    agent = Agent(TestModel(custom_output_text='hi'), capabilities=[Instrumentation()])

    first = agent.run_sync('one')
    second = agent.run_sync('two', message_history=first.all_messages(), usage=first.usage)

    reported = [
        span['attributes']['gen_ai.aggregated_usage.input_tokens']
        for span in capfire.exporter.exported_spans_as_dict()
        if 'gen_ai.aggregated_usage.input_tokens' in span['attributes']
    ]
    assert reported == snapshot([51, 52])
    assert second.usage.input_tokens == snapshot(103)


async def _run_delegating_agent(*, share_usage: bool, sequential: bool, instrument_delegate: bool = True) -> None:
    """Run a parent agent whose tool delegates to a second agent, once per tool call."""

    async def delegate_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Force the two delegate runs to overlap, so a sibling's usage lands inside this run's window.
        await asyncio.sleep(0.05)
        return ModelResponse(parts=[TextPart('joke')], usage=RequestUsage(input_tokens=10, output_tokens=1))

    delegate = Agent(
        FunctionModel(delegate_fn), name='delegate', capabilities=[Instrumentation()] if instrument_delegate else []
    )

    async def parent_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=1000, output_tokens=5)
        if len(messages) == 1:
            calls = [ToolCallPart('pick', {'n': n}, tool_call_id=str(n)) for n in (1, 2)]
            return ModelResponse(parts=calls, usage=usage)
        return ModelResponse(parts=[TextPart('done')], usage=usage)

    parent = Agent(FunctionModel(parent_fn), name='parent', capabilities=[Instrumentation()])

    @parent.tool(sequential=sequential)
    async def pick(ctx: RunContext[Any], n: int) -> str:
        return (await delegate.run('x', usage=ctx.usage if share_usage else None)).output

    await parent.run('go')


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('share_usage', [True, False])
@pytest.mark.parametrize('sequential', [True, False])
@pytest.mark.anyio
async def test_run_span_reports_its_own_usage_under_concurrent_delegation(
    capfire: CaptureLogfire, share_usage: bool, sequential: bool
) -> None:
    """Each agent-run span reports its own requests, whatever the delegates do with the usage object.

    Concurrent delegates handed the parent's `RunUsage` (the `usage=ctx.usage` pattern in
    `docs/multi-agent-applications.md`) overlap in time, so neither the shared object's contents nor
    an end-minus-start delta on it can say which run added what: each delegate would otherwise
    absorb its sibling's tokens, by more the wider the fan-out. Crediting the run that made the
    request instead makes all four combinations agree, and makes the spans sum to the run's total
    rather than counting a delegate's tokens again on the parent that contains it.
    """
    await _run_delegating_agent(share_usage=share_usage, sequential=sequential)

    agent_spans = [
        (span['name'], span['attributes']['gen_ai.aggregated_usage.input_tokens'])
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('gen_ai.operation.name') == 'invoke_agent'
    ]
    assert set(agent_spans) == snapshot({('invoke_agent delegate', 10), ('invoke_agent parent', 2000)})
    # The delegates' tokens are reported once, on the delegates, so summing every agent-run span
    # gives the run's total rather than counting them again on the parent containing them.
    assert sum(tokens for _, tokens in agent_spans) == snapshot(2020)


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('share_usage', [True, False])
@pytest.mark.parametrize('sequential', [True, False])
@pytest.mark.anyio
async def test_run_span_excludes_an_uninstrumented_delegates_usage(
    capfire: CaptureLogfire, share_usage: bool, sequential: bool
) -> None:
    """A delegate without a span of its own doesn't report its usage on the caller's.

    The caller's span reports what the caller spent whether or not the runs it starts are
    instrumented: crediting an unspanned delegate's tokens to the nearest instrumented run would
    make that run's span disagree with its own `result.usage`, and look no different from a run
    that really spent that much.
    """
    await _run_delegating_agent(share_usage=share_usage, sequential=sequential, instrument_delegate=False)

    agent_spans = [
        (span['name'], span['attributes']['gen_ai.aggregated_usage.input_tokens'])
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('gen_ai.operation.name') == 'invoke_agent'
    ]
    assert agent_spans == snapshot([('invoke_agent parent', 2000)])


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_nested_delegation_spans_sum_to_the_runs_total(capfire: CaptureLogfire) -> None:
    """Every run in a three-deep tree reports its own requests, so the spans still sum to the total.

    Depth is what separates reporting a run's own usage from reporting its subtree: a leaf's tokens
    belong to one span, not to that span and to each of the runs above it. Both agents here fan out
    to two concurrent children sharing one `RunUsage`, so a leaf's tokens would otherwise be counted
    on the leaf, on both middle runs, and on the top.
    """

    def leaf_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('leaf')], usage=RequestUsage(input_tokens=1))

    leaf = Agent(FunctionModel(leaf_fn), name='leaf', capabilities=[Instrumentation()])

    async def middle_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=10)
        if len(messages) == 1:
            calls = [ToolCallPart('ask_leaf', {'n': n}, tool_call_id=f'm{n}') for n in (1, 2)]
            return ModelResponse(parts=calls, usage=usage)
        return ModelResponse(parts=[TextPart('middle')], usage=usage)

    middle = Agent(FunctionModel(middle_fn), name='middle', capabilities=[Instrumentation()])

    @middle.tool
    async def ask_leaf(ctx: RunContext[Any], n: int) -> str:
        return (await leaf.run('x', usage=ctx.usage)).output

    async def top_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        usage = RequestUsage(input_tokens=100)
        if len(messages) == 1:
            calls = [ToolCallPart('ask_middle', {'n': n}, tool_call_id=str(n)) for n in (1, 2)]
            return ModelResponse(parts=calls, usage=usage)
        return ModelResponse(parts=[TextPart('top')], usage=usage)

    top = Agent(FunctionModel(top_fn), name='top', capabilities=[Instrumentation()])

    @top.tool
    async def ask_middle(ctx: RunContext[Any], n: int) -> str:
        return (await middle.run('x', usage=ctx.usage)).output

    result = await top.run('go')

    reported: dict[str, list[int]] = {}
    for span in capfire.exporter.exported_spans_as_dict():
        if span['attributes'].get('gen_ai.operation.name') == 'invoke_agent':
            name = span['attributes']['gen_ai.agent.name']
            reported.setdefault(name, []).append(span['attributes']['gen_ai.aggregated_usage.input_tokens'])

    # Each run's own requests: four leaf runs of one, two middle runs of two-by-ten, one top run of
    # two-by-a-hundred. Reporting subtrees instead would put 22 on each middle and 244 on the top.
    assert reported == snapshot({'leaf': [1, 1, 1, 1], 'middle': [20, 20], 'top': [200]})
    assert sum(tokens for tokens_per_run in reported.values() for tokens in tokens_per_run) == snapshot(244)
    assert result.usage.input_tokens == snapshot(244)


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.parametrize('include_content', [True, False])
def test_model_request_exception_events_honor_include_content(capfire: CaptureLogfire, include_content: bool) -> None:
    """The model request span follows the same rule as the tool and agent run spans.

    A provider error carries the response body in its message -- providers echo request content
    into those, moderation and invalid-content responses in particular -- so the exception event
    and the status description on the `chat` span have to follow the setting like the rest.
    """

    def fail(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name='fn', body='invalid content: prompt-secret')

    agent = Agent(
        FunctionModel(fail),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=include_content))],
    )

    with pytest.raises(ModelHTTPError):
        agent.run_sync('Hello')

    spans = capfire.exporter.exported_spans_as_dict()
    events = [
        (span['name'], event['attributes'])
        for span in spans
        for event in span.get('events', [])
        if event['name'] == 'exception'
    ]
    assert [(name, attributes['exception.type'], attributes['exception.escaped']) for name, attributes in events] == [
        ('chat function:fail:', 'pydantic_ai.exceptions.ModelHTTPError', 'False'),
        ('invoke_agent agent', 'pydantic_ai.exceptions.ModelHTTPError', 'False'),
    ]
    descriptions = [span.status.description for span in capfire.exporter.exported_spans if span.status.description]
    if include_content:
        assert all({'exception.message', 'exception.stacktrace'} <= set(attributes) for _, attributes in events)
        assert 'secret' in events[0][1]['exception.message']
        assert descriptions == [IsStr(regex=r'\w+: [\s\S]+')] * 2
    else:
        assert all(set(attributes) == {'exception.type', 'exception.escaped'} for _, attributes in events)
        assert 'secret' not in str(spans)
        assert descriptions == []


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
@pytest.mark.anyio
async def test_run_span_records_failures_from_its_own_finalization(capfire: CaptureLogfire) -> None:
    """The run span's finalization is inside the scope `use_span` used to cover.

    `record_exception=False` / `set_status_on_exception=False` switch the SDK's recording off for
    the whole span, so the stand-in has to wrap the whole span body rather than just the run. With
    warnings raised as errors the end-of-run mutation warning comes out of the `finally`, and the
    span still has to end up ERROR with the exception recorded on it.
    """

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('corrupt_history', {}, tool_call_id='call_1')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(model=FunctionModel(model_function), capabilities=[Instrumentation()])

    @agent.tool
    async def corrupt_history(ctx: RunContext) -> str:
        first_part = ctx.messages[0].parts[0]
        assert isinstance(first_part, UserPromptPart)
        first_part.content = 'mutated prompt'
        return 'ok'

    with warnings.catch_warnings():
        warnings.simplefilter('error', MessageHistoryMutatedWarning)
        with pytest.raises(MessageHistoryMutatedWarning):
            await agent.run('original prompt')

    [run_span] = [
        span
        for span in capfire.exporter.exported_spans
        # Logfire exports a pending span alongside the real one; only the latter carries the status.
        if span.name.startswith('invoke_agent') and (span.attributes or {}).get('logfire.span_type') != 'pending_span'
    ]
    assert run_span.status.status_code is StatusCode.ERROR
    assert [event.name for event in run_span.events] == ['exception']


@pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
def test_exception_recording_skipped_when_span_is_not_recording() -> None:
    """`use_span` left a non-recording span's exception alone, and so does the stand-in.

    The SDK formats the traceback before `add_event` discards it, so recording on a sampled-out
    span would surface a `__str__` failure in place of the error that actually happened.
    """

    class Unformattable(Exception):
        def __str__(self) -> str:
            raise RuntimeError('formatting blew up')

    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise Unformattable

    agent = Agent(
        FunctionModel(boom),
        capabilities=[
            Instrumentation(settings=InstrumentationSettings(tracer_provider=TracerProvider(sampler=ALWAYS_OFF)))
        ],
    )
    with pytest.raises(Unformattable):
        agent.run_sync('hello')
