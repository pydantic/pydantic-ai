from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from typing import Any, ClassVar, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolset, ModelResponse, RunContext, TextPart
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC, JSON_CODEC
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    Lifecycle,
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _ToolFailed,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
    _ValidationError,  # pyright: ignore[reportPrivateUsage]
    _ValidationErrorDetail,  # pyright: ignore[reportPrivateUsage]
    unwrap_recorded_tool_call_result,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

JOURNAL_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request.registered',
    'compat__model.request_stream',
    'compat__model.request_stream.registered',
    'compat__model.cancel_suspended_response',
    'compat__model.cancel_suspended_response.registered',
    'compat__event_stream_handler',
    'compat__function_toolset__functions.call_tool:function_tool',
    'compat__function_toolset__functions.validate_args',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
}

PREFECT_OPERATION_NAMES = {
    'Model Request: test',
    'Model Request (Streaming): test',
    'Cancel Suspended Response: test',
    'Handle Stream Event',
    'Call Tool: function_tool',
    'Validate Tool Args: function_tool',
    'Call MCP Tool: mcp_tool',
    'Call Tool: dynamic_tool',
    'Validate Tool Args: dynamic_tool',
}

TEMPORAL_ACTIVITY_NAMES = {
    'agent__compat__model_request',
    'agent__compat__model_request_stream',
    'agent__compat__model_cancel_suspended_response',
    'agent__compat__event_stream_handler',
    'agent__compat__toolset__<agent>__call_tool',
    'agent__compat__toolset__<agent>__validate_args',
    'agent__compat__toolset__functions__call_tool',
    'agent__compat__toolset__functions__validate_args',
    'agent__compat__mcp_server__mcp__get_tools',
    'agent__compat__mcp_server__mcp__get_instructions',
    'agent__compat__mcp_server__mcp__call_tool',
    'agent__compat__dynamic_toolset__dynamic__get_tools',
    'agent__compat__dynamic_toolset__dynamic__call_tool',
    'agent__compat__dynamic_toolset__dynamic__validate_args',
}

DBOS_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request_stream',
    'compat__model.cancel_suspended_response',
    'compat__event_stream_handler',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
}


class JournalDurability(BaseDurabilityCapability[Any]):
    engine_name = 'Journal compatibility stub'
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar = frozenset()
    _wrapped_toolset_kinds: ClassVar = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    _durable_unit_noun = 'unit'
    _durable_container_noun = 'journal'

    @property
    def in_durable_context(self) -> bool:
        return True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorded_names: list[str] = []

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        self.recorded_names.append(name)
        return await fn()


async def test_journal_operation_name_assembly_sequence() -> None:
    async def function_tool() -> str:
        return 'function'

    async def dynamic_tool() -> str:
        return 'dynamic'

    function_toolset = FunctionToolset(tools=[function_tool], id='functions')
    dynamic_toolset = DynamicToolset(lambda _: FunctionToolset(tools=[dynamic_tool]), id='dynamic')
    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=[function_toolset, dynamic_toolset],
        capabilities=[JournalDurability(event_stream_handler=_event_handler)],
    )

    result = await agent.run('Call every tool')

    assert result.output == '{"function_tool":"function","dynamic_tool":"dynamic"}'
    durability = JournalDurability.from_agent(agent)
    assert durability is not None
    assert durability.recorded_names == [
        'compat__dynamic_toolset__dynamic.get_tools',
        'compat__model.request_stream',
        'compat__event_stream_handler',
        'compat__event_stream_handler',
        'compat__function_toolset__functions.call_tool:function_tool',
        'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool',
        'compat__event_stream_handler',
        'compat__event_stream_handler',
        'compat__dynamic_toolset__dynamic.get_tools',
        'compat__model.request_stream',
    ]
    assert set(durability.recorded_names) <= JOURNAL_OPERATION_NAMES


def test_default_journal_operation_name_matrix() -> None:
    durability = JournalDurability(name='compat')
    names = {
        durability._unit_name('model.request'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('model.request', suffix='.registered'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('model.request_stream'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('model.request_stream', suffix='.registered'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('model.cancel_suspended_response'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.cancel_suspended_response', suffix='.registered'
        ),
        durability._unit_name('event_stream_handler'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'function_toolset', prefix='compat__function_toolset__functions', tool_name='function_tool'
        ),
        'compat__function_toolset__functions.validate_args',
        durability._unit_name('mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_tools'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_instructions'
        ),
        durability._unit_name('mcp_server', prefix='compat__mcp_server__mcp', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', suffix='.get_tools'
        ),
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', tool_name='dynamic_tool'
        ),
        'compat__dynamic_toolset__dynamic.validate_args',
    }
    assert names == JOURNAL_OPERATION_NAMES


def test_prefect_operation_name_matrix() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec.prefect import PrefectDurability

    durability = PrefectDurability(name='compat')
    names = {
        durability._unit_name('model.request', label='Model Request', model_name='test'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.request_stream', label='Model Request (Streaming)', model_name='test'
        ),
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.cancel_suspended_response', label='Cancel Suspended Response', model_name='test'
        ),
        durability._unit_name('event_stream_handler', label='Handle Stream Event'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('function_toolset', label='Call Tool', tool_name='function_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('function_toolset', label='Validate Tool Args', tool_name='function_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('mcp_server', label='Call MCP Tool', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('dynamic_toolset', label='Call Tool', tool_name='dynamic_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('dynamic_toolset', label='Validate Tool Args', tool_name='dynamic_tool'),  # pyright: ignore[reportPrivateUsage]
    }
    assert names == PREFECT_OPERATION_NAMES


def test_prefect_operation_name_assembly_completeness() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableFunctionToolset, DurableMCPToolset
    from pydantic_ai.durable_exec.prefect import PrefectDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[PrefectDurability(event_stream_handler=_event_handler)],
    )
    durability = PrefectDurability.from_agent(agent)
    assert durability is not None
    assert {type(toolset) for toolset in durability._toolsets_by_id.values()} >= {  # pyright: ignore[reportPrivateUsage]
        DurableFunctionToolset,
        DurableMCPToolset,
        DurableDynamicToolset,
    }
    assembled_names = {
        durability._unit_name('model.request', label='Model Request', model_name='test'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.request_stream', label='Model Request (Streaming)', model_name='test'
        ),
        durability._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.cancel_suspended_response', label='Cancel Suspended Response', model_name='test'
        ),
        durability._unit_name('event_stream_handler', label='Handle Stream Event'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('function_toolset', label='Call Tool', tool_name='function_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('function_toolset', label='Validate Tool Args', tool_name='function_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('mcp_server', label='Call MCP Tool', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('dynamic_toolset', label='Call Tool', tool_name='dynamic_tool'),  # pyright: ignore[reportPrivateUsage]
        durability._unit_name('dynamic_toolset', label='Validate Tool Args', tool_name='dynamic_tool'),  # pyright: ignore[reportPrivateUsage]
    }
    assert assembled_names == PREFECT_OPERATION_NAMES


def test_dbos_operation_name_matrix_and_assembly_completeness() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableMCPToolset
    from pydantic_ai.durable_exec.dbos import DBOSDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[DBOSDurability(event_stream_handler=_event_handler)],
    )
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None
    assert {type(toolset) for toolset in durability._toolsets_by_id.values()} >= {  # pyright: ignore[reportPrivateUsage]
        DurableMCPToolset,
        DurableDynamicToolset,
    }
    backend = durability._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    registered_names = {cast(Any, registration).dbos_function_name for registration in backend.registrations()}
    assert registered_names == DBOS_OPERATION_NAMES


def _synthetic_toolsets() -> tuple[FunctionToolset[Any], DynamicToolset[Any], Any]:
    pytest.importorskip('mcp')
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset

    async def function_tool() -> str:
        return 'function'

    function_toolset = FunctionToolset(id='functions')
    function_toolset.add_function(function_tool)
    dynamic_toolset = DynamicToolset(lambda _: FunctionToolset(tools=[function_tool]), id='dynamic')
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp')
    return function_toolset, dynamic_toolset, mcp_toolset


async def _event_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass


def test_temporal_activity_name_matrix_and_assembly_completeness() -> None:
    pytest.importorskip('temporalio')
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.temporal import TemporalDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[TemporalDurability(event_stream_handler=_event_handler)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    names = {
        ActivityDefinition.must_from_callable(item).name  # pyright: ignore[reportUnknownMemberType]
        for item in durability.temporal_activities
    }
    assert names == TEMPORAL_ACTIVITY_NAMES


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (_ToolReturn('ok'), {'result': 'ok', 'kind': 'tool_return'}),
        (_ApprovalRequired({'scope': 'write'}), {'metadata': {'scope': 'write'}, 'kind': 'approval_required'}),
        (_CallDeferred({'ticket': 7}), {'metadata': {'ticket': 7}, 'kind': 'call_deferred'}),
        (_ModelRetry('retry me'), {'message': 'retry me', 'kind': 'model_retry'}),
        (
            _ValidationError('int', [_ValidationErrorDetail('int_parsing', ['value'], 'bad integer', 'x')]),
            {
                'title': 'int',
                'errors': [{'type': 'int_parsing', 'loc': ['value'], 'msg': 'bad integer', 'input': 'x'}],
                'kind': 'validation_error',
            },
        ),
        (_ToolFailed('failed'), {'message': 'failed', 'kind': 'tool_failed'}),
    ],
)
def test_call_tool_result_json_payload_goldens(value: CallToolResult, expected: dict[str, Any]) -> None:
    assert JSON_CODEC.dump(CallToolResult, value) == expected
    assert IDENTITY_CODEC.dump(CallToolResult, value) is value
    assert IDENTITY_CODEC.load(CallToolResult, value) is value


@pytest.mark.parametrize(
    ('tp', 'value', 'expected'),
    [
        (
            ModelResponse,
            ModelResponse(parts=[TextPart('hello')]),
            {
                'parts': [
                    {
                        'content': 'hello',
                        'id': None,
                        'provider_name': None,
                        'provider_details': None,
                        'part_kind': 'text',
                    }
                ],
                'usage': {
                    'input_tokens': 0,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 0,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                    'cost': None,
                },
                'model_name': None,
                'timestamp': '2020-01-01T00:00:00Z',
                'kind': 'response',
                'provider_name': None,
                'provider_url': None,
                'provider_details': None,
                'provider_response_id': None,
                'finish_reason': None,
                'run_id': None,
                'conversation_id': None,
                'metadata': None,
                'state': 'complete',
            },
        ),
        (
            dict[str, ToolDefinition],
            {'tool': ToolDefinition(name='tool', description='Do it', parameters_json_schema={'type': 'object'})},
            {
                'tool': {
                    'name': 'tool',
                    'parameters_json_schema': {'type': 'object'},
                    'description': 'Do it',
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                    'timeout': None,
                    'defer_loading': False,
                    'unless_native': None,
                    'with_native': None,
                    'tool_kind': None,
                    'return_schema': None,
                    'include_return_schema': None,
                    'toolset_id': None,
                    'capability_id': None,
                }
            },
        ),
        (str | None, 'instructions', 'instructions'),
        (str | None, None, None),
        (type(None), None, None),
    ],
)
def test_json_and_identity_codec_payload_goldens(tp: Any, value: Any, expected: Any) -> None:
    if isinstance(value, ModelResponse):
        value.timestamp = value.timestamp.replace(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    assert JSON_CODEC.dump(tp, value) == expected
    assert IDENTITY_CODEC.dump(tp, value) is value
    assert IDENTITY_CODEC.load(tp, value) is value


def test_pre_wrapper_tool_result_upgrade_paths() -> None:
    raw_payload = {'answer': 42}
    assert unwrap_recorded_tool_call_result(raw_payload) is raw_payload
    with pytest.raises(ValidationError):
        JSON_CODEC.load(CallToolResult, raw_payload)


async def test_validation_error_crosses_call_tool_result_boundary() -> None:
    async def invalid() -> None:
        TypeAdapter(int).validate_python('not-an-int')

    payload = await wrap_tool_call_result(invalid())
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)
