from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import pytest
from typing_extensions import assert_never

from pydantic_ai import Agent, AgentStreamEvent, FunctionToolset, ModelResponse, RunContext, TextPart
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import (
    CacheIdentity,
    CallToolId,
    CancelSuspendedResponseId,
    DurableOperation,
    DurableOperationId,
    EventStreamHandlerId,
    GetInstructionsId,
    GetToolsId,
    IdentityParameterTransport,
    ModelRequestId,
    NoCacheIdentity,
    OperationConfigRole,
    OperationInvocation,
    TypedResultCodec,
    ValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import (
    DBOSOperationNamer,
    DurableOperationNamer,
    JournalOperationNamer,
    PrefectOperationNamer,
    TemporalOperationNamer,
)
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    Lifecycle,
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _ToolFailed,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models.test import TestModel
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
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool',
}
PREFECT_OPERATION_NAMES = {
    'Model Request: test',
    'Model Request (Streaming): test',
    'Cancel Suspended Response: test',
    'Handle Stream Event',
    'Call Tool: function_tool',
    'Call MCP Tool: mcp_tool',
    'Call Tool: dynamic_tool',
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
}
TEMPORAL_ACTIVITY_NAMES = {
    'agent__compat__model_request',
    'agent__compat__model_request_stream',
    'agent__compat__model_cancel_suspended_response',
    'agent__compat__event_stream_handler',
    'agent__compat__toolset__<agent>__call_tool',
    'agent__compat__toolset__functions__call_tool',
    'agent__compat__mcp_server__mcp__get_tools',
    'agent__compat__mcp_server__mcp__get_instructions',
    'agent__compat__mcp_server__mcp__call_tool',
    'agent__compat__dynamic_toolset__dynamic__get_tools',
    'agent__compat__dynamic_toolset__dynamic__call_tool',
}


class JournalDurability(BaseDurabilityCapability[Any]):
    engine_name = 'Journal operation test stub'
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar = frozenset()
    _wrapped_toolset_kinds: ClassVar = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }

    @property
    def in_durable_context(self) -> bool:
        return True

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        return await fn()


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


@dataclass(frozen=True)
class _ToolParams:
    name: str


def _ids() -> list[DurableOperationId]:
    return [
        ModelRequestId(None, False, 'test'),
        ModelRequestId('registered', False, 'test'),
        ModelRequestId(None, True, 'test'),
        ModelRequestId('registered', True, 'test'),
        CancelSuspendedResponseId(None, 'test'),
        CancelSuspendedResponseId('registered', 'test'),
        EventStreamHandlerId(),
        GetToolsId('mcp', 'mcp'),
        GetInstructionsId('mcp'),
        CallToolId('function', 'functions'),
        CallToolId('mcp', 'mcp'),
        GetToolsId('dynamic', 'dynamic'),
        CallToolId('dynamic', 'dynamic'),
    ]


def _params(operation_id: DurableOperationId) -> object:
    if isinstance(operation_id, CallToolId):
        names = {'function': 'function_tool', 'mcp': 'mcp_tool', 'dynamic': 'dynamic_tool'}
        return _ToolParams(names[operation_id.toolset_kind])
    return object()


def test_journal_name_parity_with_live_old_implementation_and_table() -> None:
    old = JournalDurability(name='compat')
    live = [
        old._unit_name('model.request'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('model.request', suffix='.registered'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('model.request_stream'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('model.request_stream', suffix='.registered'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('model.cancel_suspended_response'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('model.cancel_suspended_response', suffix='.registered'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('event_stream_handler'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_tools'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_instructions'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name(  # pyright: ignore[reportPrivateUsage]
            'function_toolset', prefix='compat__function_toolset__functions', tool_name='function_tool'
        ),
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', suffix='.get_tools'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name(  # pyright: ignore[reportPrivateUsage]
            'dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', tool_name='dynamic_tool'
        ),
    ]
    namer = JournalOperationNamer('compat')
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in _ids()]
    assert actual == live
    assert set(actual) == JOURNAL_OPERATION_NAMES


def test_prefect_name_parity_with_live_old_implementation_and_table() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec.prefect import PrefectDurability

    old = PrefectDurability(name='compat')
    ids = [*_ids()[:1], _ids()[2], _ids()[4], _ids()[6], _ids()[9], _ids()[10], _ids()[12]]
    live = [
        old._unit_name('model.request', label='Model Request', model_name='test'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.request_stream', label='Model Request (Streaming)', model_name='test'
        ),
        old._unit_name(  # pyright: ignore[reportPrivateUsage]
            'model.cancel_suspended_response', label='Cancel Suspended Response', model_name='test'
        ),
        old._unit_name('event_stream_handler', label='Handle Stream Event'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('function_toolset', label='Call Tool', tool_name='function_tool'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', label='Call MCP Tool', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('dynamic_toolset', label='Call Tool', tool_name='dynamic_tool'),  # pyright: ignore[reportPrivateUsage]
    ]
    namer = PrefectOperationNamer()
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in ids]
    assert actual == live
    assert set(actual) == PREFECT_OPERATION_NAMES


def test_dbos_name_parity_with_live_old_implementation_and_table() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec.dbos import DBOSDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[DBOSDurability(event_stream_handler=_event_handler)],
    )
    old = DBOSDurability.from_agent(agent)
    assert old is not None
    live = [
        old._request_step.dbos_function_name,  # pyright: ignore[reportPrivateUsage]
        old._request_stream_step.dbos_function_name,  # pyright: ignore[reportPrivateUsage]
        old._cancel_suspended_response_step.dbos_function_name,  # pyright: ignore[reportPrivateUsage]
        old._event_stream_handler_step.dbos_function_name,  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_tools'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', suffix='.get_instructions'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('mcp_server', prefix='compat__mcp_server__mcp', tool_name='mcp_tool'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name('dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', suffix='.get_tools'),  # pyright: ignore[reportPrivateUsage]
        old._unit_name(  # pyright: ignore[reportPrivateUsage]
            'dynamic_toolset', prefix='compat__dynamic_toolset__dynamic', tool_name='dynamic_tool'
        ),
    ]
    ids = [_ids()[0], _ids()[2], _ids()[4], _ids()[6], _ids()[7], _ids()[8], _ids()[10], _ids()[11], _ids()[12]]
    namer = DBOSOperationNamer('compat')
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in ids]
    assert actual == live
    assert set(actual) == DBOS_OPERATION_NAMES
    assert namer.operation_name(ModelRequestId('registered', False, 'test')) == live[0]


def test_temporal_name_parity_with_live_registered_activities_and_table() -> None:
    pytest.importorskip('temporalio')
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.temporal import TemporalDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[TemporalDurability(event_stream_handler=_event_handler)],
    )
    old = TemporalDurability.from_agent(agent)
    assert old is not None
    live = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in old.temporal_activities
    }
    ids: list[DurableOperationId] = [
        ModelRequestId(None, False, 'test'),
        ModelRequestId(None, True, 'test'),
        CancelSuspendedResponseId(None, 'test'),
        EventStreamHandlerId(),
        CallToolId('function', '<agent>'),
        CallToolId('function', 'functions'),
        GetToolsId('mcp', 'mcp'),
        GetInstructionsId('mcp'),
        CallToolId('mcp', 'mcp'),
        GetToolsId('dynamic', 'dynamic'),
        CallToolId('dynamic', 'dynamic'),
    ]
    namer = TemporalOperationNamer('compat')
    actual = {namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in ids}
    assert actual == live
    assert actual == TEMPORAL_ACTIVITY_NAMES


def _exhaustive_identity(operation_id: DurableOperationId) -> str:
    match operation_id:
        case ModelRequestId():
            return 'model'
        case CancelSuspendedResponseId():
            return 'cancel'
        case EventStreamHandlerId():
            return 'event'
        case GetToolsId():
            return 'tools'
        case GetInstructionsId():
            return 'instructions'
        case ValidateToolArgumentsId():
            return 'validation'
        case CallToolId():
            return 'call'
    assert_never(operation_id)


def test_operation_identity_union_is_exhaustively_constructible() -> None:
    identities: list[DurableOperationId] = [
        ModelRequestId(None, False, 'model'),
        CancelSuspendedResponseId(None, 'model'),
        EventStreamHandlerId(),
        GetToolsId('function', 'tools'),
        GetInstructionsId('mcp'),
        ValidateToolArgumentsId('dynamic', 'dynamic'),
        CallToolId('mcp', 'mcp'),
    ]
    assert [_exhaustive_identity(operation_id) for operation_id in identities] == [
        'model',
        'cancel',
        'event',
        'tools',
        'instructions',
        'validation',
        'call',
    ]


@pytest.mark.parametrize(
    'namer',
    [
        JournalOperationNamer('agent'),
        PrefectOperationNamer(),
        DBOSOperationNamer('agent'),
        TemporalOperationNamer('agent'),
    ],
)
def test_validation_names_wait_for_pr_6906(namer: DurableOperationNamer) -> None:
    with pytest.raises(RuntimeError, match=r'pinned by PR #6906 integration'):
        namer.operation_name(ValidateToolArgumentsId('function', 'tools'))


def test_namer_error_paths_and_unrepresented_formats() -> None:
    with pytest.raises(TypeError, match='must expose'):
        JournalOperationNamer('agent').invocation_name(CallToolId('function', 'tools'), object())
    prefect = PrefectOperationNamer()
    with pytest.raises(RuntimeError, match='do not have durable unit names'):
        prefect.operation_name(GetToolsId('dynamic', 'tools'))
    with pytest.raises(RuntimeError, match='do not have durable unit names'):
        prefect.operation_name(GetInstructionsId('mcp'))
    assert (
        JournalOperationNamer('agent').operation_name(GetToolsId('function', 'tools'))
        == 'agent__function_toolset__tools.get_tools'
    )


@pytest.mark.parametrize(
    'value',
    [
        _ToolReturn('ok'),
        _ApprovalRequired({'scope': 'write'}),
        _CallDeferred({'ticket': 7}),
        _ModelRetry('retry'),
        _ToolFailed('failed'),
    ],
)
def test_typed_result_codec_matches_json_codec(value: CallToolResult) -> None:
    codec = TypedResultCodec[CallToolResult](CallToolResult)
    payload = codec.dump(value)
    assert payload == JSON_CODEC.dump(CallToolResult, value)
    assert codec.load(payload) == JSON_CODEC.load(CallToolResult, payload)


def test_typed_result_codec_json_model_and_identity_parity() -> None:
    value = ModelResponse(parts=[TextPart('hello')])
    json_codec = TypedResultCodec[ModelResponse](ModelResponse)
    payload = json_codec.dump(value)
    assert payload == JSON_CODEC.dump(ModelResponse, value)
    assert json_codec.load(payload) == JSON_CODEC.load(ModelResponse, payload)
    identity_codec = TypedResultCodec[ModelResponse](ModelResponse, mode='identity')
    assert identity_codec.dump(value) is value
    assert identity_codec.load(value) is value


class _Config:
    def __init__(self) -> None:
        self.base_calls: list[tuple[OperationConfigRole, DurableOperationId]] = []

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> dict[str, str]:
        self.base_calls.append((role, operation_id))
        return {'source': 'role-default'}

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> dict[str, str] | Literal[False]:
        return False if tool is None else {'tool': tool_name}


class _CacheIdentity(CacheIdentity[int]):
    def project(self, params: int) -> object:
        return ('cache', params)


class _RecordingBackend(CallableOperationBackend[dict[str, str]]):
    def __init__(self, config: _Config) -> None:
        super().__init__(namer=JournalOperationNamer('agent'), config=config)
        self.calls: list[tuple[str, object, object]] = []

    async def _execute(
        self, *, name: str, body: Callable[[], Awaitable[object]], cache_key: object, config: object
    ) -> object:
        self.calls.append((name, cache_key, config))
        return await body()


async def test_callable_operation_backend_resolves_and_round_trips() -> None:
    async def handler(params: int) -> int:
        return params + 1

    config = _Config()
    backend = _RecordingBackend(config)
    operation = DurableOperation(
        operation_id=ModelRequestId('registered', False, 'model'),
        handler=handler,
        parameter_transport=IdentityParameterTransport[int](),
        cache_identity=_CacheIdentity(),
        result_codec=TypedResultCodec[int](int),
        config_role=OperationConfigRole.MODEL,
    )
    bound = backend.bind(operation)
    assert bound.operation is operation
    assert await bound(4) == 5
    assert backend.calls == [('agent__model.request.registered', ('cache', 4), {'source': 'role-default'})]
    assert config.base_calls == [(OperationConfigRole.MODEL, operation.operation_id)]
    assert await bound(8, config={'source': 'explicit'}) == 9
    assert backend.calls[-1] == ('agent__model.request.registered', ('cache', 8), {'source': 'explicit'})
    assert len(config.base_calls) == 1
    assert backend.registrations() == ()
    assert config.for_tool(OperationConfigRole.TOOL_CALL, CallToolId('function', 'tools'), None, 'tool') is False
    assert config.for_tool(OperationConfigRole.TOOL_CALL, CallToolId('function', 'tools'), object(), 'tool') == {
        'tool': 'tool'
    }


def test_trivial_transport_cache_and_invocation_helpers() -> None:
    value = object()
    transport = IdentityParameterTransport[object]()
    assert transport.dump(value) is value
    assert transport.load(value, runtime=object()) is value
    assert NoCacheIdentity[object]().project(value) is None
    invocation = OperationInvocation(params=value, config='config')
    assert invocation.params is value
    assert invocation.config == 'config'
