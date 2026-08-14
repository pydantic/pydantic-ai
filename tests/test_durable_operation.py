from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast

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
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend, LegacyCallableBackend
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
    _durable_unit_noun = 'unit'
    _durable_container_noun = 'journal'
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[tuple[str, tuple[Any, ...], Any]] = []

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        self.calls.append((name, inputs, config))
        return await fn()


class _OverrideNameDurability(JournalDurability):
    def _unit_name(self, kind: str, **parts: Any) -> str:
        return f'override:{kind}:{parts["label"]}'


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
    assert old._model_id_suffix('registered') == ''  # pyright: ignore[reportPrivateUsage]
    assert old._legacy_operation_name(CancelSuspendedResponseId(None, 'test')) == 'Cancel Suspended Response: test'  # pyright: ignore[reportPrivateUsage]


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
    backend = old._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    live = {cast(Any, registration).dbos_function_name for registration in backend.registrations()}
    ids = [_ids()[0], _ids()[2], _ids()[4], _ids()[6], _ids()[7], _ids()[8], _ids()[10], _ids()[11], _ids()[12]]
    namer = DBOSOperationNamer('compat')
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in ids]
    assert set(actual) == live
    assert set(actual) == DBOS_OPERATION_NAMES
    assert namer.operation_name(ModelRequestId('registered', False, 'test')) in live


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


def test_temporal_backend_preserves_sdk_visible_activity_definitions() -> None:
    pytest.importorskip('temporalio')
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.durable_exec.temporal._durability import (
        _CancelParams,  # pyright: ignore[reportPrivateUsage]
        _EventStreamHandlerParams,  # pyright: ignore[reportPrivateUsage]
        _RequestParams,  # pyright: ignore[reportPrivateUsage]
        _StreamedActivityPayload,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._toolset import toolset_temporal_activities

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[TemporalDurability(event_stream_handler=_event_handler)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    backend = durability._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None

    # Rebuild the pre-backend registration list through the old ownership path: the four
    # capability activities followed by each wrapped toolset's activities.
    legacy_registrations = [
        durability.request_activity,
        durability.request_stream_activity,
        durability.event_stream_handler_activity,
        durability.cancel_suspended_response_activity,
    ]
    for wrapped in durability._toolsets_by_id.values():  # pyright: ignore[reportPrivateUsage]
        legacy_registrations.extend(toolset_temporal_activities(wrapped))

    registrations = list(backend.registrations())
    assert registrations == legacy_registrations

    def sdk_definition(item: Callable[..., object]) -> tuple[str | None, inspect.Signature]:
        definition = ActivityDefinition.must_from_callable(item)  # pyright: ignore[reportUnknownMemberType]
        fn = cast(Callable[..., object], definition.fn)  # pyright: ignore[reportUnknownMemberType]
        return definition.name, inspect.signature(fn)

    assert [sdk_definition(item) for item in registrations] == [sdk_definition(item) for item in legacy_registrations]

    expected_signatures = {
        durability.request_activity: (_RequestParams, ModelResponse),
        durability.request_stream_activity: (_RequestParams, _StreamedActivityPayload),
        durability.event_stream_handler_activity: (_EventStreamHandlerParams, type(None)),
        durability.cancel_suspended_response_activity: (_CancelParams, type(None)),
    }
    for activity_fn, (params_type, result_type) in expected_signatures.items():
        signature = inspect.signature(activity_fn)
        assert signature.parameters['params'].annotation is params_type
        assert signature.parameters['deps'].annotation == agent.deps_type | None
        assert signature.parameters['deps'].default is None
        assert signature.return_annotation == result_type


async def test_temporal_backend_binds_existing_positional_activity() -> None:
    pytest.importorskip('temporalio')
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal._operation_backend import (
        TemporalOperationBackend,
        TemporalOperationConfig,
    )

    model_config: ActivityConfig = {'summary': 'model'}
    event_config: ActivityConfig = {'summary': 'event'}
    tool_config: ActivityConfig = {'summary': 'tool'}
    config = TemporalOperationConfig(model=model_config, event=event_config, tool=tool_config)
    assert config.base(OperationConfigRole.TOOL_CALL, CallToolId('function', 'tools')) is tool_config
    assert config.for_tool(OperationConfigRole.TOOL_CALL, CallToolId('function', 'tools'), None, 'tool') is tool_config

    backend = TemporalOperationBackend(
        agent_name='compat',
        deps_type=int,
        model_config=model_config,
        event_config=event_config,
        tool_config=tool_config,
    )

    async def existing_activity(params: str, deps: int | None = None) -> str:
        return f'{params}:{deps}'

    bound = backend.register_activity(
        existing_activity,
        operation_id=ModelRequestId(None, False, 'test'),
        config_role=OperationConfigRole.MODEL,
    )
    assert bound.operation.operation_id == ModelRequestId(None, False, 'test')
    assert await bound.operation.handler(('payload', 42)) == 'payload:42'
    assert await bound(('payload', 42)) == 'payload:42'
    assert backend.config_for_tool(bound.operation, None, 'tool') is tool_config


async def test_temporal_backend_dispatches_cancel_with_legacy_contextless_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip('temporalio')
    from pydantic_ai.durable_exec._base import CancelSuspendedResponseOperationParams
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.durable_exec.temporal._durability import _CancelParams  # pyright: ignore[reportPrivateUsage]

    agent = Agent(TestModel(), name='cancel-dispatch', capabilities=[TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None

    dispatched: list[tuple[Callable[..., object], Sequence[object], object]] = []

    async def execute_activity(activity: Callable[..., object], *, args: Sequence[object], **config: object) -> None:
        dispatched.append((activity, args, config['summary']))

    monkeypatch.setattr(
        'pydantic_ai.durable_exec.temporal._operation_backend.execute_activity',
        execute_activity,
    )
    operations = durability._bound_model_operations  # pyright: ignore[reportPrivateUsage]
    assert operations is not None
    cancel_operation = operations[2]
    response = ModelResponse(parts=[TextPart('cancel')])
    await cancel_operation(CancelSuspendedResponseOperationParams(None, response, None))

    assert dispatched == [
        (
            durability.cancel_suspended_response_activity,
            (_CancelParams(response=response), None),
            'cancel suspended response: test:test',
        )
    ]

    with pytest.raises(RuntimeError, match='requires a serialized run context'):
        await JournalDurability()._cancel_suspended_response_operation(  # pyright: ignore[reportPrivateUsage]
            CancelSuspendedResponseOperationParams(None, response, None)
        )


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
    def project(self, params: int) -> tuple[object, ...]:
        return ('cache', params)


class _RecordingBackend(CallableOperationBackend[dict[str, str]]):
    def __init__(self, config: _Config) -> None:
        super().__init__(namer=JournalOperationNamer('agent'), config=config)
        self.calls: list[tuple[str, object, object]] = []

    async def _execute(
        self, *, name: str, body: Callable[[], Awaitable[object]], cache_key: tuple[object, ...], config: object
    ) -> object:
        self.calls.append((name, cache_key, config))
        return await body()


class _RecordingLegacyCapability:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], object, object]] = []

    async def run_durable_unit(
        self,
        name: str,
        fn: Callable[[], Awaitable[object]],
        *,
        inputs: tuple[object, ...],
        config: object,
    ) -> object:
        payload = await fn()
        self.calls.append((name, inputs, config, payload))
        return payload


@dataclass(frozen=True)
class _DispatchParams:
    inputs: tuple[object, ...]
    name: str = ''


class _LogicalInputs(CacheIdentity[_DispatchParams]):
    def project(self, params: _DispatchParams) -> tuple[object, ...]:
        return params.inputs


@dataclass(frozen=True)
class _LegacyCase:
    operation_id: DurableOperationId
    role: OperationConfigRole
    params: _DispatchParams
    result: object
    result_type: object


_CTX = object()
_TOOL = object()
_MODEL_MESSAGES = ['request']
_MODEL_SETTINGS = {'temperature': 0}
_MODEL_PARAMETERS = {'allow_text_output': True}
_MODEL_RESPONSE = ModelResponse(parts=[TextPart('hello')])
_TOOL_ARGS = {'value': 1}


@pytest.mark.parametrize(
    'case',
    [
        _LegacyCase(
            ModelRequestId(None, False, 'test'),
            OperationConfigRole.MODEL,
            _DispatchParams((None, _MODEL_MESSAGES, _MODEL_SETTINGS, _MODEL_PARAMETERS, _CTX)),
            _MODEL_RESPONSE,
            ModelResponse,
        ),
        _LegacyCase(
            ModelRequestId(None, True, 'test'),
            OperationConfigRole.MODEL,
            _DispatchParams((None, _MODEL_MESSAGES, _MODEL_SETTINGS, _MODEL_PARAMETERS, _CTX)),
            {'response': 'streamed', 'events': []},
            object,
        ),
        _LegacyCase(
            CancelSuspendedResponseId(None, 'test'),
            OperationConfigRole.MODEL,
            _DispatchParams((None, _MODEL_RESPONSE, _CTX)),
            None,
            type(None),
        ),
        _LegacyCase(
            EventStreamHandlerId(),
            OperationConfigRole.EVENT,
            _DispatchParams(({'event': 'part-start'},)),
            None,
            type(None),
        ),
        _LegacyCase(
            CallToolId('function', 'functions'),
            OperationConfigRole.TOOL_CALL,
            _DispatchParams(('function_tool', _TOOL_ARGS, _CTX, _TOOL), 'function_tool'),
            _ToolReturn('ok'),
            CallToolResult,
        ),
        _LegacyCase(
            GetToolsId('dynamic', 'dynamic'),
            OperationConfigRole.TOOL_DISCOVERY,
            _DispatchParams((_CTX,)),
            {'tool': 'definition'},
            object,
        ),
        _LegacyCase(
            CallToolId('dynamic', 'dynamic'),
            OperationConfigRole.TOOL_CALL,
            _DispatchParams(('dynamic_tool', _TOOL_ARGS, _CTX), 'dynamic_tool'),
            _ToolReturn('ok'),
            CallToolResult,
        ),
        _LegacyCase(
            GetToolsId('mcp', 'mcp'),
            OperationConfigRole.TOOL_DISCOVERY,
            _DispatchParams((_CTX,)),
            {'mcp_tool': 'definition'},
            object,
        ),
        _LegacyCase(
            GetInstructionsId('mcp'),
            OperationConfigRole.TOOL_DISCOVERY,
            _DispatchParams((_CTX,)),
            'instructions',
            str | None,
        ),
        _LegacyCase(
            CallToolId('mcp', 'mcp'),
            OperationConfigRole.TOOL_CALL,
            _DispatchParams(('mcp_tool', _TOOL_ARGS, _CTX, _TOOL), 'mcp_tool'),
            _ToolReturn('ok'),
            CallToolResult,
        ),
    ],
    ids=[
        'model-request',
        'model-request-stream',
        'cancel-suspended-response',
        'event-stream-handler',
        'function-tool-call',
        'dynamic-get-tools',
        'dynamic-call',
        'mcp-get-tools',
        'mcp-get-instructions',
        'mcp-call',
    ],
)
async def test_legacy_callable_backend_dispatch_parity(case: _LegacyCase) -> None:
    handled: list[_DispatchParams] = []

    async def handler(params: _DispatchParams) -> object:
        handled.append(params)
        return case.result

    config = _Config()
    capability = _RecordingLegacyCapability()
    namer = JournalOperationNamer('compat')
    operation = DurableOperation(
        operation_id=case.operation_id,
        handler=handler,
        parameter_transport=IdentityParameterTransport[_DispatchParams](),
        cache_identity=_LogicalInputs(),
        result_codec=TypedResultCodec[object](
            case.result_type, mode='identity' if case.result_type is object else 'json'
        ),
        config_role=case.role,
    )
    bound = LegacyCallableBackend(capability, namer=namer, config=config).bind(operation)

    assert bound.operation is operation
    assert await bound(case.params) == case.result
    name = namer.invocation_name(case.operation_id, case.params).operation_name
    assert capability.calls == [(name, case.params.inputs, {'source': 'role-default'}, capability.calls[0][3])]
    assert handled == [case.params]
    assert config.base_calls == [(case.role, case.operation_id)]
    if isinstance(case.operation_id, CallToolId):
        assert capability.calls[0][3] == {'result': 'ok', 'kind': 'tool_return'}

    assert await bound(case.params, config={'source': 'explicit'}) == case.result
    assert capability.calls[-1][0:3] == (name, case.params.inputs, {'source': 'explicit'})
    assert len(config.base_calls) == 1


async def test_legacy_callable_backend_preserves_handler_exception() -> None:
    error = RuntimeError('handler failed')

    async def handler(params: _DispatchParams) -> None:
        raise error

    operation = DurableOperation(
        operation_id=EventStreamHandlerId(),
        handler=handler,
        parameter_transport=IdentityParameterTransport[_DispatchParams](),
        cache_identity=_LogicalInputs(),
        result_codec=TypedResultCodec[None](type(None)),
        config_role=OperationConfigRole.EVENT,
    )
    backend = LegacyCallableBackend(
        _RecordingLegacyCapability(), namer=JournalOperationNamer('compat'), config=_Config()
    )

    with pytest.raises(RuntimeError, match='handler failed') as exc_info:
        await backend.bind(operation)(_DispatchParams((object(),)))
    assert exc_info.value is error


class _NoneConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> None:
        return None

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> Literal[False]:
        return False


async def test_legacy_callable_backend_matches_live_production_assembly_inputs() -> None:
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

    await agent.run('Call every tool')

    production = JournalDurability.from_agent(agent)
    assert production is not None
    assert production._legacy_operation_name(GetInstructionsId('mcp')) == 'compat__mcp_server__mcp.get_instructions'  # pyright: ignore[reportPrivateUsage]
    assert production.calls
    operation_ids: dict[str, DurableOperationId] = {
        'compat__model.request_stream': ModelRequestId(None, True, 'test'),
        'compat__event_stream_handler': EventStreamHandlerId(),
        'compat__function_toolset__functions.call_tool:function_tool': CallToolId('function', 'functions'),
        'compat__dynamic_toolset__dynamic.get_tools': GetToolsId('dynamic', 'dynamic'),
        'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool': CallToolId('dynamic', 'dynamic'),
    }
    comparable = [call for call in production.calls if call[0] in operation_ids]
    assert set(operation_ids) <= {name for name, _, _ in comparable}

    declaration_capability = _RecordingLegacyCapability()
    backend = LegacyCallableBackend(declaration_capability, namer=JournalOperationNamer('compat'), config=_NoneConfig())
    for name, inputs, config in comparable:

        async def handler(params: _DispatchParams) -> None:
            return None

        operation_id = operation_ids[name]
        operation = DurableOperation(
            operation_id=operation_id,
            handler=handler,
            parameter_transport=IdentityParameterTransport[_DispatchParams](),
            cache_identity=_LogicalInputs(),
            result_codec=TypedResultCodec[None](type(None)),
            config_role=OperationConfigRole.TOOL_CALL,
        )
        tool_name = inputs[0] if isinstance(operation_id, CallToolId) and isinstance(inputs[0], str) else ''
        await backend.bind(operation)(_DispatchParams(inputs, tool_name), config=config)

    assert [(name, inputs, config) for name, inputs, config, _ in declaration_capability.calls] == comparable


async def test_live_model_declaration_honors_legacy_unit_name_override() -> None:
    agent = Agent(TestModel(), name='compat', capabilities=[_OverrideNameDurability()])

    await agent.run('hello')

    durability = _OverrideNameDurability.from_agent(agent)
    assert durability is not None
    assert [name for name, _, _ in durability.calls] == ['override:model.request:Model Request']


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
    assert NoCacheIdentity[object]().project(value) == ()
    invocation = OperationInvocation(params=value, config='config')
    assert invocation.params is value
    assert invocation.config == 'config'


def test_dbos_registered_backend_exposes_bound_operation_and_rejects_unsupported_ids() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec.dbos._operation_backend import DBOSOperationBackend, DBOSOperationConfig

    async def handler(params: _DispatchParams) -> None:
        return None

    backend = DBOSOperationBackend(
        agent_name='registered',
        config=DBOSOperationConfig(model={}, event={}, tool={}),
    )
    operation = DurableOperation(
        operation_id=ModelRequestId(None, False, 'test'),
        handler=handler,
        parameter_transport=IdentityParameterTransport[_DispatchParams](),
        cache_identity=_LogicalInputs(),
        result_codec=TypedResultCodec[None](type(None), mode='identity'),
        config_role=OperationConfigRole.MODEL,
    )
    assert backend.bind(operation).operation is operation

    unsupported = DurableOperation(
        operation_id=ValidateToolArgumentsId('dynamic', 'tools'),
        handler=handler,
        parameter_transport=IdentityParameterTransport[_DispatchParams](),
        cache_identity=_LogicalInputs(),
        result_codec=TypedResultCodec[None](type(None), mode='identity'),
        config_role=OperationConfigRole.TOOL_VALIDATION,
    )
    with pytest.raises(RuntimeError, match='not yet assigned'):
        backend.bind(unsupported)

    with pytest.raises(TypeError, match='not a model or event operation'):
        backend._bind_model_or_event(unsupported, 'unsupported', {})  # pyright: ignore[reportPrivateUsage]

    unsupported_call = replace(unsupported, operation_id=CallToolId('function', 'tools'))
    with pytest.raises(TypeError, match='not registered'):
        backend.bind(unsupported_call)
