from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal, cast

import pytest
from pydantic import TypeAdapter
from typing_extensions import assert_never

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentStreamEvent,
    FunctionToolset,
    ModelResponse,
    RunContext,
    TextPart,
    Tool,
    ToolsetTool,
)
from pydantic_ai.durable_exec._base import (
    BaseDurabilityCapability,
    CompactMessagesOperationParams,
    ToolsetKind,
)
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import (
    CacheIdentity,
    CallToolId,
    CancelSuspendedResponseId,
    CapabilityOperationId,
    CompactMessagesId,
    DurableOperation,
    DurableOperationId,
    EventStreamHandlerId,
    GetInstructionsId,
    GetToolsId,
    IdentityParameterTransport,
    ModelRequestId,
    NoCacheIdentity,
    OperationConfigRole,
    TypedResultCodec,
    ValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import (
    BoundDurableOperation,
    CallableOperationBackend,
    DurableOperationBackend,
    RegisteredOperationBackend,
)
from pydantic_ai.durable_exec._operation_names import DurableOperationNamer, JournalOperationNamer
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DurableFunctionToolset,
    DynamicToolInfo,
    DynamicToolsResult,
    Lifecycle,
    ToolConfig,
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _ToolFailed,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
    call_dynamic_tool,
    get_dynamic_tools,
    run_args_validator,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.durable_exec._utils import DurableModel, StreamedActivityResult
from pydantic_ai.durable_exec.dbos._operation_names import DBOSOperationNamer
from pydantic_ai.durable_exec.prefect._operation_names import PrefectOperationNamer
from pydantic_ai.durable_exec.temporal._operation_names import TemporalOperationNamer
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.usage import RunUsage

JOURNAL_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request.registered',
    'compat__model.request_stream',
    'compat__model.request_stream.registered',
    'compat__model.cancel_suspended_response',
    'compat__model.cancel_suspended_response.registered',
    'compat__model.compact_messages',
    'compat__model.compact_messages.registered',
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
    'Compact Messages: test',
    'Handle Stream Event',
    'Call Tool: function_tool',
    'Validate Tool Args: function_tool',
    'Call MCP Tool: mcp_tool',
    'Call Tool: dynamic_tool',
    'Validate Tool Args: dynamic_tool',
}
DBOS_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request_stream',
    'compat__model.cancel_suspended_response',
    'compat__model.compact_messages',
    'compat__event_stream_handler',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
}
TEMPORAL_ACTIVITY_NAMES = {
    'agent__compat__model_request',
    'agent__compat__model_request_stream',
    'agent__compat__model_cancel_suspended_response',
    'agent__compat__model_compact_messages',
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


class _JournalConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> ToolConfig:
        return {}

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> ToolConfig:
        return {}


class _JournalBackend(CallableOperationBackend[ToolConfig]):
    def __init__(self, durability: JournalDurability) -> None:
        super().__init__(namer=JournalOperationNamer(durability.name), config=_JournalConfig())
        self._durability = durability

    async def _execute(
        self, *, name: str, body: Callable[[], Awaitable[object]], cache_key: tuple[object, ...], config: object
    ) -> object:
        self._durability.calls.append((name, cache_key, config))
        return await body()


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

    def get_durable_operation_backend(self) -> DurableOperationBackend[ToolConfig]:
        return _JournalBackend(self)


async def test_durability_forces_sequential_tools_inside_durable_context() -> None:
    class SequentialJournalDurability(JournalDurability):
        _force_sequential_tools_in_durable_context = True

    durability = SequentialJournalDurability()
    agent = Agent(TestModel(), name='sequential', capabilities=[durability])
    bound = JournalDurability.from_agent(agent)
    assert bound is not None
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    async def handler() -> AgentRunResult[Any]:
        assert ToolManager(FunctionToolset()).get_parallel_execution_mode() == 'sequential'
        return cast(AgentRunResult[Any], object())

    await bound.wrap_run(ctx, handler=handler)


def test_prepare_run_context_without_agent() -> None:
    durability = JournalDurability()
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    durability._prepare_run_context(ctx)  # pyright: ignore[reportPrivateUsage]

    assert ctx.__dict__['_durable_operations'] == {}


def _synthetic_toolsets() -> tuple[FunctionToolset[Any], DynamicToolset[Any], Any]:
    pytest.importorskip('mcp')
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset

    # Assembly inspects this tool's definition but never executes its body.
    async def function_tool() -> str: ...  # pragma: no branch

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
        ValidateToolArgumentsId('function', 'functions'),
        CallToolId('mcp', 'mcp'),
        GetToolsId('dynamic', 'dynamic'),
        CallToolId('dynamic', 'dynamic'),
        ValidateToolArgumentsId('dynamic', 'dynamic'),
        CompactMessagesId(None, 'test'),
        CompactMessagesId('registered', 'test'),
    ]


def _params(operation_id: DurableOperationId) -> object:
    if isinstance(operation_id, CallToolId | ValidateToolArgumentsId):
        names = {'function': 'function_tool', 'mcp': 'mcp_tool', 'dynamic': 'dynamic_tool'}
        return _ToolParams(names[operation_id.toolset_kind])
    return object()


def test_journal_operation_names() -> None:
    namer = JournalOperationNamer('compat')
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in _ids()]
    assert set(actual) == JOURNAL_OPERATION_NAMES


def test_prefect_operation_names() -> None:
    ids = [
        *_ids()[:1],
        _ids()[2],
        _ids()[4],
        _ids()[6],
        _ids()[9],
        _ids()[10],
        _ids()[11],
        _ids()[13],
        _ids()[14],
        _ids()[15],
    ]
    namer = PrefectOperationNamer()
    actual = [namer.invocation_name(operation_id, _params(operation_id)).operation_name for operation_id in ids]
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
    backend = old._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    live = {cast(Any, registration).dbos_function_name for registration in backend.registrations()}
    ids = [
        _ids()[0],
        _ids()[2],
        _ids()[4],
        _ids()[6],
        _ids()[7],
        _ids()[8],
        _ids()[11],
        _ids()[12],
        _ids()[13],
        _ids()[14],
        _ids()[15],
    ]
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
        CompactMessagesId(None, 'test'),
        EventStreamHandlerId(),
        CallToolId('function', '<agent>'),
        ValidateToolArgumentsId('function', '<agent>'),
        CallToolId('function', 'functions'),
        ValidateToolArgumentsId('function', 'functions'),
        GetToolsId('mcp', 'mcp'),
        GetInstructionsId('mcp'),
        CallToolId('mcp', 'mcp'),
        GetToolsId('dynamic', 'dynamic'),
        CallToolId('dynamic', 'dynamic'),
        ValidateToolArgumentsId('dynamic', 'dynamic'),
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
    from pydantic_ai.durable_exec.temporal._transports import (
        _CompactMessagesParams,  # pyright: ignore[reportPrivateUsage]
    )

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
        durability.compact_messages_activity,
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
        durability.compact_messages_activity: (_CompactMessagesParams, ModelResponse),
        durability.event_stream_handler_activity: (_EventStreamHandlerParams, type(None)),
        durability.cancel_suspended_response_activity: (_CancelParams, type(None)),
    }
    for activity_fn, (params_type, result_type) in expected_signatures.items():
        signature = inspect.signature(activity_fn)
        assert signature.parameters['params'].annotation is params_type
        assert signature.parameters['deps'].annotation == agent.deps_type | None
        assert signature.parameters['deps'].default is None
        assert signature.return_annotation == result_type


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
    cancel_operation = operations[3]
    assert cancel_operation.operation.operation_id == CancelSuspendedResponseId(None, 'test:test')
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


async def test_temporal_backend_dispatches_compact_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('temporalio')
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.durable_exec.temporal._transports import (
        _CompactMessagesParams,  # pyright: ignore[reportPrivateUsage]
    )

    agent = Agent(TestModel(), name='compact-dispatch', capabilities=[TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    expected = ModelResponse(parts=[TextPart('compacted')])
    dispatched: list[tuple[Callable[..., object], Sequence[object], object]] = []

    async def execute_activity(
        activity: Callable[..., object], *, args: Sequence[object], **config: object
    ) -> ModelResponse:
        dispatched.append((activity, args, config['summary']))
        return expected

    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._operation_backend.execute_activity', execute_activity)
    model = TestModel()
    ctx = RunContext[None](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    operations = durability._bound_model_operations  # pyright: ignore[reportPrivateUsage]
    assert operations is not None
    operation = operations[2]
    result = cast(
        ModelResponse,
        await operation(CompactMessagesOperationParams(None, request_context, 'Keep decisions', ctx)),
    )

    assert result == expected
    activity, args, summary = dispatched[0]
    assert activity is durability.compact_messages_activity
    assert isinstance(args[0], _CompactMessagesParams)
    assert args[0].instructions == 'Keep decisions'
    assert summary == 'compact messages: test:test'


async def test_temporal_compaction_payload_round_trips_live_durable_model() -> None:
    pytest.importorskip('temporalio')
    from temporalio.contrib.pydantic import pydantic_data_converter

    from pydantic_ai.durable_exec._utils import DurableModel
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.durable_exec.temporal._transports import (
        _CompactMessagesParams,  # pyright: ignore[reportPrivateUsage]
        _CompactMessagesTransport,
    )

    # Converter round-tripping inspects these callables but cannot dispatch them.
    async def request_segment(request: ModelRequestContext) -> ModelResponse: ...  # pragma: no branch

    async def stream_segment(request: ModelRequestContext) -> StreamedActivityResult: ...

    async def compact_segment(  # pragma: no cover
        request: ModelRequestContext, instructions: str | None
    ) -> ModelResponse: ...

    async def cancel_segment(response: ModelResponse) -> None:
        pass

    model = DurableModel(
        TestModel(),
        request_segment=request_segment,
        request_stream_segment=stream_segment,
        compact_messages_segment=compact_segment,
        cancel_suspended_response_segment=cancel_segment,
    )
    agent = Agent(TestModel(), name='compact-converter', capabilities=[TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), agent=agent)
    request_context = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    transport = _CompactMessagesTransport(durability)
    wire = transport.dump(CompactMessagesOperationParams(None, request_context, None, ctx))

    payloads = await pydantic_data_converter.encode([wire])
    [decoded] = await pydantic_data_converter.decode(payloads, [tuple[_CompactMessagesParams, type(None)]])

    assert decoded[0].messages == []
    assert decoded[0].model_request_parameters == ModelRequestParameters()


async def test_temporal_backend_labels_validation_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip('temporalio')
    from pydantic_ai.durable_exec.temporal._operation_backend import TemporalBoundOperation

    # The workflow dispatcher receives these callables as identities and must not invoke them.
    async def handler(params: _ToolParams) -> None: ...  # pragma: no branch

    async def registration(params: _ToolParams) -> None: ...  # pragma: no branch

    operation = DurableOperation(
        operation_id=ValidateToolArgumentsId('dynamic', 'tools'),
        handler=handler,
        parameter_transport=IdentityParameterTransport[_ToolParams](),
        cache_identity=NoCacheIdentity(),
        result_codec=TypedResultCodec[None](type(None), mode='identity'),
        config_role=OperationConfigRole.TOOL_VALIDATION,
    )
    dispatched: list[tuple[Callable[..., object], Sequence[object], object]] = []

    async def execute_activity(activity: Callable[..., object], *, args: Sequence[object], **config: object) -> None:
        dispatched.append((activity, args, config['summary']))

    monkeypatch.setattr(
        'pydantic_ai.durable_exec.temporal._operation_backend.execute_activity',
        execute_activity,
    )
    bound = TemporalBoundOperation(operation, registration, {})
    params = _ToolParams('guarded')
    await bound(params)

    assert dispatched == [(registration, params, 'validate tool args: tools:guarded')]


def _exhaustive_identity(operation_id: DurableOperationId) -> str:
    match operation_id:
        case ModelRequestId():
            return 'model'
        case CancelSuspendedResponseId():
            return 'cancel'
        case CompactMessagesId():
            return 'compact'
        case CapabilityOperationId():
            return 'capability'
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
        CompactMessagesId(None, 'model'),
        CapabilityOperationId('capability', 'operation'),
        EventStreamHandlerId(),
        GetToolsId('function', 'tools'),
        GetInstructionsId('mcp'),
        ValidateToolArgumentsId('dynamic', 'dynamic'),
        CallToolId('mcp', 'mcp'),
    ]
    assert [_exhaustive_identity(operation_id) for operation_id in identities] == [
        'model',
        'cancel',
        'compact',
        'capability',
        'event',
        'tools',
        'instructions',
        'validation',
        'call',
    ]


@pytest.mark.parametrize(
    ('namer', 'expected'),
    [
        (JournalOperationNamer('agent'), 'agent__function_toolset__tools.validate_args'),
        (PrefectOperationNamer(), 'Validate Tool Args'),
        (DBOSOperationNamer('agent'), 'agent__function_toolset__tools.validate_args'),
        (TemporalOperationNamer('agent'), 'agent__agent__toolset__tools__validate_args'),
    ],
)
def test_validation_operation_names(namer: DurableOperationNamer, expected: str) -> None:
    assert namer.operation_name(ValidateToolArgumentsId('function', 'tools')) == expected


@pytest.mark.parametrize(
    ('namer', 'expected_name'),
    [
        (PrefectOperationNamer(), 'Compact Messages: test'),
        (DBOSOperationNamer('agent'), 'agent__model.compact_messages'),
        (TemporalOperationNamer('agent'), 'agent__agent__model_compact_messages'),
    ],
)
async def test_durable_model_compact_messages_dispatches_operation(
    namer: DurableOperationNamer, expected_name: str
) -> None:
    config = _Config()
    backend = _RecordingBackend(config, namer=namer)
    response = ModelResponse(parts=[TextPart('compacted')])

    async def handler(params: CompactMessagesOperationParams) -> ModelResponse:
        assert params.instructions == 'Keep decisions'
        return response

    operation = backend.bind(
        DurableOperation(
            operation_id=CompactMessagesId(None, 'test'),
            handler=handler,
            parameter_transport=IdentityParameterTransport[CompactMessagesOperationParams](),
            cache_identity=NoCacheIdentity(),
            result_codec=TypedResultCodec[ModelResponse](ModelResponse),
            config_role=OperationConfigRole.MODEL,
        )
    )
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    async def compact_messages_segment(request_context: ModelRequestContext, instructions: str | None) -> ModelResponse:
        return await operation(CompactMessagesOperationParams(None, request_context, instructions, ctx))

    # Compact-message dispatch must not enter the other model segment callables.
    async def unused_request(request_context: ModelRequestContext) -> ModelResponse: ...  # pragma: no branch

    async def unused_stream(request_context: ModelRequestContext) -> StreamedActivityResult: ...  # pragma: no branch

    async def unused_cancel(response: ModelResponse) -> None: ...  # pragma: no branch

    model = DurableModel(
        TestModel(),
        request_segment=unused_request,
        request_stream_segment=unused_stream,
        compact_messages_segment=compact_messages_segment,
        cancel_suspended_response_segment=unused_cancel,
    )
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    assert await model.compact_messages(request_context, instructions='Keep decisions') == response
    assert backend.calls[0][0] == expected_name


async def test_base_durable_model_compact_messages_dispatches_bound_operation() -> None:
    response = ModelResponse(parts=[TextPart('compacted')])

    class CompactModel(TestModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            assert instructions == 'Keep decisions'
            return response

    model = CompactModel()
    agent = Agent(model, name='compact', capabilities=[JournalDurability()])
    durability = JournalDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext[None](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    async def handler(context: ModelRequestContext) -> ModelResponse:
        assert isinstance(context.model, DurableModel)
        return await context.model.compact_messages(context, instructions='Keep decisions')

    assert await durability.wrap_model_request(ctx, request_context=request_context, handler=handler) == response
    assert durability.calls[0][0] == 'compact__model.compact_messages'


async def test_dbos_compact_messages_operation_dispatches_step() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.dbos._operation_backend import DBOSBoundOperation

    expected = ModelResponse(parts=[TextPart('compacted')])

    class CompactModel(TestModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            assert instructions == 'Keep decisions'
            return expected

    model = CompactModel()
    agent = Agent(model, name='compact-dbos', capabilities=[DBOSDurability()])
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None
    operations = durability._bound_model_operations  # pyright: ignore[reportPrivateUsage]
    assert operations is not None
    operation = operations[2]
    assert isinstance(operation, DBOSBoundOperation)
    step_body = cast(Callable[..., Awaitable[object]], cast(Any, operation.step).__wrapped__)

    async def step(*args: object) -> object:
        return await step_body(*args)

    operation.use_step_getter(lambda: step)
    ctx = RunContext[None](deps=None, model=model, usage=RunUsage())
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    result = cast(
        ModelResponse,
        await operation(CompactMessagesOperationParams(None, request_context, 'Keep decisions', ctx)),
    )
    assert result == expected


def test_namer_error_paths_and_unrepresented_formats() -> None:
    with pytest.raises(TypeError, match='must expose'):
        JournalOperationNamer('agent').invocation_name(CallToolId('function', 'tools'), object())
    prefect = PrefectOperationNamer()
    with pytest.raises(RuntimeError, match='bug in the durability integration; please report it'):
        prefect.operation_name(GetToolsId('dynamic', 'tools'))
    with pytest.raises(RuntimeError, match='bug in the durability integration; please report it'):
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
    def __init__(self, config: _Config, *, namer: DurableOperationNamer | None = None) -> None:
        super().__init__(namer=namer or JournalOperationNamer('agent'), config=config)
        self.calls: list[tuple[str, object, object]] = []

    async def _execute(
        self, *, name: str, body: Callable[[], Awaitable[object]], cache_key: tuple[object, ...], config: object
    ) -> object:
        self.calls.append((name, cache_key, config))
        return await body()


class _RegisteredBoundOperation:
    def __init__(self, operation: DurableOperation[Any, Any, Any]) -> None:
        self.operation = operation

    async def __call__(self, params: Any, *, config: object | None = None) -> Any:
        return await self.operation.handler(params)


class _RegisteredBackend(RegisteredOperationBackend[dict[str, str]]):
    def __init__(self, config: _Config) -> None:
        super().__init__(namer=JournalOperationNamer('agent'), config=config)
        self.operations: list[DurableOperation[Any, Any, Any]] = []

    def _register(
        self,
        operation: DurableOperation[Any, Any, Any],
        *,
        name: str,
        config: dict[str, str],
    ) -> tuple[BoundDurableOperation[Any, Any, Any], Sequence[Callable[..., object]]]:
        self.operations.append(operation)
        return _RegisteredBoundOperation(operation), (operation.handler,)


class _RegisteredDurability(JournalDurability):
    def __init__(self) -> None:
        super().__init__()
        self.backend = _RegisteredBackend(_Config())

    def get_durable_operation_backend(self) -> _RegisteredBackend:
        return self.backend


@dataclass(frozen=True)
class _DispatchParams:
    inputs: tuple[object, ...]
    name: str = ''


class _LogicalInputs(CacheIdentity[_DispatchParams]):
    def project(self, params: _DispatchParams) -> tuple[object, ...]:
        return params.inputs


class _NoneConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> None:
        return None

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> Literal[False]:
        return False


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


async def test_registered_backend_binds_model_operations_during_agent_assembly() -> None:
    agent = Agent(TestModel(), name='registered', capabilities=[_RegisteredDurability()])

    durability = _RegisteredDurability.from_agent(agent)
    assert durability is not None
    backend = durability.backend
    assert backend.registrations() == [operation.handler for operation in backend.operations]
    model_operation_ids = [
        operation.operation_id
        for operation in backend.operations
        if isinstance(operation.operation_id, (ModelRequestId, CompactMessagesId, CancelSuspendedResponseId))
    ]
    assert model_operation_ids == [
        ModelRequestId(None, False, 'default'),
        ModelRequestId(None, True, 'default'),
        CompactMessagesId(None, 'default'),
        CancelSuspendedResponseId(None, 'default'),
    ]

    async def handler(params: int) -> int:
        return params + 1

    operation = DurableOperation(
        operation_id=ModelRequestId('registered', False, 'model'),
        handler=handler,
        parameter_transport=IdentityParameterTransport[int](),
        cache_identity=_CacheIdentity(),
        result_codec=TypedResultCodec[int](int),
        config_role=OperationConfigRole.MODEL,
    )
    bound = backend.bind(operation)
    assert await bound(4) == 5


def test_trivial_transport_cache_and_invocation_helpers() -> None:
    value = object()
    transport = IdentityParameterTransport[object]()
    assert transport.dump(value) is value
    assert transport.load(value, runtime=object()) is value
    assert NoCacheIdentity[object]().project(value) == ()
    assert _NoneConfig().for_tool(OperationConfigRole.TOOL_CALL, CallToolId('function', 'tools'), None, 'tool') is False


def test_dbos_registered_backend_exposes_bound_operation_and_rejects_unsupported_ids() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec.dbos._operation_backend import DBOSOperationBackend, DBOSOperationConfig

    # Registration is inspected without dispatching this operation.
    async def handler(params: _DispatchParams) -> None: ...  # pragma: no branch

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
    assert backend.bind(unsupported).operation is unsupported

    with pytest.raises(TypeError, match='not a model or event operation'):
        backend._bind_model_or_event(unsupported, 'unsupported', {})  # pyright: ignore[reportPrivateUsage]

    unsupported_call = replace(unsupported, operation_id=CallToolId('function', 'tools'))
    with pytest.raises(TypeError, match='not registered'):
        backend.bind(unsupported_call)


async def test_dynamic_args_validator_runs_in_declarative_unit_and_preserves_schedule() -> None:
    calls: list[str] = []

    def validate_path(ctx: RunContext[None], path: str) -> None:
        if path == '/etc/shadow':
            raise ModelRetry('forbidden path')

    async def read_file(path: str) -> str:
        calls.append(path)
        return path

    async def stat_file(path: str) -> str:
        calls.append(path)
        return path

    def model(messages: list[Any], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        prompt = messages[0].parts[-1]
        assert isinstance(prompt, UserPromptPart)
        tool_name, path = str(prompt.content).split(' ', 1)
        return ModelResponse(parts=[ToolCallPart(tool_name, {'path': path})])

    inner = FunctionToolset(tools=[Tool(read_file, args_validator=validate_path), Tool(stat_file)], id='inner')
    durability = JournalDurability()
    agent = Agent[None, str](
        FunctionModel(model),
        name='validation',
        deps_type=type(None),
        toolsets=[DynamicToolset(lambda _: inner, id='files')],
        capabilities=[durability],
    )

    rejected = await agent.run('read_file /etc/shadow')
    assert calls == []
    assert [
        str(part.content)
        for message in rejected.all_messages()
        for part in message.parts
        if isinstance(part, RetryPromptPart)
    ] == ['forbidden path']
    recorded_names = [name for name, _, _ in durability.calls]
    assert 'validation__dynamic_toolset__files.validate_args' in recorded_names
    assert 'validation__dynamic_toolset__files.call_tool:read_file' not in recorded_names

    durability.calls.clear()
    await agent.run('stat_file /tmp/file')
    assert calls == ['/tmp/file']
    recorded_names = [name for name, _, _ in durability.calls]
    assert 'validation__dynamic_toolset__files.validate_args' not in recorded_names
    assert 'validation__dynamic_toolset__files.call_tool:stat_file' in recorded_names

    static_durability = JournalDurability()
    static_agent = Agent[None, str](
        FunctionModel(model),
        name='static_validation',
        deps_type=type(None),
        toolsets=[inner],
        capabilities=[static_durability],
    )
    await static_agent.run('read_file /tmp/static')
    assert calls[-1] == '/tmp/static'
    static_names = [name for name, _, _ in static_durability.calls]
    assert 'static_validation__function_toolset__inner.validate_args' in static_names
    assert 'static_validation__function_toolset__inner.call_tool:read_file' in static_names


async def test_tool_body_validation_error_preserves_detailed_retry() -> None:
    invalid_input = 'not-an-integer'
    model_messages: list[list[Any]] = []

    async def inspect_record() -> None:
        TypeAdapter(int).validate_python(invalid_input)

    def model(messages: list[Any], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('inspect_record', {})])

    agent = Agent[None, str](
        FunctionModel(model),
        name='tool_body_validation',
        deps_type=type(None),
        tools=[inspect_record],
        capabilities=[JournalDurability()],
    )

    result = await agent.run('inspect the record')

    assert result.output == 'done'
    assert len(model_messages) == 2
    retry = next(part for message in model_messages[1] for part in message.parts if isinstance(part, RetryPromptPart))
    assert retry.content == [
        {
            'type': 'int_parsing',
            'loc': (),
            'msg': 'Input should be a valid integer, unable to parse string as an integer',
            'input': invalid_input,
        }
    ]

    class NonSerializableInput:
        def __repr__(self) -> str:
            return '<non-serializable-input>'

    async def inspect_non_serializable_input() -> None:
        TypeAdapter(int).validate_python(NonSerializableInput())

    payload = await wrap_tool_call_result(inspect_non_serializable_input())
    assert JSON_CODEC.dump(CallToolResult, payload) == {
        'title': 'int',
        'errors': [
            {
                'type': 'int_type',
                'loc': [],
                'msg': 'Input should be a valid integer',
                'input': {
                    'type': f'{NonSerializableInput.__module__}.{NonSerializableInput.__qualname__}',
                    'repr': '<non-serializable-input>',
                },
            }
        ],
        'kind': 'validation_error',
    }


async def test_legacy_dynamic_execution_unit_preserves_argument_validation_retry() -> None:
    tool_calls: list[int] = []
    retries = 0

    async def double(x: int) -> str:
        tool_calls.append(x)
        return f'got {x}'

    inner = FunctionToolset(tools=[double], id='inner')
    dynamic = DynamicToolset(lambda _: inner, id='dynamic')

    async def call_tool(
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        config: Mapping[str, Any],
    ) -> Any:
        del config
        payload = await wrap_tool_call_result(call_dynamic_tool(dynamic, name, tool_args, ctx, tool_def=tool.tool_def))
        return unwrap_tool_call_result(payload)

    durable = DurableDynamicToolset(
        dynamic,
        in_durable_context=lambda: True,
        get_tools_operation=lambda ctx: get_dynamic_tools(dynamic, ctx),
        call_tool_operation=call_tool,
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-never',
    )

    def model(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal retries
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        if any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts):
            retries += 1
            return ModelResponse(parts=[ToolCallPart('double', {'x': 5})])
        return ModelResponse(parts=[ToolCallPart('double', {'x': 'not-a-number'})])

    result = await Agent(FunctionModel(model), toolsets=[durable]).run('double it')

    assert result.output == 'done'
    assert retries == 1
    assert tool_calls == [5]


async def test_dynamic_validator_without_durable_unit_is_a_hard_error() -> None:
    async def get_tools(ctx: RunContext[None]) -> DynamicToolsResult:
        return DynamicToolsResult(
            tools={
                'guarded': DynamicToolInfo(
                    tool_def=ToolDefinition(name='guarded'), max_retries=1, has_args_validator=True
                )
            },
            instructions=None,
        )

    # Missing validation is rejected before tool execution can be dispatched.
    async def never_called(*args: Any) -> Any: ...  # pragma: no branch

    durable = DurableDynamicToolset(
        DynamicToolset(lambda _: None, id='missing_validation'),
        in_durable_context=lambda: True,
        get_tools_operation=get_tools,
        call_tool_operation=never_called,
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-never',
    )
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    with pytest.raises(UserError, match=r"Tool 'guarded'.*has an `args_validator`"):
        await durable.get_tools(ctx)


async def test_legacy_validation_fallbacks_remain_inline() -> None:
    calls: list[str] = []

    def validate_value(ctx: RunContext[None], value: str) -> None:
        calls.append(value)

    # This test invokes only the validator wrappers, not the tool body.
    async def tool(value: str) -> str: ...

    async def unused_operation(  # pragma: no cover
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        config: Mapping[str, Any],
    ) -> Any: ...

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    function_toolset = FunctionToolset([Tool(tool, args_validator=validate_value)], id='function')
    durable_function = DurableFunctionToolset(
        function_toolset,
        in_durable_context=lambda: True,
        call_tool_operation=unused_operation,
        resolve_tool_config=lambda tool, name: {},
        lifecycle='enter-never',
    )
    function_tools = await durable_function.get_tools(ctx)
    function_validator = function_tools['tool'].args_validator_func
    assert function_validator is not None
    result = function_validator(ctx, value='function')
    assert result is None

    dynamic_toolset = DynamicToolset(lambda _: function_toolset, id='dynamic')
    dynamic = DurableDynamicToolset(
        dynamic_toolset,
        in_durable_context=lambda: True,
        get_tools_operation=lambda ctx: get_dynamic_tools(dynamic_toolset, ctx),
        call_tool_operation=unused_operation,
        resolve_tool_config=lambda tool, name: False,
        lifecycle='enter-never',
    )
    dynamic_tools = await dynamic.get_tools(ctx)
    dynamic_validator = dynamic_tools['tool'].args_validator_func
    assert dynamic_validator is not None
    await dynamic_validator(ctx, value='dynamic')

    assert calls == ['function', 'dynamic']


async def test_args_validator_disappearing_after_discovery_is_rejected() -> None:
    tool = ToolsetTool(
        toolset=FunctionToolset(),
        tool_def=ToolDefinition(name='changed'),
        max_retries=0,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(UserError, match="Tool 'changed' has no `args_validator`"):
        await run_args_validator(tool, {}, ctx)
