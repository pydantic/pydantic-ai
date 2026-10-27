from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass, fields
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    DeferredToolRequests,
    DeferredToolResults,
    FunctionToolset,
    ModelResponse,
    RunContext,
    TextPart,
    Tool,
    durable_exec,
)
from pydantic_ai._tool_execution import _ToolCallProcessor  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._utils import get_union_args
from pydantic_ai.capabilities import AbstractCapability, SetToolMetadata, durable_operation
from pydantic_ai.durable_exec import (
    IDENTITY_CODEC,
    JSON_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    CapabilityOperationId,
    DurabilityEngineSpec,
    DurableOperationId,
    EventStreamHandlerId,
    JournalOperationNamer,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    OperationConfigRole,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityOperationResult,
    ModelRequestContextProjection,
    capability_operation_result_type,
)
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    ToolConfig,
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _RunCancelled,  # pyright: ignore[reportPrivateUsage]
    _ToolFailed,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
    _ValidationError,  # pyright: ignore[reportPrivateUsage]
    _ValidationErrorDetail,  # pyright: ignore[reportPrivateUsage]
    run_args_validator,
    unwrap_recorded_tool_call_result,
    unwrap_tool_call_result,
    validate_tool_args,
    wrap_tool_call_result,
)
from pydantic_ai.durable_exec._utils import unwrap_model
from pydantic_ai.durable_exec._workspace import WORKSPACE_OPERATION_ID
from pydantic_ai.exceptions import RunCancelled, RunTimedOut
from pydantic_ai.messages import (
    BinaryContent,
    CapabilityEvent,
    CompactionPart,
    CustomEvent,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    EnqueuedMessagesEvent,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    ModelResponsePartDelta,
    ModelSettingsDeltaEvent,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
    RetryFeedbackPart,
    RetryInfo,
    SpeechPart,
    SpeechPartDelta,
    SystemPromptPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolAvailabilityDeltaEvent,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ..workspace_fakes import FakeWorkspace


def test_public_engine_builder_exports() -> None:
    assert durable_exec.__all__ == [
        'BaseDurabilityCapability',
        'conversation_id_from_run_id',
        'ToolsetCallToolId',
        'CallableOperationBackend',
        'ModelCancelSuspendedResponseId',
        'CapabilityOperationId',
        'ModelCompactMessagesId',
        'DurabilityCodec',
        'DurabilityEngineSpec',
        'DurableOperationBackend',
        'DurableOperationId',
        'DurableOperationNamer',
        'EventStreamHandlerId',
        'ToolsetGetInstructionsId',
        'ToolsetGetToolsId',
        'IDENTITY_CODEC',
        'JSON_CODEC',
        'JournalCallableOperationBackend',
        'JournalOperationNamer',
        'ModelRequestId',
        'OperationConfigRole',
        'RegisteredOperationBackend',
        'RoleBasedOperationConfig',
        'ToolsetKind',
        'ToolsetValidateToolArgumentsId',
        'BoundDurableOperation',
        'CacheIdentity',
        'CallToolResult',
        'capability_operation_result_type',
        'CapabilityMethodDeclaration',
        'CapabilityOperationParams',
        'durable_unit_scope',
        'DurableOperation',
        'DurableOperationConfig',
        'DurableRunContext',
        'DynamicToolsetCallToolParams',
        'DynamicToolsResult',
        'EventStreamHandlerParams',
        'ModelCancelSuspendedResponseParams',
        'ModelCompactMessagesParams',
        'ModelRequestContextProjection',
        'ModelRequestParams',
        'ParameterTransport',
        'ParameterTransports',
        'ResultCodec',
        'ToolsetCallToolParams',
        'ToolsetGetToolsParams',
        'revalidate_function_call',
        'WorkspaceCall',
        'WorkspaceCallParams',
        'WorkspaceCallResult',
    ]
    assert all(getattr(durable_exec, name) is not None for name in durable_exec.__all__)


def test_unwrap_model_removes_nested_wrappers() -> None:
    model = TestModel()
    assert unwrap_model(WrapperModel(WrapperModel(model))) is model


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
    'compat__capability__compat.operation',
}

# Bound only for an agent with a construction-time workspace supplier; the sets above stay as they are without one.
JOURNAL_WORKSPACE_NAMES = {'compat__capability__workspace.call'}
PREFECT_WORKSPACE_NAMES = {'Capability: workspace.call'}
TEMPORAL_WORKSPACE_NAMES = {'agent__compat__capability__workspace__call'}

PREFECT_OPERATION_NAMES = {
    'Model Request: test',
    'Model Request (Streaming): test',
    'Cancel Suspended Response: test',
    'Compact Messages: test',
    'Handle Stream Event',
    'Get MCP Tools: mcp',
    'Get MCP Instructions: mcp',
    'Discover Tools: dynamic',
    'Call Tool: function_tool',
    'Validate Tool Args: function_tool',
    'Call MCP Tool: mcp_tool',
    'Call Tool: dynamic_tool',
    'Validate Tool Args: dynamic_tool',
    'Capability: compat.operation',
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
    'agent__compat__capability__compat__operation',
}

TEMPORAL_CHILD_WORKFLOW_NAMES = {
    'agent__compat__toolset__<agent>__call_tool__child_workflow',
    'agent__compat__toolset__functions__call_tool__child_workflow',
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
    'compat__capability__compat.operation',
}

RENDER_TASK_NAMES = {
    'compat__model.request',
    'compat__model.request_stream',
    'compat__model.cancel_suspended_response',
    'compat__model.compact_messages',
    'compat__event_stream_handler',
    'compat__function_toolset__<agent>.call_tool',
    'compat__function_toolset__<agent>.validate_args',
    'compat__function_toolset__functions.call_tool',
    'compat__function_toolset__functions.validate_args',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
    'compat__capability__compat.operation',
}


def _operation_ids() -> list[DurableOperationId]:
    return [
        ModelRequestId(None, streaming=False, model_name='test'),
        ModelRequestId('registered', streaming=False, model_name='test'),
        ModelRequestId(None, streaming=True, model_name='test'),
        ModelRequestId('registered', streaming=True, model_name='test'),
        ModelCancelSuspendedResponseId(None, model_name='test'),
        ModelCancelSuspendedResponseId('registered', model_name='test'),
        ModelCompactMessagesId(None, model_name='test'),
        ModelCompactMessagesId('registered', model_name='test'),
        EventStreamHandlerId(),
        ToolsetCallToolId('function', toolset_id='functions'),
        ToolsetValidateToolArgumentsId('function', toolset_id='functions'),
        ToolsetGetToolsId('mcp', toolset_id='mcp'),
        ToolsetGetInstructionsId('mcp'),
        ToolsetCallToolId('mcp', toolset_id='mcp'),
        ToolsetGetToolsId('dynamic', toolset_id='dynamic'),
        ToolsetCallToolId('dynamic', toolset_id='dynamic'),
        ToolsetValidateToolArgumentsId('dynamic', toolset_id='dynamic'),
        CapabilityOperationId('compat', operation='operation'),
    ]


class CompatWorkspaceSupplier(AbstractCapability[Any]):
    id = 'compat_workspace'

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
        return FakeWorkspace('compat', ref=ref)


def _operation_label(operation_id: DurableOperationId) -> str | None:
    if isinstance(operation_id, (ToolsetCallToolId, ToolsetValidateToolArgumentsId)):
        return {'function': 'function_tool', 'mcp': 'mcp_tool', 'dynamic': 'dynamic_tool'}[operation_id.toolset_kind]
    return None


class CompatCapability(AbstractCapability[Any]):
    id = 'compat'

    @durable_operation('operation')
    async def operation(self, ctx: RunContext[Any]) -> None:
        pass


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

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        self._durability.recorded_names.append(name)
        return await body()


class JournalDurability(BaseDurabilityCapability[Any]):
    engine_spec = DurabilityEngineSpec(
        engine_name='Journal compatibility stub',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
    )

    @property
    def in_durable_context(self) -> bool:
        return True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorded_names: list[str] = []

    def get_durable_operation_backend(self) -> durable_exec.DurableOperationBackend[ToolConfig]:
        return _JournalBackend(self)


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
        capabilities=[CompatCapability(), JournalDurability(event_stream_handler=_event_handler)],
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
    namer = JournalOperationNamer('compat')
    names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
    }
    assert names == JOURNAL_OPERATION_NAMES
    assert {namer.operation_name(WORKSPACE_OPERATION_ID)} == JOURNAL_WORKSPACE_NAMES


async def test_journal_workspace_operation_binds_only_with_a_supplier() -> None:
    plain = JournalDurability()
    await Agent(TestModel(), name='compat', capabilities=[CompatCapability(), plain]).run('go')
    assert not [name for name in plain.recorded_names if 'workspace' in name]

    durability = JournalDurability()
    agent = Agent(TestModel(), name='compat', capabilities=[CompatWorkspaceSupplier(), durability])
    result = await agent.run('go')
    await result.workspace.write_text('a.txt', 'a')
    assert [name for name in durability.recorded_names if 'workspace' in name] == [
        'compat__capability__workspace.call',
        'compat__capability__workspace.call',
    ]


def test_prefect_operation_name_matrix() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec.prefect._operation_names import PrefectOperationNamer

    namer = PrefectOperationNamer()
    names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
    }
    assert names == PREFECT_OPERATION_NAMES
    assert {namer.operation_name(WORKSPACE_OPERATION_ID)} == PREFECT_WORKSPACE_NAMES


def test_prefect_operation_name_assembly_completeness() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableFunctionToolset, DurableMCPToolset
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.prefect._operation_names import PrefectOperationNamer

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[CompatCapability(), PrefectDurability(event_stream_handler=_event_handler)],
    )
    durability = PrefectDurability.from_agent(agent)
    assert durability is not None
    assert {type(toolset) for toolset in durability._toolsets_by_id.values()} >= {  # pyright: ignore[reportPrivateUsage]
        DurableFunctionToolset,
        DurableMCPToolset,
        DurableDynamicToolset,
    }
    namer = PrefectOperationNamer()
    assembled_names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
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
        capabilities=[CompatCapability(), DBOSDurability(event_stream_handler=_event_handler)],
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

    with_workspace = Agent(
        TestModel(),
        name='compat',
        capabilities=[CompatWorkspaceSupplier(), DBOSDurability()],
    )
    workspace_durability = DBOSDurability.from_agent(with_workspace)
    assert workspace_durability is not None
    workspace_backend = workspace_durability._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert workspace_backend is not None
    workspace_names = {cast(Any, registration).dbos_function_name for registration in workspace_backend.registrations()}
    assert workspace_names - DBOS_OPERATION_NAMES == JOURNAL_WORKSPACE_NAMES


@pytest.mark.parametrize('workspace', [False, True])
def test_render_task_name_matrix_and_assembly_completeness(workspace: bool) -> None:
    pytest.importorskip('render')
    pytest.importorskip('pydantic_ai_harness')
    from pydantic_ai_harness import RenderWorkflows

    from ..harness.render.conftest import RecordingWorkflows

    app = RecordingWorkflows()
    capabilities: list[AbstractCapability[Any]] = [
        CompatCapability(),
        RenderWorkflows(app, models={'registered': TestModel()}, event_stream_handler=_event_handler),
    ]
    if workspace:
        capabilities.append(CompatWorkspaceSupplier())
    Agent(TestModel(), name='compat', toolsets=list(_synthetic_toolsets()), capabilities=capabilities)

    names = app.registered_task_names
    expected = RENDER_TASK_NAMES | (JOURNAL_WORKSPACE_NAMES if workspace else set[str]())
    assert set(names) == expected
    assert len(names) == len(expected)


def _synthetic_toolsets() -> tuple[FunctionToolset[Any], DynamicToolset[Any], Any]:
    pytest.importorskip('mcp')
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset

    # Assembly inspects this tool's definition but never executes its body.
    async def function_tool() -> None: """Declaration used only for toolset assembly."""  # fmt: skip  # pragma: no branch

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
    from pydantic_ai.durable_exec.temporal._durability import worker_registrations

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[CompatCapability(), TemporalDurability(event_stream_handler=_event_handler)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    names = {
        ActivityDefinition.must_from_callable(item).name  # pyright: ignore[reportUnknownMemberType]
        for item in durability.temporal_activities
    }
    assert names == TEMPORAL_ACTIVITY_NAMES

    from temporalio.workflow import ChildWorkflowConfig, _Definition as WorkflowDefinition

    # Child workflow definitions are registered only for toolsets that can select one.
    assert worker_registrations(durability)[1] == []
    child_workflow_agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[
            SetToolMetadata(temporal={'child_workflow': ChildWorkflowConfig()}),
            TemporalDurability(event_stream_handler=_event_handler),
        ],
    )
    child_workflow_durability = TemporalDurability.from_agent(child_workflow_agent)
    assert child_workflow_durability is not None
    workflow_names = {
        WorkflowDefinition.must_from_class(item).name for item in worker_registrations(child_workflow_durability)[1]
    }
    assert workflow_names == TEMPORAL_CHILD_WORKFLOW_NAMES
    with_workspace = Agent(
        TestModel(),
        name='compat',
        capabilities=[CompatWorkspaceSupplier(), TemporalDurability()],
    )
    workspace_durability = TemporalDurability.from_agent(with_workspace)
    assert workspace_durability is not None
    workspace_names = {
        ActivityDefinition.must_from_callable(item).name  # pyright: ignore[reportUnknownMemberType]
        for item in workspace_durability.temporal_activities
    }
    assert workspace_names - TEMPORAL_ACTIVITY_NAMES == TEMPORAL_WORKSPACE_NAMES


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (_ToolReturn('ok'), {'result': 'ok', 'kind': 'tool_return'}),
        (_ApprovalRequired({'scope': 'write'}), {'metadata': {'scope': 'write'}, 'kind': 'approval_required'}),
        (_CallDeferred({'ticket': 7}), {'metadata': {'ticket': 7}, 'kind': 'call_deferred'}),
        (_ModelRetry('retry me'), {'message': 'retry me', 'kind': 'model_retry'}),
        (
            _ValidationError(
                'int',
                errors=[_ValidationErrorDetail('int_parsing', loc=['value'], msg='bad integer', input='x')],
            ),
            {
                'title': 'int',
                'errors': [{'type': 'int_parsing', 'loc': ['value'], 'msg': 'bad integer', 'input': 'x'}],
                'kind': 'validation_error',
            },
        ),
        (_ToolFailed('failed'), {'message': 'failed', 'kind': 'tool_failed'}),
        (_RunCancelled('stopped'), {'message': 'stopped', 'timed_out': False, 'kind': 'run_cancelled'}),
        (
            _RunCancelled('deadline', timed_out=True),
            {'message': 'deadline', 'timed_out': True, 'kind': 'run_cancelled'},
        ),
    ],
)
def test_call_tool_result_json_payload_goldens(value: CallToolResult, expected: dict[str, Any]) -> None:
    assert JSON_CODEC.dump(CallToolResult, value) == expected
    assert IDENTITY_CODEC.dump(CallToolResult, value) is value
    assert IDENTITY_CODEC.load(CallToolResult, value) is value


def _source_tree(source: type | Callable[..., object]) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(source)))


def _caught_exception_names(tree: ast.AST) -> set[str]:
    return {
        exception.attr if isinstance(exception, ast.Attribute) else ast.unparse(exception)
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler) and handler.type is not None
        for exception in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])
    }


def test_call_tool_result_carries_every_tool_manager_control_flow_exception() -> None:
    """Every exception core handles around a toolset call must cross a durable unit as a value.

    A durable toolset runs the tool call (or its argument validation) inside a unit:
    `wrap_tool_call_result` turns each control-flow exception into a `CallToolResult` member there,
    and `unwrap_tool_call_result` raises it again on the other side, where `ToolManager` or
    `_ToolCallProcessor._call_tool` handles it. An exception either of them starts handling without a
    matching arm leaves the unit as an ordinary error, so the engine retries or fails the unit instead
    of the agent seeing a retry, a deferral or a failed tool return. The sets are read from the source
    rather than copied, so a new `except` arm in either fails here until it gets a `CallToolResult`
    member or a `handled_caller_side_only` entry with its reason. Not a VCR test: no request differs
    when an arm is missing.
    """
    handled_caller_side_only = {
        'BaseException',  # `handle_call` tells a realtime session its call failed validation, then re-raises
        'Exception',  # routes an ordinary tool error to `on_tool_execute_error`; such an error fails the unit as itself
        'SkipToolExecution',  # raised by tool-execute hooks, which run around the durable tool call, not inside it
        'SkipToolValidation',  # raised by tool-validate hooks, which run around the durable validation, not inside it
        'ToolRetryError',  # core's own wrapper, raised once an unwrapped `ModelRetry` or `ValidationError` is handled
        'ToolFailedError',  # core's own wrapper, raised once an unwrapped `ToolFailed` is handled
        '_ValidationDeferral',  # core's own wrapper for a deferral its argument validator raised
    }
    core_handled = _caught_exception_names(_source_tree(ToolManager)) | _caught_exception_names(
        _source_tree(_ToolCallProcessor[Any, Any]._call_tool)  # pyright: ignore[reportPrivateUsage]
    )
    carried = _caught_exception_names(_source_tree(wrap_tool_call_result))
    reraised = {
        ast.unparse(raised.exc.func.value if isinstance(raised.exc.func, ast.Attribute) else raised.exc.func)
        for raised in ast.walk(_source_tree(unwrap_tool_call_result))
        if isinstance(raised, ast.Raise) and isinstance(raised.exc, ast.Call)
    }
    recorded_members = {
        node.id
        for call in ast.walk(_source_tree(unwrap_recorded_tool_call_result))
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'isinstance'
        for node in ast.walk(call.args[1])
        if isinstance(node, ast.Name)
    }

    # `unwrap_tool_call_result` may raise a subclass of the exception a member carries (`RunTimedOut`).
    toolset_namespace = vars(sys.modules[wrap_tool_call_result.__module__])
    reraised_arms = {
        next((base.__name__ for base in toolset_namespace[name].__mro__ if base.__name__ in carried), name)
        for name in reraised
    }
    assert carried == reraised_arms, '`wrap_tool_call_result` and `unwrap_tool_call_result` disagree'
    assert handled_caller_side_only - core_handled == set(), 'stale `handled_caller_side_only` entries'
    assert carried & handled_caller_side_only == set()
    assert core_handled - handled_caller_side_only == carried, (
        'Add a `CallToolResult` member and wrap/unwrap arms for each new exception core handles, '
        'or list it in `handled_caller_side_only` with its reason.'
    )
    assert recorded_members == {member.__name__ for member in get_union_args(CallToolResult)}, (
        '`unwrap_recorded_tool_call_result` passes a `CallToolResult` member through as a raw recorded value'
    )


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
                        'citations': None,
                        'part_kind': 'text',
                    }
                ],
                'usage': {
                    'input_tokens': 0,
                    'cache_write_tokens': 0,
                    'cache_write_1h_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 0,
                    'output_reasoning_tokens': 0,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'audio_seconds': 0.0,
                    'details': {},
                    'cost': None,
                },
                'model_name': None,
                'timestamp': '2020-01-01T00:00:00Z',
                'kind': 'response',
                'provider_name': None,
                'provider_url': None,
                'canonical_system': None,
                'provider_details': None,
                'provider_response_id': None,
                'finish_reason': None,
                'run_id': None,
                'conversation_id': None,
                'metadata': None,
                'workspace_ref': None,
                'history_digest': None,
                'failed_attempts': None,
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
                    'unless_tool_kind': None,
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


def test_capability_operation_result_payload_golden() -> None:
    delta = RunUsage(requests=1, tool_calls=2, input_tokens=3, details={'cached': 4})
    result = CapabilityOperationResult(5, usage_delta=delta)
    result_type = capability_operation_result_type(int)

    assert JSON_CODEC.dump(result_type, result) == {
        'value': 5,
        'usage_delta': {
            'input_tokens': 3,
            'cache_write_tokens': 0,
            'cache_write_1h_tokens': 0,
            'cache_read_tokens': 0,
            'output_tokens': 0,
            'output_reasoning_tokens': 0,
            'input_audio_tokens': 0,
            'cache_audio_read_tokens': 0,
            'output_audio_tokens': 0,
            'audio_seconds': 0.0,
            'details': {'cached': 4},
            'cost': None,
            'requests': 1,
            'tool_calls': 2,
        },
    }
    assert IDENTITY_CODEC.dump(result_type, result) is result


def test_model_request_context_projection_payload_golden() -> None:
    projection = ModelRequestContextProjection(
        [],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        model_id='restricted',
        streaming=False,
    )

    assert JSON_CODEC.dump(ModelRequestContextProjection, projection) == {
        'messages': [],
        'model_settings': None,
        'model_request_parameters': {
            'function_tools': [],
            'native_tools': [],
            'tool_visibility': None,
            'revealed_tool_names': [],
            'introduced_tool_names': [],
            'deferred_capability_ids': [],
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [],
            'prompted_output_template': None,
            'allow_text_output': True,
            'allow_image_output': False,
            'instruction_parts': None,
            'instruction_visibility': None,
            'thinking': None,
            'cache': None,
        },
        'model_id': 'restricted',
        'streaming': False,
    }


def test_model_request_context_projection_is_exhaustive() -> None:
    """Every `ModelRequestContext` field must be projected across a durable boundary or excluded with a reason.

    `ModelRequestContextProjection` carries a model-request hook's context into a durable unit
    (`build_context`) and the hook's changes back out (`apply`). A field it leaves out is reset to
    its default inside the unit, and whatever a durable hook sets on it is dropped on the way back.
    This fails when a field is added to `ModelRequestContext` until it is projected or listed in
    `intentionally_not_projected` with its reason. Not a VCR test: the projection never reaches a
    provider request.
    """
    intentionally_not_projected = {
        # Live `Model`, not serializable; crosses as `model_id`, re-resolved worker-side, and
        # caller-side only when the hook changed it.
        'model',
        # Private usage record the agent graph appends to caller-side as the model responds; `apply`
        # updates the caller's own context in place, so the record stays with the run.
        '_usage_response_ledger',
        # The live wrapper a capability's `get_wrapper_model` built around `model`, with the model it wraps; the
        # durable unit re-applies the hook to the model it rebuilds, so it's never carried across.
        '_serving_wrapper',
    }
    projected = {field.name for field in fields(ModelRequestContextProjection)}
    context_fields = {field.name for field in fields(ModelRequestContext)}

    assert projected - context_fields == set(), 'projection fields with no `ModelRequestContext` counterpart'
    assert projected & intentionally_not_projected == set(), 'fields both projected and excluded'
    assert context_fields - projected - intentionally_not_projected == set(), (
        'Project each new `ModelRequestContext` field in `ModelRequestContextProjection` (`from_context`, '
        '`build_context` and `apply`), or list it in `intentionally_not_projected` with its reason.'
    )

    # Field names alone don't prove the wiring: every projected field must also make it through
    # `from_context` and back out through both `build_context` (worker-side) and `apply` (caller-side).
    model = TestModel()
    model_settings: ModelSettings = {'temperature': 0.5}
    context = ModelRequestContext(
        model=model,
        messages=[ModelRequest(parts=[UserPromptPart('hello')])],
        model_settings=model_settings,
        model_request_parameters=ModelRequestParameters(allow_text_output=False),
        model_id='test:alias',
        streaming=True,
    )
    default_context = ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    assert [name for name in projected if getattr(context, name) == getattr(default_context, name)] == [], (
        'give each projected field a non-default value in `context` so the round trip below checks it'
    )
    projection = ModelRequestContextProjection.from_context(context)
    assert projection.build_context(model) == context
    projection.apply(default_context)
    assert default_context == context


@dataclass(kw_only=True)
class GoldenProgressEvent(CustomEvent, name='golden_progress'):
    percent: int


@dataclass(kw_only=True)
class GoldenCheckpointEvent(CapabilityEvent, namespace='golden', name='checkpoint'):
    label: str


def test_event_payload_goldens() -> None:
    """Pin the wire shape of both event families, tag included.

    An event's tag is derived from its class name unless an explicit `name=` overrides it, and it
    rides Temporal activity history, Prefect cache keys, and the buffered event stream in
    `GraphAgentState`. Renaming a class would silently change the tag, so the tags below are a
    compatibility surface rather than an implementation detail.
    """
    assert JSON_CODEC.dump(AgentStreamEvent, GoldenProgressEvent(percent=50)) == {
        'name': 'golden_progress',
        'tool_call_id': None,
        'tool_name': None,
        'event_kind': 'custom',
        'percent': 50,
    }
    assert JSON_CODEC.dump(AgentStreamEvent, GoldenCheckpointEvent(label='start', capability_id='golden')) == {
        'kind': 'golden.checkpoint',
        'capability_id': 'golden',
        'tool_call_id': None,
        'tool_name': None,
        'event_kind': 'capability',
        'label': 'start',
    }


def _union_members(union: object) -> set[type]:
    """The classes a union admits, with nested unions flattened and `Annotated` wrappers removed."""
    members: set[type] = set()
    for member in get_union_args(union):
        if get_union_args(member):
            members |= _union_members(member)
        else:
            members.add(member)
    return members


def test_every_stream_event_and_message_part_round_trips_through_the_json_codec() -> None:
    """Every member of the event and part unions must survive the codec a durable boundary uses.

    Events cross a durable unit one at a time as `AgentStreamEvent` (the event-stream-handler
    operation), and parts cross inside `ModelMessage` lists (model request parameters, the model
    request context projection, the returned `ModelResponse`). A new union member with a field the
    codec can't serialize or rebuild breaks every durable run that produces it. The coverage asserts
    fail until a new member has an instance below, so it is round-tripped too. Not a VCR test: the
    round trip happens between engine and worker, never on a provider request.
    """
    request_parts: list[ModelRequestPart] = [
        SystemPromptPart('system'),
        UserPromptPart('user'),
        SpeechPart(speaker='user'),
        ToolSearchReturnPart(content={'discovered_tools': []}),
        LoadCapabilityReturnPart(content={'instructions': 'loaded'}),
        ToolReturnPart('tool', 'result', tool_call_id='call'),
        RetryFeedbackPart('retry', retry=RetryInfo('model_retry')),
        ToolAvailabilityDeltaPart(),
        ModelSettingsDeltaPart(settings={'thinking': 'high'}),
        InstructionDeltaPart(id=None, content='delta'),
        InstructionBaselinePart(instructions=[InstructionPart('baseline')]),
        CompactionPart(content='summary'),
    ]
    response_parts: list[ModelResponsePart] = [
        TextPart('text'),
        ToolSearchCallPart(),
        LoadCapabilityCallPart(),
        ToolCallPart('tool', {'arg': 1}, tool_call_id='call'),
        NativeToolSearchCallPart(),
        NativeToolCallPart('web_search', {'query': 'q'}, tool_call_id='native'),
        NativeToolSearchReturnPart(content={'discovered_tools': []}),
        NativeToolReturnPart('web_search', {'results': []}, tool_call_id='native'),
        ThinkingPart('thinking'),
        CompactionPart(),
        FilePart(content=BinaryContent(data=b'file', media_type='application/pdf')),
        SpeechPart(speaker='assistant'),
    ]
    deltas: list[ModelResponsePartDelta] = [
        TextPartDelta('text'),
        ThinkingPartDelta(content_delta='thinking'),
        ToolCallPartDelta(args_delta='{}'),
        SpeechPartDelta(),
    ]
    events: list[AgentStreamEvent] = [
        *(PartStartEvent(index=0, part=part) for part in response_parts),
        *(PartDeltaEvent(index=0, delta=delta) for delta in deltas),
        PartEndEvent(index=0, part=TextPart('text')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        EnqueuedMessagesEvent(enqueue_id='enqueued', messages=(ModelRequest(parts=[UserPromptPart('steer')]),)),
        FunctionToolCallEvent(part=ToolCallPart('tool', tool_call_id='call')),
        FunctionToolResultEvent(part=ToolReturnPart('tool', 'result', tool_call_id='call')),
        ToolAvailabilityDeltaEvent(part=ToolAvailabilityDeltaPart()),
        ModelSettingsDeltaEvent(part=ModelSettingsDeltaPart(settings={'thinking': 'high'})),
        OutputToolCallEvent(part=ToolCallPart('final_result', tool_call_id='output')),
        OutputToolResultEvent(part=ToolReturnPart('final_result', 'done', tool_call_id='output')),
        DeferredToolRequestsEvent(requests=DeferredToolRequests(calls=[ToolCallPart('tool', tool_call_id='call')])),
        DeferredToolResultsEvent(results=DeferredToolResults(calls={'call': 'result'})),
        RealtimeTurnCompleteEvent(),
        RealtimeInputSpeechStartEvent(),
        RealtimeInputSpeechEndEvent(),
        RealtimeOutputSpeechStartEvent(),
        RealtimeOutputSpeechEndEvent(),
        RealtimeResponseInterruptedEvent(),
        RealtimeInputTranscriptionErrorEvent(message='transcription failed'),
        RealtimeSessionReconnectEvent(),
        RealtimeSessionErrorEvent(message='session failed'),
        GoldenProgressEvent(percent=50),
        GoldenCheckpointEvent(label='start', capability_id='golden'),
    ]

    # `RetryPromptPart` stays in the union for type checking and the published JSON schema: the
    # discriminator translates a legacy payload into one of its replacements, so it never
    # round-trips as itself. Parts and deltas match by exact type, since several members subclass
    # another member; events match by `isinstance`, as `CustomEvent` and `CapabilityEvent` are
    # covered by registered subclasses.
    assert {
        member.__name__
        for member in _union_members(ModelRequestPart)
        if not any(type(part) is member for part in request_parts)
    } == {'RetryPromptPart'}
    assert [
        member.__name__
        for member in _union_members(ModelResponsePart)
        if not any(type(part) is member for part in response_parts)
    ] == []
    assert [
        member.__name__
        for member in _union_members(ModelResponsePartDelta)
        if not any(type(delta) is member for delta in deltas)
    ] == []
    assert [
        member.__name__
        for member in _union_members(AgentStreamEvent)
        if not any(isinstance(event, member) for event in events)
    ] == []

    assert [
        event
        for event in events
        if JSON_CODEC.load(AgentStreamEvent, JSON_CODEC.dump(AgentStreamEvent, event)) != event
    ] == []
    messages: list[ModelMessage] = [ModelRequest(parts=request_parts), ModelResponse(parts=response_parts)]
    assert JSON_CODEC.load(list[ModelMessage], JSON_CODEC.dump(list[ModelMessage], messages)) == messages


def test_pre_wrapper_tool_result_upgrade_paths() -> None:
    raw_payload = {'answer': 42}
    assert unwrap_recorded_tool_call_result(raw_payload) is raw_payload
    with pytest.raises(ValidationError):
        JSON_CODEC.load(CallToolResult, raw_payload)


async def test_validation_error_crosses_call_tool_result_boundary() -> None:
    async def invalid_tool_body() -> None:
        TypeAdapter(int).validate_python('invalid-from-tool-body')

    payload = await wrap_tool_call_result(invalid_tool_body())
    assert isinstance(payload, _ValidationError)
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    async def typed(value: int) -> None:
        pass

    toolset = FunctionToolset(tools=[typed])
    tool = (await toolset.get_tools(RunContext(deps=None, model=TestModel(), usage=RunUsage())))['typed']
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    payload = await wrap_tool_call_result(validate_tool_args(tool, {'value': 'not-an-int'}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    def non_serializable_input(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python(object())

    unsafe_toolset = FunctionToolset(tools=[Tool(typed, args_validator=non_serializable_input)])
    unsafe_tool = (await unsafe_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(unsafe_tool, {'value': 1}, ctx))
    dumped = JSON_CODEC.dump(CallToolResult, payload)
    sanitized = dumped['errors'][0]['input']
    assert sanitized['type'] == 'builtins.object'
    assert sanitized['repr'].startswith('<object object at 0x')

    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError('broken repr')

    def broken_repr_input(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python(BrokenRepr())

    broken_toolset = FunctionToolset(tools=[Tool(typed, args_validator=broken_repr_input)])
    broken_tool = (await broken_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(broken_tool, {'value': 1}, ctx))
    dumped = JSON_CODEC.dump(CallToolResult, payload)
    assert dumped['errors'][0]['input'] == {
        'type': f'{BrokenRepr.__module__}.{BrokenRepr.__qualname__}',
        'repr': '<repr failed>',
    }

    def invalid_args_validator(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python('invalid-from-args-validator')

    validated_toolset = FunctionToolset(tools=[Tool(typed, args_validator=invalid_args_validator)])
    validated_tool = (await validated_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(validated_tool, {'value': 1}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    async def invalid_async_args_validator(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python('invalid-from-async-args-validator')

    async_validated_toolset = FunctionToolset(tools=[Tool(typed, args_validator=invalid_async_args_validator)])
    async_validated_tool = (await async_validated_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(async_validated_tool, {'value': 1}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)


@pytest.mark.parametrize('exc_type', [RunCancelled, RunTimedOut])
async def test_run_cancelled_crosses_call_tool_result_boundary(exc_type: type[RunCancelled]) -> None:
    """A sub-agent's `RunCancelled` raised in a durable unit is raised again, with its type, on the other side.

    `_ToolCallProcessor._call_tool` then turns it into a failed tool return, as it does in-process.
    The payload goes through `JSON_CODEC` as it would across a Temporal activity boundary. Not a VCR
    test: the exception type never reaches a model request, only hooks on the calling side.
    """

    async def cancelled_sub_agent() -> None:
        raise exc_type('the sub-agent stopped')

    payload = JSON_CODEC.load(
        CallToolResult, JSON_CODEC.dump(CallToolResult, await wrap_tool_call_result(cancelled_sub_agent()))
    )
    with pytest.raises(exc_type, match='the sub-agent stopped') as exc_info:
        unwrap_recorded_tool_call_result(payload)
    assert type(exc_info.value) is exc_type
