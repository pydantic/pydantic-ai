from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, with_config

from pydantic_ai import messages as _messages
from pydantic_ai._instrumentation import (
    mcp_telemetry_url_ids,
    mcp_telemetry_url_toolset_ids,
    model_request_telemetry_url_ids,
)
from pydantic_ai.capabilities._dynamic import DynamicCapability
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    capability_operation_result_type,
)
from pydantic_ai.durable_exec._operation import (
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams as _SemanticEventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransports,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._toolset import CallToolResult, DynamicToolsResult, Instructions, revalidate_function_call
from pydantic_ai.durable_exec._workspace import WorkspaceCall, WorkspaceCallParams, WorkspaceCallResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.native_tools import MCPServerTool
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.workspaces import WorkspaceRef

from ._operation_backend import TemporalParameterTransport
from ._spec import TEMPORAL_ENGINE_SPEC
from ._toolset import CallToolParams, GetToolsParams

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.output import OutputDataT
    from pydantic_ai.tools import AgentDepsT
    from pydantic_ai.toolsets import DynamicToolset

    from ._durability import TemporalDurability

__all__ = (
    '_CancelParams',
    '_CapabilityOperationParams',
    '_CapabilityOperationTransport',
    '_CancelTransport',
    '_CompactMessagesTransport',
    '_DynamicCallTransport',
    '_DynamicGetToolsTransport',
    '_EventStreamHandlerParams',
    '_EventStreamHandlerTransport',
    '_FunctionCallTransport',
    '_GetToolsTransport',
    '_MCPCallTransport',
    '_ModelRequestTransport',
    '_RequestParams',
    '_WorkspaceCallTransport',
    '_WorkspaceCallWire',
)


class _FunctionCallTransport(TemporalParameterTransport[ToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: FunctionToolset[Any]) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        assert params.tool is not None
        tool = params.tool
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=tool.tool_def,
                original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> ToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        try:
            tool = (
                self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx, original_name=params.original_name)
                if params.tool_def is not None
                else None
            )
        except KeyError as exc:
            raise UserError(
                f'Tool {params.name!r} not found in toolset {self._toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from exc
        return ToolsetCallToolParams(params.name, tool_args=params.tool_args, ctx=ctx, tool=tool)


class _GetToolsTransport(TemporalParameterTransport[ToolsetGetToolsParams, tuple[GetToolsParams, Any]]):
    wire_type = GetToolsParams
    result_type = dict[str, ToolDefinition]

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ToolsetGetToolsParams) -> tuple[GetToolsParams, Any]:
        return (
            GetToolsParams(serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx)),
            params.ctx.deps,
        )

    def load(self, payload: tuple[GetToolsParams, Any], *, runtime: object) -> ToolsetGetToolsParams:
        params, deps = payload
        return ToolsetGetToolsParams(
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        )


class _GetInstructionsTransport(_GetToolsTransport):
    """An MCP server's `get_instructions` takes the parameters `get_tools` does, and returns its instructions.

    The result type is what the workflow decodes the activity's result as, so it can't be `get_tools`'.
    """

    result_type = Instructions


class _MCPCallTransport(TemporalParameterTransport[ToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: Any) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        assert params.tool is not None
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=params.tool.tool_def,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> ToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        assert params.tool_def is not None
        return ToolsetCallToolParams(
            params.name,
            tool_args=params.tool_args,
            ctx=ctx,
            tool=self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx),
        )


class _DynamicCallTransport(TemporalParameterTransport[DynamicToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: DynamicToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=params.tool_def,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> DynamicToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return DynamicToolsetCallToolParams(params.name, tool_args=params.tool_args, ctx=ctx, tool_def=params.tool_def)


class _DynamicGetToolsTransport(_GetToolsTransport):
    result_type = DynamicToolsResult


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    """Serializable arguments for the model-request Temporal activity."""

    messages: list[_messages.ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    telemetry_mcp_url_ids: set[str] | None = None
    telemetry_mcp_url_toolset_ids: set[str] | None = None
    telemetry_mcp_url_tool_names: set[str] | None = None
    serialized_run_context: Any
    model_id: str | None = None


@dataclass(kw_only=True)
class _CapabilityOperationParams:
    arguments: dict[str, Any]
    serialized_run_context: Any
    model_id: str | None = None


class _CapabilityOperationTransport(
    TemporalParameterTransport[CapabilityOperationParams, tuple[_CapabilityOperationParams, Any]]
):
    wire_type = _CapabilityOperationParams

    def __init__(self, durability: TemporalDurability[Any], declaration: CapabilityMethodDeclaration) -> None:
        self._durability = durability
        self.result_type = capability_operation_result_type(declaration.result_type)

    def dump(self, params: CapabilityOperationParams) -> tuple[_CapabilityOperationParams, Any]:
        return (
            _CapabilityOperationParams(
                arguments=params.arguments,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.run_context),
                model_id=params.model_id,
            ),
            params.run_context.deps,
        )

    def load(self, payload: tuple[_CapabilityOperationParams, Any], *, runtime: object) -> CapabilityOperationParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return CapabilityOperationParams(ctx, arguments=params.arguments, model_id=params.model_id)


@dataclass(kw_only=True)
class _CancelParams:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: Any = None


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CompactMessagesParams:
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    telemetry_mcp_url_ids: set[str] | None = None
    telemetry_mcp_url_toolset_ids: set[str] | None = None
    telemetry_mcp_url_tool_names: set[str] | None = None
    streaming: bool
    instructions: str | None
    serialized_run_context: Any
    model_id: str | None = None


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


def restore_telemetry_mcp_url_ids(
    parameters: ModelRequestParameters,
    url_ids: set[str] | None,
    toolset_ids: set[str] | None,
    tool_names: set[str] | None,
    agent: AbstractAgent[AgentDepsT, OutputDataT] | None,
) -> None:
    """Restore URL provenance omitted from the provider request payload."""
    legacy_dynamic_local_ids: set[str] = set()
    if url_ids is None:
        url_ids = model_request_telemetry_url_ids(parameters)
        explicit_ids: set[tuple[str | None, str | None]] = set()
        if agent is not None:
            configured_capabilities = list(leaf_capabilities(agent.root_capability))
            url_ids.update(mcp_telemetry_url_ids(configured_capabilities))
            if any(isinstance(capability, DynamicCapability) for capability in configured_capabilities):
                # Legacy payloads cannot resolve a dynamic capability's local MCP toolset.
                # Match the metadata MCPToolset writes before treating equal IDs as MCP-owned.
                legacy_dynamic_local_ids = {
                    tool.capability_id
                    for tool in parameters.function_tools
                    if tool.capability_id is not None
                    and tool.capability_id == tool.toolset_id
                    and tool.metadata is not None
                    and {'meta', 'annotations', 'task'} <= tool.metadata.keys()
                    and isinstance(tool.metadata['task'], bool)
                }
                url_ids.update(legacy_dynamic_local_ids)
            for toolset in agent.toolsets:
                url_ids.update(mcp_telemetry_url_ids([], toolset))
            # Legacy payloads omit per-run provenance. Static explicit IDs can be recovered
            # from the agent; dynamic and per-run overrides leave only the resolved native tool.
            for capability in configured_capabilities:
                if isinstance(capability, MCP) and capability._telemetry_url_derived_id is None:  # pyright: ignore[reportPrivateUsage]
                    if isinstance(capability.native, MCPServerTool):
                        explicit_ids.add((capability.native.id, capability.native.url))
                    elif capability.native is not False:
                        # A configured native factory can own an explicit ID before it resolves.
                        explicit_ids.add((capability.id, capability.url))
        # An unconfigured native tool could come from a dynamic or per-run capability, or a
        # direct request. Legacy wire data cannot distinguish an equal caller-chosen ID, so
        # withhold that spelling from telemetry when no configured explicit ID claims it.
        for tool in parameters.native_tools:
            if (
                isinstance(tool, MCPServerTool)
                and (tool.id, tool.url) not in explicit_ids
                and MCP(tool.url, native=True, local=False).id == tool.id
            ):
                url_ids.add(tool.id)
    parameters._telemetry_mcp_url_ids = url_ids  # pyright: ignore[reportPrivateUsage]
    if toolset_ids is None:
        toolset_ids = legacy_dynamic_local_ids
        if agent is not None:
            if isinstance(capability_toolset := agent.root_capability.get_toolset(), AbstractToolset):
                toolset_ids.update(mcp_telemetry_url_toolset_ids(capability_toolset))  # pyright: ignore[reportUnknownArgumentType]
            for toolset in agent.toolsets:
                toolset_ids.update(mcp_telemetry_url_toolset_ids(toolset))
    parameters._telemetry_mcp_url_toolset_ids = toolset_ids  # pyright: ignore[reportPrivateUsage]
    parameters._telemetry_mcp_url_tool_names = tool_names  # pyright: ignore[reportPrivateUsage]


class _ModelRequestTransport(TemporalParameterTransport[ModelRequestParams, tuple[_RequestParams, Any]]):
    wire_type = _RequestParams

    def __init__(self, durability: TemporalDurability[Any], *, result_type: object) -> None:
        self._durability = durability
        self.result_type = result_type

    def dump(self, params: ModelRequestParams) -> tuple[_RequestParams, Any]:
        ctx = params.run_context
        return (
            _RequestParams(
                messages=params.messages,
                model_settings=cast(dict[str, Any] | None, params.model_settings),
                model_request_parameters=params.model_request_parameters,
                telemetry_mcp_url_ids=model_request_telemetry_url_ids(params.model_request_parameters),
                telemetry_mcp_url_toolset_ids=params.model_request_parameters._telemetry_mcp_url_toolset_ids,  # pyright: ignore[reportPrivateUsage]
                telemetry_mcp_url_tool_names=params.model_request_parameters._telemetry_mcp_url_tool_names,  # pyright: ignore[reportPrivateUsage]
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_RequestParams, Any], *, runtime: object) -> ModelRequestParams:
        request, deps = payload
        restore_telemetry_mcp_url_ids(
            request.model_request_parameters,
            request.telemetry_mcp_url_ids,
            request.telemetry_mcp_url_toolset_ids,
            request.telemetry_mcp_url_tool_names,
            self._durability.agent,
        )
        ctx = self._durability.deserialize_operation_run_context(request.serialized_run_context, deps)
        return ModelRequestParams(
            request.model_id,
            messages=request.messages,
            model_settings=cast(ModelSettings | None, request.model_settings),
            model_request_parameters=request.model_request_parameters,
            run_context=ctx,
        )


class _CompactMessagesTransport(
    TemporalParameterTransport[ModelCompactMessagesParams, tuple[_CompactMessagesParams, Any]]
):
    wire_type = _CompactMessagesParams
    result_type = ModelResponse

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ModelCompactMessagesParams) -> tuple[_CompactMessagesParams, Any]:
        ctx = params.run_context
        return (
            _CompactMessagesParams(
                messages=params.request_context.messages,
                model_settings=cast(dict[str, Any] | None, params.request_context.model_settings),
                model_request_parameters=params.request_context.model_request_parameters,
                telemetry_mcp_url_ids=model_request_telemetry_url_ids(params.request_context.model_request_parameters),
                telemetry_mcp_url_toolset_ids=params.request_context.model_request_parameters._telemetry_mcp_url_toolset_ids,  # pyright: ignore[reportPrivateUsage]
                telemetry_mcp_url_tool_names=params.request_context.model_request_parameters._telemetry_mcp_url_tool_names,  # pyright: ignore[reportPrivateUsage]
                streaming=params.request_context.streaming,
                instructions=params.instructions,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_CompactMessagesParams, Any], *, runtime: object) -> ModelCompactMessagesParams:
        params, deps = payload
        restore_telemetry_mcp_url_ids(
            params.model_request_parameters,
            params.telemetry_mcp_url_ids,
            params.telemetry_mcp_url_toolset_ids,
            params.telemetry_mcp_url_tool_names,
            self._durability.agent,
        )
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        request_context = ModelRequestContext(
            model=cast(Model[Any], None),
            messages=params.messages,
            model_settings=cast(ModelSettings | None, params.model_settings),
            model_request_parameters=params.model_request_parameters,
        )
        request_context.model_id = params.model_id
        request_context.streaming = params.streaming
        return ModelCompactMessagesParams(
            params.model_id,
            request_context=request_context,
            instructions=params.instructions,
            run_context=ctx,
        )


class _CancelTransport(TemporalParameterTransport[ModelCancelSuspendedResponseParams, tuple[_CancelParams, Any]]):
    wire_type = _CancelParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ModelCancelSuspendedResponseParams) -> tuple[_CancelParams, Any]:
        ctx = params.run_context
        return (
            _CancelParams(
                response=params.response,
                model_id=params.model_id,
                serialized_run_context=(
                    self._durability.run_context_type.serialize_run_context(ctx) if ctx is not None else None
                ),
            ),
            ctx.deps if ctx is not None else None,
        )

    def load(self, payload: tuple[_CancelParams, Any], *, runtime: object) -> ModelCancelSuspendedResponseParams:
        params, deps = payload
        ctx = (
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
            if params.serialized_run_context is not None
            else None
        )
        return ModelCancelSuspendedResponseParams(params.model_id, response=params.response, run_context=ctx)


class _EventStreamHandlerTransport(
    TemporalParameterTransport[_SemanticEventStreamHandlerParams, tuple[_EventStreamHandlerParams, Any]]
):
    wire_type = _EventStreamHandlerParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _SemanticEventStreamHandlerParams) -> tuple[_EventStreamHandlerParams, Any]:
        ctx = params.run_context
        return (
            _EventStreamHandlerParams(
                event=params.event,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
            ),
            ctx.deps,
        )

    def load(
        self, payload: tuple[_EventStreamHandlerParams, Any], *, runtime: object
    ) -> _SemanticEventStreamHandlerParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return _SemanticEventStreamHandlerParams(params.event, run_context=ctx)


@dataclass(kw_only=True)
class _WorkspaceCallWire:
    call: WorkspaceCall
    ref: WorkspaceRef | None
    serialized_run_context: Any


class _WorkspaceCallTransport(TemporalParameterTransport[WorkspaceCallParams, tuple[_WorkspaceCallWire, Any]]):
    wire_type = _WorkspaceCallWire
    result_type = WorkspaceCallResult

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: WorkspaceCallParams) -> tuple[_WorkspaceCallWire, Any]:
        ctx = params.run_context
        serialized_run_context = self._durability.run_context_type.serialize_run_context(ctx)
        return _WorkspaceCallWire(
            call=params.call, ref=params.ref, serialized_run_context=serialized_run_context
        ), ctx.deps

    def load(self, payload: tuple[_WorkspaceCallWire, Any], *, runtime: object) -> WorkspaceCallParams:
        wire, deps = payload
        ctx = self._durability.deserialize_operation_run_context(wire.serialized_run_context, deps)
        return WorkspaceCallParams(run_context=ctx, ref=wire.ref, call=wire.call)


class TemporalParameterTransports(ParameterTransports):
    """Carry every operation's parameters into a Temporal activity."""

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def model_request(self, result_type: object) -> _ModelRequestTransport:
        return _ModelRequestTransport(self._durability, result_type=result_type)

    def compact_messages(self) -> _CompactMessagesTransport:
        return _CompactMessagesTransport(self._durability)

    def cancel_suspended_response(self) -> _CancelTransport:
        return _CancelTransport(self._durability)

    def event_stream_handler(self) -> _EventStreamHandlerTransport:
        return _EventStreamHandlerTransport(self._durability)

    def capability_operation(self, declaration: CapabilityMethodDeclaration) -> _CapabilityOperationTransport:
        return _CapabilityOperationTransport(self._durability, declaration)

    def workspace_call(self) -> _WorkspaceCallTransport:
        return _WorkspaceCallTransport(self._durability)

    def function_call(self, toolset: FunctionToolset[Any]) -> _FunctionCallTransport:
        return _FunctionCallTransport(self._durability, toolset)

    async def prepare_function_call(
        self, toolset: FunctionToolset[Any], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        return await revalidate_function_call(toolset, params, engine_spec=TEMPORAL_ENGINE_SPEC)

    def mcp_call(self, toolset: AbstractToolset[Any]) -> _MCPCallTransport:
        return _MCPCallTransport(self._durability, toolset)

    def get_tools(self, toolset: AbstractToolset[Any]) -> _GetToolsTransport:
        return _GetToolsTransport(self._durability)

    def get_instructions(self, toolset: AbstractToolset[Any]) -> _GetInstructionsTransport:
        return _GetInstructionsTransport(self._durability)

    def dynamic_get_tools(self, toolset: DynamicToolset[Any]) -> _DynamicGetToolsTransport:
        return _DynamicGetToolsTransport(self._durability)

    def dynamic_call(self, toolset: DynamicToolset[Any]) -> _DynamicCallTransport:
        return _DynamicCallTransport(self._durability)
