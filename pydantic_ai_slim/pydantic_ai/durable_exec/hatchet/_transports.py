"""Carry each durable operation's parameters into a Hatchet task as JSON.

A Hatchet task can run on another worker, so its input is JSON: each transport dumps an operation's
semantic parameters to a wire dataclass, reduced with `TypeAdapter` to a JSON object alongside the
run's `deps`, and the worker validates them back and rebuilds the `RunContext` from the agent it has.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, fields
from functools import cache
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, with_config

from pydantic_ai import messages as _messages
from pydantic_ai._instrumentation import model_request_telemetry_url_ids
from pydantic_ai.durable_exec._capability_operation import CapabilityMethodDeclaration, CapabilityOperationParams
from pydantic_ai.durable_exec._operation import (
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ParameterTransports,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._toolset import revalidate_function_call
from pydantic_ai.durable_exec._workspace import WorkspaceCall, WorkspaceCallParams
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.workspaces import WorkspaceRef

from ._run_context import stable_capability_ids
from ._spec import HATCHET_ENGINE_SPEC

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.toolsets import DynamicToolset

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')


class TaskInput(BaseModel):
    """The JSON input of every Hatchet task that runs a durable operation."""

    params: JsonValue
    deps: JsonValue


class TransportRuntime(Protocol):
    """What the transports need from `HatchetDurability`, without importing it."""

    @property
    def agent(self) -> AbstractAgent[Any, Any] | None: ...

    def serialize_operation_run_context(self, ctx: RunContext[Any]) -> dict[str, Any]: ...

    def deserialize_operation_run_context(self, serialized: dict[str, Any], deps: Any) -> RunContext[Any]: ...

    def dump_deps(self, deps: Any) -> JsonValue: ...

    def load_deps(self, payload: JsonValue) -> Any: ...


@cache
def _adapter(wire_type: type[Any]) -> TypeAdapter[Any]:
    return TypeAdapter(wire_type)


class _JSONTransport(ParameterTransport[ParamsT, TaskInput], Generic[ParamsT, WireT]):
    """Reduce a wire dataclass and the run's `deps` to a `TaskInput`; subclasses map it to and from `ParamsT`."""

    wire_type: type[WireT]

    def __init__(self, runtime: TransportRuntime) -> None:
        self._runtime = runtime

    def to_wire(self, params: ParamsT) -> tuple[WireT, Any]:
        raise NotImplementedError

    def from_wire(self, wire: WireT, ctx_deps: Any) -> ParamsT:
        raise NotImplementedError

    def dump(self, params: ParamsT) -> TaskInput:
        wire, deps = self.to_wire(params)
        payload = _sort_sets(self.wire_type, _adapter(self.wire_type).dump_python(wire, mode='json'))
        # Tool definitions name their capability by its key too, so the run's synthetic keys are replaced there.
        if (ctx := _params_run_context(params)) is not None and (replacements := stable_capability_ids(ctx)):
            payload = _replace_capability_ids(payload, replacements)
        return TaskInput(params=payload, deps=self._runtime.dump_deps(deps))

    def load(self, payload: TaskInput, *, runtime: object) -> ParamsT:
        wire = cast(WireT, _adapter(self.wire_type).validate_python(payload.params))
        return self.from_wire(wire, self._runtime.load_deps(payload.deps))

    def _serialize(self, ctx: RunContext[Any]) -> dict[str, Any]:
        return self._runtime.serialize_operation_run_context(ctx)

    def _deserialize(self, serialized: dict[str, Any], deps: Any) -> RunContext[Any]:
        return self._runtime.deserialize_operation_run_context(serialized, deps)


def _is_set(annotation: object) -> bool:
    if get_origin(annotation) in (types.UnionType, Union):
        return any(_is_set(arg) for arg in get_args(annotation))
    return get_origin(annotation) in (set, frozenset)


@cache
def _set_fields(wire_type: type[Any]) -> frozenset[str]:
    """The fields of a wire dataclass, or of `ModelRequestParameters` inside one, that hold a set."""
    hints = get_type_hints(wire_type)
    return frozenset(field.name for field in fields(wire_type) if _is_set(hints[field.name]))


def _sort_sets(wire_type: type[Any], payload: Any) -> JsonValue:
    """Sort the lists the wire's sets were dumped as, in place.

    A set's order depends on the process's hash seed, so a replay on a restarted worker would dump it in another
    order, and Hatchet would find the task's input changed.
    """
    for name in _set_fields(wire_type):
        if isinstance(payload.get(name), list):
            payload[name] = sorted(payload[name])
    if isinstance(parameters := payload.get('model_request_parameters'), dict):
        _sort_sets(ModelRequestParameters, parameters)
    return cast(JsonValue, payload)


def _params_run_context(params: object) -> RunContext[Any] | None:
    ctx: object = getattr(params, 'run_context', None) or getattr(params, 'ctx', None)
    return cast('RunContext[Any]', ctx) if isinstance(ctx, RunContext) else None


def _replace_capability_ids(value: JsonValue, replacements: dict[str, str]) -> JsonValue:
    """Replace synthetic capability keys in the fields that hold one, leaving every other value as the caller wrote it."""
    if isinstance(value, list):
        return [_replace_capability_ids(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_field(key, item, replacements) for key, item in value.items()}
    return value


def _replace_field(key: str, value: JsonValue, replacements: dict[str, str]) -> JsonValue:
    if key == 'capability_id' and isinstance(value, str):
        return replacements.get(value, value)
    if key.endswith('capability_ids') and isinstance(value, list):
        return [replacements.get(item, item) if isinstance(item, str) else item for item in value]
    return _replace_capability_ids(value, replacements)


@dataclass(kw_only=True)
class _CallToolWire:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: dict[str, Any]
    tool_def: ToolDefinition | None
    original_name: str | None = None


@dataclass(kw_only=True)
class _RunContextWire:
    serialized_run_context: dict[str, Any]


class _FunctionCallTransport(_JSONTransport[ToolsetCallToolParams, _CallToolWire]):
    wire_type = _CallToolWire

    def __init__(self, runtime: TransportRuntime, toolset: FunctionToolset[Any]) -> None:
        super().__init__(runtime)
        self._toolset = toolset

    def to_wire(self, params: ToolsetCallToolParams) -> tuple[_CallToolWire, Any]:
        assert params.tool is not None
        tool = params.tool
        return (
            _CallToolWire(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._serialize(params.ctx),
                tool_def=tool.tool_def,
                original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
            ),
            params.ctx.deps,
        )

    def from_wire(self, wire: _CallToolWire, ctx_deps: Any) -> ToolsetCallToolParams:
        ctx = self._deserialize(wire.serialized_run_context, ctx_deps)
        try:
            tool = (
                self._toolset.tool_for_tool_def(wire.tool_def, ctx=ctx, original_name=wire.original_name)
                if wire.tool_def is not None
                else None
            )
        except KeyError as exc:
            raise UserError(
                f'Tool {wire.name!r} not found in toolset {self._toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Hatchet.'
            ) from exc
        return ToolsetCallToolParams(wire.name, tool_args=wire.tool_args, ctx=ctx, tool=tool)


class _MCPCallTransport(_JSONTransport[ToolsetCallToolParams, _CallToolWire]):
    wire_type = _CallToolWire

    def __init__(self, runtime: TransportRuntime, toolset: Any) -> None:
        super().__init__(runtime)
        self._toolset = toolset

    def to_wire(self, params: ToolsetCallToolParams) -> tuple[_CallToolWire, Any]:
        assert params.tool is not None
        return (
            _CallToolWire(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._serialize(params.ctx),
                tool_def=params.tool.tool_def,
            ),
            params.ctx.deps,
        )

    def from_wire(self, wire: _CallToolWire, ctx_deps: Any) -> ToolsetCallToolParams:
        ctx = self._deserialize(wire.serialized_run_context, ctx_deps)
        assert wire.tool_def is not None
        return ToolsetCallToolParams(
            wire.name, tool_args=wire.tool_args, ctx=ctx, tool=self._toolset.tool_for_tool_def(wire.tool_def, ctx=ctx)
        )


class _DynamicCallTransport(_JSONTransport[DynamicToolsetCallToolParams, _CallToolWire]):
    wire_type = _CallToolWire

    def to_wire(self, params: DynamicToolsetCallToolParams) -> tuple[_CallToolWire, Any]:
        return (
            _CallToolWire(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._serialize(params.ctx),
                tool_def=params.tool_def,
            ),
            params.ctx.deps,
        )

    def from_wire(self, wire: _CallToolWire, ctx_deps: Any) -> DynamicToolsetCallToolParams:
        ctx = self._deserialize(wire.serialized_run_context, ctx_deps)
        return DynamicToolsetCallToolParams(wire.name, tool_args=wire.tool_args, ctx=ctx, tool_def=wire.tool_def)


class _GetToolsTransport(_JSONTransport[ToolsetGetToolsParams, _RunContextWire]):
    wire_type = _RunContextWire

    def to_wire(self, params: ToolsetGetToolsParams) -> tuple[_RunContextWire, Any]:
        return _RunContextWire(serialized_run_context=self._serialize(params.ctx)), params.ctx.deps

    def from_wire(self, wire: _RunContextWire, ctx_deps: Any) -> ToolsetGetToolsParams:
        return ToolsetGetToolsParams(self._deserialize(wire.serialized_run_context, ctx_deps))


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestWire:
    messages: list[_messages.ModelMessage]
    # Not a `ModelSettings`, which would drop fields only its subclasses define.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    telemetry_mcp_url_ids: set[str] | None = None
    telemetry_mcp_url_toolset_ids: set[str] | None = None
    telemetry_mcp_url_tool_names: set[str] | None = None
    serialized_run_context: dict[str, Any]
    model_id: str | None = None


def _restore_telemetry(parameters: ModelRequestParameters, wire: _RequestWire | _CompactMessagesWire) -> None:
    """Restore the MCP URL provenance the provider request payload omits; telemetry reads it worker-side."""
    parameters._telemetry_mcp_url_ids = wire.telemetry_mcp_url_ids or set()  # pyright: ignore[reportPrivateUsage]
    parameters._telemetry_mcp_url_toolset_ids = wire.telemetry_mcp_url_toolset_ids or set()  # pyright: ignore[reportPrivateUsage]
    parameters._telemetry_mcp_url_tool_names = wire.telemetry_mcp_url_tool_names  # pyright: ignore[reportPrivateUsage]


class _ModelRequestTransport(_JSONTransport[ModelRequestParams, _RequestWire]):
    wire_type = _RequestWire

    def to_wire(self, params: ModelRequestParams) -> tuple[_RequestWire, Any]:
        ctx = params.run_context
        parameters = params.model_request_parameters
        return (
            _RequestWire(
                messages=params.messages,
                model_settings=cast(dict[str, Any] | None, params.model_settings),
                model_request_parameters=parameters,
                telemetry_mcp_url_ids=model_request_telemetry_url_ids(parameters),
                telemetry_mcp_url_toolset_ids=parameters._telemetry_mcp_url_toolset_ids,  # pyright: ignore[reportPrivateUsage]
                telemetry_mcp_url_tool_names=parameters._telemetry_mcp_url_tool_names,  # pyright: ignore[reportPrivateUsage]
                serialized_run_context=self._serialize(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def from_wire(self, wire: _RequestWire, ctx_deps: Any) -> ModelRequestParams:
        _restore_telemetry(wire.model_request_parameters, wire)
        return ModelRequestParams(
            wire.model_id,
            messages=wire.messages,
            model_settings=cast(ModelSettings | None, wire.model_settings),
            model_request_parameters=wire.model_request_parameters,
            run_context=self._deserialize(wire.serialized_run_context, ctx_deps),
        )


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CompactMessagesWire:
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    telemetry_mcp_url_ids: set[str] | None = None
    telemetry_mcp_url_toolset_ids: set[str] | None = None
    telemetry_mcp_url_tool_names: set[str] | None = None
    streaming: bool
    instructions: str | None
    serialized_run_context: dict[str, Any]
    model_id: str | None = None


class _CompactMessagesTransport(_JSONTransport[ModelCompactMessagesParams, _CompactMessagesWire]):
    wire_type = _CompactMessagesWire

    def to_wire(self, params: ModelCompactMessagesParams) -> tuple[_CompactMessagesWire, Any]:
        ctx = params.run_context
        request = params.request_context
        parameters = request.model_request_parameters
        return (
            _CompactMessagesWire(
                messages=request.messages,
                model_settings=cast(dict[str, Any] | None, request.model_settings),
                model_request_parameters=parameters,
                telemetry_mcp_url_ids=model_request_telemetry_url_ids(parameters),
                telemetry_mcp_url_toolset_ids=parameters._telemetry_mcp_url_toolset_ids,  # pyright: ignore[reportPrivateUsage]
                telemetry_mcp_url_tool_names=parameters._telemetry_mcp_url_tool_names,  # pyright: ignore[reportPrivateUsage]
                streaming=request.streaming,
                instructions=params.instructions,
                serialized_run_context=self._serialize(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def from_wire(self, wire: _CompactMessagesWire, ctx_deps: Any) -> ModelCompactMessagesParams:
        _restore_telemetry(wire.model_request_parameters, wire)
        request_context = ModelRequestContext(
            model=cast(Model[Any], None),
            messages=wire.messages,
            model_settings=cast(ModelSettings | None, wire.model_settings),
            model_request_parameters=wire.model_request_parameters,
        )
        request_context.model_id = wire.model_id
        request_context.streaming = wire.streaming
        return ModelCompactMessagesParams(
            wire.model_id,
            request_context=request_context,
            instructions=wire.instructions,
            run_context=self._deserialize(wire.serialized_run_context, ctx_deps),
        )


@dataclass(kw_only=True)
class _CancelWire:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: dict[str, Any] | None = None


class _CancelTransport(_JSONTransport[ModelCancelSuspendedResponseParams, _CancelWire]):
    wire_type = _CancelWire

    def to_wire(self, params: ModelCancelSuspendedResponseParams) -> tuple[_CancelWire, Any]:
        ctx = params.run_context
        return (
            _CancelWire(
                response=params.response,
                model_id=params.model_id,
                serialized_run_context=self._serialize(ctx) if ctx is not None else None,
            ),
            ctx.deps if ctx is not None else None,
        )

    def from_wire(self, wire: _CancelWire, ctx_deps: Any) -> ModelCancelSuspendedResponseParams:
        ctx = (
            self._deserialize(wire.serialized_run_context, ctx_deps)
            if wire.serialized_run_context is not None
            else None
        )
        return ModelCancelSuspendedResponseParams(wire.model_id, response=wire.response, run_context=ctx)


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventWire:
    event: AgentStreamEvent
    serialized_run_context: dict[str, Any]


class _EventStreamHandlerTransport(_JSONTransport[EventStreamHandlerParams, _EventWire]):
    wire_type = _EventWire

    def to_wire(self, params: EventStreamHandlerParams) -> tuple[_EventWire, Any]:
        ctx = params.run_context
        return _EventWire(event=params.event, serialized_run_context=self._serialize(ctx)), ctx.deps

    def from_wire(self, wire: _EventWire, ctx_deps: Any) -> EventStreamHandlerParams:
        return EventStreamHandlerParams(
            wire.event, run_context=self._deserialize(wire.serialized_run_context, ctx_deps)
        )


@dataclass(kw_only=True)
class _CapabilityOperationWire:
    arguments: dict[str, Any]
    serialized_run_context: dict[str, Any]
    model_id: str | None = None


class _CapabilityOperationTransport(_JSONTransport[CapabilityOperationParams, _CapabilityOperationWire]):
    wire_type = _CapabilityOperationWire

    def to_wire(self, params: CapabilityOperationParams) -> tuple[_CapabilityOperationWire, Any]:
        return (
            _CapabilityOperationWire(
                arguments=params.arguments,
                serialized_run_context=self._serialize(params.run_context),
                model_id=params.model_id,
            ),
            params.run_context.deps,
        )

    def from_wire(self, wire: _CapabilityOperationWire, ctx_deps: Any) -> CapabilityOperationParams:
        ctx = self._deserialize(wire.serialized_run_context, ctx_deps)
        return CapabilityOperationParams(ctx, arguments=wire.arguments, model_id=wire.model_id)


@dataclass(kw_only=True)
class _WorkspaceCallWire:
    call: WorkspaceCall
    ref: WorkspaceRef | None
    serialized_run_context: dict[str, Any]


class _WorkspaceCallTransport(_JSONTransport[WorkspaceCallParams, _WorkspaceCallWire]):
    wire_type = _WorkspaceCallWire

    def to_wire(self, params: WorkspaceCallParams) -> tuple[_WorkspaceCallWire, Any]:
        ctx = params.run_context
        return (
            _WorkspaceCallWire(call=params.call, ref=params.ref, serialized_run_context=self._serialize(ctx)),
            ctx.deps,
        )

    def from_wire(self, wire: _WorkspaceCallWire, ctx_deps: Any) -> WorkspaceCallParams:
        ctx = self._deserialize(wire.serialized_run_context, ctx_deps)
        return WorkspaceCallParams(run_context=ctx, ref=wire.ref, call=wire.call)


class HatchetParameterTransports(ParameterTransports):
    """Carry every operation's parameters into a Hatchet task as JSON."""

    def __init__(self, runtime: TransportRuntime) -> None:
        self._runtime = runtime

    def model_request(self, result_type: object) -> _ModelRequestTransport:
        return _ModelRequestTransport(self._runtime)

    def compact_messages(self) -> _CompactMessagesTransport:
        return _CompactMessagesTransport(self._runtime)

    def cancel_suspended_response(self) -> _CancelTransport:
        return _CancelTransport(self._runtime)

    def event_stream_handler(self) -> _EventStreamHandlerTransport:
        return _EventStreamHandlerTransport(self._runtime)

    def capability_operation(self, declaration: CapabilityMethodDeclaration) -> _CapabilityOperationTransport:
        return _CapabilityOperationTransport(self._runtime)

    def workspace_call(self) -> _WorkspaceCallTransport:
        return _WorkspaceCallTransport(self._runtime)

    def function_call(self, toolset: FunctionToolset[Any]) -> _FunctionCallTransport:
        return _FunctionCallTransport(self._runtime, toolset)

    async def prepare_function_call(
        self, toolset: FunctionToolset[Any], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        # The arguments crossed as JSON, so the tool gets them revalidated against its schema.
        return await revalidate_function_call(toolset, params, engine_spec=HATCHET_ENGINE_SPEC)

    def mcp_call(self, toolset: AbstractToolset[Any]) -> _MCPCallTransport:
        return _MCPCallTransport(self._runtime, toolset)

    def get_tools(self, toolset: AbstractToolset[Any]) -> _GetToolsTransport:
        return _GetToolsTransport(self._runtime)

    def get_instructions(self, toolset: AbstractToolset[Any]) -> _GetToolsTransport:
        return _GetToolsTransport(self._runtime)

    def dynamic_get_tools(self, toolset: DynamicToolset[Any]) -> _GetToolsTransport:
        return _GetToolsTransport(self._runtime)

    def dynamic_call(self, toolset: DynamicToolset[Any]) -> _DynamicCallTransport:
        return _DynamicCallTransport(self._runtime)
