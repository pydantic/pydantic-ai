"""Compatibility boundary for unpublished Pydantic AI durability semantics.

Cross-process operation parameters, capability ownership, capability-operation
discovery, effective instrumentation settings, and partial `RunContext`
reconstruction do not yet have public APIs.
Their private imports and unavoidable dynamic typing stay here so an upstream
change has one repair point. Vendor and general framework internals do not belong
in this module.
"""

from __future__ import annotations

import json
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar, overload, runtime_checkable

from opentelemetry.trace import NoOpTracer, Tracer
from pydantic import TypeAdapter
from typing_extensions import TypeVar as TypeVarExtensions

from pydantic_ai import Agent
from pydantic_ai._run_context import AnchoredEvidence, CapabilityEventT, CustomEventT
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities, select_workspace
from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    ModelRequestContextProjection,
    capability_operation_result_type,
    collect_capability_operations,
)
from pydantic_ai.durable_exec._operation import (
    DurableOperation,
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DynamicToolsResult,
    EnqueueGuard,
    enqueue_not_supported_message,
    validation_context_from_agent,
)
from pydantic_ai.durable_exec._workspace import WorkspaceCall, WorkspaceCallParams, WorkspaceCallResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE, UnavailableWorkspace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = (
    'BoundDurableOperation',
    'CallToolResult',
    'CapabilityMethodDeclaration',
    'CapabilityOperationParams',
    'CapabilityOwnedToolset',
    'DynamicToolsResult',
    'DynamicToolsetCallToolParams',
    'DurableOperation',
    'EventStreamHandlerParams',
    'JSONObject',
    'JSONValue',
    'ModelCancelSuspendedResponseParams',
    'ModelCompactMessagesParams',
    'ModelRequestContextProjection',
    'ModelRequestParams',
    'RenderJsonTransport',
    'RenderRunContext',
    'RenderRunContextCodec',
    'ToolsetCallToolParams',
    'ToolsetGetToolsParams',
    'WorkspaceCall',
    'WorkspaceCallParams',
    'WorkspaceCallResult',
    'capability_operation_result_type',
    'get_capability_operation_declaration',
    'dump_json_object',
    'dump_operation_params',
    'function_tool_original_name',
    'load_json_object',
    'load_json_type',
    'load_operation_params',
    'make_model_request_context',
    'model_settings_from_json',
    'model_settings_to_json',
    'normalize_json_value',
    'operation_run_context',
    'prepare_function_call_params',
    'reject_unidentified_operation_capabilities',
    'resolve_function_tool_for_definition',
    'resolve_mcp_tool_for_definition',
    'to_json_object',
)

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list['JSONValue'] | dict[str, 'JSONValue']
# Object values remain `object` statically because Python 3.10 cannot declare a
# named recursive alias that Pydantic 2.12 can rebuild reliably on Python 3.14.
# `to_json_object` performs the real recursive JSON validation at runtime.
JSONObject: TypeAlias = dict[str, object]

T = TypeVar('T')
ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
AgentDepsT = TypeVarExtensions('AgentDepsT', default=object)
ToolDepsT = TypeVar('ToolDepsT')

_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_OPEN_OBJECT_ADAPTER: TypeAdapter[dict[object, object]] = TypeAdapter(dict[object, object])
_LIST_ADAPTER: TypeAdapter[list[object]] = TypeAdapter(list[object])
_TUPLE_ADAPTER: TypeAdapter[tuple[object, ...]] = TypeAdapter(tuple[object, ...])


class RenderJsonTransport(ParameterTransport[ParamsT, JSONObject], ABC):
    """Nominal base for every parameter transport this integration installs.

    The framework's transport contract is generic over its wire type. Every
    Render operation crosses as a JSON object, so fixing the wire here lets
    `load_operation_params` narrow on a declared base instead of asserting a
    wire type it cannot see.
    """

    wire_type = dict


@runtime_checkable
class _ToolDefinitionResolver(Protocol):
    """The tool-reconstruction hook an MCP toolset publishes with no shared base.

    `FunctionToolset` declares `tool_for_tool_def` statically, so function tools
    are rebuilt through that public signature. `AbstractToolset` does not declare
    it, and the capability hands MCP toolsets over as the abstract type, so this
    structural protocol is the checked narrowing point for that one call.

    The deps type is erased exactly as Pydantic AI erases it on the operation
    parameter records this feeds (`ToolsetCallToolParams.ctx` is `RunContext[Any]`
    and its `tool` is `ToolsetTool[Any]`). Parameterizing it instead would make
    `isinstance` narrow to a partially unknown type, which is strictly worse.
    """

    def tool_for_tool_def(self, tool_def: ToolDefinition, *, ctx: RunContext[Any]) -> ToolsetTool[Any]: ...


class _CompactionModelPlaceholder(Model):
    """Inert public-model adapter replaced before compaction is invoked."""

    @property
    def model_name(self) -> str:
        return 'render-compaction-placeholder'

    @property
    def system(self) -> str:
        return 'render'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        del messages, model_settings, model_request_parameters
        raise RuntimeError('The compaction model placeholder must be replaced before use.')


def to_json_object(value: object) -> JSONObject:
    normalized = normalize_json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError(f'Expected a JSON object, got {type(normalized).__name__}.')
    return _JSON_OBJECT_ADAPTER.validate_python(normalized, strict=True)


def normalize_json_value(value: object) -> JSONValue:
    """Validate and normalize an encoded value without coercing mapping keys."""
    normalized = _normalize_json_value(value)
    # The recursive check gives mapping keys their Python semantics before the
    # encoder can turn keys such as `1` into strings.
    json.dumps(normalized, allow_nan=False)
    return normalized


def _normalize_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in _LIST_ADAPTER.validate_python(value, strict=True)]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in _TUPLE_ADAPTER.validate_python(value, strict=True)]
    if isinstance(value, dict):
        mapping = _OPEN_OBJECT_ADAPTER.validate_python(value, strict=True)
        normalized: dict[str, JSONValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f'JSON object keys must be strings, got {type(key).__name__}.')
            normalized[key] = _normalize_json_value(item)
        return normalized
    raise TypeError(f'Expected a JSON value, got {type(value).__name__}.')


def dump_json_object(type_form: object, value: object) -> JSONObject:
    """Encode a typed value using Pydantic AI's registered JSON codec."""
    return to_json_object(JSON_CODEC.dump(type_form, value))


def dump_operation_params(operation: DurableOperation[ParamsT, WireT, ResultT], params: ParamsT) -> JSONObject:
    """Dump one operation through a transport that promises a JSON-object wire."""
    return to_json_object(operation.parameter_transport.dump(params))


def load_operation_params(
    operation: DurableOperation[ParamsT, WireT, ResultT],
    payload: JSONObject,
    *,
    runtime: object,
) -> ParamsT:
    """Load one operation from the common Render JSON-object wire.

    Pydantic's generic registered-backend contract permits any `WireT`, while
    this integration only installs transports whose wire is `JSONObject`.
    Reject an incompatible transport rather than asserting its wire type.
    """
    transport = operation.parameter_transport
    if not isinstance(transport, RenderJsonTransport):
        raise TypeError(f'{type(transport).__name__} does not accept the Render JSON-object wire.')
    json_transport: RenderJsonTransport[ParamsT] = transport
    return json_transport.load(payload, runtime=runtime)


def operation_run_context(params: object) -> RunContext[Any] | None:
    """Return the live caller context carried by a Pydantic AI operation."""
    if isinstance(params, ToolsetCallToolParams | DynamicToolsetCallToolParams | ToolsetGetToolsParams):
        return params.ctx
    if isinstance(
        params,
        ModelRequestParams
        | ModelCompactMessagesParams
        | CapabilityOperationParams
        | EventStreamHandlerParams
        | ModelCancelSuspendedResponseParams
        | WorkspaceCallParams,
    ):
        return params.run_context
    return None


async def prepare_function_call_params(
    agent: AbstractAgent[ToolDepsT, Any],
    toolset: FunctionToolset[ToolDepsT],
    params: ToolsetCallToolParams,
) -> ToolsetCallToolParams:
    """Restore typed function arguments after the Render JSON round trip."""
    tool = params.tool
    if tool is None:
        try:
            tool = (await toolset.get_tools(params.ctx))[params.name]
        except KeyError as exc:
            raise UserError(
                f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Render Workflows.'
            ) from exc
    from ._protocol import current_effect_recorder

    if recorder := current_effect_recorder():
        recorder.set_event_capability(tool.tool_def.capability_id)
    args = tool.args_validator.validate_python(
        params.tool_args,
        context=validation_context_from_agent(agent)(params.ctx),
    )
    return ToolsetCallToolParams(params.name, tool_args=args, ctx=params.ctx, tool=tool)


def load_json_type(type_form: type[T], payload: object) -> T:
    """Decode a concrete runtime type from a JSON value.

    `DurabilityCodec.load` is intentionally untyped because Pydantic accepts
    arbitrary type forms. The runtime check contains that unavoidable boundary.
    """
    value = JSON_CODEC.load(type_form, payload)
    if not isinstance(value, type_form):  # pragma: no cover - Pydantic validates concrete classes
        raise TypeError(f'Expected {type_form.__name__}, got {type(value).__name__}.')
    return value


def load_json_object(payload: JSONObject) -> dict[str, Any]:
    """Decode a JSON object for a private semantic parameter with `Any` values."""
    value = JSON_CODEC.load(dict[str, Any], payload)
    if not isinstance(value, dict):  # pragma: no cover - Pydantic validates the declared mapping
        raise TypeError(f'Expected dict, got {type(value).__name__}.')
    # Pydantic validates this exact open mapping type before the adapter narrows it.
    return TypeAdapter(dict[str, Any]).validate_python(value)


def model_settings_to_json(value: ModelSettings | None) -> JSONObject | None:
    """Preserve provider-specific model settings as an open JSON mapping."""
    if value is None:
        return None
    return dump_json_object(dict[str, Any], value)


def model_settings_from_json(value: JSONObject | None) -> ModelSettings | None:
    """Restore the open mapping accepted by the `ModelSettings` TypedDict API."""
    if value is None:
        return None
    # `ModelSettings` declares `timeout` as `httpx.Timeout`, so Pydantic cannot build a schema
    # for the TypedDict itself and `TypeAdapter(ModelSettings)` raises at import time. The
    # decoded open mapping is validated instead, which also keeps the extra keys provider
    # subclasses add. Re-typing that validated mapping as the TypedDict is the one irreducible
    # `Any` seam here: a `dict[str, Any]` is not assignable to a TypedDict.
    settings: Any = load_json_object(value)
    return settings


def function_tool_original_name(tool: object) -> str | None:
    """Read function-tool identity without leaking its private concrete type."""
    if isinstance(tool, FunctionToolsetTool):
        return tool.original_name
    return None


def resolve_function_tool_for_definition(
    toolset: FunctionToolset[ToolDepsT],
    tool_def: ToolDefinition,
    *,
    ctx: RunContext[ToolDepsT],
    original_name: str | None = None,
) -> ToolsetTool[ToolDepsT]:
    """Rebuild a function tool from its definition through the public toolset API."""
    return toolset.tool_for_tool_def(tool_def, ctx=ctx, original_name=original_name)


def resolve_mcp_tool_for_definition(
    toolset: AbstractToolset[ToolDepsT],
    tool_def: ToolDefinition,
    *,
    ctx: RunContext[ToolDepsT],
) -> ToolsetTool[Any]:
    """Rebuild an MCP tool from its definition, narrowing the undeclared hook once."""
    if not isinstance(toolset, _ToolDefinitionResolver):
        raise TypeError(f'{type(toolset).__name__} cannot rebuild a tool from its definition.')
    return toolset.tool_for_tool_def(tool_def, ctx=ctx)


def make_model_request_context(
    *,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    model_id: str | None,
    streaming: bool,
) -> ModelRequestContext:
    """Build the compaction context whose live model is restored by the handler."""
    # The registered child-task handler resolves the model before invoking
    # `model.compact_messages`. This mirrors Pydantic AI's Temporal transport.
    context = ModelRequestContext(
        model=_CompactionModelPlaceholder(),
        messages=messages,
        model_settings=model_settings,
        model_request_parameters=model_request_parameters,
    )
    context.model_id = model_id
    context.streaming = streaming
    return context


def get_capability_operation_declaration(
    capability: AbstractCapability[ToolDepsT], operation: str
) -> CapabilityMethodDeclaration:
    """Resolve one declaration through Pydantic AI's private collector."""
    try:
        return collect_capability_operations(capability)[operation]
    except KeyError as exc:
        raise ValueError(f'Capability {type(capability).__name__!r} has no operation {operation!r}.') from exc


def reject_unidentified_operation_capabilities(root_capability: AbstractCapability[Any]) -> None:
    """Run Pydantic AI's capability-identity check ahead of any task registration.

    `BaseDurabilityCapability.for_agent` reaches the same check in
    `_bind_capability_operations`, which runs only after the engine has bound its
    toolset, model, and event operations. Every one of those registers a task on the
    `Workflows` app, and Render's public API cannot unregister a task, so a capability
    rejected there leaves the app holding a partial, permanent task set. Running the
    same traversal and the same collector first moves the rejection ahead of the first
    `app.task` call.

    The message is restated rather than reached through, because the check lives inside
    the loop that performs the registration. It must be kept identical to the one in
    `pydantic_ai.durable_exec._base.BaseDurabilityCapability._bind_capability_operations`.
    """
    for capability in leaf_capabilities(root_capability):
        # A capability contributing no durable operations needs no `id`, so the collector
        # runs first here exactly as it does upstream.
        if not collect_capability_operations(capability):
            continue
        if capability.id is None:
            raise UserError(
                f'Capability {type(capability).__name__!r} contributes durable operations and needs an explicit '
                '`id` because persisted operation identity and worker-side recovery must remain stable. '
                f"Construct it as `{type(capability).__name__}(id='...')`."
            )


_STR_SET_ADAPTER: TypeAdapter[set[str]] = TypeAdapter(set[str])
_REHYDRATORS: tuple[tuple[str, type[Any], TypeAdapter[Any]], ...] = (
    ('usage', dict, TypeAdapter(RunUsage)),
    ('usage_limits', dict, TypeAdapter(UsageLimits)),
    ('loaded_capability_ids', list, _STR_SET_ADAPTER),
    ('discovered_tool_names', list, _STR_SET_ADAPTER),
    ('available_tool_names', list, _STR_SET_ADAPTER),
    ('active_capability_ids', list, _STR_SET_ADAPTER),
    ('_deferred_capability_ids', list, _STR_SET_ADAPTER),
    ('_anchored_evidence', dict, TypeAdapter(AnchoredEvidence)),
    ('workspace_ref', dict, TypeAdapter(WorkspaceRef)),
)

_NONE_UNLESS_ATTACHED = (
    'agent',
    'root_capability',
    'pending_messages',
    'validation_context',
    'tool_manager',
    'realtime_session',
    '_durable_operations',
    '_run_capabilities_by_id',
    '_run_held_toolsets',
)
_DEFAULTED_UNLESS_CARRIED: tuple[tuple[str, Any], ...] = (('_anchored_evidence', AnchoredEvidence()),)
_RENAMED_FIELDS: tuple[tuple[str, str], ...] = (
    ('capability_loaded', 'capability_active'),
    ('available_capability_ids', 'active_capability_ids'),
)
_GUARDED_FIELDS = frozenset(RunContext.__dataclass_fields__) - {'deps', *_NONE_UNLESS_ATTACHED}


class RenderRunContext(RunContext[AgentDepsT]):
    """Restricted run context reconstructed inside a Render child task."""

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        self.__dict__.setdefault('tracer', NoOpTracer())
        self.__dict__.setdefault('workspace', Workspace(NO_WORKSPACE))
        for old_name, new_name in _RENAMED_FIELDS:
            if old_name in self.__dict__:
                self.__dict__.setdefault(new_name, self.__dict__.pop(old_name))
        for name in _NONE_UNLESS_ATTACHED:
            self.__dict__.setdefault(name, None)
        for name, default in _DEFAULTED_UNLESS_CARRIED:
            self.__dict__.setdefault(name, default)
        for name, wire_type, adapter in _REHYDRATORS:
            if isinstance(value := self.__dict__.get(name), wire_type):
                self.__dict__[name] = adapter.validate_python(value)
        from ._protocol import current_effect_recorder

        usage = self.__dict__.get('usage')
        if isinstance(usage, RunUsage) and (recorder := current_effect_recorder()):
            recorder.watch_usage(usage)
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        if name in _GUARDED_FIELDS and name not in object.__getattribute__(self, '__dataclass_fields__'):
            raise UserError(f'{name!r} is not available on {self.__class__.__name__!r} inside a Render child task.')
        return super().__getattribute__(name)

    @property
    def available_tool_names(self) -> set[str]:
        if (snapshot := self.__dict__.get('available_tool_names')) is not None:
            return snapshot
        return super().available_tool_names

    @property
    def active_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('active_capability_ids')) is not None:
            return snapshot
        return super().active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('_deferred_capability_ids')) is not None:
            return snapshot
        return super()._deferred_capability_ids

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        from ._protocol import RenderProtocolError, current_effect_recorder

        recorder = current_effect_recorder()
        if recorder is None:
            raise UserError(
                'Emitting events from a tool or event stream handler is not supported inside a Render child task.'
            )
        capability_id = recorder.event_capability_id
        if isinstance(event, CapabilityEvent):
            if event.event_dispatch == 'immediate':
                raise RenderProtocolError(
                    'Immediate capability events are unsupported inside a Render child task because '
                    'their listener decision must be available before the operation continues.'
                )
            if event.capability_id is None:
                if capability_id is None:
                    raise UserError(
                        'Capability events belong to capabilities and cannot be emitted from an application tool.'
                    )
                event.capability_id = capability_id
        elif capability_id is not None:
            raise UserError('Capability-contributed tools must emit `CapabilityEvent`, not `CustomEvent`.')
        if event.tool_call_id is None and self.tool_call_id is not None:
            event.tool_call_id = self.tool_call_id
            event.tool_name = self.tool_name
        recorder.record_event(event)
        return event

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Project the serializable context state needed by child operations."""
        return {
            'run_id': ctx.run_id,
            'conversation_id': ctx.conversation_id,
            # Models contain provider clients and cannot be serialized. Their
            # stable IDs can cross the boundary and resolve worker-side.
            '_model_id': ctx.model_id,
            'metadata': ctx.metadata,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'tool_call_metadata': ctx.tool_call_metadata,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'partial_output': ctx.partial_output,
            'trace_include_content': ctx.trace_include_content,
            'tracer_enabled': not isinstance(ctx.tracer, NoOpTracer),
            'instrumentation_version': ctx.instrumentation_version,
            'usage': ctx.usage,
            'usage_limits': ctx.usage_limits,
            'loaded_capability_ids': ctx.loaded_capability_ids,
            'discovered_tool_names': ctx.discovered_tool_names,
            '_anchored_evidence': ctx._anchored_evidence,
            'available_tool_names': ctx.available_tool_names,
            'active_capability_ids': ctx.active_capability_ids,
            '_deferred_capability_ids': ctx._deferred_capability_ids,
            'capability_active': ctx.capability_active,
            'workspace_ref': ctx.workspace.ref,
        }

    @classmethod
    def deserialize_run_context(
        cls, ctx: Mapping[str, Any], deps: AgentDepsT, model: Model | None = None
    ) -> RenderRunContext[AgentDepsT]:
        """Rebuild a restricted context and attach its worker-local model."""
        fields = {**ctx, 'model': model} if model is not None else ctx
        return cls(**fields, deps=deps)


class RenderRunContextCodec(Generic[AgentDepsT]):
    """Encode a run context and dependencies for a fresh Render process."""

    def __init__(
        self,
        *,
        deps_type: type[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None,
        resolve_model: Callable[[str | None], Model | None] | None = None,
        run_context_type: type[RenderRunContext[Any]] = RenderRunContext,
    ) -> None:
        self._deps_type = deps_type
        self._agent = agent
        self._model_resolver = resolve_model
        self._run_context_type = run_context_type

    def dump(self, ctx: RunContext[AgentDepsT]) -> JSONObject:
        context = self._run_context_type.serialize_run_context(ctx)
        return {
            'version': 1,
            'context': dump_json_object(dict[str, Any], context),
            'deps': normalize_json_value(JSON_CODEC.dump(self._deps_type, ctx.deps)),
        }

    def load(self, payload: JSONObject) -> RunContext[AgentDepsT]:
        if payload.get('version') != 1:
            raise ValueError(f'Unsupported Render run-context version: {payload.get("version")!r}.')
        context_payload = payload.get('context')
        if not isinstance(context_payload, dict):
            raise TypeError('Render run-context payload requires a JSON object in `context`.')
        if 'deps' not in payload:
            raise TypeError('Render run-context payload requires `deps`.')

        context = load_json_object(to_json_object(payload['context']))
        tracer_enabled = context.pop('tracer_enabled', False)
        if not isinstance(tracer_enabled, bool):
            raise TypeError('Serialized tracer enablement must be a boolean.')
        model = self._resolve_model(context)
        context['tracer'] = self._resolve_tracer(model) if tracer_enabled else NoOpTracer()
        # The codec validates against the complete type form. A second `isinstance`
        # check would reject valid forms such as `dict[str, str]` and `TypedDict`.
        deps_value = JSON_CODEC.load(self._deps_type, payload['deps'])
        ctx = self._run_context_type.deserialize_run_context(
            context,
            deps=deps_value,
            model=model,
        )
        if self._agent is not None:
            ctx.__dict__['agent'] = self._agent
            ctx.__dict__['root_capability'] = self._agent.root_capability
            self._restore_workspace(ctx)
            ctx.__dict__['validation_context'] = validation_context_from_agent(self._agent)(ctx)
        ctx.__dict__['pending_messages'] = EnqueueGuard(enqueue_not_supported_message('task', 'workflow'))
        return ctx

    def _restore_workspace(self, ctx: RunContext[AgentDepsT]) -> None:
        ref = ctx.__dict__.get('workspace_ref')
        if self._agent is None or not isinstance(ref, WorkspaceRef):
            return
        if ctx.workspace.backend is not NO_WORKSPACE:
            return
        workspace = select_workspace(self._agent.root_capability, ctx, ref=ref)
        if workspace is None:
            workspace = Workspace(
                UnavailableWorkspace(
                    f'No capability on agent {self._agent.name!r} can supply workspace {ref.id!r} '
                    f'from provider {ref.provider!r} inside this Render task. '
                    'Construct the worker with the same workspace capabilities as the entry task.'
                )
            )
        ctx.__dict__['workspace'] = workspace

    def _resolve_tracer(self, model: Model | None) -> Tracer:
        """Recover instrumentation from worker configuration, never from the wire."""
        if isinstance(model, InstrumentedModel):
            return model.instrumentation_settings.tracer
        if isinstance(self._agent, Agent):
            if isinstance(self._agent.model, InstrumentedModel):
                return self._agent.model.instrumentation_settings.tracer
            # Core has no public getter for the effective agent/global settings.
            # Keep that lookup in this compatibility seam instead of duplicating
            # its precedence or losing Agent.instrument_all's custom provider.
            settings = self._agent._resolve_instrumentation_settings()  # pyright: ignore[reportPrivateUsage]
            if settings is not None:
                return settings.tracer
        return NoOpTracer()

    def _resolve_model(self, context: Mapping[str, Any]) -> Model | None:
        """Resolve the serialized model ID against this worker's registry.

        The callback returns the plain model, not the workflow-side wrapper.
        This keeps work performed by a child inside that child task. If the ID
        is unknown, `ctx.model` remains guarded.
        """
        if self._model_resolver is None:
            return None
        model_id = context.get('_model_id')
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError('Serialized model ID must be a string or null.')
        return self._model_resolver(model_id)
