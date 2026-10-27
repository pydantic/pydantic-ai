"""Pydantic AI contracts used at Render's process boundary.

The operation parameters, results and the reconstructed `RunContext` come from the public
`pydantic_ai.durable_exec` builder surface. Private context attributes and the effective
tracer lookup remain here. `test_local_runtime_live.py` checks context, usage, events and
tracing in separate Render processes.

Public backend contract: https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/
"""

from __future__ import annotations

import json
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar, overload, runtime_checkable

from opentelemetry.trace import NoOpTracer, Tracer
from pydantic import TypeAdapter
from typing_extensions import TypeVar as TypeVarExtensions

from pydantic_ai import Agent
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BoundDurableOperation,
    CallToolResult,
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    DurabilityEngineSpec,
    DurableOperation,
    DurableRunContext,
    DynamicToolsetCallToolParams,
    DynamicToolsResult,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestContextProjection,
    ModelRequestParams,
    ParameterTransport,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
    WorkspaceCall,
    WorkspaceCallParams,
    WorkspaceCallResult,
    capability_operation_result_type,
    revalidate_function_call,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = (
    'BoundDurableOperation',
    'CallToolResult',
    'CapabilityMethodDeclaration',
    'CapabilityOperationParams',
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
CustomEventT = TypeVar('CustomEventT', bound=CustomEvent)
CapabilityEventT = TypeVar('CapabilityEventT', bound=CapabilityEvent)

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
    def model_name(self) -> str:  # pragma: no cover - placeholder is replaced before model use
        return 'render-compaction-placeholder'

    @property
    def system(self) -> str:  # pragma: no cover - placeholder is replaced before model use
        return 'render'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:  # pragma: no cover - placeholder is replaced before model use
        del messages, model_settings, model_request_parameters
        raise RuntimeError('The compaction model placeholder must be replaced before use.')


def to_json_object(value: object) -> JSONObject:
    normalized = normalize_json_value(value)
    if not isinstance(normalized, dict):  # pragma: no cover - callers pass validated JSON objects
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
    if isinstance(value, tuple):  # pragma: no cover - public codecs normalize tuples before this boundary
        return [_normalize_json_value(item) for item in _TUPLE_ADAPTER.validate_python(value, strict=True)]
    if isinstance(value, dict):
        mapping = _OPEN_OBJECT_ADAPTER.validate_python(value, strict=True)
        normalized: dict[str, JSONValue] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):  # pragma: no cover - public codecs already normalize mapping keys
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
    if not isinstance(transport, RenderJsonTransport):  # pragma: no cover - installed by this adapter
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
    return None  # pragma: no cover - every installed operation has a known context parameter


async def prepare_function_call_params(
    toolset: FunctionToolset[ToolDepsT],
    params: ToolsetCallToolParams,
    *,
    engine_spec: DurabilityEngineSpec,
) -> ToolsetCallToolParams:
    """Restore typed function arguments after the Render JSON round trip."""
    params = await revalidate_function_call(toolset, params, engine_spec=engine_spec)
    assert params.tool is not None
    from ._protocol import current_effect_recorder

    if recorder := current_effect_recorder():  # pragma: no branch - called inside operation_task
        recorder.set_event_capability(params.tool.tool_def.capability_id)
    return params


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
    if not isinstance(toolset, _ToolDefinitionResolver):  # pragma: no cover - core selects MCP toolsets
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


class RenderRunContext(DurableRunContext[AgentDepsT]):
    """Restricted run context reconstructed inside a Render child task."""

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        kwargs.setdefault('tracer', NoOpTracer())
        super().__init__(deps, **kwargs)
        from ._protocol import current_effect_recorder

        usage = self.__dict__.get('usage')
        if isinstance(usage, RunUsage) and (recorder := current_effect_recorder()):
            recorder.watch_usage(usage)

    def unavailable_field_message(self, name: str) -> str:
        return f'{name!r} is not available on {self.__class__.__name__!r} inside a Render child task.'

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        from ._protocol import RenderProtocolError, current_effect_recorder

        recorder = current_effect_recorder()
        if recorder is None:  # pragma: no cover - reconstructed context belongs to operation_task
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
            **super().serialize_run_context(ctx),
            # Models contain provider clients and cannot be serialized. Their
            # stable IDs can cross the boundary and resolve worker-side.
            '_model_id': ctx.model_id,
            'tracer_enabled': not isinstance(ctx.tracer, NoOpTracer),
        }


class RenderRunContextCodec(Generic[AgentDepsT]):
    """Encode a run context and dependencies for a fresh Render process."""

    def __init__(
        self,
        *,
        deps_type: type[AgentDepsT],
        agent: AbstractAgent[AgentDepsT, Any] | None,
        resolve_model: Callable[[str | None], Model | None] | None = None,
        engine_spec: DurabilityEngineSpec,
        run_context_type: type[RenderRunContext[Any]] = RenderRunContext,
    ) -> None:
        self._deps_type = deps_type
        self._engine_spec = engine_spec
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
        if model is not None:
            context['model'] = model
        # The codec validates against the complete type form. A second `isinstance`
        # check would reject valid forms such as `dict[str, str]` and `TypedDict`.
        deps_value = JSON_CODEC.load(self._deps_type, payload['deps'])
        return self._run_context_type.restore(
            context,
            deps=deps_value,
            agent=self._agent,
            engine_spec=self._engine_spec,
        )

    def _resolve_tracer(self, model: Model | None) -> Tracer:
        """Recover instrumentation from worker configuration, never from the wire."""
        if isinstance(model, InstrumentedModel):
            return model.instrumentation_settings.tracer
        if isinstance(self._agent, Agent):  # pragma: no branch - binding supplies a concrete Agent
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
        if self._model_resolver is None:  # pragma: no cover - the bound capability supplies its resolver
            return None
        model_id = context.get('_model_id')
        if model_id is not None and not isinstance(model_id, str):
            raise TypeError('Serialized model ID must be a string or null.')
        return self._model_resolver(model_id)
