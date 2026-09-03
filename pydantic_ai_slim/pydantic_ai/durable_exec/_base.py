from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from functools import partial
from typing import Any, ClassVar, Literal, NamedTuple, Protocol, TypeVar, cast, runtime_checkable
from weakref import ReferenceType, ref

from pydantic_core import PydanticSerializationError
from typing_extensions import Self

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai._utils import aclose_if_supported, get_union_args
from pydantic_ai.agent import Agent, EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
    leaf_capabilities,
)
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    infer_model,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._capability_operation import (
    CapabilityBoundOperation,
    CapabilityCacheIdentity,
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    CapabilityOperationResult,
    ModelRequestContextProjection,
    _ResolvedModelRequestContext,  # pyright: ignore[reportPrivateUsage]
    bind_arguments,
    call_declaration,
    capability_operation_result_type,
    collect_capability_operations,
    recover_capability,
)
from ._codec import IDENTITY_CODEC
from ._operation import (
    CacheIdentity,
    CapabilityOperationId,
    DurableOperation,
    DynamicToolsetCallToolParams,
    EventStreamHandlerId,
    EventStreamHandlerParams,
    IdentityParameterTransport,
    ModelCancelSuspendedResponseId,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesId,
    ModelCompactMessagesParams,
    ModelRequestId,
    ModelRequestParams,
    ParameterTransport,
    ResultCodec,
    ToolsetCallToolId,
    ToolsetCallToolParams,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetGetToolsParams,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
    TypedResultCodec,
)
from ._operation_backend import BoundDurableOperation, DurableOperationBackend, RegisteredOperationBackend
from ._runtime_toolsets import (
    cancellation_token_unsupported_error,
    reject_unsupported_runtime_toolsets,
)
from ._spec import DurabilityEngineSpec
from ._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DurableFunctionToolset,
    DurableMCPToolset,
    DynamicToolsResult,
    Instructions,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
    guard_run_context,
    resolve_tool_durable_config,
    run_args_validator,
    unwrap_recorded_tool_call_result,
    unwrap_tool_call_result,
    validate_dynamic_tool_args,
    wrap_tool_call_result,
)
from ._utils import DurableModel, StreamedActivityResult, capture_event_stream, managed_model_scope, unwrap_model

_T = TypeVar('_T')


def construction_toolsets(agent: AbstractAgent[AgentDepsT, Any]) -> Sequence[AbstractToolset[AgentDepsT]]:
    """The toolsets `agent` was built with, ignoring anything added after construction.

    `AbstractAgent.toolsets` is the wrong list to ask for here: it reflects an active
    `override(toolsets=...)` and includes toolsets a `@agent.toolset` decorator registered after
    construction. Those would land in the known-good set, and the runtime-toolset guard would wave
    through the very thing it exists to catch -- toolsets that arrive after binding and were
    therefore never wrapped for the durable engine.

    Only `Agent` can tell the two lists apart, and only durable execution needs them told apart, so
    the question is asked here rather than widened into a hook that every `AbstractAgent`
    implementation would have to answer. An agent that supports no post-construction additions has
    nothing to subtract, which is what its `toolsets` already reports.
    """
    if isinstance(agent, WrapperAgent):
        return construction_toolsets(agent.wrapped)
    if isinstance(agent, Agent):
        return agent._construction_toolsets  # pyright: ignore[reportPrivateUsage]
    return agent.toolsets


@runtime_checkable
class _RestrictedRunContext(Protocol):
    def _expose_field(self, name: str) -> None: ...


_MODEL_RESPONSE_STREAM_EVENT_TYPES = get_union_args(ModelResponseStreamEvent)


class _BoundModelOperations(NamedTuple):
    request: BoundDurableOperation[ModelRequestParams, Any, ModelResponse]
    request_stream: BoundDurableOperation[ModelRequestParams, Any, StreamedActivityResult]
    compact_messages: BoundDurableOperation[ModelCompactMessagesParams, Any, ModelResponse]
    cancel_suspended_response: BoundDurableOperation[ModelCancelSuspendedResponseParams, Any, None]


class _ResolvedRequestModel(NamedTuple):
    model_ref: ReferenceType[Model]
    model_id: str | None
    registered: bool


class _ModelRequestCacheIdentity(CacheIdentity[ModelRequestParams]):
    def project(self, params: ModelRequestParams) -> tuple[object, ...]:
        return (
            params.model_id,
            params.messages,
            params.model_settings,
            params.model_request_parameters,
            params.run_context,
        )


class _CancelSuspendedResponseCacheIdentity(CacheIdentity[ModelCancelSuspendedResponseParams]):
    def project(self, params: ModelCancelSuspendedResponseParams) -> tuple[object, ...]:
        return (params.model_id, params.response, params.run_context)


class _CompactMessagesCacheIdentity(CacheIdentity[ModelCompactMessagesParams]):
    def project(self, params: ModelCompactMessagesParams) -> tuple[object, ...]:
        return (params.model_id, params.request_context, params.instructions, params.run_context)


class _EventStreamHandlerCacheIdentity(CacheIdentity[EventStreamHandlerParams]):
    def project(self, params: EventStreamHandlerParams) -> tuple[object, ...]:
        return (params.event,)


class _GetToolsCacheIdentity(CacheIdentity[ToolsetGetToolsParams]):
    def project(self, params: ToolsetGetToolsParams) -> tuple[object, ...]:
        return (params.ctx,)


class _FunctionCallToolCacheIdentity(CacheIdentity[ToolsetCallToolParams]):
    def project(self, params: ToolsetCallToolParams) -> tuple[object, ...]:
        return (params.name, params.tool_args, params.ctx, params.tool)


class _DynamicCallToolCacheIdentity(CacheIdentity[DynamicToolsetCallToolParams]):
    def project(self, params: DynamicToolsetCallToolParams) -> tuple[object, ...]:
        return (params.name, params.tool_args, params.ctx, params.tool_def)


class _TypedResultCodec(ResultCodec[_T]):
    """Apply the capability codec and its engine-specific serialization error mapping."""

    def __init__(self, dump: Callable[[_T], object], load: Callable[[object], _T]) -> None:
        self._dump = dump
        self._load = load

    def dump(self, value: _T) -> object:
        return self._dump(value)

    def load(self, payload: object) -> _T:
        return self._load(payload)


class BaseDurabilityCapability(AbstractCapability[AgentDepsT]):
    """Base for building a durable execution engine as an agent capability.

    Owns the model registry and the model round-trip across the durable boundary:
    a `Model` instance can't be serialized into an activity/step/task, so a request
    carries a `model_id` string (`None` for the agent's default, a `models=` registry
    key, or a model-name string) and the model is rebuilt on the other side -- deps-aware,
    via the agent's full [`resolve_model_id`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id]
    capability chain, with the registry as backstop. Only strings cross: a `Model` instance that
    isn't registered in `models=` is rejected workflow-side, because rebuilding it from its own
    `model_id` would quietly reach a different endpoint with other credentials.
    Subclasses call
    [`_bind_models`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._bind_models] on the
    bound copy in `for_agent`, [`_find_model_id`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._find_model_id]
    on the workflow/flow side, and
    [`_resolve_model_for_request`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._resolve_model_for_request]
    inside the activity/step/task.
    Engine authors declare serialization, toolset lifecycle, discovery, and concurrency behavior,
    then supply a `DurableOperationBackend` that connects operations to the engine SDK. See
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    engine_spec: ClassVar[DurabilityEngineSpec]
    """Declarative configuration for this durable execution engine."""

    @property
    def engine_name(self) -> str:
        """Human-readable engine name used in error messages."""
        return self.engine_spec.engine_name

    @property
    def durable_unit_noun(self) -> str:
        """Name for one durable unit of work."""
        return self.engine_spec.durable_unit_noun

    @property
    def durable_unit_plural(self) -> str:
        """Plural name for durable units of work."""
        return self.engine_spec.durable_unit_plural or f'{self.durable_unit_noun}s'

    @property
    def durable_container_noun(self) -> str:
        """Name for the durable container."""
        return self.engine_spec.durable_container_noun

    @property
    def agent(self) -> AbstractAgent[AgentDepsT, Any] | None:
        """The agent bound to this capability, or `None` before binding."""
        return self._agent

    @property
    def default_model_id(self) -> str | None:
        """Persisted-name ID for the agent's default model, or `None` when it is not a string ID."""
        return self._default_model_id

    name: str
    """Unique name used to identify the agent's durable units (activities/steps/tasks). Defaults to the agent's `name`."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
    ) -> None:
        self.name: str = name or ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._extra_models: dict[str, Model] = dict(models) if models else {}
        self._models_by_id: dict[str, Model] = {}
        self._default_model_id: str | None = None
        self._event_stream_handler = event_stream_handler
        self._process_event_stream = ProcessEventStream(event_stream_handler) if event_stream_handler else None
        self._toolsets_by_id: dict[str, WrapperToolset[AgentDepsT]] = {}
        self._bound_model_operations: _BoundModelOperations | None = None
        self._bound_event_operation: BoundDurableOperation[EventStreamHandlerParams, Any, None] | None = None
        self._bound_capability_operations: dict[tuple[str, str], CapabilityBoundOperation] = {}
        self._capability_declarations: dict[tuple[str, str], CapabilityMethodDeclaration] = {}
        self._resolved_request_models: dict[int, _ResolvedRequestModel] = {}

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> Self:
        """Bind to the agent and register this engine's durable units on a new copy."""
        self._check_bindable()
        if not (self.name or agent.name):
            raise UserError(
                f'An agent needs to have a unique `name` in order to be used with {self.engine_name} '
                f'(or pass `name=` to `{type(self).__name__}`). The name is used to identify the '
                f"agent's durable {self.durable_unit_plural}."
            )
        bound = copy.copy(self)
        bound.name = self.name or agent.name or ''
        bound._agent = agent
        bound._resolved_request_models = {}
        bound._bind_models(agent)
        bound._toolsets_by_id = {}
        bound._bind_to_agent(agent)
        backend = bound.get_durable_operation_backend()
        # Registered engines need the complete registration set before a worker starts. Callable
        # engines bind per request because their durable unit names can depend on that request's
        # `model_id`.
        if isinstance(backend, RegisteredOperationBackend) and bound._bound_model_operations is None:
            bound._bound_model_operations = bound._bind_model_operations(backend, model_id=None, model_name='default')
        bound._bind_capability_operations(agent)
        return bound

    def _bind_capability_operations(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._bound_capability_operations = {}
        self._capability_declarations = {}
        backend = self.get_durable_operation_backend()
        durability_ref = ref(self)
        for capability in leaf_capabilities(agent.root_capability):
            declarations = collect_capability_operations(capability)
            if not declarations:
                continue
            capability_id = capability.id
            if capability_id is None:
                raise UserError(
                    f'Capability {type(capability).__name__!r} contributes durable operations and needs an explicit '
                    '`id` because persisted operation identity and worker-side recovery must remain stable. '
                    f"Construct it as `{type(capability).__name__}(id='...')`."
                )
            for operation_name, declaration in declarations.items():
                key = (capability_id, operation_name)

                async def handler(
                    params: CapabilityOperationParams,
                    *,
                    declaration: CapabilityMethodDeclaration = declaration,
                    capability_id: str = capability_id,
                ) -> Any:
                    arguments = self.engine_spec.codec.load(
                        dict[str, Any], self.engine_spec.codec.dump(dict[str, Any], params.arguments)
                    )
                    validated = cast(dict[str, Any], declaration.schema.validator.validate_python(arguments))
                    semantic_params = CapabilityOperationParams(
                        run_context=params.run_context, arguments=validated, model_id=params.model_id
                    )
                    recovered = await recover_capability(params.run_context, capability_id=capability_id)
                    if declaration.model_request_parameter is not None:
                        projection = cast(
                            ModelRequestContextProjection,
                            semantic_params.arguments[declaration.model_request_parameter],
                        )
                        async with self._durable_model_scope(projection.model_id, params.run_context) as (
                            model,
                            durable_ctx,
                        ):
                            request_context = projection.build_context(model)
                            usage_before = copy.copy(durable_ctx.usage)
                            result = await call_declaration(
                                declaration,
                                recovered,
                                params=semantic_params,
                                model_request_context=request_context,
                            )
                            operation_result = ModelRequestContextProjection.from_context(result)
                            if result.model is not model:
                                operation_result.model_id = self._find_registered_model_id_for_hook(result.model)
                        return CapabilityOperationResult(
                            value=operation_result, usage_delta=durable_ctx.usage - usage_before
                        )
                    async with self._durable_model_scope(params.model_id, params.run_context) as (_, durable_ctx):
                        semantic_params = CapabilityOperationParams(
                            run_context=durable_ctx,
                            arguments=semantic_params.arguments,
                            model_id=params.model_id,
                        )
                        usage_before = copy.copy(durable_ctx.usage)
                        result = await call_declaration(declaration, recovered, params=semantic_params)
                    return CapabilityOperationResult(value=result, usage_delta=durable_ctx.usage - usage_before)

                operation = DurableOperation(
                    operation_id=CapabilityOperationId(capability_id, operation=operation_name),
                    handler=handler,
                    parameter_transport=self._capability_operation_parameter_transport(declaration),
                    cache_identity=CapabilityCacheIdentity(),
                    result_codec=TypedResultCodec(
                        capability_operation_result_type(declaration.result_type),
                        mode='identity' if self.engine_spec.codec is IDENTITY_CODEC else 'json',
                    ),
                    config_role='capability',
                )
                self._bound_capability_operations[key] = backend.bind(operation)
                self._capability_declarations[key] = declaration

                async def dispatch_for_run_context(
                    ctx: RunContext[object],
                    args: tuple[object, ...],
                    kwargs: dict[str, object],
                    _capability: AbstractCapability[Any] = capability,
                    _operation_name: str = operation_name,
                ) -> Any:
                    durability = durability_ref()
                    if durability is None:  # pragma: no cover
                        raise RuntimeError('The durability capability bound to this agent is no longer available.')
                    return await durability._invoke_capability_operation(
                        _capability,
                        _operation_name,
                        ctx=ctx,
                        args=args,
                        kwargs=kwargs,
                    )

                bindings = capability._get_durable_operation_bindings()
                bindings.setdefault(agent)[operation_name] = dispatch_for_run_context

    def _prepare_run_context(self, ctx: RunContext[AgentDepsT]) -> None:
        """Register dispatchers on `RunContext` for worker-side and per-run capability recovery."""
        ctx._durable_operations = {}  # pyright: ignore[reportPrivateUsage]
        if ctx.agent is None:
            return
        operations: dict[tuple[str, str], Callable[..., Awaitable[object]]] = {}
        run_capabilities = ctx._run_capabilities_by_id or {}  # pyright: ignore[reportPrivateUsage]
        for capability_id, capability in run_capabilities.items():
            for bound_capability_id, operation_name in self._bound_capability_operations:
                if capability_id != bound_capability_id:
                    continue

                async def dispatch(
                    *args: object,
                    _capability: AbstractCapability[Any] = capability,
                    _operation_name: str = operation_name,
                    **kwargs: object,
                ) -> object:
                    return await self._invoke_capability_operation(
                        _capability, _operation_name, ctx=ctx, args=args, kwargs=kwargs
                    )

                operations[(capability_id, operation_name)] = dispatch
        ctx._durable_operations = operations  # pyright: ignore[reportPrivateUsage]

    async def _invoke_capability_operation(
        self,
        capability: AbstractCapability[Any],
        operation: str,
        *,
        ctx: RunContext[Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Dispatch through serialized context so per-run capability instances recover worker-side.

        Outside a durable container, calling the original operation preserves live context mutations
        without rebuilding resources solely to round-trip them through the durable projection.
        """
        capability_id = capability.id
        if capability_id is None:
            raise RuntimeError('A durable operation capability must have an explicit `id`.')
        key = (capability_id, operation)
        declaration = self._capability_declarations[key]
        if not self.in_durable_context:
            bound = declaration.function.__get__(capability, type(capability))
            return await bound(*args, **kwargs)

        request_context = next(
            (value for value in (*args, *kwargs.values()) if isinstance(value, ModelRequestContext)), None
        )
        if request_context is not None:
            projection = ModelRequestContextProjection.from_context(request_context)
            args = tuple(projection if value is request_context else value for value in args)
            kwargs = {key: projection if value is request_context else value for key, value in kwargs.items()}
        arguments = bind_arguments(declaration, ctx=ctx, args=args, kwargs=kwargs)
        model = ctx.model
        if not isinstance(model, Model):
            raise UserError('Durable capability operations require a non-realtime `Model` on `RunContext`.')
        model_id = ctx.model_id if ctx.model_id is not None else self._find_model_id(cast('Model[Any]', model))
        usage_before = copy.copy(ctx.usage)
        result = cast(
            CapabilityOperationResult[Any],
            await self._bound_capability_operations[key](
                CapabilityOperationParams(run_context=ctx, arguments=arguments, model_id=model_id)
            ),
        )
        if declaration.model_request_parameter is not None:
            projection = cast(ModelRequestContextProjection, result.value)
            inbound = cast(ModelRequestContextProjection, arguments[declaration.model_request_parameter])
            resolved_model = None
            if projection.model_id != inbound.model_id:
                resolved_model = await self._resolve_model_for_request(projection.model_id, ctx)
                registered, _ = self._registered_model_id(resolved_model)
                self._record_resolved_request_model(resolved_model, projection.model_id, registered=registered)
            value: Any = _ResolvedModelRequestContext(projection=projection, model=resolved_model)
        else:
            value = result.value
        if not (ctx.usage - usage_before).has_values():
            ctx.usage.incr(result.usage_delta)
        return value

    def _capability_operation_parameter_transport(
        self, declaration: CapabilityMethodDeclaration
    ) -> ParameterTransport[CapabilityOperationParams, Any]:
        return IdentityParameterTransport[CapabilityOperationParams]()

    def _check_bindable(self) -> None:
        """Validate that the capability can be bound in the current context."""

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Bind engine-specific durable state. Default = wrap + index the agent's leaf toolsets.

        Sufficient for ad-hoc-primitive engines (Restate/Lambda/Absurd/Prefect): their durable
        units are created at call time. Pre-registration engines override to also register units
        up front (Temporal: worker activities) or decorate them by name (DBOS: `@DBOS.step`).
        """
        self._register_toolsets(agent)

    @classmethod
    def from_agent(cls, agent: AbstractAgent[Any, Any]) -> Self | None:
        """Return the bound instance of this durability capability on an agent, if any.

        [`for_agent`][pydantic_ai.capabilities.AbstractCapability.for_agent] returns a new bound
        copy and leaves the user's original capability reference pristine, so use this to retrieve
        the instance the agent actually runs with -- e.g. the `TemporalDurability` whose activities
        are registered with the worker. Walks the agent's capability chain and returns the single
        match or `None`, raising a `UserError` if multiple instances are attached.
        """
        found: list[Self] = []
        for capability in leaf_capabilities(agent.root_capability):
            while isinstance(capability, WrapperCapability):
                capability = capability.wrapped
            if isinstance(capability, cls):
                found.append(capability)
        if len(found) > 1:
            raise UserError(f'Multiple {cls.__name__} capabilities are attached to this agent; attach at most one.')
        return found[0] if found else None

    def _reject_runtime_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Reject executing toolsets added per-run inside a durable workflow or flow.

        Construction-time toolsets are registered with the durable engine when the
        capability is bound. Executing runtime additions would bypass that registration
        and could re-execute on recovery, while non-executing toolsets can pass through.
        Outside a durable context the capability remains transparent.
        """
        if not self.in_durable_context:
            return

        construction_leaves: set[int] = set()
        # `for_agent` always binds before a run.
        if self._agent is not None:  # pragma: no branch
            for agent_toolset in construction_toolsets(self._agent):
                agent_toolset.apply(lambda leaf: construction_leaves.add(id(leaf)))

        runtime_leaves: list[AbstractToolset[AgentDepsT]] = []

        def collect(leaf: AbstractToolset[AgentDepsT]) -> None:
            if id(leaf) in construction_leaves:
                return
            if isinstance(leaf, CapabilityOwnedToolset):
                # The run re-collects capability contributions in a fresh `CapabilityOwnedToolset`
                # whenever `for_run` changed the capability tree (e.g. a `DynamicCapability`
                # resolved, or a per-run capability was added). The wrapper itself is
                # non-executing packaging; the toolset it wraps is visited separately by this
                # same walk and judged on its own identity.
                return
            runtime_leaves.append(leaf)

        toolset.apply(collect)
        reject_unsupported_runtime_toolsets(
            runtime_leaves,
            unsupported_kinds=self.engine_spec.unsupported_runtime_toolset_kinds,
            engine=self.engine_name,
            tool_config_key=self.engine_spec.tool_config_key,
        )

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        """Reject capabilities added per-run inside a durable workflow or flow."""
        if not self.in_durable_context or not isinstance(
            self.get_durable_operation_backend(), RegisteredOperationBackend
        ):
            return
        unsafe_capabilities = [capability for capability in capabilities if not capability._safe_at_runtime]
        if not unsafe_capabilities:
            return
        names = ', '.join(sorted(type(capability).__name__ for capability in unsafe_capabilities))
        raise UserError(
            f'Capabilities added per-run inside a {self.engine_name} {self.durable_container_noun} are not '
            f'supported: {names}. {self.engine_name} registers durable {self.durable_unit_plural} when a '
            f'capability is bound to the agent, before the {self.durable_container_noun} starts. A capability '
            f'added per-run therefore has no registered durable {self.durable_unit_plural} for the toolsets it '
            f'contributes or its own `@durable_operation` methods. Attach all capabilities at agent construction '
            f'time so `{type(self).__name__}.for_agent()` can register their durable {self.durable_unit_plural}.'
        )

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        # A `CancellationToken` is a same-process handle that cannot cross the durable execution
        # boundary, and firing it inside a workflow/flow would cancel the durable task out of band
        # — non-deterministic on replay. Reject it here, but only inside the durable container, so a
        # durable-capable agent used *outside* a workflow keeps accepting tokens like a normal agent.
        # The token is attached to the run's controller during `Agent` setup, before this hook fires.
        # Read via `__dict__` so a restricted run-context subclass (e.g. `TemporalRunContext`) whose
        # `__getattribute__` rejects absent fields doesn't raise a misleading error instead.
        if not self.in_durable_context:
            return
        cancellation = ctx.__dict__.get('_cancellation')
        if cancellation is not None and cancellation.has_token:
            raise cancellation_token_unsupported_error(self.engine_name)

    def _effective_event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """The handler in-boundary event delivery targets for the current run.

        Engines may override to consult per-run state -- e.g. DBOS honors the
        `event_stream_handler` recorded in a wrapper-era workflow's inputs, delivering
        it exactly the way the wrapper did so recovery replays the recorded step
        sequence.
        """
        return self._event_stream_handler

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return self._effective_event_stream_handler() is not None

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        event_stream_handler = self._effective_event_stream_handler()
        dispatch_events = False
        if event_stream_handler is not None and not self.in_durable_context:
            assert self._process_event_stream is not None
            stream = self._process_event_stream.wrap_run_event_stream(ctx, stream=stream)
        elif event_stream_handler is not None:
            dispatch_events = True

        try:
            async for event in stream:
                # `ModelResponseStreamEvent`s were already delivered live to the handler inside the
                # model-request boundary; workflow-side they're the replay, so only `HandleResponseEvent`s
                # are dispatched to the handler here.
                if dispatch_events and not isinstance(event, _MODEL_RESPONSE_STREAM_EVENT_TYPES):
                    await self._dispatch_event_stream_event(ctx, event)
                yield event
        finally:
            await aclose_if_supported(stream)

    @property
    @abstractmethod
    def in_durable_context(self) -> bool:
        """Whether execution is currently inside this engine's durable container (workflow or flow)."""

    def _register_toolsets(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Wrap the agent's leaf toolsets in engine wrappers and index them by toolset `id`."""
        for toolset in agent.toolsets:
            toolset.visit_and_replace(self._wrap_and_register_leaf)

    def _wrap_and_register_leaf(self, ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        ts_id = ts.id
        if ts_id is None and isinstance(ts, DynamicToolset):
            raise UserError(
                f"Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) "
                f'need to have a unique `id` in order to be used with {self.engine_name}. '
                f"The ID will be used to identify the toolset's {self.durable_unit_plural} within the "
                f'{self.durable_container_noun}. Set the dynamic toolset ID with `DynamicToolset(id=...)`, '
                "or, when it is contributed by a capability, set the capability's `id` (for example, "
                "`DynamicCapability(..., id='user-tools')`). A capability function passed directly to "
                '`capabilities=` cannot carry an `id`; wrap it explicitly: '
                "`DynamicCapability(my_func, id='...')`."
            )
        if ts_id is not None and (existing := self._toolsets_by_id.get(ts_id)) is not None:
            if existing.wrapped is ts:
                # The same toolset instance can appear in more than one place in the tree;
                # reuse its wrapper so its durable units register exactly once.
                return existing
            # A distinct toolset under an already-registered `id` would silently replace it
            # in the registry and route both toolsets' calls to one wrapper.
            raise UserError(
                f'Two toolsets have the same `id` {ts_id!r}. Toolset `id`s must be unique among all '
                f"toolsets registered with the same agent, as they identify the toolset's "
                f'{self.durable_unit_plural} within the {self.durable_container_noun}.'
            )
        wrapped = self._wrap_leaf_toolset(ts)
        if wrapped is None:
            return ts
        if ts_id is None:
            raise UserError(
                f"Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) "
                f'need to have a unique `id` in order to be used with {self.engine_name}. '
                f"The ID will be used to identify the toolset's {self.durable_unit_plural} within the "
                f'{self.durable_container_noun}. Set it on the toolset itself with '
                '`FunctionToolset(id=...)` or `MCPToolset(..., id=...)`, or, when the toolset is '
                "contributed by a capability, set the capability's `id` (for example, "
                "`WebSearch(local='duckduckgo', id='search')` or `MCP(url='...', id='...')`)."
            )
        self._toolsets_by_id[ts_id] = wrapped
        return wrapped

    @abstractmethod
    def get_durable_operation_backend(self) -> DurableOperationBackend[Any]:
        """Return the backend that dispatches this capability's durable operations."""

    def _typed_result_codec(self, result_type: object) -> _TypedResultCodec[Any]:
        """Build a typed result codec with the engine's serialization-failure mapping."""
        return _TypedResultCodec(partial(self._encode, result_type), partial(self.engine_spec.codec.load, result_type))

    def _encode(self, tp: Any, value: Any) -> Any:
        """Encode a durable-unit result, mapping deterministic serialization failures when configured."""
        try:
            return self.engine_spec.codec.dump(tp, value)
        except (PydanticSerializationError, TypeError) as exc:
            if mapper := self.engine_spec.serialization_failure:
                raise mapper(exc) from exc
            raise

    def _toolset_base_config(self, kind: ToolsetKind) -> Any:
        """Engine base config for a toolset kind's durable units (merged with per-tool config)."""
        return None

    def _toolset_operation_config(self, kind: ToolsetKind, toolset_id: str) -> Any:
        """Return the base config for one concrete toolset's operations."""
        return self._toolset_base_config(kind)

    def _durable_run_context(self, ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
        """Guard `ctx.enqueue()` and `ctx.cancel()` for user code that runs inside a durable unit (#6666)."""
        return guard_run_context(ctx, unit_noun=self.durable_unit_noun, container_noun=self.durable_container_noun)

    @contextmanager
    def _durable_run_context_scope(self, ctx: RunContext[AgentDepsT]) -> Generator[RunContext[AgentDepsT]]:
        """Guard `ctx.enqueue()` and install the guarded context as the ambient run context."""
        guarded = self._durable_run_context(ctx)
        with set_current_run_context(guarded):
            yield guarded

    @asynccontextmanager
    async def _durable_model_scope(
        self, model_id: str | None, run_context: RunContext[AgentDepsT]
    ) -> AsyncGenerator[tuple[Model, RunContext[AgentDepsT]]]:
        """Enter a model durable unit: guard the run context, then rebuild the request's model.

        Every model unit (request, streaming request, suspended-response cancellation) needs
        both halves, and the guard is not optional for any of them: the unit's recorded result is
        replayed on recovery or a cache hit without re-running its body, so a `ctx.enqueue()` from
        the model -- or from a `resolve_model_id` capability rebuilding it -- would be dropped.
        Models rebuilt inside the unit are owned and context-managed here; the agent's default and
        `models=` registry instances keep their existing external lifecycle owner.
        Pairing the two here means a unit can't get its model without the guard, instead of each
        engine remembering to install it per unit (Temporal has its own chokepoint in
        `deserialize_run_context`, so it doesn't use this).
        """
        with self._durable_run_context_scope(run_context) as ctx:
            model = await self._resolve_model_for_request(model_id, ctx)
            registered, _ = self._registered_model_id(model)
            async with managed_model_scope(model, owned=not registered) as active_model:
                ctx.model = active_model
                if isinstance(ctx, _RestrictedRunContext):
                    ctx._expose_field('model')  # pyright: ignore[reportPrivateUsage]
                yield active_model, ctx

    def _build_resolve_tool_config(self, base_config: Any) -> Callable[[ToolsetTool[Any] | None, str], ToolConfig]:
        """Build the per-tool config resolver from declarative fields (metadata key + polarity)."""
        metadata_key = self.engine_spec.tool_config_key

        def resolve(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            if metadata_key is None:
                # An engine that declares no config key takes no per-tool config at all, so tool
                # metadata is not consulted. Collapsing `None` to `''` here would instead read the
                # empty-string key, letting `metadata={'': False}` opt a tool out of its durable
                # unit -- un-checkpointing the call and shifting the recorded unit sequence, which
                # is exactly what DBOS's "tool metadata is ignored" contract exists to prevent.
                return self._normalize_unit_config(base_config)

            config = resolve_tool_durable_config(
                tool,
                tool_name,
                {},
                metadata_key=metadata_key,
                config_type_label=f'{self.engine_name} durable config',
            )
            if config is False:
                # `fallback_config` is deliberately empty above, so `False` can only come from
                # metadata on a concrete tool.
                assert tool is not None
                return False
            if not config:
                return self._normalize_unit_config(base_config)
            combined: dict[str, Any] = {}
            if base_config:
                combined.update(base_config)
            combined.update(config)
            return self._normalize_unit_config(combined)

        return resolve

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Force sequential tool execution when required by a sequence-keyed durable engine."""
        agent = self._agent
        if not self.engine_spec.sequential_tools_in_durable_context or agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode('sequential'):
            return await handler()

    def _normalize_unit_config(self, config: Any) -> Any:
        """Post-process a resolved config (e.g. Prefect/Temporal ensure non-retryable errors)."""
        return config

    def _unwrap_tool_result(self, payload: CallToolResult) -> Any:
        """Turn a recorded tool payload back into a value/exception (control-flow-as-values seam).

        `tool_call_result_upgrade_lenient` engines (Prefect cache, DBOS/Lambda recovery) also
        accept raw pre-value-wrapping recordings; strict journal engines assert the wire shape.
        """
        if self.engine_spec.tool_call_result_upgrade_lenient:
            return unwrap_recorded_tool_call_result(payload)
        return unwrap_tool_call_result(payload)

    async def _prepare_function_call_params(
        self, toolset: FunctionToolset[AgentDepsT], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        """Prepare engine-transported function call parameters for the common handler."""
        return params

    @contextmanager
    def _tool_call_payload_errors(self, tool_name: str) -> Generator[None]:
        """Map engine-specific failures while dispatching a tool call."""
        yield

    @contextmanager
    def _tool_run_context_scope(self, ctx: RunContext[AgentDepsT]) -> Generator[RunContext[AgentDepsT]]:
        """Install the durable run-context policy used by common tool handlers."""
        with self._durable_run_context_scope(ctx) as durable_ctx:
            yield durable_ctx

    def _function_call_parameter_transport(
        self, toolset: FunctionToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetCallToolParams, Any]:
        return IdentityParameterTransport[ToolsetCallToolParams]()

    def _get_tools_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, Any]:
        return IdentityParameterTransport[ToolsetGetToolsParams]()

    def _get_instructions_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, Any]:
        return IdentityParameterTransport[ToolsetGetToolsParams]()

    def _dynamic_get_tools_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, Any]:
        return IdentityParameterTransport[ToolsetGetToolsParams]()

    def _dynamic_call_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> ParameterTransport[DynamicToolsetCallToolParams, Any]:
        return IdentityParameterTransport[DynamicToolsetCallToolParams]()

    def _validation_context(self, ctx: RunContext[Any]) -> Any:
        return ctx.validation_context

    def _bind_validate_tool_arguments_operation(
        self,
        backend: DurableOperationBackend[Any],
        toolset: FunctionToolset[AgentDepsT] | DynamicToolset[AgentDepsT],
        kind: Literal['function', 'dynamic'],
    ) -> Any:
        """Bind the single shared tool-argument-validation declaration."""
        toolset_id = cast(str, toolset.id)
        parameter_transport: Any
        cache_identity: Any
        if kind == 'function':
            function_toolset = cast(FunctionToolset[AgentDepsT], toolset)
            parameter_transport = self._function_call_parameter_transport(function_toolset)
            cache_identity = _FunctionCallToolCacheIdentity()
        else:
            dynamic_toolset = cast(DynamicToolset[AgentDepsT], toolset)
            parameter_transport = self._dynamic_call_parameter_transport(dynamic_toolset)
            cache_identity = _DynamicCallToolCacheIdentity()

        async def handler(params: ToolsetCallToolParams | DynamicToolsetCallToolParams) -> CallToolResult:
            if isinstance(params, ToolsetCallToolParams):
                function_params = await self._prepare_function_call_params(
                    cast(FunctionToolset[AgentDepsT], toolset), params
                )
                assert function_params.tool is not None
                with self._tool_run_context_scope(function_params.ctx) as durable_ctx:
                    return await wrap_tool_call_result(
                        run_args_validator(function_params.tool, function_params.tool_args, durable_ctx)
                    )
            with self._tool_run_context_scope(params.ctx) as durable_ctx:
                return await wrap_tool_call_result(
                    validate_dynamic_tool_args(
                        cast(DynamicToolset[AgentDepsT], toolset),
                        params.name,
                        params.tool_args,
                        durable_ctx,
                        tool_def=params.tool_def,
                        validation_context=self._validation_context,
                    )
                )

        return backend.bind(
            DurableOperation(
                operation_id=ToolsetValidateToolArgumentsId(kind, toolset_id=toolset_id),
                handler=handler,
                parameter_transport=parameter_transport,
                cache_identity=cache_identity,
                result_codec=self._typed_result_codec(CallToolResult),
                config_role='tool',
                invocation_label=lambda params: params.name,
            )
        )

    def _mcp_call_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetCallToolParams, Any]:
        return IdentityParameterTransport[ToolsetCallToolParams]()

    def _mcp_discovery_registrations(
        self, get_tools: object, get_instructions: object | None
    ) -> list[Callable[..., Any]]:
        return [
            *self._bound_operation_registrations(get_tools),
            *(self._bound_operation_registrations(get_instructions) if get_instructions is not None else []),
        ]

    def _bound_operation_registrations(self, *operations: object) -> list[Callable[..., Any]]:
        """Return engine registrations contributed by bound operations, if any."""
        return []

    def _toolset_in_durable_context(self) -> bool:
        """Whether durable toolset operations should use their durable boundary."""
        return self.in_durable_context

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        """Base-owned dispatch: build the right `Durable*Toolset` for a leaf toolset kind.

        Consults `engine_spec.wrapped_toolset_kinds` (DBOS omits `'function'`) and
        `engine_spec.toolset_lifecycles`. The
        operation closures dispatch through the backend, so the codec + control-flow-value wrapping
        + upgrade-lenient decoding are all framework-owned; the engine only supplies the primitive.
        """
        if isinstance(ts, FunctionToolset):
            if 'function' not in self.engine_spec.wrapped_toolset_kinds:
                return None
            return self._build_function_toolset(ts)
        if isinstance(ts, DynamicToolset):
            if 'dynamic' not in self.engine_spec.wrapped_toolset_kinds:
                return None
            return self._build_dynamic_toolset(ts)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover
            return None
        if isinstance(ts, MCPToolset):
            if 'mcp' not in self.engine_spec.wrapped_toolset_kinds:
                return None
            return self._build_mcp_toolset(ts)
        return self._wrap_other_leaf_toolset(ts)

    def _wrap_other_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        """Wrap an engine-specific leaf toolset not handled by the built-in kind dispatch."""
        return None

    def _build_function_toolset(self, toolset: FunctionToolset[AgentDepsT]) -> DurableFunctionToolset[AgentDepsT]:
        base_config = self._toolset_operation_config('function', cast(str, toolset.id))

        async def call_tool_handler(params: ToolsetCallToolParams) -> CallToolResult:
            params = await self._prepare_function_call_params(toolset, params)
            assert params.tool is not None
            with self._tool_run_context_scope(params.ctx) as durable_ctx:
                return await wrap_tool_call_result(
                    toolset.call_tool(params.name, params.tool_args, durable_ctx, params.tool)
                )

        backend = self.get_durable_operation_backend()
        operation = DurableOperation(
            operation_id=ToolsetCallToolId('function', toolset_id=cast(str, toolset.id)),
            handler=call_tool_handler,
            parameter_transport=self._function_call_parameter_transport(toolset),
            cache_identity=_FunctionCallToolCacheIdentity(),
            result_codec=self._typed_result_codec(CallToolResult),
            config_role='tool',
            invocation_label=lambda params: params.name,
        )
        call_tool = backend.bind(operation)
        validate_args = self._bind_validate_tool_arguments_operation(backend, toolset, 'function')

        def resolve_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            return backend.config_for_tool(operation, tool=tool, tool_name=tool_name)

        def resolve_validation_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            return backend.config_for_tool(validate_args.operation, tool=tool, tool_name=tool_name)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            with self._tool_call_payload_errors(name):
                payload = await call_tool(
                    ToolsetCallToolParams(name, tool_args=tool_args, ctx=ctx, tool=tool), config=config
                )
            return self._unwrap_tool_result(payload)

        async def validate_args_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> None:
            with self._tool_call_payload_errors(name):
                payload = await validate_args(
                    ToolsetCallToolParams(name, tool_args=tool_args, ctx=ctx, tool=tool), config=config
                )
            self._unwrap_tool_result(payload)

        return DurableFunctionToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            call_tool_operation=call_tool_operation,
            validate_args_operation=validate_args_operation,
            resolve_tool_config=resolve_tool_config,
            resolve_validation_config=resolve_validation_config,
            lifecycle=self.engine_spec.toolset_lifecycles['function'],
            durable_registrations=self._bound_operation_registrations(call_tool, validate_args),
            durable_config=base_config,
        )

    def _build_dynamic_toolset(self, toolset: DynamicToolset[AgentDepsT]) -> DurableDynamicToolset[AgentDepsT]:
        base_config = self._toolset_operation_config('dynamic', cast(str, toolset.id))

        async def get_tools_handler(params: ToolsetGetToolsParams) -> DynamicToolsResult:
            with self._tool_run_context_scope(params.ctx) as durable_ctx:
                return await get_dynamic_tools(toolset, durable_ctx)

        async def call_tool_handler(params: DynamicToolsetCallToolParams) -> CallToolResult:
            with self._tool_run_context_scope(params.ctx) as durable_ctx:
                return await wrap_tool_call_result(
                    call_dynamic_tool(
                        toolset,
                        params.name,
                        params.tool_args,
                        durable_ctx,
                        tool_def=params.tool_def,
                        validation_context=self._validation_context,
                    )
                )

        backend = self.get_durable_operation_backend()
        get_tools = backend.bind(
            DurableOperation(
                operation_id=ToolsetGetToolsId('dynamic', toolset_id=cast(str, toolset.id)),
                handler=get_tools_handler,
                parameter_transport=self._dynamic_get_tools_parameter_transport(toolset),
                cache_identity=_GetToolsCacheIdentity(),
                result_codec=self._typed_result_codec(DynamicToolsResult),
                config_role='tool',
            )
        )
        call_operation = DurableOperation(
            operation_id=ToolsetCallToolId('dynamic', toolset_id=cast(str, toolset.id)),
            handler=call_tool_handler,
            parameter_transport=self._dynamic_call_parameter_transport(toolset),
            cache_identity=_DynamicCallToolCacheIdentity(),
            result_codec=self._typed_result_codec(CallToolResult),
            config_role='tool',
            invocation_label=lambda params: params.name,
        )
        call_tool = backend.bind(call_operation)
        validate_args = self._bind_validate_tool_arguments_operation(backend, toolset, 'dynamic')

        def resolve_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            return backend.config_for_tool(call_operation, tool=tool, tool_name=tool_name)

        def resolve_validation_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            return backend.config_for_tool(validate_args.operation, tool=tool, tool_name=tool_name)

        async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
            if not self.engine_spec.journal_discovery:
                return await get_dynamic_tools(toolset, ctx)

            return await get_tools(ToolsetGetToolsParams(ctx), config=base_config)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            with self._tool_call_payload_errors(name):
                payload = await call_tool(
                    DynamicToolsetCallToolParams(name, tool_args=tool_args, ctx=ctx, tool_def=tool.tool_def),
                    config=config,
                )
            return self._unwrap_tool_result(payload)

        async def validate_args_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> None:
            with self._tool_call_payload_errors(name):
                payload = await validate_args(
                    DynamicToolsetCallToolParams(name, tool_args=tool_args, ctx=ctx, tool_def=tool.tool_def),
                    config=config,
                )
            self._unwrap_tool_result(payload)

        return DurableDynamicToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            get_tools_operation=get_tools_operation,
            call_tool_operation=call_tool_operation,
            validate_args_operation=validate_args_operation,
            resolve_tool_config=resolve_tool_config,
            resolve_validation_config=resolve_validation_config,
            lifecycle=self.engine_spec.toolset_lifecycles['dynamic'],
            durable_registrations=self._bound_operation_registrations(get_tools, call_tool, validate_args),
            durable_config=base_config,
        )

    def _build_mcp_toolset(self, toolset: Any) -> DurableMCPToolset[AgentDepsT]:
        base_config = self._toolset_operation_config('mcp', cast(str, toolset.id))
        get_tools_registration_source: object | None = None

        if self.engine_spec.journal_discovery:
            get_tools = self._bind_mcp_get_tools_operation(toolset)
            get_tools_registration_source = get_tools

            async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
                return await get_tools(ToolsetGetToolsParams(ctx), config=base_config)

            registrations = self._bound_operation_registrations(get_tools)
        else:
            get_tools_operation = toolset.get_tools
            registrations = []

        return self._build_mcp_toolset_after_get_tools(
            toolset,
            base_config=base_config,
            get_tools_operation=get_tools_operation,
            get_tools=get_tools_registration_source,
            get_tools_registration=registrations,
        )

    def _bind_mcp_get_tools_operation(self, toolset: Any) -> Any:
        async def get_tools_handler(params: ToolsetGetToolsParams) -> dict[str, ToolDefinition]:
            with self._tool_run_context_scope(params.ctx) as durable_ctx:
                tools = await toolset.get_tools(durable_ctx)
            return {name: tool.tool_def for name, tool in tools.items()}

        operation = DurableOperation(
            operation_id=ToolsetGetToolsId('mcp', toolset_id=cast(str, toolset.id)),
            handler=get_tools_handler,
            parameter_transport=self._get_tools_parameter_transport(toolset),
            cache_identity=_GetToolsCacheIdentity(),
            result_codec=self._typed_result_codec(dict[str, ToolDefinition]),
            config_role='tool',
        )
        return self.get_durable_operation_backend().bind(operation)

    def _build_mcp_toolset_after_get_tools(
        self,
        toolset: Any,
        *,
        base_config: Any,
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, ToolDefinition]]],
        get_tools: object | None,
        get_tools_registration: list[Callable[..., Any]],
    ) -> DurableMCPToolset[AgentDepsT]:
        get_instructions = (
            self._bind_mcp_get_instructions_operation(toolset) if self.engine_spec.journal_discovery else None
        )

        async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> Instructions:
            assert get_instructions is not None
            return await get_instructions(ToolsetGetToolsParams(ctx), config=base_config)

        return self._build_mcp_toolset_after_discovery(
            toolset,
            base_config=base_config,
            get_tools_operation=get_tools_operation,
            get_instructions_operation=get_instructions_operation,
            discovery_registrations=(
                self._mcp_discovery_registrations(get_tools, get_instructions) if get_tools_registration else []
            ),
        )

    def _bind_mcp_get_instructions_operation(self, toolset: Any) -> Any:
        async def get_instructions_handler(params: ToolsetGetToolsParams) -> Instructions:
            with self._durable_run_context_scope(params.ctx) as durable_ctx:
                # A server's instructions are captured during `__aenter__`, so it has to be
                # connected *inside* this unit: an engine whose lifecycle never enters the
                # toolset (DBOS's `enter-never`) would otherwise journal `None` and silently
                # drop the instructions. Entry is refcounted, so this is a no-op when the
                # toolset is already entered (`enter-always`/`enter-outside-durable`).
                async with toolset:
                    return await toolset.get_instructions(durable_ctx)

        operation = DurableOperation(
            operation_id=ToolsetGetInstructionsId(cast(str, toolset.id)),
            handler=get_instructions_handler,
            parameter_transport=self._get_instructions_parameter_transport(toolset),
            cache_identity=_GetToolsCacheIdentity(),
            result_codec=self._typed_result_codec(Instructions),
            config_role='tool',
        )
        return self.get_durable_operation_backend().bind(operation)

    def _build_mcp_toolset_after_discovery(
        self,
        toolset: Any,
        *,
        base_config: Any,
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, ToolDefinition]]],
        get_instructions_operation: Callable[[RunContext[AgentDepsT]], Awaitable[Instructions]],
        discovery_registrations: list[Callable[..., Any]],
    ) -> DurableMCPToolset[AgentDepsT]:
        async def call_tool_handler(params: ToolsetCallToolParams) -> CallToolResult:
            assert params.tool is not None
            with self._durable_run_context_scope(params.ctx) as durable_ctx:
                return await wrap_tool_call_result(
                    toolset.call_tool(params.name, params.tool_args, durable_ctx, params.tool)
                )

        backend = self.get_durable_operation_backend()
        call_operation = DurableOperation(
            operation_id=ToolsetCallToolId('mcp', toolset_id=cast(str, toolset.id)),
            handler=call_tool_handler,
            parameter_transport=self._mcp_call_parameter_transport(toolset),
            cache_identity=_FunctionCallToolCacheIdentity(),
            result_codec=self._typed_result_codec(CallToolResult),
            config_role='tool',
            invocation_label=lambda params: params.name,
        )
        call_tool = backend.bind(call_operation)

        def resolve_tool_config(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
            if tool is not None and tool.tool_def.metadata is not None:
                metadata_key = self.engine_spec.tool_config_key or self.engine_name.lower()
                if tool.tool_def.metadata.get(metadata_key) is False:
                    raise UserError(
                        f'{self.engine_name} durable config for MCP tool {tool_name!r} has been explicitly '
                        'set to `False` (durable execution disabled), but MCP tools perform I/O and cannot '
                        f'run outside a durable {self.durable_unit_noun}. Remove the metadata so the call '
                        'stays durable.'
                    )
            return backend.config_for_tool(call_operation, tool=tool, tool_name=tool_name)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            with self._tool_call_payload_errors(name):
                payload = await call_tool(
                    ToolsetCallToolParams(name, tool_args=tool_args, ctx=ctx, tool=tool), config=config
                )
            return self._unwrap_tool_result(payload)

        return DurableMCPToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            get_tools_operation=get_tools_operation if self.engine_spec.journal_discovery else None,
            get_instructions_operation=get_instructions_operation if self.engine_spec.journal_discovery else None,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=resolve_tool_config,
            lifecycle=self.engine_spec.toolset_lifecycles['mcp'],
            durable_registrations=[*discovery_registrations, *self._bound_operation_registrations(call_tool)],
            durable_config=base_config,
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Base-owned: assemble a `DurableModel` from four segment executors when in-context.

        Each segment runs its model call through the durable operation backend; the model is
        rebuilt worker-side from `model_id` (`_resolve_model_for_request`). Identical for every
        callable engine -- the only per-engine input is the durable primitive + codec + naming.
        """
        if not self.in_durable_context:
            return await handler(request_context)

        resolved = self._resolved_request_model(request_context.model)
        owned = resolved is not None and not resolved.registered
        model_id = self._model_id_for_request(ctx, request_context)
        async with managed_model_scope(request_context.model, owned=owned) as active_model:
            request_context.model = active_model
            self._validate_model_request_parameters(request_context.model_request_parameters)
            model_name = request_context.model.model_name
            backend = self.get_durable_operation_backend()
            operations = self._bound_model_operations or self._bind_model_operations(
                backend, model_id=model_id, model_name=model_name
            )

            async def request_segment(request: ModelRequestContext) -> ModelResponse:
                return await operations.request(
                    ModelRequestParams(
                        model_id,
                        messages=request.messages,
                        model_settings=request.model_settings,
                        model_request_parameters=request.model_request_parameters,
                        run_context=ctx,
                    )
                )

            async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
                result = await operations.request_stream(
                    ModelRequestParams(
                        model_id,
                        messages=request.messages,
                        model_settings=request.model_settings,
                        model_request_parameters=request.model_request_parameters,
                        run_context=ctx,
                    )
                )
                return await self._load_streamed_activity_result(result, request.model_request_parameters)

            async def cancel_suspended_response_segment(response: ModelResponse) -> None:
                await operations.cancel_suspended_response(
                    ModelCancelSuspendedResponseParams(model_id, response=response, run_context=ctx)
                )

            async def compact_messages_segment(
                compact_context: ModelRequestContext, instructions: str | None
            ) -> ModelResponse:
                return await operations.compact_messages(
                    ModelCompactMessagesParams(
                        model_id,
                        request_context=compact_context,
                        instructions=instructions,
                        run_context=ctx,
                    )
                )

            request_context.model = DurableModel(
                request_context.model,
                request_segment=request_segment,
                request_stream_segment=request_stream_segment,
                compact_messages_segment=compact_messages_segment,
                cancel_suspended_response_segment=cancel_suspended_response_segment,
            )
            return await handler(request_context)

    async def _load_streamed_activity_result(
        self, result: object, model_request_parameters: ModelRequestParameters
    ) -> StreamedActivityResult:
        return cast(StreamedActivityResult, result)

    def _bind_model_operations(
        self, backend: DurableOperationBackend[Any], *, model_id: str | None, model_name: str
    ) -> _BoundModelOperations:
        request_operation = backend.bind(
            DurableOperation(
                operation_id=ModelRequestId(model_id, streaming=False, model_name=model_name),
                handler=self._model_request_operation,
                parameter_transport=self._model_request_parameter_transport(ModelResponse),
                cache_identity=_ModelRequestCacheIdentity(),
                result_codec=self._typed_result_codec(ModelResponse),
                config_role='model',
            )
        )
        request_stream_operation = backend.bind(
            DurableOperation(
                operation_id=ModelRequestId(model_id, streaming=True, model_name=model_name),
                handler=self._model_request_stream_operation,
                parameter_transport=self._model_request_parameter_transport(StreamedActivityResult),
                cache_identity=_ModelRequestCacheIdentity(),
                result_codec=self._typed_result_codec(StreamedActivityResult),
                config_role='model',
            )
        )
        compact_messages_operation = backend.bind(
            DurableOperation(
                operation_id=ModelCompactMessagesId(model_id, model_name=model_name),
                handler=self._compact_messages_operation,
                parameter_transport=self._compact_messages_parameter_transport(),
                cache_identity=_CompactMessagesCacheIdentity(),
                result_codec=self._typed_result_codec(ModelResponse),
                config_role='model',
            )
        )
        cancel_suspended_response_operation = backend.bind(
            DurableOperation(
                operation_id=ModelCancelSuspendedResponseId(model_id, model_name=model_name),
                handler=self._cancel_suspended_response_operation,
                parameter_transport=self._cancel_suspended_response_parameter_transport(),
                cache_identity=_CancelSuspendedResponseCacheIdentity(),
                result_codec=self._typed_result_codec(type(None)),
                config_role='model',
            )
        )

        return _BoundModelOperations(
            request_operation,
            request_stream_operation,
            compact_messages_operation,
            cancel_suspended_response_operation,
        )

    def _model_request_parameter_transport(self, result_type: object) -> ParameterTransport[ModelRequestParams, Any]:
        return IdentityParameterTransport[ModelRequestParams]()

    def _cancel_suspended_response_parameter_transport(
        self,
    ) -> ParameterTransport[ModelCancelSuspendedResponseParams, Any]:
        return IdentityParameterTransport[ModelCancelSuspendedResponseParams]()

    def _compact_messages_parameter_transport(self) -> ParameterTransport[ModelCompactMessagesParams, Any]:
        return IdentityParameterTransport[ModelCompactMessagesParams]()

    async def _model_request_operation(self, params: ModelRequestParams) -> ModelResponse:
        async with self._durable_model_scope(params.model_id, params.run_context) as (model, _):
            response = await model.request(params.messages, params.model_settings, params.model_request_parameters)
        self._stamp_response(response, params.messages)
        return response

    async def _model_request_stream_operation(self, params: ModelRequestParams) -> StreamedActivityResult:
        async with self._durable_model_scope(params.model_id, params.run_context) as (model, durable_ctx):
            async with model.request_stream(
                params.messages, params.model_settings, params.model_request_parameters, durable_ctx
            ) as streamed:
                events = await capture_event_stream(
                    run_context=durable_ctx,
                    stream=streamed,
                    handler=self._effective_event_stream_handler(),
                )
        response = streamed.get()
        self._stamp_response(response, params.messages)
        return StreamedActivityResult(response=response, events=events)

    async def _cancel_suspended_response_operation(self, params: ModelCancelSuspendedResponseParams) -> None:
        if params.run_context is None:
            await self._cancel_suspended_response_without_run_context(params.model_id, params.response)
            return
        async with self._durable_model_scope(params.model_id, params.run_context) as (model, _):
            await model.cancel_suspended_response(params.response)

    async def _compact_messages_operation(self, params: ModelCompactMessagesParams) -> ModelResponse:
        async with self._durable_model_scope(params.model_id, params.run_context) as (model, _):
            params.request_context.model = model
            return await model.compact_messages(params.request_context, instructions=params.instructions)

    async def _cancel_suspended_response_without_run_context(
        self, model_id: str | None, response: ModelResponse
    ) -> None:
        raise RuntimeError('Cancelling a suspended response requires a serialized run context')

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        """Validate engine-specific model request restrictions before durable dispatch."""

    def _stamp_response(self, response: ModelResponse, messages: list[Any]) -> None:
        """Stamp run provenance on a response before an engine persists/caches it. No-op default."""
        return None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their durable-wrapped versions."""
        self._reject_runtime_toolsets(toolset)
        if not self._toolsets_by_id:
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is None or (registered := self._toolsets_by_id.get(ts_id)) is None:
                return ts
            if registered.wrapped is not ts:
                # A toolset that arrived after binding (via `run(toolsets=...)`,
                # `override(toolsets=...)`, or a per-run capability) under an `id` that is already
                # registered would be replaced here by the wrapper around the *construction-time*
                # toolset, silently running that toolset's tools instead of its own. The
                # construction-time counterpart of this is caught in `_wrap_and_register_leaf`.
                #
                # Only inside a workflow, flow, or step. `get_wrapper_toolset` runs on every run,
                # and core toolsets carry no global uniqueness requirement, so raising outside a
                # durable context would stop a durability capability being transparent for ordinary
                # runs -- moving behavior on the plain-`Agent` surface to fix a durable-only defect.
                # Outside one there is no durable unit to dispatch to, so the toolset that actually
                # arrived is simply used as-is.
                if not self.in_durable_context:
                    return ts
                raise UserError(
                    f'A toolset added at run time has the same `id` {ts_id!r} as one the agent was '
                    f'constructed with. Toolset `id`s must be unique: the `id` identifies which registered '
                    f"toolset's {self.durable_unit_noun} a tool call is dispatched to inside the "
                    f'{self.durable_container_noun}, so this run would have called the construction-time '
                    "toolset's tools instead. Give the toolset a different `id`."
                )
            return registered

        return toolset.visit_and_replace(swap)

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost: durable dispatch must be the last wrapper around the model handler so every
        # other capability's contribution is already applied inside the durable unit.
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-loadable: the useful configuration (`models=` Model instances, `event_stream_handler`
        # callables, run-context classes, activity/step/task configs holding timedeltas and retry-policy
        # objects) is not spec-serializable, and a durable agent additionally has to be constructed in
        # worker-setup code for its durable units to be registered.
        return None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        """Base-owned: deliver one workflow-side event inside a durable unit.

        Sufficient for SEQUENCE-keyed engines (Restate/Lambda/Absurd/DBOS/Temporal), where the
        durable unit's identity is its encounter order, so content-identical events map to distinct
        journal entries automatically. HASH-keyed engines (Prefect) key replay on input hash, so
        two identical events collide; those engines override this to inject a per-container sequence
        (#5477 requirement 4). That override is the one genuine behavioral difference the hash-keyed
        family forces.
        """
        bound_operation = self._bound_event_operation or self._bind_event_operation(
            self.get_durable_operation_backend()
        )
        await bound_operation(EventStreamHandlerParams(event, run_context=ctx))

    def _bind_event_operation(
        self, backend: DurableOperationBackend[Any]
    ) -> BoundDurableOperation[EventStreamHandlerParams, Any, None]:
        operation = DurableOperation(
            operation_id=EventStreamHandlerId(),
            handler=self._event_stream_handler_operation,
            parameter_transport=self._event_stream_handler_parameter_transport(),
            cache_identity=_EventStreamHandlerCacheIdentity(),
            result_codec=self._typed_result_codec(type(None)),
            config_role='event',
        )
        return backend.bind(operation)

    def _event_stream_handler_parameter_transport(self) -> ParameterTransport[EventStreamHandlerParams, Any]:
        return IdentityParameterTransport[EventStreamHandlerParams]()

    async def _event_stream_handler_operation(self, params: EventStreamHandlerParams) -> None:
        handler = self._effective_event_stream_handler()
        assert handler is not None
        with self._durable_run_context_scope(params.run_context) as durable_ctx:
            await handler(durable_ctx, self._single_event_stream(params.event))

    @staticmethod
    async def _single_event_stream(
        event: AgentStreamEvent,
    ) -> AsyncIterator[AgentStreamEvent]:
        yield event

    def _bind_models(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Build the model registry on a bound copy from the agent's default model and `models=` extras.

        Called from `for_agent`. A concrete default -- a `Model` instance, or a string the user
        explicitly mapped to one via `models=` (so it *is* the default) -- is registered as
        `'default'` (that key is reserved), and a `models=` string is also kept under its raw
        string so run-time resolution of the default yields the same instance.

        A plain string default is deliberately *not* resolved here: constructing it eagerly could
        build the wrong provider -- with authentication/configuration side effects -- before a
        sibling [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] gets to reinterpret it.
        Instead no `'default'` is registered; every request for the default carries the raw string
        and re-resolves through the capability chain (or `infer_model`) on the worker.
        """
        if agent.model is None:
            raise UserError(
                f'An agent needs to have a `model` in order to be used with {self.engine_name}, '
                'it cannot be set at agent run time.'
            )
        self._default_model_id = agent.model if isinstance(agent.model, str) else None
        default_model: Model | None
        if isinstance(agent.model, str):
            # Only a `models=` mapping resolves the string to a concrete default here; any other
            # string defers to run-time resolution (a sibling `ResolveModelId`, this capability's
            # registry, or `infer_model`) so it's built worker-side with the run's deps.
            default_model = self._extra_models.get(agent.model)
        else:
            default_model = agent.model

        self._models_by_id = {} if default_model is None else {'default': default_model}
        for model_id, model_instance in self._extra_models.items():
            if model_id == 'default':
                raise UserError("Model ID 'default' is reserved for the agent's primary model.")
            self._models_by_id[model_id] = model_instance

    async def resolve_model_id(
        self,
        ctx: ModelResolutionContext[AgentDepsT],
        *,
        model_id: KnownModelName | str,
    ) -> Model | None:
        """Map a model-name string to its `models=` registry instance, or `None` to defer.

        Registry hits resolve to the registered instance; anything else defers to the
        default `infer_model` flow, so a durable run can accept arbitrary
        `agent.run(model='openai:gpt-5.2')` values without pre-registering each one in
        `models=`. To customize how strings are built (e.g. a custom provider), add a
        [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability -- its
        position relative to this one doesn't matter for non-registry strings.
        """
        return self._models_by_id.get(model_id)

    def _model_id_for_request(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> str | None:
        """The cross-boundary identifier for this request's model.

        Prefer the original model-id string the run's model was resolved from
        ([`ModelRequestContext.model_id`][pydantic_ai.models.ModelRequestContext.model_id]) when the
        request still targets the run's model: it survives aliases that the resolved model's own
        `model_id` doesn't (the worker-side chain re-resolves the same string the caller wrote). A
        model swapped in by an outer capability's `before_model_request` invalidates the provenance,
        so it falls back to `_find_model_id`.
        """
        provenance = request_context.model_id
        # A durable run always targets a regular `Model`, never a realtime model, so `ctx.model`
        # (typed as the wider `AbstractModel`) is a `Model` here; the guard narrows it for `unwrap_model`.
        run_model = ctx.model
        if (
            provenance is not None
            and isinstance(run_model, Model)
            and unwrap_model(request_context.model) is unwrap_model(cast('Model[Any]', run_model))
        ):
            return provenance
        resolved = self._resolved_request_model(request_context.model)
        if resolved is not None:
            return resolved.model_id
        return self._find_model_id(request_context.model)

    def _record_resolved_request_model(self, model: Model, model_id: str | None, *, registered: bool) -> None:
        model_key = id(model)

        def discard(_model_ref: ReferenceType[Model]) -> None:
            self._resolved_request_models.pop(model_key, None)

        self._resolved_request_models[model_key] = _ResolvedRequestModel(ref(model, discard), model_id, registered)

    def _resolved_request_model(self, model: Model) -> _ResolvedRequestModel | None:
        resolved = self._resolved_request_models.get(id(model))
        if resolved is not None and resolved.model_ref() is model:
            return resolved
        return None

    def _find_model_id(self, model: Model) -> str | None:
        """Find the cross-boundary identifier for a registered `Model` instance.

        Returns `None` for the agent's default model (no extra info needed) or a registry key when
        an instance from `models=` is being used. The activity/step/task uses the result to look the
        same `Model` up on the other side via `_resolve_model_for_request`.

        `WrapperModel` layers are peeled off the request's model one at a time, matching
        registered instances as-is at each depth and preferring the shallowest match: a
        registered behavior-changing wrapper keeps its own ID -- even under further
        unregistered wrapping, e.g. an `InstrumentedModel` around it -- while an
        unregistered wrapper around the default still takes the default's fast path.
        The registered side is never unwrapped: a registered wrapper's identity holds at
        its registered depth, so its bare inner model doesn't inherit the wrapper's ID.

        An instance that matches nothing is rejected rather than round-tripped as its own
        `model_id`: a `Model` can't be serialized across the boundary, and rebuilding one from its
        `model_id` would build a *different* model -- the same model name on whatever provider the
        worker's environment implies -- so the request would quietly go to another endpoint with
        other credentials. Registering the instance in `models=`, or passing a model-name string
        that a [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability builds
        worker-side, are the two ways to get a specific instance into a durable run.
        """
        candidate: Model | None = model
        while candidate is not None:
            registered, model_id = self._registered_model_id(candidate)
            if registered:
                return model_id
            candidate = candidate.wrapped if isinstance(candidate, WrapperModel) else None
        raise UserError(
            f'The model instance {model.model_id!r} was not registered with `{type(self).__name__}`, so it '
            f'cannot be used inside a {self.durable_container_noun}. A `Model` instance cannot be serialized '
            f'across the {self.durable_unit_noun} boundary, and rebuilding it from its `model_id` would build '
            'a different model — the same model name on the provider the worker environment implies — so the '
            f'request would go to another endpoint with other credentials. {self._model_rebuild_escape_hatches()}'
        )

    def _find_registered_model_id_for_hook(self, model: Model) -> str | None:
        registered, model_id = self._registered_model_id(model)
        if registered:
            return model_id
        raise UserError(
            'A durable `before_model_request` hook replaced `request_context.model` with the unregistered model '
            f'instance {model.model_id!r}. A live `Model` instance cannot be transported across the '
            f'{self.durable_unit_noun} boundary. Register it in `models=` on `{type(self).__name__}` and select '
            'that registered model by ID.'
        )

    def _registered_model_id(self, model: Model) -> tuple[bool, str | None]:
        for model_id, registered in self._models_by_id.items():
            if registered is model:
                return True, None if model_id == 'default' else model_id
        return False, None

    async def _resolve_model_for_request(self, model_id: str | None, run_context: RunContext[AgentDepsT]) -> Model:
        """Rebuild the `Model` for a request inside the activity/step/task, deps-aware.

        Mirrors the workflow-side resolution in `Agent._resolve_model_selection`: run the agent's
        full `resolve_model_id` capability chain -- deps-aware user capabilities like
        `ResolveModelId` get first crack, and this capability's registry resolution
        acts as the durable backstop -- so a model whose provider depends on the run's
        deps is rebuilt with the *actual* deps on the worker rather than deps-blind.

        Only strings reach here: a `model_id` is `None`, a `models=` key, or a model-name string a
        caller wrote (or that a resolver resolved from), because `_find_model_id` rejects
        unregistered instances workflow-side.
        """
        if model_id is None:
            return self._models_by_id['default']
        agent = run_context.agent
        root_capability = run_context.root_capability
        # The boundary carries both.
        if agent is not None and root_capability is not None:  # pragma: no branch
            resolution_ctx = ModelResolutionContext(agent=agent, deps=run_context.deps)
            # Exceptions raised by user resolvers in the chain propagate unchanged;
            # only the `infer_model` backstop below gets the translated error.
            resolved = await root_capability.resolve_model_id(resolution_ctx, model_id=model_id)
            if resolved is not None:
                return resolved
        try:
            return infer_model(model_id)
        except (UserError, ValueError) as e:
            # The string a caller wrote (or a resolver resolved from) is not one `infer_model` can
            # build, and no capability in the chain claimed it — e.g. an alias that only a
            # `ResolveModelId` on the workflow side knows. Point at the escape hatches instead of
            # surfacing a bare 'Unknown model'.
            raise UserError(
                f'The model {model_id!r} could not be rebuilt on the {self.engine_name} worker: it is not '
                'a model name `infer_model` can build, and no `resolve_model_id` capability claimed it. '
                f'{self._model_rebuild_escape_hatches()}'
            ) from e

    def _model_rebuild_escape_hatches(self) -> str:
        """The two supported ways to use a specific `Model` instance in a durable run."""
        return (
            f'Register the instance in `models=` on `{type(self).__name__}` and reference it by key '
            '(or pass the registered instance), or pass a model-name string and build the instance '
            'from it with a `ResolveModelId` capability.'
        )
