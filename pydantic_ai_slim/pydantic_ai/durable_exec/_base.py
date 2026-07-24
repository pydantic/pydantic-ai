from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar, Literal, TypeVar

from pydantic_core import PydanticSerializationError
from typing_extensions import Self

from pydantic_ai import FunctionToolset, ToolsetTool
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai._utils import get_union_args
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
    leaf_capabilities,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext, ModelResolutionContext, infer_model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._codec import IDENTITY_CODEC, DurabilityCodec
from ._runtime_toolsets import RuntimeToolsetKind, reject_unsupported_runtime_toolsets
from ._toolset import (
    CallToolResult,
    DurableDynamicToolset,
    DurableFunctionToolset,
    DurableMCPToolset,
    DynamicToolsResult,
    Instructions,
    Lifecycle,
    ToolConfig,
    call_dynamic_tool,
    get_dynamic_tools,
    guard_run_context_enqueue,
    resolve_tool_durable_config,
    unwrap_recorded_tool_call_result,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from ._utils import DurableModel, StreamedActivityResult, capture_event_stream, unwrap_model

_T = TypeVar('_T')
ToolsetKind = Literal['function', 'mcp', 'dynamic']

_MODEL_RESPONSE_STREAM_EVENT_TYPES = get_union_args(ModelResponseStreamEvent)


class BaseDurabilityCapability(AbstractCapability[AgentDepsT]):
    """Shared base for the durable-execution capabilities (Temporal, DBOS, Prefect).

    Owns the model registry and the model round-trip across the durable boundary:
    a `Model` instance can't be serialized into an activity/step/task, so a request
    carries a `model_id` string (`None` for the agent's default, a `models=` registry
    key, or a model-name string) and the model is rebuilt on the other side -- deps-aware,
    via the agent's full [`resolve_model_id`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id]
    capability chain, with the registry as backstop. Subclasses call
    [`_bind_models`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._bind_models] on the
    bound copy in `for_agent`, [`_find_model_id`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._find_model_id]
    on the workflow/flow side, and
    [`_resolve_model_for_request`][pydantic_ai.durable_exec._base.BaseDurabilityCapability._resolve_model_for_request]
    inside the activity/step/task.
    """

    engine_name: ClassVar[str]
    """Human-readable engine name used in error messages (e.g. `'Temporal'`)."""

    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]]
    _durable_unit_noun: ClassVar[str]
    _durable_container_noun: ClassVar[str]
    _tool_config_key: ClassVar[str | None] = None

    # --- Declarative Shape-D surface (prototype) -----------------------------------------------
    # Everything below is DATA an engine sets rather than behavior it overrides. The base's
    # concrete `_wrap_leaf_toolset` / `wrap_model_request` / `_dispatch_event_stream_event`
    # (built on `run_durable_unit` + `_codec`) consult these; a callable engine implements only
    # `run_durable_unit` + `in_durable_context` and fills these in.

    _codec: ClassVar[DurabilityCodec] = IDENTITY_CODEC
    """How the base serializes at every durable boundary. Identity for object-passing engines
    (Temporal/DBOS/Prefect), JSON for journal engines (Restate/Lambda/Absurd)."""

    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    """Which leaf-toolset kinds this engine wraps in a durable unit. DBOS omits `'function'`
    (function tools run inline via `@DBOS.step`)."""

    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    """Per-kind lifecycle profile (`enter-always` / `enter-outside-durable` / `enter-never`).
    Forced explicit because two real bugs came from defaulted gates (#5477 requirement 3).
    Restate opts function tools out of entry (`enter-never`)."""

    _tool_call_result_upgrade_lenient: ClassVar[bool] = False
    """When True, recorded tool payloads are decoded leniently for library-upgrade compat
    (`unwrap_recorded_tool_call_result`) -- engines that replay stored outputs (Prefect cache,
    DBOS/Lambda recovery). Journal engines that never cross an upgrade set False."""

    _journal_discovery: ClassVar[bool] = True
    """Whether toolset DISCOVERY (`get_tools`/`get_instructions`) runs in its own durable unit.
    Journal engines (Restate/Lambda/Absurd) journal it; Prefect deliberately runs discovery in
    flow code (flow retries re-resolve anyway) and journals only tool CALLS. THE odd one out."""

    _force_sequential_tools_in_durable_context: ClassVar[bool] = False
    """Whether tool calls must run sequentially inside the durable container."""

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

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> Self:
        """Bind to the agent and register this engine's durable units on a new copy."""
        self._validate_declarative_contract()
        self._check_bindable()
        if not (self.name or agent.name):
            raise UserError(
                f'An agent needs to have a unique `name` in order to be used with {self.engine_name} '
                f'(or pass `name=` to `{type(self).__name__}`). The name is used to identify the '
                f"agent's durable {self._durable_unit_noun}s."
            )
        bound = copy.copy(self)
        bound.name = self.name or agent.name or ''
        bound._agent = agent
        bound._bind_models(agent)
        bound._toolsets_by_id = {}
        bound._bind_to_agent(agent)
        return bound

    def _validate_declarative_contract(self) -> None:
        """Fail at binding when an engine's declarative durability configuration is incomplete."""
        cls = type(self)
        engine_name = getattr(cls, 'engine_name', '') or cls.__name__
        missing_fields = [
            field
            for field in ('engine_name', '_durable_unit_noun', '_durable_container_noun')
            if not getattr(cls, field, None)
        ]
        invalid_kinds = self._wrapped_toolset_kinds - {'function', 'mcp', 'dynamic'}
        missing_lifecycles = self._wrapped_toolset_kinds - self._toolset_lifecycles.keys()
        errors: list[str] = []
        if missing_fields:
            errors.append(f'required ClassVars are unset: {", ".join(missing_fields)}')
        if invalid_kinds:
            errors.append(f'unsupported wrapped toolset kinds: {sorted(invalid_kinds)!r}')
        if missing_lifecycles:
            errors.append(f'missing toolset lifecycles for: {sorted(missing_lifecycles)!r}')
        if errors:
            raise UserError(f'Invalid {engine_name} declarative durability contract: {"; ".join(errors)}.')

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
        found = [cap for cap in leaf_capabilities(agent.root_capability) if isinstance(cap, cls)]
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
        if self._agent is not None:  # pragma: no branch -- `for_agent` always binds before a run
            for agent_toolset in self._agent.toolsets:
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
            unsupported_kinds=self._unsupported_runtime_toolset_kinds,
            engine=self.engine_name,
            tool_config_key=self._tool_config_key,
        )

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
        if self._effective_event_stream_handler() is None:
            async for event in stream:
                yield event
            return
        if not self.in_durable_context:
            assert self._process_event_stream is not None
            async for event in self._process_event_stream.wrap_run_event_stream(ctx, stream=stream):
                yield event
            return

        async for event in stream:
            # `ModelResponseStreamEvent`s were already delivered
            # live to the handler inside the model-request boundary; workflow-side they're
            # the replay, so only `HandleResponseEvent`s are dispatched to the handler here.
            if not isinstance(event, _MODEL_RESPONSE_STREAM_EVENT_TYPES):
                await self._dispatch_event_stream_event(ctx, event)
            yield event

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
                f"The ID will be used to identify the toolset's {self._durable_unit_noun}s within the "
                f'{self._durable_container_noun}. Set the dynamic toolset ID with `DynamicToolset(id=...)`, '
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
                f'{self._durable_unit_noun}s within the {self._durable_container_noun}.'
            )
        wrapped = self._wrap_leaf_toolset(ts)
        if wrapped is None:
            return ts
        if ts_id is None:
            raise UserError(
                f"Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) "
                f'need to have a unique `id` in order to be used with {self.engine_name}. '
                f"The ID will be used to identify the toolset's {self._durable_unit_noun}s within the "
                f'{self._durable_container_noun}.'
            )
        self._toolsets_by_id[ts_id] = wrapped
        return wrapped

    # ===========================================================================================
    #  Base-owned Shape-D assembly (prototype)
    #
    #  These methods used to be overridden by every engine. They are now concrete defaults built
    #  on the declarative fields above plus two behavioral hooks -- `run_durable_unit` (the durable
    #  primitive) and `in_durable_context`. A callable engine (Prefect/DBOS/Restate/Lambda/Absurd)
    #  inherits all of them. A pre-registration engine (Temporal) still overrides them, because a
    #  call-time closure can't cross its workflow→activity boundary (see `run_durable_unit`).
    # ===========================================================================================

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        """Run one framework-built operation `fn` durably and return its (codec-dumped) payload.

        THE durable primitive for callable engines: Prefect wraps `fn` in a `@task`, Restate calls
        `ctx.run_typed(name, fn)`, Lambda bridges `fn` onto `context.step(name, ...)`. `fn` already
        applies `self._codec.dump`, so the returned payload is journal-ready; the base applies
        `self._codec.load` around this call.

        `inputs` are the operation's logical arguments (messages, tool args, run context, ...).
        SEQUENCE-keyed engines (Restate/Lambda/Absurd/DBOS) ignore them -- a step's identity is its
        encounter order. HASH-keyed engines (Prefect) MUST feed them to the durable primitive so its
        cache key hashes them; a bare no-arg closure would hide the inputs and collapse distinct
        calls onto one cache entry. This parameter is exactly the seam that lets one primitive serve
        both keying families.

        Temporal CANNOT implement this: its durable unit is a worker-registered activity dispatched
        by name (`activity.defn(name=...)` + `workflow.execute_activity(...)`), not an arbitrary
        call-time callable. Temporal therefore overrides the assembly methods below instead.
        """
        raise NotImplementedError

    async def _durable_operation(
        self, name: str, fn: Callable[[], Awaitable[_T]], *, tp: Any, inputs: tuple[Any, ...], config: Any
    ) -> _T:
        """Run `fn` in a durable unit with the codec applied: `dump` inside, `load` outside.

        Mirrors what the JSON engines hand-write (dump inside `_inner`, validate outside). For the
        identity codec both are no-ops, so object engines pass the live value straight through.
        """

        async def unit() -> Any:
            return self._encode(tp, await fn())

        payload = await self.run_durable_unit(name, unit, inputs=inputs, config=config)
        return self._codec.load(tp, payload)

    def _encode(self, tp: Any, value: Any) -> Any:
        """Encode a durable-unit result, mapping deterministic serialization failures when configured."""
        try:
            return self._codec.dump(tp, value)
        except (PydanticSerializationError, TypeError) as exc:
            mapped = self._serialization_failure(exc)
            if mapped is not None:
                raise mapped from exc
            raise

    def _serialization_failure(self, exc: Exception) -> BaseException | None:
        """Map serialization failure to an engine's non-retryable error, or return `None`.

        JSON-journal engines override this to return their terminal/non-retryable error type.
        """
        return None

    def _unit_name(self, kind: str, **parts: Any) -> str:
        """Compose the durable-unit name for one operation.

        Naming is compat surface (#5477 req 5), so this is a pure function of `(kind, parts)` an engine can override. Default is
        the journal-engine scheme (`{agent}__{kind}...`); Prefect overrides with display names.
        """
        prefix = parts.get('prefix')
        name = prefix if isinstance(prefix, str) else f'{self.name}__{kind}'
        if suffix := parts.get('suffix'):
            name = f'{name}{suffix}'
        if (tool_name := parts.get('tool_name')) is not None:
            name = f'{name}.call_tool'
            if kind != 'mcp_server':
                name = f'{name}:{tool_name}'
        return name

    def _model_unit_config(self) -> Any:
        """Engine config for model-request durable units. Override for a custom config type."""
        return None

    def _event_unit_config(self) -> Any:
        """Engine config for the event-stream-handler durable unit."""
        return None

    def _toolset_base_config(self, kind: ToolsetKind) -> Any:
        """Engine base config for a toolset kind's durable units (merged with per-tool config)."""
        return None

    def _durable_run_context(self, ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
        """Guard `ctx.enqueue()` for user code that runs inside a durable unit (#6666)."""
        return guard_run_context_enqueue(
            ctx, unit_noun=self._durable_unit_noun, container_noun=self._durable_container_noun
        )

    @contextmanager
    def _durable_run_context_scope(self, ctx: RunContext[AgentDepsT]) -> Generator[RunContext[AgentDepsT]]:
        """Guard `ctx.enqueue()` and install the guarded context as the ambient run context."""
        guarded = self._durable_run_context(ctx)
        with set_current_run_context(guarded):
            yield guarded

    def _build_resolve_tool_config(self, base_config: Any) -> Callable[[ToolsetTool[Any] | None, str], ToolConfig]:
        """Build the per-tool config resolver from declarative fields (metadata key + polarity)."""
        metadata_key = self._tool_config_key or ''

        def resolve(tool: ToolsetTool[Any] | None, tool_name: str) -> ToolConfig:
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
                from pydantic_ai.mcp import MCPToolset

                if isinstance(tool.toolset, MCPToolset):
                    raise UserError(
                        f'{self.engine_name} durable config for MCP tool {tool_name!r} has been explicitly '
                        'set to `False` (durable execution disabled), but MCP tools perform I/O and cannot '
                        f'run outside a durable {self._durable_unit_noun}. Remove the metadata so the call '
                        'stays durable.'
                    )
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
        if not self._force_sequential_tools_in_durable_context or agent is None or not self.in_durable_context:
            return await handler()
        with agent.parallel_tool_call_execution_mode('sequential'):
            return await handler()

    def _normalize_unit_config(self, config: Any) -> Any:
        """Post-process a resolved config (e.g. Prefect/Temporal ensure non-retryable errors)."""
        return config

    def _unwrap_tool_result(self, payload: CallToolResult) -> Any:
        """Turn a recorded tool payload back into a value/exception (control-flow-as-values seam).

        `_tool_call_result_upgrade_lenient` engines (Prefect cache, DBOS/Lambda recovery) also
        accept raw pre-value-wrapping recordings; strict journal engines assert the wire shape.
        """
        if self._tool_call_result_upgrade_lenient:
            return unwrap_recorded_tool_call_result(payload)
        return unwrap_tool_call_result(payload)

    def _toolset_in_durable_context(self) -> bool:
        """Whether durable toolset operations should use their durable boundary."""
        return self.in_durable_context

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        """Base-owned dispatch: build the right `Durable*Toolset` for a leaf toolset kind.

        Consults `_wrapped_toolset_kinds` (DBOS omits `'function'`) and `_toolset_lifecycles`. The
        operation closures call `self._durable_operation`, so the codec + control-flow-value wrapping
        + upgrade-lenient decoding are all framework-owned; the engine only supplies the primitive.
        """
        if isinstance(ts, FunctionToolset):
            if 'function' not in self._wrapped_toolset_kinds:
                return None
            return self._build_function_toolset(ts)
        if isinstance(ts, DynamicToolset):
            if 'dynamic' not in self._wrapped_toolset_kinds:
                return None
            return self._build_dynamic_toolset(ts)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover
            return None
        if isinstance(ts, MCPToolset):
            if 'mcp' not in self._wrapped_toolset_kinds:
                return None
            return self._build_mcp_toolset(ts)
        return None

    def _build_function_toolset(self, toolset: FunctionToolset[AgentDepsT]) -> DurableFunctionToolset[AgentDepsT]:
        base_config = self._toolset_base_config('function')
        prefix = f'{self.name}__function_toolset__{toolset.id}'

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            async def fn() -> CallToolResult:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    return await wrap_tool_call_result(toolset.call_tool(name, tool_args, durable_ctx, tool))

            unit_name = self._unit_name('function_toolset', prefix=prefix, tool_name=name, label='Call Tool')
            payload = await self._durable_operation(
                unit_name, fn, tp=CallToolResult, inputs=(name, tool_args, ctx, tool), config=config
            )
            return self._unwrap_tool_result(payload)

        return DurableFunctionToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=self._build_resolve_tool_config(base_config),
            lifecycle=self._toolset_lifecycles['function'],
            durable_config=base_config,
        )

    def _build_dynamic_toolset(self, toolset: DynamicToolset[AgentDepsT]) -> DurableDynamicToolset[AgentDepsT]:
        base_config = self._toolset_base_config('dynamic')
        prefix = f'{self.name}__dynamic_toolset__{toolset.id}'

        async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
            if not self._journal_discovery:
                # Prefect resolves the dynamic toolset in flow code, not a durable unit.
                return await get_dynamic_tools(toolset, ctx)

            async def fn() -> DynamicToolsResult:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    return await get_dynamic_tools(toolset, durable_ctx)

            return await self._durable_operation(
                self._unit_name('dynamic_toolset', prefix=prefix, suffix='.get_tools'),
                fn,
                tp=DynamicToolsResult,
                inputs=(ctx,),
                config=base_config,
            )

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            async def fn() -> CallToolResult:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    return await wrap_tool_call_result(call_dynamic_tool(toolset, name, tool_args, durable_ctx))

            unit_name = self._unit_name('dynamic_toolset', prefix=prefix, tool_name=name, label='Call Tool')
            payload = await self._durable_operation(
                unit_name, fn, tp=CallToolResult, inputs=(name, tool_args, ctx), config=config
            )
            return self._unwrap_tool_result(payload)

        return DurableDynamicToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            get_tools_operation=get_tools_operation,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=self._build_resolve_tool_config(base_config),
            lifecycle=self._toolset_lifecycles['dynamic'],
            durable_config=base_config,
        )

    def _build_mcp_toolset(self, toolset: Any) -> DurableMCPToolset[AgentDepsT]:
        base_config = self._toolset_base_config('mcp')
        prefix = f'{self.name}__mcp_server__{toolset.id}'

        async def get_tools_operation(ctx: RunContext[AgentDepsT]) -> dict[str, ToolDefinition]:
            async def fn() -> dict[str, ToolDefinition]:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    tools = await toolset.get_tools(durable_ctx)
                return {n: t.tool_def for n, t in tools.items()}

            return await self._durable_operation(
                self._unit_name('mcp_server', prefix=prefix, suffix='.get_tools'),
                fn,
                tp=dict[str, ToolDefinition],
                inputs=(ctx,),
                config=base_config,
            )

        async def get_instructions_operation(ctx: RunContext[AgentDepsT]) -> Instructions:
            async def fn() -> Instructions:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    return await toolset.get_instructions(durable_ctx)

            return await self._durable_operation(
                self._unit_name('mcp_server', prefix=prefix, suffix='.get_instructions'),
                fn,
                tp=Instructions,
                inputs=(ctx,),
                config=base_config,
            )

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Any,
        ) -> Any:
            async def fn() -> CallToolResult:
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    return await wrap_tool_call_result(toolset.call_tool(name, tool_args, durable_ctx, tool))

            unit_name = self._unit_name('mcp_server', prefix=prefix, tool_name=name, label='Call MCP Tool')
            payload = await self._durable_operation(
                unit_name, fn, tp=CallToolResult, inputs=(name, tool_args, ctx, tool), config=config
            )
            return self._unwrap_tool_result(payload)

        return DurableMCPToolset(
            toolset,
            in_durable_context=self._toolset_in_durable_context,
            # Prefect runs MCP discovery in flow code, not a durable unit (`_journal_discovery`).
            get_tools_operation=get_tools_operation if self._journal_discovery else None,
            get_instructions_operation=get_instructions_operation if self._journal_discovery else None,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=self._build_resolve_tool_config(base_config),
            lifecycle=self._toolset_lifecycles['mcp'],
            durable_config=base_config,
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Base-owned: assemble a `DurableModel` from three segment executors when in-context.

        Each segment runs its model call in a durable unit via `_durable_operation`; the model is
        rebuilt worker-side from `model_id` (`_resolve_model_for_request`). Identical for every
        callable engine -- the only per-engine input is the durable primitive + codec + naming.
        """
        if not self.in_durable_context:
            return await handler(request_context)

        model_id = self._model_id_for_request(ctx, request_context)
        suffix = self._model_id_suffix(model_id)
        model_name = request_context.model.model_name
        config = self._model_unit_config()

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            async def fn() -> ModelResponse:
                model = await self._resolve_model_for_request(model_id, ctx)
                with self._durable_run_context_scope(ctx):
                    response = await model.request(
                        request.messages, request.model_settings, request.model_request_parameters
                    )
                self._stamp_response(response, request.messages)
                return response

            return await self._durable_operation(
                self._unit_name('model.request', suffix=suffix, model_name=model_name, label='Model Request'),
                fn,
                tp=ModelResponse,
                inputs=(model_id, request.messages, request.model_settings, request.model_request_parameters, ctx),
                config=config,
            )

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            async def fn() -> StreamedActivityResult:
                model = await self._resolve_model_for_request(model_id, ctx)
                with self._durable_run_context_scope(ctx) as durable_ctx:
                    async with model.request_stream(
                        request.messages, request.model_settings, request.model_request_parameters, durable_ctx
                    ) as streamed:
                        events = await capture_event_stream(
                            run_context=durable_ctx,
                            stream=streamed,
                            handler=self._effective_event_stream_handler(),
                        )
                response = streamed.get()
                self._stamp_response(response, request.messages)
                return StreamedActivityResult(response=response, events=events)

            return await self._durable_operation(
                self._unit_name(
                    'model.request_stream', suffix=suffix, model_name=model_name, label='Model Request (Streaming)'
                ),
                fn,
                tp=StreamedActivityResult,
                inputs=(model_id, request.messages, request.model_settings, request.model_request_parameters, ctx),
                config=config,
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            async def fn() -> None:
                model = await self._resolve_model_for_request(model_id, ctx)
                with self._durable_run_context_scope(ctx):
                    await model.cancel_suspended_response(response)
                return None

            await self._durable_operation(
                self._unit_name(
                    'model.cancel_suspended_response',
                    suffix=suffix,
                    model_name=model_name,
                    label='Cancel Suspended Response',
                ),
                fn,
                tp=type(None),
                inputs=(model_id, response, ctx),
                config=config,
            )

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)

    def _model_id_suffix(self, model_id: str | None) -> str:
        """Suffix non-default model units while keeping the agent's default model names stable."""
        if model_id is None or model_id == self._default_model_id:
            return ''
        return f'.{model_id}'

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
            if ts_id is not None and ts_id in self._toolsets_by_id:
                return self._toolsets_by_id[ts_id]
            return ts

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
        handler = self._effective_event_stream_handler()
        assert handler is not None

        async def fn() -> None:
            with self._durable_run_context_scope(ctx) as durable_ctx:
                await handler(durable_ctx, self._single_event_stream(event))
            return None

        await self._durable_operation(
            self._unit_name('event_stream_handler', label='Handle Stream Event'),
            fn,
            tp=type(None),
            inputs=(event,),
            config=self._event_unit_config(),
        )

    @staticmethod
    async def _single_event_stream(event: AgentStreamEvent) -> AsyncIterator[AgentStreamEvent]:
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
        if provenance is not None and unwrap_model(request_context.model) is unwrap_model(ctx.model):
            return provenance
        return self._find_model_id(request_context.model)

    def _find_model_id(self, model: Model) -> str | None:
        """Find the cross-boundary identifier for a `Model` instance.

        Returns `None` for the agent's default model (no extra info needed),
        a registry key when an instance from `models=` is being used, or the
        model's own `model_id` string otherwise. The activity/step/task uses the
        result to rebuild the same `Model` on the other side via
        `_resolve_model_for_request`.

        `WrapperModel` layers are peeled off the request's model one at a time, matching
        registered instances as-is at each depth and preferring the shallowest match: a
        registered behavior-changing wrapper keeps its own ID -- even under further
        unregistered wrapping, e.g. an `InstrumentedModel` around it -- while an
        unregistered wrapper around the default still takes the default's fast path.
        The registered side is never unwrapped: a registered wrapper's identity holds at
        its registered depth, so its bare inner model doesn't inherit the wrapper's ID. The
        `model_id` fallback covers models built from a run-time
        string (via `resolve_model_id`) and models an outer capability swaps in
        via `before_model_request`: the worker rebuilds them by looking the
        `model_id` up in the registry, then falling back to the `resolve_model_id`
        capability chain / `infer_model`. This round-trip only reproduces a model
        that the chain or `infer_model` (or the registry under that `model_id`)
        can rebuild -- a pre-built instance with a custom provider, client, or
        settings that isn't registered in `models=` will not survive it faithfully.
        """
        candidate: Model | None = model
        while candidate is not None:
            for model_id, registered in self._models_by_id.items():
                if registered is candidate:
                    return None if model_id == 'default' else model_id
            candidate = candidate.wrapped if isinstance(candidate, WrapperModel) else None
        # Runtime-built or swapped-in Model: round-trip via its model_id string. The worker
        # rebuilds it the same way (registry lookup → resolve_model_id chain → infer_model).
        return model.model_id

    async def _resolve_model_for_request(self, model_id: str | None, run_context: RunContext[AgentDepsT]) -> Model:
        """Rebuild the `Model` for a request inside the activity/step/task, deps-aware.

        Mirrors the workflow-side resolution in `Agent._resolve_model_selection`: run the agent's
        full `resolve_model_id` capability chain -- deps-aware user capabilities like
        `ResolveModelId` get first crack, and this capability's registry resolution
        acts as the durable backstop -- so a model whose provider depends on the run's
        deps is rebuilt with the *actual* deps on the worker rather than deps-blind.
        """
        if model_id is None:
            return self._models_by_id['default']
        agent = run_context.agent
        root_capability = run_context.root_capability
        if agent is not None and root_capability is not None:  # pragma: no branch - the boundary carries both
            resolution_ctx = ModelResolutionContext(agent=agent, deps=run_context.deps)
            # Exceptions raised by user resolvers in the chain propagate unchanged;
            # only the `infer_model` backstop below gets the translated error.
            resolved = await root_capability.resolve_model_id(resolution_ctx, model_id=model_id)
            if resolved is not None:
                return resolved
        try:
            return infer_model(model_id)
        except (UserError, ValueError) as e:
            # The usual culprit: an unregistered `Model` instance was passed at run time,
            # crossed the boundary as its `model_id` string, and that string can't be fed
            # back through `infer_model` (e.g. `'function:...'`, `'test:test'`). Point at
            # the registration escape hatches instead of surfacing a bare 'Unknown model'.
            raise UserError(
                f'The model {model_id!r} could not be rebuilt on the {self.engine_name} worker. '
                'A `Model` instance cannot be serialized across the durable boundary, so it is '
                'sent as its `model_id` string and rebuilt on the other side. Register the '
                f'instance in `models=` on `{type(self).__name__}` and reference it by key '
                '(or pass the registered instance), or resolve the string with a '
                '`ResolveModelId` capability.'
            ) from e
