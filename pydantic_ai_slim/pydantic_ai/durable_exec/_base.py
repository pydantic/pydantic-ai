from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, ClassVar

from typing_extensions import Self

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext, ModelResolutionContext, infer_model
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._runtime_toolsets import RuntimeToolsetKind, reject_unsupported_runtime_toolsets
from ._utils import unwrap_model


class BaseDurabilityCapability(AbstractCapability[AgentDepsT]):
    """Shared base for the durable-execution capabilities (Temporal, DBOS, Prefect).

    Owns the model registry and the model round-trip across the durable boundary:
    a `Model` instance can't be serialized into an activity/step/task, so a request
    carries a `model_id` string (`None` for the agent's default, a `models=` registry
    key, or a model-name string) and the model is rebuilt on the other side — deps-aware,
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

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> None:
        self._agent: AbstractAgent[Any, Any] | None = None
        self._extra_models: dict[str, Model] = dict(models) if models else {}
        self._models_by_id: dict[str, Model] = {}
        self._event_stream_handler = event_stream_handler
        self._process_event_stream = ProcessEventStream(event_stream_handler) if event_stream_handler else None

    @classmethod
    def from_agent(cls, agent: AbstractAgent[Any, Any]) -> Self | None:
        """Return the bound instance of this durability capability on an agent, if any.

        [`for_agent`][pydantic_ai.capabilities.AbstractCapability.for_agent] returns a new bound
        copy and leaves the user's original capability reference pristine, so use this to retrieve
        the instance the agent actually runs with — e.g. the `TemporalDurability` whose activities
        are registered with the worker. Walks the agent's capability chain and returns the single
        match or `None`, raising a `UserError` if multiple instances are attached.
        """
        found: list[Self] = []

        def visitor(cap: Any) -> None:
            if isinstance(cap, cls):
                found.append(cap)

        agent.root_capability.apply(visitor)
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
        if not self._in_durable_context():
            return

        construction_leaves: set[int] = set()
        if self._agent is not None:  # pragma: no branch — `for_agent` always binds before a run
            for agent_toolset in self._agent.toolsets:
                agent_toolset.apply(lambda leaf: construction_leaves.add(id(leaf)))

        runtime_leaves: list[AbstractToolset[AgentDepsT]] = []

        def collect(leaf: AbstractToolset[AgentDepsT]) -> None:
            if id(leaf) not in construction_leaves:
                runtime_leaves.append(leaf)

        toolset.apply(collect)
        reject_unsupported_runtime_toolsets(
            runtime_leaves,
            unsupported_kinds=self._unsupported_runtime_toolset_kinds,
            engine=self.engine_name,
        )

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return self._event_stream_handler is not None

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        if self._event_stream_handler is None:
            async for event in stream:
                yield event
            return
        if not self._in_durable_context():
            assert self._process_event_stream is not None
            async for event in self._process_event_stream.wrap_run_event_stream(ctx, stream=stream):
                yield event
            return

        async for event in stream:
            # `ModelResponseStreamEvent`s (exactly the types below) were already delivered
            # live to the handler inside the model-request boundary; workflow-side they're
            # the replay, so only `HandleResponseEvent`s are dispatched to the handler here.
            if not isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent, FinalResultEvent)):
                await self._dispatch_event_stream_event(ctx, event)
            yield event

    @abstractmethod
    def _in_durable_context(self) -> bool:
        """Whether execution is currently inside this engine's workflow or flow."""

    @abstractmethod
    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        """Deliver one workflow-side event inside an engine-specific durable boundary."""

    @staticmethod
    async def _single_event_stream(event: AgentStreamEvent) -> AsyncIterator[AgentStreamEvent]:
        yield event

    def _bind_models(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Build the model registry on a bound copy from the agent's default model and `models=` extras.

        Called from `for_agent`. A concrete default — a `Model` instance, or a string the user
        explicitly mapped to one via `models=` (so it *is* the default) — is registered as
        `'default'` (that key is reserved), and a `models=` string is also kept under its raw
        string so run-time resolution of the default yields the same instance.

        A plain string default is deliberately *not* resolved here: constructing it eagerly could
        build the wrong provider — with authentication/configuration side effects — before a
        sibling [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] gets to reinterpret it.
        Instead no `'default'` is registered; every request for the default carries the raw string
        and re-resolves through the capability chain (or `infer_model`) on the worker.
        """
        if agent.model is None:
            raise UserError(
                f'An agent needs to have a `model` in order to be used with {self.engine_name}, '
                'it cannot be set at agent run time.'
            )
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
        [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability — its
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

        Instances are matched by identity after stripping `WrapperModel` layers,
        so e.g. an `InstrumentedModel`-wrapped default still takes the default's
        fast path. The `model_id` fallback covers models built from a run-time
        string (via `resolve_model_id`) and models an outer capability swaps in
        via `before_model_request`: the worker rebuilds them by looking the
        `model_id` up in the registry, then falling back to the `resolve_model_id`
        capability chain / `infer_model`. This round-trip only reproduces a model
        that the chain or `infer_model` (or the registry under that `model_id`)
        can rebuild — a pre-built instance with a custom provider, client, or
        settings that isn't registered in `models=` will not survive it faithfully.
        """
        unwrapped = unwrap_model(model)
        for model_id, registered in self._models_by_id.items():
            if unwrap_model(registered) is unwrapped:
                return None if model_id == 'default' else model_id
        # Runtime-built or swapped-in Model: round-trip via its model_id string. The worker
        # rebuilds it the same way (registry lookup → resolve_model_id chain → infer_model).
        return model.model_id

    async def _resolve_model_for_request(self, model_id: str | None, run_context: RunContext[AgentDepsT]) -> Model:
        """Rebuild the `Model` for a request inside the activity/step/task, deps-aware.

        Mirrors the workflow-side resolution in `Agent._resolve_model_selection`: run the agent's
        full `resolve_model_id` capability chain — deps-aware user capabilities like
        `ResolveModelId` get first crack, and this capability's registry resolution
        acts as the durable backstop — so a model whose provider depends on the run's
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
