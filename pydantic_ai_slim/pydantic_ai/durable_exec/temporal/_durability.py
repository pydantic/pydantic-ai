from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from pydantic import ConfigDict, with_config
from pydantic.errors import PydanticUserError
from pydantic_core import PydanticSerializationError
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from pydantic_ai import messages as _messages
from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
    disable_threads,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    TemporalWrapperToolset,
    temporalize_toolset as _default_temporalize_toolset,
)


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    """Serializable arguments for the model-request Temporal activity."""

    messages: list[_messages.ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None


@dataclass
class _CancelParams:
    response: ModelResponse
    serialized_run_context: Any
    model_id: str | None = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


_DEFAULT_MODEL_HEARTBEAT_TIMEOUT = timedelta(seconds=30)
"""Default `heartbeat_timeout` for the model-request activities.

A model request activity can legitimately run for a long time while waiting for one
provider round trip. Heartbeating lets Temporal distinguish that long-but-healthy
activity from a crashed worker, and makes workflow cancellation deliverable
mid-request (cancellation reaches an activity as a response to a heartbeat).
"""


@asynccontextmanager
async def _heartbeating() -> AsyncGenerator[None]:
    """Emit periodic activity heartbeats in the background while the wrapped request runs.

    The beat interval is derived from the activity's configured `heartbeat_timeout` so a
    custom (shorter or longer) timeout keeps working; the SDK additionally throttles
    outgoing heartbeats on its own. Without a configured timeout, heartbeats are inert but
    harmless, so a plain 5-second cadence is fine.

    The heartbeat task is supervised: if `beat()` itself crashes, the failure surfaces
    once the wrapped request completes, so the activity fails loudly instead of having
    silently run without heartbeats (the server would have failed the attempt via
    `heartbeat_timeout` anyway had the crash come early). An exception from the wrapped
    request always wins — a heartbeat failure never replaces it.
    """

    async def beat() -> None:
        timeout = activity.info().heartbeat_timeout
        interval = timeout.total_seconds() / 2 if timeout else 5.0
        while True:
            activity.heartbeat()
            await asyncio.sleep(interval)

    task = asyncio.create_task(beat())
    try:
        yield
    except BaseException:
        # The request's exception is already propagating; a heartbeat failure must not
        # replace it.
        task.cancel()
        with suppress(BaseException):
            await task
        raise
    else:
        task.cancel()
        with suppress(asyncio.CancelledError):
            # Anything but our own cancellation is a `beat()` crash — propagate it.
            await task


def _with_non_retryable_errors(retry_policy: RetryPolicy | None) -> RetryPolicy:
    """Return a copy of `retry_policy` with the framework's non-retryable errors ensured.

    `UserError` and `PydanticUserError` won't be fixed by re-running the activity, and an
    `UnexpectedModelBehavior` (e.g. a model staying suspended past the continuation ceiling)
    would only re-incur the request's cost. A user-supplied `retry_policy` in any activity
    config would otherwise replace the base policy wholesale and silently drop these, so the
    guarantee is re-applied after every merge that may override the policy.
    """
    retry_policy = copy.copy(retry_policy) if retry_policy else RetryPolicy()
    existing = retry_policy.non_retryable_error_types or []
    additional = [UserError.__name__, PydanticUserError.__name__, UnexpectedModelBehavior.__name__]
    retry_policy.non_retryable_error_types = [*existing, *(name for name in additional if name not in existing)]
    return retry_policy


@dataclass(init=False)
class TemporalDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Temporal activities.

    When added to an agent, this capability intercepts model requests and
    wraps toolsets to route their I/O through Temporal activities.
    Outside of workflows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via `for_agent()`. Only Temporal-specific configuration
    needs to be passed to the constructor.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        durability = TemporalDurability()
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'Temporal'

    name: str
    """Unique agent name used as a prefix for Temporal activity names."""

    run_context_type: type[TemporalRunContext[AgentDepsT]]
    """The `TemporalRunContext` subclass used to serialize/deserialize the run context."""

    activity_config: ActivityConfig
    """Base Temporal activity config used for all activities."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        deps_type: type[AgentDepsT] | None = None,
        activity_config: ActivityConfig | None = None,
        model_activity_config: ActivityConfig | None = None,
        event_stream_handler_activity_config: ActivityConfig | None = None,
        toolset_activity_config: dict[str, ActivityConfig] | None = None,
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    ):
        """Create a TemporalDurability capability.

        The agent's model, name, and toolsets are discovered automatically
        when the capability is attached to an agent (via `for_agent()`).

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                activity boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                is sent as its `model_id` string and rebuilt on the worker by
                registry lookup, then the agent's `resolve_model_id` capability
                chain / `infer_model`. Register an instance here (and reference it
                by key or pass the registered instance) whenever its `model_id`
                alone wouldn't rebuild it faithfully — e.g. a custom provider,
                client, or settings. Model-name strings never need registering;
                to customize how they're built (e.g. a custom provider), use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside model-request activities, and tool events are handled in
                per-event activities.
            deps_type: The type of the agent's dependencies, needed for Temporal
                serialization of activity parameters. Defaults to the agent's own
                `deps_type`, discovered when the capability binds via `for_agent()`.
            activity_config: Base Temporal activity config for all activities.
                Defaults to a 60-second `start_to_close_timeout`.
            model_activity_config: Activity config merged on top of the base for
                model request activities.
            event_stream_handler_activity_config: Activity config merged on top of the base for
                event stream handler activities.
            toolset_activity_config: Per-toolset activity configs keyed by toolset ID,
                merged on top of the base config.
            run_context_type: The `TemporalRunContext` subclass for run context
                serialization/deserialization.

        Note:
            Per-tool activity config (custom timeouts, retry policies, or disabling
            activity wrapping entirely) is configured via tool metadata:

            ```python {test="skip" lint="skip"}
            @my_toolset.tool(metadata={'temporal': ActivityConfig(...)})
            async def my_slow_tool(...): ...
            ```

            or via the `SetToolMetadata` capability for selector-based config.
            Setting the `'temporal'` key to `False` skips activity wrapping
            (only valid for async tool functions).
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler)
        self.run_context_type = run_context_type
        self._deps_type = deps_type

        # Normalize the activity config on copies: mutating the caller's `ActivityConfig` or a
        # `RetryPolicy` shared with other activities would leak the non-retryable entries into
        # them, and repeated construction from the same config would accumulate duplicates.
        activity_config = (
            copy.copy(activity_config)
            if activity_config
            else ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
        )
        activity_config['retry_policy'] = _with_non_retryable_errors(activity_config.get('retry_policy'))
        self.activity_config = activity_config
        # The model activities heartbeat in the background (see `_heartbeating`), so give them a
        # heartbeat timeout by default; an explicit `heartbeat_timeout` in either config wins.
        self._model_activity_config: ActivityConfig = {
            'heartbeat_timeout': _DEFAULT_MODEL_HEARTBEAT_TIMEOUT,
            **activity_config,
            **(model_activity_config or {}),
        }
        # A `retry_policy` in `model_activity_config` would otherwise replace the normalized
        # base policy and drop the non-retryable entries.
        self._model_activity_config['retry_policy'] = _with_non_retryable_errors(
            self._model_activity_config.get('retry_policy')
        )
        self._event_stream_handler_activity_config: ActivityConfig = {
            **activity_config,
            **(event_stream_handler_activity_config or {}),
        }
        self._event_stream_handler_activity_config['retry_policy'] = _with_non_retryable_errors(
            self._event_stream_handler_activity_config.get('retry_policy')
        )
        self._toolset_activity_config = toolset_activity_config or {}

        # These are populated by for_agent()
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._temporal_toolsets_by_id: dict[str, WrapperToolset[AgentDepsT]] = {}
        self._temporal_activities: list[Callable[..., Any]] = []
        self._bound_capability_classes: frozenset[type[AbstractCapability[Any]]] = frozenset()

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> TemporalDurability[AgentDepsT]:
        """Bind to the agent: discover model, name, toolsets and register Temporal activities.

        Returns a new bound instance; the original capability is left pristine so the
        same instance can be passed to multiple agents. Use
        `TemporalDurability.from_agent(agent)` to retrieve the bound copy.
        """
        if workflow.in_workflow():
            raise UserError(
                'An agent with `TemporalDurability` must be constructed outside of a Temporal workflow, '
                'so its activities can be registered with the worker before the workflow runs. '
                'Construct the agent at module level (or in worker setup code) and reference it from the workflow.'
            )
        if not agent.name:
            raise UserError(
                'An agent needs to have a unique `name` in order to be used with Temporal. '
                "The name will be used to identify the agent's activities within the workflow."
            )
        bound = copy.copy(self)
        bound.name = agent.name
        bound._agent = agent

        # Build model registry (shared with the other durability capabilities)
        bound._bind_models(agent)

        # Snapshot the leaf capability classes registered with the agent so we can
        # detect runtime additions (which would bypass activity registration).
        bound_classes: set[type[AbstractCapability[Any]]] = set()

        def _collect_class(cap: AbstractCapability[Any]) -> None:
            bound_classes.add(type(cap))

        agent.root_capability.apply(_collect_class)
        bound._bound_capability_classes = frozenset(bound_classes)

        # Discover the deps type from the agent unless explicitly configured.
        if bound._deps_type is None:
            bound._deps_type = cast('type[AgentDepsT]', agent.deps_type)

        # Register activities on the bound copy
        bound._temporal_toolsets_by_id = {}
        bound._temporal_activities = []
        bound._register_activities(agent)

        return bound

    def _register_activities(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Register all Temporal activities for model requests, event streaming, and toolsets."""
        activity_name_prefix = f'agent__{self.name}'
        assert self._deps_type is not None  # set by `for_agent` before activities are registered
        deps_type = self._deps_type
        run_context_type = self.run_context_type
        activities: list[Callable[..., Any]] = []

        # --- Model request activities ---

        async def request_activity(params: _RequestParams, deps: Any | None = None) -> ModelResponse:
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = await self._resolve_model_for_request(params.model_id, run_context)
            async with _heartbeating():
                with set_current_run_context(run_context):
                    return await model_for_request.request(
                        params.messages,
                        cast(ModelSettings | None, params.model_settings),
                        params.model_request_parameters,
                    )

        request_activity.__annotations__['deps'] = deps_type | None
        self.request_activity = activity.defn(name=f'{activity_name_prefix}__model_request')(request_activity)
        activities.append(self.request_activity)

        async def request_stream_activity(params: _RequestParams, deps: AgentDepsT) -> StreamedActivityResult:
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = await self._resolve_model_for_request(params.model_id, run_context)
            async with _heartbeating():
                with set_current_run_context(run_context):
                    async with model_for_request.request_stream(
                        params.messages,
                        cast(ModelSettings | None, params.model_settings),
                        params.model_request_parameters,
                        run_context,
                    ) as streamed_response:
                        events = await capture_event_stream(
                            run_context=run_context,
                            stream=streamed_response,
                            handler=self._event_stream_handler,
                        )
                return StreamedActivityResult(response=streamed_response.get(), events=events)

        request_stream_activity.__annotations__['deps'] = deps_type | None
        self.request_stream_activity = activity.defn(name=f'{activity_name_prefix}__model_request_stream')(
            request_stream_activity
        )
        activities.append(self.request_stream_activity)

        if self._event_stream_handler is not None:
            handler = self._event_stream_handler

            async def event_stream_handler_activity(params: _EventStreamHandlerParams, deps: AgentDepsT) -> None:
                run_context = deserialize_run_context(
                    run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
                )
                await handler(run_context, self._single_event_stream(params.event))

            event_stream_handler_activity.__annotations__['deps'] = deps_type | None
            self.event_stream_handler_activity = activity.defn(name=f'{activity_name_prefix}__event_stream_handler')(
                event_stream_handler_activity
            )
            activities.append(self.event_stream_handler_activity)

        async def cancel_suspended_response_activity(params: _CancelParams, deps: AgentDepsT) -> None:
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model = await self._resolve_model_for_request(params.model_id, run_context)
            with set_current_run_context(run_context):
                await model.cancel_suspended_response(params.response)

        cancel_suspended_response_activity.__annotations__['deps'] = deps_type | None
        self.cancel_suspended_response_activity = activity.defn(
            name=f'{activity_name_prefix}__model_cancel_suspended_response'
        )(cancel_suspended_response_activity)
        activities.append(self.cancel_suspended_response_activity)

        # --- Toolset wrapping ---
        for toolset in agent.toolsets:
            self._temporalize_leaf_toolsets(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activities=activities,
                deps_type=deps_type,
            )

        self._temporal_activities = activities

    def _temporalize_leaf_toolsets(
        self,
        toolset: AbstractToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activities: list[Callable[..., Any]],
        deps_type: type[AgentDepsT],
    ) -> None:
        """Wrap each leaf toolset in a Temporal wrapper and collect activities."""

        def temporalize(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is None:
                raise UserError(
                    "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) "
                    'need to have a unique `id` in order to be used with Temporal. '
                    "The ID will be used to identify the toolset's activities within the workflow."
                )

            existing = self._temporal_toolsets_by_id.get(ts_id)
            if existing is not None:
                if existing.wrapped is ts:
                    # The same toolset instance can appear in more than one place in the
                    # tree; reuse its wrapper so its activities register exactly once.
                    return existing
                # A distinct toolset under an already-registered `id` would silently
                # replace it in the registry and route both toolsets' calls to one wrapper.
                raise UserError(
                    f'Two toolsets have the same `id` {ts_id!r}. Toolset `id`s must be unique among all '
                    "toolsets registered with the same agent, as they identify the toolset's activities "
                    'within the workflow.'
                )

            toolset_activity_config: ActivityConfig = {
                **self.activity_config,
                **self._toolset_activity_config.get(ts_id, {}),
            }
            # A `retry_policy` in the per-toolset config would otherwise replace the
            # normalized base policy and drop the non-retryable entries.
            toolset_activity_config['retry_policy'] = _with_non_retryable_errors(
                toolset_activity_config.get('retry_policy')
            )
            wrapped = _default_temporalize_toolset(
                ts,
                activity_name_prefix,
                toolset_activity_config,
                {},  # per-tool config comes from tool metadata on the capability path
                deps_type,
                self.run_context_type,
                self._agent,
            )
            if isinstance(wrapped, TemporalWrapperToolset):
                activities.extend(wrapped.temporal_activities)
                self._temporal_toolsets_by_id[ts_id] = wrapped
            return wrapped

        toolset.visit_and_replace(temporalize)

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        """All Temporal activities registered by this capability.

        Register these with the Temporal worker, either directly or via
        `AgentPlugin`.
        """
        return self._temporal_activities

    # --- Capability hooks ---

    def _in_durable_context(self) -> bool:
        return workflow.in_workflow()

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        config: ActivityConfig = {
            'summary': f'handle event: {event.event_kind}',
            **self._event_stream_handler_activity_config,
        }
        await workflow.execute_activity(
            activity=self.event_stream_handler_activity,
            args=[
                _EventStreamHandlerParams(event=event, serialized_run_context=serialized_run_context),
                ctx.deps,
            ],
            **config,
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Disable threads and catch serialization errors inside Temporal workflows."""
        if not workflow.in_workflow():
            return await handler()

        self._validate_per_run_capabilities(ctx)

        with disable_threads(), set_agent_graph_sleep(workflow.sleep):
            try:
                return await handler()
            except PydanticSerializationError as e:  # pragma: lax no cover
                raise UserError(
                    'The `deps` object failed to be serialized. Temporal requires all objects that are passed '
                    "to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e

    def _validate_per_run_capabilities(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reject per-run capabilities not registered at agent construction time.

        Temporal needs activities registered with the worker before a workflow runs.
        Capabilities added per-run (via `agent.run(capabilities=[...])`) bypass
        `for_agent()` activity registration, so any toolsets or model wrappers they
        contribute would silently execute in workflow code (non-deterministic, no
        retry semantics). Reject by class identity: if a leaf in `ctx.root_capability`
        has a type the bound chain didn't see, raise `UserError`.

        Skipped when the bound tree contains a `DynamicCapability` — the resolved
        factory result replaces the `DynamicCapability` in the run-time tree, so the
        runtime-class check would falsely reject any class produced by the factory.
        Issue #5253 tracks proper end-to-end durable support for `DynamicCapability`
        toolsets; until that lands, the static class check is relaxed for any agent
        that uses dynamic capabilities.

        No equivalent check on DBOS/Prefect: their durable units (steps, tasks) are
        plain decorated callables registered at first-use rather than worker boot, so
        per-run capabilities can register on the fly without violating durability.
        """
        assert ctx.root_capability is not None

        if any(issubclass(cls, DynamicCapability) for cls in self._bound_capability_classes):
            return

        runtime_classes: set[type[AbstractCapability[Any]]] = set()

        def _collect(cap: AbstractCapability[Any]) -> None:
            runtime_classes.add(type(cap))

        ctx.root_capability.apply(_collect)
        # Capabilities that opt in via `_safe_at_runtime = True` (e.g. `Instrumentation`,
        # auto-injected per-run by `Agent.iter()` when `instrument=…` / `LogfirePlugin` is
        # used) don't introduce new toolsets, native tools, or model wrapping, so they
        # don't need activities registered with the worker upfront.
        extra = {cls for cls in runtime_classes - self._bound_capability_classes if not cls._safe_at_runtime}
        if extra:
            names = ', '.join(sorted(c.__name__ for c in extra))
            raise UserError(
                f'Capabilities added per-run inside a Temporal workflow are not supported: {names}. '
                'Temporal activities must be registered with the worker before the workflow runs. '
                'Attach all capabilities at agent construction time so `TemporalDurability.for_agent()` '
                'can register their activities.'
            )

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through Temporal activities when inside a workflow."""
        if not workflow.in_workflow():
            return await handler(request_context)

        self._validate_model_request_parameters(request_context.model_request_parameters)

        # Prefer the run's original model-id string (provenance) as the selection token;
        # a model swapped in by an outer capability falls back to `_find_model_id` on
        # `request_context.model` (which an outer instrumentation capability may have
        # already unwrapped — instances are unwrap-matched by identity).
        model_id = self._model_id_for_request(ctx, request_context)
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        model_name = model_id or request_context.model.model_id
        deps = ctx.deps

        def params(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> _RequestParams:
            return _RequestParams(
                messages, cast(dict[str, Any] | None, settings), parameters, serialized_run_context, model_id
            )

        async def request_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> ModelResponse:
            config: ActivityConfig = {'summary': f'request model: {model_name}', **self._model_activity_config}
            return await workflow.execute_activity(
                activity=self.request_activity, args=[params(messages, settings, parameters), deps], **config
            )

        async def request_stream_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> StreamedActivityResult:
            config: ActivityConfig = {
                'summary': f'request model: {model_name} (stream)',
                **self._model_activity_config,
            }
            return await workflow.execute_activity(
                activity=self.request_stream_activity, args=[params(messages, settings, parameters), deps], **config
            )

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            config: ActivityConfig = {
                'summary': f'cancel suspended response: {model_name}',
                **self._model_activity_config,
            }
            await workflow.execute_activity(
                activity=self.cancel_suspended_response_activity,
                args=[_CancelParams(response, serialized_run_context, model_id), deps],
                **config,
            )

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError('Image output is not supported with Temporal because of the 2MB payload size limit.')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their Temporal-wrapped versions."""
        self._reject_runtime_toolsets(toolset)

        if not self._temporal_toolsets_by_id:
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None and ts_id in self._temporal_toolsets_by_id:
                return self._temporal_toolsets_by_id[ts_id]
            return ts

        return toolset.visit_and_replace(swap)

    def _reject_runtime_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Reject executing toolsets added per-run inside a workflow.

        The run toolset assembled by the agent contains both construction-time toolsets
        (whose activities `for_agent` registered with the worker) and any extras passed
        via `run(toolsets=...)`. Executing extras would run un-wrapped inside the workflow
        — no activity, non-deterministic on replay — so they're rejected explicitly, like
        the deprecated `TemporalAgent` does. Non-executing toolsets like `ExternalToolset`
        pass through. Only applies inside a workflow; outside one the capability is
        transparent and any toolset is fine.
        """
        if not workflow.in_workflow():
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
            runtime_leaves, unsupported_kinds=frozenset({'function', 'mcp', 'dynamic'}), engine='Temporal'
        )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
