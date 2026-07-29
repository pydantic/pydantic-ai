from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar, TypeAlias, cast

from pydantic import ConfigDict, with_config
from pydantic_core import PydanticSerializationError
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import messages as _messages
from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import DurableToolsetBase
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
    disable_threads,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import (
    CompletedStreamedResponse,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    infer_model,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    TemporalWrapperToolset,
    temporalize_toolset as _default_temporalize_toolset,
    toolset_temporal_activities,
    with_non_retryable_errors,
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
    model_id: str | None = None
    serialized_run_context: Any = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


# The `ModelResponse` arm decodes histories recorded by the deprecated `TemporalAgent`, whose
# stream activity returned the bare response. Remove it (and the workflow-side event synthesis
# in `request_stream_segment`) once those histories have aged out, along with `TemporalAgent`.
_StreamedActivityPayload: TypeAlias = StreamedActivityResult | ModelResponse


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
        agent = Agent('openai:gpt-5.6-sol', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'Temporal'
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )

    _durable_unit_noun = 'activity'
    _durable_container_noun = 'workflow'
    _tool_config_key = 'temporal'

    run_context_type: type[TemporalRunContext[AgentDepsT]]
    """The `TemporalRunContext` subclass used to serialize/deserialize the run context."""

    activity_config: ActivityConfig
    """Base Temporal activity config used for all activities."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
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
            name: Unique agent name used in the Temporal activity names. Defaults to the agent's
                `name` when the capability is bound.
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
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self.run_context_type = run_context_type
        self._deps_type = deps_type

        # Normalize the activity config on copies: mutating the caller's `ActivityConfig` or a
        # `RetryPolicy` shared with other activities would leak the non-retryable entries into
        # them, and repeated construction from the same config would accumulate duplicates.
        activity_config = (
            activity_config.copy() if activity_config else ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
        )
        activity_config['retry_policy'] = with_non_retryable_errors(activity_config.get('retry_policy'))
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
        self._model_activity_config['retry_policy'] = with_non_retryable_errors(
            self._model_activity_config.get('retry_policy')
        )
        self._event_stream_handler_activity_config: ActivityConfig = {
            **activity_config,
            **(event_stream_handler_activity_config or {}),
        }
        self._event_stream_handler_activity_config['retry_policy'] = with_non_retryable_errors(
            self._event_stream_handler_activity_config.get('retry_policy')
        )
        self._toolset_activity_config = toolset_activity_config or {}

        # These are populated by for_agent()
        self._temporal_activities: list[Callable[..., Any]] = []

    def _check_bindable(self) -> None:
        if self.in_durable_context:
            raise UserError(
                'An agent with `TemporalDurability` must be constructed outside of a Temporal workflow, '
                'so its activities can be registered with the worker before the workflow runs. '
                'Construct the agent at module level (or in worker setup code) and reference it from the workflow.'
            )

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Discover the deps type from the agent unless explicitly configured.
        if self._deps_type is None:
            self._deps_type = cast('type[AgentDepsT]', agent.deps_type)

        # Register activities on the bound copy
        self._temporal_activities = []
        self._register_activities(agent)

    def _register_activities(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Register all Temporal activities for model requests, event streaming, and toolsets."""
        activity_name_prefix = f'agent__{self.name}'
        assert self._deps_type is not None  # set by `for_agent` before activities are registered
        deps_type = self._deps_type
        run_context_type = self.run_context_type
        activities: list[Callable[..., Any]] = []

        def register_activity(fn: Callable[..., Any], *, name: str) -> Callable[..., Any]:
            # Temporal's Pydantic payload converter deserializes `deps` by introspecting the activity's
            # annotation, and the concrete deps type is only known once the capability is bound to an agent.
            # Set it here so serialization uses the real type instead of the placeholder the closure declares.
            fn.__annotations__['deps'] = deps_type | None
            return activity.defn(name=name)(fn)

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

        self.request_activity = register_activity(request_activity, name=f'{activity_name_prefix}__model_request')
        activities.append(self.request_activity)

        async def request_stream_activity(params: _RequestParams, deps: Any) -> _StreamedActivityPayload:
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

        self.request_stream_activity = register_activity(
            request_stream_activity, name=f'{activity_name_prefix}__model_request_stream'
        )
        activities.append(self.request_stream_activity)

        if self._event_stream_handler is not None:
            handler = self._event_stream_handler

            async def event_stream_handler_activity(params: _EventStreamHandlerParams, deps: Any) -> None:
                run_context = deserialize_run_context(
                    run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
                )
                await handler(run_context, self._single_event_stream(params.event))

            self.event_stream_handler_activity = register_activity(
                event_stream_handler_activity, name=f'{activity_name_prefix}__event_stream_handler'
            )
            activities.append(self.event_stream_handler_activity)

        async def cancel_suspended_response_activity(params: _CancelParams, deps: Any = None) -> None:
            if params.serialized_run_context is None:
                model = self._models_by_id.get(params.model_id or 'default')
                if model is None:
                    assert params.model_id is not None
                    model = infer_model(params.model_id)
                run_context = None
            else:
                run_context = deserialize_run_context(
                    run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
                )
                model = await self._resolve_model_for_request(params.model_id, run_context)
            # The cancel activity shares `_model_activity_config`, whose default `heartbeat_timeout`
            # would otherwise fail a slow provider-teardown call for missed heartbeats.
            async with _heartbeating():
                with nullcontext() if run_context is None else set_current_run_context(run_context):
                    await model.cancel_suspended_response(params.response)

        self.cancel_suspended_response_activity = register_activity(
            cancel_suspended_response_activity,
            name=f'{activity_name_prefix}__model_cancel_suspended_response',
        )
        activities.append(self.cancel_suspended_response_activity)

        # --- Toolset wrapping ---
        self._register_toolsets(agent)
        for wrapped in self._toolsets_by_id.values():
            activities.extend(toolset_temporal_activities(wrapped))

        self._temporal_activities = activities

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        ts_id = ts.id
        toolset_activity_config = self.activity_config.copy()
        if ts_id is not None:
            toolset_activity_config.update(self._toolset_activity_config.get(ts_id, {}))
        toolset_activity_config['retry_policy'] = with_non_retryable_errors(toolset_activity_config.get('retry_policy'))
        assert self._deps_type is not None
        wrapped = _default_temporalize_toolset(
            ts,
            f'agent__{self.name}',
            toolset_activity_config,
            {},
            self._deps_type,
            self.run_context_type,
            self._agent,
        )
        return wrapped if isinstance(wrapped, (TemporalWrapperToolset, DurableToolsetBase)) else None

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        """All Temporal activities registered by this capability.

        Register these with the Temporal worker, either directly or via
        `AgentPlugin`.
        """
        return self._temporal_activities

    # --- Capability hooks ---

    @property
    def in_durable_context(self) -> bool:
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
        if not self.in_durable_context:
            return await handler()

        with disable_threads(), set_agent_graph_sleep(workflow.sleep):
            try:
                return await handler()
            except PydanticSerializationError as e:  # pragma: lax no cover
                raise UserError(
                    'The `deps` object failed to be serialized. Temporal requires all objects that are passed '
                    "to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        """Reject per-run capabilities whose activities were not registered with the worker."""
        if self.in_durable_context:
            unsafe_capabilities = [capability for capability in capabilities if not capability._safe_at_runtime]
        else:
            unsafe_capabilities = []
        if unsafe_capabilities:
            names = ', '.join(sorted(type(capability).__name__ for capability in unsafe_capabilities))
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
        if not self.in_durable_context:
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

        def params(request: ModelRequestContext) -> _RequestParams:
            return _RequestParams(
                request.messages,
                cast(dict[str, Any] | None, request.model_settings),
                request.model_request_parameters,
                serialized_run_context,
                model_id,
            )

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            config: ActivityConfig = {'summary': f'request model: {model_name}', **self._model_activity_config}
            return await workflow.execute_activity(
                activity=self.request_activity, args=[params(request), deps], **config
            )

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            config: ActivityConfig = {
                'summary': f'request model: {model_name} (stream)',
                **self._model_activity_config,
            }
            result = await workflow.execute_activity(
                activity=self.request_stream_activity, args=[params(request), deps], **config
            )
            if isinstance(result, ModelResponse):
                stream = CompletedStreamedResponse(
                    result,
                    model_request_parameters=request.model_request_parameters,
                    replay_events=True,
                )
                return StreamedActivityResult(response=result, events=[event async for event in stream])
            return result

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            config: ActivityConfig = {
                'summary': f'cancel suspended response: {model_name}',
                **self._model_activity_config,
            }
            await workflow.execute_activity(
                activity=self.cancel_suspended_response_activity,
                args=[
                    _CancelParams(
                        response=response,
                        model_id=model_id,
                        serialized_run_context=serialized_run_context,
                    ),
                    deps,
                ],
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
