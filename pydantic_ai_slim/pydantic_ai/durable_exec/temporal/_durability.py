from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast

from pydantic import ConfigDict, with_config
from pydantic.errors import PydanticUserError
from pydantic_core import PydanticSerializationError
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from pydantic_ai import _utils, messages as _messages, models
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._model import RequestParams, TemporalProviderFactory
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    TemporalWrapperToolset,
    temporalize_toolset as _default_temporalize_toolset,
)


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    """Params for the event stream handler activity."""

    event: _messages.AgentStreamEvent
    serialized_run_context: Any


@dataclass(init=False)
class TemporalDurability(AbstractCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Temporal activities.

    When added to an agent, this capability intercepts model requests and
    wraps toolsets to route their I/O through Temporal activities.
    Outside of workflows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via ``for_agent()``. Only Temporal-specific configuration
    needs to be passed to the constructor.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        durability = TemporalDurability()
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

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
        deps_type: type[AgentDepsT] = type(None),
        provider_factory: TemporalProviderFactory[AgentDepsT] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        activity_config: ActivityConfig | None = None,
        model_activity_config: ActivityConfig | None = None,
        toolset_activity_config: dict[str, ActivityConfig] | None = None,
        tool_activity_config: dict[str, dict[str, ActivityConfig | Literal[False]]] | None = None,
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        temporalize_toolset_func: Callable[
            [
                AbstractToolset[AgentDepsT],
                str,
                ActivityConfig,
                dict[str, ActivityConfig | Literal[False]],
                type[AgentDepsT],
                type[TemporalRunContext[AgentDepsT]],
            ],
            AbstractToolset[AgentDepsT],
        ] = _default_temporalize_toolset,
    ):
        """Create a TemporalDurability capability.

        The agent's model, name, and toolsets are discovered automatically
        when the capability is attached to an agent (via ``for_agent()``).

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                ``'default'``.
            deps_type: The type of the agent's dependencies, needed for Temporal
                serialization of activity parameters.
            provider_factory: Optional callable for instantiating models from
                provider strings at runtime inside activities.
            event_stream_handler: Optional handler for streaming events. When set,
                model requests use a streaming activity that invokes this handler
                inside the activity.
            activity_config: Base Temporal activity config for all activities.
                Defaults to a 60-second ``start_to_close_timeout``.
            model_activity_config: Activity config merged on top of the base for
                model request activities.
            toolset_activity_config: Per-toolset activity configs keyed by toolset ID,
                merged on top of the base config.
            tool_activity_config: Per-tool activity configs keyed by toolset ID then
                tool name. Set a tool's config to ``False`` to skip activity wrapping
                (only valid for async tool functions).
            run_context_type: The `TemporalRunContext` subclass for run context
                serialization/deserialization.
            temporalize_toolset_func: Custom function for wrapping leaf toolsets.
                Defaults to the built-in ``temporalize_toolset``.
        """
        self.run_context_type = run_context_type
        self._provider_factory = provider_factory
        self._event_stream_handler = event_stream_handler
        self._extra_models = dict(models) if models else {}
        self._deps_type = deps_type
        self._temporalize_toolset_func = temporalize_toolset_func

        # Normalize activity config
        activity_config = activity_config or ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
        retry_policy = activity_config.get('retry_policy') or RetryPolicy()
        retry_policy.non_retryable_error_types = [
            *(retry_policy.non_retryable_error_types or []),
            UserError.__name__,
            PydanticUserError.__name__,
        ]
        activity_config['retry_policy'] = retry_policy
        self.activity_config = activity_config
        self._model_activity_config: ActivityConfig = {**activity_config, **(model_activity_config or {})}
        self._toolset_activity_config = toolset_activity_config or {}
        self._tool_activity_config = tool_activity_config or {}

        # These are populated by for_agent()
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._models_by_id: dict[str, Model] = {}
        self._temporal_toolsets_by_id: dict[str, AbstractToolset[AgentDepsT]] = {}
        self._temporal_activities: list[Callable[..., Any]] = []

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> TemporalDurability[AgentDepsT]:
        """Bind to the agent: discover model, name, toolsets and register Temporal activities."""
        if not agent.name:
            raise UserError(
                'An agent needs to have a unique `name` in order to be used with Temporal. '
                "The name will be used to identify the agent's activities within the workflow."
            )
        if not isinstance(agent.model, Model):
            raise UserError(
                'An agent needs to have a concrete `model` in order to be used with Temporal, '
                'it cannot be set at agent run time.'
            )

        self.name = agent.name
        self._agent = agent

        # Build model registry
        self._models_by_id = {'default': agent.model}
        for model_id, model_instance in self._extra_models.items():
            if model_id == 'default':
                raise UserError("Model ID 'default' is reserved for the agent's primary model.")
            self._models_by_id[model_id] = model_instance

        # Register activities
        self._register_activities(agent)

        return self

    def _register_activities(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Register all Temporal activities for model requests, event streaming, and toolsets."""
        activity_name_prefix = f'agent__{self.name}'
        deps_type = self._deps_type
        run_context_type = self.run_context_type
        event_stream_handler = self._event_stream_handler
        activities: list[Callable[..., Any]] = []

        # --- Model request activities ---

        async def request_activity(params: RequestParams, deps: Any | None = None) -> ModelResponse:
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            return await model_for_request.request(
                params.messages,
                cast(ModelSettings | None, params.model_settings),
                params.model_request_parameters,
            )

        request_activity.__annotations__['deps'] = deps_type | None
        self.request_activity = activity.defn(name=f'{activity_name_prefix}__model_request')(request_activity)
        activities.append(self.request_activity)

        async def request_stream_activity(params: RequestParams, deps: AgentDepsT) -> ModelResponse:
            assert event_stream_handler is not None
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            async with model_for_request.request_stream(
                params.messages,
                cast(ModelSettings | None, params.model_settings),
                params.model_request_parameters,
                run_context,
            ) as streamed_response:
                await event_stream_handler(run_context, streamed_response)
                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        request_stream_activity.__annotations__['deps'] = deps_type | None
        self.request_stream_activity = activity.defn(name=f'{activity_name_prefix}__model_request_stream')(
            request_stream_activity
        )
        activities.append(self.request_stream_activity)

        # --- Event stream handler activity ---

        async def event_stream_handler_activity_fn(params: _EventStreamHandlerParams, deps: AgentDepsT) -> None:
            assert event_stream_handler is not None
            run_context = deserialize_run_context(
                run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )

            async def streamed_response():
                yield params.event

            await event_stream_handler(run_context, streamed_response())

        event_stream_handler_activity_fn.__annotations__['deps'] = deps_type
        self.event_stream_handler_activity = activity.defn(name=f'{activity_name_prefix}__event_stream_handler')(
            event_stream_handler_activity_fn
        )
        activities.append(self.event_stream_handler_activity)

        # --- Toolset wrapping ---
        for toolset in agent.toolsets:
            self._temporalize_leaf_toolsets(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activities=activities,
            )

        self._temporal_activities = activities

    def _temporalize_leaf_toolsets(
        self,
        toolset: AbstractToolset[AgentDepsT],
        *,
        activity_name_prefix: str,
        activities: list[Callable[..., Any]],
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

            wrapped = self._temporalize_toolset_func(
                ts,
                activity_name_prefix,
                {**self.activity_config, **self._toolset_activity_config.get(ts_id, {})},
                self._tool_activity_config.get(ts_id, {}),
                self._deps_type,
                self.run_context_type,
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
        ``DurabilityPlugin``.
        """
        return self._temporal_activities

    # --- Model resolution ---

    def _find_model_id(self, model: Model) -> str | None:
        """Find the registry key for a Model instance. Returns None for default."""
        for model_id, registered in self._models_by_id.items():
            if registered is model:
                return None if model_id == 'default' else model_id

        target = model.model_id
        for model_id, registered in self._models_by_id.items():
            if registered.model_id == target:
                return None if model_id == 'default' else model_id

        return target

    def _resolve_model_id(self, model_id: str | None, run_context: RunContext[Any] | None = None) -> Model:
        """Resolve a model ID to a Model instance (used inside activities)."""
        if model_id is None:
            default = self._models_by_id.get('default')
            if default is not None:
                return default
            return next(iter(self._models_by_id.values()))

        if model_id in self._models_by_id:
            return self._models_by_id[model_id]

        return self._infer_model(model_id, run_context)  # pragma: lax no cover

    def _infer_model(self, model_id: str, run_context: RunContext[Any] | None) -> Model:  # pragma: lax no cover
        if self._provider_factory is None or run_context is None:
            return models.infer_model(model_id)
        return models.infer_model(model_id, provider_factory=functools.partial(self._provider_factory, run_context))

    # --- Capability hooks ---

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Disable threads and catch serialization errors inside Temporal workflows."""
        if not workflow.in_workflow():
            return await handler()

        with _utils.disable_threads():
            try:
                return await handler()
            except PydanticSerializationError as e:
                raise UserError(
                    'The `deps` object failed to be serialized. Temporal requires all objects that are passed '
                    "to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e

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

        model_id = self._find_model_id(ctx.model)
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        model_name = model_id or ctx.model.model_id

        if self._event_stream_handler is not None:
            activity_config: ActivityConfig = {
                'summary': f'request model: {model_name} (stream)',
                **self._model_activity_config,
            }
            return await workflow.execute_activity(
                activity=self.request_stream_activity,
                args=[
                    RequestParams(
                        messages=request_context.messages,
                        model_settings=cast(dict[str, Any] | None, request_context.model_settings),
                        model_request_parameters=request_context.model_request_parameters,
                        serialized_run_context=serialized_run_context,
                        model_id=model_id,
                    ),
                    ctx.deps,
                ],
                **activity_config,
            )

        activity_config = {'summary': f'request model: {model_name}', **self._model_activity_config}
        return await workflow.execute_activity(
            activity=self.request_activity,
            args=[
                RequestParams(
                    messages=request_context.messages,
                    model_settings=cast(dict[str, Any] | None, request_context.model_settings),
                    model_request_parameters=request_context.model_request_parameters,
                    serialized_run_context=serialized_run_context,
                    model_id=model_id,
                ),
                ctx.deps,
            ],
            **activity_config,
        )

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError('Image output is not supported with Temporal because of the 2MB payload size limit.')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their Temporal-wrapped versions."""
        if not self._temporal_toolsets_by_id:
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None and ts_id in self._temporal_toolsets_by_id:
                return self._temporal_toolsets_by_id[ts_id]
            return ts

        return toolset.visit_and_replace(swap)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
