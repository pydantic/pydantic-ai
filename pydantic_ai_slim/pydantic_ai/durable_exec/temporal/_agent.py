from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, overload

from pydantic import ConfigDict, with_config
from pydantic.errors import PydanticUserError
from pydantic_core import PydanticSerializationError
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig
from typing_extensions import Never

from pydantic_ai import (
    AbstractToolset,
    AgentRunResultEvent,
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai.agent import AbstractAgent, AgentRun, AgentRunResult, EventStreamHandler, WrapperAgent
from pydantic_ai.agent.abstract import AgentMetadata, Instructions, RunOutputDataT
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    BuiltinToolFunc,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolFuncEither,
)

from ._model import TemporalModel, TemporalProviderFactory
from ._run_context import TemporalRunContext
from ._toolset import TemporalWrapperToolset, temporalize_toolset


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: _messages.AgentStreamEvent
    serialized_run_context: Any


class TemporalAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        models: Mapping[str, Model] | None = None,
        provider_factory: TemporalProviderFactory | None = None,
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
        ] = temporalize_toolset,
    ):
        """Wrap an agent to enable it to be used inside a Temporal workflow, by automatically offloading model requests, tool calls, and MCP server communication to Temporal activities.

        After wrapping, the original agent can still be used as normal outside of the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use in the Temporal activities' names. If not provided, the agent's `name` will be used.
            models:
                Optional mapping of model instances to register with the agent.
                Keys define the names that can be referenced at runtime and the values are `Model` instances.
                Registered model instances can be passed directly to `run(model=...)`.
                If the wrapped agent doesn't have a model set and none is provided to `run()`,
                the first model in this mapping will be used as the default.
            provider_factory:
                Optional callable used when instantiating models from provider strings (those supplied at runtime).
                The callable receives the provider name and the current run context, allowing custom configuration such as injecting API keys stored on `deps`.
                Note: This factory is only used inside Temporal workflows. Outside workflows, model strings are resolved using the default provider behavior.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            activity_config: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
            model_activity_config: The Temporal activity config to use for model request activities. This is merged with the base activity config.
            toolset_activity_config: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.
            tool_activity_config: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name.
                This is merged with the base and toolset-specific activity configs.
                If a tool does not use IO, you can specify `False` to disable using an activity.
                Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.
            run_context_type: The `TemporalRunContext` subclass to use to serialize and deserialize the run context for use inside a Temporal activity.
                By default, only the `deps`, `run_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` attributes will be available.
                To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute.
            temporalize_toolset_func: Optional function to use to prepare "leaf" toolsets (i.e. those that implement their own tool listing and calling) for Temporal by wrapping them in a `TemporalWrapperToolset` that moves methods that require IO to Temporal activities.
                If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Temporal.
                The function takes the toolset, the activity name prefix, the toolset-specific activity config, the tool-specific activity configs and the run context type.
        """
        super().__init__(wrapped)

        self._name = name
        self._event_stream_handler = event_stream_handler
        self.run_context_type = run_context_type

        if self.name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
            )
        # start_to_close_timeout is required
        activity_config = activity_config or ActivityConfig(start_to_close_timeout=timedelta(seconds=60))

        # `pydantic_ai.exceptions.UserError` and `pydantic.errors.PydanticUserError` are not retryable
        retry_policy = activity_config.get('retry_policy') or RetryPolicy()
        retry_policy.non_retryable_error_types = [
            *(retry_policy.non_retryable_error_types or []),
            UserError.__name__,
            PydanticUserError.__name__,
        ]
        activity_config['retry_policy'] = retry_policy
        self.activity_config = activity_config

        model_activity_config = model_activity_config or {}
        toolset_activity_config = toolset_activity_config or {}
        tool_activity_config = tool_activity_config or {}

        activity_name_prefix = f'agent__{self.name}'

        activities: list[Callable[..., Any]] = []

        async def event_stream_handler_activity(params: _EventStreamHandlerParams, deps: AgentDepsT) -> None:
            # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
            # and that only ends up calling `event_stream_handler` if it is set.
            assert self.event_stream_handler is not None

            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)

            async def streamed_response():
                yield params.event

            await self.event_stream_handler(run_context, streamed_response())

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        event_stream_handler_activity.__annotations__['deps'] = self.deps_type

        self.event_stream_handler_activity = activity.defn(name=f'{activity_name_prefix}__event_stream_handler')(
            event_stream_handler_activity
        )
        activities.append(self.event_stream_handler_activity)

        # Get wrapped agent's model if it's a Model instance
        wrapped_model = wrapped.model if isinstance(wrapped.model, Model) else None
        temporal_model = TemporalModel(
            wrapped_model,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config | model_activity_config,
            deps_type=self.deps_type,
            run_context_type=self.run_context_type,
            event_stream_handler=self.event_stream_handler,
            models=models,
            provider_factory=provider_factory,
        )
        activities.extend(temporal_model.temporal_activities)
        self._temporal_model = temporal_model

        def temporalize_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            id = toolset.id
            if id is None:
                raise UserError(
                    "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
                )

            toolset = temporalize_toolset_func(
                toolset,
                activity_name_prefix,
                activity_config | toolset_activity_config.get(id, {}),
                tool_activity_config.get(id, {}),
                self.deps_type,
                self.run_context_type,
            )
            if isinstance(toolset, TemporalWrapperToolset):
                activities.extend(toolset.temporal_activities)
            return toolset

        temporal_toolsets = [toolset.visit_and_replace(temporalize_toolset) for toolset in wrapped.toolsets]

        self._toolsets = temporal_toolsets
        self._temporal_activities = activities

        self._temporal_overrides_active: ContextVar[bool] = ContextVar('_temporal_overrides_active', default=False)

    @property
    def name(self) -> str | None:
        return self._name or super().name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._temporal_model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        elif workflow.in_workflow():
            return self._call_event_stream_handler_activity
        else:
            return handler

    async def _call_event_stream_handler_activity(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        serialized_run_context = self.run_context_type.serialize_run_context(ctx)
        async for event in stream:
            activity_config: ActivityConfig = {'summary': f'handle event: {event.event_kind}', **self.activity_config}
            await workflow.execute_activity(
                activity=self.event_stream_handler_activity,
                args=[
                    _EventStreamHandlerParams(
                        event=event,
                        serialized_run_context=serialized_run_context,
                    ),
                    ctx.deps,
                ],
                **activity_config,
            )

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._temporal_overrides(force=True):
            return super().toolsets

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return self._temporal_activities

    @contextmanager
    def _temporal_overrides(
        self, *, model: models.Model | models.KnownModelName | str | None = None, force: bool = False
    ) -> Iterator[None]:
        """Context manager for workflow-specific overrides.

        When called outside a workflow, this is a no-op.
        When called inside a workflow, it overrides the model and toolsets.
        """
        if not workflow.in_workflow() and not force:
            yield
            return

        # We reset tools here as the temporalized function toolset is already in self._toolsets.
        # Override model and set the model for workflow execution
        with (
            super().override(model=self._temporal_model, toolsets=self._toolsets, tools=[]),
            self._temporal_model.using_model(model),
            _utils.disable_threads(),
        ):
            temporal_active_token = self._temporal_overrides_active.set(True)
            try:
                yield
            except PydanticSerializationError as e:
                raise UserError(
                    "The `deps` object failed to be serialized. Temporal requires all objects that are passed to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e
            finally:
                self._temporal_overrides_active.reset(temporal_active_token)

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
                Inside workflows, only registered model instances, registered names, or provider strings are valid.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            if event_stream_handler is not None:
                raise UserError(
                    'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )
            resolved_model = None
        else:
            resolved_model = self._temporal_model.resolve_model(model)

        with self._temporal_overrides(model=model):
            return await super().run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=resolved_model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                event_stream_handler=event_stream_handler or self.event_stream_handler,
                **_deprecated_kwargs,
            )

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.output)
        #> The capital of Italy is Rome.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_sync()` cannot be used inside a Temporal workflow. Use `await agent.run()` instead.'
            )

        return super().run_sync(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_stream()` cannot be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        async with super().run_stream(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            event_stream_handler=event_stream_handler,
            builtin_tools=builtin_tools,
            **_deprecated_kwargs,
        ) as result:
            yield result

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-4o')

        async def main():
            events: list[AgentStreamEvent | AgentRunResultEvent] = []
            async for event in agent.run_stream_events('What is the capital of France?'):
                events.append(event)
            print(events)
            '''
            [
                PartStartEvent(index=0, part=TextPart(content='The capital of ')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='France is Paris. ')),
                PartEndEvent(
                    index=0, part=TextPart(content='The capital of France is Paris. ')
                ),
                AgentRunResultEvent(
                    result=AgentRunResult(output='The capital of France is Paris. ')
                ),
            ]
            '''
        ```

        Arguments are the same as for [`self.run`][pydantic_ai.agent.AbstractAgent.run],
        except that `event_stream_handler` is now allowed.

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_stream_events()` cannot be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        return super().run_stream_events(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
        )

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    instructions_functions=[],
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                            )
                        ],
                        timestamp=datetime.datetime(...),
                        run_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                    )
                ),
                End(data=FinalResult(output='The capital of France is Paris.')),
            ]
            '''
            print(agent_run.result.output)
            #> The capital of France is Paris.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            if not self._temporal_overrides_active.get():
                raise UserError(
                    '`agent.iter()` cannot be used inside a Temporal workflow. '
                    'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
                )

            assert model is None, 'Temporal overrides must set the model before `agent.iter()` is invoked'

            if toolsets is not None:
                raise UserError(
                    'Toolsets cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )

            resolved_model = None
        else:
            resolved_model = self._temporal_model.resolve_model(model)

        async with super().iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=resolved_model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            **_deprecated_kwargs,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent name, dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if workflow.in_workflow():
            if _utils.is_set(model):
                raise UserError(
                    'Model cannot be contextually overridden inside a Temporal workflow, it must be set at agent creation time.'
                )
            if _utils.is_set(toolsets):
                raise UserError(
                    'Toolsets cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
                )
            if _utils.is_set(tools):
                raise UserError(
                    'Tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
                )

        with super().override(
            name=name,
            deps=deps,
            model=model,
            toolsets=toolsets,
            tools=tools,
            instructions=instructions,
        ):
            yield
