from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import timedelta
from typing import Any, Callable, Literal, overload

from pydantic.errors import PydanticUserError
from pydantic_core import PydanticSerializationError
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig
from typing_extensions import Never

from pydantic_ai import (
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent import AbstractAgent, AgentRun, AgentRunResult, EventStreamHandler, RunOutputDataT, WrapperAgent
from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    Tool,
    ToolFuncEither,
)
from pydantic_ai.toolsets import AbstractToolset

from ._model import TemporalModel
from ._toolset import TemporalWrapperToolset, temporalize_toolset


class TemporalAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
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
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            activity_config: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
            model_activity_config: The Temporal activity config to use for model request activities. This is merged with the base activity config.
            toolset_activity_config: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.
            tool_activity_config: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name.
                This is merged with the base and toolset-specific activity configs.
                If a tool does not use IO, you can specify `False` to disable using an activity.
                Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.
            run_context_type: The `TemporalRunContext` subclass to use to serialize and deserialize the run context for use inside a Temporal activity.
                By default, only the `deps`, `retries`, `tool_call_id`, `tool_name`, `retry` and `run_step` attributes will be available.
                To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute.
            temporalize_toolset_func: Optional function to use to prepare "leaf" toolsets (i.e. those that implement their own tool listing and calling) for Temporal by wrapping them in a `TemporalWrapperToolset` that moves methods that require IO to Temporal activities.
                If not provided, only `FunctionToolset` and `MCPServer` will be prepared for Temporal.
                The function takes the toolset, the activity name prefix, the toolset-specific activity config, the tool-specific activity configs and the run context type.
        """
        super().__init__(wrapped)

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

        model_activity_config = model_activity_config or {}
        toolset_activity_config = toolset_activity_config or {}
        tool_activity_config = tool_activity_config or {}

        self._name = name or wrapped.name
        if self._name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
            )

        activity_name_prefix = f'agent__{self.name}'

        activities: list[Callable[..., Any]] = []
        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Temporal, it cannot be set at agent run time.'
            )

        temporal_model = TemporalModel(
            wrapped.model,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config | model_activity_config,
            deps_type=self.deps_type,
            run_context_type=run_context_type,
            event_stream_handler=event_stream_handler or wrapped.event_stream_handler,
        )
        activities.extend(temporal_model.temporal_activities)

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
                run_context_type,
            )
            if isinstance(toolset, TemporalWrapperToolset):
                activities.extend(toolset.temporal_activities)
            return toolset

        temporal_toolsets = [toolset.visit_and_replace(temporalize_toolset) for toolset in wrapped.toolsets]

        self._model = temporal_model
        self._toolsets = temporal_toolsets
        self._temporal_activities = activities

        self._temporal_overrides_active: ContextVar[bool] = ContextVar('_temporal_overrides_active', default=False)

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._temporal_overrides():
            return super().toolsets

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return self._temporal_activities

    @contextmanager
    def _temporal_overrides(self) -> Iterator[None]:
        # We reset tools here as the temporalized function toolset is already in self._toolsets.
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            token = self._temporal_overrides_active.set(True)
            try:
                yield
            except PydanticSerializationError as e:
                raise UserError(
                    "The `deps` object failed to be serialized. Temporal requires all objects that are passed to activities to be serializable using Pydantic's `TypeAdapter`."
                ) from e
            finally:
                self._temporal_overrides_active.reset(token)

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
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
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow() and event_stream_handler is not None:
            raise UserError(
                'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            )

        with self._temporal_overrides():
            return await super().run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                **_deprecated_kwargs,
            )

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
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
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run.

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
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
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
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            raise UserError(
                '`agent.run_stream()` cannot currently be used inside a Temporal workflow. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead. '
                'Please file an issue if this is not sufficient for your use case.'
            )

        async with super().run_stream(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            event_stream_handler=event_stream_handler,
            **_deprecated_kwargs,
        ) as result:
            yield result

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
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
                    instructions=None,
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
                        ]
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=Usage(
                            requests=1, request_tokens=56, response_tokens=7, total_tokens=63
                        ),
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
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
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

        Returns:
            The result of the run.
        """
        if workflow.in_workflow():
            if not self._temporal_overrides_active.get():
                raise UserError(
                    '`agent.iter()` cannot currently be used inside a Temporal workflow. '
                    'Set an `event_stream_handler` on the agent and use `agent.run()` instead. '
                    'Please file an issue if this is not sufficient for your use case.'
                )

            if model is not None:
                raise UserError(
                    'Model cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )
            if toolsets is not None:
                raise UserError(
                    'Toolsets cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
                )

        async with super().iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            **_deprecated_kwargs,
        ) as run:
            yield run

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, toolsets, or tools.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
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

        with super().override(deps=deps, model=model, toolsets=toolsets, tools=tools):
            yield
