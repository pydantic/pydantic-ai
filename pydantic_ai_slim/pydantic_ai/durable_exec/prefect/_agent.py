from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import timedelta
from typing import TYPE_CHECKING, Any, overload

from prefect import flow, get_run_logger, task
from prefect.context import FlowRunContext
from prefect.runner import Runner
from prefect.settings import get_current_settings
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import run_coro_as_sync
from prefect.utilities.slugify import slugify
from typing_extensions import Never

from pydantic_ai import (
    AbstractToolset,
    _utils,
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai.agent import AbstractAgent, AgentRun, AgentRunResult, EventStreamHandler, WrapperAgent
from pydantic_ai.agent.abstract import Instructions, RunOutputDataT
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolFuncEither,
)

from ._model import PrefectModel
from ._types import TaskConfig, default_task_config


class PrefectAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
    ):
        """Wrap an agent to enable it with Prefect durable flows, by automatically offloading model requests, tool calls, and MCP server communication to Prefect tasks.

        After wrapping, the original agent can still be used as normal outside of the Prefect flow.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use as the Prefect flow name prefix. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            mcp_task_config: The base Prefect task config to use for MCP server tasks. If no config is provided, use the default settings of Prefect.
            model_task_config: The Prefect task config to use for model request tasks. If no config is provided, use the default settings of Prefect.
            tool_task_config: The default Prefect task config to use for tool calls. If no config is provided, use the default settings of Prefect.
            tool_task_config_by_name: Per-tool task configuration. Keys are tool names, values are TaskConfig or None (None disables task wrapping for that tool).
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        if self._name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Prefect. The name will be used to identify the agent's flows and tasks."
            )

        # Merge the config with the default Prefect config
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._model_task_config = default_task_config | (model_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._tool_task_config_by_name = tool_task_config_by_name or {}

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Prefect, it cannot be set at agent run time.'
            )

        prefect_model = PrefectModel(
            wrapped.model,
            task_config=self._model_task_config,
            event_stream_handler=self.event_stream_handler,
        )
        self._model = prefect_model

        prefect_toolsets = [toolset.visit_and_replace(self._prefectify_toolset) for toolset in wrapped.toolsets]
        self._toolsets = prefect_toolsets

        # Context variable to track when we're inside this agent's Prefect flow
        self._in_prefect_agent_flow: ContextVar[bool] = ContextVar(
            f'_in_prefect_agent_flow_{self._name}', default=False
        )

    def _prefectify_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Convert a toolset to its Prefect equivalent."""
        # Replace MCPServer with PrefectMCPServer
        try:
            from pydantic_ai.mcp import MCPServer

            from ._mcp_server import PrefectMCPServer
        except ImportError:
            pass
        else:
            if isinstance(toolset, MCPServer):
                return PrefectMCPServer(
                    wrapped=toolset,
                    task_config=self._mcp_task_config,
                )

        # Replace FunctionToolset with PrefectFunctionToolset
        from pydantic_ai import FunctionToolset

        from ._function_toolset import PrefectFunctionToolset

        if isinstance(toolset, FunctionToolset):
            return PrefectFunctionToolset(
                wrapped=toolset,
                task_config=self._tool_task_config,
                tool_task_config=self._tool_task_config_by_name,
            )

        return toolset

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
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        elif FlowRunContext.get() is not None:
            # Special case if it's in a Prefect flow, we need to iterate through all events and call the handler.
            return self._call_event_stream_handler_in_flow
        else:
            return handler

    async def _call_event_stream_handler_in_flow(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        handler = self._event_stream_handler or super().event_stream_handler
        assert handler is not None

        # Create a task to handle each event
        @task(name='Handle Stream Event')
        async def event_stream_handler_task(event: _messages.AgentStreamEvent) -> None:
            async def streamed_response():
                yield event

            await handler(ctx, streamed_response())

        async for event in stream:
            await event_stream_handler_task(event)

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._prefect_overrides():
            return super().toolsets

    @contextmanager
    def _prefect_overrides(self) -> Iterator[None]:
        # Override with PrefectModel and PrefectMCPServer in the toolsets.
        with (
            super().override(model=self._model, toolsets=self._toolsets, tools=[]),
            self.sequential_tool_calls(),
        ):
            yield

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
            deferred_tool_results: Optional results for deferred tool calls in the message history.
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

        @flow(name=f'{self._name} Run')
        async def wrapped_run_flow() -> AgentRunResult[Any]:
            logger = get_run_logger()
            prompt_str = str(user_prompt)
            logger.info(f'Starting agent run with prompt: {prompt_str}')

            # Mark that we're inside a PrefectAgent flow
            token = self._in_prefect_agent_flow.set(True)
            try:
                with self._prefect_overrides():
                    result = await super(WrapperAgent, self).run(
                        user_prompt,
                        output_type=output_type,
                        message_history=message_history,
                        deferred_tool_results=deferred_tool_results,
                        model=model,
                        deps=deps,
                        model_settings=model_settings,
                        usage_limits=usage_limits,
                        usage=usage,
                        infer_name=infer_name,
                        toolsets=toolsets,
                        event_stream_handler=event_stream_handler,
                    )
                    logger.info(
                        f'Agent run completed. Requests: {result.usage().requests}, Tool calls: {result.usage().tool_calls}'
                    )
                    return result
            finally:
                self._in_prefect_agent_flow.reset(token)

        return await wrapped_run_flow()

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
            deferred_tool_results: Optional results for deferred tool calls in the message history.
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

        @flow(name=f'{self._name} Sync Run')
        def wrapped_run_sync_flow() -> AgentRunResult[Any]:
            logger = get_run_logger()
            prompt_str = str(user_prompt)
            prompt_preview = prompt_str[:100] + '...' if len(prompt_str) > 100 else prompt_str
            logger.info(f'Starting sync agent run with prompt: {prompt_preview}')

            # Mark that we're inside a PrefectAgent flow
            token = self._in_prefect_agent_flow.set(True)
            try:
                with self._prefect_overrides():
                    # Using `run_coro_as_sync` from Prefect with async `run` to avoid event loop conflicts.
                    result = run_coro_as_sync(
                        super(PrefectAgent, self).run(
                            user_prompt,
                            output_type=output_type,
                            message_history=message_history,
                            deferred_tool_results=deferred_tool_results,
                            model=model,
                            deps=deps,
                            model_settings=model_settings,
                            usage_limits=usage_limits,
                            usage=usage,
                            infer_name=infer_name,
                            toolsets=toolsets,
                            event_stream_handler=event_stream_handler,
                        )
                    )
                    logger.info(
                        f'Sync agent run completed. Requests: {result.usage().requests}, Tool calls: {result.usage().tool_calls}'
                    )
                    return result
            finally:
                self._in_prefect_agent_flow.reset(token)

        return wrapped_run_sync_flow()

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
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
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
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
            deferred_tool_results: Optional results for deferred tool calls in the message history.
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
        if model is not None and not isinstance(model, PrefectModel):
            raise UserError(
                'Non-Prefect model cannot be set at agent run time inside a Prefect flow, it must be set at agent creation time.'
            )

        with self._prefect_overrides():
            async with super().iter(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
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
        instructions: Instructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies, model, toolsets, tools, or instructions.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            instructions: The instructions to use instead of the instructions registered with the agent.
        """
        if _utils.is_set(model) and not isinstance(model, (PrefectModel)):
            raise UserError(
                'Non-Prefect model cannot be contextually overridden inside a Prefect flow, it must be set at agent creation time.'
            )

        with super().override(deps=deps, model=model, toolsets=toolsets, tools=tools, instructions=instructions):
            yield

    async def serve(
        self,
        *,
        name: str | None = None,
        interval: int | float | timedelta | None = None,
        cron: str | None = None,
        rrule: str | None = None,
        paused: bool | None = None,
        triggers: list[Any] | None = None,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        version: str | None = None,
        enforce_parameter_schema: bool = True,
        pause_on_shutdown: bool = True,
        print_starting_message: bool = True,
        limit: int | None = None,
        webserver: bool = False,
    ) -> None:
        """Serve the agent as a Prefect deployment.

        This method creates a Prefect deployment for the agent's run flow and starts
        a long-running process that monitors for scheduled work from the Prefect server.

        Example:
        ```python {title="prefect_agent_serve.py" test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.prefect import PrefectAgent

        agent = Agent('openai:gpt-4o', name='my_agent')
        prefect_agent = PrefectAgent(agent)

        # Serve with a schedule
        prefect_agent.serve(
            name='my-agent-deployment',
            cron='0 9 * * *',  # Run daily at 9am
            tags=['production', 'daily'],
        )
        ```

        Args:
            name: Name for the created deployment.
            interval: Schedule interval in seconds, or as a timedelta.
            cron: Cron schedule string (e.g., '0 9 * * *' for daily at 9am).
            rrule: iCalendar RRule schedule string.
            paused: Whether the schedule should be paused initially.
            triggers: Event triggers for the deployment.
            parameters: Default parameters to pass to agent runs (e.g., {'user_prompt': 'default prompt'}).
            description: Description for the deployment. Defaults to the agent's docstring.
            tags: Tags for the deployment for filtering and organization.
            version: Version identifier for the deployment.
            enforce_parameter_schema: Whether to validate parameters against the flow's schema.
            pause_on_shutdown: Whether to pause the schedule when the serve process is stopped.
            print_starting_message: Whether to print a message when the server starts.
            limit: Maximum number of concurrent runs for this deployment.
            webserver: Whether to start a webserver for the deployment.

        Returns:
            None. This method blocks indefinitely until interrupted.
        """
        if TYPE_CHECKING:
            assert self._name is not None

        # Serve the flow with the provided configuration
        runner = Runner(
            name=f'{self._name} Runner',
            limit=limit,
            webserver=webserver,
            pause_on_shutdown=pause_on_shutdown,
        )

        # validate parameters is False because the type definitions get lost during serialization, which
        # we'll need to fix on the Prefect side.
        @flow(name=f'run-{slugify(self._name)}', validate_parameters=False)
        # we can't support all `.run` args because Prefect needs to be able to generate a static schema
        # for parameters at runtime (doesn't work for generic types) and all parameters need to be JSON
        # serializable to allow providing them via the REST API
        async def served_run(
            user_prompt: str,
            *,
            model: str | None = None,
            model_settings: ModelSettings | None = None,
            usage_limits: _usage.UsageLimits | None = None,
        ):
            logger = get_run_logger()
            prompt_str = str(user_prompt)
            logger.info(f'Starting agent run with prompt: {prompt_str}')

            with self._prefect_overrides():
                # Mark that we're inside a PrefectAgent flow
                token = self._in_prefect_agent_flow.set(True)
                try:
                    result = await super(WrapperAgent, self).run(
                        user_prompt,
                        deps=None,  # type: ignore[arg-type]
                        model=model,
                        model_settings=model_settings,
                        usage_limits=usage_limits,
                    )
                    logger.info(
                        f'Agent run completed. Requests: {result.usage().requests}, Tool calls: {result.usage().tool_calls}'
                    )
                finally:
                    self._in_prefect_agent_flow.reset(token)
                logger.info('Result: %s', result.output)

        deployment_name = name or served_run.name
        maybe_coro = runner.add_flow(
            served_run,
            name=deployment_name,
            interval=interval,
            cron=cron,
            rrule=rrule,
            paused=paused,
            triggers=triggers,
            parameters=parameters,
            description=description,
            tags=tags,
            version=version,
            enforce_parameter_schema=enforce_parameter_schema,
            entrypoint_type=EntrypointType.MODULE_PATH,
        )
        if inspect.isawaitable(maybe_coro):
            await maybe_coro

        if print_starting_message:
            from rich.console import Console, Group

            help_message_top = '[green]Your agent is ready for requests!\n[/]'

            help_message_bottom = (
                '\nTo start an agent run, run:\n[blue]\n\t$ prefect deployment run'
                f' "{served_run.name}/{deployment_name}" -p user_prompt="your prompt"\n[/]'
            )
            if prefect_ui_url := get_current_settings().ui_url:
                help_message_bottom += f'\nYou can also trigger your deployments via the Prefect UI: [blue]{prefect_ui_url}/deployments[/]\n'

            console = Console()
            console.print(Group(help_message_top, help_message_bottom), soft_wrap=True)

        await runner.start()
