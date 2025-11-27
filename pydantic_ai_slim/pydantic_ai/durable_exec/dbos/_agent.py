from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import Any, overload

from dbos import DBOS, DBOSConfiguredInstance
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
from pydantic_ai.agent.abstract import Instructions, RunOutputDataT
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolFuncEither,
)

from ._model import DBOSModel
from ._utils import StepConfig


@DBOS.dbos_class()
class DBOSAgent(WrapperAgent[AgentDepsT, OutputDataT], DBOSConfiguredInstance):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        *,
        name: str | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        mcp_step_config: StepConfig | None = None,
        model_step_config: StepConfig | None = None,
    ):
        """Wrap an agent to enable it with DBOS durable workflows, by automatically offloading model requests, tool calls, and MCP server communication to DBOS steps.

        After wrapping, the original agent can still be used as normal outside of the DBOS workflow.

        Args:
            wrapped: The agent to wrap.
            name: Optional unique agent name to use as the DBOS configured instance name. If not provided, the agent's `name` will be used.
            event_stream_handler: Optional event stream handler to use instead of the one set on the wrapped agent.
            mcp_step_config: The base DBOS step config to use for MCP server steps. If no config is provided, use the default settings of DBOS.
            model_step_config: The DBOS step config to use for model request steps. If no config is provided, use the default settings of DBOS.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._event_stream_handler = event_stream_handler
        if self._name is None:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
            )

        # Merge the config with the default DBOS config
        self._mcp_step_config = mcp_step_config or {}
        self._model_step_config = model_step_config or {}

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.'
            )

        dbos_model = DBOSModel(
            wrapped.model,
            step_name_prefix=self._name,
            step_config=self._model_step_config,
            event_stream_handler=self.event_stream_handler,
        )
        self._model = dbos_model

        dbosagent_name = self._name

        def dbosify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            # Replace MCPServer with DBOSMCPServer
            try:
                from pydantic_ai.mcp import MCPServer

                from ._mcp_server import DBOSMCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    return DBOSMCPServer(
                        wrapped=toolset,
                        step_name_prefix=dbosagent_name,
                        step_config=self._mcp_step_config,
                    )

            return toolset

        dbos_toolsets = [toolset.visit_and_replace(dbosify_toolset) for toolset in wrapped.toolsets]
        self._toolsets = dbos_toolsets
        DBOSConfiguredInstance.__init__(self, self._name)

        # Wrap the `run` method in a DBOS workflow
        @DBOS.workflow(name=f'{self._name}.run')
        async def wrapped_run_workflow(
            user_prompt: str | Sequence[_messages.UserContent] | None = None,
            *,
            output_type: OutputSpec[RunOutputDataT] | None = None,
            message_history: Sequence[_messages.ModelMessage] | None = None,
            deferred_tool_results: DeferredToolResults | None = None,
            model: models.Model | models.KnownModelName | str | None = None,
            instructions: Instructions[AgentDepsT] = None,
            deps: AgentDepsT,
            model_settings: ModelSettings | None = None,
            usage_limits: _usage.UsageLimits | None = None,
            usage: _usage.RunUsage | None = None,
            infer_name: bool = True,
            toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
            builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
            event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
            **_deprecated_kwargs: Never,
        ) -> AgentRunResult[Any]:
            with self._dbos_overrides():
                return await super(WrapperAgent, self).run(
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
                    infer_name=infer_name,
                    toolsets=toolsets,
                    builtin_tools=builtin_tools,
                    event_stream_handler=event_stream_handler,
                    **_deprecated_kwargs,
                )

        self.dbos_wrapped_run_workflow = wrapped_run_workflow

        # Wrap the `run_sync` method in a DBOS workflow
        @DBOS.workflow(name=f'{self._name}.run_sync')
        def wrapped_run_sync_workflow(
            user_prompt: str | Sequence[_messages.UserContent] | None = None,
            *,
            output_type: OutputSpec[RunOutputDataT] | None = None,
            message_history: Sequence[_messages.ModelMessage] | None = None,
            deferred_tool_results: DeferredToolResults | None = None,
            model: models.Model | models.KnownModelName | str | None = None,
            deps: AgentDepsT,
            model_settings: ModelSettings | None = None,
            instructions: Instructions[AgentDepsT] = None,
            usage_limits: _usage.UsageLimits | None = None,
            usage: _usage.RunUsage | None = None,
            infer_name: bool = True,
            toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
            builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
            event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
            **_deprecated_kwargs: Never,
        ) -> AgentRunResult[Any]:
            with self._dbos_overrides():
                return super(DBOSAgent, self).run_sync(
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
                    infer_name=infer_name,
                    toolsets=toolsets,
                    builtin_tools=builtin_tools,
                    event_stream_handler=event_stream_handler,
                    **_deprecated_kwargs,
                )

        self.dbos_wrapped_run_sync_workflow = wrapped_run_sync_workflow

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
        elif DBOS.workflow_id is not None and DBOS.step_id is None:
            # Special case if it's in a DBOS workflow but not a step, we need to iterate through all events and call the handler.
            return self._call_event_stream_handler_in_workflow
        else:
            return handler

    async def _call_event_stream_handler_in_workflow(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[_messages.AgentStreamEvent]
    ) -> None:
        handler = self._event_stream_handler or super().event_stream_handler
        assert handler is not None

        async def streamed_response(event: _messages.AgentStreamEvent):
            yield event

        async for event in stream:
            await handler(ctx, streamed_response(event))

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._dbos_overrides():
            return super().toolsets

    @contextmanager
    def _dbos_overrides(self) -> Iterator[None]:
        # Override with DBOSModel and DBOSMCPServer in the toolsets.
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
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        return await self.dbos_wrapped_run_workflow(
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
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            event_stream_handler=event_stream_handler,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        return self.dbos_wrapped_run_sync_workflow(
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        deps: AgentDepsT = None,
        instructions: Instructions[AgentDepsT] = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run. It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.

        Returns:
            The result of the run.
        """
        if DBOS.workflow_id is not None and DBOS.step_id is None:
            raise UserError(
                '`agent.run_stream()` cannot be used inside a DBOS workflow. '
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
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            event_stream_handler=event_stream_handler,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            An async iterable of stream events `AgentStreamEvent` and finally a `AgentRunResultEvent` with the final
            run result.
        """
        raise UserError(
            '`agent.run_stream_events()` cannot be used with DBOS. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
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
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools for this run.

        Returns:
            The result of the run.
        """
        if model is not None and not isinstance(model, DBOSModel):
            raise UserError(
                'Non-DBOS model cannot be set at agent run time inside a DBOS workflow, it must be set at agent creation time.'
            )

        with self._dbos_overrides():
            async with super().iter(
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
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
        if _utils.is_set(model) and not isinstance(model, (DBOSModel)):
            raise UserError(
                'Non-DBOS model cannot be contextually overridden inside a DBOS workflow, it must be set at agent creation time.'
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
