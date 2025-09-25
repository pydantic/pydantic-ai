from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Generic, Never, overload

from restate import Context, TerminalError

from pydantic_ai import models
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent.abstract import AbstractAgent, EventStreamHandler, RunOutputDataT
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ._model import RestateModelWrapper
from ._toolset import RestateContextRunToolSet


class RestateAgentProvider(Generic[AgentDepsT, OutputDataT]):
    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT], *, max_attempts: int = 3):
        if not isinstance(wrapped.model, Model):
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        # here we collect all the configuration that will be passed to the RestateAgent
        # the actual context will be provided at runtime.
        self.wrapped = wrapped
        self.model = wrapped.model
        self.max_attempts = max_attempts

    def create_agent(self, context: Context) -> AbstractAgent[AgentDepsT, OutputDataT]:
        """Create an agent instance with the given Restate context.

        Use this method to create an agent that is tied to a specific Restate context.
        With this agent, all operations will be executed within the provided context,
        enabling features like retries and durable steps.
        Note that the agent will automatically wrap tool calls with restate's `ctx.run()`.

        Example:
           ...
           agent_provider = RestateAgentProvider(weather_agent)

           weather = restate.Service('weather')

           @weather.handler()
           async def get_weather(ctx: restate.Context, city: str):
                agent = agent_provider.create_agent_from_context(ctx)
                result = await agent.run(f'What is the weather in {city}?')
                return result.output
           ...

        Args:
            context: The Restate context to use for the agent.
            auto_wrap_tool_calls: Whether to automatically wrap tool calls with restate's ctx.run() (durable step), True by default.

        Returns:
            A RestateAgent instance that uses the provided context for its operations.
        """

        def get_context(_unused: AgentDepsT) -> Context:
            return context

        builder = self
        return RestateAgent(builder=builder, get_context=get_context, auto_wrap_tools=True)

    def create_agent_with_advanced_tools(
        self, get_context: Callable[[AgentDepsT], Context]
    ) -> AbstractAgent[AgentDepsT, OutputDataT]:
        """Create an agent instance that is able to obtain Restate context from its dependencies.

        Use this method, if you are planning to use restate's context inside the tools (for rpc, timers, multi step tools etc.)
        To obtain a context inside a tool you can add a dependency that has a `restate_context` attribute, and provide a `get_context` extractor
        function that extracts the context from the dependencies at runtime.

        Note: that the agent will NOT automatically wrap tool calls with restate's `ctx.run()`
        since the tools may use the context in different ways.

        Example:
           ...

           @dataclass
           WeatherDeps:
                ...
                restate_context: Context

           weather_agent = Agent(..., deps_type=WeatherDeps, ...)

           @weather_agent.tool
           async def get_lat_lng(ctx: RunContext[WeatherDeps], location_description: str) -> LatLng:
                restate_context = ctx.deps.restate_context
                lat = await restate_context.run(...) # <---- note the direct usage of the restate context
                lng = await restate_context.run(...)
                return LatLng(lat, lng)

           agent = RestateAgentProvider(weather_agent).create_agent_from_deps(lambda deps: deps.restate_context)

           weather = restate.Service('weather')

           @weather.handler()
           async def get_weather(ctx: restate.Context, city: str):
                result = await agent.run(f'What is the weather in {city}?', deps=WeatherDeps(restate_context=ctx, ...))
                return result.output
           ...

        Args:
            get_context: A callable that extracts the Restate context from the agent's dependencies.

        Returns:
            A RestateAgent instance that uses the provided context extractor to obtain the Restate context at runtime.

        """
        builder = self
        return RestateAgent(builder=builder, get_context=get_context, auto_wrap_tools=False)


class RestateAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """An agent that integrates with the Restate framework for resilient applications."""

    def __init__(
        self,
        builder: RestateAgentProvider[AgentDepsT, OutputDataT],
        get_context: Callable[[AgentDepsT], Context],
        auto_wrap_tools: bool,
    ):
        super().__init__(builder.wrapped)
        self._builder = builder
        self._get_context = get_context
        self._auto_wrap_tools = auto_wrap_tools

    @contextmanager
    def _restate_overrides(self, context: Context) -> Iterator[None]:
        model = RestateModelWrapper(self._builder.model, context, max_attempts=self._builder.max_attempts)

        def set_context(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            """Set the Restate context for the toolset, wrapping tools if needed."""
            if isinstance(toolset, FunctionToolset) and self._auto_wrap_tools:
                return RestateContextRunToolSet(toolset, context)
            try:
                from pydantic_ai.mcp import MCPServer

                from ._toolset import RestateMCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    return RestateMCPServer(toolset, context)

            return toolset

        toolsets = [toolset.visit_and_replace(set_context) for toolset in self._builder.wrapped.toolsets]

        with (
            super().override(model=model, toolsets=toolsets, tools=[]),
            self.sequential_tool_calls(),
        ):
            yield

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
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
        if model is not None:
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        context = self._get_context(deps)
        with self._restate_overrides(context):
            return await super(WrapperAgent, self).run(
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
                event_stream_handler=event_stream_handler,
            )
