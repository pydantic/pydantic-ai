from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, overload

from restate import Context, TerminalError

from pydantic_ai import models
from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.agent.abstract import AbstractAgent, EventStreamHandler, RunOutputDataT
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
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


class RestateAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """An agent that integrates with Restate framework for building resilient applications.

    This agent wraps an existing agent with Restate context capabilities, providing
    automatic retries and durable execution for all operations. By default, tool calls
    are automatically wrapped with Restate's execution model.

    Example:
       ...

       weather = restate.Service('weather')

       @weather.handler()
       async def get_weather(ctx: restate.Context, city: str):
            agent = RestateAgent(weather_agent, context=ctx)
            result = await agent.run(f'What is the weather in {city}?')
            return result.output
       ...

    For advanced scenarios, you can disable automatic tool wrapping by setting
    `disable_auto_wrapping_tools=True`. This allows direct usage of Restate context
    within your tools for features like RPC calls, timers, and multi-step operations.

    When automatic wrapping is disabled, function tools will NOT be automatically executed
    within Restate's `ctx.run()` context, giving you full control over how the
    Restate context is used within your tool implementations.
    But model calls, and MCP tool calls will still be automatically wrapped.

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


       weather = restate.Service('weather')

       @weather.handler()
       async def get_weather(ctx: restate.Context, city: str):
            agent = RestateAgent(weather_agent, context=ctx)
            result = await agent.run(f'What is the weather in {city}?', deps=WeatherDeps(restate_context=ctx, ...))
            return result.output
       ...

    """

    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        restate_context: Context,
        *,
        disable_auto_wrapping_tools: bool = False,
    ):
        super().__init__(wrapped)
        if not isinstance(wrapped.model, Model):
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        self._model = RestateModelWrapper(wrapped.model, restate_context, max_attempts=3)

        def set_context(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            """Set the Restate context for the toolset, wrapping tools if needed."""
            if isinstance(toolset, FunctionToolset) and not disable_auto_wrapping_tools:
                return RestateContextRunToolSet(toolset, restate_context)
            try:
                from pydantic_ai.mcp import MCPServer

                from ._toolset import RestateMCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    return RestateMCPServer(toolset, restate_context)

            return toolset

        self._toolsets = [toolset.visit_and_replace(set_context) for toolset in wrapped.toolsets]

    @contextmanager
    def _restate_overrides(self) -> Iterator[None]:
        with (
            super().override(model=self._model, toolsets=self._toolsets, tools=[]),
            self.sequential_tool_calls(),
        ):
            yield

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
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
            builtin_tools: Optional additional builtin tools for this run.
            event_stream_handler: Optional event stream handler to use for this run.

        Returns:
            The result of the run.
        """
        if model is not None:
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        with self._restate_overrides():
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
