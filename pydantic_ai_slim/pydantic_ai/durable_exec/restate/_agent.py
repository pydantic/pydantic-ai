from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from typing import Any, overload

from pydantic.errors import PydanticUserError
from typing_extensions import Never

from pydantic_ai import _utils, models
from pydantic_ai.agent.abstract import AbstractAgent, AgentMetadata, EventStreamHandler, Instructions, RunOutputDataT
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, UserContent
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.run import AgentRun, AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, DeferredToolResults, RunContext, Tool, ToolFuncEither
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ._dynamic_toolset import RestateDynamicToolset
from ._model import RestateModelWrapper
from ._restate_types import Context, TerminalError
from ._toolset import RestateContextRunToolset


class RestateAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    """Durable wrapper for running an agent inside a Restate handler.

    This ensures I/O-bound operations (model calls, MCP/FastMCP tool calls, and tool execution)
    run inside `restate.Context.run_typed(...)` so Restate can retry/deduplicate them durably.

    Notes:
    - `run_stream()` and `run_stream_events()` are not supported in Restate handlers. Use an
      `event_stream_handler` and call `run()` instead.
    - If `disable_auto_wrapping_tools=True`, function tools are executed outside `run_typed` so tools
      can use the Restate context directly.
    - See `docs/durable_execution/restate.md` for usage examples.
    """

    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        restate_context: Context,
        *,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        disable_auto_wrapping_tools: bool = False,
        max_attempts: int | None = 3,
    ):
        """Wrap an agent to run durably inside a Restate handler.

        Args:
            wrapped: The agent to wrap.
            restate_context: The Restate handler context (`restate.Context`).
            event_stream_handler: Optional event stream handler. Must be set at creation time, not at `run()` time.
            disable_auto_wrapping_tools: If `True`, function tools are executed outside `ctx.run_typed()`, allowing
                tools to use the Restate context directly. Model calls and MCP/FastMCP tool calls are still wrapped.
            max_attempts: Maximum retry attempts for durable model calls. Tool calls use Restate defaults. `None`
                uses the Restate default for model calls.
        """
        super().__init__(wrapped)
        if isinstance(wrapped, RestateAgent):
            raise TerminalError('A `RestateAgent` cannot wrap another `RestateAgent`. Wrap the base `Agent` once.')

        if not isinstance(wrapped.model, Model):
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )

        self._restate_context = restate_context
        self._event_stream_handler = event_stream_handler
        self._disable_auto_wrapping_tools = disable_auto_wrapping_tools

        model_event_stream_handler = event_stream_handler or super().event_stream_handler
        self._restate_model = RestateModelWrapper(
            wrapped.model,
            restate_context,
            event_stream_handler=model_event_stream_handler,
            max_attempts=max_attempts,
        )

        def set_context(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            """Set the Restate context for the toolset, wrapping tools if needed."""
            if isinstance(toolset, FunctionToolset) and not disable_auto_wrapping_tools:
                return RestateContextRunToolset(toolset, restate_context)

            # Dynamic toolsets may resolve to I/O toolsets; ensure tool discovery and calls are durable.
            if isinstance(toolset, DynamicToolset):
                return RestateDynamicToolset(
                    toolset,
                    restate_context,
                    disable_auto_wrapping_tools=disable_auto_wrapping_tools,
                )

            # FastMCP toolsets require I/O, so they must be wrapped for durability.
            try:
                from pydantic_ai.toolsets.fastmcp import FastMCPToolset
            except ImportError:
                pass
            else:
                if isinstance(toolset, FastMCPToolset):
                    from ._fastmcp_toolset import RestateFastMCPToolset

                    return RestateFastMCPToolset(toolset, restate_context)
            try:
                from pydantic_ai.mcp import MCPServer
            except ImportError:
                pass
            else:
                if isinstance(toolset, MCPServer):
                    from ._mcp_server import RestateMCPServer

                    return RestateMCPServer(toolset, restate_context)

            return toolset

        self._toolsets = [toolset.visit_and_replace(set_context) for toolset in wrapped.toolsets]

    @contextmanager
    def _restate_overrides(self) -> Iterator[None]:
        with (
            # Function tools are already included in `self._toolsets` (wrapped where needed),
            # so pass `tools=[]` to avoid double-registration from the wrapped agent.
            super().override(model=self._restate_model, toolsets=self._toolsets, tools=[]),
            # Restate tool calls use `ctx.run_typed(...)`, which should not be executed concurrently within a handler.
            self.parallel_tool_call_execution_mode('sequential'),
        ):
            yield

    @property
    def restate_context(self) -> Context:
        return self._restate_context

    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        """No-op: entering wrapped toolsets outside durable steps breaks Restate guarantees."""
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        """No-op: toolset cleanup should happen inside durable steps."""
        return None

    @property
    def model(self) -> models.Model | models.KnownModelName | str | None:
        return self._restate_model

    @property
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        handler = self._event_stream_handler or super().event_stream_handler
        if handler is None:
            return None
        if self._disable_auto_wrapping_tools:
            return handler
        return self._wrapped_event_stream_handler

    async def _wrapped_event_stream_handler(
        self, ctx: RunContext[AgentDepsT], stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        handler = self._event_stream_handler or super().event_stream_handler
        # `_wrapped_event_stream_handler` is only used when a handler exists.
        if handler is None:  # pragma: no cover
            return
        async for event in stream:
            event_kind = event.event_kind

            # Wrap each event in its own durable step (similar to Temporal's per-event activity),
            # so event handler side effects are retried/deduplicated at event granularity.

            async def call_handler() -> None:
                async def single_event() -> AsyncIterator[AgentStreamEvent]:
                    yield event

                try:
                    await handler(ctx, single_event())
                except (UserError, PydanticUserError) as e:
                    raise TerminalError(str(e)) from e

            await self._restate_context.run_typed(f'handle event: {event_kind}', call_handler)

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._restate_overrides():
            return super().toolsets

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
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
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
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
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent durably inside a Restate handler.

        `model` and `event_stream_handler` must be configured when creating `RestateAgent`.
        """
        if model is not None:
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        if event_stream_handler is not None:
            raise TerminalError(
                'Event stream handler cannot be set at agent run time inside a Restate handler, it must be set at agent creation time.'
            )
        if toolsets is not None:
            raise TerminalError(
                'Toolsets cannot be set at agent run time inside a Restate handler, they must be set at agent creation time.'
            )
        with self._restate_overrides():
            return await super().run(
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
                metadata=metadata,
                infer_name=infer_name,
                builtin_tools=builtin_tools,
                event_stream_handler=self.event_stream_handler,
                **_deprecated_kwargs,
            )

    def run_sync(self, *args: Any, **kwargs: Any) -> AgentRunResult[Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise TerminalError(
                '`agent.run_sync()` cannot be used inside a restate handler. Use `await agent.run()` instead.'
            )

        return super().run_sync(*args, **kwargs)

    @asynccontextmanager
    async def run_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamedRunResult[AgentDepsT, Any]]:
        """Not supported in Restate handlers. Use `event_stream_handler` and `run()` instead."""
        raise TerminalError(
            '`agent.run_stream()` cannot be used inside a restate handler. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        )

        yield

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: Instructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        if model is not None:
            raise TerminalError(
                'An agent needs to have a `model` in order to be used with Restate, it cannot be set at agent run time.'
            )
        if toolsets is not None:
            raise TerminalError(
                'Toolsets cannot be set at agent run time inside a Restate handler, they must be set at agent creation time.'
            )
        with self._restate_overrides():
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
                metadata=metadata,
                infer_name=infer_name,
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
        if _utils.is_set(model):
            raise TerminalError(
                'Model cannot be contextually overridden inside a Restate handler, it must be set at agent creation time.'
            )
        if _utils.is_set(toolsets):
            raise TerminalError(
                'Toolsets cannot be contextually overridden inside a Restate handler, they must be set at agent creation time.'
            )
        if _utils.is_set(tools):
            raise TerminalError(
                'Tools cannot be contextually overridden inside a Restate handler, they must be set at agent creation time.'
            )

        with super().override(
            name=name,
            deps=deps,
            instructions=instructions,
        ):
            yield

    def run_stream_events(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]:
        # Match the base agent's overload-friendly pattern: return an async generator.
        async def _event_stream() -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]]:
            raise TerminalError(
                '`agent.run_stream_events()` cannot be used inside a restate handler. '
                'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

            if False:  # pragma: no cover
                yield

        return _event_stream()
