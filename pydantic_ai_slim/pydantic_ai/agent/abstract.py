from __future__ import annotations as _annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Sequence,
)
from concurrent.futures import Executor
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, cast, overload

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream
from pydantic import TypeAdapter
from typing_extensions import Self, TypedDict, TypeIs, TypeVar

from pydantic_graph import End

from .. import (
    _agent_graph,
    _instructions,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    tool_manager,
    usage as _usage,
)
from .._json_schema import JsonSchema
from .._output import types_from_output_spec
from .._template import TemplateStr
from ..capabilities import AgentCapability
from ..output import OutputDataT, OutputSpec
from ..result import AgentStream, FinalResult, StreamedRunResult
from ..run import AgentRun, AgentRunResult, AgentRunResultEvent
from ..settings import ModelSettings
from ..tool_manager import ToolManager
from ..tools import (
    AgentDepsT,
    AgentNativeTool,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolFuncEither,
)
from ..toolsets import AbstractToolset

if TYPE_CHECKING:
    from pydantic_ai.agent.spec import AgentSpec
    from pydantic_ai.capabilities import CombinedCapability


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
RunOutputDataT = TypeVar('RunOutputDataT')
"""Type variable for the result data of a run where `output_type` was customized on the run call."""

EventStreamHandler: TypeAlias = Callable[
    [RunContext[AgentDepsT], AsyncIterable[_messages.AgentStreamEvent]], Awaitable[None]
]
"""A function that receives agent [`RunContext`][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools."""

EventStreamProcessor: TypeAlias = Callable[
    [RunContext[AgentDepsT], AsyncIterable[_messages.AgentStreamEvent]],
    AsyncIterator[_messages.AgentStreamEvent],
]
"""An async generator that receives agent [`RunContext`][pydantic_ai.tools.RunContext] and an async iterable of events and yields a potentially modified stream.

Used with the [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] capability to modify, drop, or add events visible to the rest of the capability chain."""


AgentMetadata = dict[str, Any] | Callable[[RunContext[AgentDepsT]], dict[str, Any]]

AgentInstructions = _instructions.AgentInstructions
"""Type alias for agent instructions — a string, `TemplateStr`, callable, or sequence thereof."""

Instructions = AgentInstructions
"""Deprecated: use `AgentInstructions` instead."""

AgentModelSettings = ModelSettings | Callable[[RunContext[AgentDepsT]], ModelSettings]
"""Type alias for agent model settings — a static `ModelSettings` dict, or a callable receiving `RunContext` that returns one dynamically per request."""


class AgentRetries(TypedDict, total=False):
    """Per-category retry budgets for an [`Agent`][pydantic_ai.agent.Agent].

    Pass to `Agent(retries=...)` as a dict to set different budgets per category.

    `int` semantics differ by call site:

    - At `Agent(retries=N)` construction time, an `int` sets both `tools` and `output`
      to `N`.
    - At `run()` / `iter()` / `override()` time, an `int` overrides only the `output`
      budget. Tool retries cannot be overridden per run or via `override()` — passing
      `retries={'tools': ...}` at those call sites raises a `UserError`, since the tool
      manager is built once at agent construction.

    Keys:
        tools: Default number of retries for tool calls before raising an error.
        output: Maximum number of retries for output validation. On the text path
            this is a global per-run budget; on the tool path it is the default
            per-tool `max_retries` for each output tool, overridable via
            [`ToolOutput(max_retries=...)`][pydantic_ai.output.ToolOutput.max_retries].
    """

    tools: int
    output: int


_RunStreamEventsRunner: TypeAlias = Callable[[EventStreamHandler[Any]], Awaitable[AgentRunResult[Any]]]
"""Starts the background agent run with the internal event-forwarding handler and returns its result."""


class _RunStreamEventsIterator(AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]):
    """The event iterator returned by [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events].

    Lazily starts a background `run()` task on the first `__anext__()` and forwards its events over a memory
    object stream, ending with a single trailing `AgentRunResultEvent` that carries the run's result. Entering
    the context manager without iterating therefore never starts a run (#6162).

    This is a hand-written iterator class rather than an `async def` generator on purpose: generator cleanup
    runs by throwing `GeneratorExit` into the suspended frame during finalization, which on Python 3.10/3.11
    can resume the frame under a different `Context` and raise the `pydantic_ai.current_run_context` token
    error (#5132). Driving cleanup explicitly through `aclose()` keeps teardown in the caller's task and
    context.
    """

    def __init__(self, run_agent: _RunStreamEventsRunner) -> None:
        self._run_agent = run_agent
        self._receive_stream: (
            MemoryObjectReceiveStream[_messages.AgentStreamEvent | AgentRunResultEvent[Any]] | None
        ) = None
        self._task: asyncio.Task[AgentRunResult[Any]] | None = None
        # Set once the trailing `AgentRunResultEvent` has been produced, so further `__anext__()` calls stop.
        self._result_yielded = False
        self._closed = False

    def __aiter__(self) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        return self

    async def __anext__(self) -> _messages.AgentStreamEvent | AgentRunResultEvent[Any]:
        if self._closed or self._result_yielded:
            raise StopAsyncIteration

        await self._ensure_started()
        assert self._receive_stream is not None
        assert self._task is not None

        try:
            return await self._receive_stream.receive()
        except anyio.EndOfStream:
            # The run closed its send stream, so all events have been delivered: surface the run result as a
            # final event. Awaiting the task here also re-raises any error it failed with, to the consumer.
            await self._receive_stream.aclose()
            self._result_yielded = True
            result = await self._task
            return AgentRunResultEvent(result)

    async def aclose(self) -> None:
        """Cancel the background run (if started) and close the receive stream, idempotently."""
        if self._closed:
            return

        self._closed = True
        # Cancel the run first so it tears down via its own cancellation, unblocking a run that's
        # parked pushing an event into the zero-buffer stream. But if the run *absorbs* that
        # cancellation (e.g. a durable step under Temporal's cooperative cancellation) it can resume
        # and block again on `send`, so close the receive end before draining: the blocked `send` then
        # fails with `BrokenResourceError` and the drain can complete instead of deadlocking. A run
        # that unwound normally is unaffected. If iteration was never started, `_task` is `None`.
        if self._task is not None:
            self._task.cancel()
        if self._receive_stream is not None:
            await self._receive_stream.aclose()
        if self._task is not None:
            await _utils.cancel_and_drain(self._task)

    async def _ensure_started(self) -> None:
        if self._task is not None:
            return

        # Zero-buffer stream: the run blocks on `send` until this iterator pulls, giving natural backpressure
        # and keeping the run no more than one event ahead of the consumer.
        send_stream, receive_stream = anyio.create_memory_object_stream[
            _messages.AgentStreamEvent | AgentRunResultEvent[Any]
        ]()
        self._receive_stream = receive_stream

        async def event_stream_handler(_: RunContext[Any], events: AsyncIterable[_messages.AgentStreamEvent]) -> None:
            async for event in events:
                await send_stream.send(event)

        async def run_agent() -> AgentRunResult[Any]:
            # Closing the send stream on exit is what surfaces `EndOfStream` to the consumer once the run ends.
            async with send_stream:
                return await self._run_agent(event_stream_handler)

        self._task = asyncio.create_task(run_agent())


class _RunStreamEventsContext(
    AbstractAsyncContextManager[AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]]
):
    """The async context manager returned by [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events].

    Hands out a single `_RunStreamEventsIterator` on entry and closes it on exit, so an early `break` out of
    the event loop still cancels and drains the background run.
    """

    def __init__(self, run_agent: _RunStreamEventsRunner) -> None:
        self._run_agent = run_agent
        self._iterator: _RunStreamEventsIterator | None = None

    async def __aenter__(self) -> AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]:
        # Single-entry: re-entering would orphan a first iterator that had already started (and leak its
        # background task), so fail loudly instead of silently. `__aexit__` still cleans up the one live
        # iterator.
        if self._iterator is not None:
            raise RuntimeError('`run_stream_events()` context manager cannot be entered more than once')
        self._iterator = _RunStreamEventsIterator(self._run_agent)
        return self._iterator

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._iterator is not None:
            await self._iterator.aclose()


class AbstractAgent(Generic[AgentDepsT, OutputDataT], ABC):
    """Abstract superclass for [`Agent`][pydantic_ai.agent.Agent], [`WrapperAgent`][pydantic_ai.agent.WrapperAgent], and your own custom agent implementations."""

    @property
    @abstractmethod
    def model(self) -> models.Model | models.KnownModelName | str | None:
        """The default model configured for this agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str | None:
        """The name of the agent, used for logging.

        If `None`, we try to infer the agent name from the call frame when the agent is first run.
        """
        raise NotImplementedError

    @name.setter
    @abstractmethod
    def name(self, value: str | None) -> None:
        """Set the name of the agent, used for logging."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str | None:
        """A human-readable description of the agent."""
        raise NotImplementedError

    @description.setter
    @abstractmethod
    def description(self, value: TemplateStr[AgentDepsT] | str | None) -> None:
        """Set the description of the agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def deps_type(self) -> type:
        """The type of dependencies used by the agent."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_type(self) -> OutputSpec[OutputDataT]:
        """The type of data output by agent runs, used to validate the data returned by the model, defaults to `str`."""
        raise NotImplementedError

    @property
    @abstractmethod
    def event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        """Optional handler for events from the model's streaming response and the agent's execution of tools."""
        raise NotImplementedError

    @property
    def root_capability(self) -> CombinedCapability[AgentDepsT]:
        """The root capability of the agent, containing all registered capabilities."""
        raise NotImplementedError

    @property
    @abstractmethod
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        """All toolsets registered on the agent.

        Output tools are not included.
        """
        raise NotImplementedError

    async def system_prompt_parts(
        self,
        *,
        deps: AgentDepsT = None,
        model: models.Model | models.KnownModelName | str | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        prompt: str | Sequence[_messages.UserContent] | None = None,
        usage: _usage.RunUsage | None = None,
        model_settings: ModelSettings | None = None,
    ) -> list[_messages.SystemPromptPart]:
        """Resolve the agent's configured system prompts into `SystemPromptPart`s.

        Returns a list suitable for prepending to a `ModelRequest`. Static strings and
        runners decorated with [`@agent.system_prompt`][pydantic_ai.Agent.system_prompt]
        are evaluated using a minimal `RunContext` built from the provided kwargs — useful
        when reconstructing a `message_history` that should carry the agent's configured
        system prompt (e.g. in UI adapters or after history compaction).

        Dynamic runners produce parts with `dynamic_ref` set so they can continue to be
        re-evaluated by the standard agent graph path on subsequent turns.

        Args:
            deps: Optional dependencies for dynamic system prompt functions.
            model: Optional model to use for `RunContext.model`. Falls back to the
                agent's configured model; required only if the agent has no model set.
            message_history: Optional message history to expose as `RunContext.messages`.
            prompt: Optional user prompt to expose as `RunContext.prompt`.
            usage: Optional usage to expose as `RunContext.usage`.
            model_settings: Optional settings to expose as `RunContext.model_settings`.
        """
        return []  # pragma: no cover — concrete subclasses override this

    def output_json_schema(self, output_type: OutputSpec[OutputDataT | RunOutputDataT] | None = None) -> JsonSchema:
        """The output return JSON schema."""
        if output_type is None:
            output_type = self.output_type

        return_types = types_from_output_spec(output_spec=output_type)

        json_schemas: list[JsonSchema] = []
        for return_type in return_types:
            json_schema = TypeAdapter(return_type).json_schema(mode='serialization')
            if json_schema not in json_schemas:
                json_schemas.append(json_schema)

        if len(json_schemas) == 1:
            return json_schemas[0]
        else:
            json_schemas, all_defs = _utils.merge_json_schema_defs(json_schemas)
            json_schema: JsonSchema = {'anyOf': json_schemas}
            if all_defs:
                json_schema['$defs'] = all_defs
            return json_schema

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')

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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        event_stream_handler = event_stream_handler or self.event_stream_handler

        async with self.iter(
            user_prompt=user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            instructions=instructions,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            retries=retries,
            toolsets=toolsets,
            capabilities=capabilities,
            spec=spec,
        ) as agent_run:
            # Drive via next() so capability hooks fire for each node.
            # When event_stream_handler is set or a capability overrides wrap_run_event_stream,
            # streaming must happen AFTER before_node_run (which may replace the node) and
            # INSIDE wrap_node_run. We achieve this by passing a custom step function that
            # streams before advancing the graph.
            _stream_step: (
                Callable[
                    [_agent_graph.AgentNode[AgentDepsT, Any]],
                    Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
                ]
                | None
            ) = None
            _needs_streaming = (
                event_stream_handler is not None or agent_run.ctx.deps.root_capability.has_wrap_run_event_stream
            )
            if _needs_streaming:
                _handler = event_stream_handler

                async def _stream_and_advance(
                    n: _agent_graph.AgentNode[AgentDepsT, Any],
                ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
                    if self.is_model_request_node(n) or self.is_call_tools_node(n):
                        async with n.stream(agent_run.ctx) as stream:
                            run_ctx = _agent_graph.build_run_context(agent_run.ctx)
                            wrapped = agent_run.ctx.deps.root_capability.wrap_run_event_stream(run_ctx, stream=stream)
                            if _handler is not None:
                                await _handler(run_ctx, wrapped)
                            # If the handler returns normally, drain whatever it left unconsumed so the
                            # node can finish through any stream wrappers. Cancellation paths interrupt
                            # the handler and do not reach this drain.
                            async for _ in wrapped:
                                pass
                    return await agent_run._advance_graph(n)  # pyright: ignore[reportPrivateUsage]

                _stream_step = _stream_and_advance

            node = agent_run.next_node
            while not isinstance(node, End):
                # Handle wrap_run short-circuit: result is already available, skip the graph.
                if agent_run.result is not None:
                    break
                if _stream_step is not None:
                    node = await agent_run._run_node_with_hooks(node, _stream_step)  # pyright: ignore[reportPrivateUsage]
                else:
                    node = await agent_run.next(node)  # pyright: ignore[reportArgumentType]

        assert agent_run.result is not None, 'The graph run did not finish properly'
        return agent_run.result

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')

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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        return _utils.run_until_complete(
            self.run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                retries=retries,
                infer_name=False,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                capabilities=capabilities,
                spec=spec,
            )
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, OutputDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    async def run_stream(  # noqa: C901
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncGenerator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async streaming mode.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph until the model produces output matching the `output_type`, for example text or structured data.
        At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
        and -- once this output has completed streaming -- get the complete output, message history, and usage.

        As this method will consider the first output matching the `output_type` to be the final output,
        it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
        If you want to always run the agent graph to completion and stream events and output at the same time,
        use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')

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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
                It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
                Note that it does _not_ receive any events after the final result is found.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        event_stream_handler = event_stream_handler or self.event_stream_handler

        yielded = False
        async with self.iter(
            user_prompt,
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            deps=deps,
            instructions=instructions,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            retries=retries,
            infer_name=False,
            toolsets=toolsets,
            capabilities=capabilities,
            spec=spec,
        ) as agent_run:
            # Handle wrap_run short-circuit: result is already available
            if agent_run.result is not None:
                graph_ctx = agent_run.ctx
                yield StreamedRunResult(
                    graph_ctx.state.message_history,
                    graph_ctx.deps.new_message_index,
                    run_result=agent_run.result,
                )
                yielded = True

            first_node = agent_run.next_node  # start with the first node
            assert isinstance(first_node, _agent_graph.UserPromptNode)  # the first node should be a user prompt node
            node: _agent_graph.AgentNode[Any, Any] = first_node
            while not yielded:
                graph_ctx = agent_run.ctx
                # Fire before_node_run BEFORE streaming so that node replacement
                # happens before any model call, avoiding double execution.
                run_ctx = _agent_graph.build_run_context(graph_ctx)
                cap = graph_ctx.deps.root_capability
                node = await cap.before_node_run(run_ctx, node=node)

                if self.is_model_request_node(node):
                    async with node.stream(graph_ctx) as stream:
                        final_result_event = None

                        async def stream_to_final(
                            stream: AgentStream,
                        ) -> AsyncIterator[_messages.ModelResponseStreamEvent]:
                            nonlocal final_result_event
                            async for event in stream:
                                yield event
                                if isinstance(event, _messages.FinalResultEvent):
                                    final_result_event = event
                                    break

                        wrapped = cap.wrap_run_event_stream(run_ctx, stream=stream_to_final(stream))
                        if event_stream_handler is not None:
                            await event_stream_handler(run_ctx, wrapped)
                        # Drain after the handler (same as the `run()` path) so the response is fully
                        # built and any `wrap_run_event_stream` wrapper finalizes; cancellation/`break`
                        # interrupt the handler and don't reach here.
                        async for _ in wrapped:
                            pass

                        if final_result_event is not None:
                            final_result = FinalResult(
                                None, final_result_event.tool_name, final_result_event.tool_call_id
                            )
                            yielded = True

                            messages = graph_ctx.state.message_history.copy()

                            async def on_complete() -> None:
                                """Called when the stream has completed.

                                The model response will have been added to messages by now
                                by `StreamedRunResult._marked_completed`.
                                """
                                nonlocal final_result
                                final_result = FinalResult(
                                    await stream.get_output(), final_result.tool_name, final_result.tool_call_id
                                )

                                # When we get here, the `ModelRequestNode` has completed streaming after the final result was found.
                                # When running an agent with `agent.run`, we'd then move to `CallToolsNode` to execute the tool calls and
                                # find the final result.
                                # We also want to execute tool calls (in case `agent.end_strategy` is not `'early'`) here, but
                                # we don't want to run the `CallToolsNode` logic to determine the final output, as it would be
                                # wasteful and could produce a different result (e.g. when text output is followed by tool calls).
                                # So we call `process_tool_calls` directly and then end the run with the found final result.

                                parts: list[_messages.ModelRequestPart] = []
                                async for _event in _agent_graph.process_tool_calls(
                                    tool_manager=graph_ctx.deps.tool_manager,
                                    tool_calls=stream.response.tool_calls,
                                    tool_call_results=None,
                                    tool_call_metadata=None,
                                    final_result=final_result,
                                    ctx=graph_ctx,
                                    output_parts=parts,
                                ):
                                    pass

                                # To allow this message history to be used in a future run without dangling tool calls,
                                # append a new ModelRequest using the tool returns and retries
                                if parts:
                                    messages.append(
                                        _messages.ModelRequest(
                                            parts,
                                            run_id=graph_ctx.state.run_id,
                                            conversation_id=graph_ctx.state.conversation_id,
                                            timestamp=_utils.now_utc(),
                                        )
                                    )

                                await agent_run.next(_agent_graph.SetFinalResult(final_result))

                            yield StreamedRunResult(
                                messages,
                                graph_ctx.deps.new_message_index,
                                stream,
                                on_complete,
                            )
                            # Note: wrap_node_run/after_node_run are intentionally skipped here.
                            # before_node_run fired above; on_complete() later calls
                            # agent_run.next(SetFinalResult(...)) which fires the full lifecycle
                            # for SetFinalResult, but not for this ModelRequestNode.
                            break
                elif self.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        wrapped = cap.wrap_run_event_stream(run_ctx, stream=stream)
                        if event_stream_handler is not None:
                            await event_stream_handler(run_ctx, wrapped)
                        # Drain `wrapped` after the handler, same as the `ModelRequestNode` branch above:
                        # `CallToolsNode.stream()` self-drains its own events, but `wrapped` is a separate
                        # `wrap_run_event_stream` layer that must finalize here too, to match the `run()` path.
                        async for _ in wrapped:
                            pass

                # Advance graph with remaining hooks (before_node_run already fired above).
                # Rebuild run_ctx after streaming so hooks see post-streaming state (e.g. run_step).
                run_ctx = _agent_graph.build_run_context(graph_ctx)
                next_node = await agent_run._wrap_and_advance(run_ctx, node, agent_run._advance_graph)  # pyright: ignore[reportPrivateUsage]
                if isinstance(next_node, End) and agent_run.result is not None:
                    # A final output could have been produced by the CallToolsNode rather than the ModelRequestNode,
                    # if a tool function raised CallDeferred or ApprovalRequired.
                    # In this case there's no response to stream, but we still let the user access the output etc as normal.
                    yield StreamedRunResult(
                        graph_ctx.state.message_history,
                        graph_ctx.deps.new_message_index,
                        run_result=agent_run.result,
                    )
                    yielded = True
                    break
                if not isinstance(next_node, _agent_graph.AgentNode):
                    raise exceptions.AgentRunError(  # pragma: lax no cover
                        'Should have produced a StreamedRunResult before getting here'
                    )
                node = cast(_agent_graph.AgentNode[Any, Any], next_node)

        if not yielded:
            raise exceptions.AgentRunError('Agent run finished without producing a final result')  # pragma: no cover

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, OutputDataT]: ...

    @overload
    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, RunOutputDataT]: ...

    def run_stream_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> result.StreamedRunResultSync[AgentDepsT, Any]:
        """Run the agent with a user prompt in sync streaming mode.

        This is a convenience method that wraps [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream],
        running all of the agent's async work on a dedicated event loop thread.
        You therefore can't use this method inside async code or if there's an active event loop.

        The returned [`StreamedRunResultSync`][pydantic_ai.result.StreamedRunResultSync] is a synchronous
        context manager and should be used with a `with` block so the stream and event loop thread are
        cleaned up when you're done.

        This method builds an internal agent graph (using system prompts, tools and output schemas) and then
        runs the graph until the model produces output matching the `output_type`, for example text or structured data.
        At this point, a streaming run result object is yielded from which you can stream the output as it comes in,
        and -- once this output has completed streaming -- get the complete output, message history, and usage.

        As this method will consider the first output matching the `output_type` to be the final output,
        it will stop running the agent graph and will not execute any tool calls made by the model after this "final" output.
        If you want to always run the agent graph to completion and stream events and output at the same time,
        use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an `event_stream_handler` or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2')

        def main():
            with agent.run_stream_sync('What is the capital of the UK?') as response:
                print(response.get_output())
                #> The capital of the UK is London.
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            event_stream_handler: Optional handler for events from the model's streaming response and the agent's execution of tools to use for this run.
                It will receive all the events up until the final result is found, which you can then read or stream from inside the context manager.
                Note that it does _not_ receive any events after the final result is found.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        return result.StreamedRunResultSync(
            self.run_stream(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                retries=retries,
                infer_name=False,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                capabilities=capabilities,
                spec=spec,
            )
        )

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[OutputDataT]]]: ...

    @overload
    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[
        AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[RunOutputDataT]]
    ]: ...

    def run_stream_events(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AsyncIterator[_messages.AgentStreamEvent | AgentRunResultEvent[Any]]]:
        """Run the agent with a user prompt in async mode and stream events from the run.

        This is a convenience method that wraps [`self.run`][pydantic_ai.agent.AbstractAgent.run] and
        uses the `event_stream_handler` kwarg to get a stream of events from the run.

        The background run starts on the first iteration of the event stream, not on entering the
        context manager, so entering and exiting without iterating never calls the model.

        Must be used as an async context manager so the background run task is deterministically
        cleaned up when the consumer stops iterating early.

        Example:
        ```python
        from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

        agent = Agent('openai:gpt-5.2')

        async def main():
            collected: list[AgentStreamEvent | AgentRunResultEvent] = []
            async with agent.run_stream_events('What is the capital of France?') as events:
                async for event in events:
                    collected.append(event)
            print(collected)
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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            An async context manager that yields an async iterator over `AgentStreamEvent`s ending with a final
            `AgentRunResultEvent` carrying the run result.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())

        async def run_agent(event_stream_handler: EventStreamHandler[AgentDepsT]) -> AgentRunResult[Any]:
            return await self.run(
                user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                retries=retries,
                infer_name=False,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                capabilities=capabilities,
                spec=spec,
            )

        return _RunStreamEventsContext(run_agent)

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, OutputDataT]]: ...

    @overload
    def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AbstractAsyncContextManager[AgentRun[AgentDepsT, RunOutputDataT]]: ...

    @asynccontextmanager
    @abstractmethod
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: Sequence[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: AgentModelSettings[AgentDepsT] | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        retries: int | AgentRetries | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> AsyncGenerator[AgentRun[AgentDepsT, Any]]:
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

        agent = Agent('openai:gpt-5.2')

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
                        conversation_id='...',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.')],
                        usage=RequestUsage(input_tokens=56, output_tokens=7),
                        model_name='gpt-5.2',
                        timestamp=datetime.datetime(...),
                        run_id='...',
                        conversation_id='...',
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
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request, or a callable
                that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns settings.
                Callables are called before each model request, allowing dynamic per-step settings.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            retries: Override the agent-level retry budgets for this run. Pass an `int` to override the
                output-validation budget (`AgentRetries(output=...)` equivalent), or an
                [`AgentRetries`][pydantic_ai.AgentRetries] dict for finer control. Tool retries cannot
                be overridden per run. See
                [`Agent.__init__`][pydantic_ai.agent.Agent.__init__] for semantics of the two enforcement paths.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://ai.pydantic.dev/capabilities/) for this run, merged with the agent's configured capabilities.
            spec: Optional agent spec to apply for this run. At run time, spec values are additive.

        Returns:
            The result of the run.
        """
        raise NotImplementedError
        yield

    @contextmanager
    @abstractmethod
    def override(
        self,
        *,
        name: str | _utils.Unset = _utils.UNSET,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | _utils.Unset = _utils.UNSET,
        native_tools: Sequence[AgentNativeTool[AgentDepsT]] | _utils.Unset = _utils.UNSET,
        instructions: _instructions.AgentInstructions[AgentDepsT] | _utils.Unset = _utils.UNSET,
        model_settings: AgentModelSettings[AgentDepsT] | _utils.Unset = _utils.UNSET,
        retries: int | AgentRetries | _utils.Unset = _utils.UNSET,
        spec: dict[str, Any] | AgentSpec | None = None,
    ) -> Generator[None]:
        """Context manager to temporarily override agent configuration.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            name: The name to use instead of the name passed to the agent constructor and agent run.
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
            toolsets: The toolsets to use instead of the toolsets passed to the agent constructor and agent run.
            tools: The tools to use instead of the tools registered with the agent.
            native_tools: The native tools to use instead of the agent's configured native tools.
            instructions: The instructions to use instead of the instructions registered with the agent.
            model_settings: The model settings to use instead of the model settings passed to the agent constructor.
                When set, any per-run `model_settings` argument is ignored.
            retries: The retry budgets to use instead of the agent-level configuration. Pass an `int` to
                override the output-validation budget, or an [`AgentRetries`][pydantic_ai.AgentRetries]
                dict for finer control. When set, any per-run `retries` argument is ignored. Tool retries
                cannot be overridden via `override()`.
            spec: Optional agent spec providing defaults for override.
        """
        raise NotImplementedError
        yield

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        RunUsage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:  # pragma: no branch
                    # if we couldn't find the agent in locals and globals are a different dict, try globals
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

    @staticmethod
    @contextmanager
    def parallel_tool_call_execution_mode(mode: tool_manager.ParallelExecutionMode = 'parallel') -> Generator[None]:
        """Set the parallel execution mode during the context.

        Args:
            mode: The execution mode for tool calls:
                - 'parallel': Run tool calls in parallel, yielding events as they complete (default).
                - 'sequential': Run tool calls one at a time in order.
                - 'parallel_ordered_events': Run tool calls in parallel, but events are emitted in order, after all calls complete.
        """
        with ToolManager.parallel_execution_mode(mode):
            yield

    @staticmethod
    @contextmanager
    def using_thread_executor(executor: Executor) -> Generator[None]:
        """Use a custom executor for running sync functions in threads during the context.

        By default, sync tool functions and other sync callbacks are run in threads using
        [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync], which creates ephemeral threads.
        In long-running servers (e.g. FastAPI), this can lead to thread accumulation under sustained load.

        This context manager lets you provide a bounded
        [`ThreadPoolExecutor`][concurrent.futures.ThreadPoolExecutor] (or any
        [`Executor`][concurrent.futures.Executor]) to control thread lifecycle:

        ```python {test="skip" lint="skip"}
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import asynccontextmanager

        from pydantic_ai import Agent

        @asynccontextmanager
        async def lifespan(app):
            executor = ThreadPoolExecutor(max_workers=16)
            with Agent.using_thread_executor(executor):
                yield
            executor.shutdown(wait=True)
        ```

        For per-agent configuration, use the
        [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] capability instead.

        Args:
            executor: The executor to use for running sync functions.
        """
        with _utils.using_thread_executor(executor):
            yield

    @staticmethod
    def is_model_request_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.ModelRequestNode[T, S]]:
        """Check if the node is a `ModelRequestNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.ModelRequestNode)

    @staticmethod
    def is_call_tools_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.CallToolsNode[T, S]]:
        """Check if the node is a `CallToolsNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.CallToolsNode)

    @staticmethod
    def is_user_prompt_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[_agent_graph.UserPromptNode[T, S]]:
        """Check if the node is a `UserPromptNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.UserPromptNode)

    @staticmethod
    def is_end_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeIs[End[result.FinalResult[S]]]:
        """Check if the node is a `End`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, End)

    @abstractmethod
    async def __aenter__(self) -> AbstractAgent[AgentDepsT, OutputDataT]:
        raise NotImplementedError

    @abstractmethod
    async def __aexit__(self, *args: Any) -> bool | None:
        raise NotImplementedError

    async def to_cli(
        self: Self,
        deps: AgentDepsT = None,
        prog_name: str = 'pydantic-ai',
        message_history: Sequence[_messages.ModelMessage] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        """Run the agent in a CLI chat interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
            message_history: History of the conversation so far.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            model: Optional model to use for the agent run.

        Example:
        ```python {title="agent_to_cli.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2', instructions='You always respond in Italian.')

        async def main():
            await agent.to_cli()
        ```
        """
        from rich.console import Console

        from pydantic_ai._cli import run_chat

        await run_chat(
            stream=True,
            agent=self,
            deps=deps,
            console=Console(),
            code_theme='monokai',
            prog_name=prog_name,
            message_history=message_history,
            model=model,
            model_settings=model_settings,
            usage_limits=usage_limits,
        )

    def to_cli_sync(
        self: Self,
        deps: AgentDepsT = None,
        prog_name: str = 'pydantic-ai',
        message_history: Sequence[_messages.ModelMessage] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
    ) -> None:
        """Run the agent in a CLI chat interface with the non-async interface.

        Args:
            deps: The dependencies to pass to the agent.
            prog_name: The name of the program to use for the CLI. Defaults to 'pydantic-ai'.
            message_history: History of the conversation so far.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            model: Optional model to use for the agent run.

        ```python {title="agent_to_cli_sync.py" test="skip"}
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-5.2', instructions='You always respond in Italian.')
        agent.to_cli_sync()
        agent.to_cli_sync(prog_name='assistant')
        ```
        """
        return _utils.run_until_complete(
            self.to_cli(
                deps=deps,
                prog_name=prog_name,
                message_history=message_history,
                model=model,
                model_settings=model_settings,
                usage_limits=usage_limits,
            )
        )
