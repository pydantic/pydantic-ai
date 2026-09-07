from __future__ import annotations as _annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

from pydantic_graph import BaseNode, End, EndMarker, ErrorMarker, GraphRun, GraphRunContext, GraphTaskRequest, JoinItem
from pydantic_graph.step import NodeStep

from . import (
    _agent_graph,
    _utils,
    exceptions,
    messages as _messages,
    usage as _usage,
)
from ._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority
from ._instrumentation import current_otel_traceparent
from ._run_context import CustomEventT
from .output import OutputDataT
from .tools import AgentDepsT

if TYPE_CHECKING:
    from ._run_context import RunContext
    from .result import FinalResult


@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, OutputDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.basenode.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-5.2')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
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
                    usage=RequestUsage(
                        cost=Decimal('0.000196'), input_tokens=56, output_tokens=7
                    ),
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

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[OutputDataT]
    ]
    _result_override: AgentRunResult[OutputDataT] | None = dataclasses.field(default=None, repr=False, init=False)
    _node_error: BaseException | None = dataclasses.field(default=None, repr=False, init=False)
    """Stores the original exception from node execution, before context manager __aexit__ may transform it."""
    _last_yielded_node: _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]] | None = (
        dataclasses.field(default=None, repr=False, init=False)
    )
    """The node most recently yielded by `__anext__`, run on the following iteration."""

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        traceparent = self._graph_run._traceparent(required=False)  # type: ignore[reportPrivateUsage]
        if traceparent is None:
            # Fall back to the active OTel span, which is the agent run span
            # when the Instrumentation capability is active.
            traceparent = current_otel_traceparent()
        if traceparent is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return traceparent

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            state=self._graph_run.state, deps=self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        task = self._graph_run.next_task
        if isinstance(task, ErrorMarker):
            raise task.error
        return self._task_to_node(task)

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.basenode.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        if self._result_override is not None:
            return self._result_override
        graph_run_output = self._graph_run.output
        if graph_run_output is None:
            return None
        return AgentRunResult(
            graph_run_output.output,
            graph_run_output.tool_name,
            self._graph_run.state,
            self._graph_run.deps.new_message_index,
            self._traceparent(required=False),
        )

    def all_messages(self) -> list[_messages.ModelMessage]:
        """Return all messages for the run so far.

        Messages from older runs are included.
        """
        return self.ctx.state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRun.all_messages] as JSON bytes.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[_messages.ModelMessage]:
        """Return the messages produced during this run so far.

        Messages provided via `message_history` and messages from older runs are excluded.
        """
        return self.all_messages()[self.ctx.deps.new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRun.new_messages] as JSON bytes.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages())

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Advance to the next node automatically based on the last returned node.

        Yields each node before it runs, ending with the [`End`][pydantic_graph.basenode.End] node.
        Advancing goes through [`next()`][pydantic_ai.run.AgentRun.next], so capability hooks fire
        exactly as they do for [`agent.run()`][pydantic_ai.agent.AbstractAgent.run].
        """
        if self._result_override is not None:
            raise StopAsyncIteration

        previous = self._last_yielded_node
        if isinstance(previous, End):
            # The run already completed; a trailing `__anext__` (e.g. the one `async for` performs
            # after yielding `End`) must stop cleanly even if `cancel()` was requested as the final
            # step finished — cancelling an already-finished run is a no-op.
            raise StopAsyncIteration

        self.ctx.deps.cancellation.bind()
        if self.ctx.deps.cancellation.cancel_requested:
            # Honor a first-party cancellation (`cancel()` on this run, possibly from the caller's
            # previous loop body) before yielding another node the caller would go on to run. The
            # first node is yielded before it runs, so unlike `self.next(previous)` this boundary
            # has no awaited step to carry the pending `task.cancel()`: yield to the event loop so
            # it's delivered here on every Python version (`raise_if_cancelling` only re-asserts on
            # 3.11+, and neither path awaits).
            await asyncio.sleep(0)
        _utils.raise_if_cancelling()

        if previous is None:
            # The first node hasn't run yet: yield it so the caller can inspect (or replace) it,
            # and run it on the next iteration.
            node = self.next_node
        elif (current := self.next_node) is not previous:
            # The loop body advanced the run itself, e.g. by calling `next()` on the node we just
            # yielded. Running `previous` again here would execute it — and its hooks — a second
            # time, so surface where the graph actually is instead.
            node = current
        else:
            try:
                node = await self.next(previous)
            except BaseException as exc:
                self._node_error = exc
                raise

        self._last_yielded_node = node
        return node

    def _task_to_node(
        self, task: EndMarker[FinalResult[OutputDataT]] | JoinItem | Sequence[GraphTaskRequest]
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        if isinstance(task, Sequence) and len(task) == 1:
            first_task = task[0]
            if isinstance(first_task.inputs, BaseNode):  # pragma: no branch
                base_node: BaseNode[  # pyright: ignore[reportUnknownVariableType]
                    _agent_graph.GraphAgentState,
                    _agent_graph.GraphAgentDeps[AgentDepsT, OutputDataT],
                    FinalResult[OutputDataT],
                ] = first_task.inputs  # pyright: ignore[reportUnknownMemberType]
                if _agent_graph.is_agent_node(node=base_node):  # pragma: no branch
                    return base_node
        if isinstance(task, EndMarker):
            return End(task.value)
        raise exceptions.AgentRunError(f'Unexpected node: {task}')  # pragma: no cover

    def _node_to_task(self, node: _agent_graph.AgentNode[AgentDepsT, OutputDataT]) -> GraphTaskRequest:
        return GraphTaskRequest(NodeStep(type(node)).id, inputs=node, fork_stack=())

    def _sync_graph_state(self, result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]) -> None:
        """Synchronize the graph runner's state to match a hook-modified result.

        After a capability hook changes the result (e.g. `on_node_run_error` recovering,
        or `after_node_run` converting End↔node), the graph runner's internal `_next` must
        be updated so that `output` and `next_node` reflect the hook's decision.
        """
        if isinstance(result, End):
            self._graph_run.override_next(EndMarker(result.data))
        else:
            self._graph_run.override_next([self._node_to_task(result)])

    def _graph_pending_node(self) -> _agent_graph.AgentNode[AgentDepsT, Any] | None:
        """The node the graph runner is pending on, or `None` if it isn't pointing at one.

        Unlike `next_node` and `_task_to_node`, this never raises: an `ErrorMarker`, an `EndMarker`
        or a shape it doesn't recognise all read as "no pending node".
        """
        task = self._graph_run.next_task
        if isinstance(task, Sequence) and len(task) == 1:
            node = task[0].inputs
            if isinstance(node, BaseNode):  # pragma: no branch
                base_node: BaseNode[  # pyright: ignore[reportUnknownVariableType]
                    _agent_graph.GraphAgentState,
                    _agent_graph.GraphAgentDeps[AgentDepsT, Any],
                    FinalResult[Any],
                ] = node
                if _agent_graph.is_agent_node(base_node):  # pragma: no branch
                    return base_node
        return None

    def _graph_reflects(self, result: _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]) -> bool:
        """Whether the graph runner's own state already records `result` as the step's outcome.

        False whenever the two have diverged, whatever the cause: the graph is still pending on the
        node a hook short-circuited past, or holds an `ErrorMarker` for an error a hook handled, or
        advanced to the handler's node while the hook returned something else.
        """
        if isinstance(result, End):
            task = self._graph_run.next_task
            return isinstance(task, EndMarker) and task.value is result.data
        return self._graph_pending_node() is result

    async def _advance_graph(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Execute a single graph step without firing capability hooks."""
        task = [self._node_to_task(node)]
        try:
            task = await self._graph_run.next(task)
        except StopAsyncIteration:
            pass
        return self._task_to_node(task)

    async def _wrap_and_advance(
        self,
        run_context: RunContext[AgentDepsT],
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        step_fn: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Execute `wrap_node_run(step_fn)` → `on_node_run_error` → `after_node_run`.

        This is the portion of the hook lifecycle after `before_node_run` has already fired.
        Used by both `_run_node_with_hooks` and directly by `run_stream()` which calls
        `before_node_run` separately (before streaming).
        """
        self.ctx.deps.cancellation.bind()
        cap = self.ctx.deps.root_capability
        try:
            if cap._has_wrap_node_run:  # pyright: ignore[reportPrivateUsage]
                result = await cap.wrap_node_run(run_context, node=node, handler=step_fn)
            else:
                result = await step_fn(node)
        except Exception as e:
            if not cap._has_on_node_run_error:  # pyright: ignore[reportPrivateUsage]
                raise
            result = await cap.on_node_run_error(run_context, node=node, error=e)
            # on_node_run_error recovered by returning a result.
            # The graph runner is in ErrorMarker state; update it to match.
            self._sync_graph_state(result)
        else:
            # `wrap_node_run` owns the outcome, but the graph runner only knows what the handler
            # did: nothing at all if the hook short-circuited past it, an error the hook went on to
            # swallow, or a step whose result the hook then replaced. Sync whenever they disagree,
            # so `next_node` and `result` follow the hook rather than the graph.
            if not self._graph_reflects(result):
                self._sync_graph_state(result)
        # If the step (or a hook wrapping it) absorbed an external cancellation, re-assert it
        # before `after_node_run` fires; the step's messages are already recorded.
        _utils.raise_if_cancelling()
        pre_hook_result = result
        result = await cap.after_node_run(run_context, node=node, result=result)

        # If after_node_run changed the result, sync the graph runner state so
        # agent_run.result correctly reflects whether the run is finished.
        if result is not pre_hook_result:
            self._sync_graph_state(result)

        _utils.raise_if_cancelling()
        return result

    async def _run_node_with_hooks(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        step_fn: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Run a node through the full capability hook lifecycle with a custom step function.

        Fires hooks in order: `before_node_run` → `wrap_node_run(step_fn)` → `after_node_run`,
        with `on_node_run_error` handling exceptions from `wrap_node_run`.
        """
        # Bind before `before_node_run` awaits: when the run is driven from a different task than
        # the previous step (manual `next()` from another task), the controller must target this
        # driver so `cancel()` interrupts the hook, not the old task. `_wrap_and_advance` re-binds
        # (idempotently) for the `run_stream()` path, which fires `before_node_run` separately.
        self.ctx.deps.cancellation.bind()
        run_context = _agent_graph.build_run_context(self.ctx)
        cap = self.ctx.deps.root_capability
        node = await cap.before_node_run(run_context, node=node)
        # A `before_node_run` hook that absorbed an external cancellation must not
        # let the node itself start.
        _utils.raise_if_cancelling()
        return await self._wrap_and_advance(run_context, node, step_fn)

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, OutputDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, OutputDataT] | End[FinalResult[OutputDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.basenode.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-5.2')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
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
                            usage=RequestUsage(
                                cost=Decimal('0.000196'), input_tokens=56, output_tokens=7
                            ),
                            model_name='gpt-5.2',
                            timestamp=datetime.datetime(...),
                            run_id='...',
                            conversation_id='...',
                        )
                    ),
                    End(data=FinalResult(output='The capital of France is Paris.')),
                ]
                '''
                print('Final result:', agent_run.result.output)
                #> Final result: The capital of France is Paris.
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.basenode.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        return await self._run_node_with_hooks(node, self._stream_and_advance)

    async def _stream_and_advance(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Execute a single graph step, streaming the node first if capabilities need its events.

        A capability that overrides `wrap_run_event_stream` only sees events if the node is
        streamed, so streaming is enabled for it here the same way `agent.run()` enables it.
        `node.stream()` applies the capability chain itself, so draining it is all that's needed.
        """
        if self.ctx.deps.root_capability.has_wrap_run_event_stream or self.ctx.deps.root_capability.has_on_event:
            await _agent_graph.drain_node_event_stream(node, self.ctx)
        return await self._advance_graph(node)

    @property
    def usage(self) -> _usage.RunUsage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._graph_run.state.metadata

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._graph_run.state.run_id

    @property
    def conversation_id(self) -> str:
        """The unique identifier for the conversation this run belongs to."""
        return self._graph_run.state.conversation_id

    @property
    def pending_messages(self) -> list[PendingMessage]:
        """Internal: live view of the queue mutated by `enqueue` and drained by the internal `PendingMessageDrainCapability`.

        Exposed for inspection / debugging; use [`enqueue`][pydantic_ai.run.AgentRun.enqueue] to add messages.
        """
        return self._graph_run.state.pending_messages

    async def emit(self, event: CustomEventT, /) -> CustomEventT:
        """Emit a [`CustomEvent`][pydantic_ai.messages.CustomEvent] into this run's event stream.

        Lets code driving [`Agent.iter`][pydantic_ai.agent.AbstractAgent.iter] inject application-defined
        events (e.g. from an external harness or event bus) into the stream, alongside events emitted from
        tools via [`RunContext.emit`][pydantic_ai.tools.RunContext.emit].
        The event surfaces on the next pull from the run's node stream.

        Designed to be called from the same event loop driving `agent.iter()`. If you're forwarding events
        from a different thread, submit the coroutine to the agent's loop
        (e.g. `asyncio.run_coroutine_threadsafe(agent_run.emit(event), loop)`).

        Args:
            event: The [`CustomEvent`][pydantic_ai.messages.CustomEvent] to emit.

        Returns:
            The event as emitted.

        Raises:
            UserError: If the run has already ended (no stream remains to deliver the event), or if
                passed a [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent]: those belong
                to capabilities, and code driving the run is application code.
        """
        if self.result is not None:
            raise exceptions.UserError(
                '`emit` cannot be called after the run has ended: nothing remains to deliver the event.'
            )
        if isinstance(event, _messages.CapabilityEvent):
            raise exceptions.UserError(
                'Capability events belong to capabilities and can only be emitted from a capability hook or '
                'capability-contributed tool. Application code should emit a `CustomEvent`; it can re-emit a '
                'received capability event as one.'
            )
        self._graph_run.state.event_stream_buffer.append(event)
        return event

    def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
    ) -> str | None:
        """Enqueue content to be injected into the conversation.

        Designed to be called from the same event loop driving `agent.iter()`. If
        you're forwarding events from a different thread (e.g. a webhook handler
        running on its own loop or thread), marshal the call back onto the agent's
        loop first (e.g. `loop.call_soon_threadsafe(agent_run.enqueue, msg)`).
        The drain's `queue[:] = remaining` pattern in `_drain_by_priority` isn't
        atomic against concurrent appends from a different thread.

        Args:
            *content: One or more [`EnqueueContent`][pydantic_ai.run.EnqueueContent] items.
                Adjacent [`UserContent`][pydantic_ai.messages.UserContent] (a `str` or multi-modal
                content like an [`ImageUrl`][pydantic_ai.messages.ImageUrl]) is gathered into one
                [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], and each
                [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart] (e.g. a
                [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]) is coalesced with adjacent
                part-style items into one [`ModelRequest`][pydantic_ai.messages.ModelRequest]; a complete
                [`ModelRequest`][pydantic_ai.messages.ModelRequest] or
                [`ModelResponse`][pydantic_ai.messages.ModelResponse] is kept as its own message. The
                assembled sequence must end in a request. Calling with no positional args is a no-op.
            priority: When to deliver:
                `'asap'` (default) — at the earliest opportunity (next model request,
                    or a redirect if the agent would otherwise end).
                `'when_idle'` — only when the agent would otherwise end, after `'asap'` messages.

        Returns:
            The `enqueue_id` of the queued message, echoed on the
            [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] emitted when it's
            delivered, or `None` when there was nothing to enqueue (an empty call).
        """
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            return None
        self._graph_run.state.pending_messages.append(pending)
        return pending.enqueue_id

    def cancel(self) -> None:
        """Cancel the whole agent run.

        The run stops what it is doing — the in-flight model request is torn down, in-flight tool
        tasks are cancelled and drained, a suspended server-side job is best-effort cancelled — and
        the code driving the run sees `asyncio.CancelledError`. When the `agent.iter()` context
        exits, this becomes [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] (including for
        `agent.run()`, which wraps `iter()`). Everything that completed before the cancellation took
        effect is preserved in message history.
        [`RunCancelled.all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages] returns a
        complete snapshot that can be passed to a new run as `message_history` to resume the
        conversation.

        Cancellation is terminal: capability hooks (`wrap_run`, `wrap_node_run`, `on_run_error`)
        may observe it and clean up, but cannot recover the run into a successful result.

        Unlike [`StreamedRunResult.cancel()`][pydantic_ai.result.StreamedRunResult.cancel], which
        only stops the current model response and lets the run continue, this ends the run itself.

        Safe to call from another task or thread (e.g. a TUI's key handler while the run is
        awaited elsewhere). Idempotent; a no-op once the run has finished — where "finished" means
        the `agent.iter()`/`agent.run()` context has exited. A `cancel()` issued inside the context
        after the run has already produced its result (e.g. after iterating to `End`) is still
        honored and surfaces as `RunCancelled` on exit, so that a hook running at context exit (like
        `after_run`) can still cancel the run; only after the context has exited is `cancel()` a true
        no-op. Externally cancelling the task running the agent (`asyncio.Task.cancel()`) remains
        supported and keeps raising `asyncio.CancelledError` instead; when both happen, the external
        cancellation wins.
        """
        self.ctx.deps.cancellation.cancel()

    def __repr__(self) -> str:  # pragma: no cover
        result = self._graph_run.output
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage}>'


@dataclasses.dataclass
class AgentRunResult(Generic[OutputDataT]):
    """The final result of an agent run."""

    output: OutputDataT
    """The output data from the agent run."""

    _output_tool_name: str | None = dataclasses.field(repr=False, compare=False, default=None)
    _state: _agent_graph.GraphAgentState = dataclasses.field(
        repr=False, compare=False, default_factory=_agent_graph.GraphAgentState
    )
    _new_message_index: int = dataclasses.field(repr=False, compare=False, default=0)
    _traceparent_value: str | None = dataclasses.field(repr=False, compare=False, default=None)

    @overload
    def _traceparent(self, *, required: Literal[False]) -> str | None: ...
    @overload
    def _traceparent(self) -> str: ...
    def _traceparent(self, *, required: bool = True) -> str | None:
        if self._traceparent_value is None and required:  # pragma: no cover
            raise AttributeError('No span was created for this agent run')
        return self._traceparent_value

    def _set_output_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the output tool.

        Useful if you want to continue the conversation and want to set the response to the output tool call.
        """
        if not self._output_tool_name:
            raise ValueError('Cannot set output tool return content when the return type is `str`.')

        messages = self._state.message_history
        last_message = messages[-1]
        for idx, part in enumerate(last_message.parts):
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._output_tool_name:
                # Only do deepcopy when we have to modify
                copied_messages = list(messages)
                copied_last = deepcopy(last_message)
                copied_last.parts[idx].content = return_content  # type: ignore[misc]
                copied_messages[-1] = copied_last
                return copied_messages

        raise LookupError(f'No tool call found with tool name {self._output_tool_name!r}.')

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        if output_tool_return_content is not None:
            return self._set_output_tool_return(output_tool_return_content)
        else:
            return self._state.message_history

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the messages produced during this run.

        Messages provided via `message_history` and messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the last response from the message history."""
        # The response may not be the very last item if it contained an output tool call. See `CallToolsNode._handle_final_result`.
        for message in reversed(self.all_messages()):
            if isinstance(message, _messages.ModelResponse):
                return message
        raise ValueError('No response found in the message history')  # pragma: no cover

    @property
    def usage(self) -> _usage.RunUsage:
        """Return the usage of the whole run."""
        return self._state.usage

    @property
    def timestamp(self) -> datetime:
        """Return the timestamp of last response."""
        return self.response.timestamp

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._state.metadata

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._state.run_id

    @property
    def conversation_id(self) -> str:
        """The unique identifier for the conversation this run belongs to."""
        return self._state.conversation_id


@dataclasses.dataclass(repr=False)
class AgentRunResultEvent(Generic[OutputDataT]):
    """An event indicating the agent run ended and containing the final result of the agent run."""

    result: AgentRunResult[OutputDataT]
    """The result of the run."""

    _: dataclasses.KW_ONLY

    event_kind: Literal['agent_run_result'] = 'agent_run_result'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr
