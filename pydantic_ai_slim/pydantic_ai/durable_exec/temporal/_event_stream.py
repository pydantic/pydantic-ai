from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Any, Generic, cast
from weakref import WeakValueDictionary

import pydantic
from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.workflow_streams import (
    WorkflowStream,
    WorkflowStreamClient,
    WorkflowStreamItem,
    WorkflowStreamState,
)
from temporalio.service import RPCError

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from pydantic_ai.ui import NativeEvent

__all__ = [
    'WorkflowStreamTopic',
    'AgentEventStream',
    'DurableAgentRunEvents',
    'workflow_stream_event_handler',
    'stream_agent_events',
]

_DRAINED_SIGNAL = '__pydantic_ai_agent_events_drained'
"""Signal a subscriber sends once it has consumed the terminal event, releasing the workflow to finish."""

_DEFAULT_BATCH_INTERVAL = timedelta(milliseconds=100)
"""How often an activity flushes buffered events to the workflow.

The Temporal SDK defaults to 2 seconds, a reasonable batch size for status updates but far too coarse
for driving a UI off a token stream.
"""

_DEFAULT_POLL_COOLDOWN = timedelta(milliseconds=100)
_DEFAULT_DRAIN_TIMEOUT = timedelta(seconds=30)


@dataclass(repr=False)
class _DurableAgentRunResultEvent(AgentRunResultEvent[OutputDataT]):
    """Wire event carrying the token that makes its drain acknowledgment idempotent."""

    drain_token: str = field(kw_only=True)


@dataclass(frozen=True)
class WorkflowStreamTopic:
    """The Workflow Stream topic a durable agent run publishes its events to.

    Pass this — or, for the default settings, just the topic name — as
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]'s
    `event_stream_topic`. The capability keeps it, so
    [`stream_agent_events()`][pydantic_ai.durable_exec.temporal.TemporalDurability.stream_agent_events]
    knows which topic to subscribe to without the name being written out again.
    """

    name: str
    """The topic name. Different agents (or different purposes) belong on different topics."""

    events: Callable[[AgentStreamEvent], bool] | None = None
    """Optional predicate selecting which events to publish; by default every event is published.

    A model stream emits a [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] per token, and
    every published event stays in workflow state for the life of the run, so dropping deltas can
    substantially cut both cost and workflow size.

    The terminal [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] is always published: it
    is what tells a subscriber the run is over.
    """

    batch_interval: timedelta = _DEFAULT_BATCH_INTERVAL
    """How often a model-request activity flushes buffered events to the workflow."""

    def __post_init__(self) -> None:
        if not self.name:
            raise UserError('A Workflow Stream topic needs a name.')
        if self.batch_interval <= timedelta(0):
            raise UserError('The Workflow Stream batch interval must be greater than zero.')

    @staticmethod
    def coerce(topic: str | WorkflowStreamTopic) -> WorkflowStreamTopic:
        return topic if isinstance(topic, WorkflowStreamTopic) else WorkflowStreamTopic(topic)


# Workflow code has no per-instance storage a capability can reach, so the stream registers itself
# under the run it belongs to. Weak values mean the entry goes away with the workflow instance
# holding the stream; replay re-runs `@workflow.init` and re-registers under the same run id.
_streams: WeakValueDictionary[str, AgentEventStream] = WeakValueDictionary()


class AgentEventStream:
    """Hosts the [Workflow Stream](https://docs.temporal.io/develop/python/workflows/workflow-streams) a durable agent run publishes its events to.

    Construct one in your workflow's `@workflow.init` and use it as an async context manager around
    the agent run:

    ```python {test="skip"}
    from temporalio import workflow

    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import AgentEventStream, TemporalDurability

    agent = Agent(
        'openai:gpt-5.6-sol',
        name='assistant',
        capabilities=[TemporalDurability(event_stream_topic='agent-events')],
    )


    @workflow.defn
    class AssistantWorkflow:
        @workflow.init
        def __init__(self, prompt: str) -> None:
            self.events = AgentEventStream()

        @workflow.run
        async def run(self, prompt: str) -> str:
            async with self.events:
                result = await agent.run(prompt)
            return result.output
    ```

    Leaving the block waits for a subscriber to acknowledge that it consumed the terminal
    [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent]. That wait is not bookkeeping: a
    Workflow Stream is served by the workflow itself, so once the workflow returns, its stream can no
    longer be read and anything a subscriber had not yet polled is gone. Nobody watching, or a
    subscriber that went away, is not an error — the wait is bounded by `drain_timeout` and the run's
    result stays authoritative either way.
    """

    def __init__(
        self,
        stream: WorkflowStream | None = None,
        *,
        prior_state: WorkflowStreamState | None = None,
        drain_timeout: timedelta = _DEFAULT_DRAIN_TIMEOUT,
    ) -> None:
        """Construct the stream. Must be called from the workflow's `@workflow.init` method.

        Args:
            stream: An existing `WorkflowStream` to publish to, if your workflow already hosts one for
                its own topics. By default a new one is created. Only one `WorkflowStream` can be
                registered per workflow.
            prior_state: Stream state carried across continue-as-new, when the stream is created here.
                Ignored when `stream` is given, as that one was constructed with its own.
            drain_timeout: How long to wait for a subscriber to acknowledge the terminal event before
                finishing anyway, so a run nobody is watching can't hang.
        """
        self._stream = stream if stream is not None else WorkflowStream(prior_state)
        self._drain_timeout = drain_timeout
        self._pending_drains: set[str] = set()
        self._next_drain_token = 0
        workflow.set_signal_handler(_DRAINED_SIGNAL, self._on_drained)  # pyright: ignore[reportUnknownMemberType]
        _streams[workflow.info().run_id] = self

    @property
    def stream(self) -> WorkflowStream:
        """The underlying `WorkflowStream`, for publishing your own topics or for `continue_as_new()`."""
        return self._stream

    def _on_drained(self, drain_token: str) -> None:
        # Several subscribers may acknowledge the same terminal event; the barrier only needs one.
        self._pending_drains.discard(drain_token)

    def _publish(self, topic: str, event: NativeEvent) -> None:
        """Append an event to the log from workflow code.

        Workflow-side publishing costs nothing beyond the workflow task that is already running, and
        replay rebuilds the log identically, so these events are delivered exactly once.
        """
        self._stream.topic(topic).publish(event)

    def _publish_result(self, topic: str, result: AgentRunResult[Any]) -> None:
        """Append a terminal event and raise its execution-scoped drain barrier."""
        run_id = workflow.info().run_id
        drain_token = f'{run_id}:{self._next_drain_token}'
        self._next_drain_token += 1
        self._stream.topic(topic).publish(_DurableAgentRunResultEvent(result, drain_token=drain_token))
        self._pending_drains.add(drain_token)

    async def __aenter__(self) -> AgentEventStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Wait for a subscriber to drain the stream, then release any that are still polling."""
        try:
            await workflow.wait_condition(lambda: not self._pending_drains, timeout=self._drain_timeout)
        except asyncio.TimeoutError:
            pass
        # A subscription is a long-poll update, and a workflow can't return while one is parked.
        # Detaching releases the waiters (and rejects new polls) so the run can finish.
        self._stream.detach_pollers()
        await workflow.wait_condition(workflow.all_handlers_finished)


def _get_agent_event_stream() -> AgentEventStream:
    """Return the stream hosted by the current workflow."""
    stream = _streams.get(workflow.info().run_id)
    if stream is None:
        raise UserError(
            'An agent with `event_stream_topic` set needs its workflow to host an `AgentEventStream`. '
            "Assign one to an attribute in the workflow's `@workflow.init` method: "
            '`self.events = AgentEventStream()`.'
        )
    return stream


def publish_agent_event(topic: str, event: NativeEvent) -> None:
    """Publish one event to `topic` on the current workflow's `AgentEventStream`.

    Called from workflow code for every event the activity-side publisher doesn't produce.
    """
    stream = _get_agent_event_stream()
    stream._publish(topic, event)  # pyright: ignore[reportPrivateUsage]


def publish_agent_result(topic: str, result: AgentRunResult[Any]) -> None:
    """Publish a terminal event with an idempotent drain token."""
    stream = _get_agent_event_stream()
    stream._publish_result(topic, result)  # pyright: ignore[reportPrivateUsage]


def workflow_stream_event_handler(
    topic: str | WorkflowStreamTopic,
    *,
    handler: EventStreamHandler[AgentDepsT] | None = None,
) -> EventStreamHandler[AgentDepsT]:
    """Build an [`EventStreamHandler`][pydantic_ai.agent.EventStreamHandler] that publishes the events it sees to a Workflow Stream topic.

    This is the building block `TemporalDurability(event_stream_topic=...)` uses for the live model
    stream, exposed so it can be composed or wrapped. Reach for the capability argument first: it
    additionally publishes the run's workflow-side events from workflow code — no activity, no signal,
    exactly once — and the terminal
    [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] that ends a subscription. Installed
    on its own as an `event_stream_handler`, this handler publishes only what it is handed, and every
    event it publishes goes out from an activity, so an activity retry republishes it.

    The handler publishes to the workflow that scheduled the activity it runs in. Outside an activity
    — an agent run outside a workflow, or a workflow-side replay through
    [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] — there is nothing to publish
    to, so the events pass through untouched. Events are serialized with the integration's Pydantic
    payload converter, so subscribers decode them back into typed events.

    Args:
        topic: The topic to publish to.
        handler: An optional handler to run alongside publishing. Each event is published and then
            passed on to this handler, which sees exactly the stream it would have seen on its own.
    """
    topic = WorkflowStreamTopic.coerce(topic)

    async def publishing_handler(run_context: RunContext[AgentDepsT], stream: AsyncIterable[AgentStreamEvent]) -> None:
        if not activity.in_activity():
            # Publishing needs an activity. Outside one -- an agent run outside a workflow, or a
            # workflow-side replay through `ProcessEventStream` -- pass the stream on unchanged so a
            # wrapped handler still sees it.
            if handler is not None:
                await handler(run_context, stream)
                return
            async for _ in stream:
                pass
            return

        # Pin publishing to the run that scheduled this activity. `from_within_activity()` builds its
        # handle from the workflow ID alone, which resolves to the *latest* execution: if this run is
        # cancelled while its activities are still shutting down and the workflow ID is then reused,
        # trailing events would land in the new execution's stream.
        info = activity.info()
        if info.workflow_id is None or info.workflow_run_id is None:
            raise UserError(
                'A Workflow Stream event handler can only publish from an activity scheduled by a workflow. '
                'This activity was started directly on the client, so it has no workflow stream to publish to.'
            )
        temporal_client = activity.client()
        activity_handle = temporal_client.get_workflow_handle(info.workflow_id, run_id=info.workflow_run_id)
        client = WorkflowStreamClient(activity_handle, client=temporal_client, batch_interval=topic.batch_interval)
        topic_handle = client.topic(topic.name)

        async def publishing(events: AsyncIterable[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
            async for event in events:
                if topic.events is None or topic.events(event):
                    topic_handle.publish(event)
                yield event

        async with client:  # background flusher; flushes what's left on exit
            published = publishing(stream)
            try:
                if handler is not None:
                    await handler(run_context, published)
            finally:
                # Keep publishing whatever a handler that returned early left behind: the topic's
                # subscribers are not that handler's business.
                async for _ in published:
                    pass

    return publishing_handler


class DurableAgentRunEvents(Generic[OutputDataT], AsyncIterator['NativeEvent']):
    """The event iterator returned by [`TemporalDurability.stream_agent_events()`][pydantic_ai.durable_exec.temporal.TemporalDurability.stream_agent_events].

    A durable [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events]: it yields the
    run's [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s in order, ending with the
    trailing [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] that carries the result —
    except that these events crossed a workflow boundary, so the run itself may be executing in
    another process entirely.

    One iterator covers one agent run. A workflow that runs the agent repeatedly publishes a terminal
    event per run, so a consumer that wants the next run reconnects with `from_offset=offset + 1`.
    """

    def __init__(
        self,
        subscription: AsyncIterator[WorkflowStreamItem[NativeEvent]],
        handle: WorkflowHandle[Any, Any],
    ) -> None:
        self._subscription = subscription
        self._handle = handle
        self._offset = -1
        self._result: AgentRunResult[OutputDataT] | None = None
        self._done = False

    @property
    def offset(self) -> int:
        """The stream offset of the last event yielded, or `-1` before the first.

        Workflow Streams are offset-addressed, so a consumer that checkpoints this can reconnect with
        `from_offset=offset + 1` and pick up exactly where it left off — which is more than ordinary
        in-process streaming can offer. Offsets run over the whole stream rather than per topic, so
        they skip whatever the workflow published to its other topics.
        """
        return self._offset

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The run's result, once the terminal event has been yielded."""
        return self._result

    def __aiter__(self) -> AsyncIterator[NativeEvent]:
        return self

    async def __anext__(self) -> NativeEvent:
        if self._done:
            raise StopAsyncIteration
        try:
            item = await anext(self._subscription)
        except StopAsyncIteration:
            # The workflow reached a terminal state without publishing a terminal event, e.g. because
            # the run failed or the workflow was cancelled. `handle.result()` says which.
            self._done = True
            raise
        self._offset = item.offset
        event = item.data
        if isinstance(event, _DurableAgentRunResultEvent):
            self._result = event.result
            self._done = True
            # Stop polling before acknowledging: the workflow finishes on that signal, and a poll
            # still in flight when it does would fail rather than return.
            await self.aclose()
            try:
                await self._handle.signal(_DRAINED_SIGNAL, event.drain_token)
            except RPCError as exc:
                # The terminal result has already crossed the workflow boundary. An acknowledgment
                # failure only makes the producer wait for its bounded drain timeout; it must not
                # turn that received result into a consumer-side stream failure.
                warnings.warn(
                    f'Failed to acknowledge the terminal agent event; the workflow may wait for its '
                    f'`drain_timeout`: {exc}',
                    RuntimeWarning,
                    stacklevel=2,
                )
            return AgentRunResultEvent(event.result)
        return event

    async def __aenter__(self) -> DurableAgentRunEvents[OutputDataT]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop the underlying subscription.

        Reaching the terminal event does this for you. Call it (or use the iterator as an async
        context manager) when you stop early, so the long-poll against Temporal doesn't stay open
        until the generator is collected.
        """
        await cast('AsyncGeneratorLike', self._subscription).aclose()


if TYPE_CHECKING:

    class AsyncGeneratorLike:
        async def aclose(self) -> None: ...


def stream_agent_events(
    client: Client,
    handle: WorkflowHandle[Any, Any],
    topic: str | WorkflowStreamTopic,
    *,
    output_type: type[OutputDataT] = cast('type[Any]', Any),
    from_offset: int = 0,
    poll_cooldown: timedelta = _DEFAULT_POLL_COOLDOWN,
) -> DurableAgentRunEvents[OutputDataT]:
    """Subscribe to the events a durable agent run publishes to a Workflow Stream topic.

    Prefer [`TemporalDurability.stream_agent_events()`][pydantic_ai.durable_exec.temporal.TemporalDurability.stream_agent_events],
    which fills in the topic and output type from the capability. Use this function from a consumer
    that holds neither, only a workflow handle.

    Args:
        client: A Temporal `Client` configured with [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin],
            so events decode back into typed Pydantic AI events.
        handle: The handle for the workflow running the agent.
        topic: The topic the agent publishes to.
        output_type: The agent's output type, so the terminal event's result is decoded into it.
        from_offset: The stream offset to start from, inclusive; pass `offset + 1` to resume.
        poll_cooldown: How long to wait between polls when no new events are ready. Passed through to
            the SDK, including `timedelta(0)`, which busy-polls: that grows the workflow's history
            toward its limit, so it belongs in tests rather than in a running application.
    """
    topic = WorkflowStreamTopic.coerce(topic)
    # Pin the subscription to the run the handle refers to, so a reused workflow ID can't redirect us
    # to a different execution: `WorkflowStreamClient.create` resolves a workflow ID to the latest
    # one. Passing `client=` keeps continue-as-new following, which re-targets to the successor.
    run_id = handle.run_id or handle.first_execution_run_id
    pinned = client.get_workflow_handle(handle.id, run_id=run_id) if run_id else handle
    result_type = Annotated[
        AgentStreamEvent | _DurableAgentRunResultEvent[output_type], pydantic.Discriminator('event_kind')
    ]
    subscription = WorkflowStreamClient(pinned, client=client).subscribe(
        [topic.name],
        from_offset=from_offset,
        result_type=cast('type[Any]', result_type),
        poll_cooldown=poll_cooldown,
    )
    # Polling follows continue-as-new by retargeting to the latest execution. Acknowledgments need
    # to do the same; their execution-scoped token makes signaling a later reused workflow ID safe.
    acknowledgment_handle = client.get_workflow_handle(handle.id)
    return DurableAgentRunEvents(
        cast('AsyncIterator[WorkflowStreamItem[NativeEvent]]', subscription), acknowledgment_handle
    )
