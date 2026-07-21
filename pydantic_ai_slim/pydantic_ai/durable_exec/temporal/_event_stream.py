from __future__ import annotations

import math
from collections.abc import AsyncIterator, Callable
from datetime import timedelta
from typing import Any, cast

import anyio
from temporalio import activity
from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.workflow_streams import WorkflowStreamClient, WorkflowStreamItem

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import AgentDepsT, RunContext

__all__ = ['workflow_stream_event_handler', 'stream_agent_events']

AgentStreamEventFilter = Callable[[AgentStreamEvent], bool]
"""A predicate selecting which `AgentStreamEvent`s to publish to a Workflow Stream topic."""

_DEFAULT_BATCH_INTERVAL = timedelta(milliseconds=100)
_DEFAULT_POLL_COOLDOWN = timedelta(milliseconds=100)


def workflow_stream_event_handler(
    topic: str,
    *,
    events: AgentStreamEventFilter | None = None,
    batch_interval: timedelta = _DEFAULT_BATCH_INTERVAL,
) -> EventStreamHandler[AgentDepsT]:
    """Build an [`EventStreamHandler`][pydantic_ai.agent.EventStreamHandler] that publishes agent events to a Temporal Workflow Stream.

    The [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] capability runs its
    event stream handler inside activities, which is exactly where `WorkflowStreamClient.from_within_activity`
    needs to be called. The returned handler publishes each `AgentStreamEvent` to `topic` on the parent
    workflow's `WorkflowStream`, so an external consumer (e.g. an HTTP layer driving a UI) can subscribe to
    the workflow and observe events in real time — see
    [`stream_agent_events`][pydantic_ai.durable_exec.temporal.stream_agent_events].

    Events are serialized with the same Pydantic payload converter the Temporal integration uses elsewhere,
    so the consumer decodes them back into typed `AgentStreamEvent`s.

    This is the building block behind the `event_stream_topic=` argument of `TemporalDurability`; pass the
    returned handler as an `event_stream_handler` (or combine it with your own) to use it explicitly.

    When invoked outside a Temporal activity — the agent run directly, or a workflow-side replay through a
    [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] handler — publishing isn't possible,
    so the events are simply drained.

    Args:
        topic: The Workflow Stream topic to publish events to.
        events: Optional predicate to select which events to publish; by default every event is published.
            A model stream emits a `PartDeltaEvent` per token, so filtering (e.g. dropping deltas) can
            significantly reduce the number of durable batches.
        batch_interval: How often the client flushes buffered events to the workflow. Defaults to 100ms.
    """

    async def handler(run_context: RunContext[AgentDepsT], stream: Any) -> None:
        if not activity.in_activity():
            async for _ in stream:
                pass
            return

        client = WorkflowStreamClient.from_within_activity(batch_interval=batch_interval)
        topic_handle = client.topic(topic)
        async with client:  # background flusher; flushes remaining events on exit
            async for event in stream:
                if events is None or events(event):
                    topic_handle.publish(event)

    return handler


def combine_event_stream_handlers(
    *handlers: EventStreamHandler[AgentDepsT],
) -> EventStreamHandler[AgentDepsT]:
    """Fan one event stream out to several handlers concurrently, each seeing every event.

    Used to keep `event_stream_topic=` orthogonal to `event_stream_handler=`: the topic publisher is
    installed as one more independent consumer of the same stream a user-supplied handler sees, so neither
    has to merge the other in by hand.
    """

    async def combined(run_context: RunContext[AgentDepsT], stream: Any) -> None:
        senders: list[Any] = []
        async with anyio.create_task_group() as tg:
            for handler in handlers:
                send, receive = anyio.create_memory_object_stream[_messages.AgentStreamEvent](math.inf)
                senders.append(send)

                async def run(handler: EventStreamHandler[AgentDepsT] = handler, receive: Any = receive) -> None:
                    async with receive:
                        await handler(run_context, receive)

                tg.start_soon(run)

            try:
                async for event in stream:
                    for send in senders:
                        try:
                            send.send_nowait(event)
                        except anyio.BrokenResourceError:  # pragma: no cover - a consumer stopped early
                            pass
            finally:
                for send in senders:
                    send.close()

    return combined


async def stream_agent_events(
    client: Client,
    handle: WorkflowHandle[Any, Any],
    topic: str,
    *,
    from_offset: int = 0,
    poll_cooldown: timedelta = _DEFAULT_POLL_COOLDOWN,
) -> AsyncIterator[AgentStreamEvent]:
    """Subscribe to a durable agent run's events published to a Workflow Stream topic.

    This is effectively a durable [`Agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events]
    across the Temporal workflow boundary: an agent configured with
    `TemporalDurability(event_stream_topic=...)` (or the
    [`workflow_stream_event_handler`][pydantic_ai.durable_exec.temporal.workflow_stream_event_handler] it wraps)
    publishes each `AgentStreamEvent` to `topic`, and this async iterator yields them back as typed events, in
    order, until the workflow reaches a terminal state.

    Because Workflow Streams are offset-addressed, a reconnecting consumer can resume from its last seen offset
    by passing `from_offset`, which is more robust than ordinary in-process streaming. Use
    `WorkflowStreamClient` directly if you need each event's offset.

    Args:
        client: A Temporal `Client` configured with the Pydantic AI plugin (so events decode into typed
            `AgentStreamEvent`s).
        handle: The `WorkflowHandle` for the durable agent run.
        topic: The Workflow Stream topic the agent publishes to (the capability's `event_stream_topic`).
        from_offset: The stream offset to start from; pass the offset after the last event seen to resume.
        poll_cooldown: How long to wait between polls when no new events are ready.
    """
    # Pin the subscription to the run the handle refers to, so a reused workflow ID can't redirect
    # us to a different execution. `WorkflowStreamClient.create` resolves `handle.id` to the *latest*
    # execution; building the client from a run-pinned handle instead targets the caller's run, while
    # `subscribe` still follows continue-as-new (which it does whenever a `client` is supplied).
    run_id = handle.run_id or handle.first_execution_run_id
    pinned_handle = client.get_workflow_handle(handle.id, run_id=run_id) if run_id else handle
    stream_client = WorkflowStreamClient(pinned_handle, client=client)
    subscription = cast(
        AsyncIterator[WorkflowStreamItem[AgentStreamEvent]],
        stream_client.subscribe(
            [topic],
            from_offset=from_offset,
            result_type=AgentStreamEvent,  # type: ignore[arg-type]
            poll_cooldown=poll_cooldown,
        ),
    )
    async for item in subscription:
        yield item.data
