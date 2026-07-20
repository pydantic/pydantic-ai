from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from datetime import timedelta

from temporalio.contrib.workflow_streams import WorkflowStreamClient

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.tools import AgentDepsT, RunContext


def make_workflow_stream_event_handler(
    topic: str,
    batch_interval: timedelta,
    inner: EventStreamHandler[AgentDepsT] | None,
) -> EventStreamHandler[AgentDepsT]:
    """Build an event stream handler that publishes events to a Temporal Workflow Stream.

    The returned handler is used inside the Temporal activities that run the agent's model
    request stream and its per-event handling. It publishes every `AgentStreamEvent` to `topic`
    on the parent workflow's [`WorkflowStream`][temporalio.contrib.workflow_streams.WorkflowStream]
    via [`WorkflowStreamClient.from_within_activity`][temporalio.contrib.workflow_streams.WorkflowStreamClient.from_within_activity],
    so an external consumer (e.g. an HTTP layer driving a UI) can subscribe to the workflow and
    observe events in real time.

    Publishing is a side effect: each event is forwarded to `inner` (if set) unchanged, so a
    user-supplied `event_stream_handler` still sees every event.

    Args:
        topic: The Workflow Stream topic to publish events to.
        batch_interval: How often the client flushes buffered events to the workflow.
        inner: An optional user-supplied handler to forward events to after publishing.
    """

    async def handler(
        run_context: RunContext[AgentDepsT],
        stream: AsyncIterable[_messages.AgentStreamEvent],
    ) -> None:
        client = WorkflowStreamClient.from_within_activity(batch_interval=batch_interval)
        topic_handle = client.topic(topic)
        async with client:  # background flusher; flushes remaining events on exit

            async def republish() -> AsyncIterator[_messages.AgentStreamEvent]:
                async for event in stream:
                    topic_handle.publish(event)
                    yield event

            if inner is not None:
                await inner(run_context, republish())
            else:
                async for _ in republish():
                    pass

    return handler
