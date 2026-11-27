"""Publish a durable agent run's events to a Hatchet durable stream, and read them back typed.

A [durable stream](https://docs.hatchet.run/home/durable-streams) topic stores its messages independently of
any run, so a consumer can subscribe while the agent runs, after it finished, or again from a cursor.

The live model stream is published from inside the model request task that produces it, so tokens reach the
consumer while the model is still streaming. Every other event, and the terminal
[`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent], is published from the durable task. A replayed
durable task publishes those again, so every message carries an ID that a replay reproduces, and
[`HatchetAgentRunEvents`][pydantic_ai.durable_exec.hatchet.HatchetAgentRunEvents] yields each ID once.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Generic, cast

import grpc
import pydantic
from hatchet_sdk import Context, Hatchet
from pydantic import TypeAdapter

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.output import OutputDataT
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tools import AgentDepsT, RunContext

if TYPE_CHECKING:
    from hatchet_sdk.features.streams import StreamEvent

__all__ = ('HatchetAgentRunEvents', 'HatchetEventStream')


def _default_topic(agent_name: str, workflow_run_id: str) -> str:
    return f'pydantic-ai.{agent_name}.{workflow_run_id}'


@dataclass(frozen=True, kw_only=True)
class HatchetEventStream:
    """Where and what a durable agent run publishes to Hatchet's durable streams.

    Pass this, or `True` for the defaults, as `HatchetDurability`'s `event_stream`.
    """

    topic: Callable[[str, str], str] = _default_topic
    """The topic for a run, from the agent name and the workflow run ID of the durable task running it.

    Defaults to `pydantic-ai.<agent name>.<workflow run ID>`, a topic per run.
    """

    namespace: str = ''
    """The durable stream namespace the topics are in."""

    events: Callable[[AgentStreamEvent], bool] | None = None
    """Optional predicate selecting which events to publish; by default every event is published.

    The terminal [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] is always published.
    """


@dataclass(kw_only=True)
class _StreamMessage(Generic[OutputDataT]):
    id: str
    """Identifies the message across replays of the durable task, so a consumer yields it once."""
    event: (
        Annotated[AgentStreamEvent | AgentRunResultEvent[OutputDataT], pydantic.Discriminator('event_kind')] | None
    ) = None
    error: str | None = None
    """Set on the terminal message of a run that failed, which has no result to carry."""


_publish_adapter: TypeAdapter[_StreamMessage[Any]] = TypeAdapter(_StreamMessage[Any])


async def publish_event(
    hatchet: Hatchet,
    config: HatchetEventStream,
    topic: str,
    *,
    message_id: str,
    event: AgentStreamEvent | AgentRunResultEvent[Any] | None = None,
    error: str | None = None,
) -> None:
    message = _StreamMessage[Any](id=message_id, event=event, error=error)
    try:
        await hatchet.streams.aio_publish(topic, _publish_adapter.dump_json(message), namespace=config.namespace)
    except grpc.aio.AioRpcError as rpc_error:
        if rpc_error.code() == grpc.StatusCode.PERMISSION_DENIED:
            # A retry can't fix a missing entitlement, and a model task would repeat its request on every attempt.
            raise UserError(
                f'Publishing agent events to Hatchet durable streams was denied: {rpc_error.details()}. Enable durable '
                "streams for the tenant, or remove `HatchetDurability`'s `event_stream`."
            ) from rpc_error
        raise


def task_event_stream_handler(
    hatchet: Hatchet,
    config: HatchetEventStream,
    agent_name: str,
    *,
    handler: EventStreamHandler[AgentDepsT] | None = None,
) -> EventStreamHandler[AgentDepsT]:
    """An event stream handler that publishes the live model stream from the model request task producing it.

    Outside a task spawned by a durable task, there is no run topic to publish to, so the events pass through.
    """

    async def publishing_handler(run_context: RunContext[AgentDepsT], stream: AsyncIterable[AgentStreamEvent]) -> None:
        ctx = hatchet.current_context
        parent_run_id = ctx.parent_workflow_run_id if isinstance(ctx, Context) else None
        if ctx is not None and parent_run_id is not None:
            topic = config.topic(agent_name, parent_run_id)
            # A retried attempt streams the response again, under new IDs: a consumer sees both attempts' events.
            id_prefix = f'{ctx.task_run_id}:{ctx.attempt_number}'
            stream = _publishing(hatchet, config, topic, id_prefix, stream)
        await _drain(run_context, stream, handler)

    return publishing_handler


async def _publishing(
    hatchet: Hatchet,
    config: HatchetEventStream,
    topic: str,
    id_prefix: str,
    events: AsyncIterable[AgentStreamEvent],
) -> AsyncIterator[AgentStreamEvent]:
    """Publish each selected event to the run's topic as it passes through."""
    index = 0
    async for event in events:
        if config.events is None or config.events(event):
            await publish_event(hatchet, config, topic, message_id=f'{id_prefix}:{index}', event=event)
            index += 1
        yield event


async def _drain(
    run_context: RunContext[AgentDepsT],
    stream: AsyncIterable[AgentStreamEvent],
    handler: EventStreamHandler[AgentDepsT] | None,
) -> None:
    """Hand the stream to `handler`, then consume whatever it left behind, so every event is still published.

    A handler that raises fails the task with its own error, which a stream error from draining would replace.
    """
    if handler is not None:
        await handler(run_context, stream)
    async for _ in stream:
        pass


class HatchetAgentRunEvents(Generic[OutputDataT], AsyncIterator[AgentStreamEvent | AgentRunResultEvent[OutputDataT]]):
    """The event iterator returned by [`HatchetDurability.stream_agent_events()`][pydantic_ai.durable_exec.hatchet.HatchetDurability.stream_agent_events].

    A durable [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events]: it yields the run's
    [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s in order, ending with the
    [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] that carries the result, while the run itself
    executes on a Hatchet worker. A run that fails publishes no result: the iterator then ends early, with the
    failure in `error`. A durable task that runs the agent more than once publishes a terminal event
    per run, so this iterator ends at the first; resume from its `cursor` to read the next run's events.
    """

    def __init__(
        self,
        subscribe: Callable[[str | None], AsyncIterator[StreamEvent]],
        *,
        output_type: type[OutputDataT],
        cursor: str | None = None,
    ) -> None:
        self._subscribe = subscribe
        self._subscription = subscribe(cursor)
        self._adapter = cast(
            TypeAdapter[_StreamMessage[OutputDataT]], TypeAdapter(cast(Any, _StreamMessage)[output_type])
        )
        self._seen: set[str] = set()
        self._cursor = cursor
        self._result: AgentRunResult[OutputDataT] | None = None
        self._error: str | None = None
        self._done = False

    @property
    def cursor(self) -> str | None:
        """The cursor of the last message read, to resume a subscription right after it."""
        return self._cursor

    @property
    def result(self) -> AgentRunResult[OutputDataT] | None:
        """The run's result, once the terminal event has been yielded."""
        return self._result

    @property
    def error(self) -> str | None:
        """Why the run failed, if the iterator ended without a result because it did.

        The durable task's own result says how it ended, including a failure after the run published its result.
        """
        return self._error

    def __aiter__(self) -> HatchetAgentRunEvents[OutputDataT]:
        return self

    async def __anext__(self) -> AgentStreamEvent | AgentRunResultEvent[OutputDataT]:
        if self._done:
            raise StopAsyncIteration
        while True:
            try:
                item = await anext(self._subscription)
            except StopAsyncIteration:
                # The server hangs up a subscription that has been idle, like one waiting on a long tool call, before
                # the run ends: pick it up again after the last message read.
                self._subscription = self._subscribe(self._cursor)
                continue
            self._cursor = item.cursor
            message = self._adapter.validate_json(item.payload)
            if message.id in self._seen:
                # A replayed durable task published this event again.
                continue
            self._seen.add(message.id)
            if message.error is not None:
                self._error = message.error
                self._done = True
                await self.aclose()
                raise StopAsyncIteration
            event = message.event
            assert event is not None
            if isinstance(event, AgentRunResultEvent):
                self._result = event.result
                self._done = True
                await self.aclose()
            return event

    async def __aenter__(self) -> HatchetAgentRunEvents[OutputDataT]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Stop the underlying subscription; reaching the terminal event does this for you."""
        self._done = True
        await cast(Any, self._subscription).aclose()
