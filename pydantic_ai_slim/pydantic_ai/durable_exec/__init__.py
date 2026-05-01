"""Building blocks for writing durable-execution capabilities.

These helpers are re-exported here so third-party packages can implement
`AbstractCapability` subclasses that route model requests through external
durable execution systems (Temporal, DBOS, Prefect, ...) without reaching
into Pydantic AI's private modules.

The built-in capabilities live in submodules: `pydantic_ai.durable_exec.temporal`,
`pydantic_ai.durable_exec.dbos`, and `pydantic_ai.durable_exec.prefect`.
"""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai._agent_graph import call_model, open_model_stream
from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

__all__ = [
    'StreamedActivityResult',
    'call_model',
    'disable_threads',
    'open_model_stream',
    'process_event_stream',
]


@dataclass
class StreamedActivityResult:
    """Bundle returned across an activity/step/task boundary in durable-execution flows.

    Carries both the final `ModelResponse` and the events emitted by the model
    stream while the chain consumed it inside the boundary, so the workflow side
    can replay the events through any per-run `event_stream_handler`.
    """

    response: ModelResponse
    events: list[ModelResponseStreamEvent]


async def process_event_stream(
    run_context: RunContext[Any],
    request_context: ModelRequestContext,
    stream: AsyncIterable[AgentStreamEvent],
    handler: EventStreamHandler[Any] | None = None,
) -> None:
    """Run the capability chain's `wrap_run_event_stream` hooks against a live model stream.

    Use from inside a durable-execution boundary (Temporal activity, DBOS step,
    Prefect task) to make sure capabilities like `ProcessEventStream` see real,
    in-time-order events rather than synthetic events replayed on the workflow side.

    Captures the events emerging from the chain into `request_context` so the
    outer agent loop can replay them through any per-run `event_stream_handler`
    (a runtime handler can't cross the activity boundary, so without this it
    would silently miss the model events).

    Marks `request_context` as having had the chain run (signals the outer agent
    loop to skip re-firing on the replay, which would double-emit hook side
    effects like OTel spans). The helper is the only public path that sets that
    flag — durability capabilities should always go through this helper.

    Args:
        run_context: The current agent run context. The capability chain is read
            from `run_context.root_capability`.
        request_context: The model request context. Mutated as a side effect to
            mark the chain as applied and stash the captured events.
        stream: The live model stream (a `StreamedResponse` or any async iterable
            of `AgentStreamEvent`).
        handler: Optional event stream handler to invoke against the wrapped
            stream. When `None`, the wrapped stream is drained without observation.
    """
    wrapped = (
        run_context.root_capability.wrap_run_event_stream(run_context, stream=stream)
        if run_context.root_capability is not None
        else stream
    )

    captured: list[AgentStreamEvent] = []

    async def teed() -> AsyncIterator[AgentStreamEvent]:
        async for event in wrapped:
            captured.append(event)
            yield event

    if handler is not None:
        await handler(run_context, teed())
    else:
        async for _ in teed():
            pass
    # The model-request path only produces ModelResponseStreamEvent (the chain
    # wraps but doesn't change the type); cast at the boundary to satisfy the
    # typed buffer field on `ModelRequestContext`.
    request_context._capabilities_already_applied = True  # pyright: ignore[reportPrivateUsage]
    request_context._buffered_stream_events = cast(list[ModelResponseStreamEvent], captured)  # pyright: ignore[reportPrivateUsage]
