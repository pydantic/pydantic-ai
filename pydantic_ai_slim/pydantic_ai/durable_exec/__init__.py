"""Building blocks for writing durable-execution capabilities.

These helpers are re-exported here so third-party packages can implement
`AbstractCapability` subclasses that route model requests through external
durable execution systems (Temporal, DBOS, Prefect, ...) without reaching
into Pydantic AI's private modules.

The built-in capabilities live in submodules: `pydantic_ai.durable_exec.temporal`,
`pydantic_ai.durable_exec.dbos`, and `pydantic_ai.durable_exec.prefect`.
"""

from collections.abc import AsyncIterable
from typing import Any

from pydantic_ai._agent_graph import call_model, open_model_stream
from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

__all__ = ['call_model', 'disable_threads', 'open_model_stream', 'process_event_stream']


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

    Marks `request_context` as having had the chain run (signals the outer agent
    loop to skip re-firing on the replay, which would double-emit hook side
    effects like OTel spans). The helper is the only public path that sets that
    flag — durability capabilities should always go through this helper.

    Args:
        run_context: The current agent run context. The capability chain is read
            from `run_context.root_capability`.
        request_context: The model request context. Mutated as a side effect to
            mark the chain as applied.
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
    if handler is not None:
        await handler(run_context, wrapped)
    else:
        async for _ in wrapped:
            pass
    request_context._capabilities_already_applied = True  # pyright: ignore[reportPrivateUsage]
