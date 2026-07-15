"""Internal building blocks shared by the bundled `durable_exec` integrations.

Not public API. The surface third-party durable-execution integrations should
build on is the wrapper hierarchy ([`WrapperAgent`][pydantic_ai.agent.WrapperAgent]
/ [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] /
[`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset]) plus the
[`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] hooks.
A first-class integration surface for runtimes is tracked as
[#5477](https://github.com/pydantic/pydantic-ai/issues/5477); until then these
helpers are reserved for the bundled `temporal`, `dbos`, and `prefect`
integrations.
"""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import RunContext

__all__ = [
    'StreamedActivityResult',
    'disable_threads',
    'process_event_stream',
    'unwrap_model',
]


def unwrap_model(model: Model) -> Model:
    """Strip [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] layers to the underlying model.

    Durability capabilities close over the agent's construction-time model and need to
    detect when a *different* model is supplied at run time (via `run(model=...)` /
    `override(model=...)`). Comparing `model_id` strings is too coarse — two distinct
    instances (e.g. the same model name on different providers, base URLs, or API keys)
    share a `model_id` — while comparing the wrapped instances directly is too strict,
    because an [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability
    wraps the model in an [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]
    before the request runs. Unwrapping both sides and comparing by identity gets it
    right: a normal run's instrumented model unwraps to the same underlying instance,
    while a genuine runtime override unwraps to a different one.
    """
    while isinstance(model, WrapperModel):
        model = model.wrapped
    return model


@dataclass
class StreamedActivityResult:
    """Bundle returned across an activity/step/task boundary in durable-execution flows.

    Carries both the final `ModelResponse` and the events emitted by the model
    stream while the chain consumed it inside the boundary, so the workflow side
    can replay the events through any per-run `event_stream_handler`. This is the
    serializable counterpart of a
    [`CompletedStreamedResponse`][pydantic_ai.models.CompletedStreamedResponse]
    with buffered events: the workflow side calls `apply_to()` and the agent loop
    rebuilds the equivalent stream from the request context.
    """

    response: ModelResponse
    events: list[ModelResponseStreamEvent]

    def apply_to(self, request_context: ModelRequestContext) -> ModelResponse:
        """Transfer this result onto the workflow-side `request_context` and return the response.

        Marks the chain as already applied (the hooks ran inside the boundary) and
        stashes the captured events so the agent loop replays them through any
        per-run `event_stream_handler`. The single place the workflow side touches
        the coordination fields.
        """
        request_context._hooks_already_applied = True  # pyright: ignore[reportPrivateUsage]
        buffered = request_context._buffered_stream_events  # pyright: ignore[reportPrivateUsage]
        request_context._buffered_stream_events = [*(buffered or []), *self.events]  # pyright: ignore[reportPrivateUsage]
        return self.response


async def process_event_stream(
    *,
    run_context: RunContext[Any],
    request_context: ModelRequestContext,
    stream: AsyncIterable[AgentStreamEvent],
    handler: EventStreamHandler[Any] | None = None,
) -> list[ModelResponseStreamEvent]:
    """Run the capability chain's `wrap_run_event_stream` hooks against a live model stream.

    Use from inside a durable-execution boundary (Temporal activity, DBOS step,
    Prefect task) to make sure capabilities like `ProcessEventStream` see real,
    in-time-order events rather than synthetic events replayed on the workflow side.

    Returns the events that emerged from the chain (also stashed on
    `request_context`) so the boundary can ship them back to the workflow side,
    where they're replayed through any per-run `event_stream_handler` (a runtime
    handler can't cross the activity boundary, so without this it would silently
    miss the model events).

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
    events = cast(list[ModelResponseStreamEvent], captured)
    request_context._hooks_already_applied = True  # pyright: ignore[reportPrivateUsage]
    request_context._buffered_stream_events = events  # pyright: ignore[reportPrivateUsage]
    return events
