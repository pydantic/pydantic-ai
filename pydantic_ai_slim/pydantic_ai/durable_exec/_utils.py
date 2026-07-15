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

from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, cast

from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

__all__ = [
    'DurableSegmentModel',
    'SegmentExecutor',
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


_ResultT = TypeVar('_ResultT')

SegmentExecutor: TypeAlias = Callable[
    [list[ModelMessage], 'ModelSettings | None', ModelRequestParameters], Awaitable[_ResultT]
]
"""Executes one model-request segment inside an engine's durable unit (activity/step/task)."""


class DurableSegmentModel(WrapperModel):
    """Dispatches each model-request segment through its own durable unit.

    The bundled durability capabilities swap this in for `request_context.model` in
    `wrap_model_request` and run the innermost handler in workflow/flow code, so the
    continuation loop (Anthropic `pause_turn`, OpenAI background mode) checkpoints every
    suspended segment durably and a failed segment retries alone, while everything else
    (`profile`, `settings`, `continuation_delay`, ...) is answered by the wrapped
    workflow-side model. Everything engine-specific lives in the three executors, each
    running one request / streamed request / cancellation inside the engine's
    activity, step, or task.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        request_context: ModelRequestContext,
        event_stream_handler: EventStreamHandler[Any] | None,
        request_segment: SegmentExecutor[ModelResponse],
        request_stream_segment: SegmentExecutor[StreamedActivityResult],
        cancel_suspended_response_segment: Callable[[ModelResponse], Awaitable[None]],
    ):
        super().__init__(wrapped)
        self._request_context = request_context
        self._event_stream_handler = event_stream_handler
        self._request_segment = request_segment
        self._request_stream_segment = request_stream_segment
        self._cancel_suspended_response_segment = cancel_suspended_response_segment

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self._event_stream_handler is not None:
            # The capability's construction-time handler must fire against the live stream
            # inside the boundary, so a non-streaming run still uses the streaming unit;
            # the buffered events land on the request context for workflow-side replay.
            result = await self._request_stream_segment(messages, model_settings, model_request_parameters)
            return result.apply_to(self._request_context)
        return await self._request_segment(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[CompletedStreamedResponse]:
        result = await self._request_stream_segment(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(
            result.response,
            model_request_parameters=model_request_parameters,
            events=result.events,
            # The chain's `wrap_run_event_stream` hooks already fired against the live
            # stream inside the boundary; the workflow-side replay must not re-fire them.
            hooks_already_applied=True,
        )

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await self._cancel_suspended_response_segment(response)


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
