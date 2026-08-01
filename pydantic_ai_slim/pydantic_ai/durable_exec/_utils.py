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
from typing import Any, TypeAlias, TypeVar

from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

__all__ = [
    'DurableModel',
    'SegmentExecutor',
    'StreamedActivityResult',
    'disable_threads',
    'capture_event_stream',
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

    Carries both the final `ModelResponse` and the raw events captured from the live
    model stream inside the boundary. The chain consumes the replayed events workflow-side.
    This is the serializable counterpart of a
    [`CompletedStreamedResponse`][pydantic_ai.models.CompletedStreamedResponse].
    """

    response: ModelResponse
    events: list[ModelResponseStreamEvent]


_ResultT = TypeVar('_ResultT')

SegmentExecutor: TypeAlias = Callable[[ModelRequestContext], Awaitable[_ResultT]]
"""Executes one model-request segment inside an engine's durable unit (activity/step/task).

Receives a fresh `ModelRequestContext` carrying the segment's messages/settings/parameters
(each continuation segment of a suspended response differs from the original request).
"""


class DurableModel(WrapperModel):
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
        request_segment: SegmentExecutor[ModelResponse],
        request_stream_segment: SegmentExecutor[StreamedActivityResult],
        cancel_suspended_response_segment: Callable[[ModelResponse], Awaitable[None]],
    ):
        super().__init__(wrapped)
        self._request_segment = request_segment
        self._request_stream_segment = request_stream_segment
        self._cancel_suspended_response_segment = cancel_suspended_response_segment

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        segment_context = ModelRequestContext(
            model=self.wrapped,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        return await self._request_segment(segment_context)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[CompletedStreamedResponse]:
        segment_context = ModelRequestContext(
            model=self.wrapped,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        result = await self._request_stream_segment(segment_context)
        yield CompletedStreamedResponse(
            result.response,
            model_request_parameters=model_request_parameters,
            replay_events=result.events,
        )

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await self._cancel_suspended_response_segment(response)


async def capture_event_stream(
    *,
    run_context: RunContext[Any],
    stream: AsyncIterable[ModelResponseStreamEvent],
    handler: EventStreamHandler[Any] | None,
) -> list[ModelResponseStreamEvent]:
    """Capture a live model stream inside a durable-execution boundary.

    If a handler is provided, it consumes the live stream inside the boundary. Any
    events it leaves unconsumed are drained and captured. The returned raw events are
    shipped back to the workflow, where the capability chain and any per-run handler
    consume the replay.

    Args:
        run_context: The current agent run context.
        stream: The live model stream.
        handler: Optional handler to run inside the durable boundary.
    """
    captured: list[ModelResponseStreamEvent] = []

    async def teed() -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in stream:
            captured.append(event)
            yield event

    teed_stream = teed()
    if handler is not None:
        await handler(run_context, teed_stream)

    async for _ in teed_stream:
        pass
    return captured
