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

import sys
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

from pydantic_ai._utils import disable_threads
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models._request_timeout import RequestDeadline
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

__all__ = [
    'DurableModel',
    'SegmentExecutor',
    'StreamedActivityResult',
    'StreamedModelRequestResult',
    'disable_threads',
    'consume_event_stream',
    'managed_model_scope',
    'mark_streamed_before_failing',
    'reject_serving_wrapper',
    'streamed_before_failing',
    'unwrap_model',
]


@asynccontextmanager
async def managed_model_scope(model: Model, *, owned: bool) -> AsyncGenerator[Model]:
    """Context-manage a model when the current durable unit owns it.

    The model's `__aexit__` return value is deliberately discarded, so it cannot
    suppress an error raised by the durable unit's body.
    """
    if not owned:
        yield model
        return

    active_model = await model.__aenter__()
    try:
        yield active_model
    except BaseException:
        await model.__aexit__(*sys.exc_info())
        raise
    else:
        await model.__aexit__(None, None, None)


def reject_serving_wrapper(
    model: WrapperModel, wrap: Callable[[Model], Model], agent_class: str, durability_class: str
) -> Model:
    """Refuse a capability's model wrapper on a deprecated durable agent, which sends requests without applying it."""
    if wrap(model.wrapped) is model.wrapped:
        return model
    raise UserError(
        f'`{agent_class}` does not support capabilities that wrap the model, such as `BatchModelRequests`. '
        f'Use `{durability_class}` instead.'
    )


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


@dataclass(kw_only=True)
class StreamedActivityResult:
    """The streamed model request result recorded by durability capabilities before 2.60.

    Carries both the final `ModelResponse` and the raw events captured from the live
    model stream inside the boundary. New results are the bare `ModelResponse`; this
    shape is only decoded from histories, steps, and task results recorded earlier, whose
    events are replayed exactly as recorded.
    """

    response: ModelResponse
    events: list[ModelResponseStreamEvent]


StreamedModelRequestResult: TypeAlias = ModelResponse | StreamedActivityResult
"""What a streamed model request's durable unit returns: the final response, or a recorded `StreamedActivityResult`."""


_STREAMED_BEFORE_FAILING = '_pydantic_ai_streamed_before_failing'


def mark_streamed_before_failing(error: BaseException, response: ModelResponse) -> None:
    """Mark `error` as having ended a model stream in a durable unit after it produced output, with that output.

    The unit consumes the stream, delivering its events to any in-boundary event handler, before the run sees any
    of it. A failure partway through therefore reaches the run before the stream it opened, where it would read as
    a failure to open one: something to retry or fall back from, with no response to count the usage of. The mark
    lets the run treat it as the failure of a stream that had started, as it does outside durable execution.
    """
    setattr(error, _STREAMED_BEFORE_FAILING, response)


def streamed_before_failing(error: BaseException) -> ModelResponse | None:
    """What a durable unit's model stream produced before `error` ended it, if `mark_streamed_before_failing` marked it."""
    response = getattr(error, _STREAMED_BEFORE_FAILING, None)
    return response if isinstance(response, ModelResponse) else None


def _completed_stream(
    result: StreamedModelRequestResult, model_request_parameters: ModelRequestParameters
) -> CompletedStreamedResponse:
    """The stream the run consumes for what a durable unit's streamed model request returned."""
    if isinstance(result, StreamedActivityResult):
        response, replay_events = result.response, result.events
    else:
        # The durable unit consumed the live stream, delivering it to any in-boundary handler;
        # workflow-side, each part replays as one event carrying its full content.
        response, replay_events = result, True
    return CompletedStreamedResponse(
        response,
        model_request_parameters=model_request_parameters,
        replay_events=replay_events,
    )


class _StreamedThenFailed(CompletedStreamedResponse):
    """Replays what a stream produced in its durable unit, then raises the error that ended it there."""

    def __init__(
        self, response: ModelResponse, *, model_request_parameters: ModelRequestParameters, error: Exception
    ) -> None:
        super().__init__(response, model_request_parameters=model_request_parameters, replay_events=True)
        self._error = error

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in super()._get_event_iterator():
            yield event
        raise self._error


_ResultT = TypeVar('_ResultT')

SegmentExecutor: TypeAlias = Callable[[ModelRequestContext], Awaitable[_ResultT]]
"""Executes one model-request segment inside an engine's durable unit (activity/step/task).

Receives a fresh `ModelRequestContext` carrying the segment's messages/settings/parameters
(each continuation segment of a suspended response differs from the original request).
"""


class DurableModel(WrapperModel):
    """Dispatches each model-request segment through its own durable unit.

    The bundled durability capabilities install this before the wrapped request lifecycle,
    then repair it in their innermost `before_model_request` hook if an outer hook swaps the model, so the
    continuation loop (Anthropic `pause_turn`, OpenAI background mode) checkpoints every
    suspended segment durably and a failed segment retries alone, while everything else
    (`profile`, `settings`, `continuation_delay`, ...) is answered by the wrapped
    workflow-side model. Everything engine-specific lives in the four executors, each
    running one request / streamed request / cancellation inside the engine's
    activity, step, or task. Installing it before the before-hook chain also routes model-side
    compaction through the durable operation.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        request_segment: SegmentExecutor[ModelResponse],
        request_stream_segment: SegmentExecutor[StreamedModelRequestResult],
        compact_messages_segment: Callable[[ModelRequestContext, str | None], Awaitable[ModelResponse]],
        cancel_suspended_response_segment: Callable[[ModelResponse], Awaitable[None]],
    ):
        super().__init__(wrapped)
        self._request_segment = request_segment
        self._request_stream_segment = request_stream_segment
        self._compact_messages_segment = compact_messages_segment
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
        try:
            result = await self._request_stream_segment(segment_context)
        except Exception as error:
            if (partial := streamed_before_failing(error)) is None:
                raise
            # The stream failed after it produced output, so it's handed over as a stream that fails partway
            # through rather than one that failed to open.
            stream: CompletedStreamedResponse = _StreamedThenFailed(
                partial, model_request_parameters=model_request_parameters, error=error
            )
        else:
            stream = _completed_stream(result, model_request_parameters)
        yield stream

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await self._cancel_suspended_response_segment(response)

    def _start_request_deadline(self, model_settings: ModelSettings | None, *, stream: bool) -> RequestDeadline | None:
        # This side only gets a streamed segment once the durable unit has consumed it, so the unit enforces the
        # `first_event_timeout` deadline, and this side just `request_timeout`'s.
        return super()._start_request_deadline(model_settings, stream=False)

    def _with_serving_wrapper(self, wrap: Callable[[Model], Model]) -> Model:
        # The requests are sent from the durable unit, which wraps the model it rebuilds there, so this side only
        # wraps the model that answers `continuation_delay` and the like.
        if (wrapped := wrap(self.wrapped)) is self.wrapped:
            return self
        return DurableModel(
            wrapped,
            request_segment=self._request_segment,
            request_stream_segment=self._request_stream_segment,
            compact_messages_segment=self._compact_messages_segment,
            cancel_suspended_response_segment=self._cancel_suspended_response_segment,
        )

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        return await self._compact_messages_segment(request_context, instructions)


async def consume_event_stream(
    *,
    run_context: RunContext[Any],
    stream: AsyncIterable[ModelResponseStreamEvent],
    handler: EventStreamHandler[Any] | None,
) -> None:
    """Consume a live model stream inside a durable-execution boundary.

    If a handler is provided, it consumes the live stream inside the boundary. Any
    events it leaves unconsumed are drained, so the stream's final response is complete.
    The events aren't shipped back to the workflow, which replays the final response instead.

    Args:
        run_context: The current agent run context.
        stream: The live model stream.
        handler: Optional handler to run inside the durable boundary.
    """

    async def passthrough() -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in stream:
            yield event

    # One iterator, so draining resumes where the handler stopped.
    events = passthrough()
    if handler is not None:
        await handler(run_context, events)

    async for _ in events:
        pass
