from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import ModelResponseStreamEvent, PartEndEvent
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse, ModelResponsePart
    from ..settings import ModelSettings

FallbackOnResponse = Callable[['ModelResponse', list['ModelMessage']], bool]
FallbackOnPart = Callable[['ModelResponsePart', list['ModelMessage']], bool]


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _model_name: str = field(repr=False)
    _fallback_on: Callable[[Exception], bool]
    _fallback_on_response: FallbackOnResponse | None
    _fallback_on_part: FallbackOnPart | None

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelAPIError,),
        fallback_on_response: FallbackOnResponse | None = None,
        fallback_on_part: FallbackOnPart | None = None,
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: A callable or tuple of exceptions that should trigger a fallback.
            fallback_on_response: A callable that inspects the model response and message history,
                returning `True` if fallback should be triggered. This enables fallback based on
                response content (e.g., a builtin tool indicating failure) rather than exceptions.
            fallback_on_part: A callable that inspects each model response part during streaming,
                returning `True` if fallback should be triggered. This enables early abort of
                streaming when a failure condition is detected (e.g., a builtin tool failure in
                the first chunk). Only applies to streaming requests.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        if isinstance(fallback_on, tuple):
            self._fallback_on = _default_fallback_condition_factory(fallback_on)
        else:
            self._fallback_on = fallback_on

        self._fallback_on_response = fallback_on_response
        self._fallback_on_part = fallback_on_part

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'fallback:{",".join(model.model_name for model in self.models)}'

    @property
    def system(self) -> str:
        return f'fallback:{",".join(model.system for model in self.models)}'

    @property
    def base_url(self) -> str | None:
        return self.models[0].base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try each model in sequence until one succeeds.

        In case of failure, raise a FallbackExceptionGroup with all exceptions.
        """
        exceptions: list[Exception] = []
        response_rejections: int = 0

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if self._fallback_on_response is not None and self._fallback_on_response(response, messages):
                response_rejections += 1
                continue

            self._set_span_attributes(model, prepared_parameters)
            return response

        _raise_fallback_exception_group(exceptions, response_rejections)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds."""
        exceptions: list[Exception] = []
        response_rejections: int = 0
        part_rejections: int = 0

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if self._fallback_on(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                if self._fallback_on_part is not None:
                    buffered_events: list[ModelResponseStreamEvent] = []
                    should_fallback = False

                    async for event in response:
                        buffered_events.append(event)
                        if isinstance(event, PartEndEvent) and self._fallback_on_part(event.part, messages):
                            should_fallback = True
                            break

                    if should_fallback:
                        part_rejections += 1
                        continue

                    if self._fallback_on_response is not None and self._fallback_on_response(response.get(), messages):
                        response_rejections += 1
                        continue

                    self._set_span_attributes(model, prepared_parameters)
                    yield BufferedStreamedResponse(_wrapped=response, _buffered_events=buffered_events)
                    return

                elif self._fallback_on_response is not None:
                    buffered_events = []
                    async for event in response:
                        buffered_events.append(event)

                    if self._fallback_on_response(response.get(), messages):
                        response_rejections += 1
                        continue

                    self._set_span_attributes(model, prepared_parameters)
                    yield BufferedStreamedResponse(_wrapped=response, _buffered_events=buffered_events)
                    return

                self._set_span_attributes(model, prepared_parameters)
                yield response
                return

        _raise_fallback_exception_group(exceptions, response_rejections, part_rejections)

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters) -> None:
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span.set_attributes(
                        {
                            **InstrumentedModel.model_attributes(model),
                            **InstrumentedModel.model_request_parameters_attributes(model_request_parameters),
                        }
                    )


def _default_fallback_condition_factory(exceptions: tuple[type[Exception], ...]) -> Callable[[Exception], bool]:
    """Create a default fallback condition for the given exceptions."""

    def fallback_condition(exception: Exception) -> bool:
        return isinstance(exception, exceptions)

    return fallback_condition


def _raise_fallback_exception_group(
    exceptions: list[Exception], response_rejections: int, part_rejections: int = 0
) -> NoReturn:
    """Raise a FallbackExceptionGroup combining exceptions and rejections."""
    all_errors: list[Exception] = list(exceptions)
    if part_rejections > 0:
        all_errors.append(RuntimeError(f'{part_rejections} model(s) rejected by fallback_on_part during streaming'))
    if response_rejections > 0:
        all_errors.append(RuntimeError(f'{response_rejections} model response(s) rejected by fallback_on_response'))

    if all_errors:
        raise FallbackExceptionGroup('All models from FallbackModel failed', all_errors)
    else:
        raise FallbackExceptionGroup(
            'All models from FallbackModel failed',
            [RuntimeError('No models available')],
        )  # pragma: no cover


@dataclass
class BufferedStreamedResponse(StreamedResponse):
    """A StreamedResponse wrapper that replays buffered events."""

    _wrapped: StreamedResponse
    _buffered_events: list[ModelResponseStreamEvent]

    model_request_parameters: ModelRequestParameters = field(init=False)

    def __post_init__(self) -> None:
        self.model_request_parameters = self._wrapped.model_request_parameters
        self._parts_manager = self._wrapped._parts_manager
        self._usage = self._wrapped._usage
        self.final_result_event = self._wrapped.final_result_event
        self.provider_response_id = self._wrapped.provider_response_id
        self.provider_details = self._wrapped.provider_details
        self.finish_reason = self._wrapped.finish_reason
        self._event_iterator = None  # reset so __aiter__ uses _get_event_iterator()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for event in self._buffered_events:
            yield event

    @property
    def model_name(self) -> str:
        return self._wrapped.model_name

    @property
    def provider_name(self) -> str | None:
        return self._wrapped.provider_name

    @property
    def timestamp(self) -> datetime:
        return self._wrapped.timestamp
