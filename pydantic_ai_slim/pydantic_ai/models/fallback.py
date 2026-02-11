from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from ..settings import ModelSettings


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _model_name: str = field(repr=False)
    _fallback_on: Callable[[Exception], bool]

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelAPIError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: A callable or tuple of exceptions that should trigger a fallback.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        if isinstance(fallback_on, tuple):
            self._fallback_on = _default_fallback_condition_factory(fallback_on)  # pyright: ignore[reportUnknownArgumentType]
        else:
            self._fallback_on = fallback_on

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

        If a previous response set `expects_continuation`, the request is routed directly
        to the model identified by the latest response identity, bypassing the fallback chain.
        """
        if continuation_route := _continuation_route(messages, self.models):
            model = continuation_route
            _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
            response = await model.request(messages, model_settings, model_request_parameters)
            self._set_span_attributes(model, prepared_parameters)
            return response

        exceptions: list[Exception] = []

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            self._set_span_attributes(model, prepared_parameters)
            return response

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds.

        If a previous response set `expects_continuation`, the request is routed directly
        to the model identified by the latest response identity, bypassing the fallback chain.
        """
        if continuation_route := _continuation_route(messages, self.models):
            model = continuation_route
            async with AsyncExitStack() as stack:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                streamed_response = await stack.enter_async_context(
                    model.request_stream(messages, model_settings, model_request_parameters, run_context)
                )
                self._set_span_attributes(model, prepared_parameters)
                yield streamed_response
                return

        exceptions: list[Exception] = []

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if self._fallback_on(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                self._set_span_attributes(model, prepared_parameters)
                yield streamed_response
                return

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters):
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


def _latest_model_response(messages: list[ModelMessage]) -> ModelResponse | None:
    """Return the most recent model response in the message history."""
    from ..messages import ModelResponse

    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return message
    return None


def _continuation_route(messages: list[ModelMessage], models: list[Model]) -> Model | None:
    """Resolve the pinned continuation model from the latest response identity."""
    if (latest_response := _latest_model_response(messages)) is None or not latest_response.expects_continuation:
        return None

    matches = [model for model in models if _model_matches_response(model, latest_response)]
    if len(matches) == 1:
        return matches[0]
    return None


def _model_matches_response(model: Model, response: ModelResponse) -> bool:
    """Return True if model identity matches the response identity."""
    matched = False
    if model_name := response.model_name:
        matched = True
        if model.model_name != model_name:
            return False
    if provider_name := response.provider_name:
        matched = True
        if model.system != provider_name:
            return False
    if provider_url := response.provider_url:
        matched = True
        if model.base_url != provider_url:
            return False
    return matched


def _default_fallback_condition_factory(exceptions: tuple[type[Exception], ...]) -> Callable[[Exception], bool]:
    """Create a default fallback condition for the given exceptions."""

    def fallback_condition(exception: Exception) -> bool:
        return isinstance(exception, exceptions)

    return fallback_condition
