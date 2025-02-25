from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..exceptions import FallbackExceptionGroup, ModelHTTPError
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage, ModelResponse
    from ..settings import ModelSettings
    from ..usage import Usage


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
        default_model: Model | KnownModelName,
        *fallback_models: Model | KnownModelName,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelHTTPError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: A callable or tuple of exceptions that should trigger a fallback.
        """
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]
        self._model_name = f'FallBackModel[{", ".join(model.model_name for model in self.models)}]'

        if isinstance(fallback_on, tuple):
            self._fallback_on = _default_fallback_condition_factory(fallback_on)
        else:
            self._fallback_on = fallback_on

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Try each model in sequence until one succeeds.

        In case of failure, raise a FallbackExceptionGroup with all exceptions.
        """
        exceptions: list[Exception] = []

        for model in self.models:
            try:
                return await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds."""
        exceptions: list[Exception] = []

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters)
                    )
                except Exception as exc:
                    if self._fallback_on(exc):
                        exceptions.append(exc)
                        continue
                    raise exc
                yield response
                return

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider, n/a for fallback models."""
        return None


def _default_fallback_condition_factory(exceptions: tuple[type[Exception], ...]) -> Callable[[Exception], bool]:
    """Create a default fallback condition for the given exceptions."""

    def fallback_condition(exception: Exception) -> bool:
        return isinstance(exception, exceptions)

    return fallback_condition
