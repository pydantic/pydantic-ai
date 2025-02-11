from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    _system: str | None = field(default=None, repr=False)

    def __init__(
        self,
        default_model: Model | KnownModelName,
        *fallback_models: Model | KnownModelName,
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
        """
        # TODO: should we do this lazily?
        default_model_ = default_model if isinstance(default_model, Model) else infer_model(default_model)
        fallback_models_ = [model if isinstance(model, Model) else infer_model(model) for model in fallback_models]

        self.models = [default_model_, *fallback_models_]

        self._model_name = f'FallBackModel[{", ".join(model.model_name for model in self.models)}]'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Try each model in sequence until one succeeds."""
        errors: list[Exception] = []

        for model in self.models:
            try:
                return await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc_info:
                errors.append(exc_info)
                continue

        raise RuntimeError(f'All fallback models failed: {errors}')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds."""
        errors: list[Exception] = []

        for model in self.models:
            try:
                async with model.request_stream(messages, model_settings, model_request_parameters) as response:
                    yield response
            except Exception as exc_info:
                errors.append(exc_info)
                continue

        raise RuntimeError(f'All fallback models failed: {errors}')
