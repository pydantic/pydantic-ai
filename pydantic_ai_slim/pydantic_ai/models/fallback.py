from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..exceptions import ModelStatusError
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
        self.models = [infer_model(model) for model in [default_model, *fallback_models]]  # pyright: ignore[reportArgumentType]

        self._model_name = f'FallBackModel[{", ".join(model.model_name for model in self.models)}]'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Try each model in sequence until one succeeds."""
        errors: list[ModelStatusError] = []

        for model in self.models:
            try:
                return await model.request(messages, model_settings, model_request_parameters)
            except ModelStatusError as exc_info:
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
        errors: list[ModelStatusError] = []

        for model in self.models:
            try:
                async with model.request_stream(messages, model_settings, model_request_parameters) as response:
                    yield response
                    return
            except ModelStatusError as exc_info:
                errors.append(exc_info)
                continue

        raise RuntimeError(f'All fallback models failed: {errors}')
