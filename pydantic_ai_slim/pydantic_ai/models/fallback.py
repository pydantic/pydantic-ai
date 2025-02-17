from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..exceptions import FallbackModelFailure, ModelStatusError
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
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]
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

        raise FallbackModelFailure(errors=errors)

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
            async with AsyncExitStack() as stack:
                try:
                    response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters)
                    )
                except ModelStatusError as exc_info:
                    errors.append(exc_info)
                    continue
                yield response
                return

        raise FallbackModelFailure(errors=errors)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider, n/a for fallback models."""
        return None
