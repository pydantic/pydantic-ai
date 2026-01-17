from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from .._run_context import RunContext
from ..messages import ModelMessage, ModelResponse
from ..profiles import ModelProfile
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model


@dataclass(init=False)
class WrapperModel(Model):
    """Model which wraps another model.

    Does nothing on its own, used as a base class.
    """

    wrapped: Model
    """The underlying model being wrapped."""

    def __init__(self, wrapped: Model | KnownModelName):
        super().__init__()
        self.wrapped = infer_model(wrapped)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        return await self.wrapped.count_tokens(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return self.wrapped.customize_request_parameters(model_request_parameters)  # pragma: no cover

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return self.wrapped.prepare_request(model_settings, model_request_parameters)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @cached_property
    def profile(self) -> ModelProfile:
        return self.wrapped.profile

    @property
    def settings(self) -> ModelSettings | None:
        """Get the settings from the wrapped model."""
        return self.wrapped.settings

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)
