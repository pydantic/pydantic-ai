from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from genai_prices.types import PriceCalculation
from opentelemetry.util.types import AttributeValue

from pydantic_ai._instrumentation import (
    ANY_ADAPTER,
    GEN_AI_REQUEST_MODEL_ATTRIBUTE,
    CostCalculationFailedWarning,
)
from pydantic_ai.models.instrumented import InstrumentationSettings

from .base import ImageGenerationInput, ImageGenerationModel
from .result import ImageGenerationResult
from .settings import ImageGenerationSettings
from .wrapper import WrapperImageGenerationModel

__all__ = 'InstrumentedImageGenerationModel', 'instrument_image_generation_model'

GEN_AI_PROVIDER_NAME_ATTRIBUTE = 'gen_ai.provider.name'


def instrument_image_generation_model(
    model: ImageGenerationModel, instrument: InstrumentationSettings | bool
) -> ImageGenerationModel:
    """Instrument an image generation model with OpenTelemetry/logfire."""
    if instrument and not isinstance(model, InstrumentedImageGenerationModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedImageGenerationModel(model, instrument)

    return model


@dataclass(init=False)
class InstrumentedImageGenerationModel(WrapperImageGenerationModel):
    """Image generation model which wraps another model for OpenTelemetry instrumentation."""

    instrumentation_settings: InstrumentationSettings
    """Instrumentation settings for this model."""

    def __init__(
        self,
        wrapped: ImageGenerationModel | str,
        options: InstrumentationSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.instrumentation_settings = options or InstrumentationSettings()

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        prompt, images, settings = self.prepare_generate(prompt, images=images, settings=settings)
        with self._instrument(prompt, images, settings) as finish:
            result = await super().generate(prompt, images=images, settings=settings)
            finish(result)
            return result

    @contextmanager
    def _instrument(
        self,
        prompt: str,
        images: Sequence[ImageGenerationInput],
        settings: ImageGenerationSettings | None,
    ) -> Generator[Callable[[ImageGenerationResult], None]]:
        operation = 'image_generation'
        span_name = f'{operation} {self.model_name}'

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            **self.model_attributes(self.wrapped),
            'prompt_length': len(prompt),
            'input_image_count': len(images),
        }

        if settings:
            attributes['image_generation_settings'] = json.dumps(self.serialize_any(settings))

        if self.instrumentation_settings.include_content:
            attributes['prompt'] = prompt

        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    'prompt_length': {'type': 'integer'},
                    'input_image_count': {'type': 'integer'},
                    'image_generation_settings': {'type': 'object'},
                    'image_count': {'type': 'integer'},
                    **({'prompt': {'type': 'string'}} if self.instrumentation_settings.include_content else {}),
                },
            }
        )

        record_metrics: Callable[[], None] | None = None
        try:
            with self.instrumentation_settings.tracer.start_as_current_span(span_name, attributes=attributes) as span:

                def finish(result: ImageGenerationResult):
                    provider_name = str(attributes[GEN_AI_PROVIDER_NAME_ATTRIBUTE])
                    request_model = str(attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE])
                    response_model = result.model_name or request_model
                    price_calculation = self._price_calculation(result)
                    nonlocal record_metrics
                    record_metrics = self._metric_recorder(
                        provider_name, request_model, response_model, result, price_calculation
                    )

                    if not span.is_recording():
                        return  # pragma: lax no cover

                    span.set_attributes(self._response_attributes(result, response_model, price_calculation))

                yield finish
        finally:
            if record_metrics:
                record_metrics()

    def _price_calculation(self, result: ImageGenerationResult) -> PriceCalculation | None:
        try:
            return result.cost()
        except LookupError:
            return None
        except Exception as e:  # pragma: no cover
            warnings.warn(f'Failed to get cost from response: {type(e).__name__}: {e}', CostCalculationFailedWarning)
            return None

    def _metric_recorder(
        self,
        provider_name: str,
        request_model: str,
        response_model: str,
        result: ImageGenerationResult,
        price_calculation: PriceCalculation | None,
    ) -> Callable[[], None]:
        def record_metrics() -> None:
            metric_attributes = {
                GEN_AI_PROVIDER_NAME_ATTRIBUTE: provider_name,
                'gen_ai.operation.name': 'image_generation',
                GEN_AI_REQUEST_MODEL_ATTRIBUTE: request_model,
                'gen_ai.response.model': response_model,
            }
            if result.usage.input_tokens:
                self.instrumentation_settings.tokens_histogram.record(
                    result.usage.input_tokens,
                    {**metric_attributes, 'gen_ai.token.type': 'input'},
                )
            if result.usage.output_tokens:
                self.instrumentation_settings.tokens_histogram.record(
                    result.usage.output_tokens,
                    {**metric_attributes, 'gen_ai.token.type': 'output'},
                )
            if price_calculation is not None:
                self.instrumentation_settings.cost_histogram.record(
                    float(price_calculation.total_price),
                    metric_attributes,
                )

        return record_metrics

    @staticmethod
    def _response_attributes(
        result: ImageGenerationResult, response_model: str, price_calculation: PriceCalculation | None
    ) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            **result.usage.opentelemetry_attributes(),
            'gen_ai.response.model': response_model,
            'image_count': len(result.images),
        }

        for image_index, image in enumerate(result.images):
            prefix = f'image.{image_index}'
            if image.size:
                attributes[f'{prefix}.size'] = image.size
            if image.quality:
                attributes[f'{prefix}.quality'] = image.quality
            if image.output_format:
                attributes[f'{prefix}.output_format'] = image.output_format
            if image.background:
                attributes[f'{prefix}.background'] = image.background
            attributes[f'{prefix}.media_type'] = image.content.media_type

        if price_calculation:
            attributes['operation.cost'] = float(price_calculation.total_price)

        if result.provider_response_id is not None:
            attributes['gen_ai.response.id'] = result.provider_response_id

        return attributes

    @staticmethod
    def model_attributes(model: ImageGenerationModel) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            GEN_AI_PROVIDER_NAME_ATTRIBUTE: model.system,
            GEN_AI_REQUEST_MODEL_ATTRIBUTE: model.model_name,
        }
        if base_url := model.base_url:
            try:
                parsed = urlparse(base_url)
            except Exception:  # pragma: no cover
                pass
            else:
                if parsed.hostname:
                    attributes['server.address'] = parsed.hostname
                if parsed.port:
                    attributes['server.port'] = parsed.port  # pragma: no cover

        return attributes

    @staticmethod
    def serialize_any(value: Any) -> str:
        try:
            return ANY_ADAPTER.dump_python(value, mode='json')
        except Exception:  # pragma: no cover
            try:
                return str(value)
            except Exception as e:
                return f'Unable to serialize: {e}'
