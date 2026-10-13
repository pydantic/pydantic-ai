from __future__ import annotations

import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass

from genai_prices.types import PriceCalculation
from opentelemetry.util.types import AttributeValue

from pydantic_ai._genai_prices import best_effort_price
from pydantic_ai._instrumentation import (
    GEN_AI_PROVIDER_NAME_ATTRIBUTE,
    GEN_AI_REQUEST_MODEL_ATTRIBUTE,
    record_uncaught_errors,
    serialize_any,
    server_attributes,
)
from pydantic_ai.models.instrumented import InstrumentationSettings

from .base import TranscriptionInput, TranscriptionModel
from .result import TranscriptionResult
from .settings import TranscriptionSettings
from .wrapper import WrapperTranscriptionModel

__all__ = 'InstrumentedTranscriptionModel', 'instrument_transcription_model'


def instrument_transcription_model(
    model: TranscriptionModel, instrument: InstrumentationSettings | bool
) -> TranscriptionModel:
    """Instrument a transcription model with OpenTelemetry/logfire."""
    if instrument and not isinstance(model, InstrumentedTranscriptionModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedTranscriptionModel(model, options=instrument)

    return model


@dataclass(init=False)
class InstrumentedTranscriptionModel(WrapperTranscriptionModel):
    """Transcription model which wraps another model so that requests are instrumented with OpenTelemetry.

    See the [Debugging and Monitoring guide](https://pydantic.dev/docs/ai/integrations/logfire/) for more info.
    """

    instrumentation_settings: InstrumentationSettings
    """Instrumentation settings for this model."""

    def __init__(
        self,
        wrapped: TranscriptionModel | str,
        *,
        options: InstrumentationSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.instrumentation_settings = options or InstrumentationSettings()

    async def transcribe(
        self, audio: TranscriptionInput, *, settings: TranscriptionSettings | None = None
    ) -> TranscriptionResult:
        settings = self.prepare_transcribe(audio, settings)
        with self._instrument(settings) as finish:
            result = await super().transcribe(audio, settings=settings)
            finish(result)
            return result

    @contextmanager
    def _instrument(self, settings: TranscriptionSettings) -> Generator[Callable[[TranscriptionResult], None]]:
        operation = 'transcription'
        span_name = f'{operation} {self.model_name}'

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            'gen_ai.output.type': 'text',
            **self.model_attributes(self.wrapped),
        }

        include_content = self.instrumentation_settings.include_content
        recorded_settings: dict[str, object] | None = None
        if self.instrumentation_settings.include_model_request_parameters:
            # `extra_headers` and `extra_body` can carry credentials, so they are never recorded.
            excluded = {'extra_headers', 'extra_body'}
            if not include_content:
                # A prompt is content: it can hold the transcript of the previous audio segment.
                excluded.add('openai_prompt')
            recorded_settings = {key: value for key, value in settings.items() if key not in excluded}
            if recorded_settings:
                attributes['transcription_settings'] = json.dumps(serialize_any(recorded_settings))

        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    **({'transcription_settings': {'type': 'object'}} if recorded_settings else {}),
                    'segment_count': {'type': 'integer'},
                    **({'transcript': {'type': 'string'}} if include_content else {}),
                },
            }
        )

        record_metrics: Callable[[], None] | None = None
        try:
            with (
                self.instrumentation_settings.tracer.start_as_current_span(
                    span_name, attributes=attributes, record_exception=False, set_status_on_exception=False
                ) as span,
                record_uncaught_errors(span, include_content=include_content),
            ):

                def finish(result: TranscriptionResult):
                    provider_name = str(attributes[GEN_AI_PROVIDER_NAME_ATTRIBUTE])
                    request_model = str(attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE])
                    response_model = result.model_name or request_model
                    price_calculation = best_effort_price(
                        result.usage,
                        model_name=result.model_name,
                        provider_api_url=result.provider_url,
                        provider_name=result.provider_name,
                        genai_request_timestamp=result.timestamp,
                    )
                    nonlocal record_metrics
                    record_metrics = self._metric_recorder(
                        operation, provider_name, request_model, response_model, result, price_calculation
                    )

                    if not span.is_recording():
                        return  # pragma: lax no cover

                    response_attributes = self._response_attributes(result, response_model, price_calculation)
                    if include_content:
                        response_attributes['transcript'] = result.text
                    span.set_attributes(response_attributes)

                yield finish
        finally:
            if record_metrics:
                record_metrics()

    def _metric_recorder(
        self,
        operation: str,
        provider_name: str,
        request_model: str,
        response_model: str,
        result: TranscriptionResult,
        price_calculation: PriceCalculation | None,
    ) -> Callable[[], None]:
        def record_metrics() -> None:
            metric_attributes = {
                GEN_AI_PROVIDER_NAME_ATTRIBUTE: provider_name,
                'gen_ai.operation.name': operation,
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
        result: TranscriptionResult, response_model: str, price_calculation: PriceCalculation | None
    ) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            **result.usage.opentelemetry_attributes(),
            'gen_ai.response.model': response_model,
            'segment_count': len(result.segments),
        }

        if result.usage.audio_seconds:
            attributes['audio_seconds'] = result.usage.audio_seconds

        if price_calculation:
            attributes['operation.cost'] = float(price_calculation.total_price)

        if result.provider_response_id is not None:
            attributes['gen_ai.response.id'] = result.provider_response_id

        return attributes

    @staticmethod
    def model_attributes(model: TranscriptionModel) -> dict[str, AttributeValue]:
        return {
            GEN_AI_PROVIDER_NAME_ATTRIBUTE: model.system,
            GEN_AI_REQUEST_MODEL_ATTRIBUTE: model.model_name,
            **server_attributes(model.base_url),
        }
