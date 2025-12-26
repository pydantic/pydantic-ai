from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from opentelemetry.util.types import AttributeValue

from pydantic_ai.models.instrumented import (
    ANY_ADAPTER,
    GEN_AI_REQUEST_MODEL_ATTRIBUTE,
    CostCalculationFailedWarning,
    InstrumentationSettings,
)

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings
from .wrapper import WrapperEmbeddingModel

__all__ = 'instrument_embedding_model', 'InstrumentedEmbeddingModel'

GEN_AI_PROVIDER_NAME_ATTRIBUTE = 'gen_ai.provider.name'


def instrument_embedding_model(model: EmbeddingModel, instrument: InstrumentationSettings | bool) -> EmbeddingModel:
    """Instrument an embedding model with OpenTelemetry/logfire."""
    if instrument and not isinstance(model, InstrumentedEmbeddingModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedEmbeddingModel(model, instrument)

    return model


@dataclass(init=False)
class InstrumentedEmbeddingModel(WrapperEmbeddingModel):
    """Embedding model which wraps another model so that requests are instrumented with OpenTelemetry.

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    instrumentation_settings: InstrumentationSettings
    """Instrumentation settings for this model."""

    def __init__(
        self,
        wrapped: EmbeddingModel | str,
        options: InstrumentationSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.instrumentation_settings = options or InstrumentationSettings()

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        with self._instrument(inputs, input_type, settings) as finish:
            result = await super().embed(inputs, input_type=input_type, settings=settings)
            finish(result)
            return result

    @contextmanager
    def _instrument(
        self,
        inputs: list[str],
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None,
    ) -> Iterator[Callable[[EmbeddingResult], None]]:
        operation = 'embeddings'
        span_name = f'{operation} {self.model_name}'

        inputs_count = len(inputs)

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            **self.model_attributes(self.wrapped),
            'input_type': input_type,
            'inputs_count': inputs_count,
        }

        if settings:
            attributes['embedding_settings'] = json.dumps(self.serialize_any(settings))

        if self.instrumentation_settings.include_content:
            attributes['inputs'] = json.dumps(inputs)

        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    'input_type': {'type': 'string'},
                    'inputs_count': {'type': 'integer'},
                    'embedding_settings': {'type': 'object'},
                    **(
                        {'inputs': {'type': ['array']}, 'embeddings': {'type': 'array'}}
                        if self.instrumentation_settings.include_content
                        else {}
                    ),
                },
            }
        )

        record_metrics: Callable[[], None] | None = None
        try:
            with self.instrumentation_settings.tracer.start_as_current_span(span_name, attributes=attributes) as span:

                def finish(result: EmbeddingResult):
                    # Prepare metric recording closure first so metrics are recorded
                    # even if the span is not recording.
                    provider_name = attributes[GEN_AI_PROVIDER_NAME_ATTRIBUTE]
                    request_model = attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE]
                    response_model = result.model_name or request_model
                    price_calculation = None

                    def _record_metrics():
                        token_attributes = {
                            GEN_AI_PROVIDER_NAME_ATTRIBUTE: provider_name,
                            'gen_ai.operation.name': operation,
                            GEN_AI_REQUEST_MODEL_ATTRIBUTE: request_model,
                            'gen_ai.response.model': response_model,
                            'gen_ai.token.type': 'input',
                        }
                        tokens = result.usage.input_tokens or 0
                        if tokens:  # pragma: no branch
                            self.instrumentation_settings.tokens_histogram.record(tokens, token_attributes)
                            if price_calculation is not None:
                                self.instrumentation_settings.cost_histogram.record(
                                    float(getattr(price_calculation, 'input_price', 0.0)),
                                    token_attributes,
                                )

                    nonlocal record_metrics
                    record_metrics = _record_metrics

                    if not span.is_recording():
                        return

                    attributes_to_set: dict[str, AttributeValue] = {
                        **result.usage.opentelemetry_attributes(),
                        'gen_ai.response.model': response_model,
                    }

                    try:
                        price_calculation = result.cost()
                    except LookupError:
                        # The cost of this provider/model is unknown, which is common.
                        pass
                    except Exception as e:  # pragma: no cover
                        warnings.warn(
                            f'Failed to get cost from response: {type(e).__name__}: {e}', CostCalculationFailedWarning
                        )
                    else:
                        attributes_to_set['operation.cost'] = float(price_calculation.total_price)

                    embeddings = result.embeddings
                    if embeddings:  # pragma: no branch
                        attributes_to_set['gen_ai.embeddings.dimension.count'] = len(embeddings[0])
                        if self.instrumentation_settings.include_content:
                            attributes['embeddings'] = json.dumps(embeddings)

                    if result.provider_response_id is not None:
                        attributes_to_set['gen_ai.response.id'] = result.provider_response_id

                    span.set_attributes(attributes_to_set)

                yield finish
        finally:
            if record_metrics:  # pragma: no branch
                # Record metrics after the span finishes to avoid duplication.
                record_metrics()

    @staticmethod
    def model_attributes(model: EmbeddingModel) -> dict[str, AttributeValue]:
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
                if parsed.hostname:  # pragma: no branch
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
