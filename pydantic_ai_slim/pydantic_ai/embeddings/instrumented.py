from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from opentelemetry.util.types import AttributeValue

from pydantic_ai.models.instrumented import ANY_ADAPTER, CostCalculationFailedWarning, InstrumentationSettings

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings
from .wrapper import WrapperEmbeddingModel

if TYPE_CHECKING:
    pass

__all__ = 'instrument_embedding_model', 'InstrumentedEmbeddingModel'


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
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        with self._instrument(documents, input_type, settings) as finish:
            result = await self.wrapped.embed(documents, input_type=input_type, settings=settings)
            finish(result)
            return result

    @contextmanager
    def _instrument(
        self,
        documents: str | Sequence[str],
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None,
    ) -> Iterator[Callable[[EmbeddingResult], None]]:
        operation = 'embeddings'
        span_name = f'{operation} {self.model_name}'

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            'input_type': input_type,
            **self.model_attributes(self.wrapped),
        }

        if settings:
            attributes['embedding_settings'] = json.dumps(self.serialize_any(settings))

        if self.instrumentation_settings.include_content:
            attributes['gen_ai.prompt'] = documents if isinstance(documents, str) else json.dumps(list(documents))

        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    'embedding_settings': {'type': 'object'},
                    'gen_ai.prompt': {'type': ['string', 'array']},
                },
            }
        )

        try:
            with self.instrumentation_settings.tracer.start_as_current_span(span_name, attributes=attributes) as span:

                def finish(result: EmbeddingResult):
                    if not span.is_recording():
                        return

                    attributes_to_set: dict[str, AttributeValue] = {
                        **result.usage.opentelemetry_attributes(),
                        'gen_ai.response.model': result.model_name or self.model_name,
                    }

                    try:
                        price_calculation = result.cost()
                    except LookupError:
                        # The cost of this provider/model is unknown, which is common.
                        pass
                    except Exception as e:
                        warnings.warn(
                            f'Failed to get cost from response: {type(e).__name__}: {e}', CostCalculationFailedWarning
                        )
                    else:
                        attributes_to_set['operation.cost'] = float(price_calculation.total_price)

                    embeddings = result.embeddings
                    if embeddings:
                        attributes_to_set['gen_ai.embeddings.dimension.count'] = len(embeddings[0])

                    if result.provider_response_id is not None:
                        attributes_to_set['gen_ai.response.id'] = result.provider_response_id

                    span.set_attributes(attributes_to_set)

                    # TODO (DouweM): Record cost metric

                yield finish
        finally:
            pass

    @staticmethod
    def model_attributes(model: EmbeddingModel) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            'gen_ai.provider.name': model.system,
            'gen_ai.request.model': model.model_name,
        }
        if base_url := model.base_url:
            try:
                parsed = urlparse(base_url)
            except Exception:  # pragma: no cover
                pass
            else:
                if parsed.hostname:  # pragma: no branch
                    attributes['server.address'] = parsed.hostname
                if parsed.port:  # pragma: no branch
                    attributes['server.port'] = parsed.port

        return attributes

    @staticmethod
    def serialize_any(value: Any) -> str:
        try:
            return ANY_ADAPTER.dump_python(value, mode='json')
        except Exception:
            try:
                return str(value)
            except Exception as e:
                return f'Unable to serialize: {e}'
