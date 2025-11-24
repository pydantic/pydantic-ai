from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload
from urllib.parse import urlparse

from opentelemetry.util.types import AttributeValue

from pydantic_ai.models.instrumented import ANY_ADAPTER, InstrumentationSettings

from .base import EmbeddingModel, EmbedInputType
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

    @overload
    async def embed(
        self, documents: str, *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float]: ...

    @overload
    async def embed(
        self, documents: Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]: ...

    async def embed(
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
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
    ) -> Iterator[Callable[[list[float] | list[list[float]]], None]]:
        operation = 'embed'
        span_name = f'{operation} {self.model_name}'

        num_inputs = 1 if isinstance(documents, str) else len(documents)

        # TODO (DouweM): Review all of these attributes that Claude may have hallucinated.

        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            **self.model_attributes(self.wrapped),
            'gen_ai.embedding.input_type': input_type,
            'gen_ai.embedding.num_inputs': num_inputs,
        }

        if settings:
            attributes['embedding_settings'] = json.dumps(self.serialize_any(settings))

        if self.instrumentation_settings.include_content and isinstance(documents, str):
            attributes['gen_ai.prompt'] = documents
        elif self.instrumentation_settings.include_content:
            # For sequences, store as JSON array
            attributes['gen_ai.prompt'] = json.dumps(list(documents))

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

                def finish(result: list[float] | list[list[float]]):
                    if not span.is_recording():
                        return

                    # Calculate output dimension
                    if isinstance(result, list) and result:
                        if isinstance(result[0], list):
                            # Multiple embeddings
                            output_dim = len(result[0]) if result[0] else 0
                            num_outputs = len(result)
                        else:
                            # Single embedding
                            output_dim = len(result)
                            num_outputs = 1
                    else:
                        output_dim = 0
                        num_outputs = 0

                    attributes_to_set = {
                        'gen_ai.embedding.dimension': output_dim,
                        'gen_ai.embedding.num_outputs': num_outputs,
                    }
                    span.set_attributes(attributes_to_set)

                yield finish
        finally:
            pass

    @staticmethod
    def model_attributes(model: EmbeddingModel) -> dict[str, AttributeValue]:
        attributes: dict[str, AttributeValue] = {
            'gen_ai.system': model.system,
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
