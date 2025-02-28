from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import logfire_api
from opentelemetry._events import Event, EventLogger, EventLoggerProvider, get_event_logger_provider
from opentelemetry.trace import Span, Tracer, TracerProvider, get_tracer_provider
from opentelemetry.util.types import AttributeValue
from pydantic import TypeAdapter

from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from ..settings import ModelSettings
from ..usage import Usage
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

MODEL_SETTING_ATTRIBUTES: tuple[
    Literal[
        'max_tokens',
        'top_p',
        'seed',
        'temperature',
        'presence_penalty',
        'frequency_penalty',
    ],
    ...,
] = (
    'max_tokens',
    'top_p',
    'seed',
    'temperature',
    'presence_penalty',
    'frequency_penalty',
)

ANY_ADAPTER = TypeAdapter[Any](Any)


@dataclass
class InstrumentedModel(WrapperModel):
    """Model which is instrumented with OpenTelemetry."""

    tracer: Tracer = field(repr=False)
    event_logger: EventLogger = field(repr=False)
    event_mode: Literal['attributes', 'logs'] = 'attributes'

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        tracer_provider: TracerProvider | None = None,
        event_logger_provider: EventLoggerProvider | None = None,
        event_mode: Literal['attributes', 'logs'] = 'attributes',
    ):
        super().__init__(wrapped)
        tracer_provider = tracer_provider or get_tracer_provider()
        event_logger_provider = event_logger_provider or get_event_logger_provider()
        self.tracer = tracer_provider.get_tracer('pydantic-ai')
        self.event_logger = event_logger_provider.get_event_logger('pydantic-ai')
        self.event_mode = event_mode

    @classmethod
    def from_logfire(
        cls,
        wrapped: Model | KnownModelName,
        logfire_instance: logfire_api.Logfire = logfire_api.DEFAULT_LOGFIRE_INSTANCE,
        event_mode: Literal['attributes', 'logs'] = 'attributes',
    ) -> InstrumentedModel:
        if hasattr(logfire_instance.config, 'get_event_logger_provider'):
            event_provider = logfire_instance.config.get_event_logger_provider()
        else:
            event_provider = None
        tracer_provider = logfire_instance.config.get_tracer_provider()
        return cls(wrapped, tracer_provider, event_provider, event_mode)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        with self._instrument(messages, model_settings) as finish:
            response, usage = await super().request(messages, model_settings, model_request_parameters)
            finish(response, usage)
            return response, usage

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        with self._instrument(messages, model_settings) as finish:
            response_stream: StreamedResponse | None = None
            try:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
            finally:
                if response_stream:
                    finish(response_stream.get(), response_stream.usage())

    @contextmanager
    def _instrument(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
    ) -> Iterator[Callable[[ModelResponse, Usage], None]]:
        operation = 'chat'
        model_name = self.model_name
        span_name = f'{operation} {model_name}'
        system = getattr(self.wrapped, 'system', '') or self.wrapped.__class__.__name__.removesuffix('Model').lower()
        system = {'google-gla': 'gemini', 'google-vertex': 'vertex_ai', 'mistral': 'mistral_ai'}.get(system, system)
        # TODO Missing attributes:
        #  - server.address: requires a Model.base_url abstract method or similar
        #  - server.port: to parse from the base_url
        #  - error.type: unclear if we should do something here or just always rely on span exceptions
        #  - gen_ai.request.stop_sequences/top_k: model_settings doesn't include these
        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            'gen_ai.system': system,
            'gen_ai.request.model': model_name,
        }

        if model_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if isinstance(value := model_settings.get(key), (float, int)):
                    attributes[f'gen_ai.request.{key}'] = value

        with self.tracer.start_as_current_span(span_name, attributes=attributes) as span:

            def finish(response: ModelResponse, usage: Usage):
                if not span.is_recording():
                    return

                events = self.messages_to_otel_events(messages)
                for event in self.messages_to_otel_events([response]):
                    events.append(
                        Event(
                            'gen_ai.choice',
                            body={
                                # TODO finish_reason
                                'index': 0,
                                'message': event.body,
                            },
                        )
                    )
                span.set_attributes(
                    {
                        # TODO finish_reason (https://github.com/open-telemetry/semantic-conventions/issues/1277), id
                        #  https://github.com/pydantic/pydantic-ai/issues/886
                        'gen_ai.response.model': response.model_name or model_name,
                        **usage.opentelemetry_attributes(),
                    }
                )
                self._emit_events(system, span, events)

            yield finish

    def _emit_events(self, system: str, span: Span, events: list[Event]) -> None:
        for event in events:
            event.attributes = {'gen_ai.system': system, **(event.attributes or {})}
        if self.event_mode == 'logs':
            for event in events:
                self.event_logger.emit(event)
        else:
            attr_name = 'events'
            span.set_attributes(
                {
                    attr_name: json.dumps([self.event_to_dict(event) for event in events]),
                    'logfire.json_schema': json.dumps(
                        {
                            'type': 'object',
                            'properties': {attr_name: {'type': 'array'}},
                        }
                    ),
                }
            )

    @staticmethod
    def event_to_dict(event: Event) -> dict[str, Any]:
        if not event.body:
            body = {}
        elif isinstance(event.body, Mapping):
            body = event.body  # type: ignore
        else:
            body = {'body': event.body}
        return {**body, **(event.attributes or {})}

    @staticmethod
    def messages_to_otel_events(messages: list[ModelMessage]) -> list[Event]:
        result: list[Event] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if hasattr(part, 'otel_event'):
                        result.append(part.otel_event())
            elif isinstance(message, ModelResponse):
                result.extend(message.otel_events())
        for event in result:
            try:
                event.body = ANY_ADAPTER.dump_python(event.body, mode='json')
            except Exception:
                try:
                    event.body = str(event.body)
                except Exception:
                    event.body = 'Unable to serialize event body'
        return result
