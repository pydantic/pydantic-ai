from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from urllib.parse import urlparse

from opentelemetry._events import (
    Event,  # pyright: ignore[reportPrivateImportUsage]
    EventLogger,  # pyright: ignore[reportPrivateImportUsage]
    EventLoggerProvider,  # pyright: ignore[reportPrivateImportUsage]
    get_event_logger_provider,  # pyright: ignore[reportPrivateImportUsage]
)
from opentelemetry.metrics import MeterProvider, get_meter_provider
from opentelemetry.trace import Span, Tracer, TracerProvider, get_tracer_provider
from opentelemetry.util.types import AttributeValue
from pydantic import TypeAdapter

from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from ..settings import ModelSettings
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

__all__ = 'instrument_model', 'InstrumentationSettings', 'InstrumentedModel'

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

# These are in the spec:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
TOKEN_HISTOGRAM_BOUNDARIES = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)


def instrument_model(model: Model, instrument: InstrumentationSettings | bool) -> Model:
    """Instrument a model with OpenTelemetry/logfire."""
    if instrument and not isinstance(model, InstrumentedModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedModel(model, instrument)

    return model


@dataclass(init=False)
class InstrumentationSettings:
    """Options for instrumenting models and agents with OpenTelemetry.

    Used in:

    - `Agent(instrument=...)`
    - [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
    - [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    tracer: Tracer = field(repr=False)
    event_logger: EventLogger = field(repr=False)
    event_mode: Literal['attributes', 'logs'] = 'attributes'
    include_binary_content: bool = True

    def __init__(
        self,
        *,
        event_mode: Literal['attributes', 'logs'] = 'attributes',
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        event_logger_provider: EventLoggerProvider | None = None,
        include_binary_content: bool = True,
        include_content: bool = True,
    ):
        """Create instrumentation options.

        Args:
            event_mode: The mode for emitting events. If `'attributes'`, events are attached to the span as attributes.
                If `'logs'`, events are emitted as OpenTelemetry log-based events.
            tracer_provider: The OpenTelemetry tracer provider to use.
                If not provided, the global tracer provider is used.
                Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
            meter_provider: The OpenTelemetry meter provider to use.
                If not provided, the global meter provider is used.
                Calling `logfire.configure()` sets the global meter provider, so most users don't need this.
            event_logger_provider: The OpenTelemetry event logger provider to use.
                If not provided, the global event logger provider is used.
                Calling `logfire.configure()` sets the global event logger provider, so most users don't need this.
                This is only used if `event_mode='logs'`.
            include_binary_content: Whether to include binary content in the instrumentation events.
            include_content: Whether to include prompts, completions, and tool call arguments and responses
                in the instrumentation events.
        """
        from pydantic_ai import __version__

        tracer_provider = tracer_provider or get_tracer_provider()
        meter_provider = meter_provider or get_meter_provider()
        event_logger_provider = event_logger_provider or get_event_logger_provider()
        scope_name = 'pydantic-ai'
        self.tracer = tracer_provider.get_tracer(scope_name, __version__)
        self.meter = meter_provider.get_meter(scope_name, __version__)
        self.event_logger = event_logger_provider.get_event_logger(scope_name, __version__)
        self.event_mode = event_mode
        self.include_binary_content = include_binary_content
        self.include_content = include_content

        # As specified in the OpenTelemetry GenAI metrics spec:
        # https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
        tokens_histogram_kwargs = dict(
            name='gen_ai.client.token.usage',
            unit='{token}',
            description='Measures number of input and output tokens used',
        )
        try:
            self.tokens_histogram = self.meter.create_histogram(
                **tokens_histogram_kwargs,
                explicit_bucket_boundaries_advisory=TOKEN_HISTOGRAM_BOUNDARIES,
            )
        except TypeError:
            # Older OTel/logfire versions don't support explicit_bucket_boundaries_advisory
            self.tokens_histogram = self.meter.create_histogram(
                **tokens_histogram_kwargs,  # pyright: ignore
            )

    def messages_to_otel_events(self, messages: list[ModelMessage]) -> list[Event]:
        """Convert a list of model messages to OpenTelemetry events.

        Args:
            messages: The messages to convert.

        Returns:
            A list of OpenTelemetry events.
        """
        events: list[Event] = []
        instructions = InstrumentedModel._get_instructions(messages)  # pyright: ignore [reportPrivateUsage]
        if instructions is not None:
            events.append(
                Event(
                    'gen_ai.system.message',
                    body={**({'content': instructions} if self.include_content else {}), 'role': 'system'},
                )
            )

        for message_index, message in enumerate(messages):
            message_events: list[Event] = []
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if hasattr(part, 'otel_event'):
                        message_events.append(part.otel_event(self))
            elif isinstance(message, ModelResponse):  # pragma: no branch
                message_events = message.otel_events(self)
            for event in message_events:
                event.attributes = {
                    'gen_ai.message.index': message_index,
                    **(event.attributes or {}),
                }
            events.extend(message_events)

        for event in events:
            event.body = InstrumentedModel.serialize_any(event.body)
        return events


GEN_AI_SYSTEM_ATTRIBUTE = 'gen_ai.system'
GEN_AI_REQUEST_MODEL_ATTRIBUTE = 'gen_ai.request.model'


@dataclass(init=False)
class InstrumentedModel(WrapperModel):
    """Model which wraps another model so that requests are instrumented with OpenTelemetry.

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    instrumentation_settings: InstrumentationSettings
    """Instrumentation settings for this model."""

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        options: InstrumentationSettings | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.instrumentation_settings = options or InstrumentationSettings()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        with self._instrument(messages, model_settings, model_request_parameters) as finish:
            response = await super().request(messages, model_settings, model_request_parameters)
            finish(response)
            return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        with self._instrument(messages, model_settings, model_request_parameters) as finish:
            response_stream: StreamedResponse | None = None
            try:
                async with super().request_stream(
                    messages, model_settings, model_request_parameters
                ) as response_stream:
                    yield response_stream
            finally:
                if response_stream:  # pragma: no branch
                    finish(response_stream.get())

    @contextmanager
    def _instrument(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Iterator[Callable[[ModelResponse], None]]:
        operation = 'chat'
        span_name = f'{operation} {self.model_name}'
        # TODO Missing attributes:
        #  - error.type: unclear if we should do something here or just always rely on span exceptions
        #  - gen_ai.request.stop_sequences/top_k: model_settings doesn't include these
        attributes: dict[str, AttributeValue] = {
            'gen_ai.operation.name': operation,
            **self.model_attributes(self.wrapped),
            'model_request_parameters': json.dumps(InstrumentedModel.serialize_any(model_request_parameters)),
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {'model_request_parameters': {'type': 'object'}},
                }
            ),
        }

        if model_settings:
            for key in MODEL_SETTING_ATTRIBUTES:
                if isinstance(value := model_settings.get(key), (float, int)):
                    attributes[f'gen_ai.request.{key}'] = value

        record_metrics: Callable[[], None] | None = None
        try:
            with self.instrumentation_settings.tracer.start_as_current_span(span_name, attributes=attributes) as span:

                def finish(response: ModelResponse):
                    # FallbackModel updates these span attributes.
                    attributes.update(getattr(span, 'attributes', {}))
                    request_model = attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE]
                    system = attributes[GEN_AI_SYSTEM_ATTRIBUTE]

                    response_model = response.model_name or request_model

                    def _record_metrics():
                        metric_attributes = {
                            GEN_AI_SYSTEM_ATTRIBUTE: system,
                            'gen_ai.operation.name': operation,
                            'gen_ai.request.model': request_model,
                            'gen_ai.response.model': response_model,
                        }
                        if response.usage.request_tokens:  # pragma: no branch
                            self.instrumentation_settings.tokens_histogram.record(
                                response.usage.request_tokens,
                                {**metric_attributes, 'gen_ai.token.type': 'input'},
                            )
                        if response.usage.response_tokens:  # pragma: no branch
                            self.instrumentation_settings.tokens_histogram.record(
                                response.usage.response_tokens,
                                {**metric_attributes, 'gen_ai.token.type': 'output'},
                            )

                    nonlocal record_metrics
                    record_metrics = _record_metrics

                    if not span.is_recording():
                        return

                    events = self.instrumentation_settings.messages_to_otel_events(messages)
                    for event in self.instrumentation_settings.messages_to_otel_events([response]):
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
                            **response.usage.opentelemetry_attributes(),
                            'gen_ai.response.model': response_model,
                        }
                    )
                    span.update_name(f'{operation} {request_model}')
                    for event in events:
                        event.attributes = {
                            GEN_AI_SYSTEM_ATTRIBUTE: system,
                            **(event.attributes or {}),
                        }
                    self._emit_events(span, events)

                yield finish
        finally:
            if record_metrics:
                # We only want to record metrics after the span is finished,
                # to prevent them from being redundantly recorded in the span itself by logfire.
                record_metrics()

    def _emit_events(self, span: Span, events: list[Event]) -> None:
        if self.instrumentation_settings.event_mode == 'logs':
            for event in events:
                self.instrumentation_settings.event_logger.emit(event)
        else:
            attr_name = 'events'
            span.set_attributes(
                {
                    attr_name: json.dumps([self.event_to_dict(event) for event in events]),
                    'logfire.json_schema': json.dumps(
                        {
                            'type': 'object',
                            'properties': {
                                attr_name: {'type': 'array'},
                                'model_request_parameters': {'type': 'object'},
                            },
                        }
                    ),
                }
            )

    @staticmethod
    def model_attributes(model: Model):
        attributes: dict[str, AttributeValue] = {
            GEN_AI_SYSTEM_ATTRIBUTE: model.system,
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
                if parsed.port:  # pragma: no branch
                    attributes['server.port'] = parsed.port

        return attributes

    @staticmethod
    def event_to_dict(event: Event) -> dict[str, Any]:
        if not event.body:
            body = {}  # pragma: no cover
        elif isinstance(event.body, Mapping):
            body = event.body  # type: ignore
        else:
            body = {'body': event.body}
        return {**body, **(event.attributes or {})}

    @staticmethod
    def serialize_any(value: Any) -> str:
        try:
            return ANY_ADAPTER.dump_python(value, mode='json')
        except Exception:
            try:
                return str(value)
            except Exception as e:
                return f'Unable to serialize: {e}'
