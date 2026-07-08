from __future__ import annotations

import itertools
import time
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from genai_prices.types import PriceCalculation
from opentelemetry.metrics import MeterProvider, get_meter_provider
from opentelemetry.trace import Span, Tracer, TracerProvider, get_tracer_provider
from opentelemetry.util.types import AttributeValue
from pydantic_core import to_json

from pydantic_ai._instrumentation import (
    DEFAULT_INSTRUMENTATION_VERSION,
    TIME_TO_FIRST_CHUNK_HISTOGRAM_BOUNDARIES,
    TOKEN_HISTOGRAM_BOUNDARIES,
    get_instructions,
    open_model_request_span,
    safe_to_json,
)

from .. import _otel_messages
from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
)
from ..settings import ModelSettings
from . import KnownModelName, Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from .wrapper import WrapperModel

__all__ = 'instrument_model', 'InstrumentationSettings', 'InstrumentedModel'


def instrument_model(model: Model, instrument: InstrumentationSettings | bool) -> Model:
    """Wrap `model` in an `InstrumentedModel` so OTel/Logfire spans are emitted around requests."""
    if instrument and not isinstance(model, InstrumentedModel):
        if instrument is True:
            instrument = InstrumentationSettings()

        model = InstrumentedModel(model, instrument)

    return model


@dataclass(init=False)
class InstrumentationSettings:
    """Options for instrumenting models and agents with OpenTelemetry.

    Used in:

    - [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability
    - [`Agent.instrument`][pydantic_ai.agent.Agent.instrument] / [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
    - [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]

    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    tracer: Tracer = field(repr=False)
    include_binary_content: bool = True
    include_content: bool = True
    version: Literal[2, 3, 4, 5] = DEFAULT_INSTRUMENTATION_VERSION
    use_aggregated_usage_attribute_names: bool = True

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        include_binary_content: bool = True,
        include_content: bool = True,
        version: Literal[2, 3, 4, 5] = DEFAULT_INSTRUMENTATION_VERSION,
        use_aggregated_usage_attribute_names: bool = True,
    ):
        """Create instrumentation options.

        Args:
            tracer_provider: The OpenTelemetry tracer provider to use.
                If not provided, the global tracer provider is used.
                Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
            meter_provider: The OpenTelemetry meter provider to use.
                If not provided, the global meter provider is used.
                Calling `logfire.configure()` sets the global meter provider, so most users don't need this.
            include_binary_content: Whether to include binary content in the instrumentation events.
            include_content: Whether to include prompts, completions, and tool call arguments and responses
                in the instrumentation events.
            version: Version of the data format. This is unrelated to the Pydantic AI package version.
                Defaults to version 5. Versions 2, 3, and 4 are deprecated compatibility formats
                and emit a `PydanticAIDeprecationWarning` when used.
                Version 2 uses the newer OpenTelemetry GenAI spec and stores messages in the following attributes:
                    - `gen_ai.system_instructions` for instructions passed to the agent.
                    - `gen_ai.input.messages` and `gen_ai.output.messages` on model request spans.
                    - `pydantic_ai.all_messages` on agent run spans.
                Version 3 is the same as version 2, with additional support for thinking tokens.
                Version 4 is the same as version 3, with GenAI semantic conventions for multimodal content:
                    URL-based media uses type='uri' with uri and mime_type fields (and modality for image/audio/video).
                    Inline binary content uses type='blob' with mime_type and content fields (and modality for image/audio/video).
                    https://opentelemetry.io/docs/specs/semconv/gen-ai/non-normative/examples-llm-calls/#multimodal-inputs-example
                Version 5 is the same as version 4, but CallDeferred and ApprovalRequired exceptions
                    no longer record an exception event or set the span status to ERROR — the span is left
                    as UNSET, since deferrals are control flow, not errors.
            use_aggregated_usage_attribute_names: Whether to use `gen_ai.aggregated_usage.*` attribute names
                for token usage on agent run spans instead of the standard `gen_ai.usage.*` names.
                Defaults to True to prevent double-counting in observability backends that aggregate span
                attributes across parent and child spans.
                Note: `gen_ai.aggregated_usage.*` is a custom namespace, not part of the OpenTelemetry
                Semantic Conventions. It may be updated if OTel introduces an official convention.
        """
        from pydantic_ai import __version__

        tracer_provider = tracer_provider or get_tracer_provider()
        meter_provider = meter_provider or get_meter_provider()
        scope_name = 'pydantic-ai'
        self.tracer = tracer_provider.get_tracer(scope_name, __version__)
        self.meter = meter_provider.get_meter(scope_name, __version__)
        self.include_binary_content = include_binary_content
        self.include_content = include_content

        if version not in (2, 3, 4, 5):
            raise ValueError('Instrumentation version must be one of 2, 3, 4, or 5.')
        if version in (2, 3, 4):
            warnings.warn(
                'Instrumentation format versions 2, 3, and 4 are deprecated; use `version=5` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
        self.version = version
        self.use_aggregated_usage_attribute_names = use_aggregated_usage_attribute_names

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
        except TypeError:  # pragma: lax no cover
            # Older OTel/logfire versions don't support explicit_bucket_boundaries_advisory
            self.tokens_histogram = self.meter.create_histogram(
                **tokens_histogram_kwargs,  # pyright: ignore
            )
        self.cost_histogram = self.meter.create_histogram(
            'operation.cost',
            unit='{USD}',
            description='Monetary cost',
        )
        time_to_first_chunk_histogram_kwargs = dict(
            name='gen_ai.client.operation.time_to_first_chunk',
            unit='s',
            description='Time from issuing a streaming request to the first chunk being surfaced to the consumer',
        )
        try:
            self.time_to_first_chunk_histogram = self.meter.create_histogram(
                **time_to_first_chunk_histogram_kwargs,
                explicit_bucket_boundaries_advisory=TIME_TO_FIRST_CHUNK_HISTOGRAM_BOUNDARIES,
            )
        except TypeError:  # pragma: lax no cover
            # Older OTel/logfire versions don't support explicit_bucket_boundaries_advisory
            self.time_to_first_chunk_histogram = self.meter.create_histogram(
                **time_to_first_chunk_histogram_kwargs,  # pyright: ignore
            )

    def messages_to_otel_messages(self, messages: list[ModelMessage]) -> list[_otel_messages.ChatMessage]:
        result: list[_otel_messages.ChatMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for is_system, group in itertools.groupby(message.parts, key=lambda p: isinstance(p, SystemPromptPart)):
                    message_parts: list[_otel_messages.MessagePart] = []
                    for part in group:
                        if hasattr(part, 'otel_message_parts'):
                            message_parts.extend(part.otel_message_parts(self))

                    result.append(
                        _otel_messages.ChatMessage(role='system' if is_system else 'user', parts=message_parts)
                    )
            elif isinstance(message, ModelResponse):  # pragma: no branch
                otel_message = _otel_messages.OutputMessage(role='assistant', parts=message.otel_message_parts(self))
                if message.finish_reason is not None:
                    otel_message['finish_reason'] = message.finish_reason
                result.append(otel_message)
        return result

    def handle_messages(
        self,
        input_messages: list[ModelMessage],
        response: ModelResponse,
        span: Span,
        parameters: ModelRequestParameters | None = None,
    ):
        output_messages = self.messages_to_otel_messages([response])
        assert len(output_messages) == 1
        output_message = output_messages[0]

        instructions = get_instructions(input_messages, parameters)
        system_instructions_attributes = self.system_instructions_attributes(instructions)

        attributes: dict[str, AttributeValue] = {
            'gen_ai.input.messages': safe_to_json(self.messages_to_otel_messages(input_messages)).decode(),
            'gen_ai.output.messages': safe_to_json([output_message]).decode(),
            **system_instructions_attributes,
            'logfire.json_schema': to_json(
                {
                    'type': 'object',
                    'properties': {
                        'gen_ai.input.messages': {'type': 'array'},
                        'gen_ai.output.messages': {'type': 'array'},
                        **({'gen_ai.system_instructions': {'type': 'array'}} if system_instructions_attributes else {}),
                        'model_request_parameters': {'type': 'object'},
                    },
                }
            ).decode(),
        }
        span.set_attributes(attributes)

    def system_instructions_attributes(self, instructions: str | None) -> dict[str, str]:
        if instructions and self.include_content:
            return {
                'gen_ai.system_instructions': safe_to_json(
                    [_otel_messages.TextPart(type='text', content=instructions)]
                ).decode(),
            }
        return {}

    def record_metrics(
        self,
        response: ModelResponse,
        price_calculation: PriceCalculation | None,
        attributes: dict[str, AttributeValue],
        time_to_first_chunk: float | None = None,
    ):
        for typ in ['input', 'output']:
            if not (tokens := getattr(response.usage, f'{typ}_tokens', 0)):  # pragma: no cover
                continue
            token_attributes = {**attributes, 'gen_ai.token.type': typ}
            self.tokens_histogram.record(tokens, token_attributes)
        if price_calculation:
            cost = float(price_calculation.total_price)
            self.cost_histogram.record(cost, attributes)
        if time_to_first_chunk is not None:
            self.time_to_first_chunk_histogram.record(time_to_first_chunk, attributes)


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
        request_context = ModelRequestContext(
            model=self.wrapped,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        with open_model_request_span(self.instrumentation_settings, request_context) as (finish, prepared_rc):
            response = await self.wrapped.request(
                prepared_rc.messages, prepared_rc.model_settings, prepared_rc.model_request_parameters
            )
            finish(response)
            return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        request_context = ModelRequestContext(
            model=self.wrapped,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        with open_model_request_span(self.instrumentation_settings, request_context) as (finish, prepared_rc):
            response_stream: StreamedResponse | None = None
            # Stamp the request-issue instant before the wrapped model opens the stream, so the
            # `time_to_first_chunk` delta spans from when we issue the request to when the first
            # chunk is surfaced to the consumer.
            request_start = time.perf_counter()
            try:
                async with self.wrapped.request_stream(
                    prepared_rc.messages,
                    prepared_rc.model_settings,
                    prepared_rc.model_request_parameters,
                    run_context,
                ) as response_stream:
                    yield response_stream
            finally:
                if response_stream:  # pragma: no branch
                    finish(
                        response_stream.get(),
                        time_to_first_chunk=response_stream.time_to_first_chunk(request_start),
                    )
