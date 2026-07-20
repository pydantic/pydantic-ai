from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast
from urllib.parse import urlparse

from opentelemetry import context as otel_context
from opentelemetry.baggage import get_baggage
from opentelemetry.trace import INVALID_SPAN, SpanKind, get_current_span
from opentelemetry.util.types import AttributeValue
from pydantic import ConfigDict, TypeAdapter
from pydantic_core import PydanticSerializationError, to_json

from pydantic_graph._utils import get_traceparent

if TYPE_CHECKING:
    from typing_extensions import Self

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.settings import ModelSettings

DEFAULT_INSTRUMENTATION_VERSION = 5
"""Default instrumentation version for `InstrumentationSettings`."""

AGENT_NAME_BAGGAGE_KEY = 'gen_ai.agent.name'
RUN_ID_BAGGAGE_KEY = 'gen_ai.agent.call.id'
CONVERSATION_ID_BAGGAGE_KEY = 'gen_ai.conversation.id'

GEN_AI_SYSTEM_ATTRIBUTE = 'gen_ai.system'
GEN_AI_REQUEST_MODEL_ATTRIBUTE = 'gen_ai.request.model'
GEN_AI_PROVIDER_NAME_ATTRIBUTE = 'gen_ai.provider.name'

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
_BASE64_ANY_ADAPTER = TypeAdapter[Any](Any, config=ConfigDict(ser_json_bytes='base64'))

# These are in the spec:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
TOKEN_HISTOGRAM_BOUNDARIES = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)

# These are advised by the spec (the metric is "Development" stability, so this may change):
# https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md#metric-gen_aiclientoperationtime_to_first_chunk
# Like any bucket advisory it's only advice: users can override it by configuring a View for this
# instrument on their MeterProvider, and SDKs configured for exponential-bucket histogram
# aggregation (e.g. logfire) ignore it entirely.
TIME_TO_FIRST_CHUNK_HISTOGRAM_BOUNDARIES = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)  # fmt: skip

time_to_first_chunk_ctx: ContextVar[float | None] = ContextVar('time_to_first_chunk', default=None)
"""Carries streaming TTFT (in seconds) from the agent graph's streaming request handler to the
`Instrumentation` capability, which reads it after `await handler(...)` returns — the handler runs
in the same task, so its `set` is visible there. The agent graph spawns a fresh task per streaming
request and only that handler ever sets the variable, so a value can't outlive its request;
non-streaming requests read the `None` default.

This is a context variable rather than a field on `ModelRequestContext` because that object is
public and holds only the *inputs* to `Model.request[_stream]`.
"""


class CostCalculationFailedWarning(Warning):
    """Warning raised when cost calculation fails."""


def get_agent_run_baggage_attributes() -> dict[str, Any]:
    """Read agent name, run ID, and conversation ID from OTel baggage and return as span attributes."""
    attrs: dict[str, Any] = {}
    agent_name = get_baggage(AGENT_NAME_BAGGAGE_KEY)
    if agent_name is not None:
        attrs[AGENT_NAME_BAGGAGE_KEY] = agent_name
    run_id = get_baggage(RUN_ID_BAGGAGE_KEY)
    if run_id is not None:
        attrs[RUN_ID_BAGGAGE_KEY] = run_id
    conversation_id = get_baggage(CONVERSATION_ID_BAGGAGE_KEY)
    if conversation_id is not None:
        attrs[CONVERSATION_ID_BAGGAGE_KEY] = conversation_id
    return attrs


def serialize_any(value: Any) -> str:
    try:
        try:
            return ANY_ADAPTER.dump_python(value, mode='json')
        except UnicodeDecodeError:
            return _BASE64_ANY_ADAPTER.dump_python(value, mode='json')
    except Exception:
        try:
            return str(value)
        except Exception as e:
            return f'Unable to serialize: {e}'


def safe_to_json(value: object) -> bytes:
    """Serialize `value` to compact JSON bytes, tolerating lone surrogates.

    `to_json` raises on unpaired surrogates (e.g. text decoded with `errors='surrogateescape'`),
    which would crash an otherwise-successful run from within instrumentation. The stdlib fallback
    escapes them, matching the lenient behavior callers had before adopting `to_json`.
    """
    try:
        return to_json(value)
    except PydanticSerializationError:
        return json.dumps(value, separators=(',', ':')).encode()


def model_attributes(model: Model) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        GEN_AI_PROVIDER_NAME_ATTRIBUTE: model.system,  # New OTel standard attribute
        GEN_AI_SYSTEM_ATTRIBUTE: model.system,  # Preserved for backward compatibility (deprecated)
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


def model_request_parameters_attributes(
    model_request_parameters: ModelRequestParameters,
) -> dict[str, AttributeValue]:
    return {'model_request_parameters': safe_to_json(serialize_any(model_request_parameters)).decode()}


def model_settings_attributes(model_settings: ModelSettings | None) -> dict[str, AttributeValue]:
    """Map the OTel-spec model settings (`max_tokens`, `temperature`, ...) to `gen_ai.request.*` attributes."""
    attributes: dict[str, AttributeValue] = {}
    if model_settings:
        for key in MODEL_SETTING_ATTRIBUTES:
            if isinstance(value := model_settings.get(key), float | int):
                attributes[f'gen_ai.request.{key}'] = value
    return attributes


def annotate_tool_call_otel_metadata(response: ModelResponse, parameters: ModelRequestParameters) -> None:
    """Copy OTel-relevant metadata from tool definitions onto matching tool call parts.

    This allows tool definition metadata (e.g. code language hints set by the code-mode toolset)
    to flow through to OTel events on both the model request span and the agent run span.
    """
    from pydantic_ai import _otel_messages
    from pydantic_ai.messages import BaseToolCallPart

    tool_defs = parameters.tool_defs
    if not tool_defs:
        return
    for part in response.parts:
        if isinstance(part, BaseToolCallPart) and (tool_def := tool_defs.get(part.tool_name)):
            if tool_def.metadata:
                otel_metadata: _otel_messages.ToolCallPartOtelMetadata = {}
                if code_arg_name := tool_def.metadata.get('code_arg_name'):
                    otel_metadata['code_arg_name'] = code_arg_name
                if code_arg_language := tool_def.metadata.get('code_arg_language'):
                    otel_metadata['code_arg_language'] = code_arg_language
                if otel_metadata:
                    part.otel_metadata = otel_metadata


def build_tool_definitions(model_request_parameters: ModelRequestParameters) -> list[dict[str, Any]]:
    """Build OTel-compliant tool definitions from model request parameters.

    Extracts tool metadata from function_tools and output_tools into a list of
    tool definition dicts following the OTel GenAI semantic conventions format.
    """
    all_tools = itertools.chain(
        model_request_parameters.function_tools or [],
        model_request_parameters.output_tools or [],
    )

    tool_definitions: list[dict[str, Any]] = []
    for tool in all_tools:
        tool_def: dict[str, Any] = {'type': 'function', 'name': tool.name}
        if tool.description:
            tool_def['description'] = tool.description
        if tool.parameters_json_schema:
            tool_def['parameters'] = tool.parameters_json_schema
        tool_definitions.append(tool_def)

    return tool_definitions


class _FinishModelRequestSpan(Protocol):
    """The `finish` callback yielded by `open_model_request_span`.

    `time_to_first_chunk` is the streaming-only TTFT in seconds; non-streaming
    callers omit it.
    """

    def __call__(self, response: ModelResponse, time_to_first_chunk: float | None = None) -> None: ...


@contextmanager
def open_model_request_span(
    settings: InstrumentationSettings,
    request_context: ModelRequestContext,
) -> Generator[tuple[_FinishModelRequestSpan, ModelRequestContext]]:
    """Open a `chat <model>` CLIENT span; yield `(finish, prepared_request_context)`.

    Shared between `Instrumentation.wrap_model_request` (agent flow) and
    `InstrumentedModel.request`/`request_stream` (standalone / `direct.model_request*`).
    Calls `model.prepare_request(...)` internally and yields a request context with the prepared
    settings/parameters so callers don't have to re-prepare. `finish(response)` annotates the
    response with OTel tool-call metadata and records outcome attributes. Token/cost metrics are
    recorded *after* the span closes so backends that aggregate from span attributes don't
    double-count.
    """
    # TODO Missing attributes:
    #  - error.type: unclear if we should do something here or just always rely on span exceptions
    #  - gen_ai.request.stop_sequences/top_k: model_settings doesn't include these
    model = request_context.model
    prepared_settings, prepared_parameters = model.prepare_request(
        request_context.model_settings, request_context.model_request_parameters
    )
    prepared_request_context = replace(
        request_context, model_settings=prepared_settings, model_request_parameters=prepared_parameters
    )
    operation = 'chat'
    span_name = f'{operation} {model.model_name}'
    attributes: dict[str, AttributeValue] = {
        'gen_ai.operation.name': operation,
        **model_attributes(model),
        **get_agent_run_baggage_attributes(),
    }
    json_schema_properties: dict[str, dict[str, str]] = {}
    if settings.include_model_request_parameters:
        attributes.update(model_request_parameters_attributes(prepared_parameters))
        json_schema_properties['model_request_parameters'] = {'type': 'object'}
    attributes['logfire.json_schema'] = to_json({'type': 'object', 'properties': json_schema_properties}).decode()

    tool_definitions = build_tool_definitions(prepared_parameters)
    if tool_definitions:
        attributes['gen_ai.tool.definitions'] = safe_to_json(tool_definitions).decode()

    attributes.update(model_settings_attributes(prepared_settings))

    record_metrics: Callable[[], None] | None = None
    try:
        with settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT) as span:
            # `finish` is a closure rather than inline so we can (a) set result attributes
            # inside the `with span:` block — they attach to the span — and (b) call the
            # captured `record_metrics` in the outer `finally` AFTER the span closes,
            # so observability backends that aggregate metrics from span attributes
            # don't double-count.
            def finish(response: ModelResponse, time_to_first_chunk: float | None = None) -> None:
                nonlocal record_metrics

                annotate_tool_call_otel_metadata(response, prepared_parameters)

                # FallbackModel updates these span attributes via get_current_span().
                attributes.update(getattr(span, 'attributes', {}))
                request_model = attributes[GEN_AI_REQUEST_MODEL_ATTRIBUTE]
                system = cast(str, attributes[GEN_AI_SYSTEM_ATTRIBUTE])

                response_model = response.model_name or request_model
                price_calculation = None

                def _record_metrics() -> None:
                    metric_attributes = {
                        GEN_AI_PROVIDER_NAME_ATTRIBUTE: system,
                        GEN_AI_SYSTEM_ATTRIBUTE: system,
                        'gen_ai.operation.name': operation,
                        'gen_ai.request.model': request_model,
                        'gen_ai.response.model': response_model,
                    }
                    settings.record_metrics(response, price_calculation, metric_attributes, time_to_first_chunk)

                record_metrics = _record_metrics

                # Compute cost before the `is_recording()` gate so `_record_metrics`
                # always emits cost data, even when the span is dropped by sampling.
                try:
                    price_calculation = response.cost()
                except LookupError:
                    pass
                except Exception as e:
                    warnings.warn(
                        f'Failed to get cost from response: {type(e).__name__}: {e}',
                        CostCalculationFailedWarning,
                    )

                if not span.is_recording():
                    return

                settings.handle_messages(prepared_request_context.messages, response, span, prepared_parameters)

                attributes_to_set: dict[str, Any] = {
                    **response.usage.opentelemetry_attributes(),
                    'gen_ai.response.model': response_model,
                }
                if price_calculation is not None:
                    attributes_to_set['operation.cost'] = float(price_calculation.total_price)
                if response.provider_response_id is not None:
                    attributes_to_set['gen_ai.response.id'] = response.provider_response_id
                if response.finish_reason is not None:
                    attributes_to_set['gen_ai.response.finish_reasons'] = [response.finish_reason]
                if time_to_first_chunk is not None:
                    attributes_to_set['gen_ai.client.operation.time_to_first_chunk'] = time_to_first_chunk
                span.set_attributes(attributes_to_set)
                span.update_name(f'{operation} {request_model}')

            yield finish, prepared_request_context
    finally:
        if record_metrics:
            record_metrics()


def capture_current_context() -> Callable[[], AbstractContextManager[None]]:
    """Snapshot the current OTel context so it can be re-attached in another task.

    The streaming continuation composite opens each segment's `request_stream` lazily,
    in the *consumer* task that iterates the stream, whereas the `chat` span is opened
    by `wrap_model_request` in a separate task. Those tasks don't share an OTel context,
    so without re-attaching, span updates driven by `get_current_span()` (e.g.
    `FallbackModel` recording the resolved inner model) would land on the wrong span.

    Returns a factory that yields a context manager re-attaching the captured context;
    the composite enters it around each segment without depending on OpenTelemetry itself.
    """
    captured = otel_context.get_current()

    @contextmanager
    def attach_captured_context() -> Generator[None]:
        # Restore the previous context by re-`attach`ing it rather than `detach`ing the token: this CM is
        # held across the `yield` in `_ContinuationStreamedResponse._get_event_iterator`, so when a streamed
        # run is interrupted mid-segment the async generator is finalized (`GeneratorExit`) in a different
        # contextvars `Context`, where `otel_context.detach(token)` -> `ContextVar.reset` raises
        # `ValueError: ... created in a different Context`. OTel swallows it but logs a noisy
        # 'Failed to detach context' (surfaced verbatim in the Pyodide output panel). `attach()` is a plain
        # `set`, which never fails cross-context, so it restores `previous` silently. See #6569.
        previous = otel_context.get_current()
        otel_context.attach(captured)
        try:
            yield
        finally:
            otel_context.attach(previous)

    return attach_captured_context


def get_instructions(
    messages: Sequence[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
) -> str | None:
    """Get the joined instructions string for the current request.

    When `model_request_parameters` is provided (normal model request flow), returns
    the joined content of `instruction_parts` which already includes prompted output
    instructions and is properly sorted.

    Falls back to reading `ModelRequest.instructions` from message history when
    `model_request_parameters` is not available (e.g. OTel span attributes).
    """
    from pydantic_ai.messages import InstructionPart, ModelRequest
    from pydantic_ai.models import Model

    if model_request_parameters:
        parts = Model._get_instruction_parts(messages, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
        if parts:
            return InstructionPart.join(parts)

    # Fallback: read from message history (used by OTel when model_request_parameters is unavailable)
    #
    # Get instructions from the first ModelRequest found when iterating messages in reverse.
    # In the case that a "mock" request was generated to include a tool-return part for a result tool,
    # we want to use the instructions from the second-to-most-recent request (which should correspond to the
    # original request that generated the response that resulted in the tool-return part).
    instructions = None

    last_two_requests: list[ModelRequest] = []
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            last_two_requests.append(message)
            if len(last_two_requests) == 2:
                break
            if message.instructions is not None:
                instructions = message.instructions
                break

    # If we don't have two requests, and we didn't already return instructions, there are definitely not any:
    if instructions is None and len(last_two_requests) == 2:
        most_recent_request = last_two_requests[0]
        second_most_recent_request = last_two_requests[1]

        # If we've gotten this far and the most recent request consists of only tool-return parts or retry-prompt
        # parts, we use the instructions from the second-to-most-recent request. This is necessary because when
        # handling result tools, we generate a "mock" ModelRequest with a tool-return part for it, and that
        # ModelRequest will not have the relevant instructions from the agent.

        # While it's possible that you could have a message history where the most recent request has only tool
        # returns, I believe there is no way to achieve that would _change_ the instructions without manually
        # crafting the most recent message. That might make sense in principle for some usage pattern, but it's
        # enough of an edge case that I think it's not worth worrying about, since you can work around this by
        # inserting another ModelRequest with no parts at all immediately before the request that has the tool
        # calls (that works because we only look at the two most recent ModelRequests here).

        # If you have a use case where this causes pain, please open a GitHub issue and we can discuss alternatives.

        if all(p.part_kind == 'tool-return' or p.part_kind == 'retry-prompt' for p in most_recent_request.parts):
            instructions = second_most_recent_request.instructions

    return instructions


def current_otel_traceparent() -> str | None:
    """Return the W3C traceparent of the active OTel span, or None if no valid span is set.

    Used as a fallback when the graph run was created without a span. In that case,
    the agent run span is typically set by the Instrumentation capability via
    `start_as_current_span` while the capability chain is executing, which is
    exactly when consumers like `OnlineEvaluation` read the traceparent.
    """
    span = get_current_span()
    if span is INVALID_SPAN:
        return None
    return get_traceparent(span) or None


@dataclass(frozen=True)
class InstrumentationNames:
    """Configuration for instrumentation span names and attributes based on version."""

    # Agent run span configuration
    agent_run_span_name: str
    agent_name_attr: str

    # Tool execution span configuration
    tool_span_name: str
    tool_arguments_attr: str
    tool_result_attr: str

    # Output Tool execution span configuration
    output_tool_span_name: str

    # Deferral span attributes
    tool_deferral_name_attr: ClassVar[str] = 'pydantic_ai.tool.deferral.name'
    tool_deferral_metadata_attr: ClassVar[str] = 'pydantic_ai.tool.deferral.metadata'

    @classmethod
    def for_version(cls, version: int) -> Self:
        """Create instrumentation configuration for a specific version.

        Args:
            version: The instrumentation version (2 or 3+)

        Returns:
            InstrumentationConfig instance with version-appropriate settings
        """
        if version == 2:
            return cls(
                agent_run_span_name='agent run',
                agent_name_attr='agent_name',
                tool_span_name='running tool',
                tool_arguments_attr='tool_arguments',
                tool_result_attr='tool_response',
                output_tool_span_name='running output function',
            )
        else:
            return cls(
                agent_run_span_name='invoke_agent',
                agent_name_attr='gen_ai.agent.name',
                tool_span_name='execute_tool',  # Will be formatted with tool name
                tool_arguments_attr='gen_ai.tool.call.arguments',
                tool_result_attr='gen_ai.tool.call.result',
                output_tool_span_name='execute_tool',
            )

    def get_agent_run_span_name(self, agent_name: str) -> str:
        """Get the formatted agent span name.

        Args:
            agent_name: Name of the agent being executed

        Returns:
            Formatted span name
        """
        if self.agent_run_span_name == 'invoke_agent':
            return f'invoke_agent {agent_name}'
        return self.agent_run_span_name

    def get_tool_span_name(self, tool_name: str) -> str:
        """Get the formatted tool span name.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            Formatted span name
        """
        if self.tool_span_name == 'execute_tool':
            return f'execute_tool {tool_name}'
        return self.tool_span_name

    def get_output_tool_span_name(self, tool_name: str) -> str:
        """Get the formatted output tool span name.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            Formatted span name
        """
        if self.output_tool_span_name == 'execute_tool':
            return f'execute_tool {tool_name}'
        return self.output_tool_span_name
