from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import urlparse

from opentelemetry._logs import LogRecord
from opentelemetry.baggage import get_baggage
from opentelemetry.trace import INVALID_SPAN, Span, SpanKind, Status, StatusCode, get_current_span
from opentelemetry.util.types import AttributeValue
from pydantic import TypeAdapter
from pydantic_core import to_json

from pydantic_graph._utils import get_traceparent

if TYPE_CHECKING:
    from typing_extensions import Self

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.instrumented import InstrumentationSettings

DEFAULT_INSTRUMENTATION_VERSION = 2
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

# These are in the spec:
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/#metric-gen_aiclienttokenusage
TOKEN_HISTOGRAM_BOUNDARIES = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864)


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
        return ANY_ADAPTER.dump_python(value, mode='json')
    except Exception:
        try:
            return str(value)
        except Exception as e:
            return f'Unable to serialize: {e}'


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
    model_request_parameters: ModelRequestParameters, *, include_content: bool = True
) -> dict[str, AttributeValue]:
    serialized = serialize_any(model_request_parameters)
    if not include_content:
        # Two fields here are prompt text the user wrote, which is the "proprietary prompts" half of
        # what the setting withholds: the instructions (whose dynamic parts can be built from deps)
        # and the prompted-output template. Tool and output *schemas* stay: request structure rather
        # than message content, and `include_model_request_parameters=False` drops the attribute.
        for part in instruction_parts_of(serialized):
            part.pop('content', None)
        _blank_prompted_output_template(serialized)
    return {'model_request_parameters': to_json(serialized).decode()}


def _blank_prompted_output_template(serialized_parameters: Any) -> None:
    """Blank the prompted-output template, which is prompt text the user wrote."""
    if not isinstance(serialized_parameters, dict):
        return  # pragma: no cover
    parameters = cast('dict[str, Any]', serialized_parameters)
    if parameters.get('prompted_output_template') is not None:
        parameters['prompted_output_template'] = None


def instruction_parts_of(serialized_parameters: Any) -> list[dict[str, Any]]:
    """The serialized `instruction_parts`, or nothing when the shape isn't what we expect."""
    if not isinstance(serialized_parameters, dict):
        return []  # pragma: no cover
    parts = cast('Any', serialized_parameters).get('instruction_parts')
    if not isinstance(parts, list):
        return []
    return [part for part in cast('list[Any]', parts) if isinstance(part, dict)]


def event_to_dict(event: LogRecord) -> dict[str, Any]:
    if not event.body:
        body = {}  # pragma: no cover
    elif isinstance(event.body, Mapping):
        body = event.body
    else:
        body = {'body': event.body}
    return {**body, **(event.attributes or {})}


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


def record_exception(span: Span, error: BaseException, *, include_content: bool, escaped: bool = True) -> None:
    """Record `error` on `span` as an `exception` event.

    With content capture enabled this is the OTel SDK's own `Span.record_exception`. Without it,
    only the exception type is kept: the message and stack trace of an exception raised around
    a tool, a model request or an agent run can quote content the setting is meant to withhold --
    a tool retry carries the text the model sees, an exception chained from one repeats that text
    in its stack trace, a provider's error response can echo the request, and user exceptions may
    echo the rejected arguments. The type and `escaped` formatting match what
    `Span.record_exception` would have produced.
    """
    if include_content:
        span.record_exception(error, escaped=escaped)
        return
    error_type = type(error)
    type_name = (
        f'{error_type.__module__}.{error_type.__qualname__}'
        if error_type.__module__ != 'builtins'
        else error_type.__qualname__
    )
    # The SDK stringifies `escaped`, so match its shape rather than mixing attribute types.
    span.add_event('exception', attributes={'exception.type': type_name, 'exception.escaped': str(escaped)})


def set_error_status(span: Span, error: BaseException, *, include_content: bool) -> None:
    """Set `span`'s status to ERROR, describing it the way `use_span` would have.

    The SDK's description is `f'{type(exc).__name__}: {exc}'`, which repeats the message the
    exception event carries, so it is withheld alongside it when content capture is off.
    """
    span.set_status(
        Status(StatusCode.ERROR, description=f'{type(error).__name__}: {error}' if include_content else None)
    )


include_content_ctx: ContextVar[bool | None] = ContextVar('include_content', default=None)
"""Carries the open `chat` span's `include_content` to code that updates that span without holding
the settings -- `FallbackModel`, which refreshes `model_request_parameters` once it knows which
model answered. Set by `open_model_request_span` for the span's lifetime, so a refresh redacts the
instruction content of the model it picked the way the span was opened, rather than guessing from
what is already recorded. `None` means no instrumented request is open.
"""


@contextmanager
def record_uncaught_errors(span: Span, *, include_content: bool) -> Generator[None]:
    """Record exceptions leaving `span`'s scope the way `use_span` would have.

    For spans opened with `record_exception=False` and `set_status_on_exception=False`, which hands
    both jobs to the caller. Enter this around the span's whole scope -- the scope `use_span`
    covered -- so that failures while finalizing the span still mark it.
    """
    try:
        yield
    except Exception as error:
        record_exception(span, error, include_content=include_content, escaped=False)
        set_error_status(span, error, include_content=include_content)
        raise


@contextmanager
def open_model_request_span(
    settings: InstrumentationSettings,
    request_context: ModelRequestContext,
) -> Generator[tuple[Callable[[ModelResponse], None], ModelRequestContext]]:
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
        **model_request_parameters_attributes(prepared_parameters, include_content=settings.include_content),
        **get_agent_run_baggage_attributes(),
        'logfire.json_schema': to_json(
            {
                'type': 'object',
                'properties': {'model_request_parameters': {'type': 'object'}},
            }
        ).decode(),
    }

    tool_definitions = build_tool_definitions(prepared_parameters)
    if tool_definitions:
        attributes['gen_ai.tool.definitions'] = to_json(tool_definitions).decode()

    if prepared_settings:
        for key in MODEL_SETTING_ATTRIBUTES:
            if isinstance(value := prepared_settings.get(key), float | int):
                attributes[f'gen_ai.request.{key}'] = value

    record_metrics: Callable[[], None] | None = None
    include_content_token = include_content_ctx.set(settings.include_content)
    try:
        with settings.tracer.start_as_current_span(
            span_name,
            attributes=attributes,
            kind=SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            # `finish` is a closure rather than inline so we can (a) set result attributes
            # inside the `with span:` block — they attach to the span — and (b) call the
            # captured `record_metrics` in the outer `finally` AFTER the span closes,
            # so observability backends that aggregate metrics from span attributes
            # don't double-count.
            def finish(response: ModelResponse) -> None:
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
                    settings.record_metrics(response, price_calculation, metric_attributes)

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

                settings.handle_messages(prepared_request_context.messages, response, system, span, prepared_parameters)

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
                span.set_attributes(attributes_to_set)
                span.update_name(f'{operation} {request_model}')

            try:
                yield finish, prepared_request_context
            except Exception as e:
                # Stand in for what the two `..._on_exception=False` arguments turned off, matching
                # `use_span` exactly: it records only `Exception`, does not mark what it records as
                # escaped, and describes the status with the exception. A provider's error response
                # body travels in that message, so it follows `include_content`.
                record_exception(span, e, include_content=settings.include_content, escaped=False)
                set_error_status(span, e, include_content=settings.include_content)
                raise
    finally:
        include_content_ctx.reset(include_content_token)
        if record_metrics:
            record_metrics()


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
            version: The instrumentation version (1, 2, or 3+)

        Returns:
            InstrumentationConfig instance with version-appropriate settings
        """
        if version <= 2:
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
