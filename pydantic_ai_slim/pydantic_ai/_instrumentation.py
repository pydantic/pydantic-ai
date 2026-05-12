from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

from opentelemetry._logs import LogRecord
from opentelemetry.baggage import get_baggage
from opentelemetry.util.types import AttributeValue
from pydantic import TypeAdapter
from pydantic_core import to_json

if TYPE_CHECKING:
    from typing_extensions import Self

    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import Model, ModelRequestParameters

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
    model_request_parameters: ModelRequestParameters,
) -> dict[str, AttributeValue]:
    return {'model_request_parameters': to_json(serialize_any(model_request_parameters)).decode()}


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
    # `opentelemetry` and `pydantic_graph` are hard deps of pydantic_ai_slim, so the
    # imports below cannot fail at runtime — kept lazy only to avoid pulling
    # OTel into `_instrumentation.py`'s import surface unless the fallback path actually fires.
    from opentelemetry.trace import INVALID_SPAN, get_current_span

    from pydantic_graph._utils import get_traceparent

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
