"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import assert_never

from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    BaseToolReturnPart,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ProviderDetailsDelta,
    ToolReturnPart,
)
from pydantic_ai.ui.vercel_ai.request_types import (
    DynamicToolApprovalRequestedPart,
    DynamicToolApprovalRespondedPart,
    DynamicToolInputAvailablePart,
    DynamicToolInputStreamingPart,
    DynamicToolOutputAvailablePart,
    DynamicToolOutputDeniedPart,
    DynamicToolOutputErrorPart,
    ToolApprovalRequestedPart,
    ToolApprovalResponded,
    ToolApprovalRespondedPart,
    ToolInputAvailablePart,
    ToolInputStreamingPart,
    ToolOutputAvailablePart,
    ToolOutputDeniedPart,
    ToolOutputErrorPart,
    UIMessage,
)
from pydantic_ai.ui.vercel_ai.response_types import (
    DataChunk,
    FileChunk,
    ProviderMetadata,
    SourceDocumentChunk,
    SourceUrlChunk,
)
from pydantic_ai.usage import RequestUsage

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'


class _PydanticAIMessageMetadata(BaseModel):
    """Schema for the `pydantic_ai` key in `UIMessage.metadata`.

    Internal protocol contract for round-tripping `ModelRequest` / `ModelResponse`
    fields through Vercel AI `UIMessage.metadata`. Adding a field here extends the
    wire format; field changes need a deprecation cycle.
    """

    model_config = ConfigDict(extra='ignore')

    # `instructions` is deliberately absent: it's a server-side behavior-shaping field
    # re-resolved by the agent on every request, not message state to round-trip.
    # Carrying it here would expose confidential prompt guidance to the client.
    timestamp: datetime | None = None
    run_id: str | None = None
    conversation_id: str | None = None

    usage: RequestUsage | None = None
    model_name: str | None = None
    provider_name: str | None = None
    provider_url: str | None = None
    provider_details: dict[str, Any] | None = None
    provider_response_id: str | None = None
    finish_reason: FinishReason | None = None


def tool_return_output(part: BaseToolReturnPart) -> Any:
    """Extract the return value from a tool return part.

    If the model response object contains a 'return_value' key, return its value,
    otherwise return the entire output dict. This matches the streaming output format.
    """
    output = part.model_response_object()
    return output.get('return_value', output)


def load_provider_metadata(provider_metadata: ProviderMetadata | None) -> dict[str, Any]:
    """Load the Pydantic AI metadata from the provider metadata."""
    return provider_metadata.get(PROVIDER_METADATA_KEY, {}) if provider_metadata else {}


def dump_provider_metadata(
    wrapper_key: str | None = PROVIDER_METADATA_KEY,
    **kwargs: ProviderDetailsDelta | str,
) -> dict[str, Any] | None:
    """Dump provider metadata from keyword arguments.

    Args:
        wrapper_key: The key to wrap the metadata in. Defaults to 'pydantic_ai'.
        **kwargs: The keyword arguments to dump.

    Returns:
        The dumped provider metadata.

    Examples:
        >>> dump_provider_metadata(id='test_id', provider_name='test_provider', provider_details={'test_detail': 1})
        {'pydantic_ai': {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}}

        >>> dump_provider_metadata(wrapper_key='test', id='test_id', provider_name='test_provider', provider_details={'test_detail': 1})
        {'test': {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}}

        >>> dump_provider_metadata(wrapper_key=None, id='test_id', provider_name='test_provider', provider_details={'test_detail': 1})
        {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}
    """
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    if wrapper_key:
        return {wrapper_key: filtered} if filtered else None
    else:
        return filtered if filtered else None


def dump_message_metadata(message: ModelMessage) -> dict[str, Any]:
    """Dump application metadata plus Pydantic AI message fields into UIMessage.metadata.

    May return an empty dict for a `ModelRequest` with no application metadata and no
    framework fields set (`ModelRequest.timestamp` is optional). For a `ModelResponse` the
    result always contains at least `{'pydantic_ai': {'timestamp': ...}}` since
    `ModelResponse.timestamp` is non-optional.
    """
    metadata = dict(message.metadata) if message.metadata else {}

    if isinstance(message, ModelRequest):
        pydantic_metadata = _PydanticAIMessageMetadata(
            timestamp=message.timestamp,
            run_id=message.run_id,
            conversation_id=message.conversation_id,
        )
    elif isinstance(message, ModelResponse):
        pydantic_metadata = _PydanticAIMessageMetadata(
            timestamp=message.timestamp,
            run_id=message.run_id,
            conversation_id=message.conversation_id,
            usage=message.usage if message.usage.has_values() else None,
            model_name=message.model_name,
            provider_name=message.provider_name,
            provider_url=message.provider_url,
            provider_details=message.provider_details,
            provider_response_id=message.provider_response_id,
            finish_reason=message.finish_reason,
        )
    else:
        assert_never(message)

    if pydantic_metadata_dump := pydantic_metadata.model_dump(mode='json', exclude_defaults=True):
        metadata[PROVIDER_METADATA_KEY] = pydantic_metadata_dump
    return metadata


def apply_message_metadata(message: ModelMessage, metadata: object) -> None:
    """Load UIMessage.metadata back onto a Pydantic AI message.

    Behavior-shaping fields like `instructions` are neither dumped nor restored: the agent
    re-resolves them per request, so client-controlled history must not be a source of truth
    for them. A crafted `pydantic_ai` payload carrying such a field is dropped by the
    `_PydanticAIMessageMetadata` schema (`extra='ignore'`).
    """
    if not is_str_dict(metadata):
        return

    raw_pydantic_metadata = metadata.get(PROVIDER_METADATA_KEY)
    application_metadata = {k: v for k, v in metadata.items() if k != PROVIDER_METADATA_KEY} or None
    message.metadata = application_metadata

    if not is_str_dict(raw_pydantic_metadata):
        return

    try:
        pydantic_metadata = _PydanticAIMessageMetadata.model_validate(raw_pydantic_metadata)
    except ValidationError:
        return

    if pydantic_metadata.timestamp is not None:
        message.timestamp = pydantic_metadata.timestamp
    if pydantic_metadata.run_id is not None:
        message.run_id = pydantic_metadata.run_id
    if pydantic_metadata.conversation_id is not None:
        message.conversation_id = pydantic_metadata.conversation_id

    if isinstance(message, ModelResponse):
        if pydantic_metadata.usage is not None:
            message.usage = pydantic_metadata.usage
        if pydantic_metadata.model_name is not None:
            message.model_name = pydantic_metadata.model_name
        if pydantic_metadata.provider_name is not None:
            message.provider_name = pydantic_metadata.provider_name
        if pydantic_metadata.provider_url is not None:
            message.provider_url = pydantic_metadata.provider_url
        if pydantic_metadata.provider_details is not None:
            message.provider_details = pydantic_metadata.provider_details
        if pydantic_metadata.provider_response_id is not None:
            message.provider_response_id = pydantic_metadata.provider_response_id
        if pydantic_metadata.finish_reason is not None:
            message.finish_reason = pydantic_metadata.finish_reason


# Data-carrying chunk types that have a direct UIMessagePart counterpart in the
# Vercel AI SDK (as of ai@6.0.57).  Protocol-control chunks (StartChunk,
# FinishChunk, StartStepChunk, ToolInputStartChunk, etc.) are excluded because
# they could corrupt the SSE stream state if injected from tool metadata.
# See: https://github.com/vercel/ai/blob/ai%406.0.57/packages/ai/src/ui/ui-messages.ts#L75
#
# If the Vercel AI SDK introduces new data-carrying UIMessagePart variants,
# the corresponding chunk type should be added here.
_DATA_CHUNK_TYPES = (DataChunk, SourceUrlChunk, SourceDocumentChunk, FileChunk)


def iter_metadata_chunks(
    tool_result: ToolReturnPart,
) -> Iterator[DataChunk | SourceUrlChunk | SourceDocumentChunk | FileChunk]:
    """Yield data-carrying chunks from `tool_result.metadata` (or `.content`).

    Used by both the streaming and dump paths. Only `_DATA_CHUNK_TYPES` are
    yielded; protocol-control chunks are filtered out.
    """
    possible = tool_result.metadata or tool_result.content
    if isinstance(possible, _DATA_CHUNK_TYPES):
        yield possible
    elif isinstance(possible, (str, bytes)):  # pragma: no branch
        # Avoid iterable check for strings and bytes.
        pass
    elif isinstance(possible, Iterable):  # pragma: no branch
        for item in possible:  # type: ignore[reportUnknownMemberType]
            if isinstance(item, _DATA_CHUNK_TYPES):  # pragma: no branch
                yield item


_TOOL_PART_TYPES = (
    ToolInputStreamingPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    ToolApprovalRequestedPart,
    ToolApprovalRespondedPart,
    ToolOutputDeniedPart,
    DynamicToolInputStreamingPart,
    DynamicToolInputAvailablePart,
    DynamicToolOutputAvailablePart,
    DynamicToolOutputErrorPart,
    DynamicToolApprovalRequestedPart,
    DynamicToolApprovalRespondedPart,
    DynamicToolOutputDeniedPart,
)


_APPROVAL_RESPONDED_TYPES = (
    ToolApprovalRespondedPart,
    DynamicToolApprovalRespondedPart,
)


def iter_tool_approval_responses(
    messages: list[UIMessage],
) -> Iterator[tuple[str, ToolApprovalResponded]]:
    """Yield `(tool_call_id, approval)` for each responded tool approval in assistant messages.

    Only `approval-responded` parts are matched. `output-denied` parts have
    already been materialized into the message history by `load_messages()` and
    must not be re-processed as deferred results.
    """
    for msg in messages:
        if msg.role == 'assistant':
            for part in msg.parts:
                if isinstance(part, _APPROVAL_RESPONDED_TYPES) and isinstance(part.approval, ToolApprovalResponded):
                    yield part.tool_call_id, part.approval
