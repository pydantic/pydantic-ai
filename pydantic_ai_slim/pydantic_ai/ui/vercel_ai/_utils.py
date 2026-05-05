"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, TypeVar, cast

from pydantic import TypeAdapter

from pydantic_ai.messages import (
    BaseToolReturnPart,
    FinishReason,
    ModelMessage,
    ModelRequest,
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

datetime_ta: TypeAdapter[datetime] = TypeAdapter(datetime)
request_usage_ta: TypeAdapter[RequestUsage] = TypeAdapter(RequestUsage)
T = TypeVar('T')
_FINISH_REASONS: set[FinishReason] = {'stop', 'length', 'content_filter', 'tool_call', 'error'}


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


def dump_message_metadata(message: ModelMessage) -> dict[str, Any] | None:
    """Dump application metadata plus Pydantic AI message fields into UIMessage.metadata."""
    metadata = dict(message.metadata) if message.metadata else {}
    pydantic_metadata: dict[str, Any] = {}

    if message.timestamp is not None:
        pydantic_metadata['timestamp'] = message.timestamp.isoformat()
    if message.run_id is not None:
        pydantic_metadata['run_id'] = message.run_id
    if message.conversation_id is not None:
        pydantic_metadata['conversation_id'] = message.conversation_id

    if isinstance(message, ModelRequest):
        if message.instructions is not None:
            pydantic_metadata['instructions'] = message.instructions
    else:
        if message.usage.has_values():
            pydantic_metadata['usage'] = request_usage_ta.dump_python(message.usage, mode='json', exclude_defaults=True)
        if message.model_name is not None:
            pydantic_metadata['model_name'] = message.model_name
        if message.provider_name is not None:
            pydantic_metadata['provider_name'] = message.provider_name
        if message.provider_url is not None:
            pydantic_metadata['provider_url'] = message.provider_url
        if message.provider_details is not None:
            pydantic_metadata['provider_details'] = message.provider_details
        if message.provider_response_id is not None:
            pydantic_metadata['provider_response_id'] = message.provider_response_id
        if message.finish_reason is not None:
            pydantic_metadata['finish_reason'] = message.finish_reason

    if pydantic_metadata:
        metadata[PROVIDER_METADATA_KEY] = pydantic_metadata

    return metadata or None


def apply_message_metadata(message: ModelMessage, metadata: Any) -> None:
    """Load UIMessage.metadata back onto a Pydantic AI message."""
    if not isinstance(metadata, dict):
        return
    metadata = cast(dict[str, Any], metadata)

    raw_pydantic_metadata = metadata.get(PROVIDER_METADATA_KEY)
    if not isinstance(raw_pydantic_metadata, dict):
        message.metadata = metadata or None
        return
    pydantic_metadata = cast(dict[str, Any], raw_pydantic_metadata)

    message.metadata = {k: v for k, v in metadata.items() if k != PROVIDER_METADATA_KEY} or None

    if timestamp := _validate_with(datetime_ta, pydantic_metadata.get('timestamp')):
        message.timestamp = timestamp
    if run_id := _metadata_str(pydantic_metadata, 'run_id'):
        message.run_id = run_id
    if conversation_id := _metadata_str(pydantic_metadata, 'conversation_id'):
        message.conversation_id = conversation_id

    if isinstance(message, ModelRequest):
        if instructions := _metadata_str(pydantic_metadata, 'instructions'):
            message.instructions = instructions
    else:
        if usage := _validate_with(request_usage_ta, pydantic_metadata.get('usage')):
            message.usage = usage
        if model_name := _metadata_str(pydantic_metadata, 'model_name'):
            message.model_name = model_name
        if provider_name := _metadata_str(pydantic_metadata, 'provider_name'):
            message.provider_name = provider_name
        if provider_url := _metadata_str(pydantic_metadata, 'provider_url'):
            message.provider_url = provider_url
        if provider_details := _metadata_dict(pydantic_metadata, 'provider_details'):
            message.provider_details = provider_details
        if provider_response_id := _metadata_str(pydantic_metadata, 'provider_response_id'):
            message.provider_response_id = provider_response_id
        if finish_reason := _metadata_finish_reason(pydantic_metadata):
            message.finish_reason = finish_reason


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _metadata_dict(metadata: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = metadata.get(key)
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _metadata_finish_reason(metadata: dict[str, Any]) -> FinishReason | None:
    value = _metadata_str(metadata, 'finish_reason')
    return value if value in _FINISH_REASONS else None


def _validate_with(type_adapter: TypeAdapter[T], value: Any) -> T | None:
    if value is None:
        return None
    try:
        return type_adapter.validate_python(value)
    except Exception:  # pragma: no cover
        return None


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
