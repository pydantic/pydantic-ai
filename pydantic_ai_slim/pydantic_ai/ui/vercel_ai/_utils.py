"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    BaseToolReturnPart,
    ForceDownloadMode,
    ModelMessage,
    ProviderDetailsDelta,
    ToolReturnPart,
    tool_return_ta,
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

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'


class _PydanticAIMessageMetadata(BaseModel):
    """Schema for the `pydantic_ai` key in `UIMessage.metadata`.

    Internal protocol contract for round-tripping framework-side `ModelMessage` fields
    through Vercel AI `UIMessage.metadata`. Adding a field here extends the wire format;
    field changes need a deprecation cycle.

    Only `timestamp` is carried. `UIMessage.metadata` is client-controlled, so dumping
    server fields can leak infrastructure details (e.g. `provider_url`) and loading them
    trusts client input (e.g. a forged `provider_response_id` chaining into another user's
    conversation via OpenAI's `previous_response_id='auto'`). Exposing more fields needs an
    explicit, user-controlled opt-in -- see https://github.com/pydantic/pydantic-ai/issues/5174.
    """

    model_config = ConfigDict(extra='ignore')

    timestamp: datetime | None = None


def tool_return_output(part: BaseToolReturnPart) -> Any:
    """Serialize a tool return's full content for `ToolOutputAvailablePart.output`.

    Vercel's `output` field is `Any`, so the full return — file data included — is always dumped inline
    and rehydrated on load via `ToolReturnContent`'s discriminator (`_validate_tool_output`). No gating.
    The same function serializes both the `dump_messages` history path and the live event stream
    (`tool-output-available`), so files survive either round-trip.
    """
    return tool_return_ta.dump_python(part.content, mode='json')


def load_provider_metadata(provider_metadata: ProviderMetadata | None) -> dict[str, Any]:
    """Load the Pydantic AI metadata from the provider metadata."""
    return provider_metadata.get(PROVIDER_METADATA_KEY, {}) if provider_metadata else {}


def dump_provider_metadata(
    wrapper_key: str | None = PROVIDER_METADATA_KEY,
    **kwargs: ProviderDetailsDelta | ForceDownloadMode | str | None,
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
    """Dump application metadata plus framework message fields into `UIMessage.metadata`.

    May return an empty dict for a `ModelRequest` with no application metadata, since
    `ModelRequest.timestamp` is optional. For a `ModelResponse` the result always contains
    at least `{'pydantic_ai': {'timestamp': ...}}` since `ModelResponse.timestamp` is set.

    `UIMessage.metadata` is typed as `unknown` since AI SDK v5, so older frontends will
    silently ignore the field rather than reject the message.
    """
    metadata = dict(message.metadata) if message.metadata else {}

    pydantic_metadata = _PydanticAIMessageMetadata(timestamp=message.timestamp)
    if pydantic_metadata_dump := pydantic_metadata.model_dump(mode='json', exclude_defaults=True):
        metadata[PROVIDER_METADATA_KEY] = pydantic_metadata_dump
    return metadata


def apply_message_metadata(message: ModelMessage, metadata: object) -> None:
    """Load `UIMessage.metadata` back onto a Pydantic AI message.

    Only `timestamp` is restored from the `pydantic_ai` key; see `_PydanticAIMessageMetadata`
    for why other fields are excluded. Application metadata (non-`pydantic_ai` keys) is
    restored as-is onto `message.metadata`; an empty/missing app-side dict leaves any
    previously-attached `message.metadata` untouched, which matters when consecutive
    `UIMessage`s merge into the same `ModelRequest` and only one carries application fields.
    """
    if not is_str_dict(metadata):
        return

    raw_pydantic_metadata = metadata.get(PROVIDER_METADATA_KEY)
    if application_metadata := {k: v for k, v in metadata.items() if k != PROVIDER_METADATA_KEY}:
        message.metadata = application_metadata

    if not is_str_dict(raw_pydantic_metadata):
        return

    try:
        pydantic_metadata = _PydanticAIMessageMetadata.model_validate(raw_pydantic_metadata)
    except ValidationError:
        return

    if pydantic_metadata.timestamp is not None:
        message.timestamp = pydantic_metadata.timestamp


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
