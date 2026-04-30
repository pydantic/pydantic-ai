"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from typing import Any, Final, Literal

import pydantic
from typing_extensions import Required, TypedDict

from pydantic_ai.messages import (
    BaseToolReturnPart,
    BinaryContent,
    BinaryImage,
    MultiModalContent,
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

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'

MULTIMODAL_TOOL_RETURN_KIND: Final[str] = 'pydantic_ai_multimodal_tool_return'
"""Discriminator value for the multimodal tool-return envelope used in `ToolOutputAvailablePart.output`.

The Vercel AI SDK does not define a multimodal shape for tool outputs as of ai@6.0.57, so we wrap
multimodal `ToolReturnPart.content` in a structured envelope to preserve it across the round trip.
"""


class MultimodalToolOutputEnvelope(TypedDict, total=False):
    """Envelope shape for multimodal `ToolReturnPart.content` carried in `ToolOutputAvailablePart.output`."""

    pydantic_ai_kind: Required[Literal['pydantic_ai_multimodal_tool_return']]
    data: Any
    files: Required[list[dict[str, Any]]]


multi_modal_content_ta: pydantic.TypeAdapter[MultiModalContent] = pydantic.TypeAdapter(MultiModalContent)
"""TypeAdapter for serializing/deserializing `MultiModalContent` items in the tool-return envelope."""


def tool_return_output(part: BaseToolReturnPart) -> Any:
    """Extract the return value from a tool return part.

    If the model response object contains a 'return_value' key, return its value,
    otherwise return the entire output dict. This matches the streaming output format.
    """
    output = part.model_response_object()
    return output.get('return_value', output)


def tool_return_output_for_dump(part: BaseToolReturnPart) -> Any:
    """Like `tool_return_output`, but wraps multimodal items in a `MultimodalToolOutputEnvelope`.

    Used by `dump_messages` to preserve `MultiModalContent` items inside `ToolReturnPart.content`
    across the dump -> load round trip. The Vercel AI SDK has no native multimodal tool-output
    shape, so we encode files in a structured envelope keyed by `MULTIMODAL_TOOL_RETURN_KIND`.
    """
    if not part.files:
        return tool_return_output(part)
    output = part.model_response_object()
    data = output.get('return_value', output) if output else None
    return {
        'pydantic_ai_kind': MULTIMODAL_TOOL_RETURN_KIND,
        'data': data,
        'files': [multi_modal_content_ta.dump_python(f, mode='json') for f in part.files],
    }


def decode_multimodal_tool_output(output: Any) -> tuple[Any, list[MultiModalContent]] | None:
    """Decode a multimodal tool-return envelope; returns `None` if `output` is not an envelope."""
    if not isinstance(output, dict):
        return None
    envelope: dict[str, Any] = output  # pyright: ignore[reportUnknownVariableType]
    if envelope.get('pydantic_ai_kind') != MULTIMODAL_TOOL_RETURN_KIND:
        return None
    raw_files: list[Any] = envelope.get('files') or []
    files: list[MultiModalContent] = []
    for f in raw_files:
        item = multi_modal_content_ta.validate_python(f)
        # Narrow `BinaryContent` with an image media type to `BinaryImage` so round trips through
        # the discriminator union preserve the subclass (matches `BinaryContent.from_data_uri`).
        if isinstance(item, BinaryContent) and not isinstance(item, BinaryImage):
            item = BinaryContent.narrow_type(item)
        files.append(item)
    return envelope.get('data'), files


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
