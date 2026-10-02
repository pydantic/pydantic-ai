"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, NonNegativeInt, TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict

from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import (
    BaseToolReturnPart,
    Citation,
    ForceDownloadMode,
    ModelMessage,
    ProviderDetailsDelta,
    TextPart,
    ToolReturnPart,
    WebCitationSource,
    tool_return_ta,
)
from pydantic_ai.ui._utils import INTERNAL_METADATA_KEY
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

TOOL_AVAILABILITY_DELTA_DATA_TYPE = 'data-tool-availability-delta'
"""Data chunk type for tool availability changes."""

COMPACTION_DATA_TYPE = 'data-compaction'
"""Data chunk type for compaction parts."""

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
    and rehydrated on load through the `ToolReturnContent` union (`_validate_tool_output`). No gating.
    The same function serializes both the `dump_messages` history path and the live event stream
    (`tool-output-available`), so files survive either round-trip.
    """
    return tool_return_ta.dump_python(part.content, mode='json')


def load_provider_metadata(provider_metadata: ProviderMetadata | None) -> dict[str, Any]:
    """Load the Pydantic AI metadata from the provider metadata."""
    return provider_metadata.get(PROVIDER_METADATA_KEY, {}) if provider_metadata else {}


def dump_provider_metadata(
    wrapper_key: str | None = PROVIDER_METADATA_KEY,
    **kwargs: ProviderDetailsDelta | ForceDownloadMode | list[dict[str, Any]] | str | None,
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


_citations_ta: TypeAdapter[list[Citation]] = TypeAdapter(list[Citation])
_citation_ta: TypeAdapter[Citation] = TypeAdapter(Citation)


def dump_citations(citations: Sequence[Citation] | None) -> list[dict[str, Any]] | None:
    """Dump citations to JSON-compatible data for a text part's provider metadata."""
    return _citations_ta.dump_python(list(citations), mode='json') if citations else None


def load_citations(data: object, text: str) -> list[Citation] | None:
    """Load citations from a text part's provider metadata.

    Provider metadata is client-controlled, so each citation that doesn't validate, or whose anchor doesn't fit in
    `text`, is dropped instead of failing the request. The text and the other citations still load.
    """
    if not isinstance(data, list):
        return None
    citations: list[Citation] = []
    for item in cast(list[object], data):
        try:
            citation = _citation_ta.validate_python(item)
        except ValidationError:
            continue
        if not citation.anchor or citation.anchor.end <= len(text):
            citations.append(citation)
    return citations or None


def _offset_citations(citations: Sequence[Citation], offset: int) -> list[Citation]:
    return [
        replace(
            citation,
            anchor=replace(citation.anchor, start=citation.anchor.start + offset, end=citation.anchor.end + offset),
        )
        if citation.anchor and offset
        else citation
        for citation in citations
    ]


def merged_text_citations(parts: Sequence[TextPart]) -> list[Citation]:
    """Citations of text parts merged into one UI text part, with anchors shifted onto the merged text."""
    citations: list[Citation] = []
    offset = 0
    for part in parts:
        citations.extend(_offset_citations(part.citations or [], offset))
        offset += len(part.content)
    return citations


def dump_text_metadata(parts: Sequence[TextPart]) -> dict[str, Any] | None:
    """Dump provider metadata for a UI text part holding one or more consecutive text parts.

    The first part's fields and all citations go at the top level. When several parts were merged and any of them
    has citations, `parts` also keeps each one's length, fields and citations, so that loading the message gives back
    the original parts, with each citation on the part it belongs to.
    """
    first = parts[0]
    split = len(parts) > 1 and any(part.citations for part in parts)
    return dump_provider_metadata(
        id=first.id,
        provider_name=first.provider_name,
        provider_details=first.provider_details,
        citations=dump_citations(merged_text_citations(parts)),
        parts=[_dump_text_part_metadata(part) for part in parts] if split else None,
    )


def _dump_text_part_metadata(part: TextPart) -> dict[str, Any]:
    metadata = dump_provider_metadata(
        wrapper_key=None,
        id=part.id,
        provider_name=part.provider_name,
        provider_details=part.provider_details,
        citations=dump_citations(part.citations),
    )
    return {'length': len(part.content), **(metadata or {})}


class _TextPartMetadata(TypedDict):
    length: NonNegativeInt
    id: NotRequired[str | None]
    provider_name: NotRequired[str | None]
    provider_details: NotRequired[dict[str, Any] | None]
    citations: NotRequired[object]


_text_parts_metadata_ta: TypeAdapter[list[_TextPartMetadata]] = TypeAdapter(list[_TextPartMetadata])


def load_text_parts(text: str, provider_meta: dict[str, Any]) -> list[TextPart]:
    """Load the text parts held by a UI text part, from its text and provider metadata.

    Provider metadata is client-controlled, so if `parts` doesn't validate or its lengths don't add up to `text`, the
    text loads as one part with the top-level fields and citations.
    """
    if (data := provider_meta.get('parts')) is not None:
        try:
            entries = _text_parts_metadata_ta.validate_python(data)
        except ValidationError:
            entries = None
        if entries and sum(entry['length'] for entry in entries) == len(text):
            parts: list[TextPart] = []
            start = 0
            for entry in entries:
                content = text[start : start + entry['length']]
                start += entry['length']
                parts.append(
                    TextPart(
                        content=content,
                        id=entry.get('id'),
                        provider_name=entry.get('provider_name'),
                        provider_details=entry.get('provider_details'),
                        citations=load_citations(entry.get('citations'), content),
                    )
                )
            return parts
    return [
        TextPart(
            content=text,
            id=provider_meta.get('id'),
            provider_name=provider_meta.get('provider_name'),
            provider_details=provider_meta.get('provider_details'),
            citations=load_citations(provider_meta.get('citations'), text),
        )
    ]


def iter_citation_source_chunks(citations: Iterable[Citation], seen_urls: set[str]) -> Iterator[SourceUrlChunk]:
    """Yield a `source-url` chunk for each web source URL not in `seen_urls`, adding it.

    Document sources have no chunk: Vercel AI's `source-document` requires a media type and title,
    which provider document citations don't reliably carry. They still round-trip with the text part.
    """
    for citation in citations:
        for source in citation.sources:
            if isinstance(source, WebCitationSource) and source.url not in seen_urls:
                seen_urls.add(source.url)
                yield SourceUrlChunk(source_id=source.url, url=source.url, title=source.title)


def dump_message_metadata(message: ModelMessage) -> dict[str, Any]:
    """Dump application metadata plus framework message fields into `UIMessage.metadata`.

    May return an empty dict for a `ModelRequest` with no application metadata, since
    `ModelRequest.timestamp` is optional. For a `ModelResponse` the result always contains
    at least `{'pydantic_ai': {'timestamp': ...}}` since `ModelResponse.timestamp` is set.

    `UIMessage.metadata` is typed as `unknown` since AI SDK v5, so older frontends will
    silently ignore the field rather than reject the message.
    """
    metadata = (
        {key: value for key, value in message.metadata.items() if key != INTERNAL_METADATA_KEY}
        if message.metadata
        else {}
    )

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
    if application_metadata := {
        key: value for key, value in metadata.items() if key not in (PROVIDER_METADATA_KEY, INTERNAL_METADATA_KEY)
    }:
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
DATA_CHUNK_TYPES = (DataChunk, SourceUrlChunk, SourceDocumentChunk, FileChunk)


def iter_metadata_chunks(
    tool_result: ToolReturnPart,
) -> Iterator[DataChunk | SourceUrlChunk | SourceDocumentChunk | FileChunk]:
    """Yield data-carrying chunks from `tool_result.metadata` (or `.content`).

    Used by both the streaming and dump paths. Only `DATA_CHUNK_TYPES` are
    yielded; protocol-control chunks are filtered out.
    """
    possible = tool_result.metadata or tool_result.content
    if isinstance(possible, DATA_CHUNK_TYPES):
        yield possible
    elif isinstance(possible, (str, bytes)):  # pragma: no branch
        # Avoid iterable check for strings and bytes.
        pass
    elif isinstance(possible, Iterable):  # pragma: no branch
        for item in possible:  # type: ignore[reportUnknownMemberType]
            if isinstance(item, DATA_CHUNK_TYPES):  # pragma: no branch
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
