"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from pydantic_ai._mcp_apps import get_mcp_app_tool_result
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
from pydantic_ai.ui._utils import INTERNAL_METADATA_KEY, MCP_APPS_UI_TYPE, mcp_app_ui_payload
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

MODEL_SETTINGS_DELTA_DATA_TYPE = 'data-model-settings-delta'
"""Data chunk type for changes to the thinking effort in effect."""

COMPACTION_DATA_TYPE = 'data-compaction'
"""Data chunk type for compaction parts."""

ENQUEUED_MESSAGES_DATA_TYPE = 'data-enqueued-messages'
"""Data chunk type for messages delivered into the run by `RunContext.enqueue`, encoded as `UIMessage`s."""

MCP_APPS_DATA_TYPE = f'data-{MCP_APPS_UI_TYPE}'
"""Data chunk type for the MCP Apps data of a tool call, keyed by its tool call ID."""

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
    **kwargs: ProviderDetailsDelta | ForceDownloadMode | str | list[Any] | None,
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


def _offset_raw_citation(citation: object, offset: int) -> object:
    """Shift a provider citation's positions in the answer text, which only OpenAI Responses annotations hold.

    Other providers' raw citations either hold positions in the cited document, which merging doesn't move, or
    aren't sent back to the provider, so they're left as they are.
    """
    if not offset or not is_str_dict(citation):
        return citation
    if (
        citation.get('type') == 'url_citation'
        and isinstance(start := citation.get('start_index'), int)
        and isinstance(end := citation.get('end_index'), int)
    ):
        return {**citation, 'start_index': start + offset, 'end_index': end + offset}
    if citation.get('type') == 'file_citation' and isinstance(index := citation.get('index'), int):
        return {**citation, 'index': index + offset}
    return citation


def dump_text_metadata(parts: Sequence[TextPart]) -> dict[str, Any] | None:
    """Dump provider metadata for a UI text part holding one or more consecutive text parts.

    The UI text part loads back as one text part, so it keeps the first part's fields, as merged text always has, with
    every part's citations: ours with their anchors moved onto the merged text, and the provider's, which replay
    sends back, joined under `provider_details['citations']`. The provider name comes from a part with citations
    when the first part has none, since replay only sends back citations from the provider that produced them.
    """
    first = parts[0]
    provider_name = first.provider_name or next((part.provider_name for part in parts if part.citations), None)
    provider_details = first.provider_details
    if len(parts) > 1:
        raw_citations: list[object] = []
        offset = 0
        for part in parts:
            citations = (part.provider_details or {}).get('citations')
            if part.provider_name == provider_name and isinstance(citations, list):
                raw_citations.extend(
                    _offset_raw_citation(citation, offset) for citation in cast(list[object], citations)
                )
            offset += len(part.content)
        if raw_citations:
            provider_details = {**(provider_details or {}), 'citations': raw_citations}
    return dump_provider_metadata(
        id=first.id,
        provider_name=provider_name,
        provider_details=provider_details,
        citations=dump_citations(merged_text_citations(parts)),
    )


def load_text_part(text: str, provider_meta: dict[str, Any]) -> TextPart:
    """Load the text part a UI text part holds, from its text and provider metadata."""
    # The client sends the metadata, and requests read `provider_details` as a dict.
    provider_details = provider_meta.get('provider_details')
    return TextPart(
        content=text,
        id=provider_meta.get('id'),
        provider_name=provider_meta.get('provider_name'),
        provider_details=cast(dict[str, Any], provider_details) if isinstance(provider_details, dict) else None,
        citations=load_citations(provider_meta.get('citations'), text),
    )


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
    yielded; protocol-control chunks are filtered out. MCP Apps data that
    `MCPToolset` attached becomes a single `data-mcp-apps` chunk.
    """
    if (app := get_mcp_app_tool_result(tool_result.metadata)) is not None:
        yield DataChunk(type=MCP_APPS_DATA_TYPE, id=tool_result.tool_call_id, data=mcp_app_ui_payload(app))
        return

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
