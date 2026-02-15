"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic_ai.messages import ProviderDetailsDelta, ToolReturnPart
from pydantic_ai.ui.vercel_ai.response_types import (
    DataChunk,
    FileChunk,
    ProviderMetadata,
    SourceDocumentChunk,
    SourceUrlChunk,
)

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'


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
    """Yield data-carrying chunks from ``tool_result.metadata`` (or ``.content``).

    Used by both the streaming and dump paths. Only ``_DATA_CHUNK_TYPES`` are
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
