"""Utilities for handling Pydantic AI and Vercel data streams."""

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic_ai.messages import ProviderDetailsDelta, ToolReturnPart
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, ProviderMetadata

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


def iter_metadata_chunks(tool_result: ToolReturnPart) -> Iterator[BaseChunk]:
    """Iterate over BaseChunks from ToolReturnPart metadata or content.

    Used by both the streaming path (``_event_stream.py``) and the dump path
    (``_adapter.py``) to extract user-supplied chunks from tool return metadata.

    Args:
        tool_result: The tool return part to extract chunks from.

    Yields:
        BaseChunk instances found in the metadata/content.
    """
    possible = tool_result.metadata or tool_result.content
    if isinstance(possible, BaseChunk):
        yield possible
    elif isinstance(possible, str | bytes):  # pragma: no branch
        # Avoid iterable check for strings and bytes.
        pass
    elif isinstance(possible, Iterable):  # pragma: no branch
        for item in possible:  # type: ignore[reportUnknownMemberType]
            if isinstance(item, BaseChunk):  # pragma: no branch
                yield item
