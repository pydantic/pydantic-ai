"""Utilities for handling Pydantic AI and Vercel data streams."""

from typing import Any

from pydantic_ai.messages import BaseToolReturnPart, ProviderDetailsDelta
from pydantic_ai.ui.vercel_ai.response_types import ProviderMetadata

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'


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
