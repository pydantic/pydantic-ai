"""Utilities for converting between Pydantic AI and Vercel AI data structures."""

from typing import Any

from pydantic_ai.messages import ProviderDetailsDelta
from pydantic_ai.ui.vercel_ai.response_types import ProviderMetadata

__all__ = []

PROVIDER_METADATA_KEY = 'pydantic_ai'


def load_provider_metadata(provider_metadata: ProviderMetadata | None) -> dict[str, Any]:
    """Load the Pydantic AI metadata from the provider metadata."""
    return provider_metadata.get(PROVIDER_METADATA_KEY, {}) if provider_metadata else {}


def dump_provider_metadata(
    wrapper_key: str | None = PROVIDER_METADATA_KEY,
    **kwargs: ProviderDetailsDelta | str,
) -> dict[str, Any] | None:
    """Dump provider metadata from keyword arguments."""
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    if wrapper_key:
        return {wrapper_key: filtered} if filtered else None
    else:
        return filtered if filtered else None
