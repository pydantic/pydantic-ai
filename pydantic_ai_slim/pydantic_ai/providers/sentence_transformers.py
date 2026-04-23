from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai.providers import Provider


class SentenceTransformersProvider(Provider[Any]):
    """Provider for Sentence Transformers API."""

    @property
    def name(self) -> str:
        """The provider name."""
        return 'sentence-transformers'  # pragma: no cover

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        raise NotImplementedError('The Sentence Transformers provider does not have a base URL as it runs locally.')

    @property
    def client(self) -> Any:
        """The client for the provider."""
        return None  # pragma: no cover
