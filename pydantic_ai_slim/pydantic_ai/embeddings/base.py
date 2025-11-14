from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload

from pydantic_ai.embeddings.settings import EmbeddingSettings, merge_embedding_settings


class EmbeddingModel(ABC):
    """Abstract class for a model."""

    _settings: EmbeddingSettings | None = None

    def __init__(
        self,
        *,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        """Initialize the model with optional settings and profile.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
            profile: The model profile to use.
        """
        self._settings = settings

    @property
    def settings(self) -> EmbeddingSettings | None:
        """Get the model settings."""
        return self._settings

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str:
        """The embedding model provider."""
        raise NotImplementedError()

    @overload
    async def embed(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    async def embed(self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None) -> list[list[float]]:
        pass

    @abstractmethod
    async def embed(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        raise NotImplementedError

    def prepare_embed(
        self, documents: str | Sequence[str], settings: EmbeddingSettings | None = None
    ) -> tuple[Sequence[str], bool, EmbeddingSettings]:
        """Prepare the documents and settings for the embedding."""
        is_single_document = isinstance(documents, str)
        if is_single_document:
            documents = [documents]

        settings = merge_embedding_settings(self._settings, settings) or {}

        return documents, is_single_document, settings
