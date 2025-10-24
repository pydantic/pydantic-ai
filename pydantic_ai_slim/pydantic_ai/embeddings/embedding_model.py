from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload

from pydantic_ai.embeddings.settings import EmbeddingSettings


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
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    # TODO: Add system?

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @overload
    async def embed(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    async def embed(self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None) -> list[list[float]]:
        pass

    async def embed(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        raise NotImplementedError
