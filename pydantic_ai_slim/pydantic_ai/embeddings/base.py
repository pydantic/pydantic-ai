from abc import ABC, abstractmethod
from collections.abc import Sequence

from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings, merge_embedding_settings


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

    @abstractmethod
    async def embed(
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        raise NotImplementedError

    def prepare_embed(
        self, documents: str | Sequence[str], settings: EmbeddingSettings | None = None
    ) -> tuple[list[str], EmbeddingSettings]:
        """Prepare the documents and settings for the embedding."""
        documents = [documents] if isinstance(documents, str) else list(documents)

        settings = merge_embedding_settings(self._settings, settings) or {}

        return documents, settings

    async def max_input_tokens(self) -> int | None:
        """Get the maximum number of tokens that can be input to the model.

        `None` means unknown.
        """
        return None

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the text."""
        raise NotImplementedError
