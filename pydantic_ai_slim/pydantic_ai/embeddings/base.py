from abc import ABC, abstractmethod
from collections.abc import Sequence

from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings, merge_embedding_settings


class EmbeddingModel(ABC):
    """Abstract base class for embedding models.

    Implement this class to create a custom embedding model. For most use cases,
    use one of the built-in implementations:

    - [`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel]
    - [`CohereEmbeddingModel`][pydantic_ai.embeddings.cohere.CohereEmbeddingModel]
    - [`GoogleEmbeddingModel`][pydantic_ai.embeddings.google.GoogleEmbeddingModel]
    - [`BedrockEmbeddingModel`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingModel]
    - [`SentenceTransformerEmbeddingModel`][pydantic_ai.embeddings.sentence_transformers.SentenceTransformerEmbeddingModel]
    """

    _settings: EmbeddingSettings | None = None

    def __init__(
        self,
        *,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        """Initialize the model with optional settings.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._settings = settings

    @property
    def settings(self) -> EmbeddingSettings | None:
        """Get the default settings for this model."""
        return self._settings

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the embedding model."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def system(self) -> str:
        """The embedding model provider/system identifier (e.g., 'openai', 'cohere')."""
        raise NotImplementedError()

    @abstractmethod
    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Generate embeddings for the given inputs.

        Args:
            inputs: A single string or sequence of strings to embed.
            input_type: Whether the inputs are queries or documents.
            settings: Optional settings to override the model's defaults.

        Returns:
            An [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing
            the embeddings and metadata.
        """
        raise NotImplementedError

    def prepare_embed(
        self, inputs: str | Sequence[str], settings: EmbeddingSettings | None = None
    ) -> tuple[list[str], EmbeddingSettings]:
        """Prepare the inputs and settings for embedding.

        This method normalizes inputs to a list and merges settings.
        Subclasses should call this at the start of their `embed()` implementation.

        Args:
            inputs: A single string or sequence of strings.
            settings: Optional settings to merge with defaults.

        Returns:
            A tuple of (normalized inputs list, merged settings).
        """
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)

        settings = merge_embedding_settings(self._settings, settings) or {}

        return inputs, settings

    async def max_input_tokens(self) -> int | None:
        """Get the maximum number of tokens that can be input to the model.

        Returns:
            The maximum token count, or `None` if unknown.
        """
        return None  # pragma: no cover

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The text to tokenize and count.

        Returns:
            The number of tokens.

        Raises:
            NotImplementedError: If the model doesn't support token counting.
            UserError: If the model or tokenizer is not supported.
        """
        raise NotImplementedError
