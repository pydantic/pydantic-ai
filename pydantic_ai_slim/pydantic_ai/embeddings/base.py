from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import TextContent

from ._modality import embedding_modality
from .input import EmbeddingGroup, EmbeddingInput, EmbeddingModality, embedding_parts
from .profile import DEFAULT_EMBEDDING_PROFILE, EmbeddingModelProfile
from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings, merge_embedding_settings

_TEXT_ONLY: frozenset[EmbeddingModality] = frozenset({'text'})


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

    @property
    def profile(self) -> EmbeddingModelProfile:
        """What this model can accept: its modalities, and whether it can embed a group as one vector.

        Defaults to text only, one part per input. Models that accept more should override this, keyed
        by model name where support differs between the models a class covers.
        """
        return DEFAULT_EMBEDDING_PROFILE

    @abstractmethod
    async def embed(
        self,
        inputs: EmbeddingInput | Sequence[EmbeddingInput],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        """Generate embeddings for the given inputs.

        Args:
            inputs: A single input or sequence of inputs to embed, each yielding one embedding.
            input_type: Whether the inputs are queries or documents.
            settings: Optional settings to override the model's defaults.

        Returns:
            An [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing
            the embeddings and metadata.
        """
        raise NotImplementedError

    def prepare_embed(
        self, inputs: EmbeddingInput | Sequence[EmbeddingInput], settings: EmbeddingSettings | None = None
    ) -> tuple[list[EmbeddingInput], EmbeddingSettings]:
        """Prepare the inputs and settings for embedding.

        This method normalizes inputs to a list, checks them against the model's
        [`profile`][pydantic_ai.embeddings.EmbeddingModel.profile], and merges settings.
        Subclasses should call this at the start of their `embed()` implementation, or
        [`prepare_text_embed()`][pydantic_ai.embeddings.EmbeddingModel.prepare_text_embed] if they only support text.

        Args:
            inputs: A single input or sequence of inputs.
            settings: Optional settings to merge with defaults.

        Returns:
            A tuple of (normalized inputs list, merged settings). An item may be an
            [`EmbeddingGroup`][pydantic_ai.embeddings.EmbeddingGroup] of several parts if the profile
            says the model can embed one — combine its parts into a single vector, but don't pass it
            on as if it were one part.

        Raises:
            UserError: If an input uses a modality the model doesn't support, or groups several parts
                when the model embeds one part per input.
        """
        # Test for a single input rather than for a sequence: a batch is any iterable, including
        # generators and the array types embedding callers commonly hold their corpus in.
        items = [inputs] if isinstance(inputs, EmbeddingInput) else list(inputs)

        profile = self.profile
        supported = profile.get('supported_modalities', _TEXT_ONLY)
        grouped = profile.get('supports_grouped_inputs', False)

        for item in items:
            parts = embedding_parts(item)
            for part in parts:
                if (modality := embedding_modality(part)) not in supported:
                    # Attributed to Pydantic AI rather than to the model: some models we haven't
                    # implemented the modality for do support it.
                    raise UserError(
                        f'Pydantic AI does not support {modality} inputs for `{self.model_name}`. '
                        f'Supported modalities: {", ".join(sorted(supported))}.'
                    )
            # A group of one is nothing to combine, so it embeds like the bare part it wraps.
            if len(parts) > 1 and not grouped:
                raise UserError(
                    f'`{self.model_name}` embeds one part per input and cannot combine an `EmbeddingGroup` '
                    'into a single vector; pass the parts as separate inputs to embed them separately.'
                )

        settings = merge_embedding_settings(self._settings, settings) or {}

        return items, settings

    def prepare_text_embed(
        self, inputs: EmbeddingInput | Sequence[EmbeddingInput], settings: EmbeddingSettings | None = None
    ) -> tuple[list[EmbeddingInput], list[str], EmbeddingSettings]:
        """Prepare text-only inputs and settings for embedding.

        Like [`prepare_embed()`][pydantic_ai.embeddings.EmbeddingModel.prepare_embed], but additionally unwraps
        [`TextContent`][pydantic_ai.messages.TextContent] so implementations that only send text get plain strings.

        Args:
            inputs: A single input or sequence of inputs.
            settings: Optional settings to merge with defaults.

        Returns:
            A tuple of (normalized inputs list, text to send, merged settings). Pass the inputs list to
            [`EmbeddingResult.inputs`][pydantic_ai.embeddings.EmbeddingResult.inputs] so a `TextContent`'s
            metadata survives into the result, and the text to the provider.

        Raises:
            UserError: If an input isn't a single text part.
        """
        items, settings = self.prepare_embed(inputs, settings)

        texts: list[str] = []
        for item in items:
            # A group of one is nothing to combine, so it embeds like the bare part it wraps; a longer
            # one never reaches here, as `prepare_embed()` refuses it unless the profile allows groups.
            if isinstance(item, EmbeddingGroup) and len(item.content) == 1:
                item = item.content[0]

            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, TextContent):
                texts.append(item.content)
            else:
                raise UserError(f'`{self.model_name}` only supports plain text inputs, got `{type(item).__name__}`.')

        return items, texts, settings

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
