from typing_extensions import TypedDict


class EmbeddingSettings(TypedDict, total=False):
    """Common settings for configuring embedding models.

    These settings apply across multiple embedding model providers.
    Not all settings are supported by all models - check the specific
    model's documentation for details.

    Provider-specific settings classes (e.g.,
    [`OpenAIEmbeddingSettings`][pydantic_ai.embeddings.openai.OpenAIEmbeddingSettings],
    [`CohereEmbeddingSettings`][pydantic_ai.embeddings.cohere.CohereEmbeddingSettings])
    extend this with additional provider-prefixed options.
    """

    dimensions: int
    """The number of dimensions for the output embeddings.

    Supported by:

    * OpenAI
    * Cohere
    * Google
    * Sentence Transformers
    * Bedrock
    * VoyageAI
    """

    truncate: bool
    """Whether to truncate inputs that exceed the model's context length.

    Defaults to `False`. If `True`, inputs that are too long will be truncated.
    If `False`, an error will be raised for inputs that exceed the context length.

    For more control over truncation, you can use
    [`max_input_tokens()`][pydantic_ai.embeddings.Embedder.max_input_tokens] and
    [`count_tokens()`][pydantic_ai.embeddings.Embedder.count_tokens] to implement
    your own truncation logic.

    Provider-specific truncation settings (e.g., `cohere_truncate`, `bedrock_cohere_truncate`)
    take precedence if specified.

    Supported by:

    * Cohere
    * Bedrock (Cohere and Nova models)
    * VoyageAI
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Cohere
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Cohere
    """


def merge_embedding_settings(
    base: EmbeddingSettings | None, overrides: EmbeddingSettings | None
) -> EmbeddingSettings | None:
    """Merge two sets of embedding settings, with overrides taking precedence.

    Args:
        base: Base settings (typically from the embedder or model).
        overrides: Settings that should override the base (typically per-call settings).

    Returns:
        Merged settings, or `None` if both inputs are `None`.
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
