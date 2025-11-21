from typing_extensions import TypedDict


class EmbeddingSettings(TypedDict, total=False):
    """Settings to configure an embedding model.

    Here we include only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.
    """

    dimensions: int
    """The number of dimensions the resulting output embeddings should have.

    Supported by:

    * OpenAI
    * Cohere
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
    """Merge two sets of embedding settings, preferring the overrides.

    A common use case is: merge_embedding_settings(<embedder settings>, <run settings>)
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
