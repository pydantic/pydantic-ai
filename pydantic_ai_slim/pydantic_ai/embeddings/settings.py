from typing import Literal, TypedDict


class EmbeddingSettings(TypedDict, total=False):
    # TODO: May want to add extra_headres, extra_query, extra_body, timeout, etc.

    output_dimension: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Cohere
    * OpenAI
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Cohere
    """

    # We don't support embedding_types for now because it doesn't affect the user-facing API today..
    # embedding_types: Literal["float", "int8", "uint8", "binary", "ubinary", "base64"]

    input_type: Literal['search_document', 'search_query', 'classification', 'clustering', 'image']
    """The input type of the embedding.

    Supported by:

    * Cohere (See `cohere.EmbedInputType`)
    """

    # TODO: Add more?


def merge_embedding_settings(
    base: EmbeddingSettings | None, overrides: EmbeddingSettings | None
) -> EmbeddingSettings | None:
    """Merge two sets of embedding settings, preferring the overrides.

    A common use case is: merge_embedding_settings(<agent settings>, <run settings>)
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
