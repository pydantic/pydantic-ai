from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, TextContent, VideoUrl

__all__ = 'EmbeddingModality', 'EmbeddingContentPart', 'EmbeddingContent', 'EmbeddingInput', 'embedding_parts'

EmbeddingModality: TypeAlias = Literal['text', 'image', 'audio', 'video', 'document']
"""A kind of content an embedding model may or may not be able to embed.

Every model supports `'text'`; support for the other modalities is declared per model by
[`EmbeddingModel.supported_modalities`][pydantic_ai.embeddings.EmbeddingModel.supported_modalities].
"""

EmbeddingContentPart: TypeAlias = str | TextContent | ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent
"""A single piece of content to embed, reusing the content types from [`pydantic_ai.messages`][pydantic_ai.messages].

This is [`UserContent`][pydantic_ai.messages.UserContent] without
[`CachePoint`][pydantic_ai.messages.CachePoint], which has no meaning for embeddings, and without
[`UploadedFile`][pydantic_ai.messages.UploadedFile], as provider file APIs aren't supported by the
embeddings interface yet. Python has no union subtraction, so the members are spelled out; when
`messages` gains a content type, decide here whether it can be embedded.
"""


@dataclass
class EmbeddingContent:
    """Multiple content parts that are embedded together into a single vector.

    Each item passed to [`embed()`][pydantic_ai.embeddings.Embedder.embed] yields exactly one
    embedding, so an `EmbeddingContent` is the way to say "combine these parts into one vector"
    rather than "embed each of these separately":

    ```python
    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import EmbeddingContent
    from pydantic_ai.messages import ImageUrl

    embedder = Embedder('google:gemini-embedding-2')
    image = ImageUrl(url='https://iili.io/3Hs4FMg.png')


    async def main():
        # Two inputs, two vectors
        result = await embedder.embed_documents(['a kiwi fruit', image])
        assert len(result.embeddings) == 2

        # One input, one vector combining the caption and the image
        result = await embedder.embed_documents(EmbeddingContent(['a kiwi fruit', image]))
        assert len(result.embeddings) == 1
    ```

    Only supported by models that support more than text; see
    [`EmbeddingModel.supported_modalities`][pydantic_ai.embeddings.EmbeddingModel.supported_modalities].
    """

    content: Sequence[EmbeddingContentPart]
    """The parts to embed together, in order."""

    def __post_init__(self) -> None:
        # A bare `str` satisfies `Sequence[EmbeddingContentPart]`, so `EmbeddingContent('a kiwi fruit')`
        # type-checks and would silently embed five characters as five parts.
        if isinstance(self.content, str):
            raise UserError(
                '`EmbeddingContent` takes a sequence of parts, not a single string. '
                "Wrap it in a list to embed it on its own, or pass it to `embed()` directly if it's the whole input."
            )
        # Every input must yield one embedding, and there is nothing to embed here; left unchecked this
        # reaches the provider as an empty `Content` and fails there instead.
        if not self.content:
            raise UserError('`EmbeddingContent` needs at least one part to embed.')


EmbeddingInput: TypeAlias = EmbeddingContentPart | EmbeddingContent
"""A single input to embed, yielding exactly one embedding vector."""


def embedding_parts(item: EmbeddingInput) -> Sequence[EmbeddingContentPart]:
    """The content parts that make up an input, so a single part and an `EmbeddingContent` can be handled alike."""
    return item.content if isinstance(item, EmbeddingContent) else [item]
