from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic_ai.messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, TextContent, VideoUrl

EmbeddingModality = Literal['text', 'image', 'audio', 'video', 'document']
"""A kind of content an embedding model may or may not be able to embed.

Every model supports `'text'`; support for the other modalities is declared per model by
[`EmbeddingModel.supported_modalities`][pydantic_ai.embeddings.EmbeddingModel.supported_modalities].
"""

EmbeddingFile: TypeAlias = ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent
"""A non-text item to embed, reusing the multi-modal content types from [`pydantic_ai.messages`][pydantic_ai.messages].

[`UploadedFile`][pydantic_ai.messages.UploadedFile] is not included: provider file APIs aren't
supported by the embeddings interface yet.
"""

EmbeddingContentPart: TypeAlias = str | TextContent | EmbeddingFile
"""A single piece of content to embed."""


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


EmbeddingInput: TypeAlias = EmbeddingContentPart | EmbeddingContent
"""A single input to embed, yielding exactly one embedding vector."""


def embedding_parts(item: EmbeddingInput) -> Sequence[EmbeddingContentPart]:
    """The content parts that make up an input, so a single part and an `EmbeddingContent` can be handled alike."""
    return item.content if isinstance(item, EmbeddingContent) else [item]


def embedding_modality(part: EmbeddingContentPart) -> EmbeddingModality:
    """The modality of a content part, to check it against a model's supported modalities."""
    if isinstance(part, str | TextContent):
        return 'text'
    elif isinstance(part, ImageUrl):
        return 'image'
    elif isinstance(part, AudioUrl):
        return 'audio'
    elif isinstance(part, VideoUrl):
        return 'video'
    elif isinstance(part, DocumentUrl):
        return 'document'
    elif part.is_image:
        return 'image'
    elif part.is_audio:
        return 'audio'
    elif part.is_video:
        return 'video'
    else:
        return 'document'
