from typing_extensions import assert_never

from pydantic_ai.messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, TextContent, VideoUrl

from .input import EmbeddingContentPart, EmbeddingModality


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
    elif isinstance(part, BinaryContent):
        if part.is_image:
            return 'image'
        elif part.is_audio:
            return 'audio'
        elif part.is_video:
            return 'video'
        else:
            return 'document'
    else:
        assert_never(part)
