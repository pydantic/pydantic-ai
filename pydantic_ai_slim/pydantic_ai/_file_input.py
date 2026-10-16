"""Which files a model can be sent, as its profile's file-input flags say."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Literal, get_args

from ._utils import is_text_like_media_type
from .messages import AudioUrl, BinaryContent, DocumentMediaType, DocumentUrl, FileUrl, ImageUrl, VideoUrl

if TYPE_CHECKING:
    from .profiles import ModelProfile

FileInputKind = Literal['image', 'document', 'audio', 'video']
"""The kinds of file the profile's file-input flags cover."""

KNOWN_DOCUMENT_MEDIA_TYPES: frozenset[DocumentMediaType] = frozenset(get_args(DocumentMediaType))
"""Every `DocumentMediaType`: the default for `ModelProfile.supported_document_media_types`."""

TEXT_DOCUMENT_MEDIA_TYPES: frozenset[DocumentMediaType] = frozenset(
    media_type for media_type in KNOWN_DOCUMENT_MEDIA_TYPES if is_text_like_media_type(media_type)
)
"""The known document media types that are text, which some adapters inline into the prompt as text."""

SendableFile = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent
"""A file the flags judge. An `UploadedFile` is not one: only the provider it was uploaded to can read it."""


def unsupported_file_input(item: SendableFile, profile: ModelProfile) -> str | None:
    """What `item` is that the model cannot read, such as `'audio files'`, or `None` when it can be sent.

    Binary content that is neither an image, audio, video nor a document is left to the model adapter.
    """
    if isinstance(item, ImageUrl) or (isinstance(item, BinaryContent) and item.is_image):
        return None if profile.get('supports_image_input', True) else 'images'
    if isinstance(item, AudioUrl) or (isinstance(item, BinaryContent) and item.is_audio):
        return None if profile.get('supports_audio_file_input', True) else 'audio files'
    if isinstance(item, VideoUrl) or (isinstance(item, BinaryContent) and item.is_video):
        return None if profile.get('supports_video_input', True) else 'video'
    if isinstance(item, DocumentUrl) or (isinstance(item, BinaryContent) and item.is_document):
        if not profile.get('supports_document_input', True):
            return 'documents'
        media_type = _media_type(item)
        supported = profile.get('supported_document_media_types', KNOWN_DOCUMENT_MEDIA_TYPES)
        if media_type in KNOWN_DOCUMENT_MEDIA_TYPES and media_type not in supported:
            return f'{media_type} documents'
    return None


def unsupported_file_note(item: SendableFile, unsupported: str) -> str:
    """The text sent in place of a file a tool returned that the model cannot read.

    It names the file the way the tool result would have (`See file <identifier>.`), with its media type
    and, for content held in memory, its size, so the model can tell the user which file it could not read.
    """
    details: list[str] = []
    if media_type := _media_type(item):
        details.append(media_type)
    if isinstance(item, BinaryContent):
        details.append(f'{len(item.data):,} bytes')
    described = f' ({", ".join(details)})' if details else ''
    return f'File {item.identifier}{described} was not sent: this model cannot read {unsupported}.'


def _media_type(item: SendableFile) -> str | None:
    if isinstance(item, FileUrl):
        try:
            return item.media_type
        except ValueError:
            # A URL whose extension says nothing about its media type, and none was given.
            return None
    return item.media_type
