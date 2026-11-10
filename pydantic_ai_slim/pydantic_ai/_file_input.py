"""Which files a model can be sent, as its profile's file-input flags say."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Literal, get_args

from ._utils import is_text_like_media_type
from .messages import (
    AudioMediaType,
    AudioUrl,
    BinaryContent,
    DocumentMediaType,
    DocumentUrl,
    FileUrl,
    ImageMediaType,
    ImageUrl,
    VideoMediaType,
    VideoUrl,
)

if TYPE_CHECKING:
    from .profiles import ModelProfile

FileInputKind = Literal['image', 'document', 'audio', 'video']
"""The kinds of file the profile's file-input flags cover."""

KNOWN_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset(get_args(ImageMediaType))
"""Every `ImageMediaType`: JPEG, PNG, GIF and WebP, which every provider that reads images reads."""

KNOWN_AUDIO_MEDIA_TYPES: frozenset[str] = frozenset(get_args(AudioMediaType))
"""Every `AudioMediaType`."""

KNOWN_VIDEO_MEDIA_TYPES: frozenset[str] = frozenset(get_args(VideoMediaType))
"""Every `VideoMediaType`."""

KNOWN_DOCUMENT_MEDIA_TYPES: frozenset[DocumentMediaType] = frozenset(get_args(DocumentMediaType))
"""Every `DocumentMediaType`: the default for `ModelProfile.supported_document_media_types`."""

TEXT_DOCUMENT_MEDIA_TYPES: frozenset[DocumentMediaType] = frozenset(
    media_type for media_type in KNOWN_DOCUMENT_MEDIA_TYPES if is_text_like_media_type(media_type)
)
"""The known document media types that are text, which some adapters inline into the prompt as text."""

SendableFile = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent
"""A file the flags judge. An `UploadedFile` is not one: only the provider it was uploaded to can read it."""


_MEDIA_TYPE_ALIASES = {'image/jpg': 'image/jpeg', 'audio/mp3': 'audio/mpeg', 'audio/x-wav': 'audio/wav'}
"""Spellings in common use that providers which read the registered media type also read."""


def unsupported_file_input(item: SendableFile, profile: ModelProfile) -> str | None:
    """What `item` is that the model cannot read, such as `'audio files'` or `'image/heic files'`, or `None`.

    A file is judged by its kind's flag, then by its media type when the profile lists the media types it reads
    for that kind (`supported_image_media_types` and the others): one outside the set can't be read. Without a
    set, a file is judged by its kind alone. Documents are judged by media type only for the types
    `DocumentMediaType` lists. A few common aliases, such as `image/jpg`, count as their registered media type.
    A URL whose media type can't be inferred, or is of another kind, is judged by kind alone. Binary content
    that is neither an image, audio, video nor a document is left to the model adapter.
    """
    supported: frozenset[str] | None
    match item:
        case ImageUrl() | BinaryContent(is_image=True):
            if not profile.get('supports_image_input', True):
                return 'images'
            supported, kind = profile.get('supported_image_media_types'), 'image'
        case AudioUrl() | BinaryContent(is_audio=True):
            if not profile.get('supports_audio_file_input', True):
                return 'audio files'
            supported, kind = profile.get('supported_audio_media_types'), 'audio'
        case VideoUrl() | BinaryContent(is_video=True):
            if not profile.get('supports_video_input', True):
                return 'video'
            supported, kind = profile.get('supported_video_media_types'), 'video'
        case DocumentUrl() | BinaryContent(is_document=True):
            if not profile.get('supports_document_input', True):
                return 'documents'
            supported, kind = profile.get('supported_document_media_types', KNOWN_DOCUMENT_MEDIA_TYPES), 'document'
        case _:
            return None
    media_type = inferred_media_type(item)
    if supported is None or media_type is None:
        return None
    return _unsupported_media_type(media_type, kind, supported)


def _unsupported_media_type(media_type: str, kind: FileInputKind, supported: frozenset[str]) -> str | None:
    judged_as = _MEDIA_TYPE_ALIASES.get(media_type, media_type)
    if kind == 'document':
        if judged_as not in KNOWN_DOCUMENT_MEDIA_TYPES:
            return None
    elif not judged_as.startswith(f'{kind}/'):
        # Of another kind, such as an `ImageUrl` given as a PDF, which the adapter sends as what it downloads as.
        return None
    return None if judged_as in supported else f'{media_type} files'


def unsupported_file_note(item: SendableFile, unsupported: str) -> str:
    """The text sent in place of a file a tool returned that the model cannot read.

    It names the file the way the tool result would have (`See file <identifier>.`), with its media type
    and, for content held in memory, its size, so the model can tell the user which file it could not read.
    """
    details: list[str] = []
    if media_type := inferred_media_type(item):
        details.append(media_type)
    if isinstance(item, BinaryContent):
        details.append(f'{len(item.data):,} bytes')
    described = f' ({", ".join(details)})' if details else ''
    return f'File {item.identifier}{described} was not sent: this model cannot read {unsupported}.'


def inferred_media_type(item: SendableFile) -> str | None:
    """The media type of `item`, or `None` for a URL whose media type can't be inferred and wasn't given."""
    if isinstance(item, FileUrl):
        try:
            return item.media_type
        except ValueError:
            # A URL whose extension says nothing about its media type, and none was given.
            return None
    return item.media_type
