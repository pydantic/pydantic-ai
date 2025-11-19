from __future__ import annotations as _annotations

import base64
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from datetime import datetime
from mimetypes import guess_type
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast, overload

import pydantic
import pydantic_core
from genai_prices import calc_price, types as genai_types
from opentelemetry._events import Event  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import deprecated

from . import _otel_messages, _utils
from ._utils import generate_tool_call_id as _generate_tool_call_id, now_utc as _now_utc
from .exceptions import UnexpectedModelBehavior
from .usage import RequestUsage

if TYPE_CHECKING:
    from .models.instrumented import InstrumentationSettings


AudioMediaType: TypeAlias = Literal['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aiff', 'audio/aac']
ImageMediaType: TypeAlias = Literal['image/jpeg', 'image/png', 'image/gif', 'image/webp']
DocumentMediaType: TypeAlias = Literal[
    'application/pdf',
    'text/plain',
    'text/csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'text/html',
    'text/markdown',
    'application/msword',
    'application/vnd.ms-excel',
]
VideoMediaType: TypeAlias = Literal[
    'video/x-matroska',
    'video/quicktime',
    'video/mp4',
    'video/webm',
    'video/x-flv',
    'video/mpeg',
    'video/x-ms-wmv',
    'video/3gpp',
]

AudioFormat: TypeAlias = Literal['wav', 'mp3', 'oga', 'flac', 'aiff', 'aac']
ImageFormat: TypeAlias = Literal['jpeg', 'png', 'gif', 'webp']
DocumentFormat: TypeAlias = Literal['csv', 'doc', 'docx', 'html', 'md', 'pdf', 'txt', 'xls', 'xlsx']
VideoFormat: TypeAlias = Literal['mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp']

FinishReason: TypeAlias = Literal[
    'stop',
    'length',
    'content_filter',
    'tool_call',
    'error',
]
"""Reason the model finished generating the response, normalized to OpenTelemetry values."""


@dataclass(repr=False)
class SystemPromptPart:
    """A system prompt, generally written by the application developer.

    This gives the model context and guidance on how to respond.
    """

    content: str
    """The content of the prompt."""

    _: KW_ONLY

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the prompt."""

    dynamic_ref: str | None = None
    """The ref of the dynamic system prompt function that generated this part.

    Only set if system prompt is dynamic, see [`system_prompt`][pydantic_ai.Agent.system_prompt] for more information.
    """

    part_kind: Literal['system-prompt'] = 'system-prompt'
    """Part type identifier, this is available on all parts as a discriminator."""

    def otel_event(self, settings: InstrumentationSettings) -> Event:
        return Event(
            'gen_ai.system.message',
            body={'role': 'system', **({'content': self.content} if settings.include_content else {})},
        )

    def otel_message_parts(self, settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]:
        return [_otel_messages.TextPart(type='text', **{'content': self.content} if settings.include_content else {})]

    __repr__ = _utils.dataclasses_no_defaults_repr


def _multi_modal_content_identifier(identifier: str | bytes) -> str:
    """Generate stable identifier for multi-modal content to help LLM in finding a specific file in tool call responses."""
    if isinstance(identifier, str):
        identifier = identifier.encode('utf-8')
    return hashlib.sha1(identifier).hexdigest()[:6]


@dataclass(init=False, repr=False)
class FileUrl(ABC):
    """Abstract base class for any URL-based file."""

    url: str
    """The URL of the file."""

    _: KW_ONLY

    force_download: bool = False
    """For OpenAI and Google APIs it:

    * If True, the file is downloaded and the data is sent to the model as bytes.
    * If False, the URL is sent directly to the model and no download is performed.
    """

    vendor_metadata: dict[str, Any] | None = None
    """Vendor-specific metadata for the file.

    Supported by:
    - `GoogleModel`: `VideoUrl.vendor_metadata` is used as `video_metadata`: https://ai.google.dev/gemini-api/docs/video-understanding#customize-video-processing
    - `OpenAIChatModel`, `OpenAIResponsesModel`: `ImageUrl.vendor_metadata['detail']` is used as `detail` setting for images
    """

    _media_type: Annotated[str | None, pydantic.Field(alias='media_type', default=None, exclude=True)] = field(
        compare=False, default=None
    )

    _identifier: Annotated[str | None, pydantic.Field(alias='identifier', default=None, exclude=True)] = field(
        compare=False, default=None
    )

    def __init__(
        self,
        url: str,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self._media_type = media_type
        self._identifier = identifier
        self.force_download = force_download
        self.vendor_metadata = vendor_metadata

    @pydantic.computed_field
    @property
    def media_type(self) -> str:
        """Return the media type of the file, based on the URL or the provided `media_type`."""
        return self._media_type or self._infer_media_type()

    @pydantic.computed_field
    @property
    def identifier(self) -> str:
        """The identifier of the file, such as a unique ID.

        This identifier can be provided to the model in a message to allow it to refer to this file in a tool call argument,
        and the tool can look up the file in question by iterating over the message history and finding the matching `FileUrl`.

        This identifier is only automatically passed to the model when the `FileUrl` is returned by a tool.
        If you're passing the `FileUrl` as a user message, it's up to you to include a separate text part with the identifier,
        e.g. "This is file <identifier>:" preceding the `FileUrl`.

        It's also included in inline-text delimiters for providers that require inlining text documents, so the model can
        distinguish multiple files.
        """
        return self._identifier or _multi_modal_content_identifier(self.url)

    @abstractmethod
    def _infer_media_type(self) -> str:
        """Infer the media type of the file based on the URL."""
        raise NotImplementedError

    @property
    @abstractmethod
    def format(self) -> str:
        """The file format."""
        raise NotImplementedError

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(init=False, repr=False)
class VideoUrl(FileUrl):
    """A URL to a video."""

    url: str
    """The URL of the video."""

    _: KW_ONLY

    kind: Literal['video-url'] = 'video-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        kind: Literal['video-url'] = 'video-url',
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        _media_type: str | None = None,
        _identifier: str | None = None,
    ) -> None:
        super().__init__(
            url=url,
            force_download=force_download,
            vendor_metadata=vendor_metadata,
            media_type=media_type or _media_type,
            identifier=identifier or _identifier,
        )
        self.kind = kind

    def _infer_media_type(self) -> VideoMediaType:
        """Return the media type of the video, based on the url."""
        if self.url.endswith('.mkv'):
            return 'video/x-matroska'
        elif self.url.endswith('.mov'):
            return 'video/quicktime'
        elif self.url.endswith('.mp4'):
            return 'video/mp4'
        elif self.url.endswith('.webm'):
            return 'video/webm'
        elif self.url.endswith('.flv'):
            return 'video/x-flv'
        elif self.url.endswith(('.mpeg', '.mpg')):
            return 'video/mpeg'
        elif self.url.endswith('.wmv'):
            return 'video/x-ms-wmv'
        elif self.url.endswith('.three_gp'):
            return 'video/3gpp'
        # Assume that YouTube videos are mp4 because there would be no extension
        # to infer from. This should not be a problem, as Gemini disregards media
        # type for YouTube URLs.
        elif self.is_youtube:
            return 'video/mp4'
        else:
            raise ValueError(
                f'Could not infer media type from video URL: {self.url}. Explicitly provide a `media_type` instead.'
            )

    @property
    def is_youtube(self) -> bool:
        """True if the URL has a YouTube domain."""
        return self.url.startswith(('https://youtu.be/', 'https://youtube.com/', 'https://www.youtube.com/'))

    @property
    def format(self) -> VideoFormat:
        """The file format of the video.

        The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
        """
        return _video_format_lookup[self.media_type]


@dataclass(init=False, repr=False)
class AudioUrl(FileUrl):
    """A URL to an audio file."""

    url: str
    """The URL of the audio file."""

    _: KW_ONLY

    kind: Literal['audio-url'] = 'audio-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        kind: Literal['audio-url'] = 'audio-url',
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        _media_type: str | None = None,
        _identifier: str | None = None,
    ) -> None:
        super().__init__(
            url=url,
            force_download=force_download,
            vendor_metadata=vendor_metadata,
            media_type=media_type or _media_type,
            identifier=identifier or _identifier,
        )
        self.kind = kind

    def _infer_media_type(self) -> AudioMediaType:
        """Return the media type of the audio file, based on the url.

        References:
        - Gemini: https://ai.google.dev/gemini-api/docs/audio#supported-formats
        """
        if self.url.endswith('.mp3'):
            return 'audio/mpeg'
        if self.url.endswith('.wav'):
            return 'audio/wav'
        if self.url.endswith('.flac'):
            return 'audio/flac'
        if self.url.endswith('.oga'):
            return 'audio/ogg'
        if self.url.endswith('.aiff'):
            return 'audio/aiff'
        if self.url.endswith('.aac'):
            return 'audio/aac'

        raise ValueError(
            f'Could not infer media type from audio URL: {self.url}. Explicitly provide a `media_type` instead.'
        )

    @property
    def format(self) -> AudioFormat:
        """The file format of the audio file."""
        return _audio_format_lookup[self.media_type]


@dataclass(init=False, repr=False)
class ImageUrl(FileUrl):
    """A URL to an image."""

    url: str
    """The URL of the image."""

    _: KW_ONLY

    kind: Literal['image-url'] = 'image-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        kind: Literal['image-url'] = 'image-url',
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        _media_type: str | None = None,
        _identifier: str | None = None,
    ) -> None:
        super().__init__(
            url=url,
            force_download=force_download,
            vendor_metadata=vendor_metadata,
            media_type=media_type or _media_type,
            identifier=identifier or _identifier,
        )
        self.kind = kind

    def _infer_media_type(self) -> ImageMediaType:
        """Return the media type of the image, based on the url."""
        if self.url.endswith(('.jpg', '.jpeg')):
            return 'image/jpeg'
        elif self.url.endswith('.png'):
            return 'image/png'
        elif self.url.endswith('.gif'):
            return 'image/gif'
        elif self.url.endswith('.webp'):
            return 'image/webp'
        else:
            raise ValueError(
                f'Could not infer media type from image URL: {self.url}. Explicitly provide a `media_type` instead.'
            )

    @property
    def format(self) -> ImageFormat:
        """The file format of the image.

        The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
        """
        return _image_format_lookup[self.media_type]


@dataclass(init=False, repr=False)
class DocumentUrl(FileUrl):
    """The URL of the document."""

    url: str
    """The URL of the document."""

    _: KW_ONLY

    kind: Literal['document-url'] = 'document-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        *,
        media_type: str | None = None,
        identifier: str | None = None,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        kind: Literal['document-url'] = 'document-url',
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        _media_type: str | None = None,
        _identifier: str | None = None,
    ) -> None:
        super().__init__(
            url=url,
            force_download=force_download,
            vendor_metadata=vendor_metadata,
            media_type=media_type or _media_type,
            identifier=identifier or _identifier,
        )
        self.kind = kind

    def _infer_media_type(self) -> str:
        """Return the media type of the document, based on the url."""
        # Common document types are hardcoded here as mime-type support for these
        # extensions varies across operating systems.
        if self.url.endswith(('.md', '.mdx', '.markdown')):
            return 'text/markdown'
        elif self.url.endswith('.asciidoc'):
            return 'text/x-asciidoc'
        elif self.url.endswith('.txt'):
            return 'text/plain'
        elif self.url.endswith('.pdf'):
            return 'application/pdf'
        elif self.url.endswith('.rtf'):
            return 'application/rtf'
        elif self.url.endswith('.doc'):
            return 'application/msword'
        elif self.url.endswith('.docx'):
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif self.url.endswith('.xls'):
            return 'application/vnd.ms-excel'
        elif self.url.endswith('.xlsx'):
            return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        type_, _ = guess_type(self.url)
        if type_ is None:
            raise ValueError(
                f'Could not infer media type from document URL: {self.url}. Explicitly provide a `media_type` instead.'
            )
        return type_

    @property
    def format(self) -> DocumentFormat:
        """The file format of the document.

        The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
        """
        media_type = self.media_type
        try:
            return _document_format_lookup[media_type]
        except KeyError as e:
            raise ValueError(f'Unknown document media type: {media_type}') from e


@dataclass(init=False, repr=False)
class BinaryContent:
    """Binary content, e.g. an audio or image file."""

    data: bytes
    """The binary data."""

    _: KW_ONLY

    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
    """The media type of the binary data."""

    vendor_metadata: dict[str, Any] | None = None
    """Vendor-specific metadata for the file.

    Supported by:
    - `GoogleModel`: `BinaryContent.vendor_metadata` is used as `video_metadata`: https://ai.google.dev/gemini-api/docs/video-understanding#customize-video-processing
    - `OpenAIChatModel`, `OpenAIResponsesModel`: `BinaryContent.vendor_metadata['detail']` is used as `detail` setting for images
    """

    _identifier: Annotated[str | None, pydantic.Field(alias='identifier', default=None, exclude=True)] = field(
        compare=False, default=None
    )

    kind: Literal['binary'] = 'binary'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        data: bytes,
        *,
        media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str,
        identifier: str | None = None,
        vendor_metadata: dict[str, Any] | None = None,
        kind: Literal['binary'] = 'binary',
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        _identifier: str | None = None,
    ) -> None:
        self.data = data
        self.media_type = media_type
        self._identifier = identifier or _identifier
        self.vendor_metadata = vendor_metadata
        self.kind = kind

    @staticmethod
    def narrow_type(bc: BinaryContent) -> BinaryContent | BinaryImage:
        """Narrow the type of the `BinaryContent` to `BinaryImage` if it's an image."""
        if bc.is_image:
            return BinaryImage(
                data=bc.data,
                media_type=bc.media_type,
                identifier=bc.identifier,
                vendor_metadata=bc.vendor_metadata,
            )
        else:
            return bc

    @classmethod
    def from_data_uri(cls, data_uri: str) -> BinaryContent:
        """Create a `BinaryContent` from a data URI."""
        prefix = 'data:'
        if not data_uri.startswith(prefix):
            raise ValueError('Data URI must start with "data:"')
        media_type, data = data_uri[len(prefix) :].split(';base64,', 1)
        return cls.narrow_type(cls(data=base64.b64decode(data), media_type=media_type))

    @pydantic.computed_field
    @property
    def identifier(self) -> str:
        """Identifier for the binary content, such as a unique ID.

        This identifier can be provided to the model in a message to allow it to refer to this file in a tool call argument,
        and the tool can look up the file in question by iterating over the message history and finding the matching `BinaryContent`.

        This identifier is only automatically passed to the model when the `BinaryContent` is returned by a tool.
        If you're passing the `BinaryContent` as a user message, it's up to you to include a separate text part with the identifier,
        e.g. "This is file <identifier>:" preceding the `BinaryContent`.

        It's also included in inline-text delimiters for providers that require inlining text documents, so the model can
        distinguish multiple files.
        """
        return self._identifier or _multi_modal_content_identifier(self.data)

    @property
    def data_uri(self) -> str:
        """Convert the `BinaryContent` to a data URI."""
        return f'data:{self.media_type};base64,{base64.b64encode(self.data).decode()}'

    @property
    def is_audio(self) -> bool:
        """Return `True` if the media type is an audio type."""
        return self.media_type.startswith('audio/')

    @property
    def is_image(self) -> bool:
        """Return `True` if the media type is an image type."""
        return self.media_type.startswith('image/')

    @property
    def is_video(self) -> bool:
        """Return `True` if the media type is a video type."""
        return self.media_type.startswith('video/')

    @property
    def is_document(self) -> bool:
        """Return `True` if the media type is a document type."""
        return self.media_type in _document_format_lookup

    @property
    def format(self) -> str:
        """The file format of the binary content."""
        try:
            if self.is_audio:
                return _audio_format_lookup[self.media_type]
            elif self.is_image:
                return _image_format_lookup[self.media_type]
            elif self.is_video:
                return _video_format_lookup[self.media_type]
            else:
                return _document_format_lookup[self.media_type]
        except KeyError as e:
            raise ValueError(f'Unknown media type: {self.media_type}') from e

    __repr__ = _utils.dataclasses_no_defaults_repr


class BinaryImage(BinaryContent):
    """Binary content that's guaranteed to be an image."""

    def __init__(
        self,
        data: bytes,
        *,
        media_type: str,
        identifier: str | None = None,
        vendor_metadata: dict[str, Any] | None = None,
        # Required for inline-snapshot which expects all dataclass `__init__` methods to take all field names as kwargs.
        kind: Literal['binary'] = 'binary',
        _identifier: str | None = None,
    ):
        super().__init__(
            data=data, media_type=media_type, identifier=identifier or _identifier, vendor_metadata=vendor_metadata
        )

        if not self.is_image:
            raise ValueError('`BinaryImage` must be have a media type that starts with "image/"')  # pragma: no cover


@dataclass
class CachePoint:
    """A cache point marker for prompt caching.

    Can be inserted into UserPromptPart.content to mark cache boundaries.
    Models that don't support caching will filter these out.

    Supported by:

    - Anthropic
    """

    kind: Literal['cache-point'] = 'cache-point'
    """Type identifier, this is available on all parts as a discriminator."""

    ttl: Literal['5m', '1h'] = '5m'
    """The cache time-to-live, either "5m" (5 minutes) or "1h" (1 hour).

    Supported by:

    * Anthropic. See https://docs.claude.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration for more information."""


MultiModalContent = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent
UserContent: TypeAlias = str | MultiModalContent | CachePoint


@dataclass(repr=False)
class ToolReturn:
    """A structured return value for tools that need to provide both a return value and custom content to the model.

    This class allows tools to return complex responses that include:
    - A return value for actual tool return
    - Custom content (including multi-modal content) to be sent to the model as a UserPromptPart
    - Optional metadata for application use
    """

    return_value: Any
    """The return value to be used in the tool response."""

    _: KW_ONLY

    content: str | Sequence[UserContent] | None = None
    """The content to be sent to the model as a UserPromptPart."""

    metadata: Any = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    kind: Literal['tool-return'] = 'tool-return'

    __repr__ = _utils.dataclasses_no_defaults_repr


_document_format_lookup: dict[str, DocumentFormat] = {
    'application/pdf': 'pdf',
    'text/plain': 'txt',
    'text/csv': 'csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'text/html': 'html',
    'text/markdown': 'md',
    'application/msword': 'doc',
    'application/vnd.ms-excel': 'xls',
}
_audio_format_lookup: dict[str, AudioFormat] = {
    'audio/mpeg': 'mp3',
    'audio/wav': 'wav',
    'audio/flac': 'flac',
    'audio/ogg': 'oga',
    'audio/aiff': 'aiff',
    'audio/aac': 'aac',
}
_image_format_lookup: dict[str, ImageFormat] = {
    'image/jpeg': 'jpeg',
    'image/png': 'png',
    'image/gif': 'gif',
    'image/webp': 'webp',
}
_video_format_lookup: dict[str, VideoFormat] = {
    'video/x-matroska': 'mkv',
    'video/quicktime': 'mov',
    'video/mp4': 'mp4',
    'video/webm': 'webm',
    'video/x-flv': 'flv',
    'video/mpeg': 'mpeg',
    'video/x-ms-wmv': 'wmv',
    'video/3gpp': 'three_gp',
}


@dataclass(repr=False)
class UserPromptPart:
    """A user prompt, generally written by the end user.

    Content comes from the `user_prompt` parameter of [`Agent.run`][pydantic_ai.agent.AbstractAgent.run],
    [`Agent.run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], and [`Agent.run_stream`][pydantic_ai.agent.AbstractAgent.run_stream].
    """

    content: str | Sequence[UserContent]
    """The content of the prompt."""

    _: KW_ONLY

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the prompt."""

    part_kind: Literal['user-prompt'] = 'user-prompt'
    """Part type identifier, this is available on all parts as a discriminator."""

    def otel_event(self, settings: InstrumentationSettings) -> Event:
        content = [{'kind': part.pop('type'), **part} for part in self.otel_message_parts(settings)]
        for part in content:
            if part['kind'] == 'binary' and 'content' in part:
                part['binary_content'] = part.pop('content')
        content = [
            part['content'] if part == {'kind': 'text', 'content': part.get('content')} else part for part in content
        ]
        if content in ([{'kind': 'text'}], [self.content]):
            content = content[0]
        return Event('gen_ai.user.message', body={'content': content, 'role': 'user'})

    def otel_message_parts(self, settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]:
        parts: list[_otel_messages.MessagePart] = []
        content: Sequence[UserContent] = [self.content] if isinstance(self.content, str) else self.content
        for part in content:
            if isinstance(part, str):
                parts.append(
                    _otel_messages.TextPart(type='text', **({'content': part} if settings.include_content else {}))
                )
            elif isinstance(part, ImageUrl | AudioUrl | DocumentUrl | VideoUrl):
                parts.append(
                    _otel_messages.MediaUrlPart(
                        type=part.kind,
                        **{'url': part.url} if settings.include_content else {},
                    )
                )
            elif isinstance(part, BinaryContent):
                converted_part = _otel_messages.BinaryDataPart(type='binary', media_type=part.media_type)
                if settings.include_content and settings.include_binary_content:
                    converted_part['content'] = base64.b64encode(part.data).decode()
                parts.append(converted_part)
            elif isinstance(part, CachePoint):
                # CachePoint is a marker, not actual content - skip it for otel
                pass
            else:
                parts.append({'type': part.kind})  # pragma: no cover
        return parts

    __repr__ = _utils.dataclasses_no_defaults_repr


tool_return_ta: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    Any, config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
)


@dataclass(repr=False)
class BaseToolReturnPart:
    """Base class for tool return parts."""

    tool_name: str
    """The name of the "tool" was called."""

    content: Any
    """The return value."""

    tool_call_id: str = field(default_factory=_generate_tool_call_id)
    """The tool call identifier, this is used by some models including OpenAI.

    In case the tool call id is not provided by the model, Pydantic AI will generate a random one.
    """

    _: KW_ONLY

    metadata: Any = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp, when the tool returned."""

    def model_response_str(self) -> str:
        """Return a string representation of the content for the model."""
        if isinstance(self.content, str):
            return self.content
        else:
            return tool_return_ta.dump_json(self.content).decode()

    def model_response_object(self) -> dict[str, Any]:
        """Return a dictionary representation of the content, wrapping non-dict types appropriately."""
        # gemini supports JSON dict return values, but no other JSON types, hence we wrap anything else in a dict
        json_content = tool_return_ta.dump_python(self.content, mode='json')
        if isinstance(json_content, dict):
            return json_content  # type: ignore[reportUnknownReturn]
        else:
            return {'return_value': json_content}

    def otel_event(self, settings: InstrumentationSettings) -> Event:
        return Event(
            'gen_ai.tool.message',
            body={
                **({'content': self.content} if settings.include_content else {}),
                'role': 'tool',
                'id': self.tool_call_id,
                'name': self.tool_name,
            },
        )

    def otel_message_parts(self, settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]:
        from .models.instrumented import InstrumentedModel

        part = _otel_messages.ToolCallResponsePart(
            type='tool_call_response',
            id=self.tool_call_id,
            name=self.tool_name,
        )

        if settings.include_content and self.content is not None:
            part['result'] = InstrumentedModel.serialize_any(self.content)

        return [part]

    def has_content(self) -> bool:
        """Return `True` if the tool return has content."""
        return self.content is not None  # pragma: no cover

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ToolReturnPart(BaseToolReturnPart):
    """A tool return message, this encodes the result of running a tool."""

    _: KW_ONLY

    part_kind: Literal['tool-return'] = 'tool-return'
    """Part type identifier, this is available on all parts as a discriminator."""


@dataclass(repr=False)
class BuiltinToolReturnPart(BaseToolReturnPart):
    """A tool return message from a built-in tool."""

    _: KW_ONLY

    provider_name: str | None = None
    """The name of the provider that generated the response."""

    part_kind: Literal['builtin-tool-return'] = 'builtin-tool-return'
    """Part type identifier, this is available on all parts as a discriminator."""


error_details_ta = pydantic.TypeAdapter(list[pydantic_core.ErrorDetails], config=pydantic.ConfigDict(defer_build=True))


@dataclass(repr=False)
class RetryPromptPart:
    """A message back to a model asking it to try again.

    This can be sent for a number of reasons:

    * Pydantic validation of tool arguments failed, here content is derived from a Pydantic
      [`ValidationError`][pydantic_core.ValidationError]
    * a tool raised a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception
    * no tool was found for the tool name
    * the model returned plain text when a structured response was expected
    * Pydantic validation of a structured response failed, here content is derived from a Pydantic
      [`ValidationError`][pydantic_core.ValidationError]
    * an output validator raised a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception
    """

    content: list[pydantic_core.ErrorDetails] | str
    """Details of why and how the model should retry.

    If the retry was triggered by a [`ValidationError`][pydantic_core.ValidationError], this will be a list of
    error details.
    """

    _: KW_ONLY

    tool_name: str | None = None
    """The name of the tool that was called, if any."""

    tool_call_id: str = field(default_factory=_generate_tool_call_id)
    """The tool call identifier, this is used by some models including OpenAI.

    In case the tool call id is not provided by the model, Pydantic AI will generate a random one.
    """

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp, when the retry was triggered."""

    part_kind: Literal['retry-prompt'] = 'retry-prompt'
    """Part type identifier, this is available on all parts as a discriminator."""

    def model_response(self) -> str:
        """Return a string message describing why the retry is requested."""
        if isinstance(self.content, str):
            if self.tool_name is None:
                description = f'Validation feedback:\n{self.content}'
            else:
                description = self.content
        else:
            json_errors = error_details_ta.dump_json(self.content, exclude={'__all__': {'ctx'}}, indent=2)
            plural = isinstance(self.content, list) and len(self.content) != 1
            description = (
                f'{len(self.content)} validation error{"s" if plural else ""}:\n```json\n{json_errors.decode()}\n```'
            )
        return f'{description}\n\nFix the errors and try again.'

    def otel_event(self, settings: InstrumentationSettings) -> Event:
        if self.tool_name is None:
            return Event('gen_ai.user.message', body={'content': self.model_response(), 'role': 'user'})
        else:
            return Event(
                'gen_ai.tool.message',
                body={
                    **({'content': self.model_response()} if settings.include_content else {}),
                    'role': 'tool',
                    'id': self.tool_call_id,
                    'name': self.tool_name,
                },
            )

    def otel_message_parts(self, settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]:
        if self.tool_name is None:
            return [_otel_messages.TextPart(type='text', content=self.model_response())]
        else:
            part = _otel_messages.ToolCallResponsePart(
                type='tool_call_response',
                id=self.tool_call_id,
                name=self.tool_name,
            )

            if settings.include_content:
                part['result'] = self.model_response()

            return [part]

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelRequestPart = Annotated[
    SystemPromptPart | UserPromptPart | ToolReturnPart | RetryPromptPart, pydantic.Discriminator('part_kind')
]
"""A message part sent by Pydantic AI to a model."""


@dataclass(repr=False)
class ModelRequest:
    """A request generated by Pydantic AI and sent to a model, e.g. a message from the Pydantic AI app to the model."""

    parts: Sequence[ModelRequestPart]
    """The parts of the user message."""

    _: KW_ONLY

    instructions: str | None = None
    """The instructions for the model."""

    kind: Literal['request'] = 'request'
    """Message type identifier, this is available on all parts as a discriminator."""

    run_id: str | None = None
    """The unique identifier of the agent run in which this message originated."""

    metadata: dict[str, Any] | None = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    @classmethod
    def user_text_prompt(cls, user_prompt: str, *, instructions: str | None = None) -> ModelRequest:
        """Create a `ModelRequest` with a single user prompt as text."""
        return cls(parts=[UserPromptPart(user_prompt)], instructions=instructions)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class TextPart:
    """A plain text response from a model."""

    content: str
    """The text content of the response."""

    _: KW_ONLY

    id: str | None = None
    """An optional identifier of the text part."""

    part_kind: Literal['text'] = 'text'
    """Part type identifier, this is available on all parts as a discriminator."""

    def has_content(self) -> bool:
        """Return `True` if the text content is non-empty."""
        return bool(self.content)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ThinkingPart:
    """A thinking response from a model."""

    content: str
    """The thinking content of the response."""

    _: KW_ONLY

    id: str | None = None
    """The identifier of the thinking part."""

    signature: str | None = None
    """The signature of the thinking.

    Supported by:

    * Anthropic (corresponds to the `signature` field)
    * Bedrock (corresponds to the `signature` field)
    * Google (corresponds to the `thought_signature` field)
    * OpenAI (corresponds to the `encrypted_content` field)
    """

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Signatures are only sent back to the same provider.
    """

    part_kind: Literal['thinking'] = 'thinking'
    """Part type identifier, this is available on all parts as a discriminator."""

    def has_content(self) -> bool:
        """Return `True` if the thinking content is non-empty."""
        return bool(self.content)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class FilePart:
    """A file response from a model."""

    content: Annotated[BinaryContent, pydantic.AfterValidator(BinaryImage.narrow_type)]
    """The file content of the response."""

    _: KW_ONLY

    id: str | None = None
    """The identifier of the file part."""

    provider_name: str | None = None
    """The name of the provider that generated the response.
    """

    part_kind: Literal['file'] = 'file'
    """Part type identifier, this is available on all parts as a discriminator."""

    def has_content(self) -> bool:
        """Return `True` if the file content is non-empty."""
        return bool(self.content.data)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class BaseToolCallPart:
    """A tool call from a model."""

    tool_name: str
    """The name of the tool to call."""

    args: str | dict[str, Any] | None = None
    """The arguments to pass to the tool.

    This is stored either as a JSON string or a Python dictionary depending on how data was received.
    """

    tool_call_id: str = field(default_factory=_generate_tool_call_id)
    """The tool call identifier, this is used by some models including OpenAI.

    In case the tool call id is not provided by the model, Pydantic AI will generate a random one.
    """

    _: KW_ONLY

    id: str | None = None
    """An optional identifier of the tool call part, separate from the tool call ID.

    This is used by some APIs like OpenAI Responses."""

    def args_as_dict(self) -> dict[str, Any]:
        """Return the arguments as a Python dictionary.

        This is just for convenience with models that require dicts as input.
        """
        if not self.args:
            return {}
        if isinstance(self.args, dict):
            return self.args
        args = pydantic_core.from_json(self.args)
        assert isinstance(args, dict), 'args should be a dict'
        return cast(dict[str, Any], args)

    def args_as_json_str(self) -> str:
        """Return the arguments as a JSON string.

        This is just for convenience with models that require JSON strings as input.
        """
        if not self.args:
            return '{}'
        if isinstance(self.args, str):
            return self.args
        return pydantic_core.to_json(self.args).decode()

    def has_content(self) -> bool:
        """Return `True` if the arguments contain any data."""
        if isinstance(self.args, dict):
            # TODO: This should probably return True if you have the value False, or 0, etc.
            #   It makes sense to me to ignore empty strings, but not sure about empty lists or dicts
            return any(self.args.values())
        else:
            return bool(self.args)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ToolCallPart(BaseToolCallPart):
    """A tool call from a model."""

    _: KW_ONLY

    part_kind: Literal['tool-call'] = 'tool-call'
    """Part type identifier, this is available on all parts as a discriminator."""


@dataclass(repr=False)
class BuiltinToolCallPart(BaseToolCallPart):
    """A tool call to a built-in tool."""

    _: KW_ONLY

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Built-in tool calls are only sent back to the same provider.
    """

    part_kind: Literal['builtin-tool-call'] = 'builtin-tool-call'
    """Part type identifier, this is available on all parts as a discriminator."""


ModelResponsePart = Annotated[
    TextPart | ToolCallPart | BuiltinToolCallPart | BuiltinToolReturnPart | ThinkingPart | FilePart,
    pydantic.Discriminator('part_kind'),
]
"""A message part returned by a model."""


@dataclass(repr=False)
class ModelResponse:
    """A response from a model, e.g. a message from the model to the Pydantic AI app."""

    parts: Sequence[ModelResponsePart]
    """The parts of the model message."""

    _: KW_ONLY

    usage: RequestUsage = field(default_factory=RequestUsage)
    """Usage information for the request.

    This has a default to make tests easier, and to support loading old messages where usage will be missing.
    """

    model_name: str | None = None
    """The name of the model that generated the response."""

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the response.

    If the model provides a timestamp in the response (as OpenAI does) that will be used.
    """

    kind: Literal['response'] = 'response'
    """Message type identifier, this is available on all parts as a discriminator."""

    provider_name: str | None = None
    """The name of the LLM provider that generated the response."""

    provider_details: Annotated[
        dict[str, Any] | None,
        # `vendor_details` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        pydantic.Field(validation_alias=pydantic.AliasChoices('provider_details', 'vendor_details')),
    ] = None
    """Additional provider-specific details in a serializable format.

    This allows storing selected vendor-specific data that isn't mapped to standard ModelResponse fields.
    For OpenAI models, this may include 'logprobs', 'finish_reason', etc.
    """

    provider_response_id: Annotated[
        str | None,
        # `vendor_id` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        pydantic.Field(validation_alias=pydantic.AliasChoices('provider_response_id', 'vendor_id')),
    ] = None
    """request ID as specified by the model provider. This can be used to track the specific request to the model."""

    finish_reason: FinishReason | None = None
    """Reason the model finished generating the response, normalized to OpenTelemetry values."""

    run_id: str | None = None
    """The unique identifier of the agent run in which this message originated."""

    metadata: dict[str, Any] | None = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    @property
    def text(self) -> str | None:
        """Get the text in the response."""
        texts: list[str] = []
        last_part: ModelResponsePart | None = None
        for part in self.parts:
            if isinstance(part, TextPart):
                # Adjacent text parts should be joined together, but if there are parts in between
                # (like built-in tool calls) they should have newlines between them
                if isinstance(last_part, TextPart):
                    texts[-1] += part.content
                else:
                    texts.append(part.content)
            last_part = part
        if not texts:
            return None

        return '\n\n'.join(texts)

    @property
    def thinking(self) -> str | None:
        """Get the thinking in the response."""
        thinking_parts = [part.content for part in self.parts if isinstance(part, ThinkingPart)]
        if not thinking_parts:
            return None
        return '\n\n'.join(thinking_parts)

    @property
    def files(self) -> list[BinaryContent]:
        """Get the files in the response."""
        return [part.content for part in self.parts if isinstance(part, FilePart)]

    @property
    def images(self) -> list[BinaryImage]:
        """Get the images in the response."""
        return [file for file in self.files if isinstance(file, BinaryImage)]

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Get the tool calls in the response."""
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    @property
    def builtin_tool_calls(self) -> list[tuple[BuiltinToolCallPart, BuiltinToolReturnPart]]:
        """Get the builtin tool calls and results in the response."""
        calls = [part for part in self.parts if isinstance(part, BuiltinToolCallPart)]
        if not calls:
            return []
        returns_by_id = {part.tool_call_id: part for part in self.parts if isinstance(part, BuiltinToolReturnPart)}
        return [
            (call_part, returns_by_id[call_part.tool_call_id])
            for call_part in calls
            if call_part.tool_call_id in returns_by_id
        ]

    @deprecated('`price` is deprecated, use `cost` instead')
    def price(self) -> genai_types.PriceCalculation:  # pragma: no cover
        return self.cost()

    def cost(self) -> genai_types.PriceCalculation:
        """Calculate the cost of the usage.

        Uses [`genai-prices`](https://github.com/pydantic/genai-prices).
        """
        assert self.model_name, 'Model name is required to calculate price'
        return calc_price(
            self.usage,
            self.model_name,
            provider_id=self.provider_name,
            genai_request_timestamp=self.timestamp,
        )

    def otel_events(self, settings: InstrumentationSettings) -> list[Event]:
        """Return OpenTelemetry events for the response."""
        result: list[Event] = []

        def new_event_body():
            new_body: dict[str, Any] = {'role': 'assistant'}
            ev = Event('gen_ai.assistant.message', body=new_body)
            result.append(ev)
            return new_body

        body = new_event_body()
        for part in self.parts:
            if isinstance(part, ToolCallPart):
                body.setdefault('tool_calls', []).append(
                    {
                        'id': part.tool_call_id,
                        'type': 'function',
                        'function': {
                            'name': part.tool_name,
                            **({'arguments': part.args} if settings.include_content else {}),
                        },
                    }
                )
            elif isinstance(part, TextPart | ThinkingPart):
                kind = part.part_kind
                body.setdefault('content', []).append(
                    {'kind': kind, **({'text': part.content} if settings.include_content else {})}
                )
            elif isinstance(part, FilePart):
                body.setdefault('content', []).append(
                    {
                        'kind': 'binary',
                        'media_type': part.content.media_type,
                        **(
                            {'binary_content': base64.b64encode(part.content.data).decode()}
                            if settings.include_content and settings.include_binary_content
                            else {}
                        ),
                    }
                )

        if content := body.get('content'):
            text_content = content[0].get('text')
            if content == [{'kind': 'text', 'text': text_content}]:
                body['content'] = text_content

        return result

    def otel_message_parts(self, settings: InstrumentationSettings) -> list[_otel_messages.MessagePart]:
        parts: list[_otel_messages.MessagePart] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                parts.append(
                    _otel_messages.TextPart(
                        type='text',
                        **({'content': part.content} if settings.include_content else {}),
                    )
                )
            elif isinstance(part, ThinkingPart):
                parts.append(
                    _otel_messages.ThinkingPart(
                        type='thinking',
                        **({'content': part.content} if settings.include_content else {}),
                    )
                )
            elif isinstance(part, FilePart):
                converted_part = _otel_messages.BinaryDataPart(type='binary', media_type=part.content.media_type)
                if settings.include_content and settings.include_binary_content:
                    converted_part['content'] = base64.b64encode(part.content.data).decode()
                parts.append(converted_part)
            elif isinstance(part, BaseToolCallPart):
                call_part = _otel_messages.ToolCallPart(type='tool_call', id=part.tool_call_id, name=part.tool_name)
                if isinstance(part, BuiltinToolCallPart):
                    call_part['builtin'] = True
                if settings.include_content and part.args is not None:
                    from .models.instrumented import InstrumentedModel

                    if isinstance(part.args, str):
                        call_part['arguments'] = part.args
                    else:
                        call_part['arguments'] = {k: InstrumentedModel.serialize_any(v) for k, v in part.args.items()}

                parts.append(call_part)
            elif isinstance(part, BuiltinToolReturnPart):
                return_part = _otel_messages.ToolCallResponsePart(
                    type='tool_call_response',
                    id=part.tool_call_id,
                    name=part.tool_name,
                    builtin=True,
                )
                if settings.include_content and part.content is not None:  # pragma: no branch
                    from .models.instrumented import InstrumentedModel

                    return_part['result'] = InstrumentedModel.serialize_any(part.content)

                parts.append(return_part)
        return parts

    @property
    @deprecated('`vendor_details` is deprecated, use `provider_details` instead')
    def vendor_details(self) -> dict[str, Any] | None:
        return self.provider_details

    @property
    @deprecated('`vendor_id` is deprecated, use `provider_response_id` instead')
    def vendor_id(self) -> str | None:
        return self.provider_response_id

    @property
    @deprecated('`provider_request_id` is deprecated, use `provider_response_id` instead')
    def provider_request_id(self) -> str | None:
        return self.provider_response_id

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelMessage = Annotated[ModelRequest | ModelResponse, pydantic.Discriminator('kind')]
"""Any message sent to or returned by a model."""

ModelMessagesTypeAdapter = pydantic.TypeAdapter(
    list[ModelMessage], config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
)
"""Pydantic [`TypeAdapter`][pydantic.type_adapter.TypeAdapter] for (de)serializing messages."""


@dataclass(repr=False)
class TextPartDelta:
    """A partial update (delta) for a `TextPart` to append new text content."""

    content_delta: str
    """The incremental text content to add to the existing `TextPart` content."""

    _: KW_ONLY

    part_delta_kind: Literal['text'] = 'text'
    """Part delta type identifier, used as a discriminator."""

    def apply(self, part: ModelResponsePart) -> TextPart:
        """Apply this text delta to an existing `TextPart`.

        Args:
            part: The existing model response part, which must be a `TextPart`.

        Returns:
            A new `TextPart` with updated text content.

        Raises:
            ValueError: If `part` is not a `TextPart`.
        """
        if not isinstance(part, TextPart):
            raise ValueError('Cannot apply TextPartDeltas to non-TextParts')  # pragma: no cover
        return replace(part, content=part.content + self.content_delta)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class ThinkingPartDelta:
    """A partial update (delta) for a `ThinkingPart` to append new thinking content."""

    content_delta: str | None = None
    """The incremental thinking content to add to the existing `ThinkingPart` content."""

    signature_delta: str | None = None
    """Optional signature delta.

    Note this is never treated as a delta — it can replace None.
    """

    provider_name: str | None = None
    """Optional provider name for the thinking part.

    Signatures are only sent back to the same provider.
    """

    part_delta_kind: Literal['thinking'] = 'thinking'
    """Part delta type identifier, used as a discriminator."""

    @overload
    def apply(self, part: ModelResponsePart) -> ThinkingPart: ...

    @overload
    def apply(self, part: ModelResponsePart | ThinkingPartDelta) -> ThinkingPart | ThinkingPartDelta: ...

    def apply(self, part: ModelResponsePart | ThinkingPartDelta) -> ThinkingPart | ThinkingPartDelta:
        """Apply this thinking delta to an existing `ThinkingPart`.

        Args:
            part: The existing model response part, which must be a `ThinkingPart`.

        Returns:
            A new `ThinkingPart` with updated thinking content.

        Raises:
            ValueError: If `part` is not a `ThinkingPart`.
        """
        if isinstance(part, ThinkingPart):
            new_content = part.content + self.content_delta if self.content_delta else part.content
            new_signature = self.signature_delta if self.signature_delta is not None else part.signature
            new_provider_name = self.provider_name if self.provider_name is not None else part.provider_name
            return replace(part, content=new_content, signature=new_signature, provider_name=new_provider_name)
        elif isinstance(part, ThinkingPartDelta):
            if self.content_delta is None and self.signature_delta is None:
                raise ValueError('Cannot apply ThinkingPartDelta with no content or signature')
            if self.content_delta is not None:
                part = replace(part, content_delta=(part.content_delta or '') + self.content_delta)
            if self.signature_delta is not None:
                part = replace(part, signature_delta=self.signature_delta)
            if self.provider_name is not None:
                part = replace(part, provider_name=self.provider_name)
            return part
        raise ValueError(  # pragma: no cover
            f'Cannot apply ThinkingPartDeltas to non-ThinkingParts or non-ThinkingPartDeltas ({part=}, {self=})'
        )

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class ToolCallPartDelta:
    """A partial update (delta) for a `ToolCallPart` to modify tool name, arguments, or tool call ID."""

    tool_name_delta: str | None = None
    """Incremental text to add to the existing tool name, if any."""

    args_delta: str | dict[str, Any] | None = None
    """Incremental data to add to the tool arguments.

    If this is a string, it will be appended to existing JSON arguments.
    If this is a dict, it will be merged with existing dict arguments.
    """

    tool_call_id: str | None = None
    """Optional tool call identifier, this is used by some models including OpenAI.

    Note this is never treated as a delta — it can replace None, but otherwise if a
    non-matching value is provided an error will be raised."""

    part_delta_kind: Literal['tool_call'] = 'tool_call'
    """Part delta type identifier, used as a discriminator."""

    def as_part(self) -> ToolCallPart | None:
        """Convert this delta to a fully formed `ToolCallPart` if possible, otherwise return `None`.

        Returns:
            A `ToolCallPart` if `tool_name_delta` is set, otherwise `None`.
        """
        if self.tool_name_delta is None:
            return None

        return ToolCallPart(self.tool_name_delta, self.args_delta, self.tool_call_id or _generate_tool_call_id())

    @overload
    def apply(self, part: ModelResponsePart) -> ToolCallPart | BuiltinToolCallPart: ...

    @overload
    def apply(
        self, part: ModelResponsePart | ToolCallPartDelta
    ) -> ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta: ...

    def apply(
        self, part: ModelResponsePart | ToolCallPartDelta
    ) -> ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta:
        """Apply this delta to a part or delta, returning a new part or delta with the changes applied.

        Args:
            part: The existing model response part or delta to update.

        Returns:
            Either a new `ToolCallPart` or `BuiltinToolCallPart`, or an updated `ToolCallPartDelta`.

        Raises:
            ValueError: If `part` is neither a `ToolCallPart`, `BuiltinToolCallPart`, nor a `ToolCallPartDelta`.
            UnexpectedModelBehavior: If applying JSON deltas to dict arguments or vice versa.
        """
        if isinstance(part, ToolCallPart | BuiltinToolCallPart):
            return self._apply_to_part(part)

        if isinstance(part, ToolCallPartDelta):
            return self._apply_to_delta(part)

        raise ValueError(  # pragma: no cover
            f'Can only apply ToolCallPartDeltas to ToolCallParts, BuiltinToolCallParts, or ToolCallPartDeltas, not {part}'
        )

    def _apply_to_delta(self, delta: ToolCallPartDelta) -> ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta:
        """Internal helper to apply this delta to another delta."""
        if self.tool_name_delta:
            # Append incremental text to the existing tool_name_delta
            updated_tool_name_delta = (delta.tool_name_delta or '') + self.tool_name_delta
            delta = replace(delta, tool_name_delta=updated_tool_name_delta)

        if isinstance(self.args_delta, str):
            if isinstance(delta.args_delta, dict):
                raise UnexpectedModelBehavior(
                    f'Cannot apply JSON deltas to non-JSON tool arguments ({delta=}, {self=})'
                )
            updated_args_delta = (delta.args_delta or '') + self.args_delta
            delta = replace(delta, args_delta=updated_args_delta)
        elif isinstance(self.args_delta, dict):
            if isinstance(delta.args_delta, str):
                raise UnexpectedModelBehavior(
                    f'Cannot apply dict deltas to non-dict tool arguments ({delta=}, {self=})'
                )
            updated_args_delta = {**(delta.args_delta or {}), **self.args_delta}
            delta = replace(delta, args_delta=updated_args_delta)

        if self.tool_call_id:
            delta = replace(delta, tool_call_id=self.tool_call_id)

        # If we now have enough data to create a full ToolCallPart, do so
        if delta.tool_name_delta is not None:
            return ToolCallPart(delta.tool_name_delta, delta.args_delta, delta.tool_call_id or _generate_tool_call_id())

        return delta

    def _apply_to_part(self, part: ToolCallPart | BuiltinToolCallPart) -> ToolCallPart | BuiltinToolCallPart:
        """Internal helper to apply this delta directly to a `ToolCallPart` or `BuiltinToolCallPart`."""
        if self.tool_name_delta:
            # Append incremental text to the existing tool_name
            tool_name = part.tool_name + self.tool_name_delta
            part = replace(part, tool_name=tool_name)

        if isinstance(self.args_delta, str):
            if isinstance(part.args, dict):
                raise UnexpectedModelBehavior(f'Cannot apply JSON deltas to non-JSON tool arguments ({part=}, {self=})')
            updated_json = (part.args or '') + self.args_delta
            part = replace(part, args=updated_json)
        elif isinstance(self.args_delta, dict):
            if isinstance(part.args, str):
                raise UnexpectedModelBehavior(f'Cannot apply dict deltas to non-dict tool arguments ({part=}, {self=})')
            updated_dict = {**(part.args or {}), **self.args_delta}
            part = replace(part, args=updated_dict)

        if self.tool_call_id:
            part = replace(part, tool_call_id=self.tool_call_id)
        return part

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelResponsePartDelta = Annotated[
    TextPartDelta | ThinkingPartDelta | ToolCallPartDelta, pydantic.Discriminator('part_delta_kind')
]
"""A partial update (delta) for any model response part."""


@dataclass(repr=False, kw_only=True)
class PartStartEvent:
    """An event indicating that a new part has started.

    If multiple `PartStartEvent`s are received with the same index,
    the new one should fully replace the old one.
    """

    index: int
    """The index of the part within the overall response parts list."""

    part: ModelResponsePart
    """The newly started `ModelResponsePart`."""

    previous_part_kind: (
        Literal['text', 'thinking', 'tool-call', 'builtin-tool-call', 'builtin-tool-return', 'file'] | None
    ) = None
    """The kind of the previous part, if any.

    This is useful for UI event streams to know whether to group parts of the same kind together when emitting events.
    """

    event_kind: Literal['part_start'] = 'part_start'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class PartDeltaEvent:
    """An event indicating a delta update for an existing part."""

    index: int
    """The index of the part within the overall response parts list."""

    delta: ModelResponsePartDelta
    """The delta to apply to the specified part."""

    event_kind: Literal['part_delta'] = 'part_delta'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class PartEndEvent:
    """An event indicating that a part is complete."""

    index: int
    """The index of the part within the overall response parts list."""

    part: ModelResponsePart
    """The complete `ModelResponsePart`."""

    next_part_kind: (
        Literal['text', 'thinking', 'tool-call', 'builtin-tool-call', 'builtin-tool-return', 'file'] | None
    ) = None
    """The kind of the next part, if any.

    This is useful for UI event streams to know whether to group parts of the same kind together when emitting events.
    """

    event_kind: Literal['part_end'] = 'part_end'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False, kw_only=True)
class FinalResultEvent:
    """An event indicating the response to the current model request matches the output schema and will produce a result."""

    tool_name: str | None
    """The name of the output tool that was called. `None` if the result is from text content and not from a tool."""
    tool_call_id: str | None
    """The tool call ID, if any, that this result is associated with."""
    event_kind: Literal['final_result'] = 'final_result'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelResponseStreamEvent = Annotated[
    PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent, pydantic.Discriminator('event_kind')
]
"""An event in the model response stream, starting a new part, applying a delta to an existing one, indicating a part is complete, or indicating the final result."""


@dataclass(repr=False)
class FunctionToolCallEvent:
    """An event indicating the start to a call to a function tool."""

    part: ToolCallPart
    """The (function) tool call to make."""

    _: KW_ONLY

    event_kind: Literal['function_tool_call'] = 'function_tool_call'
    """Event type identifier, used as a discriminator."""

    @property
    def tool_call_id(self) -> str:
        """An ID used for matching details about the call to its result."""
        return self.part.tool_call_id

    @property
    @deprecated('`call_id` is deprecated, use `tool_call_id` instead.')
    def call_id(self) -> str:
        """An ID used for matching details about the call to its result."""
        return self.part.tool_call_id  # pragma: no cover

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class FunctionToolResultEvent:
    """An event indicating the result of a function tool call."""

    result: ToolReturnPart | RetryPromptPart
    """The result of the call to the function tool."""

    _: KW_ONLY

    content: str | Sequence[UserContent] | None = None
    """The content that will be sent to the model as a UserPromptPart following the result."""

    event_kind: Literal['function_tool_result'] = 'function_tool_result'
    """Event type identifier, used as a discriminator."""

    @property
    def tool_call_id(self) -> str:
        """An ID used to match the result to its original call."""
        return self.result.tool_call_id

    __repr__ = _utils.dataclasses_no_defaults_repr


@deprecated(
    '`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.'
)
@dataclass(repr=False)
class BuiltinToolCallEvent:
    """An event indicating the start to a call to a built-in tool."""

    part: BuiltinToolCallPart
    """The built-in tool call to make."""

    _: KW_ONLY

    event_kind: Literal['builtin_tool_call'] = 'builtin_tool_call'
    """Event type identifier, used as a discriminator."""


@deprecated(
    '`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.'
)
@dataclass(repr=False)
class BuiltinToolResultEvent:
    """An event indicating the result of a built-in tool call."""

    result: BuiltinToolReturnPart
    """The result of the call to the built-in tool."""

    _: KW_ONLY

    event_kind: Literal['builtin_tool_result'] = 'builtin_tool_result'
    """Event type identifier, used as a discriminator."""


HandleResponseEvent = Annotated[
    FunctionToolCallEvent
    | FunctionToolResultEvent
    | BuiltinToolCallEvent  # pyright: ignore[reportDeprecated]
    | BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
    pydantic.Discriminator('event_kind'),
]
"""An event yielded when handling a model response, indicating tool calls and results."""

AgentStreamEvent = Annotated[ModelResponseStreamEvent | HandleResponseEvent, pydantic.Discriminator('event_kind')]
"""An event in the agent stream: model response stream events and response-handling events."""
