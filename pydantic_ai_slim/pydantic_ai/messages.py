from __future__ import annotations as _annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from mimetypes import guess_type
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast, overload

import pydantic
import pydantic_core
from opentelemetry._events import Event  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import TypeAlias, deprecated

from . import _utils
from ._utils import (
    generate_tool_call_id as _generate_tool_call_id,
    now_utc as _now_utc,
)
from .exceptions import UnexpectedModelBehavior
from .usage import Usage

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


@dataclass(repr=False)
class SystemPromptPart:
    """A system prompt, generally written by the application developer.

    This gives the model context and guidance on how to respond.
    """

    content: str
    """The content of the prompt."""

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

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(init=False, repr=False)
class FileUrl(ABC):
    """Abstract base class for any URL-based file."""

    url: str
    """The URL of the file."""

    force_download: bool = False
    """If the model supports it:

    * If True, the file is downloaded and the data is sent to the model as bytes.
    * If False, the URL is sent directly to the model and no download is performed.
    """

    vendor_metadata: dict[str, Any] | None = None
    """Vendor-specific metadata for the file.

    Supported by:
    - `GoogleModel`: `VideoUrl.vendor_metadata` is used as `video_metadata`: https://ai.google.dev/gemini-api/docs/video-understanding#customize-video-processing
    """

    _media_type: str | None = field(init=False, repr=False)

    def __init__(
        self,
        url: str,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
    ) -> None:
        self.url = url
        self.vendor_metadata = vendor_metadata
        self.force_download = force_download
        self._media_type = media_type

    @abstractmethod
    def _infer_media_type(self) -> str:
        """Return the media type of the file, based on the url."""

    @property
    def media_type(self) -> str:
        """Return the media type of the file, based on the url or the provided `_media_type`."""
        return self._media_type or self._infer_media_type()

    @property
    @abstractmethod
    def format(self) -> str:
        """The file format."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(init=False, repr=False)
class VideoUrl(FileUrl):
    """A URL to a video."""

    url: str
    """The URL of the video."""

    kind: Literal['video-url'] = 'video-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
        kind: Literal['video-url'] = 'video-url',
    ) -> None:
        super().__init__(url=url, force_download=force_download, vendor_metadata=vendor_metadata, media_type=media_type)
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
            raise ValueError(f'Unknown video file extension: {self.url}')

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

    kind: Literal['audio-url'] = 'audio-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
        kind: Literal['audio-url'] = 'audio-url',
    ) -> None:
        super().__init__(url=url, force_download=force_download, vendor_metadata=vendor_metadata, media_type=media_type)
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

        raise ValueError(f'Unknown audio file extension: {self.url}')

    @property
    def format(self) -> AudioFormat:
        """The file format of the audio file."""
        return _audio_format_lookup[self.media_type]


@dataclass(init=False, repr=False)
class ImageUrl(FileUrl):
    """A URL to an image."""

    url: str
    """The URL of the image."""

    kind: Literal['image-url'] = 'image-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
        kind: Literal['image-url'] = 'image-url',
    ) -> None:
        super().__init__(url=url, force_download=force_download, vendor_metadata=vendor_metadata, media_type=media_type)
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
            raise ValueError(f'Unknown image file extension: {self.url}')

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

    kind: Literal['document-url'] = 'document-url'
    """Type identifier, this is available on all parts as a discriminator."""

    def __init__(
        self,
        url: str,
        force_download: bool = False,
        vendor_metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
        kind: Literal['document-url'] = 'document-url',
    ) -> None:
        super().__init__(url=url, force_download=force_download, vendor_metadata=vendor_metadata, media_type=media_type)
        self.kind = kind

    def _infer_media_type(self) -> str:
        """Return the media type of the document, based on the url."""
        type_, _ = guess_type(self.url)
        if type_ is None:
            raise ValueError(f'Unknown document file extension: {self.url}')
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


@dataclass(repr=False)
class BinaryContent:
    """Binary content, e.g. an audio or image file."""

    data: bytes
    """The binary data."""

    media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
    """The media type of the binary data."""

    identifier: str | None = None
    """Identifier for the binary content, such as a URL or unique ID.

    This identifier can be provided to the model in a message to allow it to refer to this file in a tool call argument, and the tool can look up the file in question by iterating over the message history and finding the matching `BinaryContent`.

    This identifier is only automatically passed to the model when the `BinaryContent` is returned by a tool. If you're passing the `BinaryContent` as a user message, it's up to you to include a separate text part with the identifier, e.g. "This is file <identifier>:" preceding the `BinaryContent`.
    """

    vendor_metadata: dict[str, Any] | None = None
    """Vendor-specific metadata for the file.

    Supported by:
    - `GoogleModel`: `BinaryContent.vendor_metadata` is used as `video_metadata`: https://ai.google.dev/gemini-api/docs/video-understanding#customize-video-processing
    """

    kind: Literal['binary'] = 'binary'
    """Type identifier, this is available on all parts as a discriminator."""

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


UserContent: TypeAlias = 'str | ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent'


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

    content: Sequence[UserContent] | None = None
    """The content sequence to be sent to the model as a UserPromptPart."""

    metadata: Any = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    __repr__ = _utils.dataclasses_no_defaults_repr


# Ideally this would be a Union of types, but Python 3.9 requires it to be a string, and strings don't work with `isinstance``.
MultiModalContentTypes = (ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent)
_document_format_lookup: dict[str, DocumentFormat] = {
    'application/pdf': 'pdf',
    'text/plain': 'txt',
    'text/csv': 'csv',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'text/html': 'html',
    'text/markdown': 'md',
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

    Content comes from the `user_prompt` parameter of [`Agent.run`][pydantic_ai.Agent.run],
    [`Agent.run_sync`][pydantic_ai.Agent.run_sync], and [`Agent.run_stream`][pydantic_ai.Agent.run_stream].
    """

    content: str | Sequence[UserContent]
    """The content of the prompt."""

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp of the prompt."""

    part_kind: Literal['user-prompt'] = 'user-prompt'
    """Part type identifier, this is available on all parts as a discriminator."""

    def otel_event(self, settings: InstrumentationSettings) -> Event:
        content: str | list[dict[str, Any] | str] | dict[str, Any]
        if isinstance(self.content, str):
            content = self.content if settings.include_content else {'kind': 'text'}
        else:
            content = []
            for part in self.content:
                if isinstance(part, str):
                    content.append(part if settings.include_content else {'kind': 'text'})
                elif isinstance(part, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
                    content.append({'kind': part.kind, **({'url': part.url} if settings.include_content else {})})
                elif isinstance(part, BinaryContent):
                    converted_part = {'kind': part.kind, 'media_type': part.media_type}
                    if settings.include_content and settings.include_binary_content:
                        converted_part['binary_content'] = base64.b64encode(part.data).decode()
                    content.append(converted_part)
                else:
                    content.append({'kind': part.kind})  # pragma: no cover
        return Event('gen_ai.user.message', body={'content': content, 'role': 'user'})

    __repr__ = _utils.dataclasses_no_defaults_repr


tool_return_ta: pydantic.TypeAdapter[Any] = pydantic.TypeAdapter(
    Any, config=pydantic.ConfigDict(defer_build=True, ser_json_bytes='base64', val_json_bytes='base64')
)


@dataclass(repr=False)
class ToolReturnPart:
    """A tool return message, this encodes the result of running a tool."""

    tool_name: str
    """The name of the "tool" was called."""

    content: Any
    """The return value."""

    tool_call_id: str
    """The tool call identifier, this is used by some models including OpenAI."""

    metadata: Any = None
    """Additional data that can be accessed programmatically by the application but is not sent to the LLM."""

    timestamp: datetime = field(default_factory=_now_utc)
    """The timestamp, when the tool returned."""

    part_kind: Literal['tool-return'] = 'tool-return'
    """Part type identifier, this is available on all parts as a discriminator."""

    def model_response_str(self) -> str:
        """Return a string representation of the content for the model."""
        if isinstance(self.content, str):
            return self.content
        else:
            return tool_return_ta.dump_json(self.content).decode()

    def model_response_object(self) -> dict[str, Any]:
        """Return a dictionary representation of the content, wrapping non-dict types appropriately."""
        # gemini supports JSON dict return values, but no other JSON types, hence we wrap anything else in a dict
        if isinstance(self.content, dict):
            return tool_return_ta.dump_python(self.content, mode='json')  # pyright: ignore[reportUnknownMemberType]
        else:
            return {'return_value': tool_return_ta.dump_python(self.content, mode='json')}

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

    __repr__ = _utils.dataclasses_no_defaults_repr


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
            description = f'{len(self.content)} validation errors: {json_errors.decode()}'
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

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelRequestPart = Annotated[
    Union[SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart], pydantic.Discriminator('part_kind')
]
"""A message part sent by Pydantic AI to a model."""


@dataclass(repr=False)
class ModelRequest:
    """A request generated by Pydantic AI and sent to a model, e.g. a message from the Pydantic AI app to the model."""

    parts: list[ModelRequestPart]
    """The parts of the user message."""

    instructions: str | None = None
    """The instructions for the model."""

    kind: Literal['request'] = 'request'
    """Message type identifier, this is available on all parts as a discriminator."""

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

    id: str | None = None
    """The identifier of the thinking part."""

    signature: str | None = None
    """The signature of the thinking.

    The signature is only available on the Anthropic models.
    """

    part_kind: Literal['thinking'] = 'thinking'
    """Part type identifier, this is available on all parts as a discriminator."""

    def has_content(self) -> bool:
        """Return `True` if the thinking content is non-empty."""
        return bool(self.content)

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ToolCallPart:
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

    part_kind: Literal['tool-call'] = 'tool-call'
    """Part type identifier, this is available on all parts as a discriminator."""

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


ModelResponsePart = Annotated[Union[TextPart, ToolCallPart, ThinkingPart], pydantic.Discriminator('part_kind')]
"""A message part returned by a model."""


@dataclass(repr=False)
class ModelResponse:
    """A response from a model, e.g. a message from the model to the Pydantic AI app."""

    parts: list[ModelResponsePart]
    """The parts of the model message."""

    usage: Usage = field(default_factory=Usage)
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

    vendor_details: dict[str, Any] | None = field(default=None)
    """Additional vendor-specific details in a serializable format.

    This allows storing selected vendor-specific data that isn't mapped to standard ModelResponse fields.
    For OpenAI models, this may include 'logprobs', 'finish_reason', etc.
    """

    vendor_id: str | None = None
    """Vendor ID as specified by the model provider. This can be used to track the specific request to the model."""

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
                        'type': 'function',  # TODO https://github.com/pydantic/pydantic-ai/issues/888
                        'function': {
                            'name': part.tool_name,
                            **({'arguments': part.args} if settings.include_content else {}),
                        },
                    }
                )
            elif isinstance(part, TextPart):
                if body.get('content'):
                    body = new_event_body()
                if settings.include_content:
                    body['content'] = part.content

        return result

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelMessage = Annotated[Union[ModelRequest, ModelResponse], pydantic.Discriminator('kind')]
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


@dataclass(repr=False)
class ThinkingPartDelta:
    """A partial update (delta) for a `ThinkingPart` to append new thinking content."""

    content_delta: str | None = None
    """The incremental thinking content to add to the existing `ThinkingPart` content."""

    signature_delta: str | None = None
    """Optional signature delta.

    Note this is never treated as a delta — it can replace None.
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
            return replace(part, content=new_content, signature=new_signature)
        elif isinstance(part, ThinkingPartDelta):
            if self.content_delta is None and self.signature_delta is None:
                raise ValueError('Cannot apply ThinkingPartDelta with no content or signature')
            if self.signature_delta is not None:
                return replace(part, signature_delta=self.signature_delta)
            if self.content_delta is not None:
                return replace(part, content_delta=self.content_delta)
        raise ValueError(  # pragma: no cover
            f'Cannot apply ThinkingPartDeltas to non-ThinkingParts or non-ThinkingPartDeltas ({part=}, {self=})'
        )

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
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
    def apply(self, part: ModelResponsePart) -> ToolCallPart: ...

    @overload
    def apply(self, part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta: ...

    def apply(self, part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta:
        """Apply this delta to a part or delta, returning a new part or delta with the changes applied.

        Args:
            part: The existing model response part or delta to update.

        Returns:
            Either a new `ToolCallPart` or an updated `ToolCallPartDelta`.

        Raises:
            ValueError: If `part` is neither a `ToolCallPart` nor a `ToolCallPartDelta`.
            UnexpectedModelBehavior: If applying JSON deltas to dict arguments or vice versa.
        """
        if isinstance(part, ToolCallPart):
            return self._apply_to_part(part)

        if isinstance(part, ToolCallPartDelta):
            return self._apply_to_delta(part)

        raise ValueError(  # pragma: no cover
            f'Can only apply ToolCallPartDeltas to ToolCallParts or ToolCallPartDeltas, not {part}'
        )

    def _apply_to_delta(self, delta: ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta:
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

    def _apply_to_part(self, part: ToolCallPart) -> ToolCallPart:
        """Internal helper to apply this delta directly to a `ToolCallPart`."""
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
    Union[TextPartDelta, ThinkingPartDelta, ToolCallPartDelta], pydantic.Discriminator('part_delta_kind')
]
"""A partial update (delta) for any model response part."""


@dataclass(repr=False)
class PartStartEvent:
    """An event indicating that a new part has started.

    If multiple `PartStartEvent`s are received with the same index,
    the new one should fully replace the old one.
    """

    index: int
    """The index of the part within the overall response parts list."""

    part: ModelResponsePart
    """The newly started `ModelResponsePart`."""

    event_kind: Literal['part_start'] = 'part_start'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class PartDeltaEvent:
    """An event indicating a delta update for an existing part."""

    index: int
    """The index of the part within the overall response parts list."""

    delta: ModelResponsePartDelta
    """The delta to apply to the specified part."""

    event_kind: Literal['part_delta'] = 'part_delta'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class FinalResultEvent:
    """An event indicating the response to the current model request matches the output schema and will produce a result."""

    tool_name: str | None
    """The name of the output tool that was called. `None` if the result is from text content and not from a tool."""
    tool_call_id: str | None
    """The tool call ID, if any, that this result is associated with."""
    event_kind: Literal['final_result'] = 'final_result'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


ModelResponseStreamEvent = Annotated[Union[PartStartEvent, PartDeltaEvent], pydantic.Discriminator('event_kind')]
"""An event in the model response stream, either starting a new part or applying a delta to an existing one."""

AgentStreamEvent = Annotated[
    Union[PartStartEvent, PartDeltaEvent, FinalResultEvent], pydantic.Discriminator('event_kind')
]
"""An event in the agent stream."""


@dataclass(repr=False)
class FunctionToolCallEvent:
    """An event indicating the start to a call to a function tool."""

    part: ToolCallPart
    """The (function) tool call to make."""
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
    event_kind: Literal['function_tool_result'] = 'function_tool_result'
    """Event type identifier, used as a discriminator."""

    @property
    def tool_call_id(self) -> str:
        """An ID used to match the result to its original call."""
        return self.result.tool_call_id

    __repr__ = _utils.dataclasses_no_defaults_repr


HandleResponseEvent = Annotated[
    Union[FunctionToolCallEvent, FunctionToolResultEvent], pydantic.Discriminator('event_kind')
]
