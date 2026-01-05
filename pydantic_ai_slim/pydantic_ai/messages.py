from __future__ import annotations as _annotations

import base64
import hashlib
import mimetypes
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from datetime import datetime
from mimetypes import MimeTypes
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast, overload
from urllib.parse import urlparse

import pydantic
import pydantic_core
from genai_prices import calc_price, types as genai_types
from opentelemetry._logs import LogRecord  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import TypedDict, deprecated

from . import _otel_messages, _utils
from ._utils import generate_tool_call_id as _generate_tool_call_id, now_utc as _now_utc
from .exceptions import UnexpectedModelBehavior
from .usage import RequestUsage

if TYPE_CHECKING:
    from .models.instrumented import InstrumentationSettings

# =============================================================================
# TypedDict schemas for builtin tool return content
# =============================================================================
# These TypedDicts define the unified content structure for each builtin tool
# return part. Fields are all optional (total=False) because different providers
# populate different subsets of fields.


class CodeExecutionReturnContent(TypedDict, total=False):
    """Unified content schema for code execution results across providers.

    Different providers use different fields:
    - Anthropic: type, stdout, stderr, return_code, content
    - Google: outcome, output
    - OpenAI: status, logs
    """

    # Anthropic fields
    type: str
    """Discriminator field (e.g., 'code_execution_result')."""
    stdout: str
    """Standard output from code execution (Anthropic)."""
    stderr: str
    """Standard error from code execution (Anthropic)."""
    return_code: int
    """Process exit code (Anthropic)."""
    content: list[Any]
    """Additional content objects (Anthropic, typically empty)."""
    # Google fields
    outcome: str
    """Execution outcome status (e.g., 'OUTCOME_OK', 'OUTCOME_TIMEOUT') (Google)."""
    output: str
    """Execution output (Google)."""
    # OpenAI fields
    status: str
    """Execution status (e.g., 'completed') (OpenAI)."""
    logs: list[str]
    """Output lines from execution (OpenAI)."""


class WebSearchSource(TypedDict, total=False):
    """A single web search result source.

    Fields vary by provider - OpenAI uses 'url', Google uses 'uri'.
    """

    title: str
    """Title of the search result."""
    url: str
    """URL of the search result (OpenAI, Anthropic)."""
    uri: str
    """URI of the search result (Google uses 'uri' instead of 'url')."""
    snippet: str
    """Snippet/preview text from the result."""
    relevance_score: float
    """Relevance score of the result."""


class WebSearchReturnContent(TypedDict, total=False):
    """Unified content schema for web search results.

    OpenAI returns dict with 'status' and 'sources' keys.
    Google/Anthropic may return list directly - use WebSearchSource list type.
    """

    status: str
    """Search status (e.g., 'completed') (OpenAI)."""
    sources: list[WebSearchSource]
    """List of search result sources (OpenAI dict format)."""


class WebFetchPage(TypedDict, total=False):
    """A single fetched web page result.

    Fields vary by provider - Anthropic uses 'url', Google uses 'retrieved_url'.
    """

    content: str
    """The fetched page content/text."""
    url: str
    """URL of the fetched page (Anthropic)."""
    retrieved_url: str
    """Retrieved URL (Google uses 'retrieved_url' instead of 'url')."""
    type: str
    """MIME type of the content (e.g., 'text/html')."""
    retrieved_at: str
    """ISO timestamp when the page was fetched."""


class WebFetchReturnContent(TypedDict, total=False):
    """Unified content schema for web fetch results.

    Anthropic returns single dict, Google returns list - use WebFetchPage list type.
    """

    content: str
    """The fetched page content/text (single page format)."""
    url: str
    """URL of the fetched page (Anthropic)."""
    retrieved_url: str
    """Retrieved URL (Google)."""
    type: str
    """MIME type of the content."""
    retrieved_at: str
    """ISO timestamp when the page was fetched."""


class FileSearchResult(TypedDict, total=False):
    """A single file search result.

    OpenAI includes detailed file metadata, Google includes store context.
    """

    id: str
    """Result ID (OpenAI)."""
    file_id: str
    """File ID (OpenAI)."""
    filename: str
    """Filename (OpenAI)."""
    score: float
    """Relevance score."""
    text: str
    """Matched text content."""
    content: str
    """Content of the result."""
    file_search_store: str
    """File search store identifier (Google)."""


class FileSearchReturnContent(TypedDict, total=False):
    """Unified content schema for file search results.

    OpenAI returns dict with 'status' and 'results' keys.
    Google returns list directly - use FileSearchResult list type.
    """

    status: str
    """Search status (e.g., 'completed') (OpenAI)."""
    results: list[FileSearchResult]
    """List of search results (OpenAI dict format)."""


class ImageGenerationReturnContent(TypedDict, total=False):
    """Unified content schema for image generation results (OpenAI only).

    Note: The actual image is returned as a separate FilePart, not in content.
    """

    status: str
    """Generation status (e.g., 'completed')."""
    revised_prompt: str
    """The actual prompt used for generation (may differ from input)."""
    size: str
    """Image dimensions (e.g., '1024x1024')."""
    quality: str
    """Quality setting (e.g., 'high', 'standard')."""
    background: str
    """Background setting (e.g., 'opaque', 'transparent')."""


# =============================================================================
# UNIFIED/NORMALIZED SCHEMAS
# =============================================================================
# These TypedDicts define a provider-agnostic, normalized format for builtin tool
# results. Use the `.normalized` property on return parts to access data in this
# unified format, regardless of which provider generated the response.


CodeExecutionStatus: TypeAlias = Literal['completed', 'failed', 'timeout', 'unknown']
"""Status of a code execution result."""

BuiltinToolStatus: TypeAlias = Literal['completed', 'failed', 'unknown']
"""Generic status for builtin tool results."""


class NormalizedCodeExecutionContent(TypedDict, total=False):
    """Unified/normalized schema for code execution results.

    Access via `part.normalized` for provider-agnostic field names.
    """

    status: CodeExecutionStatus
    """Execution status: 'completed', 'failed', 'timeout', or 'unknown'."""
    output: str
    """Combined output from execution (stdout for Anthropic, output for Google, joined logs for OpenAI)."""
    error: str
    """Error output (stderr for Anthropic, not available for other providers)."""
    exit_code: int
    """Process exit code (return_code for Anthropic, not available for other providers)."""


class NormalizedWebSearchSource(TypedDict, total=False):
    """Unified schema for a single web search result source."""

    title: str
    """Title of the search result."""
    url: str
    """URL of the search result (normalized from 'uri' for Google)."""
    snippet: str
    """Snippet/preview text from the result."""
    relevance_score: float
    """Relevance score of the result."""


class NormalizedWebSearchContent(TypedDict, total=False):
    """Unified/normalized schema for web search results.

    Access via `part.normalized` for provider-agnostic field names.
    """

    status: BuiltinToolStatus
    """Search status: 'completed', 'failed', or 'unknown'."""
    sources: list[NormalizedWebSearchSource]
    """List of search result sources."""


class NormalizedWebFetchPage(TypedDict, total=False):
    """Unified schema for a single fetched web page."""

    url: str
    """URL of the fetched page (normalized from 'retrieved_url' for Google)."""
    content: str
    """The fetched page content/text."""
    content_type: str
    """MIME type of the content."""
    fetched_at: str
    """ISO timestamp when the page was fetched."""


class NormalizedWebFetchContent(TypedDict, total=False):
    """Unified/normalized schema for web fetch results.

    Access via `part.normalized` for provider-agnostic field names.
    """

    status: BuiltinToolStatus
    """Fetch status: 'completed', 'failed', or 'unknown'."""
    pages: list[NormalizedWebFetchPage]
    """List of fetched pages."""


class NormalizedFileSearchResult(TypedDict, total=False):
    """Unified schema for a single file search result."""

    id: str
    """Result ID."""
    filename: str
    """Filename of the matched file."""
    content: str
    """Matched text content."""
    score: float
    """Relevance score."""
    file_store: str
    """File store identifier."""


class NormalizedFileSearchContent(TypedDict, total=False):
    """Unified/normalized schema for file search results.

    Access via `part.normalized` for provider-agnostic field names.
    """

    status: BuiltinToolStatus
    """Search status: 'completed', 'failed', or 'unknown'."""
    results: list[NormalizedFileSearchResult]
    """List of search results."""


class NormalizedImageGenerationContent(TypedDict, total=False):
    """Unified/normalized schema for image generation results (OpenAI only).

    Access via `part.normalized` for provider-agnostic field names.
    Note: The actual image is returned as a separate FilePart, not in content.
    """

    status: Literal['completed', 'generating', 'failed', 'unknown']
    """Generation status."""
    revised_prompt: str
    """The actual prompt used for generation (may differ from input)."""
    size: str
    """Image dimensions (e.g., '1024x1024')."""
    quality: str
    """Quality setting (e.g., 'high', 'standard')."""
    background: str
    """Background setting (e.g., 'opaque', 'transparent')."""


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================


def normalize_code_execution_content(raw: dict[str, Any]) -> NormalizedCodeExecutionContent:
    """Normalize code execution content from any provider to unified format.

    Args:
        raw: Raw provider-specific content.

    Returns:
        Normalized content with unified field names.
    """
    normalized: NormalizedCodeExecutionContent = {}

    # Anthropic format: stdout, stderr, return_code
    if 'stdout' in raw or 'stderr' in raw or 'return_code' in raw:
        return_code = raw.get('return_code')
        if return_code is not None:
            normalized['status'] = 'completed' if return_code == 0 else 'failed'
            normalized['exit_code'] = return_code
        else:
            normalized['status'] = 'unknown'
        # Always include output/error (even if empty string) for consistency
        normalized['output'] = raw.get('stdout', '')
        if 'stderr' in raw:
            normalized['error'] = raw.get('stderr', '')
    # Google format: outcome, output
    elif 'outcome' in raw:
        outcome = raw.get('outcome', '')
        outcome_map: dict[str, CodeExecutionStatus] = {
            'OUTCOME_OK': 'completed',
            'OUTCOME_TIMEOUT': 'timeout',
            'OUTCOME_FAILED': 'failed',
        }
        normalized['status'] = outcome_map.get(outcome, 'unknown')
        normalized['output'] = raw.get('output', '')
    # OpenAI format: status, logs
    elif 'logs' in raw:
        status = raw.get('status', 'unknown')
        normalized['status'] = 'completed' if status == 'completed' else 'failed' if status == 'failed' else 'unknown'
        logs = raw.get('logs', [])
        normalized['output'] = '\n'.join(logs) if logs else ''
    # Unknown or empty format - provide defaults
    else:
        normalized['status'] = 'unknown'
        normalized['output'] = ''

    return normalized


def normalize_web_search_content(
    raw: dict[str, Any] | list[dict[str, Any]],
) -> NormalizedWebSearchContent:
    """Normalize web search content from any provider to unified format.

    Args:
        raw: Raw provider-specific content (dict or list).

    Returns:
        Normalized content with unified field names.
    """
    normalized: NormalizedWebSearchContent = {'status': 'completed', 'sources': []}

    def _extract_source(item: dict[str, Any]) -> NormalizedWebSearchSource:
        source: NormalizedWebSearchSource = {}
        if title := item.get('title'):
            source['title'] = title
        # Normalize uri -> url
        if url := item.get('url') or item.get('uri'):
            source['url'] = url
        if snippet := item.get('snippet'):
            source['snippet'] = snippet
        if (score := item.get('relevance_score')) is not None:
            source['relevance_score'] = score
        return source

    if isinstance(raw, list):
        # Google/Anthropic: list of sources directly
        for item in raw:
            if source := _extract_source(item):
                normalized['sources'].append(source)
    else:
        # Dict format
        if status := raw.get('status'):
            normalized['status'] = 'completed' if status == 'completed' else 'failed' if status == 'failed' else 'unknown'
        # OpenAI: sources array
        if sources := raw.get('sources'):
            for item in sources:
                if source := _extract_source(item):
                    normalized['sources'].append(source)
        # Groq: results array
        elif results := raw.get('results'):
            for item in results:
                if source := _extract_source(item):
                    normalized['sources'].append(source)
        # Groq fallback: output string
        elif output := raw.get('output'):
            normalized['sources'].append({'snippet': output})

    return normalized


def normalize_web_fetch_content(
    raw: dict[str, Any] | list[dict[str, Any]],
) -> NormalizedWebFetchContent:
    """Normalize web fetch content from any provider to unified format.

    Args:
        raw: Raw provider-specific content (dict or list).

    Returns:
        Normalized content with unified field names.
    """
    normalized: NormalizedWebFetchContent = {'status': 'completed', 'pages': []}

    def _extract_page(item: dict[str, Any]) -> NormalizedWebFetchPage:
        page: NormalizedWebFetchPage = {}
        # Normalize retrieved_url -> url
        if url := item.get('url') or item.get('retrieved_url'):
            page['url'] = url
        if content := item.get('content'):
            page['content'] = content
        if content_type := item.get('type') or item.get('content_type'):
            page['content_type'] = content_type
        if fetched_at := item.get('retrieved_at') or item.get('fetched_at'):
            page['fetched_at'] = fetched_at
        return page

    if isinstance(raw, list):
        # Google: list of pages
        for item in raw:
            if page := _extract_page(item):
                normalized['pages'].append(page)
    else:
        # Anthropic: single page dict
        if page := _extract_page(raw):
            normalized['pages'].append(page)

    return normalized


def normalize_file_search_content(
    raw: dict[str, Any] | list[dict[str, Any]],
) -> NormalizedFileSearchContent:
    """Normalize file search content from any provider to unified format.

    Args:
        raw: Raw provider-specific content (dict or list).

    Returns:
        Normalized content with unified field names.
    """
    normalized: NormalizedFileSearchContent = {'status': 'completed', 'results': []}

    def _extract_result(item: dict[str, Any]) -> NormalizedFileSearchResult:
        result: NormalizedFileSearchResult = {}
        if id_ := item.get('id') or item.get('file_id'):
            result['id'] = id_
        if filename := item.get('filename'):
            result['filename'] = filename
        # Normalize text -> content
        if content := item.get('content') or item.get('text'):
            result['content'] = content
        if (score := item.get('score')) is not None:
            result['score'] = score
        if file_store := item.get('file_search_store') or item.get('file_store'):
            result['file_store'] = file_store
        return result

    if isinstance(raw, list):
        # Google: list of results directly
        for item in raw:
            if result := _extract_result(item):
                normalized['results'].append(result)
    else:
        # OpenAI: dict with status and results
        if status := raw.get('status'):
            normalized['status'] = 'completed' if status == 'completed' else 'failed' if status == 'failed' else 'unknown'
        if results := raw.get('results'):
            for item in results:
                if result := _extract_result(item):
                    normalized['results'].append(result)

    return normalized


def normalize_image_generation_content(raw: dict[str, Any]) -> NormalizedImageGenerationContent:
    """Normalize image generation content to unified format.

    Args:
        raw: Raw provider-specific content.

    Returns:
        Normalized content with unified field names.
    """
    normalized: NormalizedImageGenerationContent = {}

    if status := raw.get('status'):
        normalized['status'] = status
    if revised_prompt := raw.get('revised_prompt'):
        normalized['revised_prompt'] = revised_prompt
    if size := raw.get('size'):
        normalized['size'] = size
    if quality := raw.get('quality'):
        normalized['quality'] = quality
    if background := raw.get('background'):
        normalized['background'] = background

    return normalized


_mime_types = MimeTypes()
# Replicate what is being done in `mimetypes.init()`
_mime_types.read_windows_registry()
for file in mimetypes.knownfiles:
    if os.path.isfile(file):
        _mime_types.read(file)
# TODO check for added mimetypes in Python 3.11 when dropping support for Python 3.10:
# Document types
_mime_types.add_type('application/rtf', '.rtf')
_mime_types.add_type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx')
_mime_types.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
_mime_types.add_type('text/markdown', '.mdx')
_mime_types.add_type('text/markdown', '.md')
_mime_types.add_type('text/x-asciidoc', '.asciidoc')

# Image types
_mime_types.add_type('image/webp', '.webp')

# Video types
_mime_types.add_type('video/3gpp', '.three_gp')
_mime_types.add_type('video/x-matroska', '.mkv')
_mime_types.add_type('video/x-ms-wmv', '.wmv')

# Audio types
# NOTE: aac is platform specific (linux: audio/x-aac, macos: audio/aac) but x-aac is deprecated https://mimetype.io/audio/aac
_mime_types.add_type('audio/aac', '.aac')
_mime_types.add_type('audio/aiff', '.aiff')
_mime_types.add_type('audio/flac', '.flac')
_mime_types.add_type('audio/ogg', '.oga')
_mime_types.add_type('audio/wav', '.wav')


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

ProviderDetailsDelta: TypeAlias = dict[str, Any] | Callable[[dict[str, Any] | None], dict[str, Any]] | None
"""Type for provider_details input: can be a static dict, a callback to update existing details, or None."""


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

    Only set if system prompt is dynamic, see [`system_prompt`][pydantic_ai.agent.Agent.system_prompt] for more information.
    """

    part_kind: Literal['system-prompt'] = 'system-prompt'
    """Part type identifier, this is available on all parts as a discriminator."""

    def otel_event(self, settings: InstrumentationSettings) -> LogRecord:
        return LogRecord(
            attributes={'event.name': 'gen_ai.system.message'},
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

    def _infer_media_type(self) -> str:
        """Return the media type of the video, based on the url."""
        # Assume that YouTube videos are mp4 because there would be no extension
        # to infer from. This should not be a problem, as Gemini disregards media
        # type for YouTube URLs.
        if self.is_youtube:
            return 'video/mp4'

        mime_type, _ = _mime_types.guess_type(self.url)
        if mime_type is None:
            raise ValueError(
                f'Could not infer media type from video URL: {self.url}. Explicitly provide a `media_type` instead.'
            )
        return mime_type

    @property
    def is_youtube(self) -> bool:
        """True if the URL has a YouTube domain."""
        parsed = urlparse(self.url)
        hostname = parsed.hostname
        return hostname in ('youtu.be', 'youtube.com', 'www.youtube.com')

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

    def _infer_media_type(self) -> str:
        """Return the media type of the audio file, based on the url.

        References:
        - Gemini: https://ai.google.dev/gemini-api/docs/audio#supported-formats
        """
        mime_type, _ = _mime_types.guess_type(self.url)
        if mime_type is None:
            raise ValueError(
                f'Could not infer media type from audio URL: {self.url}. Explicitly provide a `media_type` instead.'
            )
        return mime_type

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

    def _infer_media_type(self) -> str:
        """Return the media type of the image, based on the url."""
        mime_type, _ = _mime_types.guess_type(self.url)
        if mime_type is None:
            raise ValueError(
                f'Could not infer media type from image URL: {self.url}. Explicitly provide a `media_type` instead.'
            )
        return mime_type

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
        mime_type, _ = _mime_types.guess_type(self.url)
        if mime_type is None:
            raise ValueError(
                f'Could not infer media type from document URL: {self.url}. Explicitly provide a `media_type` instead.'
            )
        return mime_type

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
    """The binary file data.

    Use `.base64` to get the base64-encoded string.
    """

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

    @classmethod
    def from_path(cls, path: PathLike[str]) -> BinaryContent:
        """Create a `BinaryContent` from a path.

        Defaults to 'application/octet-stream' if the media type cannot be inferred.

        Raises:
            FileNotFoundError: if the file does not exist.
            PermissionError: if the file cannot be read.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'File not found: {path}')
        media_type, _ = _mime_types.guess_type(path)
        if media_type is None:
            media_type = 'application/octet-stream'

        return cls.narrow_type(cls(data=path.read_bytes(), media_type=media_type))

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
        return f'data:{self.media_type};base64,{self.base64}'

    @property
    def base64(self) -> str:
        """Return the binary data as a base64-encoded string. Default encoding is UTF-8."""
        return base64.b64encode(self.data).decode()

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
    - Amazon Bedrock (Converse API)
    """

    kind: Literal['cache-point'] = 'cache-point'
    """Type identifier, this is available on all parts as a discriminator."""

    ttl: Literal['5m', '1h'] = '5m'
    """The cache time-to-live, either "5m" (5 minutes) or "1h" (1 hour).

    Supported by:

    * Anthropic (automatically omitted for Bedrock, as it does not support explicit TTL). See https://docs.claude.com/en/docs/build-with-claude/prompt-caching#1-hour-cache-duration for more information."""


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

    def otel_event(self, settings: InstrumentationSettings) -> LogRecord:
        content: Any = [{'kind': part.pop('type'), **part} for part in self.otel_message_parts(settings)]
        for part in content:
            if part['kind'] == 'binary' and 'content' in part:
                part['binary_content'] = part.pop('content')
        content = [
            part['content'] if part == {'kind': 'text', 'content': part.get('content')} else part for part in content
        ]
        if content in ([{'kind': 'text'}], [self.content]):
            content = content[0]
        return LogRecord(attributes={'event.name': 'gen_ai.user.message'}, body={'content': content, 'role': 'user'})

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
                    converted_part['content'] = part.base64
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

    def otel_event(self, settings: InstrumentationSettings) -> LogRecord:
        return LogRecord(
            attributes={'event.name': 'gen_ai.tool.message'},
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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

    part_kind: Literal['builtin-tool-return'] = 'builtin-tool-return'
    """Part type identifier, this is available on all parts as a discriminator."""


@dataclass(repr=False)
class CodeExecutionReturnPart(BuiltinToolReturnPart):
    """A return part for code execution tool results.

    The content field contains raw provider-specific data. Use the `normalized` property
    for provider-agnostic access with unified field names.
    """

    content: CodeExecutionReturnContent | dict[str, Any]
    """The raw code execution result content from the provider."""

    part_kind: Literal['code-execution-return'] = 'code-execution-return'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def normalized(self) -> NormalizedCodeExecutionContent:
        """Get normalized content with unified field names.

        Returns provider-agnostic data with fields: status, output, error, exit_code.
        """
        return normalize_code_execution_content(cast(dict[str, Any], self.content))


@dataclass(repr=False)
class WebSearchReturnPart(BuiltinToolReturnPart):
    """A return part for web search tool results.

    The content field contains raw provider-specific data. Use the `normalized` property
    for provider-agnostic access with unified field names.
    """

    content: WebSearchReturnContent | list[WebSearchSource] | dict[str, Any] | list[dict[str, Any]]
    """The raw web search result content from the provider."""

    part_kind: Literal['web-search-return'] = 'web-search-return'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def normalized(self) -> NormalizedWebSearchContent:
        """Get normalized content with unified field names.

        Returns provider-agnostic data with fields: status, sources.
        Each source has: title, url, snippet, relevance_score.
        """
        return normalize_web_search_content(cast(dict[str, Any] | list[dict[str, Any]], self.content))


@dataclass(repr=False)
class WebFetchReturnPart(BuiltinToolReturnPart):
    """A return part for web fetch tool results.

    The content field contains raw provider-specific data. Use the `normalized` property
    for provider-agnostic access with unified field names.
    """

    content: WebFetchReturnContent | list[WebFetchPage] | dict[str, Any] | list[dict[str, Any]]
    """The raw web fetch result content from the provider."""

    part_kind: Literal['web-fetch-return'] = 'web-fetch-return'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def normalized(self) -> NormalizedWebFetchContent:
        """Get normalized content with unified field names.

        Returns provider-agnostic data with fields: status, pages.
        Each page has: url, content, content_type, fetched_at.
        """
        return normalize_web_fetch_content(cast(dict[str, Any] | list[dict[str, Any]], self.content))


@dataclass(repr=False)
class FileSearchReturnPart(BuiltinToolReturnPart):
    """A return part for file search tool results.

    The content field contains raw provider-specific data. Use the `normalized` property
    for provider-agnostic access with unified field names.
    """

    content: FileSearchReturnContent | list[FileSearchResult] | dict[str, Any] | list[dict[str, Any]]
    """The raw file search result content from the provider."""

    part_kind: Literal['file-search-return'] = 'file-search-return'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def normalized(self) -> NormalizedFileSearchContent:
        """Get normalized content with unified field names.

        Returns provider-agnostic data with fields: status, results.
        Each result has: id, filename, content, score, file_store.
        """
        return normalize_file_search_content(cast(dict[str, Any] | list[dict[str, Any]], self.content))


@dataclass(repr=False)
class ImageGenerationReturnPart(BuiltinToolReturnPart):
    """A return part for image generation tool results (metadata only, image is in FilePart).

    The content field contains raw provider-specific data. Use the `normalized` property
    for provider-agnostic access with unified field names.
    """

    content: ImageGenerationReturnContent | dict[str, Any]
    """The raw image generation result content from the provider (metadata only)."""

    part_kind: Literal['image-generation-return'] = 'image-generation-return'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def normalized(self) -> NormalizedImageGenerationContent:
        """Get normalized content with unified field names.

        Returns provider-agnostic data with fields: status, revised_prompt, size, quality, background.
        """
        return normalize_image_generation_content(cast(dict[str, Any], self.content))


# Mapping from tool_name to specialized return part class for migration
_TOOL_NAME_TO_RETURN_PART_CLASS: dict[str, type[BuiltinToolReturnPart]] = {
    'code_execution': CodeExecutionReturnPart,
    'web_search': WebSearchReturnPart,
    'web_fetch': WebFetchReturnPart,
    'url_context': WebFetchReturnPart,  # Deprecated alias
    'file_search': FileSearchReturnPart,
    'image_generation': ImageGenerationReturnPart,
}


def _migrate_builtin_tool_return_part(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate old BuiltinToolReturnPart data to specific subclass based on tool_name.

    This BeforeValidator upgrades old serialized data with part_kind='builtin-tool-return'
    to the appropriate specialized subclass (e.g., 'code-execution-return').
    """
    if not isinstance(data, dict):
        return data  # pragma: lax no cover

    part_kind = data.get('part_kind')
    tool_name = data.get('tool_name', '')

    if part_kind == 'builtin-tool-return':
        if tool_name in _TOOL_NAME_TO_RETURN_PART_CLASS:
            new_class = _TOOL_NAME_TO_RETURN_PART_CLASS[tool_name]
            data = dict(data)
            data['part_kind'] = new_class.part_kind
        # MCP and memory tools stay as base class (schemas not stable)
        elif tool_name.startswith('mcp_server:') or tool_name in ('mcp_server', 'memory'):
            pass

    return data


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

    def otel_event(self, settings: InstrumentationSettings) -> LogRecord:
        if self.tool_name is None:
            return LogRecord(
                attributes={'event.name': 'gen_ai.user.message'},
                body={'content': self.model_response(), 'role': 'user'},
            )
        else:
            return LogRecord(
                attributes={'event.name': 'gen_ai.tool.message'},
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

    # Default is None for backwards compatibility with old serialized messages that don't have this field.
    # Using a default_factory would incorrectly fill in the current time for deserialized historical messages.
    timestamp: datetime | None = None
    """The timestamp when the request was sent to the model."""

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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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


@dataclass(repr=False)
class CodeExecutionCallPart(BuiltinToolCallPart):
    """A call part for code execution tool with normalized accessor properties."""

    args: dict[str, Any] | str | None = None
    """The code execution call arguments."""

    part_kind: Literal['code-execution-call'] = 'code-execution-call'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def code(self) -> str | None:
        """Get the code to be executed."""
        args = self.args_as_dict()
        return args.get('code')


@dataclass(repr=False)
class WebSearchCallPart(BuiltinToolCallPart):
    """A call part for web search tool with normalized accessor properties."""

    args: dict[str, Any] | str | None = None
    """The web search call arguments."""

    part_kind: Literal['web-search-call'] = 'web-search-call'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def query(self) -> str | None:
        """Get the search query."""
        args = self.args_as_dict()
        return args.get('query')


@dataclass(repr=False)
class WebFetchCallPart(BuiltinToolCallPart):
    """A call part for web fetch tool with normalized accessor properties."""

    args: dict[str, Any] | str | None = None
    """The web fetch call arguments."""

    part_kind: Literal['web-fetch-call'] = 'web-fetch-call'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def urls(self) -> list[str]:
        """Get the URLs to fetch."""
        args = self.args_as_dict()
        if 'urls' in args:
            return args['urls']
        if 'url' in args:
            return [args['url']]
        return []  # pragma: lax no cover


@dataclass(repr=False)
class FileSearchCallPart(BuiltinToolCallPart):
    """A call part for file search tool with normalized accessor properties."""

    args: dict[str, Any] | str | None = None
    """The file search call arguments."""

    part_kind: Literal['file-search-call'] = 'file-search-call'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""

    @property
    def queries(self) -> list[str]:
        """Get the search queries."""
        args = self.args_as_dict()
        if 'queries' in args:
            return args['queries']
        if 'query' in args:
            return [args['query']]
        return []  # pragma: lax no cover


@dataclass(repr=False)
class ImageGenerationCallPart(BuiltinToolCallPart):
    """A call part for image generation tool (prompt comes from conversation context)."""

    part_kind: Literal['image-generation-call'] = 'image-generation-call'  # pyright: ignore[reportIncompatibleVariableOverride]
    """Part type identifier, this is available on all parts as a discriminator."""


# Mapping from tool_name to specialized call part class for migration
_TOOL_NAME_TO_CALL_PART_CLASS: dict[str, type[BuiltinToolCallPart]] = {
    'code_execution': CodeExecutionCallPart,
    'web_search': WebSearchCallPart,
    'web_fetch': WebFetchCallPart,
    'url_context': WebFetchCallPart,  # Deprecated alias
    'file_search': FileSearchCallPart,
    'image_generation': ImageGenerationCallPart,
}


def _migrate_builtin_tool_call_part(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate old BuiltinToolCallPart data to specific subclass based on tool_name.

    This BeforeValidator upgrades old serialized data with part_kind='builtin-tool-call'
    to the appropriate specialized subclass (e.g., 'code-execution-call').
    """
    if not isinstance(data, dict):
        return data  # pragma: lax no cover

    part_kind = data.get('part_kind')
    tool_name = data.get('tool_name', '')

    if part_kind == 'builtin-tool-call':
        if tool_name in _TOOL_NAME_TO_CALL_PART_CLASS:
            new_class = _TOOL_NAME_TO_CALL_PART_CLASS[tool_name]
            data = dict(data)
            data['part_kind'] = new_class.part_kind
        # MCP and memory tools stay as base class (schemas not stable)
        elif tool_name.startswith('mcp_server:') or tool_name in ('mcp_server', 'memory'):
            pass

    return data


# Union of all builtin tool call part types for the discriminator
_BuiltinToolCallPartUnion = (
    BuiltinToolCallPart
    | CodeExecutionCallPart
    | WebSearchCallPart
    | WebFetchCallPart
    | FileSearchCallPart
    | ImageGenerationCallPart
)

# Union of all builtin tool return part types for the discriminator
_BuiltinToolReturnPartUnion = (
    BuiltinToolReturnPart
    | CodeExecutionReturnPart
    | WebSearchReturnPart
    | WebFetchReturnPart
    | FileSearchReturnPart
    | ImageGenerationReturnPart
)


def _migrate_builtin_tool_parts(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate old BuiltinToolCallPart and BuiltinToolReturnPart data to specific subclasses.

    This BeforeValidator upgrades old serialized data to the appropriate specialized subclass.
    """
    data = _migrate_builtin_tool_call_part(data)
    data = _migrate_builtin_tool_return_part(data)
    return data


ModelResponsePart = Annotated[
    TextPart | ToolCallPart | _BuiltinToolCallPartUnion | _BuiltinToolReturnPartUnion | ThinkingPart | FilePart,
    pydantic.BeforeValidator(_migrate_builtin_tool_parts),
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
    """The timestamp when the response was received locally.

    This is always a high-precision local datetime. Provider-specific timestamps
    (if available) are stored in `provider_details['timestamp']`.
    """

    kind: Literal['response'] = 'response'
    """Message type identifier, this is available on all parts as a discriminator."""

    provider_name: str | None = None
    """The name of the LLM provider that generated the response."""

    provider_url: str | None = None
    """The base URL of the LLM provider that generated the response."""

    provider_details: Annotated[
        dict[str, Any] | None,
        # `vendor_details` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        pydantic.Field(validation_alias=pydantic.AliasChoices('provider_details', 'vendor_details')),
    ] = None
    """Additional data returned by the provider that can't be mapped to standard fields."""

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
        # Try matching on provider_api_url first as this is more specific, then fall back to provider_id.
        if self.provider_url:
            try:
                return calc_price(
                    self.usage,
                    self.model_name,
                    provider_api_url=self.provider_url,
                    genai_request_timestamp=self.timestamp,
                )
            except LookupError:
                pass
        return calc_price(
            self.usage,
            self.model_name,
            provider_id=self.provider_name,
            genai_request_timestamp=self.timestamp,
        )

    def otel_events(self, settings: InstrumentationSettings) -> list[LogRecord]:
        """Return OpenTelemetry events for the response."""
        result: list[LogRecord] = []

        def new_event_body():
            new_body: dict[str, Any] = {'role': 'assistant'}
            ev = LogRecord(attributes={'event.name': 'gen_ai.assistant.message'}, body=new_body)
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
                            {'binary_content': part.content.base64}
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
                    converted_part['content'] = part.content.base64
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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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
        return replace(
            part,
            content=part.content + self.content_delta,
            provider_details={**(part.provider_details or {}), **(self.provider_details or {})} or None,
        )

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

    provider_details: ProviderDetailsDelta = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    Can be a dict to merge with existing details, or a callable that takes
    the existing details and returns updated details.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

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
            # Resolve callable provider_details if needed
            resolved_details = (
                self.provider_details(part.provider_details)
                if callable(self.provider_details)
                else self.provider_details
            )
            new_provider_details = {**(part.provider_details or {}), **(resolved_details or {})} or None
            return replace(
                part,
                content=new_content,
                signature=new_signature,
                provider_name=new_provider_name,
                provider_details=new_provider_details,
            )
        elif isinstance(part, ThinkingPartDelta):
            if self.content_delta is None and self.signature_delta is None:
                raise ValueError('Cannot apply ThinkingPartDelta with no content or signature')
            if self.content_delta is not None:
                part = replace(part, content_delta=(part.content_delta or '') + self.content_delta)
            if self.signature_delta is not None:
                part = replace(part, signature_delta=self.signature_delta)
            if self.provider_name is not None:
                part = replace(part, provider_name=self.provider_name)
            if self.provider_details is not None:
                if callable(self.provider_details):
                    if callable(part.provider_details):
                        existing_fn = part.provider_details
                        new_fn = self.provider_details

                        def chained_both(d: dict[str, Any] | None) -> dict[str, Any]:
                            return new_fn(existing_fn(d))

                        part = replace(part, provider_details=chained_both)
                    else:
                        part = replace(part, provider_details=self.provider_details)  # pragma: no cover
                elif callable(part.provider_details):
                    existing_fn = part.provider_details
                    new_dict = self.provider_details

                    def chained_dict(d: dict[str, Any] | None) -> dict[str, Any]:
                        return {**existing_fn(d), **new_dict}

                    part = replace(part, provider_details=chained_dict)
                else:
                    existing = part.provider_details if isinstance(part.provider_details, dict) else {}
                    part = replace(part, provider_details={**existing, **self.provider_details})
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

    provider_details: dict[str, Any] | None = None
    """Additional data returned by the provider that can't be mapped to standard fields.

    This is used for data that is required to be sent back to APIs, as well as data users may want to access programmatically."""

    part_delta_kind: Literal['tool_call'] = 'tool_call'
    """Part delta type identifier, used as a discriminator."""

    def as_part(self) -> ToolCallPart | None:
        """Convert this delta to a fully formed `ToolCallPart` if possible, otherwise return `None`.

        Returns:
            A `ToolCallPart` if `tool_name_delta` is set, otherwise `None`.
        """
        if self.tool_name_delta is None:
            return None

        return ToolCallPart(
            self.tool_name_delta,
            self.args_delta,
            self.tool_call_id or _generate_tool_call_id(),
            provider_details=self.provider_details,
        )

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

        if self.provider_details:
            merged_provider_details = {**(delta.provider_details or {}), **self.provider_details}
            delta = replace(delta, provider_details=merged_provider_details)

        # If we now have enough data to create a full ToolCallPart, do so
        if delta.tool_name_delta is not None:
            return ToolCallPart(
                delta.tool_name_delta,
                delta.args_delta,
                delta.tool_call_id or _generate_tool_call_id(),
                provider_details=delta.provider_details,
            )

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

        if self.provider_details:
            merged_provider_details = {**(part.provider_details or {}), **self.provider_details}
            part = replace(part, provider_details=merged_provider_details)

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
        Literal[
            'text',
            'thinking',
            'tool-call',
            'builtin-tool-call',
            'builtin-tool-return',
            'file',
            'code-execution-call',
            'web-search-call',
            'web-fetch-call',
            'file-search-call',
            'image-generation-call',
            'code-execution-return',
            'web-search-return',
            'web-fetch-return',
            'file-search-return',
            'image-generation-return',
        ]
        | None
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
        Literal[
            'text',
            'thinking',
            'tool-call',
            'builtin-tool-call',
            'builtin-tool-return',
            'file',
            'code-execution-call',
            'web-search-call',
            'web-fetch-call',
            'file-search-call',
            'image-generation-call',
            'code-execution-return',
            'web-search-return',
            'web-fetch-return',
            'file-search-return',
            'image-generation-return',
        ]
        | None
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
