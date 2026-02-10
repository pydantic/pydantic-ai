"""Cross-provider matrix tests for multimodal tool return functionality.

This module tests multimodal tool returns across all providers using a cartesian
product of test dimensions:
- provider: anthropic, bedrock, google, openai_chat, openai_responses, groq, mistral, xai
- file_type: image, document, audio, video
- content_source: binary, url, url_force_download
- return_style: direct (return file), tool_return_content (via ToolReturn.content)

The test harness uses a SUPPORT_MATRIX to determine expected behavior (native,
fallback, or error) for each provider/file_type combination. Content source errors
are tracked separately in ERROR_DETAILS since they can vary by source type.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest
from typing_extensions import assert_never

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from tests.cassette_utils import CassetteContext

from pydantic_ai import Agent, BinaryContent, BinaryImage
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
    iter_message_parts,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as xai_available:
    from pydantic_ai.models.xai import XaiModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
]

Expectation = Literal['native', 'fallback', 'error']
"""Expected behavior: 'native' = in tool_result, 'fallback' = separate user msg, 'error' = raises."""

ReturnStyle = Literal['direct', 'tool_return_content']
"""Return style: 'direct' = returns file directly, 'tool_return_content' = via ToolReturn.content."""

SUPPORT_MATRIX: dict[tuple[str, str], Expectation] = {
    ('bedrock', 'image'): 'native',
    ('bedrock', 'document'): 'native',
    ('bedrock', 'audio'): 'error',
    ('bedrock', 'video'): 'native',
    ('anthropic', 'image'): 'native',
    ('anthropic', 'document'): 'native',
    ('anthropic', 'audio'): 'error',
    ('anthropic', 'video'): 'error',
    ('google', 'image'): 'native',
    ('google', 'document'): 'native',
    ('google', 'audio'): 'native',
    ('google', 'video'): 'native',
    ('google_gemini3', 'image'): 'native',
    ('google_gemini3', 'document'): 'native',
    ('google_gemini3', 'audio'): 'native',
    ('google_gemini3', 'video'): 'native',
    ('openai_chat', 'image'): 'fallback',
    ('openai_chat', 'document'): 'fallback',
    ('openai_chat', 'audio'): 'error',
    ('openai_chat', 'video'): 'error',
    ('openai_responses', 'image'): 'native',
    ('openai_responses', 'document'): 'native',
    ('openai_responses', 'audio'): 'error',
    ('openai_responses', 'video'): 'error',
    ('xai', 'image'): 'fallback',
    ('xai', 'document'): 'fallback',
    ('xai', 'audio'): 'error',
    ('xai', 'video'): 'error',
    ('groq', 'image'): 'fallback',
    ('groq', 'document'): 'error',
    ('groq', 'audio'): 'error',
    ('groq', 'video'): 'error',
    ('mistral', 'image'): 'fallback',
    ('mistral', 'document'): 'fallback',
    ('mistral', 'audio'): 'error',
    ('mistral', 'video'): 'error',
}

FileTypeType = Literal['image', 'document', 'audio', 'video']


@dataclass
class ExpectError:
    """Expected error for a provider/file_type/content_source/return_style combination.

    When `content_source` or `return_style` is None, the error applies to all values
    of that dimension. When specified, the error only applies to that specific value.
    """

    provider: str
    """Provider name matching `MODEL_CONFIGS` keys."""
    file_type: FileTypeType
    """File type that triggers this error."""
    content_source: str | None = None
    """Content source filter, or None to match all sources."""
    return_style: str | None = None
    """Return style filter, or None to match all styles."""
    error_type: type[Exception] = RuntimeError
    """Expected exception type."""
    match: str | None = None
    """Regex pattern to match against the error message."""


ERROR_DETAILS: list[ExpectError] = [
    ExpectError(
        'anthropic',
        'audio',
        content_source='binary',
        return_style='direct',
        error_type=NotImplementedError,
        match='Unsupported binary content type for Anthropic tool returns: audio/mpeg',
    ),
    ExpectError(
        'anthropic',
        'audio',
        content_source='binary',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Unsupported binary content type for Anthropic: audio/mpeg',
    ),
    ExpectError(
        'anthropic',
        'audio',
        content_source='url',
        return_style='direct',
        error_type=NotImplementedError,
        match='AudioUrl is not supported for Anthropic tool returns',
    ),
    ExpectError(
        'anthropic',
        'audio',
        content_source='url',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='AudioUrl is not supported by Anthropic',
    ),
    ExpectError(
        'anthropic',
        'audio',
        content_source='url_force_download',
        return_style='direct',
        error_type=NotImplementedError,
        match='AudioUrl is not supported for Anthropic tool returns',
    ),
    ExpectError(
        'anthropic',
        'audio',
        content_source='url_force_download',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='AudioUrl is not supported by Anthropic',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='binary',
        return_style='direct',
        error_type=NotImplementedError,
        match='Unsupported binary content type for Anthropic tool returns: video/mp4',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='binary',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Unsupported binary content type for Anthropic: video/mp4',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='url',
        return_style='direct',
        error_type=NotImplementedError,
        match='VideoUrl is not supported for Anthropic tool returns',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='url',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='VideoUrl is not supported by Anthropic',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='url_force_download',
        return_style='direct',
        error_type=NotImplementedError,
        match='VideoUrl is not supported for Anthropic tool returns',
    ),
    ExpectError(
        'anthropic',
        'video',
        content_source='url_force_download',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='VideoUrl is not supported by Anthropic',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='binary',
        return_style='direct',
        error_type=NotImplementedError,
        match='Unsupported binary content type for Bedrock tool returns: audio/mpeg',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='url',
        return_style='direct',
        error_type=NotImplementedError,
        match='AudioUrl is not supported for Bedrock tool returns',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='url_force_download',
        return_style='direct',
        error_type=NotImplementedError,
        match='AudioUrl is not supported for Bedrock tool returns',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='binary',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Unsupported content type for Bedrock user prompts: BinaryContent',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='url',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Audio is not supported yet',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='url_force_download',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Audio is not supported yet',
    ),
    ExpectError(
        'openai_chat',
        'audio',
        error_type=ModelHTTPError,
        match='Content blocks are expected to be either text or image_url',
    ),
    ExpectError('openai_chat', 'video', error_type=NotImplementedError, match='VideoUrl is not supported'),
    ExpectError('openai_responses', 'audio', error_type=Exception),  # Binary: RuntimeError, URL: ModelHTTPError
    ExpectError('openai_responses', 'video', error_type=NotImplementedError, match='VideoUrl is not supported'),
    ExpectError('xai', 'audio', error_type=NotImplementedError, match='Audio'),
    ExpectError('xai', 'video', error_type=NotImplementedError, match='Video'),
    ExpectError('groq', 'document', content_source='binary', match='Only images are supported for binary content'),
    ExpectError('groq', 'audio', content_source='binary', match='Only images are supported for binary content'),
    ExpectError('groq', 'video', content_source='binary', match='Only images are supported for binary content'),
    ExpectError(
        'groq',
        'document',
        content_source='url',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'audio',
        content_source='url',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'video',
        content_source='url',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'document',
        content_source='url',
        return_style='tool_return_content',
        match='DocumentUrl is not supported in Groq',
    ),
    ExpectError(
        'groq',
        'audio',
        content_source='url',
        return_style='tool_return_content',
        match='AudioUrl is not supported in Groq',
    ),
    ExpectError(
        'groq',
        'video',
        content_source='url',
        return_style='tool_return_content',
        match='VideoUrl is not supported in Groq',
    ),
    ExpectError(
        'bedrock',
        'document',
        return_style='tool_return_content',
        error_type=ModelHTTPError,
        match='text block must be included',
    ),
    ExpectError(
        'groq',
        'document',
        content_source='url_force_download',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'audio',
        content_source='url_force_download',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'video',
        content_source='url_force_download',
        return_style='direct',
        match='Only images are supported for multimodal content',
    ),
    ExpectError(
        'groq',
        'document',
        content_source='url_force_download',
        return_style='tool_return_content',
        match='DocumentUrl is not supported in Groq',
    ),
    ExpectError(
        'groq',
        'audio',
        content_source='url_force_download',
        return_style='tool_return_content',
        match='AudioUrl is not supported in Groq',
    ),
    ExpectError(
        'groq',
        'video',
        content_source='url_force_download',
        return_style='tool_return_content',
        match='VideoUrl is not supported in Groq',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='binary',
        return_style='direct',
        error_type=UserError,
        match='Unsupported binary content type for Mistral tool returns: audio/mpeg',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='binary',
        return_style='tool_return_content',
        match='BinaryContent other than image or PDF is not supported in Mistral',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='url',
        return_style='direct',
        error_type=UserError,
        match='AudioUrl is not supported for Mistral tool returns',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='url',
        return_style='tool_return_content',
        match='Unsupported content type',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='url_force_download',
        return_style='direct',
        error_type=UserError,
        match='AudioUrl is not supported for Mistral tool returns',
    ),
    ExpectError(
        'mistral',
        'audio',
        content_source='url_force_download',
        return_style='tool_return_content',
        match='Unsupported content type',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='binary',
        return_style='direct',
        error_type=UserError,
        match='Unsupported binary content type for Mistral tool returns: video/mp4',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='binary',
        return_style='tool_return_content',
        match='BinaryContent other than image or PDF is not supported in Mistral',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='url',
        return_style='direct',
        error_type=UserError,
        match='VideoUrl is not supported for Mistral tool returns',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='url',
        return_style='tool_return_content',
        match='VideoUrl is not supported in Mistral',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='url_force_download',
        return_style='direct',
        error_type=UserError,
        match='VideoUrl is not supported for Mistral tool returns',
    ),
    ExpectError(
        'mistral',
        'video',
        content_source='url_force_download',
        return_style='tool_return_content',
        match='VideoUrl is not supported in Mistral',
    ),
]


def get_error_details(
    provider: str, file_type: FileTypeType, content_source: str, return_style: str
) -> ExpectError | None:
    for e in ERROR_DETAILS:
        if e.provider != provider or e.file_type != file_type:
            continue
        if e.content_source is not None and e.content_source != content_source:
            continue
        if e.return_style is not None and e.return_style != return_style:
            continue
        return e
    return None


MODEL_CONFIGS: dict[str, tuple[str, Any]] = {
    'anthropic': ('claude-sonnet-4-5', anthropic_available),
    'bedrock': ('us.amazon.nova-pro-v1:0', bedrock_available),
    'google': ('gemini-2.5-flash', google_available),
    'google_gemini3': ('gemini-3-flash-preview', google_available),
    'openai_chat': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_available),
    'groq': ('meta-llama/llama-4-maverick-17b-128e-instruct', groq_available),
    'mistral': ('mistral-medium-latest', mistral_available),
    'xai': ('grok-4-1-fast-non-reasoning', xai_available),
}


def create_model(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any = None,
    xai_provider: Any = None,
) -> Model:
    model_name = MODEL_CONFIGS[provider][0]
    if provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_keys['anthropic']))
    elif provider == 'bedrock':
        assert bedrock_provider is not None
        return BedrockConverseModel(model_name, provider=bedrock_provider)
    elif provider in ('google', 'google_gemini3'):
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'openai_chat':
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'openai_responses':
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'groq':
        return GroqModel(model_name, provider=GroqProvider(api_key=api_keys['groq']))
    elif provider == 'mistral':
        return MistralModel(model_name, provider=MistralProvider(api_key=api_keys['mistral']))
    elif provider == 'xai':
        assert xai_provider is not None
        return XaiModel(model_name, provider=xai_provider)
    else:  # pragma: no cover
        raise ValueError(f'Unknown provider: {provider}')


def is_provider_available(provider: str) -> bool:
    _, available = MODEL_CONFIGS.get(provider, (None, lambda: False))
    return bool(available() if callable(available) else available)


CASSETTE_PATTERNS: dict[tuple[str, str], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): ('gstatic.com/webp/gallery3/1.png', '/9j/', '_9j_', 'UklGR', 'iVBOR'),
    ('image', 'url_force_download'): ('/9j/', '_9j_', 'UklGR'),
    ('document', 'binary'): ('%PDF', 'JVBER'),
    ('document', 'url'): ('pdfobject.com/pdf/sample.pdf', '%PDF', 'JVBER'),
    ('document', 'url_force_download'): ('%PDF', 'JVBER'),
    ('audio', 'binary'): ('//t', '__t'),
    ('audio', 'url'): ('download.samplelib.com/mp3/sample-3s.mp3', '//t', '__t'),
    ('audio', 'url_force_download'): ('//t', '__t'),
    ('video', 'binary'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
    ('video', 'url'): (
        'storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
        'ftyp',
        'ZnR5cA',
        'Z0eXB',
    ),
    ('video', 'url_force_download'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
}

XAI_CASSETTE_PATTERNS: dict[tuple[str, str], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): 'gstatic.com/webp/gallery3/1.png',
    ('image', 'url_force_download'): ('/9j/', '_9j_', 'UklGR'),
    ('document', 'binary'): 'file_id',
    ('document', 'url'): 'file_id',
    ('document', 'url_force_download'): 'file_id',
}


def get_cassette_pattern(provider: str, file_type: FileTypeType, content_source: str) -> str | tuple[str, ...] | None:
    """Get the cassette pattern for a provider/file_type/content_source combination."""
    if provider == 'xai':
        return XAI_CASSETTE_PATTERNS.get((file_type, content_source))
    return CASSETTE_PATTERNS.get((file_type, content_source))


@lru_cache
def make_image_binary(assets_path: Path) -> BinaryImage:
    """Create a binary image from the kiwi.jpg test asset."""
    return BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')


def make_image_url(_: Path) -> ImageUrl:
    """Create an ImageUrl pointing to a public test image."""
    return ImageUrl(url='https://www.gstatic.com/webp/gallery3/1.png')


@lru_cache
def make_document_binary(assets_path: Path) -> BinaryContent:
    """Create a binary PDF from the dummy.pdf test asset."""
    return BinaryContent(data=assets_path.joinpath('dummy.pdf').read_bytes(), media_type='application/pdf')


def make_document_url(_: Path) -> DocumentUrl:
    """Create a DocumentUrl pointing to a public test PDF."""
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')


@lru_cache
def make_audio_binary(assets_path: Path) -> BinaryContent:
    """Create a binary audio from the marcelo.mp3 test asset."""
    return BinaryContent(data=assets_path.joinpath('marcelo.mp3').read_bytes(), media_type='audio/mpeg')


def make_audio_url(_: Path) -> AudioUrl:
    """Create an AudioUrl pointing to a public test MP3."""
    return AudioUrl(url='https://download.samplelib.com/mp3/sample-3s.mp3')


@lru_cache
def make_video_binary(assets_path: Path) -> BinaryContent:
    """Create a binary video from the small_video.mp4 test asset."""
    return BinaryContent(data=assets_path.joinpath('small_video.mp4').read_bytes(), media_type='video/mp4')


def make_video_url(_: Path) -> VideoUrl:
    """Create a VideoUrl pointing to a public test video."""
    return VideoUrl(url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4')


def make_image_url_force_download(_: Path) -> ImageUrl:
    """Create an ImageUrl with force_download=True."""
    return ImageUrl(url='https://www.gstatic.com/webp/gallery3/1.png', force_download=True)


def make_document_url_force_download(_: Path) -> DocumentUrl:
    """Create a DocumentUrl with force_download=True."""
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf', force_download=True)


def make_audio_url_force_download(_: Path) -> AudioUrl:
    """Create an AudioUrl with force_download=True."""
    return AudioUrl(url='https://download.samplelib.com/mp3/sample-3s.mp3', force_download=True)


def make_video_url_force_download(_: Path) -> VideoUrl:
    """Create a VideoUrl with force_download=True."""
    return VideoUrl(
        url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', force_download=True
    )


CONTENT_FACTORIES: dict[tuple[str, str], Any] = {
    ('image', 'binary'): make_image_binary,
    ('image', 'url'): make_image_url,
    ('image', 'url_force_download'): make_image_url_force_download,
    ('document', 'binary'): make_document_binary,
    ('document', 'url'): make_document_url,
    ('document', 'url_force_download'): make_document_url_force_download,
    ('audio', 'binary'): make_audio_binary,
    ('audio', 'url'): make_audio_url,
    ('audio', 'url_force_download'): make_audio_url_force_download,
    ('video', 'binary'): make_video_binary,
    ('video', 'url'): make_video_url,
    ('video', 'url_force_download'): make_video_url_force_download,
}


def get_expectation(provider: str, file_type: FileTypeType) -> Expectation:
    return SUPPORT_MATRIX[(provider, file_type)]


FILE_TYPE_CLASSES: dict[str, tuple[type, ...]] = {
    'image': (BinaryImage, ImageUrl),
    'document': (DocumentUrl, BinaryContent),
    'audio': (AudioUrl, BinaryContent),
    'video': (VideoUrl, BinaryContent),
}


def _is_file_type(item: Any, file_type: FileTypeType) -> bool:
    """Check if item matches the expected file type."""
    expected_classes = FILE_TYPE_CLASSES[file_type]
    if not isinstance(item, expected_classes):
        return False  # pragma: no cover
    if isinstance(item, BinaryImage):
        return file_type == 'image'
    if isinstance(item, ImageUrl):
        return file_type == 'image'
    if isinstance(item, BinaryContent):
        media = item.media_type
        if file_type == 'document':
            return media.startswith('application/')
        elif file_type == 'audio':
            return media.startswith('audio/')
        elif file_type == 'video':
            return media.startswith('video/')
        elif file_type == 'image':  # pragma: no cover
            return media.startswith('image/')
        else:
            assert_never(file_type)
    if isinstance(item, DocumentUrl):
        return file_type == 'document'
    if isinstance(item, AudioUrl):
        return file_type == 'audio'
    if isinstance(item, VideoUrl):
        return file_type == 'video'
    return False  # pragma: no cover


def assert_file_in_tool_return(messages: list[ModelMessage], file_type: FileTypeType) -> None:
    """Assert that file content of the expected type is present in a ToolReturnPart."""
    for trp in iter_message_parts(messages, ModelRequest, ToolReturnPart):
        for f in trp.files:  # pragma: no branch
            if _is_file_type(f, file_type):  # pragma: no branch
                return
    raise AssertionError(f'No {file_type} found in any ToolReturnPart')  # pragma: no cover


def assert_file_in_user_prompt(messages: list[ModelMessage], file_type: FileTypeType) -> None:
    """Assert that file content of the expected type is present in a UserPromptPart.

    For tool_return_content style, files are moved to a separate UserPromptPart
    by design of ToolReturn.content. This verifies files ended up in user messages.
    """
    for upp in iter_message_parts(messages, ModelRequest, UserPromptPart):
        if isinstance(upp.content, list):
            for item in upp.content:  # pragma: no branch
                if _is_file_type(item, file_type):  # pragma: no branch
                    return
    raise AssertionError(f'No {file_type} found in any UserPromptPart')  # pragma: no cover


def assert_multimodal_result(
    messages: list[ModelMessage],
    expectation: Expectation,
    file_type: FileTypeType,
    return_style: ReturnStyle = 'direct',
) -> None:
    """Assert that multimodal content was handled correctly based on expectation.

    For both 'native' and 'fallback' expectations:
    - The `ToolReturn.content` should contain the file content
    - The model implementation handles sending it appropriately (native or separated)
    - We verify files are in the message history, not the API-specific format

    For 'tool_return_content' style:
    - Files go to a separate `UserPromptPart` by design of `ToolReturn.content`
    """
    match expectation:
        case 'error':
            pass
        case 'native' | 'fallback':  # pragma: no branch
            # Both native and fallback: file should be in ToolReturnPart or UserPromptPart
            # The difference is how the model implementation sends it to the API
            if return_style == 'tool_return_content':
                # For tool_return_content, files are in UserPromptPart by design
                assert_file_in_user_prompt(messages, file_type)
            else:
                # For direct return, files should be in ToolReturnPart
                assert_file_in_tool_return(messages, file_type)


@pytest.fixture
def api_keys(
    openai_api_key: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    xai_api_key: str,
) -> dict[str, str]:
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'google': gemini_api_key,
        'groq': groq_api_key,
        'mistral': mistral_api_key,
        'xai': xai_api_key,
    }


PROVIDERS = [pytest.param(name, id=name) for name in MODEL_CONFIGS]

FILE_TYPES = [
    pytest.param('image', id='image'),
    pytest.param('document', id='document'),
    pytest.param('audio', id='audio'),
    pytest.param('video', id='video'),
]

CONTENT_SOURCES = [
    pytest.param('binary', id='binary'),
    pytest.param('url', id='url'),
    pytest.param('url_force_download', id='url_force_download'),
]

RETURN_STYLES = [
    pytest.param('direct', id='direct'),
    pytest.param('tool_return_content', id='tool_return_content'),
]


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('file_type', FILE_TYPES)
@pytest.mark.parametrize('content_source', CONTENT_SOURCES)
@pytest.mark.parametrize('return_style', RETURN_STYLES)
async def test_multimodal_tool_return_matrix(
    provider: str,
    file_type: FileTypeType,
    content_source: str,
    return_style: ReturnStyle,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
    disable_ssrf_protection_for_vcr: None,
):
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    error_info = get_error_details(provider, file_type, content_source, return_style)
    expectation: Expectation = 'error' if error_info else get_expectation(provider, file_type)
    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    content_factory = CONTENT_FACTORIES[(file_type, content_source)]
    content = content_factory(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_file() -> Any:
        if return_style == 'direct':
            return content
        else:
            return ToolReturn(return_value='File attached', content=[content])

    prompt = f'Use the get_file tool now to retrieve a {file_type} file, then describe what you received.'

    if expectation == 'error':
        assert error_info is not None, (
            f'Missing error details for {provider}/{file_type}/{content_source}/{return_style}'
        )
        with pytest.raises(error_info.error_type, match=error_info.match):
            await agent.run(prompt, usage_limits=UsageLimits(output_tokens_limit=100000))
    else:
        result = await agent.run(prompt, usage_limits=UsageLimits(output_tokens_limit=100000))
        assert result.output, 'Expected non-empty response from model'
        assert_multimodal_result(result.all_messages(), expectation, file_type, return_style)
        if cassette_ctx and (pattern := get_cassette_pattern(provider, file_type, content_source)):  # pragma: no branch
            cassette_ctx.verify_contains(pattern)


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_mixed_content_ordering(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
):
    """Test that [text, image, dict] are sent to the API in the correct order.

    Returns mixed content types and verifies the cassette preserves ordering,
    catching bugs where content might be reordered or silently dropped.

    For native providers, strict ordering is verified within the tool_result.
    For fallback providers, the image is separated into a user message, so we
    only verify all content is present (ordering across messages may differ).
    """
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    expectation = get_expectation(provider, 'image')
    if expectation == 'error':  # pragma: no cover
        pytest.skip(f'{provider} does not support images')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    image = make_image_binary(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_mixed_content() -> list[Any]:
        return ['Here is the image:', image, {'metadata': 'test'}]

    result = await agent.run(
        'Call the get_mixed_content tool and describe what you received.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    if cassette_ctx:  # pragma: no branch
        if expectation == 'native':
            if provider == 'google_gemini3':
                # Gemini 3 native: parts field appears before response in JSON
                cassette_ctx.verify_ordering(('/9j/', '_9j_'), 'Here is the image:', 'metadata')
            elif provider == 'google':
                # Gemini 2.5 fallback: image as separate part after function_response
                cassette_ctx.verify_ordering('Here is the image:', 'metadata', ('/9j/', '_9j_'))
            else:
                cassette_ctx.verify_ordering('Here is the image:', ('/9j/', '_9j_'), 'metadata')
        else:
            cassette_ctx.verify_ordering('Here is the image:', 'metadata')
            cassette_ctx.verify_contains(('/9j/', '_9j_'))


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_model_sees_multiple_images(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
    disable_ssrf_protection_for_vcr: None,
):
    """Verify the model processes multiple images by identifying both.

    Returns a kiwi image (binary) and a second image (URL), then verifies:
    1. Both images are sent to the API (cassette verification)
    2. The model identifies the kiwi fruit (semantic verification)
    """
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    expectation = get_expectation(provider, 'image')
    if expectation == 'error':  # pragma: no cover
        pytest.skip(f'{provider} does not support images in tool returns')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    kiwi_image = make_image_binary(assets_path)
    url_image = make_image_url(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_images() -> list[Any]:
        return [kiwi_image, url_image]

    result = await agent.run(
        'Call the get_images tool. One image shows a fruit - what fruit is it? Just name the fruit.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert 'kiwi' in result.output.lower(), f'Model should identify kiwi fruit, got: {result.output}'
    if cassette_ctx:  # pragma: no branch
        cassette_ctx.verify_contains(('/9j/', '_9j_'))
        cassette_ctx.verify_contains(('UklGR', 'iVBOR', 'gstatic.com/webp/gallery3/1.png'))


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_empty_string_in_mixed_content(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
):
    """Test that empty strings in tool returns are skipped correctly."""
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    image = make_image_binary(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_content_with_empty_strings() -> list[Any]:
        return ['', image, '', 'Some text', '']

    result = await agent.run(
        'Call the get_content_with_empty_strings tool and describe what you received.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    if cassette_ctx:  # pragma: no branch
        cassette_ctx.verify_contains(('/9j/', '_9j_'), 'Some text')


OPENAI_PROVIDERS = [
    pytest.param('openai_responses', id='openai_responses'),
]


@pytest.mark.parametrize('provider', OPENAI_PROVIDERS)
async def test_vendor_metadata_detail(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
):
    """Test that vendor_metadata with detail setting is handled correctly."""
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    image_binary = BinaryImage(
        data=assets_path.joinpath('kiwi.jpg').read_bytes(),
        media_type='image/jpeg',
        vendor_metadata={'detail': 'high'},
    )
    image_url = ImageUrl(
        url='https://www.gstatic.com/webp/gallery3/1.png',
        vendor_metadata={'detail': 'low'},
    )

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_images_with_metadata() -> list[Any]:
        return [image_binary, image_url]

    result = await agent.run(
        'Call the get_images_with_metadata tool and describe what you see.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    if cassette_ctx:  # pragma: no branch
        cassette_ctx.verify_contains('"detail": "high"', '"detail": "low"')


async def test_text_plain_document_anthropic(
    anthropic_api_key: str,
    assets_path: Path,
    allow_model_requests: None,
    vcr: Cassette | None,
):
    """Test that text/plain documents are handled correctly by Anthropic."""
    from tests.cassette_utils import CassetteContext

    if not anthropic_available():
        pytest.skip('anthropic dependencies not installed')

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    text_content = assets_path.joinpath('dummy.txt').read_bytes()
    document = BinaryContent(data=text_content, media_type='text/plain')

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_text_document() -> BinaryContent:
        return document

    result = await agent.run(
        'Call the get_text_document tool and describe the document content.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    ctx = CassetteContext('anthropic', vcr, 'test_text_plain_document_anthropic', 'test_multimodal_tool_returns')
    ctx.verify_contains('Dummy TXT file')


@pytest.mark.skipif(not bedrock_available(), reason='bedrock dependencies not installed')
async def test_s3_document_url_bedrock():
    """Test that S3 URLs are correctly parsed for Bedrock documents."""
    document = DocumentUrl(
        url='s3://my-bucket/path/to/document.pdf?bucketOwner=123456789012', media_type='application/pdf'
    )
    result = await BedrockConverseModel._map_file_to_content_block(document, count(1))  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert 'document' in result
    assert result['document']['source']['s3Location']['uri'] == 's3://my-bucket/path/to/document.pdf'  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert result['document']['source']['s3Location']['bucketOwner'] == '123456789012'  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.skipif(not bedrock_available(), reason='bedrock dependencies not installed')
async def test_s3_image_url_bedrock():
    """Test that S3 URLs are correctly parsed for Bedrock images."""
    image = ImageUrl(url='s3://my-bucket/images/photo.png', media_type='image/png')
    result = await BedrockConverseModel._map_file_to_content_block(image, count(1))  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert 'image' in result
    assert result['image']['format'] == 'png'
    assert result['image']['source']['s3Location']['uri'] == 's3://my-bucket/images/photo.png'  # pyright: ignore[reportTypedDictNotRequiredAccess]


@pytest.mark.skipif(not bedrock_available(), reason='bedrock dependencies not installed')
async def test_s3_video_url_bedrock():
    """Test that S3 URLs are correctly parsed for Bedrock videos."""
    video = VideoUrl(url='s3://my-bucket/videos/clip.mp4', media_type='video/mp4')
    result = await BedrockConverseModel._map_file_to_content_block(video, count(1))  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert 'video' in result
    assert result['video']['format'] == 'mp4'
    assert result['video']['source']['s3Location']['uri'] == 's3://my-bucket/videos/clip.mp4'  # pyright: ignore[reportTypedDictNotRequiredAccess]
