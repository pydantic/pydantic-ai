"""Cross-provider matrix tests for multimodal tool return functionality.

This module tests multimodal tool returns across all providers using a cartesian
product of test dimensions:
- provider: see `ProviderName`
- file_type: image, document, audio, video
- content_source: binary, url, url_force_download
- return_style: direct (return file), tool_return_content (via ToolReturn.content)

The SUPPORT_MATRIX determines expected behavior for each (provider, file_type) pair:
'native', 'fallback', or an ExpectError with error type and match pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest
from typing_extensions import assert_never

from tests.cassette_utils import CassetteContext

if TYPE_CHECKING:
    from vcr.cassette import Cassette

from pydantic_ai import Agent, BinaryContent, BinaryImage
from pydantic_ai.exceptions import ModelHTTPError
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
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..conftest import iter_message_parts, try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider, VertexAILocation

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

ProviderName = Literal[
    'anthropic',
    'bedrock_nova',
    'bedrock_claude',
    'google',
    'google_gemini3',
    'google_vertex',
    'openai_chat',
    'openai_responses',
    'groq',
    'mistral',
    'xai',
]
PROVIDERS = [pytest.param(name, id=name) for name in ProviderName.__args__]

FileType = Literal['image', 'document', 'audio', 'video']
FILE_TYPES = [pytest.param(t, id=t) for t in FileType.__args__]

ContentSource = Literal['binary', 'url', 'url_force_download']
CONTENT_SOURCES = [pytest.param(s, id=s) for s in ContentSource.__args__]

ReturnStyle = Literal['direct', 'tool_return_content']
"""Return style: 'direct' = returns file directly, 'tool_return_content' = via ToolReturn.content."""
RETURN_STYLES = [pytest.param(s, id=s) for s in ReturnStyle.__args__]

Expectation = Literal['native', 'fallback']
"""Expected behavior: 'native' = in tool_result, 'fallback' = separate user msg, 'error' = raises."""


@dataclass
class ExpectError:
    """Expected error for a provider/file_type combination."""

    error_type: type[Exception] = RuntimeError
    match: str | None = None


SUPPORT_MATRIX: dict[tuple[ProviderName, FileType], Expectation | ExpectError] = {
    # Anthropic: images and documents native, audio/video unsupported
    ('anthropic', 'image'): 'native',
    ('anthropic', 'document'): 'native',
    ('anthropic', 'audio'): ExpectError(NotImplementedError, r'(?i)audio.*anthropic|anthropic.*audio'),
    ('anthropic', 'video'): ExpectError(NotImplementedError, r'(?i)video.*anthropic|anthropic.*video'),
    # Bedrock Nova: images, documents, video native; audio unsupported
    ('bedrock_nova', 'image'): 'native',
    ('bedrock_nova', 'document'): 'native',
    ('bedrock_nova', 'audio'): ExpectError(
        NotImplementedError, r'(?i)audio.*(?:bedrock|not supported)|bedrock.*audio|Unsupported.*Bedrock'
    ),
    ('bedrock_nova', 'video'): 'native',
    # Bedrock Claude: images and documents native; audio/video unsupported
    ('bedrock_claude', 'image'): 'native',
    ('bedrock_claude', 'document'): 'native',
    ('bedrock_claude', 'audio'): ExpectError(
        NotImplementedError, r'(?i)audio.*(?:bedrock|not supported)|bedrock.*audio|Unsupported.*Bedrock'
    ),
    ('bedrock_claude', 'video'): ExpectError(ModelHTTPError, r"doesn't support the video content block"),
    # Google GLA: all types native
    ('google', 'image'): 'native',
    ('google', 'document'): 'native',
    ('google', 'audio'): 'native',
    ('google', 'video'): 'native',
    # Google Gemini 3: all types native
    ('google_gemini3', 'image'): 'native',
    ('google_gemini3', 'document'): 'native',
    ('google_gemini3', 'audio'): 'native',
    ('google_gemini3', 'video'): 'native',
    # Google Vertex: all types native
    ('google_vertex', 'image'): 'native',
    ('google_vertex', 'document'): 'native',
    ('google_vertex', 'audio'): 'native',
    ('google_vertex', 'video'): 'native',
    # OpenAI Chat: images and documents via fallback, audio/video unsupported
    ('openai_chat', 'image'): 'fallback',
    ('openai_chat', 'document'): 'fallback',
    # TODO: "ModelHTTPError" happens because we're not catching the incomatibility earlier
    # update code to be consistent with the rest (throw early "NotImplementedError" with consistent wording)
    ('openai_chat', 'audio'): ExpectError(ModelHTTPError, r'expected to be either text or image_url'),
    ('openai_chat', 'video'): ExpectError(NotImplementedError, r'VideoUrl is not supported'),
    # OpenAI Responses: images and documents native, audio/video unsupported
    ('openai_responses', 'image'): 'native',
    ('openai_responses', 'document'): 'native',
    ('openai_responses', 'audio'): ExpectError(NotImplementedError, r'(?i)audio.*openai|unsupported binary'),
    ('openai_responses', 'video'): ExpectError(NotImplementedError, r'VideoUrl is not supported'),
    # xAI: images and documents via fallback, audio/video unsupported
    ('xai', 'image'): 'fallback',
    ('xai', 'document'): 'fallback',
    ('xai', 'audio'): ExpectError(NotImplementedError, r'(?i)not supported by xAI'),
    ('xai', 'video'): ExpectError(NotImplementedError, r'(?i)not supported by xAI'),
    # Groq: images via fallback, everything else unsupported
    # Unsupported types are delegated to _map_user_prompt which raises RuntimeError
    ('groq', 'image'): 'fallback',
    ('groq', 'document'): ExpectError(RuntimeError, r'(?:DocumentUrl|images are supported).*(?:Groq|binary content)'),
    ('groq', 'audio'): ExpectError(RuntimeError, r'(?:AudioUrl|images are supported).*(?:Groq|binary content)'),
    ('groq', 'video'): ExpectError(RuntimeError, r'(?:VideoUrl|images are supported).*(?:Groq|binary content)'),
    # Mistral: images and documents via fallback, audio/video unsupported
    # Unsupported types are delegated to _map_user_prompt which raises RuntimeError
    ('mistral', 'image'): 'fallback',
    ('mistral', 'document'): 'fallback',
    ('mistral', 'audio'): ExpectError(RuntimeError, r'(?:Unsupported content type|not supported in Mistral)'),
    ('mistral', 'video'): ExpectError(RuntimeError, r'(?:VideoUrl|not supported in Mistral)'),
}

# Overrides for specific (provider, file_type, content_source, return_style) combos where
# the behavior differs from the general SUPPORT_MATRIX entry. Keys use None to match all
# values of that dimension.
ERROR_OVERRIDES: dict[tuple[ProviderName, FileType, ContentSource | None, ReturnStyle | None], ExpectError] = {
    # TODO: verify if this error can be fixed by adjusting the behavior to include a text block.
    # re: https://github.com/pydantic/pydantic-ai/pull/3826#discussion_r2795658369
    # if it's not fixed, explain why not, otherwise update
    ('bedrock_nova', 'document', None, 'tool_return_content'): ExpectError(
        ModelHTTPError, r'text block must be included'
    ),
    ('bedrock_claude', 'document', None, 'tool_return_content'): ExpectError(
        ModelHTTPError, r'text block must be included'
    ),
    ('bedrock_claude', 'audio', 'binary', 'tool_return_content'): ExpectError(
        NotImplementedError, r'Unsupported content type for Bedrock user prompts'
    ),
    ('openai_responses', 'audio', 'url', None): ExpectError(ModelHTTPError, r'unsupported'),
    ('openai_responses', 'audio', 'url_force_download', None): ExpectError(ModelHTTPError, r'unsupported'),
    # Groq: tool_return_content errors match the base SUPPORT_MATRIX (all go through _map_user_prompt)
    # Mistral: tool_return_content errors match the base SUPPORT_MATRIX (all go through _map_user_prompt)
}


_OverrideKey = tuple[ProviderName, FileType, ContentSource | None, ReturnStyle | None]


def get_error_info(
    provider: ProviderName, file_type: FileType, content_source: ContentSource, return_style: ReturnStyle
) -> ExpectError | None:
    """Look up error info: first check overrides (most specific match), then the matrix."""
    keys: list[_OverrideKey] = [
        (provider, file_type, content_source, return_style),
        (provider, file_type, content_source, None),
        (provider, file_type, None, return_style),
        (provider, file_type, None, None),
    ]
    for key in keys:
        if key in ERROR_OVERRIDES:
            return ERROR_OVERRIDES[key]
    entry = SUPPORT_MATRIX.get((provider, file_type))
    if isinstance(entry, ExpectError):
        return entry
    return None


MODEL_CONFIGS: dict[ProviderName, tuple[str, Any]] = {
    'anthropic': ('claude-sonnet-4-5', anthropic_available),
    # TODO change to amazon.nova-2-lite-v1:0 and rerecord
    'bedrock_nova': ('us.amazon.nova-pro-v1:0', bedrock_available),
    'bedrock_claude': ('us.anthropic.claude-sonnet-4-5-20250929-v1:0', bedrock_available),
    # TODO change to 3-flash-preview too and rerecord
    'google': ('gemini-2.5-flash', google_available),
    'google_gemini3': ('gemini-3-flash-preview', google_available),
    'google_vertex': ('gemini-3-flash-preview', google_available),
    'openai_chat': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_available),
    'groq': ('meta-llama/llama-4-maverick-17b-128e-instruct', groq_available),
    'mistral': ('mistral-medium-latest', mistral_available),
    'xai': ('grok-4-1-fast-non-reasoning', xai_available),
}


def create_model(
    provider: ProviderName,
    api_keys: dict[str, str],
    bedrock_provider: Any = None,
    xai_provider: Any = None,
    vertex_provider: Any = None,
) -> Model:
    model_name = MODEL_CONFIGS[provider][0]
    if provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_keys['anthropic']))
    elif provider in ('bedrock_nova', 'bedrock_claude'):
        assert bedrock_provider is not None
        return BedrockConverseModel(model_name, provider=bedrock_provider)
    elif provider in ('google', 'google_gemini3'):
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'google_vertex':  # pragma: no cover
        assert vertex_provider is not None
        return GoogleModel(model_name, provider=vertex_provider)
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
    else:
        assert_never(provider)


def is_provider_available(provider: ProviderName) -> bool:
    _, available = MODEL_CONFIGS[provider]
    return bool(available() if callable(available) else available)


@lru_cache
def make_image_binary(assets_path: Path) -> BinaryImage:
    """Create a binary image from the kiwi.jpg test asset."""
    return BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')


def make_image_url() -> ImageUrl:
    """Create an ImageUrl pointing to a public test image."""
    return ImageUrl(url='https://www.gstatic.com/webp/gallery3/1.png')


@lru_cache
def make_document_binary(assets_path: Path) -> BinaryContent:
    """Create a binary PDF from the dummy.pdf test asset."""
    return BinaryContent(data=assets_path.joinpath('dummy.pdf').read_bytes(), media_type='application/pdf')


def make_document_url() -> DocumentUrl:
    """Create a DocumentUrl pointing to a public test PDF."""
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')


@lru_cache
def make_audio_binary(assets_path: Path) -> BinaryContent:
    """Create a binary audio from the marcelo.mp3 test asset."""
    return BinaryContent(data=assets_path.joinpath('marcelo.mp3').read_bytes(), media_type='audio/mpeg')


def make_audio_url() -> AudioUrl:
    """Create an AudioUrl pointing to a public test MP3."""
    return AudioUrl(url='https://download.samplelib.com/mp3/sample-3s.mp3')


@lru_cache
def make_video_binary(assets_path: Path) -> BinaryContent:
    """Create a binary video from the small_video.mp4 test asset."""
    return BinaryContent(data=assets_path.joinpath('small_video.mp4').read_bytes(), media_type='video/mp4')


def make_video_url() -> VideoUrl:
    """Create a VideoUrl pointing to a public test video."""
    return VideoUrl(url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4')


def make_image_url_force_download() -> ImageUrl:
    """Create an ImageUrl with force_download=True."""
    return ImageUrl(url='https://www.gstatic.com/webp/gallery3/1.png', force_download=True)


def make_document_url_force_download() -> DocumentUrl:
    """Create a DocumentUrl with force_download=True."""
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf', force_download=True)


def make_audio_url_force_download() -> AudioUrl:
    """Create an AudioUrl with force_download=True."""
    return AudioUrl(url='https://download.samplelib.com/mp3/sample-3s.mp3', force_download=True)


def make_video_url_force_download() -> VideoUrl:
    """Create a VideoUrl with force_download=True."""
    return VideoUrl(
        url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', force_download=True
    )


CONTENT_FACTORIES: dict[tuple[FileType, ContentSource], Any] = {
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

# Patterns to verify in cassette request/response bodies. A tuple means any one
# of the strings matching is sufficient to confirm the content is present.
CASSETTE_PATTERNS: dict[tuple[FileType, ContentSource], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): (make_image_url().url, '/9j/', '_9j_', 'UklGR', 'iVBOR'),
    ('image', 'url_force_download'): ('/9j/', '_9j_', 'UklGR'),
    ('document', 'binary'): ('%PDF', 'JVBER'),
    ('document', 'url'): (make_document_url().url, '%PDF', 'JVBER'),
    ('document', 'url_force_download'): ('%PDF', 'JVBER'),
    ('audio', 'binary'): ('//t', '__t'),
    ('audio', 'url'): (make_audio_url().url, '//t', '__t'),
    ('audio', 'url_force_download'): ('//t', '__t'),
    ('video', 'binary'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
    ('video', 'url'): (make_video_url().url, 'ftyp', 'ZnR5cA', 'Z0eXB'),
    ('video', 'url_force_download'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
}

# xAI uses a different cassette format (proto) with different content patterns
XAI_CASSETTE_PATTERNS: dict[tuple[FileType, ContentSource], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): make_image_url().url,
    ('image', 'url_force_download'): ('/9j/', '_9j_', 'UklGR'),
    ('document', 'binary'): 'file_id',
    ('document', 'url'): 'file_id',
    ('document', 'url_force_download'): 'file_id',
}


def get_cassette_pattern(
    provider: ProviderName, file_type: FileType, content_source: ContentSource
) -> str | tuple[str, ...] | None:
    """Get the cassette pattern for a provider/file_type/content_source combination."""
    if provider == 'xai':
        return XAI_CASSETTE_PATTERNS.get((file_type, content_source))
    return CASSETTE_PATTERNS.get((file_type, content_source))


FILE_TYPE_CLASSES: dict[FileType, tuple[type, ...]] = {
    'image': (BinaryImage, ImageUrl),
    'document': (DocumentUrl, BinaryContent),
    'audio': (AudioUrl, BinaryContent),
    'video': (VideoUrl, BinaryContent),
}


def _is_file_type(item: Any, file_type: FileType) -> bool:
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


def assert_file_in_tool_return(messages: list[ModelMessage], file_type: FileType) -> None:
    """Assert that file content of the expected type is present in a ToolReturnPart."""
    for trp in iter_message_parts(messages, ModelRequest, ToolReturnPart):
        for f in trp.files:  # pragma: no branch
            if _is_file_type(f, file_type):  # pragma: no branch
                return
    raise AssertionError(f'No {file_type} found in any ToolReturnPart')  # pragma: no cover


def assert_file_in_user_prompt(messages: list[ModelMessage], file_type: FileType) -> None:
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
    file_type: FileType,
    return_style: ReturnStyle = 'direct',
) -> None:
    """Assert that multimodal content was handled correctly.

    For 'tool_return_content' style, files go to a separate UserPromptPart.
    For 'direct' style, files should be in ToolReturnPart.
    """
    if return_style == 'tool_return_content':
        assert_file_in_user_prompt(messages, file_type)
    else:
        assert_file_in_tool_return(messages, file_type)


@pytest.fixture
def vertex_provider(request: pytest.FixtureRequest, vertex_provider_auth: None) -> Any:
    """Override conftest's vertex_provider to return None for non-vertex tests instead of skipping."""
    if 'google_vertex' not in request.node.name:  # pyright: ignore[reportUnknownMemberType]
        return None

    if not google_available():  # pragma: no cover
        pytest.skip('google is not installed')

    project = os.getenv('GOOGLE_CLOUD_PROJECT', os.getenv('GOOGLE_PROJECT'))
    if not project:  # pragma: no cover
        pytest.skip('GOOGLE_CLOUD_PROJECT not set')
    location = os.getenv('GOOGLE_LOCATION', 'global')  # pragma: no cover
    return GoogleProvider(project=project, location=cast(VertexAILocation, location))  # pragma: no cover


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


@pytest.mark.parametrize('provider', PROVIDERS)
@pytest.mark.parametrize('file_type', FILE_TYPES)
@pytest.mark.parametrize('content_source', CONTENT_SOURCES)
@pytest.mark.parametrize('return_style', RETURN_STYLES)
async def test_multimodal_tool_return_matrix(
    provider: ProviderName,
    file_type: FileType,
    content_source: ContentSource,
    return_style: ReturnStyle,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    vertex_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext | None,
    disable_ssrf_protection_for_vcr: None,
):
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    error_info = get_error_info(provider, file_type, content_source, return_style)
    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)
    content_factory = CONTENT_FACTORIES[(file_type, content_source)]
    content = content_factory(assets_path) if content_source == 'binary' else content_factory()

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_file() -> Any:
        if return_style == 'direct':
            return content
        else:
            return ToolReturn(return_value='File attached', content=[content])

    prompt = f'Use the get_file tool now to retrieve a {file_type} file, then describe what you received.'

    if error_info:
        with pytest.raises(error_info.error_type, match=error_info.match):
            await agent.run(prompt, usage_limits=UsageLimits(output_tokens_limit=100000))
    else:
        result = await agent.run(prompt, usage_limits=UsageLimits(output_tokens_limit=100000))
        assert result.output, 'Expected non-empty response from model'
        assert_multimodal_result(result.all_messages(), file_type, return_style)
        if cassette_ctx and (pattern := get_cassette_pattern(provider, file_type, content_source)):  # pragma: no branch
            cassette_ctx.verify_contains(pattern)
        if SUPPORT_MATRIX[(provider, file_type)] == 'fallback' and return_style == 'direct' and cassette_ctx:
            cassette_ctx.verify_contains('See file')


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_mixed_content_ordering(
    provider: ProviderName,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    vertex_provider: Any,
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

    image_support = SUPPORT_MATRIX[(provider, 'image')]
    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)
    image = make_image_binary(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_mixed_content() -> list[Any]:
        return ['Here is the image:', image, {'metadata': 'test_value'}]

    result = await agent.run(
        'Call the get_mixed_content tool and describe what you received.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    if cassette_ctx:  # pragma: no branch
        if image_support == 'native':
            if provider == 'google_gemini3':
                cassette_ctx.verify_ordering(('/9j/', '_9j_'), 'Here is the image:', 'metadata')
            elif provider == 'google':
                cassette_ctx.verify_ordering('Here is the image:', 'metadata', ('/9j/', '_9j_'))
            else:
                cassette_ctx.verify_ordering('Here is the image:', ('/9j/', '_9j_'), 'metadata')
        else:
            cassette_ctx.verify_ordering('Here is the image:', 'metadata')
            cassette_ctx.verify_contains(('/9j/', '_9j_'))
            cassette_ctx.verify_contains('See file')


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_model_sees_multiple_images(
    provider: ProviderName,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    vertex_provider: Any,
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

    if isinstance(SUPPORT_MATRIX[(provider, 'image')], ExpectError):  # pragma: no cover
        pytest.skip(f'{provider} does not support images in tool returns')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)
    kiwi_image = make_image_binary(assets_path)
    url_image = make_image_url()

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_images() -> list[Any]:
        return [kiwi_image, url_image]

    result = await agent.run(
        'Call the get_images tool. One image shows a fruit - what fruit is it? Just name the fruit.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert 'kiwi' in result.output.lower(), f'Model should identify kiwi fruit, got: {result.output}'
    image_support = SUPPORT_MATRIX[(provider, 'image')]
    if cassette_ctx:  # pragma: no branch
        cassette_ctx.verify_contains(('/9j/', '_9j_'))
        cassette_ctx.verify_contains(('UklGR', 'iVBOR', 'gstatic.com/webp/gallery3/1.png'))
        if image_support == 'fallback':
            cassette_ctx.verify_contains('See file')


@pytest.mark.skipif(not openai_available(), reason='openai dependencies not installed')
async def test_vendor_metadata_detail(
    openai_api_key: str,
    assets_path: Path,
    allow_model_requests: None,
    vcr: Cassette | None,
):
    """Test that vendor_metadata with detail setting is handled correctly."""
    model = OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    image_binary = BinaryImage(
        data=assets_path.joinpath('kiwi.jpg').read_bytes(),
        media_type='image/jpeg',
        vendor_metadata={'detail': 'high'},
    )
    image_url = ImageUrl(
        url=make_image_url().url,
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
    if vcr:  # pragma: no branch
        # Can't use cassette_ctx fixture here: it requires a 'provider' parametrize arg
        ctx = CassetteContext('openai_responses', vcr, 'test_vendor_metadata_detail', 'test_multimodal_tool_returns')
        ctx.verify_contains('"detail": "high"', '"detail": "low"')


async def test_text_plain_document_anthropic(
    anthropic_api_key: str,
    assets_path: Path,
    allow_model_requests: None,
    vcr: Cassette | None,
):
    """Test that text/plain documents are handled correctly by Anthropic."""
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
    # Can't use cassette_ctx fixture here: it requires a 'provider' parametrize arg
    ctx = CassetteContext('anthropic', vcr, 'test_text_plain_document_anthropic', 'test_multimodal_tool_returns')
    ctx.verify_contains('Dummy TXT file')


@pytest.mark.skipif(not mistral_available(), reason='mistral dependencies not installed')
async def test_non_pdf_document_url_error(
    mistral_api_key: str,
    allow_model_requests: None,
):
    """Test that Mistral raises RuntimeError for non-PDF DocumentUrl in tool returns."""
    model = MistralModel('mistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_file() -> DocumentUrl:
        return DocumentUrl(url='https://example.com/file.txt', media_type='text/plain')

    with pytest.raises(RuntimeError, match='DocumentUrl other than PDF is not supported in Mistral'):
        await agent.run(
            'Use the get_file tool to retrieve a file.',
            usage_limits=UsageLimits(output_tokens_limit=100000),
        )


@pytest.mark.skipif(not bedrock_available(), reason='bedrock dependencies not installed')
async def test_empty_tool_return(
    bedrock_provider: Any,
    allow_model_requests: None,
):
    """Test that Bedrock handles tools returning empty strings without crashing.

    Bedrock requires at least one content block in tool results. This test verifies
    that empty string returns are handled gracefully (e.g. sent as empty text block).
    """
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_empty() -> str:
        return ''

    result = await agent.run(
        'Use the get_empty tool.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert tool_returns, 'Expected at least one ToolReturnPart'
    assert tool_returns[0].model_response_str() == ''


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
