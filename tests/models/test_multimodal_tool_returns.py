"""Cross-provider matrix tests for multimodal tool return functionality.

This module tests multimodal tool returns across all providers using a cartesian
product of test dimensions:
- provider: see `ProviderName`
- file_type: image, document, audio, video
- content_source: binary, url, url_force_download
- return_style: direct (return file), tool_return_content (via ToolReturn.content)

The SUPPORT_MATRIX determines expected behavior for each (provider, file_type) pair:
'in_tool_result', 'as_user_content', or an ExpectError with error type and match pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from typing_extensions import assert_never

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
from tests.cassette_utils import CassetteContext

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
    'google_2_5',
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

Expectation = Literal['in_tool_result', 'as_user_content']


@dataclass
class ExpectError:
    """Expected error for a provider/file_type combination."""

    error_type: type[Exception] = NotImplementedError
    match: str | None = None


SUPPORT_MATRIX: dict[tuple[ProviderName, FileType], Expectation | ExpectError] = {
    # Anthropic: images and documents in_tool_result, audio/video unsupported
    ('anthropic', 'image'): 'in_tool_result',
    ('anthropic', 'document'): 'in_tool_result',
    ('anthropic', 'audio'): ExpectError(NotImplementedError, r'(?i)audio.*anthropic|anthropic.*audio'),
    ('anthropic', 'video'): ExpectError(NotImplementedError, r'(?i)video.*anthropic|anthropic.*video'),
    # Bedrock Nova: images, documents, video in_tool_result; audio unsupported
    ('bedrock_nova', 'image'): 'in_tool_result',
    ('bedrock_nova', 'document'): 'in_tool_result',
    ('bedrock_nova', 'audio'): ExpectError(
        NotImplementedError, r'(?i)audio.*(?:bedrock|not supported)|bedrock.*audio|Unsupported.*Bedrock'
    ),
    ('bedrock_nova', 'video'): 'in_tool_result',
    # Bedrock Claude: images and documents in_tool_result; audio/video unsupported
    ('bedrock_claude', 'image'): 'in_tool_result',
    ('bedrock_claude', 'document'): 'in_tool_result',
    ('bedrock_claude', 'audio'): ExpectError(
        NotImplementedError, r'(?i)audio.*(?:bedrock|not supported)|bedrock.*audio|Unsupported.*Bedrock'
    ),
    ('bedrock_claude', 'video'): ExpectError(ModelHTTPError, r"doesn't support the video content block"),
    # Google GLA (gemini-2.5): all types as_user_content (no inline tool return support)
    ('google_2_5', 'image'): 'as_user_content',
    ('google_2_5', 'document'): 'as_user_content',
    ('google_2_5', 'audio'): 'as_user_content',
    ('google_2_5', 'video'): 'as_user_content',
    # Google Gemini 3: all types in_tool_result
    ('google_gemini3', 'image'): 'in_tool_result',
    ('google_gemini3', 'document'): 'in_tool_result',
    ('google_gemini3', 'audio'): 'in_tool_result',
    ('google_gemini3', 'video'): 'in_tool_result',
    # Google Vertex: all types in_tool_result
    ('google_vertex', 'image'): 'in_tool_result',
    ('google_vertex', 'document'): 'in_tool_result',
    ('google_vertex', 'audio'): 'in_tool_result',
    ('google_vertex', 'video'): 'in_tool_result',
    # OpenAI Chat: images and documents as_user_content, audio/video unsupported
    ('openai_chat', 'image'): 'as_user_content',
    ('openai_chat', 'document'): 'as_user_content',
    ('openai_chat', 'audio'): ExpectError(
        NotImplementedError, r'AudioUrl is not supported in OpenAI Chat Completions tool returns'
    ),
    ('openai_chat', 'video'): ExpectError(
        NotImplementedError, r'VideoUrl is not supported in OpenAI Chat Completions tool returns'
    ),
    # OpenAI Responses: images and documents in_tool_result, audio/video unsupported
    ('openai_responses', 'image'): 'in_tool_result',
    ('openai_responses', 'document'): 'in_tool_result',
    ('openai_responses', 'audio'): ExpectError(NotImplementedError, r'(?i)audio.*openai responses|unsupported binary'),
    ('openai_responses', 'video'): ExpectError(NotImplementedError, r'VideoUrl is not supported in OpenAI Responses'),
    # xAI: images and documents as_user_content, audio/video unsupported
    ('xai', 'image'): 'as_user_content',
    ('xai', 'document'): 'as_user_content',
    ('xai', 'audio'): ExpectError(NotImplementedError, r'(?i)not supported in xAI'),
    ('xai', 'video'): ExpectError(NotImplementedError, r'(?i)not supported in xAI'),
    # Groq: images as_user_content, everything else unsupported
    ('groq', 'image'): 'as_user_content',
    ('groq', 'document'): ExpectError(match=r'(?:DocumentUrl|images are supported).*Groq user prompts'),
    ('groq', 'audio'): ExpectError(match=r'(?:AudioUrl|images are supported).*Groq user prompts'),
    ('groq', 'video'): ExpectError(match=r'(?:VideoUrl|images are supported).*Groq user prompts'),
    # Mistral: images and documents as_user_content, audio/video unsupported
    ('mistral', 'image'): 'as_user_content',
    ('mistral', 'document'): 'as_user_content',
    ('mistral', 'audio'): ExpectError(
        match=r'(?:AudioUrl|BinaryContent other than image or PDF) is not supported in Mistral user prompts'
    ),
    ('mistral', 'video'): ExpectError(
        match=r'(?:VideoUrl|BinaryContent other than image or PDF) is not supported in Mistral user prompts'
    ),
}

# Overrides for specific (provider, file_type, content_source, return_style) combos where
# the behavior differs from the general SUPPORT_MATRIX entry. Keys use None to match all
# values of that dimension.
ERROR_OVERRIDES: dict[tuple[ProviderName, FileType, ContentSource | None, ReturnStyle | None], ExpectError] = {
    ('bedrock_claude', 'audio', 'binary', 'tool_return_content'): ExpectError(
        NotImplementedError, r'Unsupported content type for Bedrock user prompts'
    ),
    # tool_return_content routes through UserPromptPart which uses different error messages
    ('openai_chat', 'audio', None, 'tool_return_content'): ExpectError(
        ModelHTTPError, r'expected to be either text or image_url'
    ),
    ('openai_chat', 'video', None, 'tool_return_content'): ExpectError(
        NotImplementedError, r'VideoUrl is not supported in OpenAI Chat Completions user prompts'
    ),
    ('openai_responses', 'audio', 'url', None): ExpectError(ModelHTTPError, r'unsupported'),
    ('openai_responses', 'audio', 'url_force_download', None): ExpectError(ModelHTTPError, r'unsupported'),
    # Groq: tool_return_content errors match the base SUPPORT_MATRIX (all go through _map_user_prompt)
    # Mistral: tool_return_content errors match the base SUPPORT_MATRIX (all go through _map_user_prompt)
    # Vertex AI can't crawl certain URLs blocked by robots.txt (gstatic.com, test-videos.co.uk).
    # force_download variants work since the client downloads locally before sending to Vertex.
    ('google_vertex', 'image', 'url', None): ExpectError(ModelHTTPError, r'URL_ROBOTED|ROBOTED_DENIED'),
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
    'bedrock_nova': ('us.amazon.nova-2-lite-v1:0', bedrock_available),
    'bedrock_claude': ('us.anthropic.claude-sonnet-4-5-20250929-v1:0', bedrock_available),
    'google_2_5': ('gemini-2.5-flash', google_available),
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
    elif provider in ('google_2_5', 'google_gemini3'):
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


IMAGE_URL = 'https://www.gstatic.com/webp/gallery3/1.png'
DOCUMENT_URL = 'https://pdfobject.com/pdf/sample.pdf'
AUDIO_URL = 'https://download.samplelib.com/mp3/sample-3s.mp3'
VIDEO_URL = 'https://www.w3schools.com/html/mov_bbb.mp4'


URL_FACTORIES: dict[tuple[FileType, ContentSource], Any] = {
    ('image', 'url'): lambda: ImageUrl(url=IMAGE_URL),
    ('image', 'url_force_download'): lambda: ImageUrl(url=IMAGE_URL, force_download=True),
    ('document', 'url'): lambda: DocumentUrl(url=DOCUMENT_URL),
    ('document', 'url_force_download'): lambda: DocumentUrl(url=DOCUMENT_URL, force_download=True),
    ('audio', 'url'): lambda: AudioUrl(url=AUDIO_URL),
    ('audio', 'url_force_download'): lambda: AudioUrl(url=AUDIO_URL, force_download=True),
    ('video', 'url'): lambda: VideoUrl(url=VIDEO_URL),
    ('video', 'url_force_download'): lambda: VideoUrl(url=VIDEO_URL, force_download=True),
}

# Patterns to verify in cassette request/response bodies. A tuple means any one
# of the strings matching is sufficient to confirm the content is present.
# base64 variants: standard (/9j/) vs URL-safe (_9j_) encoding of the same bytes,
# used because providers differ in which encoding they accept/return.
CASSETTE_PATTERNS: dict[tuple[FileType, ContentSource], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): (IMAGE_URL, '/9j/', '_9j_', 'UklGR', 'iVBOR'),
    ('image', 'url_force_download'): ('/9j/', '_9j_', 'UklGR'),
    ('document', 'binary'): ('%PDF', 'JVBER'),
    ('document', 'url'): (DOCUMENT_URL, '%PDF', 'JVBER'),
    ('document', 'url_force_download'): ('%PDF', 'JVBER'),
    ('audio', 'binary'): ('//t', '__t'),
    ('audio', 'url'): (AUDIO_URL, '//t', '__t'),
    ('audio', 'url_force_download'): ('//t', '__t'),
    ('video', 'binary'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
    ('video', 'url'): (VIDEO_URL, 'ftyp', 'ZnR5cA', 'Z0eXB'),
    ('video', 'url_force_download'): ('ftyp', 'ZnR5cA', 'Z0eXB'),
}

# xAI uses a different cassette format (proto) with different content patterns
XAI_CASSETTE_PATTERNS: dict[tuple[FileType, ContentSource], str | tuple[str, ...]] = {
    ('image', 'binary'): ('/9j/', '_9j_'),
    ('image', 'url'): IMAGE_URL,
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
    if not project:
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


@pytest.fixture
def binary_contents(
    image_content: BinaryImage,
    document_content: BinaryContent,
    audio_content: BinaryContent,
    video_content: BinaryContent,
) -> dict[FileType, BinaryImage | BinaryContent]:
    return {
        'image': image_content,
        'document': document_content,
        'audio': audio_content,
        'video': video_content,
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
    binary_contents: dict[FileType, BinaryImage | BinaryContent],
    allow_model_requests: None,
    cassette_ctx: CassetteContext,
    disable_ssrf_protection_for_vcr: None,
):
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    error_info = get_error_info(provider, file_type, content_source, return_style)
    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)
    content = binary_contents[file_type] if content_source == 'binary' else URL_FACTORIES[(file_type, content_source)]()

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
        if pattern := get_cassette_pattern(provider, file_type, content_source):  # pragma: no branch
            cassette_ctx.verify_contains(pattern)
        if SUPPORT_MATRIX[(provider, file_type)] == 'as_user_content' and return_style == 'direct':
            cassette_ctx.verify_contains('See file')


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_mixed_content_ordering(
    provider: ProviderName,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    vertex_provider: Any,
    image_content: BinaryImage,
    allow_model_requests: None,
    cassette_ctx: CassetteContext,
):
    """Test that [text, image, dict] are sent to the API in the correct order.

    Returns mixed content types and verifies the cassette preserves ordering,
    catching bugs where content might be reordered or silently dropped.

    For in_tool_result providers, strict ordering is verified within the tool_result.
    For as_user_content providers, the image is separated into a user message, so we
    only verify all content is present (ordering across messages may differ).
    """
    if not is_provider_available(provider):  # pragma: no cover
        pytest.skip(f'{provider} dependencies not installed')

    image_support = SUPPORT_MATRIX[(provider, 'image')]
    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_mixed_content() -> list[Any]:
        return ['Here is the image:', image_content, {'pydantic_ai_marker': 'test_42'}]

    result = await agent.run(
        'Call the get_mixed_content tool and describe what you received.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'
    if image_support == 'in_tool_result':
        if provider in ('google_2_5', 'google_gemini3', 'google_vertex'):
            # Google Gemini 3 serializes function_response.parts (in_tool_result image) before
            # function_response.response (text/dict), so image data appears first.
            cassette_ctx.verify_ordering(('/9j/', '_9j_'), 'Here is the image:', 'pydantic_ai_marker')
        else:
            cassette_ctx.verify_ordering('Here is the image:', ('/9j/', '_9j_'), 'pydantic_ai_marker')
    else:
        cassette_ctx.verify_ordering('Here is the image:', 'pydantic_ai_marker')
        cassette_ctx.verify_contains(('/9j/', '_9j_'))
        cassette_ctx.verify_contains('See file')


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_model_sees_multiple_images(
    provider: ProviderName,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    vertex_provider: Any,
    image_content: BinaryImage,
    allow_model_requests: None,
    cassette_ctx: CassetteContext,
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

    if provider == 'google_vertex':  # pragma: no cover
        pytest.skip('Vertex AI cannot crawl the test image URL (robots.txt)')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider, vertex_provider)
    kiwi_image = image_content
    url_image = URL_FACTORIES[('image', 'url')]()

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
    cassette_ctx.verify_contains(('/9j/', '_9j_'))
    cassette_ctx.verify_contains(('UklGR', 'iVBOR', IMAGE_URL))
    if image_support == 'as_user_content':
        cassette_ctx.verify_contains('See file')


@pytest.mark.skipif(not openai_available(), reason='openai dependencies not installed')
async def test_vendor_metadata_detail(
    openai_api_key: str,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext,
):
    """Test that vendor_metadata with detail setting is handled correctly."""
    model = OpenAIResponsesModel('gpt-5-mini', provider=OpenAIProvider(api_key=openai_api_key))
    image_binary = BinaryImage(
        data=assets_path.joinpath('kiwi.jpg').read_bytes(),
        media_type='image/jpeg',
        vendor_metadata={'detail': 'high'},
    )
    image_url = ImageUrl(
        url=IMAGE_URL,
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
    cassette_ctx.verify_contains('"detail": "high"', '"detail": "low"')


async def test_text_plain_document_anthropic(
    anthropic_api_key: str,
    assets_path: Path,
    allow_model_requests: None,
    cassette_ctx: CassetteContext,
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
    cassette_ctx.verify_contains('Dummy TXT file')


@pytest.mark.skipif(not mistral_available(), reason='mistral dependencies not installed')
async def test_non_pdf_document_url_error(
    mistral_api_key: str,
    allow_model_requests: None,
):
    """Test that Mistral raises NotImplementedError for non-PDF DocumentUrl in tool returns."""
    model = MistralModel('mistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_file() -> DocumentUrl:
        return DocumentUrl(url='https://example.com/file.txt', media_type='text/plain')

    with pytest.raises(
        NotImplementedError, match='DocumentUrl other than PDF is not supported in Mistral user prompts'
    ):
        await agent.run(
            'Use the get_file tool to retrieve a file.',
            usage_limits=UsageLimits(output_tokens_limit=100000),
        )


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
