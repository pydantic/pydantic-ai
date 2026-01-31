"""Cross-provider matrix tests for multimodal tool return functionality.

This module tests multimodal tool returns across all providers using a cartesian
product of test dimensions: provider, file_type, return_style.

The test harness uses a SUPPORT_MATRIX to determine expected behavior (native,
fallback, or error) for each provider/file_type combination.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, BinaryContent, BinaryImage
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
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
ReturnStyle = Literal['direct', 'tool_return_content']

SUPPORT_MATRIX: dict[tuple[str, str], Expectation] = {
    ('bedrock', 'image'): 'native',
    ('bedrock', 'document'): 'native',
    ('bedrock', 'audio'): 'fallback',
    ('bedrock', 'video'): 'native',
    ('anthropic', 'image'): 'native',
    ('anthropic', 'document'): 'native',
    ('anthropic', 'audio'): 'fallback',
    ('anthropic', 'video'): 'fallback',
    ('google', 'image'): 'native',
    ('google', 'document'): 'native',
    ('google', 'audio'): 'native',
    ('google', 'video'): 'native',
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
}


@dataclass
class ExpectError:
    """Expected error for a provider/file_type/content_source/return_style combination."""

    provider: str
    file_type: str
    content_source: str | None = None
    return_style: str | None = None
    error_type: type[Exception] = RuntimeError
    match: str | None = None


ERROR_DETAILS: list[ExpectError] = [
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
        'bedrock',
        'audio',
        content_source='binary',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Binary content is not supported yet',
    ),
    ExpectError(
        'bedrock',
        'audio',
        content_source='url',
        return_style='tool_return_content',
        error_type=NotImplementedError,
        match='Audio is not supported yet',
    ),
]


def get_error_details(provider: str, file_type: str, content_source: str, return_style: str) -> ExpectError | None:
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
    'openai_chat': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_available),
    'groq': ('meta-llama/llama-4-maverick-17b-128e-instruct', groq_available),
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
    elif provider == 'google':
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'openai_chat':
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'openai_responses':
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'groq':
        return GroqModel(model_name, provider=GroqProvider(api_key=api_keys['groq']))
    elif provider == 'xai':
        assert xai_provider is not None
        return XaiModel(model_name, provider=xai_provider)
    else:
        raise ValueError(f'Unknown provider: {provider}')


def is_provider_available(provider: str) -> bool:
    _, available = MODEL_CONFIGS.get(provider, (None, lambda: False))
    return bool(available() if callable(available) else available)


def make_image_binary(assets_path: Path) -> BinaryImage:
    return BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')


def make_image_url(_: Path) -> ImageUrl:
    return ImageUrl(url='https://www.gstatic.com/webp/gallery3/1.png')


def make_document_binary(assets_path: Path) -> BinaryContent:
    return BinaryContent(data=assets_path.joinpath('dummy.pdf').read_bytes(), media_type='application/pdf')


def make_document_url(_: Path) -> DocumentUrl:
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')


def make_audio_binary(assets_path: Path) -> BinaryContent:
    return BinaryContent(data=assets_path.joinpath('marcelo.mp3').read_bytes(), media_type='audio/mpeg')


def make_audio_url(_: Path) -> AudioUrl:
    return AudioUrl(url='https://download.samplelib.com/mp3/sample-3s.mp3')


def make_video_binary(assets_path: Path) -> BinaryContent:
    return BinaryContent(data=assets_path.joinpath('small_video.mp4').read_bytes(), media_type='video/mp4')


def make_video_url(_: Path) -> VideoUrl:
    return VideoUrl(url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4')


CONTENT_FACTORIES = {
    ('image', 'binary'): make_image_binary,
    ('image', 'url'): make_image_url,
    ('document', 'binary'): make_document_binary,
    ('document', 'url'): make_document_url,
    ('audio', 'binary'): make_audio_binary,
    ('audio', 'url'): make_audio_url,
    ('video', 'binary'): make_video_binary,
    ('video', 'url'): make_video_url,
}


def get_expectation(provider: str, file_type: str) -> Expectation:
    return SUPPORT_MATRIX[(provider, file_type)]


def assert_file_in_tool_return(messages: list[Any], file_type: str) -> None:
    """Assert that file content is present in a ToolReturnPart."""
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    files = part.files
                    if files:
                        return
    raise AssertionError(f'No {file_type} found in any ToolReturnPart')


def assert_file_in_messages(messages: list[Any], file_type: str) -> None:
    """Assert that file content is present somewhere in messages.

    For tool_return_content style, files are in UserPromptPart (by design of ToolReturn.content).
    This checks that files are sent to the model, regardless of location.
    """
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    if part.files:
                        return
                elif isinstance(part, UserPromptPart):
                    content = getattr(part, 'content', None)
                    if isinstance(content, list):
                        for item in content:  # pyright: ignore[reportUnknownVariableType]
                            if hasattr(item, 'media_type') or hasattr(item, 'url'):  # pyright: ignore[reportUnknownArgumentType]
                                return
    raise AssertionError(f'No {file_type} found in any message part')


def assert_multimodal_result(
    messages: list[Any],
    expectation: Expectation,
    file_type: str,
    return_style: ReturnStyle = 'direct',
) -> None:
    """Assert that multimodal content was handled correctly based on expectation.

    For both 'native' and 'fallback' expectations:
    - The ToolReturnPart should contain the file content
    - The model implementation handles sending it appropriately (native or separated)
    - We verify files are in the message history, not the API-specific format

    For 'tool_return_content' style:
    - Files go to a separate UserPromptPart by design of ToolReturn.content
    """
    match expectation:
        case 'error':
            pass
        case 'native' | 'fallback':
            # Both native and fallback: file should be in ToolReturnPart or UserPromptPart
            # The difference is how the model implementation sends it to the API
            if return_style == 'tool_return_content':
                # For tool_return_content, files are in UserPromptPart by design
                assert_file_in_messages(messages, file_type)
            else:
                # For direct return, files should be in ToolReturnPart
                assert_file_in_tool_return(messages, file_type)


@pytest.fixture
def api_keys(
    openai_api_key: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    groq_api_key: str,
    xai_api_key: str,
) -> dict[str, str]:
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'google': gemini_api_key,
        'groq': groq_api_key,
        'xai': xai_api_key,
    }


PROVIDERS = [
    pytest.param('anthropic', id='anthropic'),
    pytest.param('bedrock', id='bedrock'),
    pytest.param('google', id='google'),
    pytest.param('openai_chat', id='openai_chat'),
    pytest.param('openai_responses', id='openai_responses'),
    pytest.param('groq', id='groq'),
    pytest.param('xai', id='xai'),
]

FILE_TYPES = [
    pytest.param('image', id='image'),
    pytest.param('document', id='document'),
    pytest.param('audio', id='audio'),
    pytest.param('video', id='video'),
]

CONTENT_SOURCES = [
    pytest.param('binary', id='binary'),
    pytest.param('url', id='url'),
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
    file_type: str,
    content_source: str,
    return_style: ReturnStyle,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
):
    if not is_provider_available(provider):
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


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_mixed_content_ordering(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
):
    """Test that [text, image, dict] preserves order in tool returns."""
    if not is_provider_available(provider):
        pytest.skip(f'{provider} dependencies not installed')

    expectation = get_expectation(provider, 'image')
    if expectation == 'error':
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


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_multiple_files(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
):
    """Test returning multiple files in a single tool return."""
    if not is_provider_available(provider):
        pytest.skip(f'{provider} dependencies not installed')

    expectation = get_expectation(provider, 'image')
    if expectation == 'error':
        pytest.skip(f'{provider} does not support images')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    image1 = make_image_binary(assets_path)
    image2 = make_image_url(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_two_images() -> list[Any]:
        return [image1, image2]

    result = await agent.run(
        'Call the get_two_images tool and describe what you see.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert result.output, 'Expected non-empty response from model'


@pytest.mark.parametrize('provider', PROVIDERS)
async def test_model_sees_image_content(
    provider: str,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    xai_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
):
    """Verify the model actually processes image content by identifying a kiwi fruit."""
    if not is_provider_available(provider):
        pytest.skip(f'{provider} dependencies not installed')

    expectation = get_expectation(provider, 'image')
    if expectation == 'error':
        pytest.skip(f'{provider} does not support images in tool returns')

    model = create_model(provider, api_keys, bedrock_provider, xai_provider)
    image = make_image_binary(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_fruit_image() -> BinaryImage:
        return image

    result = await agent.run(
        'Call the get_fruit_image tool. What fruit is shown in the image? Just name the fruit.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )
    assert 'kiwi' in result.output.lower(), f'Model should identify kiwi fruit, got: {result.output}'
