"""Cross-provider VCR tests for multimodal tool return functionality.

This module consolidates multimodal tool return tests across all providers.
Tests are parametrized by provider and file type, using cassettes recorded
against live APIs.

Key behaviors tested:
- Tool returns with BinaryContent (images, documents, audio, video)
- Tool returns with FileUrl types (ImageUrl, DocumentUrl, AudioUrl, VideoUrl)
- Provider-specific handling: native in tool_result vs separate user message
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, BinaryContent, BinaryImage
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    VideoUrl,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import

# =============================================================================
# Provider imports (conditional)
# =============================================================================

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

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# =============================================================================
# Content factory functions
# =============================================================================


def _image_binary(assets_path: Path) -> BinaryImage:
    """Create BinaryImage from test assets."""
    return BinaryImage(data=assets_path.joinpath('kiwi.jpg').read_bytes(), media_type='image/jpeg')


def _image_url(_: Path) -> ImageUrl:
    """Create ImageUrl for testing."""
    return ImageUrl(url='https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png')


def _document_binary(assets_path: Path) -> BinaryContent:
    """Create BinaryContent PDF from test assets."""
    return BinaryContent(data=assets_path.joinpath('dummy.pdf').read_bytes(), media_type='application/pdf')


def _document_url(_: Path) -> DocumentUrl:
    """Create DocumentUrl for testing."""
    return DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')


def _audio_binary(assets_path: Path) -> BinaryContent:
    """Create BinaryContent audio from test assets."""
    return BinaryContent(data=assets_path.joinpath('marcelo.mp3').read_bytes(), media_type='audio/mpeg')


def _audio_url(_: Path) -> AudioUrl:
    """Create AudioUrl for testing."""
    return AudioUrl(url='https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')


def _video_binary(assets_path: Path) -> BinaryContent:
    """Create BinaryContent video from test assets."""
    return BinaryContent(data=assets_path.joinpath('small_video.mp4').read_bytes(), media_type='video/mp4')


def _video_url(_: Path) -> VideoUrl:
    """Create VideoUrl for testing."""
    return VideoUrl(url='https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4')


# =============================================================================
# Case dataclass
# =============================================================================


FileType = Literal['image', 'document', 'audio', 'video']
ContentSource = Literal['binary', 'url']
Behavior = Literal['native', 'user_msg', 'dropped']


@dataclass
class Case:
    """A single test case for multimodal tool return behavior."""

    id: str
    provider: str
    file_type: FileType
    source: ContentSource
    expected_behavior: Behavior
    expected_message_structure: Any  # snapshot() stored here per case
    content_factory: Callable[[Path], Any]
    prompt: str = 'Describe what the tool returned.'


# =============================================================================
# Provider configuration
# =============================================================================

PROVIDER_MODELS: dict[str, tuple[str, Callable[[], bool]]] = {
    'openai_chat': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_available),  # Same package as openai_chat
    'anthropic': ('claude-sonnet-4-5', anthropic_available),
    'google': ('gemini-3-flash-preview', google_available),
    'bedrock': ('us.amazon.nova-pro-v1:0', bedrock_available),
}


def get_model(provider: str, api_keys: dict[str, str], bedrock_provider: Any = None) -> Model:
    """Create a model instance for the given provider."""
    model_name, _ = PROVIDER_MODELS[provider]

    if provider == 'openai_chat':
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'openai_responses':
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_keys['anthropic']))
    elif provider == 'google':
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'bedrock':
        assert bedrock_provider is not None, 'bedrock_provider fixture required for bedrock tests'
        return BedrockConverseModel(model_name, provider=bedrock_provider)
    else:
        raise ValueError(f'Unknown provider: {provider}')


def should_skip_case(case: Case) -> str | None:
    """Check if a case should be skipped based on provider availability."""
    _, is_available = PROVIDER_MODELS.get(case.provider, (None, lambda: False))
    if callable(is_available):
        if not is_available():
            return f'{case.provider} not installed'
    elif not is_available:
        return f'{case.provider} not installed'

    return None


# =============================================================================
# Helper functions
# =============================================================================


class _MessageStructure(TypedDict):
    type: str
    parts: list[str]


def get_message_structure(messages: list[ModelRequest | ModelResponse]) -> list[_MessageStructure]:
    """Extract simplified message structure for snapshot comparison."""
    result: list[_MessageStructure] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            result.append(
                {
                    'type': 'request',
                    'parts': [type(p).__name__ for p in msg.parts],
                }
            )
        else:
            result.append(
                {
                    'type': 'response',
                    'parts': [type(p).__name__ for p in msg.parts],
                }
            )
    return result


# =============================================================================
# Test cases
# Behavior matrix (VCR tested cases):
# | Provider          | Image    | Document | Audio   | Video    |
# |-------------------|----------|----------|---------|----------|
# | Anthropic         | native   | user_msg | --      | --       |
# | Bedrock           | native   | user_msg | dropped | user_msg |
# | Google            | native   | native   | native  | native   |
# | OpenAI Chat       | user_msg | user_msg | --      | --       |
# | OpenAI Responses  | native   | native   | --      | --       |
#
# "--" = Not tested via VCR (raises exception, model limitation, or API bug)
# See test_multimodal_tool_returns_unit.py for exception behavior tests
# =============================================================================

# fmt: off
CASES = [
    # === Anthropic: Images native, documents via user_msg ===
    # Note: Audio/video binary not supported by Anthropic (raises RuntimeError)
    Case('anthropic-image-binary', 'anthropic', 'image', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_binary),
    Case('anthropic-image-url', 'anthropic', 'image', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_url),
    Case('anthropic-document-binary', 'anthropic', 'document', 'binary', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_binary),
    Case('anthropic-document-url', 'anthropic', 'document', 'url', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_url),

    # === Bedrock: Images native, documents/video via user_msg, audio dropped ===
    Case('bedrock-image-binary', 'bedrock', 'image', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}, {'type':'response','parts':['TextPart']}]), _image_binary),
    Case('bedrock-image-url', 'bedrock', 'image', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_url),
    Case('bedrock-document-binary', 'bedrock', 'document', 'binary', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_binary),
    Case('bedrock-document-url', 'bedrock', 'document', 'url', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_url),
    Case('bedrock-audio-binary', 'bedrock', 'audio', 'binary', 'dropped', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}, {'type':'response','parts':['TextPart']}]), _audio_binary),
    Case('bedrock-audio-url', 'bedrock', 'audio', 'url', 'dropped', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}, {'type':'response','parts':['TextPart']}]), _audio_url),
    Case('bedrock-video-binary', 'bedrock', 'video', 'binary', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _video_binary),
    Case('bedrock-video-url', 'bedrock', 'video', 'url', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _video_url),

    # === Google: All native ===
    Case('google-image-binary', 'google', 'image', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_binary),
    Case('google-image-url', 'google', 'image', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_url),
    Case('google-document-binary', 'google', 'document', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_binary),
    Case('google-document-url', 'google', 'document', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_url),
    Case('google-audio-binary', 'google', 'audio', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart','TextPart','TextPart']}]), _audio_binary),
    Case('google-audio-url', 'google', 'audio', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['RetryPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}, {'type':'response','parts':['TextPart']}]), _audio_url),
    Case('google-video-binary', 'google', 'video', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _video_binary),
    Case('google-video-url', 'google', 'video', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['RetryPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _video_url),

    # === OpenAI Chat: Images/docs via user_msg ===
    # Note: video raises NotImplementedError, audio not supported by gpt-5-mini - tested in unit tests
    Case('openai_chat-image-binary', 'openai_chat', 'image', 'binary', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_binary),
    Case('openai_chat-image-url', 'openai_chat', 'image', 'url', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _image_url),
    Case('openai_chat-document-binary', 'openai_chat', 'document', 'binary', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_binary),
    Case('openai_chat-document-url', 'openai_chat', 'document', 'url', 'user_msg', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), _document_url),

    # === OpenAI Responses: Images/docs native ===
    # Note: audio-binary/video raise NotImplementedError, audio-url not supported by gpt-5-mini - tested in unit tests
    Case('openai_responses-image-binary', 'openai_responses', 'image', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), _image_binary),
    Case('openai_responses-image-url', 'openai_responses', 'image', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), _image_url),
    Case('openai_responses-document-binary', 'openai_responses', 'document', 'binary', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), _document_binary),
    Case('openai_responses-document-url', 'openai_responses', 'document', 'url', 'native', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), _document_url),
]
# fmt: on


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_keys(
    openai_api_key: str,
    anthropic_api_key: str,
    gemini_api_key: str,
) -> dict[str, str]:
    """Collect all API keys into a dict."""
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'google': gemini_api_key,
    }


@pytest.fixture(scope='module')
def vcr_config():
    """Override vcr_config to enable cassette injection for dropped behavior verification."""
    return {
        'ignore_localhost': True,
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
        'inject_cassette': True,
    }


# =============================================================================
# Main parametrized test
# =============================================================================


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
async def test_multimodal_tool_return(
    case: Case,
    api_keys: dict[str, str],
    bedrock_provider: Any,
    assets_path: Path,
    allow_model_requests: None,
    vcr: Any,
):
    """Test multimodal tool return handling across providers.

    This test verifies:
    1. Tools can return multimodal content (images, documents, audio, video)
    2. Provider-specific handling is correct (native vs user_msg vs dropped)
    3. The message structure matches expected patterns
    """
    skip_reason = should_skip_case(case)
    if skip_reason:
        pytest.skip(skip_reason)

    model = get_model(case.provider, api_keys, bedrock_provider)
    content = case.content_factory(assets_path)

    agent: Agent[None, str] = Agent(model)

    @agent.tool_plain
    def get_file() -> Any:
        """Return a file for the model to analyze."""
        return content

    result = await agent.run(
        f'Call the get_file tool to get a {case.file_type} and describe what you see.',
        usage_limits=UsageLimits(output_tokens_limit=100000),
    )

    # Snapshot the message structure - this documents the behavior:
    # - native: file in ToolReturnPart only
    # - user_msg: file in ToolReturnPart + separate UserPromptPart
    # - dropped: no file content sent (but message structure is same as native)
    message_structure = get_message_structure(result.all_messages())
    assert message_structure == case.expected_message_structure

    # For dropped behavior, verify via cassette that no audio content blocks were sent
    # We check for audio content keys like "audio/mpeg" (media type) or '"audio":' (Bedrock format)
    if case.expected_behavior == 'dropped':
        for req in vcr.requests:
            body_raw = req.body.decode() if isinstance(req.body, bytes) else req.body
            body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
            body_str = json.dumps(body).lower()
            # Check for audio content block markers, not just the word "audio" in text
            assert '"audio":' not in body_str and 'audio/mpeg' not in body_str, (
                f'Expected no audio content in request, but found it: {body_str[:500]}...'
            )
