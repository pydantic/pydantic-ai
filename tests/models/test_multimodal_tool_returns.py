from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pytest

from pydantic_ai import Agent, BinaryContent, UserError
from pydantic_ai.messages import ModelRequest, MultiModalContent, ToolReturnPart, UserPromptPart

from ..conftest import try_import

if TYPE_CHECKING:
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='required packages not installed'),
    pytest.mark.vcr(),
]

Expectation = Literal['native', 'fallback', 'error']
FileType = Literal['image', 'document', 'audio', 'video']
ProviderName = Literal['anthropic', 'bedrock', 'google', 'openai-chat', 'openai-responses', 'xai', 'groq']

SUPPORT_MATRIX: dict[tuple[ProviderName, FileType], Expectation] = {
    ('anthropic', 'image'): 'native',
    ('anthropic', 'document'): 'native',
    ('anthropic', 'audio'): 'error',
    ('anthropic', 'video'): 'error',
    ('bedrock', 'image'): 'error',
    ('bedrock', 'document'): 'error',
    ('bedrock', 'audio'): 'error',
    ('bedrock', 'video'): 'error',
    ('google', 'image'): 'native',
    ('google', 'document'): 'native',
    ('google', 'audio'): 'native',
    ('google', 'video'): 'native',
    ('openai-chat', 'image'): 'fallback',
    ('openai-chat', 'document'): 'fallback',
    ('openai-chat', 'audio'): 'error',
    ('openai-chat', 'video'): 'error',
    ('openai-responses', 'image'): 'native',
    ('openai-responses', 'document'): 'native',
    ('openai-responses', 'audio'): 'error',
    ('openai-responses', 'video'): 'error',
    ('xai', 'image'): 'fallback',
    ('xai', 'document'): 'error',
    ('xai', 'audio'): 'error',
    ('xai', 'video'): 'error',
    ('groq', 'image'): 'fallback',
    ('groq', 'document'): 'error',
    ('groq', 'audio'): 'error',
    ('groq', 'video'): 'error',
}


def get_expectation(provider: ProviderName, file_type: FileType) -> Expectation:
    return SUPPORT_MATRIX[(provider, file_type)]


@dataclass
class Case:
    provider: ProviderName
    file_type: FileType
    model_name: str = ''
    snapshot: Any = None
    error_match: str | None = None
    media_type_override: str | None = None

    @property
    def expectation(self) -> Expectation:
        if self.error_match:
            return 'error'
        return get_expectation(self.provider, self.file_type)

    @property
    def id(self) -> str:
        base = f'{self.provider}-{self.file_type}'
        if self.media_type_override:
            fmt = self.media_type_override.split('/')[-1]
            return f'{base}-{fmt}'
        return base


def assert_file_in_tool_return(messages: list[Any], file_identifier: str) -> None:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    files = part.files
                    if any(f.identifier == file_identifier for f in files):
                        return
    raise AssertionError(f'File {file_identifier} not found in any ToolReturnPart')


def assert_no_separate_user_file(messages: list[Any]) -> None:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if isinstance(part.content, list):
                        for item in part.content:
                            if isinstance(item, MultiModalContent):
                                raise AssertionError(f'Found unexpected multimodal content in UserPromptPart: {item}')


def assert_multimodal_result(
    messages: list[Any],
    expectation: Expectation,
    file_identifier: str,
) -> None:
    # TODO: Add provider-level API request verification to check that:
    # - 'native' providers put the file directly in the tool_result API field
    # - 'fallback' providers use "See file {id}" placeholder + separate user message in API request
    # Currently we can only verify the pydantic-ai message history, where both look identical.
    if expectation == 'error':
        return
    elif expectation in ('native', 'fallback'):
        assert_file_in_tool_return(messages, file_identifier)
        assert_no_separate_user_file(messages)


def create_model(
    case: Case,
    openai_api_key: str,
    anthropic_api_key: str,
    groq_api_key: str,
    gemini_api_key: str,
    bedrock_provider: BedrockProvider,
    xai_provider: XaiProvider | None,
) -> Any:
    if case.provider == 'anthropic':
        return AnthropicModel(
            case.model_name or 'claude-sonnet-4-5',
            provider=AnthropicProvider(api_key=anthropic_api_key),
        )
    elif case.provider == 'bedrock':
        return BedrockConverseModel(
            case.model_name or 'us.amazon.nova-lite-v1:0',
            provider=bedrock_provider,
        )
    elif case.provider == 'google':
        if gemini_api_key and gemini_api_key != 'mock-api-key':
            provider = GoogleProvider(api_key=gemini_api_key)
        else:
            provider = GoogleProvider(vertexai=True)
        return GoogleModel(case.model_name or 'gemini-2.0-flash', provider=provider)
    elif case.provider == 'openai-chat':
        return OpenAIChatModel(
            case.model_name or 'gpt-4o',
            provider=OpenAIProvider(api_key=openai_api_key),
        )
    elif case.provider == 'openai-responses':
        return OpenAIResponsesModel(
            case.model_name or 'gpt-4o',
            provider=OpenAIProvider(api_key=openai_api_key),
        )
    elif case.provider == 'xai':
        assert xai_provider is not None
        return XaiModel(
            case.model_name or 'grok-4-fast-non-reasoning',
            provider=xai_provider,
        )
    elif case.provider == 'groq':
        return GroqModel(
            case.model_name or 'meta-llama/llama-4-scout-17b-16e-instruct',
            provider=GroqProvider(api_key=groq_api_key),
        )
    else:
        raise ValueError(f'Unknown provider: {case.provider}')


def create_file_content(
    file_type: FileType,
    image_content: BinaryContent,
    document_content: BinaryContent,
    audio_content: BinaryContent,
    video_content: BinaryContent,
    media_type_override: str | None = None,
) -> MultiModalContent:
    if file_type == 'image':
        base = image_content
    elif file_type == 'document':
        base = document_content
    elif file_type == 'audio':
        base = audio_content
    elif file_type == 'video':
        base = video_content
    else:
        raise ValueError(f'Unknown file type: {file_type}')

    if media_type_override:
        return BinaryContent(data=base.data, media_type=media_type_override)
    return base


CASES: list[Case] = [
    Case(provider='anthropic', file_type='image'),
    Case(provider='anthropic', file_type='document'),
    Case(provider='anthropic', file_type='audio', error_match='does not support audio'),
    Case(provider='anthropic', file_type='video', error_match='does not support video'),
    Case(provider='bedrock', file_type='image', error_match='does not support multimodal content'),
    Case(provider='bedrock', file_type='document', error_match='does not support multimodal content'),
    Case(provider='bedrock', file_type='audio', error_match='does not support multimodal content'),
    Case(provider='bedrock', file_type='video', error_match='does not support multimodal content'),
    Case(provider='google', file_type='image'),
    Case(provider='google', file_type='document'),
    Case(provider='google', file_type='audio'),
    Case(provider='google', file_type='video'),
    Case(provider='openai-chat', file_type='image'),
    Case(provider='openai-chat', file_type='document'),
    Case(provider='openai-chat', file_type='audio', error_match='does not support audio'),
    Case(provider='openai-chat', file_type='video', error_match='does not support video'),
    Case(provider='openai-responses', file_type='image'),
    Case(provider='openai-responses', file_type='document'),
    Case(provider='openai-responses', file_type='audio', error_match='does not support audio'),
    Case(provider='openai-responses', file_type='video', error_match='does not support video'),
    Case(provider='xai', file_type='image'),
    Case(provider='xai', file_type='document', error_match='does not support documents'),
    Case(provider='xai', file_type='audio', error_match='does not support audio'),
    Case(provider='xai', file_type='video', error_match='does not support video'),
    Case(provider='groq', file_type='image'),
    Case(provider='groq', file_type='document', error_match='does not support documents'),
    Case(provider='groq', file_type='audio', error_match='does not support audio'),
    Case(provider='groq', file_type='video', error_match='does not support video'),
]


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
async def test_multimodal_tool_return(
    case: Case,
    request: pytest.FixtureRequest,
    allow_model_requests: None,
    openai_api_key: str,
    anthropic_api_key: str,
    groq_api_key: str,
    gemini_api_key: str,
    bedrock_provider: BedrockProvider,
    image_content: BinaryContent,
    document_content: BinaryContent,
    audio_content: BinaryContent,
    video_content: BinaryContent,
):
    # xAI uses gRPC (not HTTP), so VCR cassettes don't work. We use xai_provider fixture
    # (proto cassettes) only for xAI tests. Using getfixturevalue avoids fixture setup for other providers.
    xai_provider: XaiProvider | None = request.getfixturevalue('xai_provider') if case.provider == 'xai' else None
    model = create_model(
        case,
        openai_api_key,
        anthropic_api_key,
        groq_api_key,
        gemini_api_key,
        bedrock_provider,
        xai_provider,
    )
    agent: Agent[None, str] = Agent(
        model,
        system_prompt='You MUST use available tools to complete tasks. Never respond without using a tool first.',
    )
    file_content = create_file_content(
        case.file_type,
        image_content,
        document_content,
        audio_content,
        video_content,
        case.media_type_override,
    )
    file_identifier = file_content.identifier

    @agent.tool_plain
    async def get_file() -> MultiModalContent:
        return file_content

    prompt = f'Call the get_file tool to retrieve a {case.file_type}, then describe what you received.'

    if case.expectation == 'error':
        with pytest.raises(UserError, match=case.error_match):
            await agent.run(prompt)
    else:
        result = await agent.run(prompt)
        messages = result.all_messages()
        assert_multimodal_result(messages, case.expectation, file_identifier)
        if case.snapshot is not None:
            assert messages == case.snapshot
