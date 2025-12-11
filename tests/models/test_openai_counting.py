from __future__ import annotations as _annotations

from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import IsNow, IsStr, try_import

with try_import() as imports_successful:
    from openai.types import chat

    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIResponsesModel,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

    MockChatCompletion = chat.ChatCompletion | Exception
    MockChatCompletionChunk = chat.ChatCompletionChunk | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.mark.vcr()
@pytest.mark.parametrize(
    'model_name,expected_token_count',
    [
        ('gpt-3.5-turbo', 115),
        ('gpt-4-0613', 115),
        ('gpt-4', 115),
        ('gpt-4o', 110),
        ('gpt-4o-mini', 110),
        ('gpt-5', 109),
    ],
)
async def test_count_tokens(
    model_name: str,
    expected_token_count: int,
):
    """Test token counting with OpenAI Chat and Response models."""
    test_messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='You are a helpful, pattern-following assistant that translates corporate jargon into plain English.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                SystemPromptPart(
                    content='New synergies will help drive top-line growth.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                SystemPromptPart(
                    content='Things working well together will increase revenue.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                SystemPromptPart(
                    content="Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
                    timestamp=IsNow(tz=timezone.utc),
                ),
                SystemPromptPart(
                    content="Let's talk later when we're less busy about how to do better.",
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content="This late pivot means we don't have time to boil the ocean for the client deliverable.",
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            run_id=IsStr(),
        )
    ]

    chat_model = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key='foobar'))
    usage_result: RequestUsage = await chat_model.count_tokens(test_messages, {}, ModelRequestParameters())
    assert usage_result.input_tokens == expected_token_count

    responses_model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key='foobar'))
    usage_result: RequestUsage = await responses_model.count_tokens(test_messages, {}, ModelRequestParameters())
    assert usage_result.input_tokens == expected_token_count


@pytest.mark.vcr()
async def test_count_tokens_with_non_string_values():
    """Test token counting with messages that have non-string values (like content arrays)."""
    test_messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Text content',
                        ImageUrl(url='https://example.com/image.jpg'),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            run_id=IsStr(),
        )
    ]

    chat_model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    usage_result: RequestUsage = await chat_model.count_tokens(test_messages, {}, ModelRequestParameters())

    assert usage_result.input_tokens > 0


@pytest.mark.vcr()
async def test_openai_model_usage_limit_not_exceeded(
    allow_model_requests: None,
    openai_api_key: str,
):
    provider = OpenAIProvider(api_key=openai_api_key)
    model = OpenAIResponsesModel('gpt-4', provider=provider)
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=25, count_tokens_before_request=True),
    )
    assert result.output == snapshot(
        'The sentence you provided is commonly used as a pangram, which is a phrase that uses every letter of the alphabet at least once. This sentence is often used for typing practice due to its use of every letter.'
    )


@pytest.mark.vcr()
async def test_openai_model_usage_limit_exceeded(
    allow_model_requests: None,
    openai_api_key: str,
):
    provider = OpenAIProvider(api_key=openai_api_key)
    model = OpenAIResponsesModel('gpt-4', provider=provider)
    agent = Agent(model=model)

    with pytest.raises(UsageLimitExceeded) as exc_info:
        _ = await agent.run(
            'The quick brown fox jumps over the lazydog. The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=25, count_tokens_before_request=True),
        )

    assert 'exceed the input_tokens_limit of' in str(exc_info.value)


@pytest.mark.vcr()
async def test_openai_model_usage_unsupported_model(
    allow_model_requests: None,
    openai_api_key: str,
):
    """Test token counting with messages that have non-string values (like content arrays)."""
    test_messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Text content',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            run_id=IsStr(),
        )
    ]

    chat_model = OpenAIChatModel('not-supported-model', provider=OpenAIProvider(api_key='foobar'))
    usage_result: RequestUsage = await chat_model.count_tokens(test_messages, {}, ModelRequestParameters())

    assert usage_result.input_tokens > 0
