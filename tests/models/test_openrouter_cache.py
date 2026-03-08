"""Tests for OpenRouter CachePoint and prompt caching support.

Tests are added to this separate file to keep the main test_openrouter.py manageable,
but they test the same OpenRouterModel class.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent, CachePoint, ModelRequest, ToolDefinition, UserPromptPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ImageUrl, ModelMessage, SystemPromptPart
from pydantic_ai.models import ModelRequestParameters

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


# ===== CachePoint in user messages =====


async def test_openrouter_cache_point_adds_cache_control() -> None:
    """Test that CachePoint adds cache_control to the preceding content part for Anthropic models."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['Some context to cache', CachePoint(), 'Now the question'])])
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Some context to cache', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Now the question'},
        ]
    )


async def test_openrouter_cache_point_multiple_markers() -> None:
    """Test multiple CachePoint markers in a single prompt."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=['First chunk', CachePoint(), 'Second chunk', CachePoint(), 'Question'],
                )
            ]
        )
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'First chunk', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Second chunk', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Question'},
        ]
    )


async def test_openrouter_cache_point_with_custom_ttl() -> None:
    """Test CachePoint with custom TTL='1h' for Anthropic model."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(ttl='1h'), 'Question'])])
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content[0] == snapshot(
        {'type': 'text', 'text': 'Context', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}
    )


async def test_openrouter_cache_point_gemini_omits_ttl() -> None:
    """Test that CachePoint omits TTL for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(), 'Question'])])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content[0] == snapshot({'type': 'text', 'text': 'Context', 'cache_control': {'type': 'ephemeral'}})


async def test_openrouter_cache_point_first_content_raises_error(allow_model_requests: None) -> None:
    """Test that CachePoint as first content raises UserError."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match='CachePoint cannot be the first content in a user message - there must be previous content',
    ):
        await agent.run([CachePoint(), 'This should fail'])


async def test_openrouter_cache_point_unsupported_provider_ignored() -> None:
    """Test that CachePoint is silently ignored for unsupported downstream providers."""
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Context', CachePoint(), 'Question'])])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    # No cache_control should be present — CachePoint silently skipped
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Context'},
            {'type': 'text', 'text': 'Question'},
        ]
    )


async def test_openrouter_cache_point_string_content_unchanged() -> None:
    """Test that plain string content is not converted to list format by CachePoint logic."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Just a plain string')])]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    # String content should remain a string, not converted to list
    assert content == 'Just a plain string'


async def test_openrouter_cache_point_with_image_content() -> None:
    """Test CachePoint attaches cache_control to a preceding image content part."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/image.jpg'),
                        CachePoint(),
                        'What is in this image?',
                    ]
                )
            ]
        )
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    content = mapped[0].get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {
                'type': 'image_url',
                'image_url': {'url': 'https://example.com/image.jpg'},
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {'type': 'text', 'text': 'What is in this image?'},
        ]
    )


async def test_openrouter_cache_messages_no_duplicate_with_explicit_cache_point() -> None:
    """Test that cache_messages doesn't conflict with an explicit CachePoint on different parts."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=['Long context to cache', CachePoint(), 'Now the question'],
                )
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )
    content = mapped[-1].get('content')
    assert isinstance(content, list)
    # CachePoint tags 'Long context to cache', cache_messages tags 'Now the question'
    assert content == snapshot(
        [
            {'type': 'text', 'text': 'Long context to cache', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'type': 'text', 'text': 'Now the question', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
        ]
    )


# ===== openrouter_cache_instructions =====


async def test_openrouter_cache_instructions_adds_cache_control() -> None:
    """Test that openrouter_cache_instructions adds cache_control to system prompt."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Find the system message
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    assert isinstance(content, list)
    assert content == snapshot(
        [
            {
                'type': 'text',
                'text': 'You are a helpful assistant.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )


async def test_openrouter_cache_instructions_custom_ttl() -> None:
    """Test openrouter_cache_instructions with custom TTL='1h'."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions='1h')

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    assert isinstance(content, list)
    assert cast(dict[str, Any], content[0])['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})


async def test_openrouter_cache_instructions_gemini_omits_ttl() -> None:
    """Test that openrouter_cache_instructions omits TTL for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    assert isinstance(content, list)
    assert cast(dict[str, Any], content[0])['cache_control'] == snapshot({'type': 'ephemeral'})


async def test_openrouter_cache_instructions_unsupported_provider_ignored() -> None:
    """Test that openrouter_cache_instructions is silently ignored for unsupported providers."""
    model = OpenRouterModel('openai/gpt-5', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    content = system_msg.get('content')
    # Should remain a plain string, not converted to list with cache_control
    assert isinstance(content, str)


# ===== openrouter_cache_messages =====


async def test_openrouter_cache_messages_adds_cache_control() -> None:
    """Test that openrouter_cache_messages adds cache_control to the last message."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt.'),
                UserPromptPart(content='User message'),
            ]
        )
    ]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Last message should have cache_control
    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert content[-1] == snapshot(
        {'type': 'text', 'text': 'User message', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
    )

    # System message should NOT have cache_control
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    system_content = system_msg.get('content')
    assert isinstance(system_content, str)


async def test_openrouter_cache_messages_custom_ttl() -> None:
    """Test openrouter_cache_messages with custom TTL='1h'."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages='1h')

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='User message')])]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert cast(dict[str, Any], content[-1])['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})


async def test_openrouter_cache_messages_empty_content_no_crash() -> None:
    """Test that openrouter_cache_messages does not crash when last message has empty content list."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[])])]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    # Should not crash — empty content is left as-is
    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert content == []


# ===== openrouter_cache_tool_definitions =====


async def test_openrouter_cache_tool_definitions_anthropic() -> None:
    """Test that openrouter_cache_tool_definitions adds cache_control to the last tool for Anthropic."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_tool_definitions=True)

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='tool_one', description='First tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
            ToolDefinition(
                name='tool_two', description='Second tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    assert len(tools) == 2
    # First tool should NOT have cache_control
    first_tool = cast(dict[str, Any], tools[0])
    assert 'cache_control' not in first_tool
    # Last tool SHOULD have cache_control
    last_tool = cast(dict[str, Any], tools[1])
    assert last_tool['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '5m'})


async def test_openrouter_cache_tool_definitions_custom_ttl() -> None:
    """Test openrouter_cache_tool_definitions with custom TTL='1h'."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_tool_definitions='1h')

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    last_tool = cast(dict[str, Any], tools[0])
    assert last_tool['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})


async def test_openrouter_cache_tool_definitions_gemini_ignored() -> None:
    """Test that openrouter_cache_tool_definitions has no effect for Gemini models."""
    model = OpenRouterModel('google/gemini-2.5-flash', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(openrouter_cache_tool_definitions=True)

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    # Gemini should NOT get cache_control on tools
    last_tool = cast(dict[str, Any], tools[0])
    assert 'cache_control' not in last_tool


# ===== Combined settings =====


async def test_openrouter_cache_all_settings_combined() -> None:
    """Test that all cache settings work together without interfering."""
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=OpenRouterProvider(api_key='test-key'))
    settings = OpenRouterModelSettings(
        openrouter_cache_instructions=True,
        openrouter_cache_messages=True,
        openrouter_cache_tool_definitions=True,
    )

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(content='User message'),
            ]
        )
    ]

    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='my_tool', description='A tool', parameters_json_schema={'type': 'object', 'properties': {}}
            ),
        ],
        allow_text_output=True,
    )

    # Check tools
    tools = model._get_tools(params, model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    last_tool = cast(dict[str, Any], tools[0])
    assert last_tool['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}

    # Check messages
    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, params, model_settings=settings
    )

    # System message should have cache_control
    system_msg = next(m for m in mapped if m.get('role') in ('system', 'developer'))
    system_content = system_msg.get('content')
    assert isinstance(system_content, list)
    assert 'cache_control' in system_content[0]

    # Last message should have cache_control
    last_msg = mapped[-1]
    last_content = last_msg.get('content')
    assert isinstance(last_content, list)
    assert 'cache_control' in last_content[-1]


# ===== E2E tests with cassettes =====


async def test_openrouter_cache_point_anthropic_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test CachePoint with Anthropic model via OpenRouter using real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify cache_control was in the request
    assert vcr is not None
    assert len(vcr.requests) >= 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    user_content = request_body['messages'][0]['content']
    # The first content part should have cache_control
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_point_gemini_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test CachePoint with Gemini model via OpenRouter using real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify cache_control was in the request (without TTL for Gemini)
    assert vcr is not None
    assert len(vcr.requests) >= 1  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    user_content = request_body['messages'][0]['content']
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral'}


async def test_openrouter_cache_instructions_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_instructions with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify cache_control was added to system message
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    content = system_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_messages_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_messages with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='Be helpful.',
        model_settings=OpenRouterModelSettings(openrouter_cache_messages=True),
    )

    result = await agent.run('Say hello in one word.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify cache_control was added to the last message
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    last_msg = request_body['messages'][-1]
    content = last_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_tool_definitions_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test openrouter_cache_tool_definitions with Anthropic model via real API."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)

    agent = Agent(
        model,
        model_settings=OpenRouterModelSettings(openrouter_cache_tool_definitions=True),
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Sunny in {city}'

    result = await agent.run('What tools do you have available? Just list them briefly.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify cache_control was added to the last tool
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    tools = request_body.get('tools', [])
    assert len(tools) >= 1
    assert 'cache_control' in tools[-1]
    assert tools[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_messages_anthropic_real_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:
    """Test that openrouter_cache_messages produces cache read metrics for Anthropic via OpenRouter.

    Forces routing to the Anthropic provider directly (not Bedrock) to ensure cache locality.
    The first call populates the cache, and the second call reads from it.

    Note: cache_write_tokens is not asserted because OpenRouter reports it via a non-standard
    prompt_tokens_details.cache_write_tokens field that genai-prices does not currently map.
    cache_read_tokens works via the standard prompt_tokens_details.cached_tokens field.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant.',
        model_settings=OpenRouterModelSettings(
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    # Must exceed 2048 tokens for Claude Sonnet caching
    result1 = await agent.run(
        'Analyze the architectural patterns used in distributed database systems and their tradeoffs. ' * 200
    )
    usage1 = result1.usage()

    assert usage1.requests == 1
    assert usage1.input_tokens > 2000
    assert usage1.output_tokens > 0

    # Second call continues the conversation — the previous cached message is still in the request
    result2 = await agent.run('Can you summarize that in one sentence?', message_history=result1.all_messages())
    usage2 = result2.usage()

    # cache_read_tokens > 0 proves caching actually worked end-to-end
    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_instructions_gemini_real_api(
    allow_model_requests: None, openrouter_api_key: str
) -> None:
    """Test that openrouter_cache_instructions produces cache read metrics for Gemini via OpenRouter.

    Uses cache_instructions with a long system prompt so the cached content is identical across
    both calls. Forces routing to Google AI Studio to ensure cache locality.
    Gemini 2.5 Flash requires 1024 tokens minimum for caching.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('google/gemini-2.5-flash', provider=provider)

    long_instructions = (
        'You are a specialized assistant that helps with distributed systems design. '
        'You have deep expertise in consensus protocols, CAP theorem, and eventual consistency. '
    ) * 80  # ~1200 tokens, well above 1024 minimum

    agent = Agent(
        model,
        instructions=long_instructions,
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_provider={'order': ['google-ai-studio'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    # First call populates the cache with the long system instructions
    result1 = await agent.run('What is the CAP theorem?')
    usage1 = result1.usage()

    assert usage1.requests == 1
    assert usage1.input_tokens > 1000
    assert usage1.output_tokens > 0

    # Second call reuses the same system instructions — should hit cache
    result2 = await agent.run('What is eventual consistency?')
    usage2 = result2.usage()

    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_streaming_e2e(
    allow_model_requests: None, openrouter_api_key: str, vcr: Cassette
) -> None:
    """Test that cache_control is correctly included in requests when using streaming."""
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_messages=True,
        ),
    )

    async with agent.run_stream('Say hello in one word.') as stream:
        result = await stream.get_output()

    assert isinstance(result, str)
    assert len(result) > 0

    # Verify cache_control was included in the request
    assert vcr is not None
    request_body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

    # System message should have cache_control from cache_instructions
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    system_content = system_msg['content']
    assert isinstance(system_content, list)
    assert 'cache_control' in system_content[-1]

    # Last user message should have cache_control from cache_messages
    last_msg = request_body['messages'][-1]
    last_content = last_msg['content']
    assert isinstance(last_content, list)
    assert 'cache_control' in last_content[-1]


async def test_openrouter_cache_all_settings_real_api(allow_model_requests: None, openrouter_api_key: str) -> None:
    """Test all cache settings combined with actual cache write+read metrics.

    Enables cache_instructions, cache_tool_definitions, and cache_messages together,
    forces Anthropic routing for cache locality, and verifies cache_read_tokens on
    the second call.
    """
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenRouterModel('anthropic/claude-sonnet-4.6', provider=provider)

    agent = Agent(
        model,
        instructions=(
            'You are an assistant that specializes in mathematics and calculations. '
            'Always show your work step by step. '
        )
        * 100,  # ~2500 tokens, above 2048 minimum
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_tool_definitions=True,
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    @agent.tool_plain
    def calculator(expression: str) -> str:  # pragma: no cover
        """Evaluate a math expression."""
        return 'result'

    result1 = await agent.run('What is 123 * 456?')
    usage1 = result1.usage()

    assert usage1.requests >= 1
    assert usage1.input_tokens > 2000
    assert usage1.output_tokens > 0

    # Second call with same agent — system instructions + tools should be cached
    result2 = await agent.run('What is 789 + 321?')
    usage2 = result2.usage()

    assert usage2.requests >= 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0
