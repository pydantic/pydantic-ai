"""Tests for OpenRouter prompt caching.

Covers the `CachePoint` pre-request guards, the `openrouter_cache_*` settings, and the
`cache_control` breakpoints they emit on the wire (public-API VCR tests against the real
OpenRouter API).
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING

import pytest
from vcr.cassette import Cassette

from pydantic_ai import (
    Agent,
    BinaryImage,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...cassette_utils import single_request_body
from ...conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider

if TYPE_CHECKING:
    from ..conftest import OpenRouterModelFactory

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


# ===== Prompt caching: pre-request guards (public API, no request) =====


async def test_openrouter_cache_point_first_content_raises_error(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory
) -> None:
    """A `CachePoint` with no preceding content raises `UserError` before any request."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model)

    with pytest.raises(
        UserError,
        match='CachePoint cannot be the first content in a user message - there must be previous content',
    ):
        await agent.run([CachePoint(), 'This should fail'])


async def test_openrouter_cache_points_exceed_limit_raises(allow_model_requests: None) -> None:
    """Exceeding the downstream provider's cache-breakpoint budget raises `UserError` before any request.

    Only reachable via a custom profile with a low `openrouter_max_cache_points`: built-in
    profiles allow 4, and normal settings can't produce more than one system + one tool
    breakpoint. Pins the pre-request budget guard through the public API (the error is raised
    while mapping messages, so no HTTP request is made).
    """
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        # custom `profile=` is not forwarded by the openrouter_model factory
        # ast-grep-ignore: prefer-model-factory
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(
            openrouter_supports_cache_control=True,
            openrouter_supports_cache_ttl=True,
            openrouter_supports_tool_cache=True,
            openrouter_max_cache_points=1,
        ),
    )
    agent = Agent(
        model,
        instructions='Be helpful.',
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_tool_definitions=True,
        ),
    )

    @agent.tool_plain
    def my_tool() -> str:  # pragma: no cover
        return 'ok'

    with pytest.raises(UserError, match='Too many cache points for downstream provider'):
        await agent.run('Hello')


# ===== Prompt caching: defensive-branch unit tests =====
# These two pin branches that no real OpenRouter model+provider combination reaches; they're
# kept as unit tests because routing them through a real request would either be impossible
# (the config can't arise) or contrived. They deliberately call the private `_map_messages` to
# exercise those branches directly. See the PR thread for the reachability analysis.


async def test_openrouter_cache_instructions_ignores_user_role_profile() -> None:
    """Instruction caching is skipped when the profile maps instructions to user messages.

    Unreachable via any built-in model: the only profile that sets `openai_system_prompt_role='user'`
    is `o1-mini` (OpenAI), for which `openrouter_supports_cache_control` is `False`, so the
    instruction-caching code never runs. Only reachable via a user-supplied custom profile.
    """
    model = OpenRouterModel(
        'anthropic/claude-sonnet-4.6',
        # custom `profile=` is not forwarded by the openrouter_model factory
        # ast-grep-ignore: prefer-model-factory
        provider=OpenRouterProvider(api_key='test-key'),
        profile=OpenRouterModelProfile(openai_system_prompt_role='user', openrouter_supports_cache_control=True),
    )
    settings = OpenRouterModelSettings(openrouter_cache_instructions=True)
    params = ModelRequestParameters(instruction_parts=[InstructionPart(content='Static instructions.', dynamic=False)])

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[UserPromptPart(content='Hello')])], params, model_settings=settings
    )

    for message in mapped:
        content = message.get('content')
        assert isinstance(content, str)
        assert 'cache_control' not in content


async def test_openrouter_cache_messages_empty_content_no_crash(openrouter_model: OpenRouterModelFactory) -> None:
    """`openrouter_cache_messages` is a no-op (not a crash) when the last message has empty content.

    The empty-content list is only reachable via a degenerate `agent.run([])`; pins the guard that
    prevents an `IndexError` on `content[-1]`.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    settings = OpenRouterModelSettings(openrouter_cache_messages=True)

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[])])]

    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, ModelRequestParameters(), model_settings=settings
    )

    last_msg = mapped[-1]
    content = last_msg.get('content')
    assert isinstance(content, list)
    assert content == []


# ===== Prompt caching: public-API wire-shape tests (cassettes) =====
# Each runs through `Agent.run()` against the real OpenRouter API and asserts the `cache_control`
# breakpoints on the recorded request body. These replace the former private-method unit tests
# that called `_map_messages` / `_get_tool_choice` directly.


async def test_openrouter_cache_point_multiple_markers_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Multiple `CachePoint`s (including a custom TTL) each tag their preceding block for Anthropic.

    The longer TTL must precede the shorter one: Anthropic rejects a `1h` breakpoint that comes
    after a `5m` breakpoint within the messages group.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model)

    result = await agent.run(
        [
            'First chunk of context to cache. ' * 20,
            CachePoint(ttl='1h'),
            'Second chunk of context to cache. ' * 20,
            CachePoint(),
            'Summarize in one sentence.',
        ]
    )

    assert isinstance(result.output, str)
    content = single_request_body(vcr)['messages'][0]['content']
    assert content[0]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}
    assert content[1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
    assert 'cache_control' not in content[2]


async def test_openrouter_cache_point_image_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, image_content: BinaryImage, vcr: Cassette
) -> None:
    """`CachePoint` attaches `cache_control` to a preceding image content part for Anthropic."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model)

    result = await agent.run([image_content, CachePoint(), 'What is in this image? Answer in one word.'])

    assert isinstance(result.output, str)
    content = single_request_body(vcr)['messages'][0]['content']
    assert content[0]['type'] == 'image_url'
    assert content[0]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
    assert 'cache_control' not in content[1]


async def test_openrouter_cache_point_unsupported_provider_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """`CachePoint` is silently dropped for downstream providers without cache support (OpenAI)."""
    model = openrouter_model('openai/gpt-5-mini')
    agent = Agent(model)

    result = await agent.run(['Some context. ' * 20, CachePoint(), 'Summarize in one sentence.'])

    assert isinstance(result.output, str)
    content = single_request_body(vcr)['messages'][0]['content']
    assert all('cache_control' not in part for part in content)


async def test_openrouter_cache_instructions_gemini_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """`openrouter_cache_instructions` caches the system prompt for Gemini, omitting TTL."""
    model = openrouter_model('google/gemini-2.5-flash')
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    system_msg = next(m for m in single_request_body(vcr)['messages'] if m['role'] in ('system', 'developer'))
    assert system_msg['content'][-1]['cache_control'] == {'type': 'ephemeral'}


async def test_openrouter_cache_instructions_static_dynamic_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """For Anthropic, the instruction cache point lands on the last static block, leaving dynamic instructions uncached."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'The current focus is distributed systems.'

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    system_messages = [m for m in single_request_body(vcr)['messages'] if m['role'] in ('system', 'developer')]
    # Last static instruction block carries the cache breakpoint; the dynamic tail does not.
    assert any(
        isinstance(m['content'], list) and any('cache_control' in part for part in m['content'])
        for m in system_messages
    )
    assert system_messages[-1]['content'] == 'The current focus is distributed systems.'


async def test_openrouter_cache_instructions_gemini_skips_dynamic_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """For Gemini, instruction caching is skipped entirely when dynamic instructions are present."""
    model = openrouter_model('google/gemini-2.5-flash')
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'The current focus is distributed systems.'

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    system_messages = [m for m in single_request_body(vcr)['messages'] if m['role'] in ('system', 'developer')]
    for m in system_messages:
        content = m['content']
        assert isinstance(content, str) or all('cache_control' not in part for part in content)


async def test_openrouter_cache_instructions_system_prompt_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """`openrouter_cache_instructions` caches the system prompt when it comes via `system_prompt=`.

    With no `instructions=`, there are no structured instruction parts, so the cache point falls back
    to the last `system`/`developer` message, which here is the `system_prompt`-derived block.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        system_prompt='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    system_msg = next(m for m in single_request_body(vcr)['messages'] if m['role'] in ('system', 'developer'))
    assert system_msg['content'][-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_instructions_system_prompt_dynamic_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """With only dynamic instructions, the Anthropic cache point falls back to the static `system_prompt` prefix.

    The dynamic instruction block stays uncached; the breakpoint anchors to the message preceding it.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        system_prompt='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'The current focus is distributed systems.'

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    system_messages = [m for m in single_request_body(vcr)['messages'] if m['role'] in ('system', 'developer')]
    # Static `system_prompt` prefix carries the breakpoint; the dynamic tail does not.
    assert isinstance(system_messages[0]['content'], list) and any(
        'cache_control' in part for part in system_messages[0]['content']
    )
    assert system_messages[-1]['content'] == 'The current focus is distributed systems.'


async def test_openrouter_cache_instructions_dynamic_only_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Instruction caching is skipped for Anthropic when the only instructions are dynamic and there is no static prefix.

    With no static block to anchor the breakpoint to, no `cache_control` is added anywhere.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model, model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True))

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'You are a helpful assistant that specializes in caching. ' * 20

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    for m in single_request_body(vcr)['messages']:
        content = m['content']
        assert isinstance(content, str) or all('cache_control' not in part for part in content)


async def test_openrouter_cache_instructions_no_instructions_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """`openrouter_cache_instructions` is a no-op when there are no instructions and no system prompt.

    With no structured instruction parts, the fallback scans for a `system`/`developer` message to
    anchor the breakpoint to; finding none, it adds no `cache_control` anywhere.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model, model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True))

    result = await agent.run('Say hello in one word.')

    assert isinstance(result.output, str)
    for m in single_request_body(vcr)['messages']:
        content = m['content']
        assert isinstance(content, str) or all('cache_control' not in part for part in content)


async def test_openrouter_cache_messages_preserves_cachepoint_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """An explicit `CachePoint(ttl='1h')` on the final block is not overwritten by `openrouter_cache_messages`."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model, model_settings=OpenRouterModelSettings(openrouter_cache_messages=True))

    result = await agent.run(['Final context to cache. ' * 20, CachePoint(ttl='1h')])

    assert isinstance(result.output, str)
    content = single_request_body(vcr)['messages'][-1]['content']
    assert content[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '1h'}


async def test_openrouter_cache_tool_definitions_gemini_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """`openrouter_cache_tool_definitions` has no effect for Gemini (tool caching is Anthropic-only)."""
    model = openrouter_model('google/gemini-2.5-flash')
    agent = Agent(model, model_settings=OpenRouterModelSettings(openrouter_cache_tool_definitions=True))

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Sunny in {city}'

    result = await agent.run('What tools do you have? List them briefly.')

    assert isinstance(result.output, str)
    tools = single_request_body(vcr).get('tools', [])
    assert tools
    assert all('cache_control' not in tool for tool in tools)


# ===== Cache E2E tests with cassettes =====


async def test_openrouter_cache_point_anthropic_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test CachePoint with Anthropic model via OpenRouter using real API."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Here is some important context to cache.' * 20,
                            CachePoint(),
                            'Summarize the context in one sentence.',
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The context consists of a repeated phrase stating that there is "important context to cache," repeated 20 times, but contains no actual substantive information.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=176,
                    output_tokens=34,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.001038,
                    'upstream_inference_cost': 0.001038,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009364-SjDV5yqNtQzbIiXso5JR',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was in the request
    assert vcr is not None
    request_body = single_request_body(vcr)
    user_content = request_body['messages'][0]['content']
    # The first content part should have cache_control
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_point_gemini_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test CachePoint with Gemini model via OpenRouter using real API."""
    model = openrouter_model('google/gemini-2.5-flash')
    agent = Agent(model)

    result = await agent.run(
        ['Here is some important context to cache.' * 20, CachePoint(), 'Summarize the context in one sentence.']
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Here is some important context to cache.' * 20,
                            CachePoint(),
                            'Summarize the context in one sentence.',
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The provided text repeatedly emphasizes the importance of caching context.')],
                usage=RequestUsage(
                    input_tokens=168,
                    output_tokens=11,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='google/gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'STOP',
                    'downstream_provider': 'Google',
                    'cost': 7.79e-05,
                    'upstream_inference_cost': 7.79e-05,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009367-3zFFs0yQRvCe01Kda6ID',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was in the request (without TTL for Gemini)
    assert vcr is not None
    request_body = single_request_body(vcr)
    user_content = request_body['messages'][0]['content']
    assert 'cache_control' in user_content[0]
    assert user_content[0]['cache_control'] == {'type': 'ephemeral'}


async def test_openrouter_cache_instructions_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test openrouter_cache_instructions with Anthropic model via real API."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        instructions='You are a helpful assistant that specializes in caching. ' * 20,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run('What do you specialize in? Answer in one sentence.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What do you specialize in? Answer in one sentence.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                instructions=IsStr(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='I specialize in caching.')],
                usage=RequestUsage(
                    input_tokens=260,
                    output_tokens=10,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.00093,
                    'upstream_inference_cost': 0.00093,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009369-muGga581yy6h5yw9o2yt',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to system message
    assert vcr is not None
    request_body = single_request_body(vcr)
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    content = system_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_messages_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test openrouter_cache_messages with Anthropic model via real API."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        instructions='Be helpful.',
        model_settings=OpenRouterModelSettings(openrouter_cache_messages=True),
    )

    result = await agent.run('Say hello in one word.')

    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Say hello in one word.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Be helpful.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello!')],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=5,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.000126,
                    'upstream_inference_cost': 0.000126,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009372-jrp75p3HzyzBsmgfbEZn',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to the last message
    assert vcr is not None
    request_body = single_request_body(vcr)
    last_msg = request_body['messages'][-1]
    content = last_msg['content']
    assert isinstance(content, list)
    assert 'cache_control' in content[-1]


async def test_openrouter_cache_tool_definitions_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test openrouter_cache_tool_definitions with Anthropic model via real API."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')

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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What tools do you have available? Just list them briefly.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
I have one tool available:

- **get_weather**: Retrieves the current weather for a specified city.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=566,
                    output_tokens=27,
                    details={'is_byok': False, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'downstream_provider': 'Amazon Bedrock',
                    'cost': 0.002103,
                    'upstream_inference_cost': 0.002103,
                    'is_byok': False,
                    'timestamp': IsDatetime(),
                },
                provider_response_id='gen-1773009374-OCscsda5VF5I9PRwtZv1',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was added to the last tool
    assert vcr is not None
    request_body = single_request_body(vcr)
    tools = request_body.get('tools', [])
    assert len(tools) >= 1
    assert 'cache_control' in tools[-1]
    assert tools[-1]['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}


async def test_openrouter_cache_messages_anthropic_real_api(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory
) -> None:
    """Test that openrouter_cache_messages produces cache write/read metrics for Anthropic via OpenRouter.

    Forces routing to the Anthropic provider directly (not Bedrock) to ensure cache locality.
    The first call populates the cache, and the second call reads from it.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        instructions='You are a helpful assistant.',
        model_settings=OpenRouterModelSettings(
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    # Must exceed Claude Sonnet's cacheable prompt minimum.
    result1 = await agent.run(
        'Analyze the architectural patterns used in distributed database systems and their tradeoffs. ' * 200
    )
    usage1 = result1.usage

    assert usage1.requests == 1
    assert usage1.input_tokens > 2000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call continues the conversation — the previous cached message is still in the request
    result2 = await agent.run('Can you summarize that in one sentence?', message_history=result1.all_messages())
    usage2 = result2.usage

    # cache_read_tokens > 0 proves caching actually worked end-to-end
    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_instructions_gemini_real_api(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory
) -> None:
    """Test that openrouter_cache_instructions produces cache write/read metrics for Gemini via OpenRouter.

    Uses cache_instructions with a long system prompt so the cached content is identical across
    both calls. Forces routing to Google AI Studio to ensure cache locality.
    """
    model = openrouter_model('google/gemini-2.5-flash')

    long_instructions = (
        'You are a specialized assistant that helps with distributed systems design. '
        'You have deep expertise in consensus protocols, CAP theorem, and eventual consistency. '
    ) * 80  # Long enough to exceed Gemini's current cacheable prompt minimum.

    agent = Agent(
        model,
        instructions=long_instructions,
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_provider={'order': ['google-ai-studio'], 'allow_fallbacks': False},
            max_tokens=300,
        ),
    )

    # First call populates the cache with the long system instructions
    result1 = await agent.run('What is the CAP theorem?')
    usage1 = result1.usage

    assert usage1.requests == 1
    assert usage1.input_tokens > 1000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call reuses the same system instructions — should hit cache
    result2 = await agent.run('What is eventual consistency?')
    usage2 = result2.usage

    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_cache_streaming_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Test that cache_control is correctly included in requests when using streaming."""
    model = openrouter_model('anthropic/claude-sonnet-4.6')
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
    assert stream.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Say hello in one word.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions=IsStr(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello!')],
                usage=RequestUsage(
                    input_tokens=254,
                    output_tokens=5,
                    details={'is_byok': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'image_tokens': 0},
                ),
                model_name='anthropic/claude-4.6-sonnet-20260217',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'timestamp': IsDatetime(),
                    'downstream_provider': 'Amazon Bedrock',
                    'finish_reason': 'stop',
                },
                provider_response_id='gen-1773012759-4u9w7As08eMtL75bWtu8',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify cache_control was included in the request
    assert vcr is not None
    request_body = single_request_body(vcr)

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


async def test_openrouter_cache_all_settings_real_api(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory
) -> None:
    """Test all cache settings combined with actual cache write+read metrics.

    Enables cache_instructions, cache_tool_definitions, and cache_messages together,
    forces Anthropic routing for cache locality, and verifies cache write/read metrics.
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')

    agent = Agent(
        model,
        instructions=(
            'You are an assistant that specializes in mathematics and calculations. '
            'Always show your work step by step. '
        )
        * 100,  # Long enough to exceed Claude Sonnet's cacheable prompt minimum.
        model_settings=OpenRouterModelSettings(
            openrouter_cache_instructions=True,
            openrouter_cache_tool_definitions=True,
            openrouter_cache_messages=True,
            openrouter_provider={'order': ['anthropic'], 'allow_fallbacks': False},
            max_tokens=100,
        ),
    )

    @agent.tool_plain
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return 'result'

    result1 = await agent.run('What is 123 * 456?')
    usage1 = result1.usage

    assert usage1.requests >= 1
    assert usage1.input_tokens > 2000
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Second call with same agent — system instructions + tools should be cached
    result2 = await agent.run('What is 789 + 321?')
    usage2 = result2.usage

    assert usage2.requests >= 1
    assert usage2.cache_read_tokens > 0
    assert usage2.output_tokens > 0


async def test_openrouter_limit_cache_points_e2e(
    allow_model_requests: None, openrouter_model: OpenRouterModelFactory, vcr: Cassette
) -> None:
    """Excess cache breakpoints are trimmed (oldest dropped) so the request stays within Anthropic's limit.

    Sends 5 `CachePoint` markers plus `cache_instructions` (6 total breakpoints) to an
    Anthropic model via OpenRouter. Without limiting, Anthropic would return a 400 error.
    Verifies the request succeeds, the recorded request has at most 4 `cache_control` breakpoints,
    the system instruction breakpoint is preserved, and the surviving message breakpoints are the
    newest ones (oldest dropped first).
    """
    model = openrouter_model('anthropic/claude-sonnet-4.6')
    agent = Agent(
        model,
        instructions='You are a helpful assistant. ' * 50,
        model_settings=OpenRouterModelSettings(openrouter_cache_instructions=True),
    )

    result = await agent.run(
        [
            'Context block one. ' * 20,
            CachePoint(),
            'Context block two. ' * 20,
            CachePoint(),
            'Context block three. ' * 20,
            CachePoint(),
            'Context block four. ' * 20,
            CachePoint(),
            'Context block five. ' * 20,
            CachePoint(),
            'Summarize everything in one sentence.',
        ]
    )

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    request_body = single_request_body(vcr)

    cache_count = 0
    for msg in request_body['messages']:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    assert cache_count <= 4

    # System instruction breakpoint is preserved.
    system_msg = next(m for m in request_body['messages'] if m['role'] in ('system', 'developer'))
    assert any('cache_control' in block for block in system_msg['content'])

    # Of the 5 CachePoints in the user message, only the newest survive (oldest dropped first):
    # 1 system + 3 newest message breakpoints = 4.
    user_msg = next(m for m in request_body['messages'] if m['role'] == 'user')
    cached_texts = [block['text'] for block in user_msg['content'] if 'cache_control' in block]
    assert all('one' not in text and 'two' not in text for text in cached_texts)
    assert any('five' in text for text in cached_texts)
