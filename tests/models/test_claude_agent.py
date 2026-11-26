"""`ClaudeAgentModel` against a scripted Claude Code process.

Claude Code is a subprocess the Claude Agent SDK talks to over stdin and stdout, so its traffic can't be recorded as a
cassette. These tests run the real SDK and the real `ClaudeAgentModel` in-process against `FakeClaudeCode`, which plays
Claude Code's side of the stream-json protocol (see `claude_code_fake.py`), and assert what each session was started
with and sent.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryImage, ImageUrl, ModelRetry, Tool, ToolReturn
from pydantic_ai._genai_prices import lookup_context_window
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelHTTPError,
    UnsupportedSettingWarning,
    UserError,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters, infer_model
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsInstance, IsStr, try_import

with try_import() as imports_successful:
    import anthropic
    from claude_agent_sdk import ProcessError

    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.claude_agent import ClaudeAgentModel
    from pydantic_ai.providers import _claude_agent_sessions
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.claude_agent import ClaudeAgentProvider

    from .claude_code_fake import FakeClaudeCode, FakeResponse, FakeSession, text, thinking, tool_use

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='claude-agent-sdk not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def claude_agent(*responses: FakeResponse) -> tuple[ClaudeAgentModel, FakeClaudeCode]:
    fake = FakeClaudeCode(list(responses))
    provider = ClaudeAgentProvider()
    provider._sessions.transport_factory = fake  # pyright: ignore[reportPrivateUsage]
    # Each test scripts every request it expects; `test_a_failed_request_is_retried` covers the default retries.
    return ClaudeAgentModel('claude-sonnet-5-5', provider=provider, settings={'max_retries': 0}), fake


def frame_shape(session: FakeSession) -> list[tuple[str, list[str], bool]]:
    """Each frame a session was sent: its role, its content block types, and whether it queried the model."""
    return [
        (frame['type'], [block['type'] for block in frame['message']['content']], frame.get('shouldQuery', True))
        for frame in session.frames
    ]


def test_infer_model():
    model = infer_model('claude-agent:claude-sonnet-5-5')
    assert isinstance(model, ClaudeAgentModel)
    assert model.system == 'claude-agent'
    assert model.provider is not None and model.provider.canonical_system == 'anthropic'
    profile = model.profile
    assert (
        profile.get('supports_cache'),
        profile.get('supports_json_schema_output'),
        profile.get('supported_native_tools'),
    ) == (
        False,
        False,
        frozenset(),
    )


async def test_safe_options_and_our_prompt():
    """Claude Code gets our instructions as its whole system prompt, and none of its own tools, settings or MCP."""
    model, fake = claude_agent(FakeResponse([text('PAX> hi')]))
    agent = Agent(model, instructions='You are Pax.')

    result = await agent.run('Hello')

    assert result.output == 'PAX> hi'
    [session] = fake.sessions
    options = session.options
    assert {
        'model': options.model,
        'system_prompt': options.system_prompt,
        'tools': options.tools,
        'mcp_servers': list(options.mcp_servers) if isinstance(options.mcp_servers, dict) else options.mcp_servers,
        'strict_mcp_config': options.strict_mcp_config,
        'setting_sources': options.setting_sources,
        'skills': options.skills,
        'verbatim_prompts': options.verbatim_prompts,
        'include_partial_messages': options.include_partial_messages,
        'client_app': options.client_app,
        'nonessential_traffic': options.env.get('CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'),
        'max_output_tokens': options.env.get('CLAUDE_CODE_MAX_OUTPUT_TOKENS'),
        'credentials_from_env': {
            name: value for name, value in options.env.items() if name.startswith(('ANTHROPIC_', 'CLAUDE_CODE_USE_'))
        },
    } == snapshot(
        {
            'model': 'claude-sonnet-5-5',
            'system_prompt': 'You are Pax.',
            'tools': [],
            'mcp_servers': ['pydantic_ai'],
            'strict_mcp_config': True,
            'setting_sources': [],
            'skills': [],
            'verbatim_prompts': True,
            'include_partial_messages': True,
            'client_app': IsStr(regex=r'pydantic-ai/.+'),
            'nonessential_traffic': '1',
            'max_output_tokens': '128000',
            'credentials_from_env': {
                'ANTHROPIC_API_KEY': '',
                'ANTHROPIC_AUTH_TOKEN': '',
                'ANTHROPIC_BASE_URL': '',
                'CLAUDE_CODE_USE_BEDROCK': '',
                'CLAUDE_CODE_USE_VERTEX': '',
                'CLAUDE_CODE_USE_FOUNDRY': '',
                'CLAUDE_CODE_USE_GATEWAY': '',
                'CLAUDE_CODE_USE_MANTLE': '',
                'CLAUDE_CODE_USE_ANTHROPIC_AWS': '',
                'CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD': '',
            },
        }
    )
    assert frame_shape(session) == snapshot([('user', ['text'], True)])
    assert session.closed


async def test_tool_calls_continue_the_same_session():
    """Claude Code runs our tools one at a time after the response; each waits for the next request's result."""
    model, fake = claude_agent(
        FakeResponse(
            [
                tool_use('toolu_1', 'get_weather', {'city': 'Lisbon'}),
                tool_use('toolu_2', 'get_time', {'city': 'Lisbon'}),
            ],
            stop_reason='tool_use',
        ),
        FakeResponse([text('Sunny at 14:05.')]),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    @agent.tool_plain
    def get_time(city: str) -> str:
        return f'14:05 in {city}'

    result = await agent.run('Weather and time in Lisbon?')

    assert result.output == 'Sunny at 14:05.'
    [session] = fake.sessions
    assert frame_shape(session) == snapshot([('user', ['text'], True)])
    assert session.tool_results == snapshot(
        [
            {'tool_use_id': 'toolu_1', 'content': [{'type': 'text', 'text': 'Sunny in Lisbon'}], 'isError': False},
            {'tool_use_id': 'toolu_2', 'content': [{'type': 'text', 'text': '14:05 in Lisbon'}], 'isError': False},
        ]
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Weather and time in Lisbon?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_weather', args='{"city": "Lisbon"}', tool_call_id='toolu_1'),
                    ToolCallPart(tool_name='get_time', args='{"city": "Lisbon"}', tool_call_id='toolu_2'),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-sonnet-5-5',
                model_id='claude-agent:claude-sonnet-5-5',
                timestamp=IsDatetime(),
                provider_name='claude-agent',
                provider_url='https://claude-agent.invalid',
                canonical_system='anthropic',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_fake',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='Sunny in Lisbon',
                        tool_call_id='toolu_1',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_time', content='14:05 in Lisbon', tool_call_id='toolu_2', timestamp=IsDatetime()
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Sunny at 14:05.')],
                usage=IsInstance(RequestUsage),
                model_name='claude-sonnet-5-5',
                model_id='claude-agent:claude-sonnet-5-5',
                timestamp=IsDatetime(),
                provider_name='claude-agent',
                provider_url='https://claude-agent.invalid',
                canonical_system='anthropic',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_fake',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_outer_scope_keeps_the_session_across_runs():
    """Inside `async with agent:`, a run that continues the conversation sends only its new user message."""
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    [session] = fake.sessions
    assert frame_shape(session) == snapshot([('user', ['text'], True), ('user', ['text'], True)])
    assert session.closed


async def test_a_new_session_replays_the_history():
    """Without an outer scope each run gets its own session, which is sent the history without querying the model."""
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'get_weather', {'city': 'Lisbon'})], stop_reason='tool_use'),
        FakeResponse([text('Sunny.')]),
        FakeResponse([text('Still sunny.')]),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    first = await agent.run('Weather?')
    second = await agent.run('And now?', message_history=first.all_messages())

    assert second.output == 'Still sunny.'
    assert [frame_shape(session) for session in fake.sessions] == snapshot(
        [
            [('user', ['text'], True)],
            [
                ('user', ['text'], False),
                ('assistant', ['tool_use'], True),
                ('user', ['tool_result'], False),
                ('assistant', ['text'], True),
                ('user', ['text'], True),
            ],
        ]
    )
    replayed_call = fake.sessions[1].frames[1]['message']['content'][0]
    assert replayed_call['name'] == 'mcp__pydantic_ai__get_weather'


async def test_changed_instructions_start_a_new_session():
    """A session's system prompt is fixed when it starts, so new instructions replay the history into a new one."""
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))

    async with model:
        first = await Agent(model, instructions='Be terse.').run('Count.')
        await Agent(model, instructions='Be verbose.').run('Again.', message_history=first.all_messages())

    assert [session.options.system_prompt for session in fake.sessions] == ['Be terse.', 'Be verbose.']
    assert frame_shape(fake.sessions[1]) == snapshot(
        [('user', ['text'], False), ('assistant', ['text'], True), ('user', ['text'], True)]
    )
    assert all(session.closed for session in fake.sessions)


async def test_image_tool_result_goes_through_mcp():
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'snapshot', {})], stop_reason='tool_use'),
        FakeResponse([text('A pixel.')]),
    )
    agent = Agent(model)

    @agent.tool_plain(name='snapshot')
    def take_snapshot() -> ToolReturn:
        return ToolReturn(return_value=BinaryImage(data=b'\x89PNG', media_type='image/png'))

    result = await agent.run('Take a snapshot.')

    assert result.output == 'A pixel.'
    [session] = fake.sessions
    assert [item['type'] for item in session.tool_results[0]['content']] == snapshot(['image'])


async def test_not_signed_in():
    model, _ = claude_agent(FakeResponse(error='authentication_failed', error_text='Not logged in · Please run /login'))

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('Hello')

    assert isinstance(exc_info.value, ModelAuthenticationError)
    assert exc_info.value.status_code == 401
    assert exc_info.value.body == snapshot(
        {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'Not logged in · Please run /login'}}
    )


async def test_subscription_limit(monkeypatch: pytest.MonkeyPatch):
    """The plan's reset time becomes `Retry-After`, and the Anthropic client is told not to retry."""
    monkeypatch.setattr(_claude_agent_sessions.time, 'time', lambda: 4_000_000_000 - 3600.5)
    model, _ = claude_agent(
        FakeResponse(error='rate_limit', error_text="You've hit your limit", rate_limit_resets_at=4_000_000_000)
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('Hello')

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 3601
    assert exc_info.value.headers is not None and exc_info.value.headers['x-should-retry'] == 'false'


async def test_count_tokens_is_not_supported():
    model, _ = claude_agent()

    with pytest.raises(NotImplementedError, match='Claude Code does not count tokens'):
        await model.count_tokens([ModelRequest.user_text_prompt('Hello')], None, ModelRequestParameters())


async def test_thinking_response_continues_the_session():
    """A response with signed thinking still matches the history that comes back, so the session continues."""
    model, fake = claude_agent(
        FakeResponse([thinking('Count up.', 'sig-1'), text('One.')]),
        FakeResponse([text('Two.')]),
    )
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        await agent.run('Again.', message_history=first.all_messages())

    [session] = fake.sessions
    assert frame_shape(session) == snapshot([('user', ['text'], True), ('user', ['text'], True)])


@pytest.mark.parametrize('edit', ['answer', 'prompt', 'extra turn'])
async def test_edited_history_replays_into_a_new_session(edit: str):
    """History that isn't what the session saw, followed by one user message, never continues it."""
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        history = first.all_messages()
        request, response = history
        assert isinstance(request, ModelRequest) and isinstance(response, ModelResponse)
        if edit == 'answer':
            history = [request, replace(response, parts=[TextPart(content='Uno.')])]
        elif edit == 'prompt':
            history = [replace(request, parts=[UserPromptPart(content='Count!')]), response]
        else:
            history = [
                *history,
                ModelRequest.user_text_prompt('And?'),
                ModelResponse(parts=[TextPart(content='Done.')]),
            ]
        await agent.run('Again.', message_history=history)

    assert len(fake.sessions) == 2
    assert frame_shape(fake.sessions[1])[-1] == ('user', ['text'], True)
    assert all(session.closed for session in fake.sessions)


async def test_a_run_ended_by_an_output_tool_replays_the_next_run():
    """A run that ends on an output tool call leaves Claude Code waiting for its result, which the next run's
    history sends together with a new prompt; that can't continue the waiting turn, so it's replayed."""
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'final_result', {'city': 'Lisbon'})], stop_reason='tool_use'),
        FakeResponse([tool_use('toolu_2', 'final_result', {'city': 'Porto'})], stop_reason='tool_use'),
    )
    agent = Agent(model, output_type=City)

    async with agent:
        first = await agent.run('Pick a city.')
        second = await agent.run('Another one?', message_history=first.all_messages())

    assert (first.output, second.output) == (City(city='Lisbon'), City(city='Porto'))
    assert frame_shape(fake.sessions[1]) == snapshot(
        [('user', ['text'], False), ('assistant', ['tool_use'], True), ('user', ['tool_result', 'text'], True)]
    )


@dataclass
class City:
    city: str


async def test_concurrent_requests_of_one_conversation_get_their_own_sessions():
    release = asyncio.Event()
    model, fake = claude_agent(
        FakeResponse([text('One.')]), FakeResponse([text('A.')], hold=release), FakeResponse([text('B.')])
    )
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        history = first.all_messages()
        # The first continuation keeps the conversation's session busy while the second is sent.
        continued = asyncio.create_task(agent.run('Again.', message_history=history))
        await fake.holding.wait()
        other = await agent.run('Again.', message_history=history)
        release.set()
        results = [await continued, other]

    assert [result.output for result in results] == ['A.', 'B.']
    assert len(fake.sessions) == 2
    assert all(session.closed for session in fake.sessions)


async def test_the_least_recently_used_idle_session_makes_room(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_claude_agent_sessions, '_MAX_SESSIONS', 1)
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        await agent.run('First conversation.')
        assert not fake.sessions[0].closed
        await agent.run('Second conversation.')
        assert fake.sessions[0].closed and not fake.sessions[1].closed

    assert fake.sessions[1].closed


async def test_a_new_conversation_is_not_kept_while_every_session_is_busy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_claude_agent_sessions, '_MAX_SESSIONS', 1)
    release = asyncio.Event()
    model, fake = claude_agent(FakeResponse([text('One.')], hold=release), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = asyncio.create_task(agent.run('First conversation.'))
        await fake.holding.wait()
        await agent.run('Second conversation.')
        assert [session.closed for session in fake.sessions] == [False, True]
        release.set()
        await first

    assert all(session.closed for session in fake.sessions)


async def test_a_session_whose_process_exited_is_replaced():
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse(exit=True), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        with pytest.raises(ModelHTTPError, match='Claude Code exited without a response'):
            await agent.run('Again.', message_history=first.all_messages())
        second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    assert len(fake.sessions) == 2


async def test_a_process_that_exits_between_turns_is_replaced():
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        await fake.sessions[0].close()
        second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    assert len(fake.sessions) == 2


async def test_a_process_that_exits_during_replay():
    fake = FakeClaudeCode([FakeResponse([text('One.')])], exit_on_history=True)
    provider = ClaudeAgentProvider()
    provider._sessions.transport_factory = fake  # pyright: ignore[reportPrivateUsage]
    agent = Agent(ClaudeAgentModel('claude-sonnet-5-5', provider=provider))
    first = await agent.run('Count.')

    with pytest.raises(ModelHTTPError, match='Claude Code exited while it was sent the history'):
        await agent.run('Again.', message_history=first.all_messages())

    assert all(session.closed for session in fake.sessions)


async def test_a_response_cut_short():
    model, fake = claude_agent(FakeResponse([text('One.')], cut='error'))

    with pytest.raises(ModelAPIError, match='Connection error'):
        await Agent(model).run('Count.')

    assert fake.sessions[0].closed


async def test_a_crash_while_waiting_for_the_response():
    model, fake = claude_agent(FakeResponse(crash=RuntimeError('Claude Code crashed')))

    with pytest.raises(ModelAPIError):
        await Agent(model).run('Count.')

    assert fake.sessions[0].closed


async def test_a_claude_code_failure_while_waiting_for_the_response():
    model, fake = claude_agent(FakeResponse(crash=ProcessError('Command failed', exit_code=1)))

    with pytest.raises(ModelHTTPError, match='Claude Code failed: Command failed') as exc_info:
        await Agent(model).run('Count.')

    assert exc_info.value.status_code == 500
    assert fake.sessions[0].closed


async def test_claude_code_that_fails_to_start():
    model, fake = claude_agent()
    fake.missing = True

    with pytest.raises(ModelHTTPError, match='Claude Code failed: Claude Code not found') as exc_info:
        await Agent(model).run('Count.')

    assert exc_info.value.status_code == 500
    assert len(fake.sessions) == 1


async def test_a_claude_code_failure_after_the_response_started():
    model, fake = claude_agent(FakeResponse([text('Partial')], cut='crash'))

    with pytest.raises(ModelAPIError, match='Claude Code failed: Command failed'):
        await Agent(model).run('Count.')

    assert fake.sessions[0].closed


async def test_an_event_newer_than_the_anthropic_types_replays_the_next_request():
    """A stop reason the installed `anthropic` types don't know still reaches the response, but isn't recorded."""
    model, fake = claude_agent(FakeResponse([text('One.')], stop_reason='a_new_reason'), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        second = await agent.run('Again.', message_history=first.all_messages())

    assert (first.output, second.output) == ('One.', 'Two.')
    assert len(fake.sessions) == 2
    assert frame_shape(fake.sessions[1]) == snapshot(
        [('user', ['text'], False), ('assistant', ['text'], True), ('user', ['text'], True)]
    )


async def test_a_block_newer_than_the_anthropic_types_replays_the_next_request():
    model, fake = claude_agent(FakeResponse([{'type': 'a_new_block'}, text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        with pytest.warns(UserWarning, match="content block type is not yet implemented.+'a_new_block'"):
            first = await agent.run('Count.')
        second = await agent.run('Again.', message_history=first.all_messages())

    assert (first.output, second.output) == ('One.', 'Two.')
    assert len(fake.sessions) == 2


async def test_a_process_that_fails_between_turns_is_replaced():
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async with agent:
        first = await agent.run('Count.')
        fake.sessions[0].write_error = ProcessError('Command failed', exit_code=1)
        second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    assert len(fake.sessions) == 2 and fake.sessions[0].closed


async def test_tool_choice_none_offers_no_tools():
    """Claude Code has no `tool_choice`, so a request that allows no tool call gets a session without tools."""
    model, fake = claude_agent(FakeResponse([text('Sunny.')]), FakeResponse([text('Rainy.')]))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather() -> str:
        return 'Sunny'  # pragma: no cover

    await agent.run('Weather?', model_settings={'tool_choice': 'none'})
    await agent.run('Weather?')

    assert [session.tool_names for session in fake.sessions] == [[], ['get_weather']]


async def test_thinking_off_on_a_model_whose_api_turns_it_off_between_tools():
    """Claude Code has its own flag to turn thinking off, so Claude Sonnet 5.5 doesn't get `between_tools`."""
    model, fake = claude_agent(FakeResponse([text('391')]))

    await Agent(model, model_settings={'thinking': False}).run('17*23?')

    assert fake.sessions[0].options.thinking == {'type': 'disabled'}


async def test_a_thinking_config_claude_code_cant_take():
    model, fake = claude_agent()

    with pytest.raises(ModelHTTPError, match="Claude Code can't serve this request") as exc_info:
        await Agent(model, model_settings=AnthropicModelSettings(anthropic_thinking={'type': 'between_tools'})).run(
            'Hi'
        )

    assert exc_info.value.status_code == 400
    assert fake.sessions == []


async def test_a_tool_schema_with_a_remote_reference_is_refused():
    """The Agent SDK validates arguments with `jsonschema`, which would fetch the reference from this process."""
    model, fake = claude_agent()
    schema = {'type': 'object', 'properties': {'city': {'$ref': 'http://127.0.0.1:9/city.json'}}}

    def lookup(city: str) -> str:
        return city  # pragma: no cover

    agent = Agent(model, tools=[Tool.from_schema(lookup, name='lookup', description='', json_schema=schema)])

    with pytest.raises(ModelHTTPError, match="Claude Code can't serve this request") as exc_info:
        await agent.run('Hi')

    assert exc_info.value.status_code == 400
    assert 'http://127.0.0.1:9/city.json' in str(exc_info.value.body)
    assert fake.sessions == []


async def test_anthropic_settings_claude_code_doesnt_send_are_reported():
    model, _ = claude_agent(FakeResponse([text('Hi.')]))
    settings = AnthropicModelSettings(anthropic_effort='low', anthropic_metadata={'user_id': 'u'})

    with pytest.warns(UnsupportedSettingWarning, match='`anthropic_metadata` setting is not supported'):
        await Agent(model, model_settings=settings).run('Hello')


async def test_tuning_settings_are_ignored_silently():
    """Claude Code always caches, and drops tuning knobs like `anthropic_cache_messages` without changing a guarantee."""
    model, _ = claude_agent(FakeResponse([text('Hi.')]))
    settings = AnthropicModelSettings(cache=True, anthropic_cache_messages=True, anthropic_service_tier='auto')

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnsupportedSettingWarning)
        await Agent(model, model_settings=settings).run('Hello')


async def test_a_failed_request_is_retried():
    """Claude Code is set not to retry, so Pydantic AI retries a request that failed, in a new session."""
    model, fake = claude_agent(FakeResponse(error='server_error', error_text='Overloaded'), FakeResponse([text('Hi.')]))

    result = await Agent(model).run('Hello', model_settings={'max_retries': 1})

    assert result.output == 'Hi.'
    assert len(fake.sessions) == 2


async def test_a_plan_limit_is_not_retried():
    """A plan's usage limit resets hours later, so it fails at once, whatever `max_retries` allows."""
    model, fake = claude_agent(
        FakeResponse(error='rate_limit', error_text="You've hit your limit", rate_limit_resets_at=4_000_000_000)
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('Hello', model_settings={'max_retries': 2})

    assert exc_info.value.status_code == 429
    assert len(fake.sessions) == 1


async def test_an_error_claude_code_has_no_status_for():
    model, _ = claude_agent(FakeResponse(error='a_new_kind', error_text='Something new went wrong'))

    with pytest.raises(ModelHTTPError, match='Something new went wrong') as exc_info:
        await Agent(model).run('Hello')

    assert exc_info.value.status_code == 500


async def test_the_api_requests_own_status_wins():
    """Claude Code reports the failed API request's status, which is more precise than its error kind's."""
    model, _ = claude_agent(FakeResponse(error='server_error', error_text='Overloaded', api_error_status=529))

    with pytest.raises(ModelHTTPError, match='Overloaded') as exc_info:
        await Agent(model).run('Hello')

    assert exc_info.value.status_code == 529


async def test_a_session_that_fails_to_continue_is_closed(monkeypatch: pytest.MonkeyPatch):
    model, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    agent = Agent(model)

    async def fail(self: object, message: object) -> bool:
        raise RuntimeError('unexpected')

    async with agent:
        first = await agent.run('Count.')
        with monkeypatch.context() as patch:
            patch.setattr(_claude_agent_sessions._Session, 'send', fail)  # pyright: ignore[reportPrivateUsage]
            with pytest.raises(ModelAPIError):
                await agent.run('Again.', message_history=first.all_messages())
        assert fake.sessions[0].closed
        second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    assert len(fake.sessions) == 2


async def test_a_process_that_exits_mid_response():
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'get_weather', {'city': 'Lisbon'})], stop_reason='tool_use'),
        FakeResponse(exit=True),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    with pytest.raises(ModelHTTPError, match='Claude Code exited'):
        await agent.run('Weather?')

    assert fake.sessions[0].closed


async def test_an_abandoned_stream_closes_its_session():
    model, fake = claude_agent(FakeResponse([text('One.')]))

    async with model.request_stream([ModelRequest.user_text_prompt('Count.')], None, ModelRequestParameters()):
        pass

    assert fake.sessions[0].closed


async def test_a_direct_request_uses_a_session_of_its_own():
    """Without an agent run or an entered model there's no lease, so the session closes with the response."""
    model, fake = claude_agent(FakeResponse([text('Hi.')]))

    response = await model_request(model, [ModelRequest.user_text_prompt('Hello')])

    assert response.parts == [TextPart(content='Hi.')]
    assert fake.sessions[0].closed


async def test_other_endpoints_are_not_served():
    client = ClaudeAgentProvider().client
    assert isinstance(client, anthropic.AsyncAnthropic)

    with pytest.raises(anthropic.NotFoundError):
        await client.beta.messages.count_tokens(
            model='claude-sonnet-5-5', messages=[{'role': 'user', 'content': 'Hello'}]
        )


def test_infer_model_needs_a_claude_agent_provider():
    with pytest.raises(UserError, match='Claude Code models require a `ClaudeAgentProvider`'):
        infer_model('claude-agent:claude-sonnet-5-5', provider_factory=lambda _: AnthropicProvider(api_key='key'))


def test_default_provider():
    model = ClaudeAgentModel('sonnet')
    assert isinstance(model.provider, ClaudeAgentProvider)


async def test_a_process_that_exits_after_the_response_started():
    model, fake = claude_agent(FakeResponse([text('One.')], cut='exit'))

    with pytest.raises(ModelAPIError):
        await Agent(model).run('Count.')

    assert fake.sessions[0].closed


async def test_a_retry_prompt_continues_the_session():
    """A tool's retry prompt reaches Claude Code as the call's error result, in the same session."""
    model, fake = claude_agent(
        FakeResponse(
            [text('Let me check.'), tool_use('toolu_1', 'get_weather', {'city': 'Atlantis'})], stop_reason='tool_use'
        ),
        FakeResponse([tool_use('toolu_2', 'get_weather', {'city': 'Lisbon'})], stop_reason='tool_use'),
        FakeResponse([text('Sunny.')]),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        if city == 'Atlantis':
            raise ModelRetry('Unknown city.')
        return f'Sunny in {city}'

    result = await agent.run('Weather?')

    assert result.output == 'Sunny.'
    [session] = fake.sessions
    assert [(item['isError'], item['content'][0]['text'][:13]) for item in session.tool_results] == snapshot(
        [(True, 'Unknown city.'), (False, 'Sunny in Lisb')]
    )


async def test_an_image_url_tool_result_replays_into_a_new_session():
    """MCP can't carry an image by URL, so its result is sent as part of the history of a new session."""
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'snapshot', {})], stop_reason='tool_use'),
        FakeResponse([text('A kiwi.')]),
    )
    agent = Agent(model)

    @agent.tool_plain(name='snapshot')
    def take_snapshot() -> ImageUrl:
        return ImageUrl(url='https://example.com/kiwi.png')

    result = await agent.run('Take a snapshot.')

    assert result.output == 'A kiwi.'
    assert len(fake.sessions) == 2
    assert frame_shape(fake.sessions[1]) == snapshot(
        [('user', ['text'], False), ('assistant', ['tool_use'], True), ('user', ['tool_result'], True)]
    )


async def test_an_empty_tool_result_continues_the_session():
    model, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'list_alerts', {})], stop_reason='tool_use'),
        FakeResponse([text('No alerts.')]),
    )
    agent = Agent(model)

    @agent.tool_plain
    def list_alerts() -> list[str]:
        return []

    result = await agent.run('Any alerts?')

    assert result.output == 'No alerts.'
    [session] = fake.sessions
    assert session.tool_results == snapshot(
        [{'tool_use_id': 'toolu_1', 'content': [{'type': 'text', 'text': ''}], 'isError': False}]
    )


def test_the_context_window_is_the_claude_apis():
    """Claude Code serves the Claude API's models, so compaction gets the same context window as on `anthropic:`."""
    window = lookup_context_window('claude-haiku-4-5', provider_name='anthropic')

    assert window is not None
    assert ClaudeAgentModel('claude-haiku-4-5').context_window == window
    assert ClaudeAgentModel('claude-not-a-model').context_window is None
