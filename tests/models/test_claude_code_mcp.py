"""`claude_code_inherit_mcp_servers`: Claude Code brings the user's own MCP servers and runs their tools itself.

Like `test_claude_code.py`, these run the real Claude Agent SDK against `claude_code_fake.py`, as Claude Code's traffic
is a subprocess's, not HTTP that a cassette could record.
"""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, NativeToolCallPart, TextPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, NativeToolReturnPart

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.claude_code import ClaudeCodeModel, ClaudeCodeModelSettings
    from pydantic_ai.providers import _claude_code_sessions
    from pydantic_ai.providers.claude_code import ClaudeCodeProvider

    from .claude_code_fake import (
        NOTHING,
        FakeClaudeCode,
        FakeResponse,
        FakeSession,
        external_tool_use,
        text,
        tool_use,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='claude-agent-sdk not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

SERVERS = ['claude.ai Docs', 'claude.ai Gmail']
ALL = ClaudeCodeModelSettings(claude_code_inherit_mcp_servers=True)
DOCS = ClaudeCodeModelSettings(claude_code_inherit_mcp_servers=['claude.ai Docs'])


@pytest.fixture(autouse=True)
def no_status_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude Code's servers load in a few seconds; the fake's load in a given number of status requests."""
    monkeypatch.setattr(_claude_code_sessions, '_MCP_STATUS_POLL', 0)


def claude_code(*responses: FakeResponse, **script: Any) -> tuple[ClaudeCodeModel, FakeClaudeCode]:
    fake = FakeClaudeCode(list(responses), mcp_servers=SERVERS, **script)
    provider = ClaudeCodeProvider()
    provider._sessions.transport_factory = fake  # pyright: ignore[reportPrivateUsage]
    return ClaudeCodeModel('claude-sonnet-5-5', provider=provider, settings={'max_retries': 0}), fake


def frame_shape(session: FakeSession) -> list[tuple[str, list[str], bool]]:
    return [
        (frame['type'], [block['type'] for block in frame['message']['content']], frame.get('shouldQuery', True))
        for frame in session.frames
    ]


def guide_call() -> FakeResponse:
    return FakeResponse(
        [external_tool_use('toolu_1', 'claude_ai_Docs', 'guide', {'items': ['index']})],
        stop_reason='tool_use',
        usage={'input_tokens': 10, 'output_tokens': 5},
    )


async def test_claude_codes_own_tool_calls_come_back_as_native_tool_parts():
    """Claude Code runs the call in the same turn, and the response holds the call, its result and the answer."""
    model, fake = claude_code(
        guide_call(), FakeResponse([text('It covers docs.')], usage={'input_tokens': 20, 'output_tokens': 7})
    )
    fake.external_results['guide'] = 'Docs are living documents.'

    result = await Agent(model).run('What does the guide cover?', model_settings=ALL)

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='mcp_server:claude.ai Docs',
                args='{"action":"call_tool","tool_name":"guide","tool_args":{"items": ["index"]}}',
                tool_call_id='toolu_1',
                provider_name='claude-code',
            ),
            NativeToolReturnPart(
                tool_name='mcp_server:claude.ai Docs',
                content={
                    'content': [{'citations': None, 'text': 'Docs are living documents.', 'type': 'text'}],
                    'is_error': False,
                },
                tool_call_id='toolu_1',
                timestamp=IsDatetime(),
                provider_name='claude-code',
            ),
            TextPart(content='It covers docs.'),
        ]
    )
    assert (response.usage.input_tokens, response.usage.output_tokens) == (30, 12)
    [session] = fake.sessions
    assert (session.options.strict_mcp_config, session.options.disallowed_tools) == (False, [])


async def test_named_servers_alone_are_offered():
    """The other servers' tools are disallowed, and a call to one anyway is denied."""
    model, fake = claude_code(
        FakeResponse([external_tool_use('toolu_1', 'claude_ai_Gmail', 'search', {})], stop_reason='tool_use'),
        FakeResponse([text('I have no access to your mail.')]),
    )

    result = await Agent(model).run('Search my mail.', model_settings=DOCS)

    assert fake.sessions[0].options.disallowed_tools == ['mcp__claude_ai_Gmail']
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [part.part_kind for part in response.parts] == snapshot(['builtin-tool-call', 'builtin-tool-return', 'text'])
    assert isinstance(returned := response.parts[1], NativeToolReturnPart)
    assert returned.content == snapshot(
        {'content': [{'citations': None, 'text': 'Permission denied', 'type': 'text'}], 'is_error': True}
    )


async def test_a_result_with_no_content():
    model, fake = claude_code(guide_call(), FakeResponse([text('Nothing there.')]))
    fake.external_results['guide'] = NOTHING

    result = await Agent(model).run('Guide?', model_settings=ALL)

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert isinstance(returned := response.parts[1], NativeToolReturnPart)
    assert returned.content == snapshot({'content': [], 'is_error': False})


async def test_a_server_claude_code_doesnt_have():
    model, fake = claude_code()

    with pytest.raises(ModelHTTPError, match='no MCP server named') as exc_info:
        await Agent(model).run('Hi', model_settings=ClaudeCodeModelSettings(claude_code_inherit_mcp_servers=['Notion']))

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'message': "Claude Code has no MCP server named 'Notion'; it has: 'claude.ai Docs', 'claude.ai Gmail'.",
            },
        }
    )
    assert fake.sessions == []


async def test_the_servers_are_found_once_after_they_load():
    model, fake = claude_code(FakeResponse([text('One.')]), FakeResponse([text('Two.')]), pending_polls=2)
    agent = Agent(model, model_settings=DOCS)

    await agent.run('First.')
    await agent.run('Second.')

    assert len(fake.discoveries) == 1
    assert len(fake.sessions) == 2


async def test_servers_that_never_finish_loading(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_claude_code_sessions, '_MCP_STATUS_WAIT', 0.05)
    model, fake = claude_code(FakeResponse([text('One.')]), pending_polls=10**9)

    result = await Agent(model, model_settings=DOCS).run('Hi')

    assert result.output == 'One.'
    assert fake.sessions[0].options.disallowed_tools == ['mcp__claude_ai_Gmail']


async def test_native_tool_parts_continue_and_replay():
    """The next request continues the session; once it's gone, the history is replayed as Claude Code's own blocks."""
    model, fake = claude_code(
        guide_call(),
        FakeResponse([text('It covers docs.')]),
        FakeResponse([text('Index.')]),
        FakeResponse([text('Docs.')]),
    )
    agent = Agent(model, model_settings=ALL)

    async with agent:
        first = await agent.run('What does the guide cover?')
        second = await agent.run('Which topic?', message_history=first.all_messages())
    third = await agent.run('And my first question?', message_history=second.all_messages())

    assert third.output == 'Docs.'
    assert [frame_shape(session) for session in fake.sessions] == snapshot(
        [
            [('user', ['text'], True), ('user', ['text'], True)],
            [
                ('user', ['text'], False),
                ('assistant', ['tool_use'], True),
                ('user', ['tool_result'], False),
                ('assistant', ['text'], True),
                ('user', ['text'], False),
                ('assistant', ['text'], True),
                ('user', ['text'], True),
            ],
        ]
    )


async def test_a_call_beside_one_of_ours():
    """When the model also calls an agent tool, the response ends there, and Claude Code's result comes in the next."""
    model, fake = claude_code(
        FakeResponse(
            [
                external_tool_use('toolu_1', 'claude_ai_Docs', 'guide', {}),
                tool_use('toolu_2', 'get_city', {}),
            ],
            stop_reason='tool_use',
        ),
        FakeResponse([text('Lisbon, and docs.')]),
        FakeResponse([text('Again.')]),
    )
    fake.external_results['guide'] = [{'type': 'text', 'text': 'Docs.'}, {'type': 'image', 'source': {}}]
    agent = Agent(model, model_settings=ALL)

    @agent.tool_plain
    def get_city() -> str:
        return 'Lisbon'

    result = await agent.run('Where, and what docs?')
    again = await agent.run('Again.', message_history=result.all_messages())

    assert [
        [part.part_kind for part in message.parts]
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
    ] == snapshot([['builtin-tool-call', 'tool-call'], ['builtin-tool-return', 'text']])
    assert again.output == 'Again.'
    assert frame_shape(fake.sessions[-1]) == snapshot(
        [
            ('user', ['text'], False),
            ('assistant', ['tool_use', 'tool_use'], True),
            ('user', ['tool_result'], False),
            ('user', ['tool_result'], False),
            ('assistant', ['text'], True),
            ('user', ['text'], True),
        ]
    )
