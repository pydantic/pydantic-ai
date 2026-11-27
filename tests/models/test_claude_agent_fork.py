"""A new Claude Code session forks the transcript of an earlier one, so it starts with the history in the prompt cache.

Like `test_claude_code.py`, these run the real Claude Agent SDK against `claude_code_fake.py`, as Claude Code's traffic
is a subprocess's, not HTTP that a cassette could record.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.claude_agent import ClaudeAgentModel
    from pydantic_ai.providers import _claude_agent_sessions
    from pydantic_ai.providers.claude_agent import ClaudeAgentProvider

    from .claude_code_fake import FakeClaudeCode, FakeResponse, FakeSession, text, tool_use

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='claude-agent-sdk not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def claude_agent(*responses: FakeResponse) -> tuple[Agent[None, str], FakeClaudeCode]:
    fake = FakeClaudeCode(list(responses))
    provider = ClaudeAgentProvider()
    provider._sessions.transport_factory = fake  # pyright: ignore[reportPrivateUsage]
    model = ClaudeAgentModel('claude-sonnet-5-5', provider=provider, settings={'max_retries': 0})
    return Agent(model), fake


def forked_from(session: FakeSession) -> tuple[str | None, str | None]:
    return session.options.resume, session.options.resume_session_at


def fork_points(agent: Agent[None, str]) -> list[tuple[str, str]]:
    model = agent.model
    assert isinstance(model, ClaudeAgentModel)
    provider = model._provider  # pyright: ignore[reportPrivateUsage]
    assert isinstance(provider, ClaudeAgentProvider)
    points = provider._sessions.fork_points.values()  # pyright: ignore[reportPrivateUsage]
    return [(point.session_id, point.message_uuid) for point in points]


def frame_shape(session: FakeSession) -> list[tuple[str, list[str], bool]]:
    return [
        (frame['type'], [block['type'] for block in frame['message']['content']], frame.get('shouldQuery', True))
        for frame in session.frames
    ]


async def test_the_next_run_forks_the_last_ones_transcript():
    """Without an outer scope each run gets its own session, which forks the last one's and is sent the new message."""
    agent, fake = claude_agent(
        FakeResponse([text('One.')]), FakeResponse([text('Two.')]), FakeResponse([text('Three.')])
    )

    first = await agent.run('Count.')
    second = await agent.run('Again.', message_history=first.all_messages())
    third = await agent.run('Once more.', message_history=second.all_messages())

    assert third.output == 'Three.'
    assert [forked_from(session) for session in fake.sessions] == snapshot(
        [(None, None), ('session-0', 'session-0/0'), ('session-1', 'session-1/1')]
    )
    assert all(session.options.fork_session == (index > 0) for index, session in enumerate(fake.sessions))
    assert [frame_shape(session) for session in fake.sessions] == snapshot(
        [[('user', ['text'], True)], [('user', ['text'], True)], [('user', ['text'], True)]]
    )


async def test_a_rewind_forks_at_the_answer_it_goes_on_from():
    agent, fake = claude_agent(
        FakeResponse([text('One.')]), FakeResponse([text('Two.')]), FakeResponse([text('Other.')])
    )

    first = await agent.run('Count.')
    await agent.run('Again.', message_history=first.all_messages())
    other = await agent.run('Something else.', message_history=first.all_messages())

    assert other.output == 'Other.'
    assert forked_from(fake.sessions[2]) == ('session-0', 'session-0/0')


async def test_the_fork_point_is_the_runs_answer_not_its_tool_call():
    """A response that calls a tool goes on in the same session, so only the answer after it is a fork point."""
    agent, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'get_city', {})], stop_reason='tool_use'),
        FakeResponse([text('Lisbon.')]),
        FakeResponse([text('Portugal.')]),
    )

    @agent.tool_plain
    def get_city() -> str:
        return 'Lisbon'

    first = await agent.run('Where?')
    second = await agent.run('Country?', message_history=first.all_messages())

    assert second.output == 'Portugal.'
    assert forked_from(fake.sessions[1]) == ('session-0', 'session-0/1')
    assert frame_shape(fake.sessions[1]) == snapshot([('user', ['text'], True)])


def edit_last_answer(messages: list[ModelMessage]) -> list[ModelMessage]:
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    return [*messages[:-1], replace(response, parts=[TextPart('Edited.')])]


async def test_an_edited_answer_forks_from_the_one_before_it():
    agent, fake = claude_agent(
        FakeResponse([text('One.')]), FakeResponse([text('Two.')]), FakeResponse([text('Three.')])
    )

    first = await agent.run('Count.')
    second = await agent.run('Again.', message_history=first.all_messages())
    await agent.run('Once more.', message_history=edit_last_answer(second.all_messages()))

    assert forked_from(fake.sessions[2]) == ('session-0', 'session-0/0')
    # What the transcript doesn't hold is sent without querying the model, then the new message.
    assert frame_shape(fake.sessions[2]) == snapshot(
        [('user', ['text'], False), ('assistant', ['text'], True), ('user', ['text'], True)]
    )


async def test_history_no_session_ended_is_replayed():
    agent, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))

    first = await agent.run('Count.')
    await agent.run('Again.', message_history=edit_last_answer(first.all_messages()))

    assert forked_from(fake.sessions[1]) == (None, None)
    assert frame_shape(fake.sessions[1]) == snapshot(
        [('user', ['text'], False), ('assistant', ['text'], True), ('user', ['text'], True)]
    )


async def test_a_transcript_claude_code_no_longer_has_is_replayed():
    """As when the transcript was deleted: the fork fails, and the history is replayed into a new session instead."""
    agent, fake = claude_agent(
        FakeResponse([text('One.')]), FakeResponse([text('Two.')]), FakeResponse([text('Three.')])
    )

    first = await agent.run('Count.')
    fake.transcripts.clear()
    second = await agent.run('Again.', message_history=first.all_messages())

    assert second.output == 'Two.'
    assert [forked_from(session) for session in fake.sessions] == snapshot(
        [(None, None), ('session-0', 'session-0/0'), (None, None)]
    )
    assert frame_shape(fake.sessions[2]) == snapshot(
        [('user', ['text'], False), ('assistant', ['text'], True), ('user', ['text'], True)]
    )
    # The fork point that failed is forgotten, and the new session's answer is the next one.
    third = await agent.run('Once more.', message_history=second.all_messages())
    assert third.output == 'Three.'
    assert forked_from(fake.sessions[3]) == ('session-2', 'session-2/0')


async def test_the_oldest_fork_points_make_room(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_claude_agent_sessions, '_MAX_FORK_POINTS', 1)
    agent, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]), FakeResponse([text('Two.')]))

    first = await agent.run('Count.')
    await agent.run('Again.', message_history=first.all_messages())
    # The second run's answer took the first's place, so the history up to the first is replayed.
    await agent.run('Again.', message_history=first.all_messages())

    assert [forked_from(session) for session in fake.sessions] == snapshot(
        [(None, None), ('session-0', 'session-0/0'), (None, None)]
    )
    # The same answer to the same history is remembered once, at its latest transcript entry.
    assert fork_points(agent) == [('session-2', 'session-2/0')]


async def test_an_answer_without_a_transcript_entry_isnt_a_fork_point():
    agent, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    fake.untracked = True

    first = await agent.run('Count.')
    await agent.run('Again.', message_history=first.all_messages())

    assert fork_points(agent) == []
    assert forked_from(fake.sessions[1]) == (None, None)


async def test_a_fork_that_exits_falls_back_to_the_replay():
    """A forked process that exits while it's sent the rest of the history gives way to one sent all of it."""
    agent, fake = claude_agent(FakeResponse([text('One.')]), FakeResponse([text('Two.')]))
    first = await agent.run('Count.')
    fake.exit_once_on_history = True

    second = await agent.run('Again.', message_history=with_exchange_after(first.all_messages()))

    assert second.output == 'Two.'
    assert [forked_from(session) for session in fake.sessions[1:]] == [('session-0', 'session-0/0'), (None, None)]
    # Only a missing transcript forgets the fork point.
    assert ('session-0', 'session-0/0') in fork_points(agent)


def with_exchange_after(messages: list[ModelMessage]) -> list[ModelMessage]:
    """The history with one more exchange after it, so a fork is sent history without querying the model."""
    return [*messages, ModelRequest.user_text_prompt('Next.'), ModelResponse(parts=[TextPart('Edited.')])]


async def test_a_turn_whose_answer_has_no_transcript_entry_isnt_a_fork_point():
    """The tool call's entry isn't where the turn's answer ends, so it isn't remembered for it."""
    agent, fake = claude_agent(
        FakeResponse([tool_use('toolu_1', 'get_city', {})], stop_reason='tool_use'),
        FakeResponse([text('Lisbon.')]),
    )

    @agent.tool_plain
    def get_city() -> str:
        fake.untracked = True
        return 'Lisbon'

    await agent.run('Where?')

    assert fork_points(agent) == []
