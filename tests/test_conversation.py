"""Tests for `Conversation`, the state a conversation carries between runs.

Unit tests: they pin what travels between runs and what a round-trip preserves, neither of which a
cassette matcher is sensitive to.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import Agent, Conversation, RunUsage
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel


def test_defaults() -> None:
    conversation = Conversation()

    assert conversation.messages == []
    assert conversation.usage == RunUsage()
    assert UUID(conversation.conversation_id).version == 7
    assert Conversation().conversation_id != conversation.conversation_id


def test_run_result_conversation_carries_the_whole_bundle() -> None:
    result = Agent(TestModel(custom_output_text='hello'), instructions='Be helpful.').run_sync('Say hello')
    conversation = result.conversation

    assert conversation.messages == result.all_messages()
    assert conversation.usage == result.usage
    assert conversation.conversation_id == result.conversation_id


def test_run_result_conversation_messages_are_a_copy() -> None:
    result = Agent(TestModel(), instructions='Be helpful.').run_sync('Say hello')
    conversation = result.conversation

    conversation.messages.clear()

    assert result.all_messages() != []


def test_round_trips_through_pydantic() -> None:
    result = Agent(TestModel(custom_output_text='stored'), instructions='Be helpful.').run_sync('Say hello')
    result.usage.tool_calls = 3
    adapter = TypeAdapter(Conversation)

    reloaded = adapter.validate_json(adapter.dump_json(result.conversation))

    assert reloaded.messages == result.all_messages()
    assert reloaded.usage == result.usage
    assert reloaded.usage.tool_calls == 3
    assert reloaded.conversation_id == result.conversation_id


def test_usable_as_a_field_on_a_model() -> None:
    class Thread(BaseModel):
        owner: str
        conversation: Conversation

    result = Agent(TestModel(), instructions='Be helpful.').run_sync('Say hello')
    thread = Thread(owner='acme', conversation=result.conversation)

    reloaded = Thread.model_validate_json(thread.model_dump_json())

    assert reloaded.owner == 'acme'
    assert reloaded.conversation.messages == result.all_messages()


def test_carrying_usage_keeps_a_conversation_total() -> None:
    """The reason `usage` is on the bundle: `message_history` alone restarts the count each run."""
    agent = Agent(TestModel(), instructions='Be helpful.')

    first = agent.run_sync('One')
    without_usage = agent.run_sync('Two', message_history=first.conversation.messages)
    with_usage = agent.run_sync(
        'Two',
        message_history=first.conversation.messages,
        usage=first.conversation.usage,
    )

    assert without_usage.usage.requests == 1
    assert with_usage.usage.requests == 2
    assert with_usage.usage.input_tokens > without_usage.usage.input_tokens


def test_run_sync_continues_a_conversation() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.')
    conversation = agent.run_sync('One').conversation

    result = agent.run_sync('Two', conversation=conversation)

    assert result.all_messages()[: len(conversation.messages)] == conversation.messages
    assert len(result.all_messages()) > len(conversation.messages)
    assert result.conversation.usage.requests == 2
    assert result.conversation_id == conversation.conversation_id


@pytest.mark.anyio
async def test_run_continues_a_conversation() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.')
    conversation = (await agent.run('One')).conversation

    result = await agent.run('Two', conversation=conversation)

    assert result.all_messages()[: len(conversation.messages)] == conversation.messages
    assert len(result.all_messages()) > len(conversation.messages)
    assert result.conversation.usage.requests == 2
    assert result.conversation_id == conversation.conversation_id


@pytest.mark.anyio
async def test_run_stream_continues_a_conversation() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.')
    conversation = (await agent.run('One')).conversation

    async with agent.run_stream('Two', conversation=conversation) as result:
        await result.get_output()

        assert result.all_messages()[: len(conversation.messages)] == conversation.messages
        assert len(result.all_messages()) > len(conversation.messages)
        assert result.usage.requests == 2
        assert result.conversation_id == conversation.conversation_id


@pytest.mark.anyio
async def test_iter_continues_a_conversation() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.')
    conversation = (await agent.run('One')).conversation

    async with agent.iter('Two', conversation=conversation) as agent_run:
        async for _ in agent_run:
            pass

    assert agent_run.result is not None
    assert agent_run.result.all_messages()[: len(conversation.messages)] == conversation.messages
    assert len(agent_run.result.all_messages()) > len(conversation.messages)
    assert agent_run.result.conversation.usage.requests == 2
    assert agent_run.result.conversation_id == conversation.conversation_id


@pytest.mark.parametrize(
    ('message_history', 'usage', 'conversation_id', 'conflicting_argument'),
    [
        ([], None, None, 'message_history'),
        (None, RunUsage(), None, 'usage'),
        (None, None, 'other-conversation', 'conversation_id'),
    ],
)
def test_conversation_rejects_separate_arguments(
    message_history: list[ModelMessage] | None,
    usage: RunUsage | None,
    conversation_id: str | None,
    conflicting_argument: str,
) -> None:
    agent = Agent(TestModel())

    with pytest.raises(UserError, match=rf'`{conflicting_argument}`'):
        agent.run_sync(
            'Two',
            conversation=Conversation(),
            message_history=message_history,
            usage=usage,
            conversation_id=conversation_id,
        )


def test_conversation_is_a_branch_point() -> None:
    agent = Agent(TestModel(), instructions='Be helpful.')
    conversation = agent.run_sync('Root').conversation
    original_requests = conversation.usage.requests
    original_message_count = len(conversation.messages)

    first_branch = agent.run_sync('First branch', conversation=conversation)
    second_branch = agent.run_sync('Second branch', conversation=conversation)

    # A conversation is usable as a branch point only when starting a run leaves it unchanged.
    assert conversation.usage.requests == original_requests
    assert len(conversation.messages) == original_message_count
    assert first_branch.all_messages()[:original_message_count] == conversation.messages
    assert second_branch.all_messages()[:original_message_count] == conversation.messages
    assert first_branch.new_messages() != second_branch.new_messages()
    assert first_branch.usage.requests == second_branch.usage.requests == original_requests + 1
