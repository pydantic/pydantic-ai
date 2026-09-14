"""Tests for `Conversation`, the state a conversation carries between runs.

Unit tests: they pin what travels between runs and what a round-trip preserves, neither of which a
cassette matcher is sensitive to.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, TypeAdapter

from pydantic_ai import Agent, Conversation, RunUsage
from pydantic_ai.models.test import TestModel


def test_defaults() -> None:
    conversation = Conversation()

    assert conversation.messages == []
    assert conversation.usage == RunUsage()
    assert conversation.metadata is None
    assert UUID(conversation.conversation_id).version == 7
    assert Conversation().conversation_id != conversation.conversation_id


def test_run_result_conversation_carries_the_whole_bundle() -> None:
    result = Agent(TestModel(custom_output_text='hello'), instructions='Be helpful.').run_sync(
        'Say hello', metadata={'tenant': 'acme'}
    )
    conversation = result.conversation

    assert conversation.messages == result.all_messages()
    assert conversation.usage == result.usage
    assert conversation.conversation_id == result.conversation_id
    assert conversation.metadata == {'tenant': 'acme'}


def test_run_result_conversation_messages_are_a_copy() -> None:
    result = Agent(TestModel(), instructions='Be helpful.').run_sync('Say hello')
    conversation = result.conversation

    conversation.messages.clear()

    assert result.all_messages() != []


def test_round_trips_through_pydantic() -> None:
    result = Agent(TestModel(custom_output_text='stored'), instructions='Be helpful.').run_sync(
        'Say hello', metadata={'tenant': 'acme'}
    )
    result.usage.tool_calls = 3
    adapter = TypeAdapter(Conversation)

    reloaded = adapter.validate_json(adapter.dump_json(result.conversation))

    assert reloaded.messages == result.all_messages()
    assert reloaded.usage == result.usage
    assert reloaded.usage.tool_calls == 3
    assert reloaded.conversation_id == result.conversation_id
    assert reloaded.metadata == {'tenant': 'acme'}


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


def test_metadata_accepts_arbitrary_values() -> None:
    payload: dict[str, Any] = {'nested': {'a': [1, 2]}}
    conversation = Conversation(metadata=payload)

    adapter = TypeAdapter(Conversation)
    assert adapter.validate_json(adapter.dump_json(conversation)).metadata == payload
