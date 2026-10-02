"""Tests for `Conversation`, the state a conversation carries between runs.

Unit tests: they pin what travels between runs and what a round-trip preserves, neither of which a
cassette matcher is sensitive to.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from pydantic import BaseModel, SerializationInfo, TypeAdapter, field_serializer

from pydantic_ai import Agent, Conversation, ConversationTypeAdapter, RunUsage
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests


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


def test_run_result_conversation_usage_is_a_copy() -> None:
    """The bundle is a branch point, so spending it must not rewrite what the run already spent."""
    result = Agent(TestModel(), instructions='Be helpful.').run_sync('Say hello')
    conversation = result.conversation

    conversation.usage.requests += 10
    conversation.usage.details['branch'] = 1

    assert result.usage.requests == 1
    assert 'branch' not in result.usage.details


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


def _refund_agent() -> Agent[None, str | DeferredToolRequests]:
    """An agent whose first run pauses on one call needing approval and one executed elsewhere."""

    def llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('refund', {'amount': 10}, tool_call_id='refund-1'),
                    ToolCallPart('look_up_order', {}, tool_call_id='lookup-1'),
                ]
            )
        return ModelResponse(parts=[TextPart('Refunded order 42.')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def refund(amount: int) -> str:
        return f'refunded {amount}'

    @agent.tool_plain
    def look_up_order() -> str:
        raise CallDeferred(metadata={'queue': 'orders'})

    return agent


def test_a_paused_conversation_carries_what_it_is_waiting_on() -> None:
    """The requests travel with the conversation: the messages can't say which answer each call needs.

    Stored and reloaded, the conversation still knows the refund wants approval and the lookup
    wants an external result with its metadata, so it can be resumed from storage alone.
    """
    agent = _refund_agent()
    paused = agent.run_sync('Refund my order.')
    assert isinstance(paused.output, DeferredToolRequests)

    stored = ConversationTypeAdapter.dump_json(paused.conversation)
    conversation = ConversationTypeAdapter.validate_json(stored)

    requests = conversation.deferred_tool_requests
    assert requests is not None
    assert requests == paused.output
    assert [call.tool_call_id for call in requests.approvals] == ['refund-1']
    assert [call.tool_call_id for call in requests.calls] == ['lookup-1']
    assert requests.metadata == {'lookup-1': {'queue': 'orders'}}

    results = requests.build_results(approve_all=True, calls={'lookup-1': 'order 42'})
    resumed = agent.run_sync(conversation=conversation, deferred_tool_results=results)

    assert resumed.output == 'Refunded order 42.'
    assert resumed.conversation.deferred_tool_requests is None


def test_a_conversation_s_requests_are_its_own() -> None:
    agent = _refund_agent()
    paused = agent.run_sync('Refund my order.')

    conversation = paused.conversation
    assert conversation.deferred_tool_requests is not None
    conversation.deferred_tool_requests.approvals.clear()

    assert isinstance(paused.output, DeferredToolRequests)
    assert [call.tool_call_id for call in paused.output.approvals] == ['refund-1']


def test_serializes_with_the_fidelity_of_the_messages_adapter() -> None:
    """Every way of storing a conversation keeps what `ModelMessagesTypeAdapter` keeps.

    A tool's raw `bytes` return lives in an `Any`-typed field that only an outermost adapter's
    `ser_json_bytes` reaches, so before the conversation routed its messages through that adapter,
    dumping it to JSON failed outright on non-UTF-8 bytes, nested in a model of the caller's or not.
    """
    image = BinaryContent(data=bytes([0x89, 0xFF, 0x00, 0x10]), media_type='image/png')
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart('read_file', bytes([0xFF, 0xFE]), tool_call_id='c1'),
                UserPromptPart(content=['What is in this image?', image]),
            ]
        )
    ]
    expected = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    conversation = Conversation(messages=messages)

    class Thread(BaseModel):
        conversation: Conversation

    thread = Thread(conversation=conversation)

    assert ConversationTypeAdapter.validate_json(ConversationTypeAdapter.dump_json(conversation)).messages == expected
    assert Thread.model_validate_json(thread.model_dump_json()).conversation.messages == expected
    assert Thread.model_validate(thread.model_dump(mode='json')).conversation.messages == expected


def test_serialization_honors_the_caller_s_dump_settings() -> None:
    """A redaction asked for is a redaction applied, on its own or nested in a model of the caller's.

    The messages are dumped through `ModelMessagesTypeAdapter`, and a plain serializer's return isn't
    shaped by the outer dump's settings, so they have to be passed on: dropping `exclude` would send
    the metadata a server meant to keep from its client, and dropping `context` would stop a
    context-aware serializer inside it from redacting itself.
    """
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('Hi')], metadata={'api_key': 'secret'})]
    conversation = Conversation(messages=messages)
    redact_metadata = {'messages': {'__all__': {'metadata'}}}

    class Thread(BaseModel):
        conversation: Conversation

    assert b'secret' not in ConversationTypeAdapter.dump_json(conversation, exclude=redact_metadata)
    assert 'secret' not in Thread(conversation=conversation).model_dump_json(exclude={'conversation': redact_metadata})

    class Token(BaseModel):
        value: str

        @field_serializer('value')
        def _redact(self, value: str, info: SerializationInfo) -> str:
            context: dict[str, bool] = info.context or {}
            return '***' if context.get('redact') else value

    with_token = Conversation(
        messages=[ModelRequest(parts=[UserPromptPart('Hi')], metadata={'token': Token(value='secret')})]
    )
    assert b'secret' not in ConversationTypeAdapter.dump_json(with_token, context={'redact': True})

    dumped = json.loads(ConversationTypeAdapter.dump_json(conversation, exclude_none=True, exclude_defaults=True))
    assert dumped['messages'] == json.loads(
        ModelMessagesTypeAdapter.dump_json(messages, exclude_none=True, exclude_defaults=True)
    )
