"""Compaction lifecycle events from stateless `OpenAICompaction`."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CompactionEndEvent, CompactionStartEvent, on_event
from pydantic_ai.messages import AgentStreamEvent, CompactionPart, ModelMessage, ModelResponse
from pydantic_ai.tools import RunContext
from tests.conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


async def test_stateless_compaction_emits_lifecycle_events(allow_model_requests: None, openai_api_key: str):
    """Stateless `OpenAICompaction` emits an immediately dispatched `CompactionStartEvent` and a `CompactionEndEvent` around `/compact`."""
    model = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    compaction = OpenAICompaction(message_count_threshold=4)
    compaction.id = 'openai_compaction'
    agent = Agent(
        model=model,
        instructions='You are a helpful math assistant. Give short answers.',
        capabilities=[compaction],
    )

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    message_history: list[ModelMessage] = []
    result = await agent.run('What is 2+2?', message_history=message_history, event_stream_handler=handler)
    message_history = result.all_messages()
    result = await agent.run('And 3+3?', message_history=message_history, event_stream_handler=handler)
    message_history = result.all_messages()
    # Third run exceeds the threshold and compacts
    result = await agent.run('And 5+5?', message_history=message_history, event_stream_handler=handler)

    assert any(
        isinstance(part, CompactionPart)
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
    )
    compaction_events = [event for event in events if isinstance(event, CompactionStartEvent | CompactionEndEvent)]
    assert compaction_events == snapshot(
        [
            CompactionStartEvent(capability_id='openai_compaction', strategy='openai', messages_before=4),
            CompactionEndEvent(
                capability_id='openai_compaction', strategy='openai', messages_before=4, messages_after=1
            ),
        ]
    )


async def test_cancelled_start_event_skips_compact(allow_model_requests: None, openai_api_key: str):
    """A listener cancelling `CompactionStartEvent` makes stateless `OpenAICompaction` skip the `/compact` call.

    The cassette holds only regular `/responses` exchanges, so a regression that still calls
    `/compact` fails to find a matching recording.
    """

    class HoldCompaction(AbstractCapability[Any]):
        @on_event(CompactionStartEvent)
        async def _hold(self, ctx: RunContext[Any], event: CompactionStartEvent) -> None:
            event.cancel('history must stay intact')

    model = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    compaction = OpenAICompaction(message_count_threshold=4)
    compaction.id = 'openai_compaction'
    agent = Agent(
        model=model,
        instructions='You are a helpful math assistant. Give short answers.',
        capabilities=[compaction, HoldCompaction()],
    )

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    message_history: list[ModelMessage] = []
    result = await agent.run('What is 2+2?', message_history=message_history, event_stream_handler=handler)
    message_history = result.all_messages()
    result = await agent.run('And 3+3?', message_history=message_history, event_stream_handler=handler)
    message_history = result.all_messages()
    result = await agent.run('And 5+5?', message_history=message_history, event_stream_handler=handler)

    all_msgs = result.all_messages()
    assert not any(
        isinstance(part, CompactionPart) for msg in all_msgs if isinstance(msg, ModelResponse) for part in msg.parts
    )
    assert len(all_msgs) == 6
    compaction_events = [event for event in events if isinstance(event, CompactionStartEvent | CompactionEndEvent)]
    assert compaction_events == snapshot(
        [
            CompactionStartEvent(
                capability_id='openai_compaction',
                strategy='openai',
                messages_before=4,
                cancelled=True,
                cancel_reason='history must stay intact',
            )
        ]
    )
