"""OpenAI Chat Completions can emit text after a tool call in the same stream.

Vercel AI SDK v6 rejects a `text-delta` for a part that has already ended. Hosted
providers did not reproduce this ordering, so these tests use synthetic chunks.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelResponse, ModelResponsePart, TextPart, ThinkingPart, ToolCallPart

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage

    from .mock_openai import MockOpenAI

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def _assert_text_part_lifecycle(events: list[Any]) -> None:
    """Vercel AI SDK v6: text-delta is invalid unless that id is currently open."""
    open_ids: set[str] = set()
    for event in events:
        if isinstance(event, str):
            continue
        kind = event.get('type')
        part_id = event.get('id')
        if kind == 'text-start':
            assert isinstance(part_id, str)
            assert part_id not in open_ids
            open_ids.add(part_id)
        elif kind == 'text-delta':
            assert part_id in open_ids, f'text-delta for missing or ended id {part_id!r}'
        elif kind == 'text-end':
            assert isinstance(part_id, str)
            assert part_id in open_ids
            open_ids.remove(part_id)
    assert not open_ids


def _chunk(delta: ChoiceDelta, finish_reason: Literal['stop', 'tool_calls'] | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='response',
        object='chat.completion.chunk',
        created=1,
        model='test',
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _tool_delta(index: int, arguments: str) -> ChoiceDelta:
    return ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=index,
                id=f'call-{index}',
                type='function',
                function=ChoiceDeltaToolCallFunction(name='lookup', arguments=arguments),
            )
        ]
    )


def _chunks(*deltas: ChoiceDelta, finish_reason: Literal['stop', 'tool_calls']) -> list[ChatCompletionChunk]:
    return [*(_chunk(delta) for delta in deltas), _chunk(ChoiceDelta(), finish_reason=finish_reason)]


def _lookup_agent(
    first_response: list[ChatCompletionChunk],
    *,
    profile: OpenAIModelProfile | None = None,
) -> Agent[None, str]:
    mock_client = MockOpenAI.create_mock_stream(
        [first_response, _chunks(ChoiceDelta(content='Done.'), finish_reason='stop')]
    )
    agent = Agent(OpenAIChatModel('test', provider=OpenAIProvider(openai_client=mock_client), profile=profile))

    @agent.tool_plain
    def lookup(value: int) -> int:
        return value

    return agent


async def _run_vercel(agent: Agent[None, str]) -> tuple[list[Any], Sequence[ModelResponsePart]]:
    request = SubmitMessage(
        id='foo',
        messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Look up values.')])],
    )
    adapter = VercelAIAdapter(agent, request, sdk_version=6)
    with capture_run_messages() as messages:
        events = [
            '[DONE]' if '[DONE]' in raw else json.loads(raw.removeprefix('data: '))
            async for raw in adapter.encode_stream(adapter.run_stream())
        ]
    return events, next(message.parts for message in messages if isinstance(message, ModelResponse))


async def test_text_then_tool_then_newline_keeps_vercel_text_lifecycle(allow_model_requests: None):
    """Reporter shape: text, tool, `\\n`, tool. The newline must not be a delta on the ended part."""
    events, parts = await _run_vercel(
        _lookup_agent(
            _chunks(
                ChoiceDelta(content='Checking now.'),
                _tool_delta(0, '{"value":1}'),
                ChoiceDelta(content='\n'),
                _tool_delta(1, '{"value":2}'),
                finish_reason='tool_calls',
            )
        )
    )

    _assert_text_part_lifecycle(events)
    assert parts == [
        TextPart('Checking now.'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart('\n'),
        ToolCallPart(tool_name='lookup', args='{"value":2}', tool_call_id='call-1'),
    ]


async def test_non_whitespace_text_after_tool_is_a_new_part(allow_model_requests: None):
    """A tool in the middle of the stream starts a new text part, not a continuation."""
    events, parts = await _run_vercel(
        _lookup_agent(
            _chunks(
                ChoiceDelta(content='Checking.'),
                _tool_delta(0, '{"value":1}'),
                ChoiceDelta(content=' More text.'),
                finish_reason='tool_calls',
            )
        )
    )

    _assert_text_part_lifecycle(events)
    assert parts == [
        TextPart('Checking.'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart(' More text.'),
    ]


async def test_closing_think_tag_after_tool_is_not_leaked_as_text(allow_model_requests: None):
    """Do not rotate `'content'` while it is still a `ThinkingPart`, or `</think>` becomes visible text."""
    events, parts = await _run_vercel(
        _lookup_agent(
            _chunks(
                ChoiceDelta(content='<think>'),
                ChoiceDelta(content='Checking'),
                _tool_delta(0, '{"value":1}'),
                ChoiceDelta(content='</think>'),
                ChoiceDelta(content=' Continued.'),
                finish_reason='tool_calls',
            ),
            profile=OpenAIModelProfile(thinking_tags=('<think>', '</think>')),
        )
    )

    _assert_text_part_lifecycle(events)
    assert parts == [
        ThinkingPart(content='Checking', id='content', provider_name='openai'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart(' Continued.'),
    ]
