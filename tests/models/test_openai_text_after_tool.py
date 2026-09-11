"""Vercel AI SDK v6 rejects a `text-delta` for a part that has already ended.

OpenAI Chat Completions can emit `delta.content` after `tool_calls` has started in the
same SSE response. `OpenAIStreamedResponse` used a constant `vendor_part_id='content'`,
so that later text was applied to the already-ended part. The adapter then reused
`self.message_id` and the frontend raised `Received text-delta for missing text part`.

Hosted Groq/Mistral/Qwen did not reproduce this intra-response ordering, so these tests
drive `OpenAIChatModel` with synthetic chunks through `Agent` + `VercelAIAdapter` rather
than a VCR cassette. The contract is the UI part lifecycle, not internal part indices.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from typing import Literal, cast

import pytest

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart

from ..conftest import try_import

UIEvent = dict[str, object]

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

FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']

_UI_PART_KINDS = {
    'text': ('text-start', 'text-delta', 'text-end'),
    'reasoning': ('reasoning-start', 'reasoning-delta', 'reasoning-end'),
}


def _event_str(event: UIEvent, key: str) -> str | None:
    value = event.get(key)
    return value if isinstance(value, str) else None


def _assert_ui_part_lifecycle(events: Sequence[UIEvent | str]) -> None:
    """Vercel AI SDK v6: start opens an id, delta requires it open, end closes it."""
    for kind, (start, delta, end) in _UI_PART_KINDS.items():
        open_ids: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = _event_str(event, 'type')
            part_id = _event_str(event, 'id')
            if event_type == start:
                assert part_id is not None
                assert part_id not in open_ids
                open_ids.add(part_id)
            elif event_type == delta:
                assert part_id in open_ids, f'{kind}-delta for missing or ended id {part_id!r}'
            elif event_type == end:
                assert part_id is not None
                assert part_id in open_ids
                open_ids.remove(part_id)
        assert not open_ids, f'{kind} parts still open: {open_ids}'


def _text_start_ids(events: Sequence[UIEvent | str]) -> list[str]:
    ids: list[str] = []
    for event in events:
        if isinstance(event, dict) and _event_str(event, 'type') == 'text-start':
            part_id = _event_str(event, 'id')
            assert part_id is not None
            ids.append(part_id)
    return ids


def _first_step_events(events: Sequence[UIEvent | str]) -> list[UIEvent | str]:
    first_step: list[UIEvent | str] = []
    for event in events:
        first_step.append(event)
        if isinstance(event, dict) and _event_str(event, 'type') == 'finish-step':
            break
    return first_step


def _first_model_response(messages: Sequence[ModelMessage]) -> ModelResponse:
    return next(message for message in messages if isinstance(message, ModelResponse))


def chunk(delta: ChoiceDelta, finish_reason: FinishReason | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='response',
        object='chat.completion.chunk',
        created=1,
        model='test',
        choices=[Choice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def tool_delta(index: int, arguments: str, *, start: bool = False) -> ChoiceDelta:
    return ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=index,
                id=f'call-{index}' if start else None,
                type='function' if start else None,
                function=ChoiceDeltaToolCallFunction(name='lookup' if start else None, arguments=arguments),
            )
        ]
    )


def _chunks(*deltas: ChoiceDelta, finish_reason: FinishReason) -> list[ChatCompletionChunk]:
    return [*(chunk(delta) for delta in deltas), chunk(ChoiceDelta(), finish_reason=finish_reason)]


def _lookup_agent(
    first_response: list[ChatCompletionChunk],
    *,
    profile: OpenAIModelProfile | None = None,
) -> Agent[None, str]:
    mock_client = MockOpenAI.create_mock_stream(
        [
            first_response,
            _chunks(ChoiceDelta(content='Done.'), finish_reason='stop'),
        ]
    )
    agent = Agent(
        OpenAIChatModel(
            'test',
            provider=OpenAIProvider(openai_client=mock_client),
            profile=profile,
        )
    )

    @agent.tool_plain
    def lookup(value: int) -> int:
        return value

    return agent


def _decode_sse(raw: str) -> UIEvent | str:
    if '[DONE]' in raw:
        return '[DONE]'
    payload = json.loads(raw.removeprefix('data: '))
    assert isinstance(payload, dict)
    return cast(UIEvent, payload)  # SSE JSON objects are string-keyed dicts


async def _run_vercel(agent: Agent[None, str]) -> tuple[list[UIEvent | str], list[ModelMessage]]:
    request = SubmitMessage(
        id='foo',
        messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Look up values.')])],
    )
    adapter = VercelAIAdapter(agent, request, sdk_version=6)
    with capture_run_messages() as messages:
        events = [_decode_sse(raw) async for raw in adapter.encode_stream(adapter.run_stream())]
    return events, list(messages)


async def test_text_then_tool_then_newline_keeps_vercel_text_lifecycle(allow_model_requests: None):
    """Reporter shape: text, tool 0, `\\n`, tool 1. The newline must not be a delta on the ended part."""
    agent = _lookup_agent(
        _chunks(
            ChoiceDelta(content='Checking now.'),
            tool_delta(0, '{', start=True),
            tool_delta(0, '"value":'),
            tool_delta(0, '1}'),
            ChoiceDelta(content='\n'),
            tool_delta(1, '{', start=True),
            tool_delta(1, '"value":'),
            tool_delta(1, '2}'),
            finish_reason='tool_calls',
        )
    )
    events, messages = await _run_vercel(agent)

    _assert_ui_part_lifecycle(events)
    first_step_ids = _text_start_ids(_first_step_events(events))
    assert len(first_step_ids) == 2
    assert len(set(first_step_ids)) == 2
    assert _first_model_response(messages).parts == [
        TextPart('Checking now.'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart('\n'),
        ToolCallPart(tool_name='lookup', args='{"value":2}', tool_call_id='call-1'),
    ]


async def test_non_whitespace_text_after_tool_is_a_new_vercel_part(allow_model_requests: None):
    """A tool in the middle of a Chat Completions stream starts a new text part, not a continuation."""
    agent = _lookup_agent(
        _chunks(
            ChoiceDelta(content='Checking.'),
            tool_delta(0, '{', start=True),
            tool_delta(0, '"value":'),
            tool_delta(0, '1}'),
            ChoiceDelta(content=' More text.'),
            finish_reason='tool_calls',
        )
    )
    events, messages = await _run_vercel(agent)

    _assert_ui_part_lifecycle(events)
    first_step_ids = _text_start_ids(_first_step_events(events))
    assert len(first_step_ids) == 2
    assert first_step_ids[0] != first_step_ids[1]
    assert _first_model_response(messages).parts == [
        TextPart('Checking.'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart(' More text.'),
    ]


async def test_closing_think_tag_after_tool_is_not_leaked_as_text(allow_model_requests: None):
    """Rotation must not run while `'content'` is still a `ThinkingPart`, or `</think>` becomes visible text."""
    agent = _lookup_agent(
        _chunks(
            ChoiceDelta(content='<think>'),
            ChoiceDelta(content='Checking'),
            tool_delta(0, '{', start=True),
            tool_delta(0, '"value":'),
            tool_delta(0, '1}'),
            ChoiceDelta(content='</think>'),
            ChoiceDelta(content=' Continued.'),
            finish_reason='tool_calls',
        ),
        profile=OpenAIModelProfile(thinking_tags=('<think>', '</think>')),
    )
    events, messages = await _run_vercel(agent)

    _assert_ui_part_lifecycle(events)
    assert _first_model_response(messages).parts == [
        ThinkingPart(content='Checking', id='content', provider_name='openai'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart(' Continued.'),
    ]
