"""OpenAI Chat Completions can resume text or reasoning after another part in the same stream.

The resumed content must start a new part: the earlier part has already ended, and UI
adapters reject a delta on an ended part. Hosted providers did not reproduce these
orderings, so these tests use synthetic chunks.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from .._inline_snapshot import snapshot
from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage

    from .mock_openai import MockOpenAI
    from .test_openai import chunk, struc_chunk, text_chunk

with try_import() as ag_ui_imports_successful:
    from ag_ui.core import RunAgentInput, UserMessage

    from pydantic_ai.ui.ag_ui import AGUIAdapter

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def _reasoning_chunk(text: str) -> ChatCompletionChunk:
    # `reasoning_content` is not on the SDK's `ChoiceDelta`; DeepSeek, vLLM and others send it as an extra field.
    return chunk([ChoiceDelta.model_construct(role='assistant', reasoning_content=text)])


def _reasoning_text_reasoning_text_agent() -> Agent:
    """The reporter's shape: reasoning resumes after text, then text resumes after it."""
    stream = [
        _reasoning_chunk('Think.'),
        text_chunk('Answer.'),
        _reasoning_chunk('Again'),
        _reasoning_chunk('.'),
        text_chunk(' More'),
        text_chunk('.'),
        chunk([ChoiceDelta()], finish_reason='stop'),
    ]
    return Agent(OpenAIChatModel('test', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream))))


def _assert_part_lifecycles(events: list[Any]) -> None:
    """Vercel AI SDK v6: a text or reasoning delta is invalid unless that id is currently open for its kind."""
    open_ids: dict[str, set[str]] = {'text': set(), 'reasoning': set()}
    for event in events:
        if isinstance(event, str):
            continue
        kind, _, stage = event['type'].partition('-')
        if kind not in open_ids:
            continue
        part_id = event['id']
        if stage == 'start':
            assert part_id not in open_ids[kind]
            open_ids[kind].add(part_id)
        elif stage == 'delta':
            assert part_id in open_ids[kind], f'{kind}-delta for missing or ended id {part_id!r}'
        else:
            assert stage == 'end'
            open_ids[kind].remove(part_id)
    assert open_ids == {'text': set(), 'reasoning': set()}


async def test_text_after_tool_is_not_a_vercel_delta_on_the_ended_part(allow_model_requests: None):
    """Reporter shape: text, tool, then more content. That content must not be a delta on the ended text id."""
    agent = Agent(
        OpenAIChatModel(
            'test',
            provider=OpenAIProvider(
                openai_client=MockOpenAI.create_mock_stream(
                    [
                        [
                            text_chunk('Checking now.'),
                            struc_chunk('lookup', '{"value":1}'),
                            text_chunk('\n', finish_reason='tool_calls'),
                        ],
                        [text_chunk('Done.', finish_reason='stop')],
                    ]
                )
            ),
        )
    )

    @agent.tool_plain
    def lookup(value: int) -> int:
        return value

    adapter = VercelAIAdapter(
        agent,
        SubmitMessage(id='foo', messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Look up.')])]),
        sdk_version=6,
    )
    events = [
        '[DONE]' if '[DONE]' in raw else json.loads(raw.removeprefix('data: '))
        async for raw in adapter.encode_stream(adapter.run_stream())
    ]
    _assert_part_lifecycles(events)


async def test_closing_think_tag_after_tool_is_not_leaked_as_text(allow_model_requests: None):
    """Do not rotate `'content'` while it is still a `ThinkingPart`, or `</think>` becomes visible text."""
    model = OpenAIChatModel(
        'test',
        provider=OpenAIProvider(
            openai_client=MockOpenAI.create_mock_stream(
                [
                    text_chunk('<think>'),
                    text_chunk('Checking'),
                    struc_chunk('lookup', '{"value":1}'),
                    text_chunk('</think>'),
                    text_chunk(' Continued.', finish_reason='tool_calls'),
                ]
            )
        ),
        profile=OpenAIModelProfile(thinking_tags=('<think>', '</think>')),
    )
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart('Look up.')])],
        None,
        ModelRequestParameters(),
    ) as response:
        async for _ in response:
            pass

    assert response.get().parts == [
        ThinkingPart(content='Checking', id='content', provider_name='openai'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id=IsStr()),
        TextPart(' Continued.'),
    ]


@pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed')
async def test_resumed_reasoning_is_a_new_ag_ui_reasoning_message(allow_model_requests: None):
    """Reporter shape: reasoning, text, reasoning, text. Each burst is its own AG-UI message and the run finishes.

    Resumed reasoning used to be a delta on the ended thinking part, and the adapter ended the run with `RUN_ERROR`.
    """
    agent = _reasoning_text_reasoning_text_agent()
    run_input = RunAgentInput(
        thread_id='thread',
        run_id='run',
        messages=[UserMessage(id='msg', content='Think twice.')],
        state={},
        context=[],
        tools=[],
        forwarded_props=None,
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input, ag_ui_version='0.1.10')
    events = [json.loads(raw.removeprefix('data: ')) async for raw in adapter.encode_stream(adapter.run_stream())]

    assert [(event['type'], event.get('delta')) for event in events] == snapshot(
        [
            ('RUN_STARTED', None),
            ('THINKING_START', None),
            ('THINKING_TEXT_MESSAGE_START', None),
            ('THINKING_TEXT_MESSAGE_CONTENT', 'Think.'),
            ('THINKING_TEXT_MESSAGE_END', None),
            ('THINKING_END', None),
            ('TEXT_MESSAGE_START', None),
            ('TEXT_MESSAGE_CONTENT', 'Answer.'),
            ('TEXT_MESSAGE_END', None),
            ('THINKING_START', None),
            ('THINKING_TEXT_MESSAGE_START', None),
            ('THINKING_TEXT_MESSAGE_CONTENT', 'Again'),
            ('THINKING_TEXT_MESSAGE_CONTENT', '.'),
            ('THINKING_TEXT_MESSAGE_END', None),
            ('THINKING_END', None),
            ('TEXT_MESSAGE_START', None),
            ('TEXT_MESSAGE_CONTENT', ' More'),
            ('TEXT_MESSAGE_CONTENT', '.'),
            ('TEXT_MESSAGE_END', None),
            ('RUN_FINISHED', None),
        ]
    )


async def test_resumed_reasoning_is_not_a_vercel_delta_on_an_ended_part(allow_model_requests: None):
    """Reporter shape: reasoning, text, reasoning, text. No reasoning or text delta may land on an ended id."""
    agent = _reasoning_text_reasoning_text_agent()
    adapter = VercelAIAdapter(
        agent,
        SubmitMessage(id='foo', messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Think twice.')])]),
        sdk_version=6,
    )
    events = [
        '[DONE]' if '[DONE]' in raw else json.loads(raw.removeprefix('data: '))
        async for raw in adapter.encode_stream(adapter.run_stream())
    ]
    _assert_part_lifecycles(events)


@pytest.mark.parametrize(
    'stream, expected_parts',
    [
        pytest.param(
            lambda: [
                _reasoning_chunk('Think.'),
                struc_chunk('lookup', '{}'),
                _reasoning_chunk('Again'),
                _reasoning_chunk('.'),
                chunk([ChoiceDelta()], finish_reason='tool_calls'),
            ],
            snapshot(
                [
                    ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                    ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                    ThinkingPart(content='Again.', id='reasoning_content', provider_name='openai'),
                ]
            ),
            id='reasoning-tool-reasoning',
        ),
        pytest.param(
            lambda: [
                text_chunk('Answer.'),
                _reasoning_chunk('Think.'),
                text_chunk(' More'),
                text_chunk('.'),
                chunk([ChoiceDelta()], finish_reason='stop'),
            ],
            snapshot(
                [
                    TextPart(content='Answer.'),
                    ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                    TextPart(content=' More.'),
                ]
            ),
            id='text-reasoning-text',
        ),
    ],
)
async def test_resumed_content_starts_a_new_part(
    allow_model_requests: None,
    stream: Callable[[], list[ChatCompletionChunk]],
    expected_parts: list[ModelResponsePart],
):
    """Content resumed after another part started gets its own part, never a delta after its part ended."""
    model = OpenAIChatModel('test', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream())))
    ended: set[int] = set()
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart('Think twice.')])],
        None,
        ModelRequestParameters(),
    ) as response:
        async for event in response:
            if isinstance(event, PartEndEvent):
                ended.add(event.index)
            elif isinstance(event, PartDeltaEvent):
                assert event.index not in ended, f'delta on ended part {event.index}'

    assert response.get().parts == expected_parts
