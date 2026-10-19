"""OpenAI Chat Completions can resume text or reasoning after another part in the same stream.

The resumed content must start a new part: the earlier part has already ended, and UI
adapters reject a delta on an ended part. These tests use synthetic chunks: text after a
tool call (#8208) did not reproduce on hosted providers, and reasoning after text (#8726)
came from vLLM's Gemma 4 reasoning parser.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
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
from pydantic_ai.profiles.openai import OpenAIModelProfile

from .._inline_snapshot import snapshot
from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage

    from .mock_openai import MockOpenAI
    from .test_openai import chunk, struc_chunk, text_chunk

with try_import() as ag_ui_imports_successful:
    from ag_ui.core import RunAgentInput, UserMessage

    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.ag_ui._utils import detect_ag_ui_version, parse_ag_ui_version

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]

_AG_UI_HAS_REASONING_EVENTS = ag_ui_imports_successful() and parse_ag_ui_version(detect_ag_ui_version()) >= (0, 1, 11)


def _reasoning_chunk(text: str) -> ChatCompletionChunk:
    # `reasoning_content` is not on the SDK's `ChoiceDelta`; DeepSeek, vLLM and others send it as an extra field.
    return chunk([ChoiceDelta.model_construct(role='assistant', reasoning_content=text)])


def _reasoning_text_reasoning_text_agent() -> Agent:
    """The #8726 shape: reasoning resumes after text, then text resumes after it."""
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
    """The #8208 shape: text, tool, then more content. That content must not be a delta on the ended text id."""
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
@pytest.mark.parametrize(
    'ag_ui_version, expected_events',
    [
        pytest.param(
            '0.1.10',
            snapshot(
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
            ),
            id='thinking-events',
        ),
        pytest.param(
            '0.1.11',
            snapshot(
                [
                    ('RUN_STARTED', None),
                    ('REASONING_START', None),
                    ('REASONING_MESSAGE_START', None),
                    ('REASONING_MESSAGE_CONTENT', 'Think.'),
                    ('REASONING_MESSAGE_END', None),
                    ('REASONING_ENCRYPTED_VALUE', None),
                    ('REASONING_END', None),
                    ('TEXT_MESSAGE_START', None),
                    ('TEXT_MESSAGE_CONTENT', 'Answer.'),
                    ('TEXT_MESSAGE_END', None),
                    ('REASONING_START', None),
                    ('REASONING_MESSAGE_START', None),
                    ('REASONING_MESSAGE_CONTENT', 'Again'),
                    ('REASONING_MESSAGE_CONTENT', '.'),
                    ('REASONING_MESSAGE_END', None),
                    ('REASONING_ENCRYPTED_VALUE', None),
                    ('REASONING_END', None),
                    ('TEXT_MESSAGE_START', None),
                    ('TEXT_MESSAGE_CONTENT', ' More'),
                    ('TEXT_MESSAGE_CONTENT', '.'),
                    ('TEXT_MESSAGE_END', None),
                    ('RUN_FINISHED', None),
                ]
            ),
            id='reasoning-events',
            marks=pytest.mark.skipif(not _AG_UI_HAS_REASONING_EVENTS, reason='requires ag-ui-protocol >= 0.1.11'),
        ),
    ],
)
async def test_resumed_reasoning_is_a_new_ag_ui_reasoning_message(
    allow_model_requests: None, ag_ui_version: str, expected_events: list[tuple[str, str | None]]
):
    """The #8726 shape: each burst is its own AG-UI message and the run ends with `RUN_FINISHED`.

    A thinking delta after its message ended would end the run with `RUN_ERROR` instead.
    """
    run_input = RunAgentInput(
        thread_id='thread',
        run_id='run',
        messages=[UserMessage(id='msg', content='Think twice.')],
        state={},
        context=[],
        tools=[],
        forwarded_props=None,
    )
    adapter = AGUIAdapter(
        agent=_reasoning_text_reasoning_text_agent(), run_input=run_input, ag_ui_version=ag_ui_version
    )
    events = [json.loads(raw.removeprefix('data: ')) async for raw in adapter.encode_stream(adapter.run_stream())]

    assert [(event['type'], event.get('delta')) for event in events] == expected_events


async def test_resumed_reasoning_is_not_a_vercel_delta_on_an_ended_part(allow_model_requests: None):
    """The #8726 shape: no reasoning or text delta may land on an ended id."""
    adapter = VercelAIAdapter(
        _reasoning_text_reasoning_text_agent(),
        SubmitMessage(id='foo', messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Think twice.')])]),
        sdk_version=6,
    )
    events = [
        '[DONE]' if '[DONE]' in raw else json.loads(raw.removeprefix('data: '))
        async for raw in adapter.encode_stream(adapter.run_stream())
    ]
    _assert_part_lifecycles(events)


@dataclass(frozen=True)
class ResumedContentCase:
    id: str
    stream: Callable[[], list[ChatCompletionChunk]]
    expected_parts: list[ModelResponsePart]
    profile: OpenAIModelProfile = field(default_factory=OpenAIModelProfile)


RESUMED_CONTENT_CASES = [
    ResumedContentCase(
        id='adjacent-reasoning-fields-keep-distinct-parts',
        stream=lambda: [
            _reasoning_chunk('Think.'),
            chunk([ChoiceDelta.model_construct(role='assistant', reasoning='Again')]),
            chunk([ChoiceDelta.model_construct(role='assistant', reasoning='.')]),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ThinkingPart(content='Again.', id='reasoning', provider_name='openai'),
            ]
        ),
    ),
    ResumedContentCase(
        id='reasoning-tool-reasoning',
        stream=lambda: [
            _reasoning_chunk('Think.'),
            struc_chunk('lookup', '{}'),
            _reasoning_chunk('Again'),
            _reasoning_chunk('.'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                ThinkingPart(content='Again.', id='reasoning_content', provider_name='openai'),
            ]
        ),
    ),
    ResumedContentCase(
        id='text-reasoning-text',
        stream=lambda: [
            text_chunk('Answer.'),
            _reasoning_chunk('Think.'),
            text_chunk(' More'),
            text_chunk('.'),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Answer.'),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                TextPart(content=' More.'),
            ]
        ),
    ),
    ResumedContentCase(
        id='custom-field-does-not-collide-with-a-resumed-reasoning-key',
        stream=lambda: [
            chunk([ChoiceDelta.model_validate({'role': 'assistant', 'reasoning-1': 'Think.'})]),
            text_chunk('Answer.'),
            chunk([ChoiceDelta.model_construct(role='assistant', reasoning='Again.')]),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                ThinkingPart(content='Think.', id='reasoning-1', provider_name='openai'),
                TextPart(content='Answer.'),
                ThinkingPart(content='Again.', id='reasoning', provider_name='openai'),
            ]
        ),
        profile=OpenAIModelProfile(openai_chat_thinking_field='reasoning-1'),
    ),
    ResumedContentCase(
        id='resumed-text-keeps-leading-whitespace-after-reasoning',
        stream=lambda: [
            _reasoning_chunk('Think.'),
            text_chunk('Answer.'),
            _reasoning_chunk('Again.'),
            text_chunk('\n\n'),
            text_chunk('More.'),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                TextPart(content='Answer.'),
                ThinkingPart(content='Again.', id='reasoning_content', provider_name='openai'),
                TextPart(content='\n\nMore.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='text-after-tool-drops-leading-whitespace',
        stream=lambda: [
            text_chunk('Checking now.'),
            struc_chunk('lookup', '{}'),
            text_chunk('\n\n'),
            text_chunk('Done.'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Checking now.'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                TextPart(content='Done.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='text-after-reasoning-then-tool-drops-leading-whitespace',
        stream=lambda: [
            text_chunk('Answer.'),
            _reasoning_chunk('Think.'),
            struc_chunk('lookup', '{}'),
            text_chunk('\n\n'),
            text_chunk('Done.'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Answer.'),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                TextPart(content='Done.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='text-after-tool-then-reasoning-drops-leading-whitespace',
        stream=lambda: [
            text_chunk('Checking now.'),
            struc_chunk('lookup', '{}'),
            _reasoning_chunk('Think.'),
            text_chunk('\n\n'),
            text_chunk('Done.'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Checking now.'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                TextPart(content='Done.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='trailing-whitespace-after-tool-is-dropped',
        stream=lambda: [
            text_chunk('Checking now.'),
            struc_chunk('lookup', '{}'),
            text_chunk('\n', finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Checking now.'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='whitespace-held-after-reasoning-carries-past-a-tool-call',
        stream=lambda: [
            text_chunk('Answer.'),
            _reasoning_chunk('Think.'),
            text_chunk('\n\n'),
            struc_chunk('lookup', '{}'),
            text_chunk('Done.'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Answer.'),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
                TextPart(content='\n\nDone.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='text-after-think-tags-following-reasoning-drops-leading-whitespace',
        stream=lambda: [
            text_chunk('Answer.'),
            _reasoning_chunk('Think.'),
            text_chunk('<think>'),
            text_chunk('More'),
            text_chunk('</think>'),
            text_chunk('\n\n'),
            text_chunk('Done.'),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Answer.'),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ThinkingPart(content='More', id='content', provider_name='openai'),
                TextPart(content='Done.'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='trailing-whitespace-after-reasoning-is-dropped',
        stream=lambda: [
            text_chunk('Answer.'),
            _reasoning_chunk('Think.'),
            text_chunk('\n'),
            chunk([ChoiceDelta()], finish_reason='stop'),
        ],
        expected_parts=snapshot(
            [
                TextPart(content='Answer.'),
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
    ResumedContentCase(
        id='leading-whitespace-before-tool-is-dropped',
        stream=lambda: [
            _reasoning_chunk('Think.'),
            text_chunk('\n\n'),
            struc_chunk('lookup', '{}'),
            chunk([ChoiceDelta()], finish_reason='tool_calls'),
        ],
        expected_parts=snapshot(
            [
                ThinkingPart(content='Think.', id='reasoning_content', provider_name='openai'),
                ToolCallPart(tool_name='lookup', args='{}', tool_call_id=IsStr()),
            ]
        ),
        profile=OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in RESUMED_CONTENT_CASES])
async def test_resumed_content_starts_a_new_part(allow_model_requests: None, case: ResumedContentCase):
    """Content resumed after another part gets its own part, never a delta after its part ended.

    `ignore_streamed_leading_whitespace` holds the whitespace-only start of text right after reasoning that interrupted
    a text part and prepends it to the next text part; it drops all other leading whitespace, and held whitespace that
    no text follows.
    """
    model = OpenAIChatModel(
        'test',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(case.stream())),
        profile=case.profile,
    )
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

    assert response.get().parts == case.expected_parts
