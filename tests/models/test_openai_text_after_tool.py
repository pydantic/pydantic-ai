"""OpenAI Chat Completions can emit text after a tool call in the same stream.

Vercel AI SDK v6 rejects a `text-delta` for a part that has already ended. Hosted
providers did not reproduce this ordering, so these tests use synthetic chunks.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, TextPart, ThinkingPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage

    from .mock_openai import MockOpenAI
    from .test_openai import struc_chunk, text_chunk

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
    _assert_text_part_lifecycle(events)


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
