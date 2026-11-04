"""Edits to a response part must reach a model that resolves item IDs to its stored copy.

The OpenAI Responses API stores a response's items by default, and when an item comes back with its
ID it uses the stored copy and ignores the content sent alongside. These are recorded tests because
that server behavior is the point: the input token count and the model's answer show what it read.
"""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import ClampOversizedMessages, ClearToolResults
from tests.conftest import RequestCapture
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY

from .test_compaction import _make_ctx  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.anyio, pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)]

LINES = '80 lines of the form `LINE <n>: <eight random lowercase English words>`, numbered from 1'
ASK = (
    'Look at your earlier output as it appears in this conversation, including any tool call arguments. '
    'Reply with only the largest N such that `LINE N:` is visible consecutively from 1, with no gap.'
)


def save(text: str) -> str:
    """Save the text."""
    return 'saved'


def _model(api_key: str, capture: RequestCapture) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-6-luna', provider=OpenAIProvider(api_key=api_key, http_client=capture.client))


def _input_shape(body: dict[str, Any]) -> list[tuple[str, bool]]:
    """Each input item's type, and whether it was sent with an item ID."""
    return [(item.get('type', 'easy'), 'id' in item) for item in body['input']]


@pytest.mark.vcr
async def test_clamped_text_reaches_the_model(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    model = _model(openai_api_key, request_capture)
    first = await Agent(model).run(f'Write exactly {LINES}. No preamble.')
    clamp = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=100, keep_tail_chars=100)
    second = await Agent(model, capabilities=[clamp]).run(ASK, message_history=first.all_messages())

    assert second.response.usage.input_tokens < first.response.usage.output_tokens / 2
    assert second.output == snapshot('2')
    # The reasoning item keeps its ID: the API accepts it before a message that has none.
    assert _input_shape(request_capture.body('/responses', 1)) == snapshot(
        [('easy', False), ('reasoning', True), ('easy', False), ('easy', False)]
    )


@pytest.mark.vcr
async def test_clamped_tool_call_args_reach_the_model(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    model = _model(openai_api_key, request_capture)
    first = await Agent(model, tools=[save]).run(
        f'Call `save` once, with `text` set to {LINES}, separated by newlines. '
        'Do not write the lines as text. After the tool returns, reply `OK`.'
    )
    clamp = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=100, keep_tail_chars=100)
    second = await Agent(model, tools=[save], capabilities=[clamp]).run(ASK, message_history=first.all_messages())

    assert second.response.usage.input_tokens < first.response.usage.input_tokens / 2
    assert second.output == snapshot('2')
    assert _input_shape(request_capture.body('/responses', 2)) == snapshot(
        [
            ('easy', False),
            ('reasoning', True),
            ('function_call', False),
            ('function_call_output', False),
            ('message', True),
            ('easy', False),
        ]
    )


@pytest.mark.vcr
async def test_cleared_tool_inputs_reach_the_model(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture
):
    model = _model(openai_api_key, request_capture)
    first = await Agent(model, tools=[save]).run(
        f'Call `save` once, with `text` set to {LINES}, separated by newlines. '
        'Do not write the lines as text. After the tool returns, reply `OK`.'
    )
    clear = ClearToolResults(keep_pairs=0, clear_tool_inputs=True, max_messages=1)
    second = await Agent(model, tools=[save], capabilities=[clear]).run(ASK, message_history=first.all_messages())

    assert second.response.usage.input_tokens < first.response.usage.input_tokens / 2
    assert second.output == snapshot('0')
    assert _input_shape(request_capture.body('/responses', 2)) == snapshot(
        [
            ('easy', False),
            ('reasoning', True),
            ('function_call', False),
            ('function_call_output', False),
            ('message', True),
            ('easy', False),
        ]
    )


async def test_only_edited_parts_lose_their_item_id():
    """Not a recorded test: no recording can pin that an untouched part stays replayable by ID."""
    kept_text = TextPart('short', id='msg_1', provider_name='openai')
    kept_call = ToolCallPart('t', '{"a": 1}', tool_call_id='c1', id='fc_1', provider_name='openai')
    clamped_text = TextPart('x' * 2_000, id='msg_2', provider_name='openai')
    clamped_call = ToolCallPart('t', 'y' * 2_000, tool_call_id='c2', id='fc_2', provider_name='openai')
    messages: list[ModelMessage] = [ModelResponse(parts=[kept_text, kept_call, clamped_text, clamped_call])]

    clamp = ClampOversizedMessages(max_part_chars=1_000, keep_head_chars=100, keep_tail_chars=100)
    result = await clamp.compact(messages, _make_ctx())

    parts = result[0].parts
    assert parts[0] is kept_text
    assert parts[1] is kept_call
    assert isinstance(parts[2], TextPart) and parts[2].id is None
    assert isinstance(parts[3], ToolCallPart) and parts[3].id is None and parts[3].tool_call_id == 'c2'
