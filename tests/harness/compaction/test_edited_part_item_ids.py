"""Edits to a response part must reach a model that resolves item IDs to its stored copy.

The OpenAI Responses API stores a response's items by default, and when an item comes back with its
ID it uses the stored copy and ignores the content sent alongside. These are recorded tests because
that server behavior is the point: the input token count and the model's answer show what it read.

The property tests draw histories with `tests/history_strategies.py` instead, and check that no part either capability
edits goes out with its item ID, in the body `tests/history_mappers.py` maps for OpenAI Responses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from hypothesis import given
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    BaseToolCallPart,
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequestPart,
    ModelResponsePart,
    SpeechPart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.compaction import ClampOversizedMessages, ClearToolResults
from tests.conftest import RequestCapture
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY
from tests.history_invariants import stale_reference_violations
from tests.history_mappers import BODY_CASES
from tests.history_strategies import IGNORE_RANDOM_STATE_NOTE, PROPERTY_SETTINGS, histories

pytestmark = [pytest.mark.vcr, pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)]

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
    """Each input item's type, or role for a message without one, and whether it was sent with an item ID."""
    return [(item.get('type', item.get('role')), 'id' in item) for item in body['input']]


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
        [('user', False), ('reasoning', True), ('assistant', False), ('user', False)]
    )


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
            ('user', False),
            ('reasoning', True),
            ('function_call', False),
            ('function_call_output', False),
            ('message', True),
            ('user', False),
        ]
    )


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
            ('user', False),
            ('reasoning', True),
            ('function_call', False),
            ('function_call_output', False),
            ('message', True),
            ('user', False),
        ]
    )


_RESPONSES = next(case for case in BODY_CASES if case.id == 'openai-responses')

Edit = Callable[[list[ModelMessage], RunContext[None]], Coroutine[object, object, list[ModelMessage]]]


def _item_id(part: ModelRequestPart | ModelResponsePart) -> str | None:
    """The provider item ID a part carries, if its kind has one."""
    if isinstance(part, TextPart | ThinkingPart | BaseToolCallPart | FilePart | CompactionPart | SpeechPart):
        return part.id
    return None


def _edited_item_ids(before: list[ModelMessage], after: list[ModelMessage]) -> set[str]:
    """The item IDs of the parts `after` holds in a changed form, matched by position in their message."""
    return {
        item_id
        for old, new in zip(before, after, strict=True)
        for old_part, new_part in zip(old.parts, new.parts, strict=True)
        if old_part != new_part and (item_id := _item_id(old_part)) is not None
    }


def _stale_item_ids(edit: Edit, history: list[ModelMessage]) -> list[str]:
    async def run() -> list[str]:
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), messages=history)
        edited = await edit(history, ctx)
        body = await _RESPONSES.body(edited)
        return stale_reference_violations(_edited_item_ids(history, edited), [body.system, body.messages])

    with asyncio.Runner() as runner:
        return runner.run(run())


@IGNORE_RANDOM_STATE_NOTE
class TestClampOversizedMessages:
    @PROPERTY_SETTINGS
    @given(histories())
    def test_clamped_parts_go_without_their_item_ids(self, history: list[ModelMessage]):
        clamp = ClampOversizedMessages[None](max_part_chars=8, keep_head_chars=2, keep_tail_chars=2)
        assert _stale_item_ids(clamp.compact, history) == []


@IGNORE_RANDOM_STATE_NOTE
class TestClearToolResults:
    @PROPERTY_SETTINGS
    @given(histories())
    def test_cleared_parts_go_without_their_item_ids(self, history: list[ModelMessage]):
        clear = ClearToolResults[None](max_messages=1, keep_pairs=0, clear_tool_inputs=True)
        assert _stale_item_ids(clear.compact, history) == []
