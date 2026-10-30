"""Canonical ID assignment invariants that provider responses cannot deterministically exercise."""

from __future__ import annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)
from pydantic_ai.models import _hashed_tool_call_id  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.models._tool_call_ids import deduplicate_tool_call_ids


def _ids(messages: list[ModelMessage]) -> list[str]:
    return [
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart)
    ]


@pytest.mark.parametrize('across_responses', [False, True])
def test_fifo_pairing_through_split_result_messages(across_responses: bool) -> None:
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart('a', {}, 'same'), ToolCallPart('b', {}, 'same'), ToolCallPart('c', {}, 'same')]
        ),
        ModelRequest(parts=[ToolReturnPart('a', 'A', 'same')]),
        ModelRequest(parts=[ToolReturnPart('b', 'B', 'same', outcome='retried'), ToolReturnPart('c', 'C', 'same')]),
    ]
    stored = ModelMessagesTypeAdapter.dump_json(messages)
    rendered = deduplicate_tool_call_ids(messages, across_responses=across_responses)
    ids = _ids(rendered)
    assert len(set(ids[:3])) == 3
    assert ids[:3] == ids[3:]
    assert ModelMessagesTypeAdapter.dump_json(messages) == stored
    assert deduplicate_tool_call_ids(rendered, across_responses=across_responses) is rendered


@pytest.mark.parametrize('across_responses', [False, True])
def test_assignment_is_deterministic_and_prefix_stable(across_responses: bool) -> None:
    # An authored ID can collide with the first generated replacement.
    collision = _hashed_tool_call_id('same#1', 32)
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart('a', {}, collision), ToolCallPart('b', {}, 'same'), ToolCallPart('c', {}, 'same')]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('a', 'A', collision),
                ToolReturnPart('b', 'B', 'same'),
                ToolReturnPart('c', 'C', 'same'),
            ]
        ),
    ]
    first = deduplicate_tool_call_ids(messages, across_responses=across_responses)
    assert len(set(_ids(first)[:3])) == 3
    assert _ids(first)[:3] == _ids(first)[3:]
    assert ModelMessagesTypeAdapter.dump_json(first) == ModelMessagesTypeAdapter.dump_json(
        deduplicate_tool_call_ids(messages, across_responses=across_responses)
    )
    messages.extend(
        [
            ModelResponse(parts=[ToolCallPart('d', {}, _ids(first)[2])]),
            ModelRequest(parts=[ToolReturnPart('d', 'D', _ids(first)[2])]),
        ]
    )
    extended = deduplicate_tool_call_ids(messages, across_responses=across_responses)
    assert ModelMessagesTypeAdapter.dump_json(extended[:2]) == ModelMessagesTypeAdapter.dump_json(first)
    if across_responses:
        assert len(set(_ids(extended)[:3] + _ids(extended)[6:7])) == 4


def test_native_parts_and_typed_search_parts_preserve_their_contracts() -> None:
    native_call = NativeToolCallPart('web_search', {}, tool_call_id='same', provider_name='anthropic')
    native_result = NativeToolReturnPart('web_search', 'found', tool_call_id='same', provider_name='anthropic')
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                native_call,
                native_result,
                ToolSearchCallPart(args={'queries': ['a']}, tool_call_id='same'),
                ToolCallPart('a', {}, 'same'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='same'),
                ToolReturnPart('a', 'A', 'same'),
            ]
        ),
    ]
    mapped = deduplicate_tool_call_ids(messages)
    assert mapped[0].parts[0] is native_call
    assert mapped[0].parts[1] is native_result
    assert isinstance(mapped[0].parts[2], ToolSearchCallPart)
    assert isinstance(mapped[1].parts[0], ToolSearchReturnPart)
    assert _ids(mapped)[:2] == _ids(mapped)[2:]


def test_valid_history_keeps_identity_and_orphan_results_are_not_repaired() -> None:
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart('before'), ToolCallPart('a', {}, 'call')]),
        ModelRequest(parts=[ToolReturnPart('a', 'A', 'call'), ToolReturnPart('other', 'orphan', 'absent')]),
    ]
    assert deduplicate_tool_call_ids(messages) is messages
    assert deduplicate_tool_call_ids(tuple(messages)) == messages


def test_later_response_reuse_shadows_an_unanswered_call() -> None:
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('a', {}, 'same'), ToolCallPart('b', {}, 'same')]),
        ModelResponse(parts=[ToolCallPart('c', {}, 'same')]),
        ModelRequest(parts=[ToolReturnPart('c', 'C', 'same')]),
    ]
    mapped = deduplicate_tool_call_ids(messages, across_responses=True)
    assert _ids(mapped)[2] == _ids(mapped)[3]


def test_rewritten_ids_fit_mistral_format() -> None:
    messages: list[ModelMessage] = [ModelResponse(parts=[ToolCallPart('a', {}, 'aB3dE5gH9') for _ in range(5)])]
    ids = _ids(deduplicate_tool_call_ids(messages, id_length=9))
    assert len(set(ids)) == 5
    assert all(len(value) == 9 and value.isascii() and value.isalnum() for value in ids)


def test_exhausted_id_space_raises_instead_of_looping() -> None:
    messages: list[ModelMessage] = [ModelResponse(parts=[ToolCallPart('a', {}, 'x') for _ in range(63)])]
    with pytest.raises(UserError, match='Too many tool calls'):
        deduplicate_tool_call_ids(messages, id_length=1)
