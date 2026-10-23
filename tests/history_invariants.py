"""Checkers for message histories, and for the provider request bodies mapped from them.

Each checker returns the stable code of every violation it finds, once per violating item, and an empty
list when its input holds. They are the executable spec `test_history_properties.py` holds
`repair_messages` and the model adapters to. `test_history_invariants.py` pins, for each code, a
hand-built input that violates it: a property no committed input can fail checks nothing.

History checkers:

- `tool.orphan_return`: a `ToolReturnPart` that answers no call of the response right before its request;
- `tool.missing_return`: a `ToolCallPart` that the request right after its response doesn't answer;
- `tool.duplicate_return`: a call answered more often than it was made;
- `history.python_round_trip` / `history.json_round_trip`: a history loaded back from its Python or JSON dump
  doesn't dump the same;
- `history.not_idempotent`: a transformation changes its own output;
- `history.part_lost` / `history.part_added`: a part one history has more often than the other, other
  than what repair is documented to drop (a return whose call no earlier response made) or to add (a
  return marked as synthesized in its metadata).

Body checkers, over a body's messages projected to `WireMessage`s by the function for its shape:

- `wire.duplicate_call_id`: a call id the body sends more than once;
- `wire.orphan_result`: a tool result with no earlier call or, where results must answer the message
  right before theirs, none in that message;
- `wire.unanswered_call`: a call no later result answers;
- `wire.consecutive_role`: two adjacent messages with the same role;
- `wire.empty_message`: a message with no blocks;
- `wire.empty_text`: a text block with no text;
- `wire.body_changed`: two mappings that should agree, such as mapping the same history twice, don't.
"""

from __future__ import annotations as _annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal, assert_never

import pydantic_core
from pydantic import JsonValue

from pydantic_ai.messages import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ToolCallPart,
    ToolReturnPart,
)


def tool_pairing_violations(messages: Sequence[ModelMessage]) -> list[str]:
    """`tool.*`: every tool call is answered exactly once, in the request right after its response."""
    violations: list[str] = []
    open_calls: Counter[str] = Counter()
    for message in messages:
        if isinstance(message, ModelResponse):
            violations += ['tool.missing_return'] * open_calls.total()
            open_calls = Counter(part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart))
        elif isinstance(message, ModelRequest):
            returns = Counter(part.tool_call_id for part in message.parts if isinstance(part, ToolReturnPart))
            for call_id in dict.fromkeys([*open_calls, *returns]):
                calls, answers = open_calls[call_id], returns[call_id]
                if not calls:
                    violations += ['tool.orphan_return'] * answers
                elif answers > calls:
                    violations += ['tool.duplicate_return'] * (answers - calls)
                else:
                    violations += ['tool.missing_return'] * (calls - answers)
            open_calls = Counter()
        else:
            assert_never(message)
    return violations + ['tool.missing_return'] * open_calls.total()


def round_trip_violations(history: list[ModelMessage], *, mode: Literal['python', 'json']) -> list[str]:
    """`history.python_round_trip` / `history.json_round_trip`: a history loads back from its dump unchanged.

    Both are compared in Python mode, so in JSON mode a value JSON can't hold, such as a tuple or a NaN, counts.
    """
    dumped = ModelMessagesTypeAdapter.dump_python(history)
    if mode == 'python':
        loaded = ModelMessagesTypeAdapter.validate_python(dumped)
        code = 'history.python_round_trip'
    else:
        loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
        code = 'history.json_round_trip'
    return [code] if ModelMessagesTypeAdapter.dump_python(loaded) != dumped else []


def idempotence_violations(
    transform: Callable[[list[ModelMessage]], list[ModelMessage]], history: list[ModelMessage]
) -> list[str]:
    """`history.not_idempotent`: applying `transform` to its own output changes nothing."""
    once = transform(history)
    return ['history.not_idempotent'] if transform(once) != once else []


def _is_synthesized(part: ModelRequestPart | ModelResponsePart) -> bool:
    return isinstance(part, ToolReturnPart) and part.metadata == {SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True}


def part_changes(before: Sequence[ModelMessage], after: Sequence[ModelMessage]) -> list[str]:
    """`history.part_lost` / `history.part_added`: `after` holds the parts of `before`, and no others.

    Parts are compared by their serialized form, every field included. A return in `before` whose call no
    earlier response made is expected to be dropped, and synthesized returns are left out on both sides.
    """
    called: set[str] = set()
    expected: Counter[bytes] = Counter()
    for message in before:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                called.add(part.tool_call_id)
            orphan = isinstance(part, ToolReturnPart) and part.tool_call_id not in called
            if not (orphan or _is_synthesized(part)):
                expected[pydantic_core.to_json(part)] += 1
    actual = Counter(
        pydantic_core.to_json(part) for message in after for part in message.parts if not _is_synthesized(part)
    )
    return ['history.part_lost'] * (expected - actual).total() + ['history.part_added'] * (actual - expected).total()


@dataclass(frozen=True)
class WireBlock:
    """One block of a provider message, reduced to what the body checkers read."""

    kind: Literal['text', 'call', 'result', 'other']
    value: str = ''
    """The text of a `'text'` block, or the call id of a `'call'` or `'result'` block."""


@dataclass(frozen=True)
class WireMessage:
    """One message of a provider request body: a role (or an item type) and its blocks."""

    role: str
    blocks: list[WireBlock]


def _items(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _fields(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _text(value: JsonValue) -> str:
    """A string value, or `''` for a missing one, which the checkers then report as empty or unpaired."""
    return value if isinstance(value, str) else ''


def _content_blocks(content: JsonValue) -> list[WireBlock]:
    """Message content given as a string, or as a list of typed parts."""
    if isinstance(content, str):
        return [WireBlock('text', content)]
    return [
        WireBlock('text', _text(fields.get('text')))
        if fields.get('type') in ('text', 'input_text', 'output_text')
        else WireBlock('other')
        for fields in map(_fields, _items(content))
    ]


def chat_completions_wire(messages: JsonValue) -> list[WireMessage]:
    """OpenAI Chat Completions messages, and those of the APIs shaped like it: Groq, Mistral, Cohere, Hugging Face.

    A tool message is the result it carries: its content is the result's, not a text block.
    """
    wire: list[WireMessage] = []
    for fields in map(_fields, _items(messages)):
        call_id = fields.get('tool_call_id')
        blocks = (
            [WireBlock('result', _text(call_id))] if call_id is not None else _content_blocks(fields.get('content'))
        )
        blocks += [WireBlock('call', _text(_fields(call).get('id'))) for call in _items(fields.get('tool_calls'))]
        wire.append(WireMessage(_text(fields.get('role')), blocks))
    return wire


def openai_responses_wire(items: JsonValue) -> list[WireMessage]:
    """OpenAI Responses input items, each a message of its own, named by its role or its item type."""
    wire: list[WireMessage] = []
    for fields in map(_fields, _items(items)):
        item_type = _text(fields.get('type', 'message'))
        if item_type == 'function_call':
            wire.append(WireMessage(item_type, [WireBlock('call', _text(fields.get('call_id')))]))
        elif item_type == 'function_call_output':
            wire.append(WireMessage(item_type, [WireBlock('result', _text(fields.get('call_id')))]))
        elif item_type == 'message':
            wire.append(WireMessage(_text(fields.get('role')), _content_blocks(fields.get('content'))))
        else:
            wire.append(WireMessage(item_type, [WireBlock('other')]))
    return wire


def anthropic_wire(messages: JsonValue) -> list[WireMessage]:
    """Anthropic Messages API messages, with consecutive same-role turns combined into one, as the API combines them.

    The adapter relies on that: reasoning carried over from another provider goes in a user turn of its own, right
    behind the one holding the tool results (`tests/models/anthropic/test_carried_thinking.py`).
    """
    wire: list[WireMessage] = []
    for fields in map(_fields, _items(messages)):
        blocks = _content_blocks(fields.get('content'))
        for index, block in enumerate(map(_fields, _items(fields.get('content')))):
            if block.get('type') == 'tool_use':
                blocks[index] = WireBlock('call', _text(block.get('id')))
            elif block.get('type') == 'tool_result':
                blocks[index] = WireBlock('result', _text(block.get('tool_use_id')))
        role = _text(fields.get('role'))
        if wire and wire[-1].role == role:
            wire[-1].blocks.extend(blocks)
        else:
            wire.append(WireMessage(role, blocks))
    return wire


def bedrock_wire(messages: JsonValue) -> list[WireMessage]:
    """Bedrock Converse messages, whose content blocks are keyed by their type."""
    wire: list[WireMessage] = []
    for fields in map(_fields, _items(messages)):
        blocks: list[WireBlock] = []
        for block in map(_fields, _items(fields.get('content'))):
            if 'text' in block:
                blocks.append(WireBlock('text', _text(block['text'])))
            elif 'toolUse' in block:
                blocks.append(WireBlock('call', _text(_fields(block['toolUse']).get('toolUseId'))))
            elif 'toolResult' in block:
                blocks.append(WireBlock('result', _text(_fields(block['toolResult']).get('toolUseId'))))
            else:
                blocks.append(WireBlock('other'))
        wire.append(WireMessage(_text(fields.get('role')), blocks))
    return wire


def google_wire(contents: JsonValue) -> list[WireMessage]:
    """Gemini contents, whose parts are keyed by their type; a thought is not a text block."""
    wire: list[WireMessage] = []
    for fields in map(_fields, _items(contents)):
        blocks: list[WireBlock] = []
        for part in map(_fields, _items(fields.get('parts'))):
            if 'function_call' in part:
                blocks.append(WireBlock('call', _text(_fields(part['function_call']).get('id'))))
            elif 'function_response' in part:
                blocks.append(WireBlock('result', _text(_fields(part['function_response']).get('id'))))
            elif 'text' in part and not part.get('thought'):
                blocks.append(WireBlock('text', _text(part['text'])))
            else:
                blocks.append(WireBlock('other'))
        wire.append(WireMessage(_text(fields.get('role')), blocks))
    return wire


def duplicate_call_id_violations(messages: Sequence[WireMessage]) -> list[str]:
    """`wire.duplicate_call_id`: every call id is sent once."""
    call_ids = Counter(block.value for message in messages for block in message.blocks if block.kind == 'call')
    return ['wire.duplicate_call_id'] * sum(count - 1 for count in call_ids.values())


def call_pairing_violations(messages: Sequence[WireMessage], *, results_follow_their_call: bool) -> list[str]:
    """`wire.orphan_result`, `wire.unanswered_call`: every result answers an earlier call, and every call is answered.

    With `results_follow_their_call`, a result must answer a call of the message right before its own, as
    Anthropic and Bedrock require.
    """
    violations: list[str] = []
    calls: list[str] = []
    answered: set[str] = set()
    previous_calls: list[str] = []
    for message in messages:
        message_calls: list[str] = []
        for block in message.blocks:
            match block.kind:
                case 'call':
                    message_calls.append(block.value)
                case 'result':
                    if block.value not in (previous_calls if results_follow_their_call else calls):
                        violations.append('wire.orphan_result')
                    answered.add(block.value)
                case 'text' | 'other':
                    pass
                case _:
                    assert_never(block.kind)
        calls += message_calls
        previous_calls = message_calls
    return violations + ['wire.unanswered_call' for call_id in dict.fromkeys(calls) if call_id not in answered]


def role_violations(messages: Sequence[WireMessage]) -> list[str]:
    """`wire.consecutive_role`: no two adjacent messages have the same role."""
    return ['wire.consecutive_role' for previous, current in pairwise(messages) if previous.role == current.role]


def empty_message_violations(messages: Sequence[WireMessage]) -> list[str]:
    """`wire.empty_message`: every message has a block."""
    return ['wire.empty_message' for message in messages if not message.blocks]


def empty_text_violations(messages: Sequence[WireMessage]) -> list[str]:
    """`wire.empty_text`: no text block is empty."""
    return [
        'wire.empty_text'
        for message in messages
        for block in message.blocks
        if block.kind == 'text' and not block.value
    ]


def body_changes(first: object, second: object) -> list[str]:
    """`wire.body_changed`: two mappings that should give the same request body did."""
    return ['wire.body_changed'] if first != second else []
