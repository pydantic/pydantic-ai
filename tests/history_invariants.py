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

- `wire.duplicate_call_id`: a call id one message sends more than once or, with `across_messages`, one the
  body sends more than once;
- `wire.orphan_result`: a tool result with no earlier call or, where results must answer the message
  right before theirs, none in that message;
- `wire.unanswered_call`: a call no later result answers;
- `wire.empty_body`: a body with no messages, unless `system_suffices` (the API takes its system prompt alone);
- `wire.empty_message`: a message with no blocks, other than an assistant turn where `empty_assistant_turns_pass`
  or a user turn where `empty_user_turns_pass`;
- `wire.empty_text`: a text block with no text, or one outside an assistant turn with only whitespace;
- `wire.body_changed`: two mappings that should agree, such as mapping the same history twice, don't.
"""

from __future__ import annotations as _annotations

from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
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
    """`history.not_idempotent`: applying `transform` to its own output changes nothing.

    Compared by value: idempotence is about what the history says. That `Model.prepare_messages` returns the same
    list object when it has nothing to change, which the agent graph relies on to skip a second repair, is a separate
    contract, pinned in `tests/models/test_model.py` by `test_prepare_messages_keeps_history_identity_when_*`.
    """
    once = transform(history)
    snapshot = deepcopy(once)
    return ['history.not_idempotent'] if transform(once) != snapshot else []


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


def duplicate_call_id_violations(messages: Sequence[WireMessage], *, across_messages: bool) -> list[str]:
    """`wire.duplicate_call_id`: no call id is sent twice in one message, or, `across_messages`, in the whole body."""
    scopes = [messages] if across_messages else [[message] for message in messages]
    call_ids = [
        Counter(block.value for message in scope for block in message.blocks if block.kind == 'call')
        for scope in scopes
    ]
    return ['wire.duplicate_call_id'] * sum(count - 1 for counts in call_ids for count in counts.values())


def call_pairing_violations(messages: Sequence[WireMessage], *, results_follow_their_call: bool) -> list[str]:
    """`wire.orphan_result`, `wire.unanswered_call`: every result answers one earlier call, and every call is answered.

    Each result consumes one call of its id, so a call repeated under one id needs as many results. With
    `results_follow_their_call`, a result must answer a call of the nearest earlier message that is not made of results
    alone, as Anthropic, Bedrock and OpenAI Chat require: OpenAI Chat rejects a `tool` message that does not follow an
    assistant message with `tool_calls`. A message that mixes results with text closes the run of result messages.
    """
    violations: list[str] = []
    pending: Counter[str] = Counter()
    previous: Counter[str] = Counter()
    for message in messages:
        message_calls: Counter[str] = Counter()
        for block in message.blocks:
            match block.kind:
                case 'call':
                    message_calls[block.value] += 1
                case 'result':
                    scope = previous if results_follow_their_call else pending
                    if scope[block.value] <= 0:
                        violations.append('wire.orphan_result')
                    for counter in (previous, pending):
                        if counter[block.value] > 0:
                            counter[block.value] -= 1
                case 'text' | 'other':
                    pass
                case _:
                    assert_never(block.kind)
        pending.update(message_calls)
        if message_calls or not message.blocks or any(block.kind != 'result' for block in message.blocks):
            previous = Counter(message_calls)
    return violations + ['wire.unanswered_call'] * sum(pending.values())


_ASSISTANT_ROLES = frozenset({'assistant', 'model'})


def empty_message_violations(
    messages: Sequence[WireMessage],
    *,
    system_suffices: bool,
    empty_assistant_turns_pass: bool,
    empty_user_turns_pass: bool,
) -> list[str]:
    """`wire.empty_body`, `wire.empty_message`: the body has a message, and every message has a block.

    With `system_suffices`, the body sends a system prompt the API takes as a request on its own, so it needs no
    message. With `empty_assistant_turns_pass` or `empty_user_turns_pass`, the API takes an assistant or a user turn
    with no block.
    """
    if not messages:
        return [] if system_suffices else ['wire.empty_body']
    return [
        'wire.empty_message'
        for message in messages
        if not message.blocks
        and not (empty_assistant_turns_pass and message.role in _ASSISTANT_ROLES)
        and not (empty_user_turns_pass and message.role == 'user')
    ]


def empty_text_violations(messages: Sequence[WireMessage]) -> list[str]:
    """`wire.empty_text`: no text block is empty, and none outside an assistant turn is whitespace-only.

    Anthropic and Bedrock reject both, but take a whitespace-only text block in an assistant turn (probed on
    2026-10-23 with claude-sonnet-5-5 and, on Bedrock, us.anthropic.claude-sonnet-4-6).
    """
    return [
        'wire.empty_text'
        for message in messages
        for block in message.blocks
        if block.kind == 'text' and not (block.value if message.role in _ASSISTANT_ROLES else block.value.strip())
    ]


def body_changes(first: object, second: object) -> list[str]:
    """`wire.body_changed`: two mappings that should give the same request body didn't."""
    return ['wire.body_changed'] if first != second else []
