"""Shared conversation-extraction primitives for conversation-level evaluators.

These utilities convert the Pydantic AI OpenTelemetry span tree into a sequence of
[`ConversationTurn`][pydantic_evals.evaluators.ConversationTurn]s, which higher-level
evaluators (e.g. [`ConversationGoalAchievement`][pydantic_evals.evaluators.ConversationGoalAchievement],
[`RoleAdherence`][pydantic_evals.evaluators.RoleAdherence]) can feed to an LLM judge.

Users can compose these primitives to build their own conversation evaluators on top.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from ..otel.span_tree import AttributeValue, SpanNode, SpanTree

__all__ = (
    'ConversationTurn',
    'extract_conversation_turns',
    'format_transcript',
)

logger = logging.getLogger(__name__)

Role = Literal['user', 'assistant', 'system', 'tool']

_VALID_ROLES: frozenset[str] = frozenset({'user', 'assistant', 'system', 'tool'})

_ALL_MESSAGES_ATTR = 'pydantic_ai.all_messages'
_NEW_MESSAGE_INDEX_ATTR = 'pydantic_ai.new_message_index'
_GEN_AI_INPUT_MESSAGES_ATTR = 'gen_ai.input.messages'
_GEN_AI_OUTPUT_MESSAGES_ATTR = 'gen_ai.output.messages'


@dataclass
class ConversationTurn:
    """A single turn in a conversation, flattened from an OTel GenAI message.

    One `ChatMessage` may expand to multiple `ConversationTurn`s — for example, an
    assistant response that contains both a text part and a tool call becomes two turns.
    This keeps rendering simple and makes it easy for judge prompts to reference specific
    actions by turn number.
    """

    role: Role
    """The conversational role of this turn.

    `'tool'` is used for tool-call responses (the output of a tool), distinct from
    the assistant's tool-call request which uses `'assistant'` with `tool_name` set.
    """

    content: str
    """Human-readable content of the turn.

    For text parts, this is the text. For tool calls, this is the tool name; the
    arguments are available separately in `tool_arguments`. Multimodal parts are
    represented as placeholders like `[image]` or `[file: image/png]`.
    """

    turn_index: int
    """0-based index of this turn in the conversation, stable across extraction calls."""

    tool_name: str | None = None
    """For tool-call and tool-return turns, the name of the tool; `None` otherwise."""

    tool_arguments: str | None = None
    """For tool-call turns, a JSON-encoded string of the arguments; `None` otherwise.

    Stored as a string rather than a parsed structure so that the exact payload sent
    to the tool is preserved for judge prompts and downstream analysis.
    """


def extract_conversation_turns(span_tree: SpanTree) -> list[ConversationTurn]:
    """Extract ordered conversation turns from a span tree.

    Looks for `pydantic_ai.all_messages` on agent-run spans first (the richest source,
    containing the full conversation). Falls back to per-request spans carrying
    `gen_ai.input.messages` / `gen_ai.output.messages` when no agent-run span is
    present (e.g. when using `direct` model calls).

    If an agent run was handed a non-empty `message_history`, the span records
    `pydantic_ai.new_message_index` — we honor this so that continued runs only return
    turns produced by the *current* run, avoiding double-scoring of prior history.

    The function is defensive: malformed JSON, missing fields, or unknown part types
    are logged at WARNING level and skipped, rather than raising. This keeps evaluators
    resilient to schema drift in Pydantic AI instrumentation.

    Args:
        span_tree: The span tree captured during the task run.

    Returns:
        A list of turns in conversation order. Empty if no recognized messages are found.
    """
    for node in span_tree:
        if _ALL_MESSAGES_ATTR in node.attributes:
            turns = _turns_from_all_messages(node)
            if turns is not None:
                return turns

    return _turns_from_gen_ai_spans(span_tree)


def format_transcript(turns: list[ConversationTurn]) -> str:
    """Format turns as a numbered transcript for use in judge prompts.

    Each line is prefixed with `[N] role:` where `N` is the turn index (so prompts
    can unambiguously reference specific turns, e.g. "role was broken at turn 3").

    Args:
        turns: Conversation turns, typically from `extract_conversation_turns`.

    Returns:
        A newline-separated transcript. Empty string if `turns` is empty.
    """
    lines: list[str] = []
    for turn in turns:
        prefix = f'[{turn.turn_index}] {turn.role}'
        if turn.role == 'assistant' and turn.tool_name is not None:
            arg_suffix = f' args={turn.tool_arguments}' if turn.tool_arguments else ''
            lines.append(f'{prefix} (tool_call {turn.tool_name!r}{arg_suffix}): {turn.content}')
        elif turn.role == 'tool' and turn.tool_name is not None:
            lines.append(f'{prefix} (tool {turn.tool_name!r}): {turn.content}')
        else:
            lines.append(f'{prefix}: {turn.content}')
    return '\n'.join(lines)


def _turns_from_all_messages(node: SpanNode) -> list[ConversationTurn] | None:
    """Parse the `pydantic_ai.all_messages` attribute on a single span node.

    Returns `None` if the attribute value isn't a valid JSON array, so the caller
    can fall back to other sources. Individual malformed messages are skipped.
    """
    raw = node.attributes[_ALL_MESSAGES_ATTR]
    messages = _parse_json_list(raw, _ALL_MESSAGES_ATTR)
    if messages is None:
        return None

    start_index = _resolve_new_message_index(node.attributes.get(_NEW_MESSAGE_INDEX_ATTR), len(messages))
    return _flatten_messages(messages[start_index:])


def _turns_from_gen_ai_spans(span_tree: SpanTree) -> list[ConversationTurn]:
    """Fallback extraction from per-request `gen_ai.input.messages` / `gen_ai.output.messages` spans.

    Iterates spans in tree order (which is start-timestamp order): for the *first*
    model-request span we include both input and output; for subsequent spans we
    include only the output, since each new input subsumes the previous conversation.
    """
    messages: list[Any] = []
    seen_first_input = False
    for node in span_tree:
        input_raw = node.attributes.get(_GEN_AI_INPUT_MESSAGES_ATTR)
        output_raw = node.attributes.get(_GEN_AI_OUTPUT_MESSAGES_ATTR)
        if input_raw is None and output_raw is None:
            continue

        if not seen_first_input and input_raw is not None:
            input_messages = _parse_json_list(input_raw, _GEN_AI_INPUT_MESSAGES_ATTR)
            if input_messages is not None:
                messages.extend(input_messages)
            seen_first_input = True

        if output_raw is not None:
            output_messages = _parse_json_list(output_raw, _GEN_AI_OUTPUT_MESSAGES_ATTR)
            if output_messages is not None:
                messages.extend(output_messages)

    return _flatten_messages(messages)


def _parse_json_list(value: AttributeValue, attr_name: str) -> list[Any] | None:
    """Parse a JSON attribute value that's expected to be an array, returning `None` on failure.

    Callers must have already verified the attribute is present (not `None`).
    """
    if not isinstance(value, str):
        logger.warning('Attribute %r has non-string value; skipping.', attr_name)
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as e:
        logger.warning('Failed to parse JSON for attribute %r: %s', attr_name, e)
        return None
    if not isinstance(parsed, list):
        logger.warning('Attribute %r did not decode to a list; skipping.', attr_name)
        return None
    return cast(list[Any], parsed)


def _resolve_new_message_index(raw: AttributeValue | None, total: int) -> int:
    """Resolve `pydantic_ai.new_message_index`, falling back to 0 on invalid values."""
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool):
        logger.warning('Attribute %r has non-integer value %r; scoring all turns.', _NEW_MESSAGE_INDEX_ATTR, raw)
        return 0
    if raw < 0 or raw > total:
        logger.warning(
            'Attribute %r value %d out of range [0, %d]; scoring all turns.',
            _NEW_MESSAGE_INDEX_ATTR,
            raw,
            total,
        )
        return 0
    return raw


def _flatten_messages(messages: list[Any]) -> list[ConversationTurn]:
    """Flatten a list of ChatMessage-shaped dicts into ConversationTurn entries."""
    turns: list[ConversationTurn] = []
    for message in messages:
        if not isinstance(message, dict):
            logger.warning('Skipping non-dict message: %r', type(message).__name__)
            continue
        message_dict = cast(dict[str, Any], message)
        role_raw = message_dict.get('role')
        if not isinstance(role_raw, str) or role_raw not in _VALID_ROLES:
            logger.warning('Skipping message with missing or invalid role: %r', role_raw)
            continue
        parts = message_dict.get('parts')
        if not isinstance(parts, list):
            logger.warning('Skipping message with missing or non-list parts: role=%r', role_raw)
            continue

        # Narrowing for type-checker
        message_role: Role = cast(Role, role_raw)
        turns.extend(_flatten_typed_parts(message_role, cast(list[Any], parts), starting_index=len(turns)))
    return turns


def _flatten_typed_parts(role: Role, parts: list[Any], starting_index: int) -> list[ConversationTurn]:
    """Convert the `parts` of one ChatMessage into one or more ConversationTurns.

    Each recognized part produces exactly one turn. Unknown or malformed parts are
    logged and skipped.
    """
    turns: list[ConversationTurn] = []
    for part in parts:
        if not isinstance(part, dict):
            logger.warning('Skipping non-dict part in role=%r message.', role)
            continue
        part_dict = cast(dict[str, Any], part)
        try:
            turn = _part_to_turn(role, part_dict, turn_index=starting_index + len(turns))
        except (TypeError, ValueError, KeyError) as e:
            logger.warning('Failed to flatten part %r: %s', part_dict.get('type'), e)
            continue
        if turn is not None:
            turns.append(turn)
    return turns


def _part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    """Convert a single typed part dict into a ConversationTurn, or return `None` to skip."""
    part_type = part.get('type')
    handler = _PART_HANDLERS.get(part_type) if isinstance(part_type, str) else None
    if handler is not None:
        return handler(role, part, turn_index)

    if part_type in ('image-url', 'audio-url', 'video-url', 'document-url'):
        # These legacy media URL types aren't in `_PART_HANDLERS` because they share one
        # derivation rule (strip the `-url` suffix); keeping them inline avoids four
        # near-identical handler functions.
        assert isinstance(part_type, str)
        modality = part_type.split('-', 1)[0]
        return ConversationTurn(role=role, content=f'[{modality}]', turn_index=turn_index)

    logger.warning('Skipping unknown part type %r at turn %d', part_type, turn_index)
    return None


def _text_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    content = part.get('content', '')
    if not isinstance(content, str):
        content = str(content)
    return ConversationTurn(role=role, content=content, turn_index=turn_index)


def _thinking_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    # Thinking parts are internal reasoning; they're not part of the visible conversation
    # that a judge should score. Skip silently (no warning — this is expected).
    del role, part, turn_index
    return None


def _tool_call_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    del role  # tool_call always produces an assistant turn
    name = part.get('name')
    if not isinstance(name, str):
        logger.warning('Skipping tool_call part with missing name at turn %d', turn_index)
        return None
    arguments = part.get('arguments')
    tool_arguments: str | None
    if arguments is None:
        tool_arguments = None
    elif isinstance(arguments, str):
        tool_arguments = arguments
    else:
        try:
            tool_arguments = json.dumps(arguments)
        except (TypeError, ValueError) as e:
            logger.warning('Failed to JSON-encode tool_call arguments at turn %d: %s', turn_index, e)
            tool_arguments = None
    return ConversationTurn(
        role='assistant',
        content=name,
        turn_index=turn_index,
        tool_name=name,
        tool_arguments=tool_arguments,
    )


def _tool_call_response_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    del role  # tool responses always produce a 'tool' turn
    name = part.get('name')
    if not isinstance(name, str):
        logger.warning('Skipping tool_call_response part with missing name at turn %d', turn_index)
        return None
    result = part.get('result')
    if result is None:
        content = ''
    elif isinstance(result, str):
        content = result
    else:
        try:
            content = json.dumps(result)
        except (TypeError, ValueError):
            content = repr(result)
    return ConversationTurn(role='tool', content=content, turn_index=turn_index, tool_name=name)


def _uri_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    modality = part.get('modality') or 'file'
    return ConversationTurn(role=role, content=f'[{modality}]', turn_index=turn_index)


def _binary_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    media_type = part.get('media_type') or 'binary'
    return ConversationTurn(role=role, content=f'[{media_type}]', turn_index=turn_index)


def _blob_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    modality = part.get('modality') or part.get('mime_type') or 'blob'
    return ConversationTurn(role=role, content=f'[{modality}]', turn_index=turn_index)


def _file_part_to_turn(role: Role, part: dict[str, Any], turn_index: int) -> ConversationTurn | None:
    modality = part.get('modality')
    mime_type = part.get('mime_type')
    if modality and mime_type:
        descriptor = f'{modality}: {mime_type}'
    elif modality:
        descriptor = modality
    elif mime_type:
        descriptor = mime_type
    else:
        return ConversationTurn(role=role, content='[file]', turn_index=turn_index)
    return ConversationTurn(role=role, content=f'[file: {descriptor}]', turn_index=turn_index)


_PartHandler = Callable[[Role, dict[str, Any], int], 'ConversationTurn | None']

# Dispatch table for known part types. Kept at module level so it's built once and
# new part types can be added by appending a handler without growing the main function.
_PART_HANDLERS: dict[str, _PartHandler] = {
    'text': _text_part_to_turn,
    'thinking': _thinking_part_to_turn,
    'tool_call': _tool_call_part_to_turn,
    'tool_call_response': _tool_call_response_part_to_turn,
    'uri': _uri_part_to_turn,
    'binary': _binary_part_to_turn,
    'blob': _blob_part_to_turn,
    'file': _file_part_to_turn,
}
