"""Breaking chat-template special tokens in the text a request carries, for routes that may tokenize them.

Many hosts of open-weight models encode message content with special tokens allowed. A literal `<|im_start|>` in a
tool result then reaches the model as the real control token, and the text after it as a turn of its own, in whatever
role it names. A note beside the text can't help, because it sits in a turn the forged one has already closed.

On a profile with [`may_tokenize_special_tokens_in_content`][pydantic_ai.profiles.ModelProfile.may_tokenize_special_tokens_in_content],
`Model.prepare_messages` puts a zero-width joiner inside each such token, so no tokenizer can match it. It only
changes the outgoing copy, and the same text always gets the same joiners, so stored history stays as written and
replaying it sends the same bytes.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

from .messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    CachePoint,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    TextContent,
    TextPart,
    ThinkingPart,
    UserPromptPart,
    is_multi_modal_content,
    tool_return_ta,
)

if TYPE_CHECKING:
    from .profiles import ModelProfile

ZERO_WIDTH_JOINER: Final = '‍'

_SPECIAL_TOKEN_START: Final = re.compile(
    # Only the token's first character is consumed, so overlapping tokens, as in `]~b]~b]`, each get a joiner.
    r'<(?=\|[^\s<>|]{1,64}\|>)'  # `<|im_start|>`, `<|eot_id|>`, `<|start_header_id|>`, `<|end|>`, `<|user|>`
    r'|<(?=｜[^\s<>｜]{1,64}｜>)'  # `<｜User｜>`, `<｜end▁of▁sentence｜>`
    r'|<(?=(?:start|end)_of_turn>)'
    r'|\[(?=/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS)\]|TOOL_CALLS\])'
    r'|\](?=~!?b[\]\[])'  # `]~b]`, `]~!b[`
    r'|\[(?=e~\[)'
)

_PartT = TypeVar('_PartT')


def break_special_tokens(text: str) -> str:
    """`text` with a zero-width joiner after the first character of each chat-template special token in it."""
    return _SPECIAL_TOKEN_START.sub(lambda match: match[0] + ZERO_WIDTH_JOINER, text)


def break_special_tokens_in_messages(messages: list[ModelMessage], profile: ModelProfile) -> list[ModelMessage]:
    """Break the special tokens in the text of every part that doesn't come from the application itself, if `profile` asks for it.

    `SystemPromptPart`s are left alone: they're the application's own channel, like instructions and tool definitions.
    So is reasoning with a signature, which the provider would reject when changed, and a compaction a provider
    produced. Ids, provider details and every other field stay as they are.

    Returns the original list when nothing changed.
    """
    if not profile.get('may_tokenize_special_tokens_in_content', False):
        return messages
    new_messages = [_break_message(message) for message in messages]
    return new_messages if any(new is not old for new, old in zip(new_messages, messages)) else messages


def _break_message(message: ModelMessage) -> ModelMessage:
    if isinstance(message, ModelRequest):
        request_parts = [_break_part(part) for part in message.parts]
        changed = any(new is not old for new, old in zip(request_parts, message.parts))
        return replace(message, parts=request_parts) if changed else message
    response_parts = [_break_part(part) for part in message.parts]
    changed = any(new is not old for new, old in zip(response_parts, message.parts))
    return replace(message, parts=response_parts) if changed else message


def _break_part(part: _PartT) -> _PartT:
    new_part: Any = part
    if isinstance(part, UserPromptPart | BaseToolReturnPart):
        content = _break_value(part.content)
        if content is not part.content:
            new_part = replace(part, content=content)
    elif isinstance(part, BaseToolCallPart):
        args = _break_value(part.args)
        if args is not part.args:
            new_part = replace(part, args=args)
    elif (
        isinstance(part, TextPart)
        or (isinstance(part, ThinkingPart) and part.signature is None)
        or (isinstance(part, CompactionPart) and part.provider_name is None)
    ):
        content = part.content and break_special_tokens(part.content)
        if content != part.content:
            new_part = replace(part, content=content)
    return new_part


def _break_value(value: Any) -> Any:
    """`value` with the special tokens in its text broken, or `value` itself when it has none.

    Strings, text content and the strings in lists and dicts are broken in place, files are left for the adapter that
    inlines them, and any other value, which the adapter sends as its JSON text, is replaced by its broken JSON form.
    """
    if isinstance(value, str):
        broken = break_special_tokens(value)
        return value if broken == value else broken
    if isinstance(value, TextContent):
        return _break_adjacent_text([value])[0]
    if isinstance(value, list | tuple):
        return _break_sequence(cast('list[Any] | tuple[Any, ...]', value))
    if isinstance(value, dict):
        return _break_mapping(cast('dict[Any, Any]', value))
    if value is None or isinstance(value, bool | int | float | CachePoint) or is_multi_modal_content(value):
        return value
    dumped = tool_return_ta.dump_python(value, mode='json', by_alias=True)
    broken = _break_value(dumped)
    return value if broken is dumped else broken


def _break_sequence(sequence: list[Any] | tuple[Any, ...]) -> Any:
    """Break each item, treating adjacent text as one text: hosts join a message's text parts before tokenizing."""
    items: list[Any] = []
    for is_text, group in groupby(sequence, key=lambda item: isinstance(item, str | TextContent)):
        items.extend(_break_adjacent_text(list(group)) if is_text else [_break_value(item) for item in group])
    return sequence if all(new is old for new, old in zip(items, sequence)) else items


def _break_mapping(mapping: dict[Any, Any]) -> Any:
    entries = [(_break_value(key), _break_value(item)) for key, item in mapping.items()]
    if all(new_key is key and new_item is item for (new_key, new_item), (key, item) in zip(entries, mapping.items())):
        return mapping
    if len({key for key, _ in entries}) < len(entries):
        # A broken key now equals one that was already written that way: send the JSON text, which keeps every entry.
        return break_special_tokens(tool_return_ta.dump_json(mapping).decode())
    return dict(entries)


def _break_adjacent_text(items: Sequence[str | TextContent]) -> list[str | TextContent]:
    texts = [item.content if isinstance(item, TextContent) else item for item in items]
    joiners = [match.end() for match in _SPECIAL_TOKEN_START.finditer(''.join(texts))]
    new_items: list[str | TextContent] = []
    offset = 0
    for item, text in zip(items, texts):
        end = offset + len(text)
        cuts = [position - offset for position in joiners if offset < position <= end]
        new_items.append(_with_joiners(item, text, cuts) if cuts else item)
        offset = end
    return new_items


def _with_joiners(item: str | TextContent, text: str, cuts: list[int]) -> str | TextContent:
    bounds = [0, *cuts, len(text)]
    broken = ZERO_WIDTH_JOINER.join(text[start:stop] for start, stop in zip(bounds, bounds[1:]))
    return replace(item, content=broken) if isinstance(item, TextContent) else broken
