"""Breaking chat-template special tokens in the text a request carries, for routes that may tokenize them.

Many hosts of open-weight models encode message content with special tokens allowed. A literal `<|im_start|>` in a
tool result then reaches the model as the real control token, and the text after it as a turn of its own, in whatever
role it names. A note beside the text can't help, because it sits in a turn the forged one has already closed.

On a profile with [`may_tokenize_special_tokens_in_content`][pydantic_ai.profiles.ModelProfile.may_tokenize_special_tokens_in_content],
`Model.prepare_messages` puts a zero-width joiner inside each such token, so no tokenizer can match it. It only
changes the outgoing copy, and the same text always gets the same joiners, so stored history stays as written and
replaying it sends the same bytes.

The model reads the joiner as part of the text and copies it into what it writes, or writes another invisible character
for it, which would silently corrupt a file it edits. So the inverse runs on each response, at the `Model` boundary:
text and tool call arguments lose a zero-width character inside a token before anything else sees them, whether the
response was streamed or not. Tools, output validation,
users and stored history get the literal token. When that text is sent back, the break restores the bytes the model
wrote.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

from .messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    CachePoint,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolCallPartDelta,
    UserPromptPart,
    is_multi_modal_content,
    tool_return_ta,
)

if TYPE_CHECKING:
    from .profiles import ModelProfile

ZERO_WIDTH_JOINER: Final = '‍'

_LITERAL_TOKEN_TAILS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ('<', ('start_of_turn>', 'end_of_turn>')),
    (
        '[',
        (
            'INST]',
            '/INST]',
            'SYSTEM_PROMPT]',
            '/SYSTEM_PROMPT]',
            'AVAILABLE_TOOLS]',
            '/AVAILABLE_TOOLS]',
            'TOOL_RESULTS]',
            '/TOOL_RESULTS]',
            'TOOL_CALLS]',
            'e~[',
        ),
    ),
    (']', ('~b]', '~!b[')),
)
"""The fixed tokens, as their first character and the rest."""

_NAMED_TOKEN_TAILS: Final = (
    r'\|[^\s<>|\u200b-\u200d\u2060\ufeff]{1,64}\|>|｜[^\s<>｜\u200b-\u200d\u2060\ufeff]{1,64}｜>'
)
"""What follows the `<` of a `<|im_start|>` or a `<｜User｜>`, whichever name a template gives it."""

_TOKEN_SHAPES: Final[tuple[tuple[str, str], ...]] = (
    ('<', _NAMED_TOKEN_TAILS),
    *((re.escape(first), '|'.join(re.escape(tail) for tail in tails)) for first, tails in _LITERAL_TOKEN_TAILS),
)

_SPECIAL_TOKEN_START: Final = re.compile('|'.join(f'{first}(?={tails})' for first, tails in _TOKEN_SHAPES))
"""Only the token's first character is consumed, so overlapping tokens, as in `]~b]~b]`, each get a joiner."""


def _joiner_inside_token(joiner: str) -> re.Pattern[str]:
    return re.compile('|'.join(f'(?<={first}){joiner}(?={tails})' for first, tails in _TOKEN_SHAPES))


_ZERO_WIDTH_CHARACTER: Final = '[\u200b\u200c\u200d\u2060\ufeff]'
"""What a model writes for the joiner when it copies it: some families reach for another invisible character instead."""
_ZERO_WIDTH_ESCAPE: Final = r'\\u(?:200[bcdBCD]|2060|[fF][eE][fF][fF])'
"""The same characters, as a model can write them in a tool call's JSON arguments."""

_JOINER_IN_TEXT: Final = _joiner_inside_token(_ZERO_WIDTH_CHARACTER)
_JOINER_IN_JSON: Final = _joiner_inside_token(f'(?:{_ZERO_WIDTH_CHARACTER}|{_ZERO_WIDTH_ESCAPE})')

_POSSIBLE_JOINER_IN_TEXT: Final = re.compile(rf'(?<=[<\[\]]){_ZERO_WIDTH_CHARACTER}')
_POSSIBLE_JOINER_IN_JSON: Final = re.compile(
    rf'(?<=[<\[\]])(?:{_ZERO_WIDTH_CHARACTER}|{_ZERO_WIDTH_ESCAPE}|\\(?:u[0-9a-fA-F]{{0,3}})?$)'
)
"""Where a joiner has started, also when a chunk ended partway through its escape."""

_TOKEN_TAIL_PREFIX: Final = re.compile(
    r'\|[^\s<>|\u200b-\u200d\u2060\ufeff]{0,64}\|?|｜[^\s<>｜\u200b-\u200d\u2060\ufeff]{0,64}｜?|'
    + '|'.join(
        re.escape(tail[:length])
        for _, tails in _LITERAL_TOKEN_TAILS
        for tail in tails
        for length in range(1, len(tail))
    )
    + '|'
)
"""The start of what can follow a joiner in a token, including nothing yet."""

_PartT = TypeVar('_PartT')


def break_special_tokens(text: str) -> str:
    """`text` with a zero-width joiner after the first character of each chat-template special token in it."""
    return _SPECIAL_TOKEN_START.sub(lambda match: match[0] + ZERO_WIDTH_JOINER, text)


def restore_special_tokens(text: str) -> str:
    """The inverse of [`break_special_tokens`][pydantic_ai._special_tokens.break_special_tokens], for a model's text.

    Any zero-width character inside a token is removed, since that's what a model's copy of the joiner can turn into.
    """
    return _JOINER_IN_TEXT.sub('', text)


def restore_special_tokens_in_parts(parts: Sequence[_PartT]) -> list[_PartT]:
    """Remove the joiner from the tokens in response text and tool call arguments.

    Reasoning is left as the model wrote it, as it can be signed, and so are native tool calls, which the provider ran.
    """
    return [_restore_part(part) for part in parts]


def restore_special_tokens_in_response(response: ModelResponse) -> ModelResponse:
    """`response` with [`restore_special_tokens_in_parts`][pydantic_ai._special_tokens.restore_special_tokens_in_parts] applied."""
    parts = restore_special_tokens_in_parts(response.parts)
    return (
        replace(response, parts=parts) if any(new is not old for new, old in zip(parts, response.parts)) else response
    )


def _restore_part(part: _PartT) -> _PartT:
    new_part: Any = part
    if isinstance(part, TextPart):
        content = restore_special_tokens(part.content)
        if content != part.content:
            new_part = replace(part, content=content)
    elif isinstance(part, ToolCallPart):
        call = cast('ToolCallPart[Any]', part)
        args = _restore_args(call.args)
        if args != call.args:
            new_part = replace(call, args=args)
    return new_part


def _restore_args(args: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if isinstance(args, str):
        return _JOINER_IN_JSON.sub('', args)
    return None if args is None else _restore_value(args)


def _restore_value(value: Any) -> Any:
    if isinstance(value, str):
        return restore_special_tokens(value)
    if isinstance(value, dict):
        mapping = cast('dict[Any, Any]', value)
        return {_restore_value(key): _restore_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        return [_restore_value(item) for item in cast('list[Any]', value)]
    return value


@dataclass
class _StreamedTextRestorer:
    """Removes the joiner from the tokens in text that arrives in chunks, however they split it.

    Only a joiner is ever held back, from where it starts until what follows shows whether it's in a token, so text
    without one goes out as it arrives.
    """

    joiner_inside_token: re.Pattern[str]
    possible_joiner: re.Pattern[str]
    previous_character: str = ''
    held: str = ''

    def feed(self, chunk: str) -> str:
        window = self.previous_character + self.held + chunk
        restored = self.joiner_inside_token.sub('', window)
        hold_from = self._hold_from(restored)
        text = restored[len(self.previous_character) : hold_from]
        self.held = restored[hold_from:]
        self.previous_character = text[-1:] or self.previous_character
        return text

    def flush(self) -> str:
        """What's still held, when no more text is coming: a joiner that didn't end up in a token stays as written."""
        held, self.held = self.held, ''
        return held

    def _hold_from(self, text: str) -> int:
        start = None
        for start in self.possible_joiner.finditer(text):
            pass
        if start is None:
            return len(text)
        if start.end() == len(text) or _TOKEN_TAIL_PREFIX.fullmatch(text, start.end()):
            return start.start()
        return len(text)


@dataclass
class StreamRestorer:
    """Applies the joiner's removal to a response's stream events, one text per part."""

    restorers: dict[int, _StreamedTextRestorer] = field(default_factory=dict[int, _StreamedTextRestorer])
    args_parts: set[int] = field(default_factory=set[int])

    def restore(self, event: ModelResponseStreamEvent) -> ModelResponseStreamEvent:
        if isinstance(event, PartStartEvent):
            return self._restore_start(event)
        if isinstance(event, PartDeltaEvent):
            return self._restore_delta(event)
        return event

    def _restore_start(self, event: PartStartEvent) -> PartStartEvent:
        part = event.part
        if isinstance(part, TextPart):
            return replace(event, part=replace(part, content=self._text_restorer(event.index).feed(part.content)))
        if isinstance(part, ToolCallPart):
            if isinstance(part.args, str):
                return replace(event, part=replace(part, args=self._args_restorer(event.index).feed(part.args)))
            return replace(event, part=_restore_part(part))
        return event

    def _restore_delta(self, event: PartDeltaEvent) -> PartDeltaEvent:
        delta = event.delta
        if isinstance(delta, TextPartDelta):
            return replace(
                event, delta=replace(delta, content_delta=self._text_restorer(event.index).feed(delta.content_delta))
            )
        if isinstance(delta, ToolCallPartDelta) and isinstance(delta.args_delta, str):
            return replace(
                event, delta=replace(delta, args_delta=self._args_restorer(event.index).feed(delta.args_delta))
            )
        return event

    def _text_restorer(self, index: int) -> _StreamedTextRestorer:
        return self.restorers.setdefault(index, _StreamedTextRestorer(_JOINER_IN_TEXT, _POSSIBLE_JOINER_IN_TEXT))

    def _args_restorer(self, index: int) -> _StreamedTextRestorer:
        self.args_parts.add(index)
        return self.restorers.setdefault(index, _StreamedTextRestorer(_JOINER_IN_JSON, _POSSIBLE_JOINER_IN_JSON))

    def restore_part_so_far(self, index: int, part: _PartT) -> _PartT:
        """`part` as the stream's events have delivered it, so a snapshot and later deltas never overlap."""
        restored: Any = _restore_part(part)
        restorer = self.restorers.get(index)
        if restorer is None or not restorer.held:
            return restored
        if isinstance(restored, TextPart):
            new_part: Any = replace(restored, content=restored.content.removesuffix(restorer.held))
            return new_part
        call = cast('ToolCallPart[Any]', restored)
        assert isinstance(call.args, str)
        new_part = replace(call, args=call.args.removesuffix(restorer.held))
        return new_part

    def flush(self) -> list[PartDeltaEvent]:
        """Send what's held for every part, as deltas: at the end of the stream, or before a new part starts."""
        events: list[PartDeltaEvent] = []
        for index, restorer in self.restorers.items():
            if held := restorer.flush():
                delta = (
                    ToolCallPartDelta(args_delta=held)
                    if index in self.args_parts
                    else TextPartDelta(content_delta=held)
                )
                events.append(PartDeltaEvent(index=index, delta=delta))
        return events


def restore_special_tokens_in_stream(
    events: AsyncIterator[ModelResponseStreamEvent], restorer: StreamRestorer | None
) -> AsyncIterator[ModelResponseStreamEvent]:
    """`events` with the joiner removed from the tokens in their text and tool call arguments, if there's a `restorer`."""
    return events if restorer is None else _restored_events(events, restorer)


async def _restored_events(
    events: AsyncIterator[ModelResponseStreamEvent], restorer: StreamRestorer
) -> AsyncIterator[ModelResponseStreamEvent]:
    async for event in events:
        if isinstance(event, PartStartEvent):
            for flushed in restorer.flush():
                yield flushed
        yield restorer.restore(event)
    for flushed in restorer.flush():
        yield flushed


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
        container = cast('UserPromptPart | BaseToolReturnPart[Any]', part)
        content = _break_value(container.content)
        if content is not container.content:
            new_part = replace(container, content=content)
    elif isinstance(part, BaseToolCallPart):
        call = cast('BaseToolCallPart[Any]', part)
        args = _break_value(call.args)
        if args is not call.args:
            new_part = replace(call, args=args)
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
