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
from collections.abc import AsyncIterator, Callable, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, TypeVar, cast

import pydantic_core

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

ZERO_WIDTH_JOINER: Final = '\u200d'

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

_NAME: Final = r'[^\s<>|\u200b-\u200d\u2060\ufeff]'
_FULL_WIDTH_NAME: Final = r'[^\s<>｜\u200b-\u200d\u2060\ufeff]'

_ZERO_WIDTH_CHARACTER: Final = '[\u200b\u200c\u200d\u2060\ufeff]'
"""What a model writes for the joiner when it copies it: some families reach for another invisible character instead."""
_ZERO_WIDTH_ESCAPE: Final = r'\\u(?:200[bcdBCD]|2060|[fF][eE][fF][fF])'
"""The same characters, as a model can write them in a tool call's JSON arguments."""
_JSON_ZERO_WIDTH: Final = f'(?:{_ZERO_WIDTH_CHARACTER}|{_ZERO_WIDTH_ESCAPE})'
_JSON_GT: Final = r'(?:>|\\u003[eE])'
"""Some servers re-encode JSON arguments with `<` and `>` escaped, as Go's `json.Marshal` does."""
_JSON_LT_LOOKBEHINDS: Final = (r'(?<=<)', r'(?<=\\u003c)', r'(?<=\\u003C)')


def _token_shapes(gt: str) -> tuple[tuple[str, str], ...]:
    """Each token's first character and what can follow it, with `gt` matching a `>`."""
    named = (
        rf'\|{_NAME}{{1,64}}\|{gt}',  # `<|im_start|>`
        rf'｜{_FULL_WIDTH_NAME}{{1,64}}｜{gt}',  # `<｜User｜>`
        rf'\|[a-z_]{{1,32}}{gt}',  # `<|turn>`
        rf'[a-z_]{{1,32}}\|{gt}',  # `<turn|>`
    )
    return (
        ('<', '|'.join(named)),
        *(
            (re.escape(first), '|'.join(re.escape(tail).replace('>', gt) for tail in tails))
            for first, tails in _LITERAL_TOKEN_TAILS
        ),
    )


_SPECIAL_TOKEN_START: Final = re.compile('|'.join(f'{first}(?={tails})' for first, tails in _token_shapes('>')))
"""Only the token's first character is consumed, so overlapping tokens, as in `]~b]~b]`, each get a joiner."""

_JOINER_IN_TEXT: Final = re.compile(
    '|'.join(f'(?<={first}){_ZERO_WIDTH_CHARACTER}(?={tails})' for first, tails in _token_shapes('>'))
)
_JOINER_IN_JSON: Final = re.compile(
    '|'.join(
        f'{lookbehind}{_JSON_ZERO_WIDTH}(?={tails})'
        for first, tails in _token_shapes(_JSON_GT)
        for lookbehind in (_JSON_LT_LOOKBEHINDS if first == '<' else (f'(?<={first})',))
    )
)

_POSSIBLE_JOINER_IN_TEXT: Final = re.compile(rf'(?<=[<\[\]]){_ZERO_WIDTH_CHARACTER}')
_POSSIBLE_JOINER_IN_JSON: Final = re.compile(
    '|'.join(
        rf'{lookbehind}(?:{_JSON_ZERO_WIDTH}|\\(?:u[0-9a-fA-F]{{0,3}})?$)'
        for lookbehind in (r'(?<=[<\[\]])', *_JSON_LT_LOOKBEHINDS[1:])
    )
)
"""Where a joiner has started, also when a chunk ended partway through its escape."""

_LONGEST_TOKEN_START: Final = len(r'\u003c')
"""How much emitted text the joiner's check has to look back on."""

_PARTIAL_ESCAPE: Final = r'(?:\\(?:u[0-9a-fA-F]{0,3})?)?'

_TOKEN_TAIL_PREFIX: Final = re.compile(
    rf'\|{_NAME}{{0,64}}(?:\|{_PARTIAL_ESCAPE})?|｜{_FULL_WIDTH_NAME}{{0,64}}(?:｜{_PARTIAL_ESCAPE})?'
    rf'|[a-z_]{{1,32}}(?:\|{_PARTIAL_ESCAPE})?|(?:start|end)_of_turn{_PARTIAL_ESCAPE}|'
    + '|'.join(
        re.escape(tail[:length])
        for _, tails in _LITERAL_TOKEN_TAILS
        for tail in tails
        for length in range(1, len(tail))
    )
    + '|'
)
"""The start of what can follow a joiner in a token, including nothing yet, up to an escaped `>` still arriving."""

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
    if not isinstance(args, str):
        return None if args is None else _restore_value(args)
    restored = _JOINER_IN_JSON.sub('', args)
    if '\\u' not in restored:
        return restored
    # The rest of the token can be escaped too, which only shows once the JSON is decoded.
    try:
        decoded = pydantic_core.from_json(restored)
    except ValueError:
        return restored
    return restored if (fully_restored := _restore_value(decoded)) == decoded else fully_restored


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
    previous: str = ''
    held: str = ''

    def feed(self, chunk: str) -> str:
        restored = self.joiner_inside_token.sub('', self.previous + self.held + chunk)
        hold_from = self._hold_from(restored)
        text = restored[len(self.previous) : hold_from]
        self.held = restored[hold_from:]
        self.previous = (self.previous + text)[-_LONGEST_TOKEN_START:]
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
        if isinstance(delta, ToolCallPartDelta):
            if isinstance(delta.args_delta, str):
                args_delta = self._args_restorer(event.index).feed(delta.args_delta)
                return replace(event, delta=replace(delta, args_delta=args_delta))
            if delta.args_delta is not None:
                return replace(event, delta=replace(delta, args_delta=_restore_args(delta.args_delta)))
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

    def flush_text(self) -> list[PartDeltaEvent]:
        """Send what's held for each text part, as a new part has started and the last one has to end whole.

        Tool call arguments stay held: parallel calls' deltas interleave, and JSON arguments never end in a joiner.
        """
        return self._flush(lambda index: index not in self.args_parts)

    def flush(self) -> list[PartDeltaEvent]:
        """Send what's held for every part, at the end of the stream."""
        return self._flush(lambda index: True)

    def _flush(self, include: Callable[[int], bool]) -> list[PartDeltaEvent]:
        events: list[PartDeltaEvent] = []
        for index, restorer in self.restorers.items():
            if include(index) and (held := restorer.flush()):
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
            for flushed in restorer.flush_text():
                yield flushed
        yield restorer.restore(event)
    for flushed in restorer.flush():
        yield flushed


def break_special_tokens_in_messages(
    messages: list[ModelMessage], profile: ModelProfile, system: str
) -> list[ModelMessage]:
    """Break the special tokens in the text of every part that doesn't come from the application itself, if `profile` asks for it.

    `SystemPromptPart`s are left alone: they're the application's own channel, like instructions and tool definitions.
    So is reasoning signed by `system`, the provider this request goes to, which it would reject when changed, and a
    compaction a provider produced. Reasoning signed by another provider is sent as text, without its signature. Ids,
    provider details and every other field stay as they are.

    Returns the original list when nothing changed.
    """
    if not profile.get('may_tokenize_special_tokens_in_content', False):
        return messages
    new_messages = [_break_message(message, system) for message in messages]
    return new_messages if any(new is not old for new, old in zip(new_messages, messages)) else messages


def _break_message(message: ModelMessage, system: str) -> ModelMessage:
    if isinstance(message, ModelRequest):
        request_parts = [_break_part(part, system) for part in message.parts]
        changed = any(new is not old for new, old in zip(request_parts, message.parts))
        return replace(message, parts=request_parts) if changed else message
    response_parts = [_break_part(part, system) for part in message.parts]
    changed = any(new is not old for new, old in zip(response_parts, message.parts))
    return replace(message, parts=response_parts) if changed else message


def _break_part(part: _PartT, system: str) -> _PartT:
    new_part: Any = part
    if isinstance(part, UserPromptPart):
        content = _break_content(part.content)
        if content is not part.content:
            new_part = replace(part, content=content)
    elif isinstance(part, BaseToolReturnPart):
        new_part = _break_tool_return(cast('BaseToolReturnPart[Any]', part))
    elif isinstance(part, BaseToolCallPart):
        new_part = _break_tool_call(cast('BaseToolCallPart[Any]', part))
    elif (
        isinstance(part, TextPart)
        or (isinstance(part, ThinkingPart) and (part.signature is None or part.provider_name != system))
        or (isinstance(part, CompactionPart) and part.provider_name is None)
    ):
        content = part.content and break_special_tokens(part.content)
        if content != part.content:
            new_part = replace(part, content=content)
    return new_part


def _break_tool_return(part: BaseToolReturnPart[Any]) -> BaseToolReturnPart[Any]:
    content = _break_content(part.content)
    name = break_special_tokens(part.tool_name)
    if content is part.content and name == part.tool_name:
        return part
    return replace(part, content=content, tool_name=name)


def _break_tool_call(call: BaseToolCallPart[Any]) -> BaseToolCallPart[Any]:
    args = call.args
    if isinstance(args, str) and '\\u' in args:
        args = _break_escaped_args(call, args)
    else:
        args = _break_value(args)
    name = break_special_tokens(call.tool_name)
    if args is call.args and name == call.tool_name:
        return call
    return replace(call, args=args, tool_name=name)


def _break_escaped_args(call: BaseToolCallPart[Any], args: str) -> str | dict[str, Any]:
    """Adapters decode JSON arguments to send them canonically, which would turn an escaped token into a literal one."""
    try:
        decoded = call.args_as_dict(raise_if_invalid=True)
    except (ValueError, AssertionError):
        # Malformed arguments go out as they are, so only the joiner changes.
        return break_special_tokens(args)
    broken = _break_value(decoded)
    return args if broken is decoded else broken


def _break_value(value: Any) -> Any:
    """`value` with the special tokens in its text broken, or `value` itself when it has none.

    Strings, text content and the strings in lists and dicts are broken in place, a file's text is left for the adapter
    that inlines it, and any other value, which the adapter sends as its JSON text, is replaced by its broken JSON form.
    """
    if isinstance(value, str):
        broken = break_special_tokens(value)
        return value if broken == value else broken
    if isinstance(value, TextContent):
        return _break_adjacent_text([value])[0]
    if isinstance(value, list | tuple):
        return _break_items(cast('list[Any] | tuple[Any, ...]', value))
    if isinstance(value, dict):
        return _break_mapping(cast('dict[Any, Any]', value))
    if is_multi_modal_content(value):
        return _break_file_identifier(value)
    if value is None or isinstance(value, bool | int | float | CachePoint):
        return value
    dumped = tool_return_ta.dump_python(value, mode='json', by_alias=True)
    broken = _break_value(dumped)
    return value if broken is dumped else broken


def _break_items(sequence: list[Any] | tuple[Any, ...]) -> Any:
    items = [_break_value(item) for item in sequence]
    return sequence if all(new is old for new, old in zip(items, sequence)) else items


def _break_file_identifier(file: Any) -> Any:
    """The file with its identifier broken: it's sent in text, which adapters build from it after this pass."""
    identifier = break_special_tokens(file.identifier)
    if identifier == file.identifier:
        return file
    # Not `replace`: these classes have `__init__`s of their own, which read the `identifier` alias.
    broken = copy(file)
    broken._identifier = identifier
    return broken


def _break_content(content: Any) -> Any:
    """Break a message's content, whose list items an adapter sends as separate text parts.

    Hosts join those before tokenizing, unlike the items of a structured value, which keep their JSON punctuation.
    """
    if isinstance(content, list | tuple):
        return _break_sequence(cast('list[Any] | tuple[Any, ...]', content))
    return _break_value(content)


def _break_sequence(sequence: list[Any] | tuple[Any, ...]) -> Any:
    """Break each item, treating adjacent text as one text."""
    items: list[Any] = []
    for is_text, group in groupby(sequence, key=lambda item: isinstance(item, str | TextContent | CachePoint)):
        items.extend(_break_adjacent_text(list(group)) if is_text else [_break_value(item) for item in group])
    return sequence if all(new is old for new, old in zip(items, sequence)) else items


def _break_mapping(mapping: dict[Any, Any]) -> Any:
    entries = [(_break_value(key), _break_value(item)) for key, item in mapping.items()]
    if all(new_key is key and new_item is item for (new_key, new_item), (key, item) in zip(entries, mapping.items())):
        return mapping
    try:
        return_as_json = len({key for key, _ in entries}) < len(entries)
    except TypeError:
        # A tuple key with a token in it became a list.
        return_as_json = True
    if return_as_json:
        # A broken key now equals one that was already written that way: send the JSON text, which keeps every entry.
        return break_special_tokens(tool_return_ta.dump_json(mapping).decode())
    return dict(entries)


def _break_adjacent_text(items: Sequence[str | TextContent | CachePoint]) -> list[str | TextContent | CachePoint]:
    # A cache marker sends no text of its own, so the text on either side of it joins up too.
    texts = [
        item.content if isinstance(item, TextContent) else '' if isinstance(item, CachePoint) else item
        for item in items
    ]
    joiners = [match.end() for match in _SPECIAL_TOKEN_START.finditer(''.join(texts))]
    new_items: list[str | TextContent | CachePoint] = []
    offset = 0
    for item, text in zip(items, texts):
        end = offset + len(text)
        cuts = [position - offset for position in joiners if offset < position <= end]
        new_items.append(_with_joiners(item, text, cuts) if cuts else item)
        offset = end
    return new_items


def _with_joiners(item: str | TextContent | CachePoint, text: str, cuts: list[int]) -> str | TextContent | CachePoint:
    bounds = [0, *cuts, len(text)]
    broken = ZERO_WIDTH_JOINER.join(text[start:stop] for start, stop in zip(bounds, bounds[1:]))
    return replace(item, content=broken) if isinstance(item, TextContent) else broken
