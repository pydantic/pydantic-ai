"""What the stream parity modules share, and which of them checks each model class's streams.

A stream parity module replays recorded traffic through a model class's `StreamedResponse` and through the model's
processing of the complete response that holds the same data, and asserts they build the same `ModelResponse`, as
`comparable_response` compares them. Where the SDK accumulates a stream, or its stream folds into its complete form
by documented rules, the module replays recorded streams and builds the complete response from their events. Where
neither holds (Groq, Mistral, Hugging Face, Cohere), it renders each recorded complete response as the stream that
sends it, as `render_chat_stream` does for Chat Completions, and replays that.

Each module also replays its streams re-chunked, and asserts each builds what the stream built unchanged, so a
streaming defect shows without waiting for a recording whose chunks happen to show it:

- `split-delta`: a seeded half of the deltas that carry text, thinking or tool call arguments are each cut in two
  at a seeded offset, as a provider may send any piece of them in a chunk of its own.
- `merge-deltas`: a seeded half of the pairs of adjacent deltas of the same block are each sent as one. A rendered
  stream sends each block whole, so it has none to merge.
- `drop-argument-deltas`, for the Responses API only: a function call's argument deltas are left out where its
  `response.function_call_arguments.done` event repeats the arguments, as the ChatGPT backend sends them.

A delta that is a whole thinking tag, like `<think>`, is never cut or merged: `ModelResponsePartsManager` reads a tag
only as a delta of its own, as a model sends the tag's one token. A re-chunking that builds something else on `main`
goes in the module's `known` entries, with the issue that tracks it, until it's fixed.

Each recording is re-chunked under each seed in `SEEDS`, so a failure names the recording, strategy and seed that
reproduce it. A sweep over many seeds would find more, but stays out of the suite on purpose: every replay costs a
whole stream, and the suite runs on every push.

A module's `DRIVER`, a `StreamParityDriver`, says how it replays and re-chunks one recording. The modules import this
one, so `STREAM_PARITY` can't import their drivers back: it names each class's module instead, with the recorded
interactions it claims. `tests/models/test_stream_parity_coverage.py` checks that every model class whose
`request_stream` calls its SDK has an entry, and that each module an entry names has a `DRIVER`.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Generic, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import JsonValue

from pydantic_ai.messages import BaseToolCallPart, ModelResponse, ModelResponsePart, NativeToolReturnPart
from pydantic_ai.profiles import DEFAULT_THINKING_TAGS

from ..manifests import Exempt
from ..stream_contract import Known, comparable_part

RecordingT = TypeVar('RecordingT')
WireT = TypeVar('WireT')
EventT = TypeVar('EventT')

Strategy = Literal['split-delta', 'merge-deltas', 'drop-argument-deltas']

SEEDS = (0, 1)

JsonObject = dict[str, JsonValue]


def comparable_response(
    response: ModelResponse, part_overrides: Callable[[ModelResponsePart], Mapping[str, object]] | None = None
) -> dict[str, object]:
    """The response's fields, without the time it was built, with each part compared as `comparable_part` does.

    `part_overrides` gives the fields of a part that a module compares by a rule of its own.
    """
    fields: dict[str, object] = {f.name: getattr(response, f.name) for f in dataclasses.fields(response)}
    del fields['timestamp']
    fields['parts'] = [
        {**comparable_part(part), **(part_overrides(part) if part_overrides else {})} for part in response.parts
    ]
    return fields


def generated_tool_call_id(part: ModelResponsePart) -> dict[str, str]:
    """A part override for a tool call ID Pydantic AI generates, as each replay generates its own.

    A provider that doesn't ID a call, like Gemini's code execution or Groq's executed tools, gets a `pyd_ai_` ID.
    """
    if isinstance(part, BaseToolCallPart | NativeToolReturnPart) and part.tool_call_id.startswith('pyd_ai_'):
        return {'tool_call_id': '<generated>'}
    return {}


@dataclass(frozen=True)
class StreamParityDriver(Generic[RecordingT, WireT]):
    """How a stream parity module replays one recording, and re-chunks the stream it replays."""

    wire: Callable[[RecordingT], WireT]
    """The stream the recording replays as: its recorded body, or the stream a complete response renders as."""
    replay: Callable[[RecordingT, WireT], Awaitable[ModelResponse]]
    """What the model class's streamed response builds when the recording's request gets this stream back."""
    rechunk: Callable[[WireT, Strategy, Random], WireT | None]
    """The stream re-chunked by a strategy under a seed, or `None` if the strategy finds nothing in it to change."""
    strategies: tuple[Strategy, ...] = ('split-delta', 'merge-deltas')
    comparable: Callable[[ModelResponse], Mapping[str, object]] = comparable_response
    """How two replays of a recording compare: a module that generates part IDs compares them by its own rule."""


async def assert_rechunked_streams_match(
    driver: StreamParityDriver[RecordingT, WireT],
    recordings: Iterable[tuple[str, RecordingT]],
    known: Mapping[str, Known] | None = None,
) -> None:
    """Assert each recording, re-chunked by each of the driver's strategies under each seed, builds what it built.

    A re-chunking `known` holds, by `'<recording> <strategy> seed=<seed>'`, must still build something else, until the
    issue that tracks it is fixed and its entry removed.
    """
    rechunked: dict[str, Mapping[str, object]] = {}
    recorded: dict[str, Mapping[str, object]] = {}
    applied: set[Strategy] = set()
    for name, recording in recordings:
        wire = driver.wire(recording)
        built = driver.comparable(await driver.replay(recording, wire))
        for strategy in driver.strategies:
            for seed in SEEDS:
                if (other := driver.rechunk(wire, strategy, Random(seed))) is not None:
                    key = f'{name} {strategy} seed={seed}'
                    rechunked[key] = driver.comparable(await driver.replay(recording, other))
                    recorded[key] = built
                    applied.add(strategy)
    # Guards against a wire format change leaving a strategy nothing to re-chunk.
    assert applied == set(driver.strategies), f'strategies with nothing to re-chunk: {set(driver.strategies) - applied}'
    for key in (known or {}).keys() & recorded.keys():
        assert rechunked.pop(key) != recorded.pop(key), (
            f'{key} builds what it built as recorded: remove it from `known`'
        )
    # This module's asserts aren't rewritten by pytest, so the failure names what differs itself.
    differing = [key for key in recorded if rechunked[key] != recorded[key]]
    assert not differing, (
        f'{len(differing)} of {len(recorded)} re-chunkings build something else than the recording, first '
        f'{differing[0]}:\n  re-chunked: {rechunked[differing[0]]}\n  recorded:   {recorded[differing[0]]}\n'
        'Fix the adapter; if it builds this on `main` and an issue tracks the fix, add each re-chunking to the '
        "`_KNOWN` table the driver's module passes as `known`"
    )


def cut(text: str, rng: Random) -> tuple[str, str] | None:
    """`text` cut in two non-empty pieces at a seeded offset, or `None` if it's too short to cut."""
    if len(text) < 2:
        return None
    offset = rng.randint(1, len(text) - 1)
    return text[:offset], text[offset:]


def split_deltas(
    events: Sequence[EventT], rng: Random, split: Callable[[EventT, Random], tuple[EventT, EventT] | None]
) -> list[EventT] | None:
    """`events` with a seeded half of the events `split` can cut each replaced by its two pieces."""
    pieces = [split(event, rng) for event in events]
    candidates = [index for index, piece in enumerate(pieces) if piece is not None]
    if not candidates:
        return None
    chosen = {index for index in candidates if rng.random() < 0.5} or {rng.choice(candidates)}
    rechunked: list[EventT] = []
    for index, (event, piece) in enumerate(zip(events, pieces)):
        rechunked.extend(piece if index in chosen and piece is not None else (event,))
    return rechunked


def merge_deltas(
    events: Sequence[EventT], rng: Random, merge: Callable[[EventT, EventT], EventT | None]
) -> list[EventT] | None:
    """`events` with a seeded half of the adjacent pairs `merge` can join each replaced by the one event it joins."""
    candidates = [index for index in range(1, len(events)) if merge(events[index - 1], events[index]) is not None]
    if not candidates:
        return None
    chosen = {index for index in candidates if rng.random() < 0.5} or {rng.choice(candidates)}
    rechunked: list[EventT] = []
    for index, event in enumerate(events):
        # A pair after a chosen one starts from the event that one joined, which carries the same block's deltas.
        if index in chosen and rechunked and (joined := merge(rechunked[-1], event)) is not None:
            rechunked[-1] = joined
        else:
            rechunked.append(event)
    return rechunked


@dataclass(frozen=True)
class SseEvent:
    """One event of a recorded server-sent event stream."""

    lines: tuple[str, ...]
    """The event's lines, as recorded."""
    data: JsonObject | None
    """The JSON object its `data:` line carries, if it carries one."""

    def with_data(self, data: JsonObject) -> SseEvent:
        lines = tuple(
            f'data: {json.dumps(data, ensure_ascii=False)}' if line.startswith('data:') else line for line in self.lines
        )
        return SseEvent(lines, data)


def parse_sse(body: str) -> tuple[list[SseEvent], str]:
    """A recorded server-sent event stream's events, and the line break it ends its lines with."""
    newline = '\r\n' if '\r\n' in body else '\n'
    events: list[SseEvent] = []
    for block in body.split(newline * 2):
        lines = tuple(block.split(newline))
        data_lines = [line.removeprefix('data:') for line in lines if line.startswith('data:')]
        try:
            data: JsonValue = json.loads(data_lines[0]) if len(data_lines) == 1 else None
        except ValueError:
            # Like the `[DONE]` that ends a Chat Completions stream.
            data = None
        events.append(SseEvent(lines, data if isinstance(data, dict) else None))
    return events, newline


def render_sse(events: Iterable[SseEvent], newline: str) -> str:
    return (newline * 2).join(newline.join(event.lines) for event in events)


def rechunk_sse(
    body: str,
    strategy: Strategy,
    rng: Random,
    *,
    split: Callable[[JsonObject, Random], tuple[JsonObject, JsonObject] | None],
    merge: Callable[[JsonObject, JsonObject], JsonObject | None],
) -> str | None:
    """A server-sent event stream re-chunked by `split-delta` or `merge-deltas`, applied to its events' JSON data."""
    events, newline = parse_sse(body)
    if strategy == 'split-delta':
        rechunked = split_deltas(events, rng, lambda event, rng: _split_sse_event(event, rng, split))
    else:
        assert strategy == 'merge-deltas', strategy
        rechunked = merge_deltas(events, rng, lambda first, second: _merge_sse_events(first, second, merge))
    return None if rechunked is None else render_sse(rechunked, newline)


def _split_sse_event(
    event: SseEvent, rng: Random, split: Callable[[JsonObject, Random], tuple[JsonObject, JsonObject] | None]
) -> tuple[SseEvent, SseEvent] | None:
    if event.data is None or (pieces := split(event.data, rng)) is None:
        return None
    return event.with_data(pieces[0]), event.with_data(pieces[1])


def _merge_sse_events(
    first: SseEvent, second: SseEvent, merge: Callable[[JsonObject, JsonObject], JsonObject | None]
) -> SseEvent | None:
    if first.data is None or second.data is None or (joined := merge(first.data, second.data)) is None:
        return None
    return second.with_data(joined)


def with_field(data: JsonObject, key: str, value: JsonValue) -> JsonObject:
    """A copy of `data` with `key` set to `value`."""
    return {**data, key: value}


def json_object(value: JsonValue) -> JsonObject:
    """`value` if it's a JSON object, else an empty one, so a missing field reads as absent."""
    return value if isinstance(value, dict) else {}


def json_objects(value: JsonValue) -> list[JsonObject]:
    """The JSON objects in `value` if it's a list, else none."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


# The fields of a Chat Completions delta that carry text or thinking, which a chunk may cut anywhere.
_CHAT_TEXT_FIELDS = ('content', 'reasoning_content', 'reasoning')


def _only_choice(chunk: JsonObject) -> tuple[JsonObject, JsonObject] | None:
    """A chunk's one choice and its delta, if it has exactly one choice."""
    choices = json_objects(chunk.get('choices'))
    return (choices[0], json_object(choices[0].get('delta'))) if len(choices) == 1 else None


def split_chat_chunk(
    chunk: JsonObject, rng: Random, continuation: Callable[[JsonObject, str], JsonValue] | None = None
) -> tuple[JsonObject, JsonObject] | None:
    """A chunk cut in two at a seeded offset of one of its text, thinking or tool call argument deltas.

    The first piece carries only the cut field's head, with the delta's role and the tool call's identity; the second
    carries the rest of the chunk, so the finish reason, usage and logprobs stay where the stream ended them. A cut
    tool call continues in the fragment `continuation` builds from the call and the rest of its arguments: by
    default, OpenAI's, which carries only the call's index.
    """
    if (only := _only_choice(chunk)) is None:
        return None
    choice, delta = only
    calls = json_objects(delta.get('tool_calls'))
    # A thinking tag stays one delta: a stream reads a tag only as a delta of its own, as a model sends its token.
    fields: list[str | int] = [
        key for key in _CHAT_TEXT_FIELDS if isinstance(delta.get(key), str) and delta[key] not in DEFAULT_THINKING_TAGS
    ]
    fields += [
        index for index, call in enumerate(calls) if isinstance(json_object(call.get('function')).get('arguments'), str)
    ]
    if not fields:
        return None
    field = rng.choice(fields)
    head_delta: JsonObject = {'role': delta['role']} if 'role' in delta else {}
    if isinstance(field, str):
        text = delta[field]
        assert isinstance(text, str)
        if (pieces := cut(text, rng)) is None:
            return None
        head_delta[field] = pieces[0]
        tail_delta = with_field(delta, field, pieces[1])
    else:
        call = calls[field]
        function = json_object(call.get('function'))
        arguments = function['arguments']
        assert isinstance(arguments, str)
        if (pieces := cut(arguments, rng)) is None:
            return None
        head_delta['tool_calls'] = [with_field(call, 'function', with_field(function, 'arguments', pieces[0]))]
        rest: JsonValue = (
            {'index': call.get('index'), 'function': {'arguments': pieces[1]}}
            if continuation is None
            else continuation(call, pieces[1])
        )
        tail_delta = with_field(
            delta, 'tool_calls', [rest if index == field else other for index, other in enumerate(calls)]
        )
    tail_delta.pop('role', None)
    head_choice: JsonObject = {'index': choice.get('index', 0), 'delta': head_delta, 'finish_reason': None}
    head: JsonObject = {key: value for key, value in chunk.items() if key != 'usage'}
    head['choices'] = [head_choice]
    return head, with_field(chunk, 'choices', [with_field(choice, 'delta', tail_delta)])


def _set_keys(data: JsonObject) -> set[str]:
    return {key for key, value in data.items() if value is not None}


def merge_chat_chunks(first: JsonObject, second: JsonObject) -> JsonObject | None:
    """The one chunk that carries two adjacent chunks' deltas of the same field, if the first carries nothing else."""
    if (first_only := _only_choice(first)) is None or (second_only := _only_choice(second)) is None:
        return None
    (first_choice, first_delta), (second_choice, second_delta) = first_only, second_only
    fields = _set_keys(first_delta) - {'role'}
    if (
        first.get('usage') is not None
        or _set_keys(first_choice) != {'index', 'delta'}
        or first_choice.get('index') != second_choice.get('index')
        or len(fields) != 1
        or second_delta.get('role') not in (None, first_delta.get('role'))
    ):
        return None
    (field,) = fields
    joined = _join_field(field, first_delta, second_delta)
    if joined is None:
        return None
    merged_delta = with_field(second_delta, field, joined)
    if first_delta.get('role') is not None:
        merged_delta['role'] = first_delta['role']
    return with_field(first | second, 'choices', [with_field(second_choice, 'delta', merged_delta)])


def _join_field(field: str, first_delta: JsonObject, second_delta: JsonObject) -> JsonValue | None:
    first_value, second_value = first_delta.get(field), second_delta.get(field)
    if field in _CHAT_TEXT_FIELDS:
        if not isinstance(first_value, str) or not isinstance(second_value, str):
            return None
        return None if {first_value, second_value} & set(DEFAULT_THINKING_TAGS) else first_value + second_value
    first_calls, second_calls = json_objects(first_value), json_objects(second_value)
    if field != 'tool_calls' or len(first_calls) != 1 or len(second_calls) != 1:
        return None
    (first_call,), (second_call,) = first_calls, second_calls
    first_function, second_function = json_object(first_call.get('function')), json_object(second_call.get('function'))
    first_arguments, second_arguments = first_function.get('arguments'), second_function.get('arguments')
    # The second fragment continues the first only if it names no other call.
    if (
        first_call.get('index') != second_call.get('index')
        or second_call.get('id') not in (None, first_call.get('id'))
        or second_function.get('name') not in (None, '', first_function.get('name'))
        or not isinstance(first_arguments, str)
        or not isinstance(second_arguments, str)
    ):
        return None
    return [
        with_field(first_call, 'function', with_field(first_function, 'arguments', first_arguments + second_arguments))
    ]


_THINKING_TAG_PATTERN = re.compile(f'({"|".join(re.escape(tag) for tag in DEFAULT_THINKING_TAGS)})')


def chat_message_deltas(message: JsonObject, fields: Sequence[str]) -> list[JsonObject]:
    """One delta per field of a complete Chat Completions message, in the order of `fields`, as a provider streams them.

    A text that holds thinking tags is sent as the deltas between them, with each tag a delta of its own, as a model
    streams a tag as one token. A tool call is left to `render_chat_stream`.
    """
    unsent = {key for key, value in message.items() if value not in (None, '', [])} - {'role', 'tool_calls', *fields}
    assert not unsent, f'the stream has no rendering for {unsent}'
    deltas: list[JsonObject] = []
    for key in fields:
        value = message.get(key)
        if isinstance(value, str) and key == 'content':
            deltas += [{key: piece} for piece in _THINKING_TAG_PATTERN.split(value) if piece]
        elif value not in (None, '', []):
            deltas.append({key: value})
    return deltas


def render_chat_stream(
    completion: JsonObject, *, chunk_fields: Sequence[str], deltas: Sequence[JsonObject], final: JsonObject
) -> str:
    """A complete Chat Completions response rendered as the stream that sends it.

    Each chunk repeats the response's `chunk_fields`, the ones the provider's recorded streams repeat. The first
    chunk carries the message's role, the next ones each carry one of `deltas`, then each tool call whole, and the
    last the finish reason and `final`'s fields, like the usage. The choice's logprobs go with the first content.
    """
    choices = json_objects(completion.get('choices'))
    assert len(choices) == 1, completion
    choice = choices[0]
    message = json_object(choice.get('message'))
    all_deltas: list[JsonObject] = [{'role': message.get('role', 'assistant')}, *deltas]
    for index, call in enumerate(json_objects(message.get('tool_calls'))):
        all_deltas.append({'tool_calls': [with_field(call, 'index', index)]})
    top: JsonObject = {key: value for key, value in completion.items() if key in chunk_fields}
    top['object'] = 'chat.completion.chunk'
    chunks: list[JsonObject] = []
    logprobs = choice.get('logprobs')
    for delta in all_deltas:
        delta_choice: JsonObject = {'index': choice.get('index', 0), 'delta': delta, 'finish_reason': None}
        if 'content' in delta and logprobs is not None:
            delta_choice['logprobs'], logprobs = logprobs, None
        chunks.append(with_field(top, 'choices', [delta_choice]))
    last_choice: JsonObject = {
        'index': choice.get('index', 0),
        'delta': {},
        'finish_reason': choice.get('finish_reason'),
    }
    chunks.append({**top, 'choices': [last_choice], **final})
    return ''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n'


@dataclass(frozen=True)
class StreamClaim:
    """The recorded interactions that hold a model class's streams, or the complete responses it renders as streams."""

    endpoint: str
    """What the request URI contains."""
    marker: str
    """What the response body contains, which also lets a scan skip the cassettes whose text lacks it."""
    host: str = ''
    """What the request host contains. An empty host claims every host that no claim on the same endpoint names."""

    def __str__(self) -> str:
        return f'at `{self.endpoint}` on {f"`{self.host}`" if self.host else "any host"}, marked `{self.marker}`'


@dataclass(frozen=True)
class ReplayedBy:
    """A model class whose recorded traffic the `DRIVER` of a stream parity module replays, whole and re-chunked."""

    module: str
    claim: StreamClaim

    @property
    def reason(self) -> str:
        return f'`{self.module}` replays the recorded interactions {self.claim}'


def claim_of(model_class: str) -> StreamClaim:
    """The recorded interactions `model_class`'s driver replays."""
    entry = STREAM_PARITY[model_class]
    assert isinstance(entry, ReplayedBy), entry
    return entry.claim


def claimant(uri: str, body: str) -> str | None:
    """The model class whose claim takes a recorded interaction: the one naming its host, else one for any host.

    An interaction on a host a claim names belongs to that claim's class even when its body lacks the claim's marker,
    so a catch-all claim on the same endpoint never takes, say, a Groq stream that `GroqModel` doesn't replay.
    """
    host = urlsplit(uri).hostname or ''
    claims = {
        name: entry.claim
        for name, entry in STREAM_PARITY.items()
        if isinstance(entry, ReplayedBy) and entry.claim.endpoint in uri
    }
    named = [name for name, claim in claims.items() if claim.host and claim.host in host]
    assert len(named) <= 1, f'{uri} is claimed by {named}'
    owners = named or [name for name, claim in claims.items() if not claim.host]
    return next((name for name in owners if claims[name].marker in body), None)


_ANTHROPIC = 'tests.models.anthropic.test_stream_parity'
_CHAT = 'tests.models.test_openai_chat_stream_parity'
_RESPONSES = 'tests.models.test_openai_responses_stream_parity'
_GOOGLE = 'tests.models.google.test_stream_parity'
_BEDROCK = 'tests.models.bedrock.test_stream_parity'
_XAI = 'tests.models.xai.test_stream_parity'

_COMPLETE_CHAT = StreamClaim('/chat/completions', 'chat.completion')
_CHAT_STREAM = StreamClaim('/chat/completions', 'data:')
_RESPONSES_STREAM = StreamClaim('/responses', '"type":"response.')

_DECISION_STREAM = Exempt(
    'streams the complete decision as one event through `DecisionStreamedResponse`, so it has no chunks to re-chunk '
    'and no stream to compare with its complete response'
)

STREAM_PARITY: dict[str, ReplayedBy | Exempt] = {
    'AnthropicModel': ReplayedBy(_ANTHROPIC, StreamClaim('/v1/messages', 'event: message_start')),
    'OpenAIChatModel': ReplayedBy(_CHAT, _CHAT_STREAM),
    'CrusoeModel': ReplayedBy(_CHAT, dataclasses.replace(_CHAT_STREAM, host='crusoecloud.com')),
    'GitHubCopilotModel': ReplayedBy(_CHAT, dataclasses.replace(_CHAT_STREAM, host='githubcopilot.com')),
    'OpenRouterModel': ReplayedBy(_CHAT, dataclasses.replace(_CHAT_STREAM, host='openrouter.ai')),
    'SnowflakeModel': ReplayedBy(_CHAT, dataclasses.replace(_CHAT_STREAM, host='snowflakecomputing.com')),
    'ZaiModel': ReplayedBy(_CHAT, dataclasses.replace(_CHAT_STREAM, host='api.z.ai')),
    'CerebrasModel': Exempt(
        "streams through `OpenAIStreamedResponse`, which `OpenAIChatModel`'s driver replays; no Cerebras stream is "
        'recorded'
    ),
    'OllamaModel': Exempt(
        "streams through `OpenAIStreamedResponse`, which `OpenAIChatModel`'s driver replays; no Ollama stream is "
        'recorded'
    ),
    'BedrockMantleChatModel': Exempt(
        "streams through `OpenAIStreamedResponse`, which `OpenAIChatModel`'s driver replays; no Chat Completions "
        'stream from Bedrock Mantle is recorded'
    ),
    'OpenAIResponsesModel': ReplayedBy(_RESPONSES, _RESPONSES_STREAM),
    'OpenAICodexModel': ReplayedBy(_RESPONSES, dataclasses.replace(_RESPONSES_STREAM, host='chatgpt.com')),
    'BedrockMantleResponsesModel': ReplayedBy(
        _RESPONSES, dataclasses.replace(_RESPONSES_STREAM, host='bedrock-mantle.')
    ),
    'GoogleModel': ReplayedBy(_GOOGLE, StreamClaim(':streamGenerateContent', 'data: ')),
    # The event stream is binary, so no text marks it: an error response, which has no stream, is left out by status.
    'BedrockConverseModel': ReplayedBy(_BEDROCK, StreamClaim('/converse-stream', '')),
    # gRPC traffic, recorded in `*.xai.yaml` proto cassettes rather than as HTTP.
    'XaiModel': ReplayedBy(_XAI, StreamClaim('chat.stream()', 'response_stream:')),
    # These SDKs have no stream accumulator, so their drivers render recorded complete responses as streams instead.
    'GroqModel': ReplayedBy(
        'tests.models.test_groq_stream_parity', dataclasses.replace(_COMPLETE_CHAT, host='api.groq.com')
    ),
    'MistralModel': ReplayedBy(
        'tests.models.test_mistral_stream_parity', dataclasses.replace(_COMPLETE_CHAT, host='api.mistral.ai')
    ),
    'HuggingFaceModel': ReplayedBy(
        'tests.models.test_huggingface_stream_parity',
        dataclasses.replace(_COMPLETE_CHAT, host='router.huggingface.co'),
    ),
    'CohereModel': ReplayedBy(
        'tests.models.test_cohere_stream_parity', StreamClaim('/v2/chat', 'finish_reason', host='api.cohere.com')
    ),
    'OpenAIDecisionsModel': _DECISION_STREAM,
    'GoogleDecisionsModel': _DECISION_STREAM,
    'TypeSafeModel': _DECISION_STREAM,
    'SystemOneModel': _DECISION_STREAM,
}
"""How each model class whose `request_stream` calls its SDK gets its streams checked, by class name."""
