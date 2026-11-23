"""Mutations of recorded provider responses along the provider SDK's pydantic response types.

A cassette holds the shapes a provider happened to send, and the SDK's response types admit more. An adapter that only
ever met the recorded shapes can crash on the others. `mutations` walks a recorded body alongside its SDK type and
yields one `Mutation` for each place where the type admits a shape the recording doesn't have:

- `null`: a field the type lets be null, and that holds a value, is null.
- `drop`: a field with a default other than `None` is left out.
- `unknown-value`: a field typed as a `Literal` or string `Enum` of two or more values holds a value it doesn't list,
  as the API sends once it adds one.
- `unknown-variant`: a discriminated union gets a member of a type it doesn't list, as the API sends once it adds one:
  first in a list, or in place of a field's value. A stream gets an event of an unknown type before the first event
  of each type.
- `text-as-parts`: a string field the type also admits as a list of parts holds the string as one text part.

A field that holds a constant, a `Literal` of one value like a union member's `type`, is never mutated.
`SPEC_SHAPES` admits a mutation at a field whose SDK type leaves out a shape the API sends, and cites what shows it.

`sample_recordings` picks the recordings to mutate. `distinct_mutations` makes each kind of mutation at each site once,
in the first recording that has the site, at every place there, so a run is deterministic and its size is bounded by
the SDK's types rather than by the cassettes.

`tests/models/test_sdk_mutation.py` serves each mutated body to the adapter that reads it. The adapter must return a
`ModelResponse` or raise `UnexpectedModelBehavior`. At a field in `OPEN_FIELDS`, whose values the API leaves open, an
unknown value must still make a `ModelResponse`. A crash an adapter is known to have goes in `KNOWN`, with what tracks
its fix.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from types import NoneType
from typing import Literal, get_args, get_origin

import httpx2
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python
from typing_extensions import TypeIs
from typing_inspection import typing_objects

from pydantic_ai._utils import get_union_args, unwrap_annotated

from ..cassette_utils import TESTS_DIR, load_cassette_yaml, recorded_request_body

MutationKind = Literal['null', 'drop', 'unknown-value', 'unknown-variant', 'text-as-parts']
JsonPath = tuple[str | int, ...]

UNKNOWN = 'pydantic_ai_unknown'
"""The value, and the union member `type`, that no SDK lists."""

STREAM_SITE = 'stream'
"""The site of an event of an unknown type inserted into a stream, as `stream[before <type of the next event>]`."""


@dataclass(frozen=True)
class Mutation:
    """One change to a recorded body that its SDK type admits."""

    kind: MutationKind
    site: str
    """The SDK field the change is at, as `<module>.<class>.<field>`, or for an event inserted into a stream, the
    `STREAM_SITE` form."""
    path: JsonPath
    """Where the change is in the body. In a stream, the path starts at the index of the event."""
    op: Literal['set', 'delete', 'insert']
    value: JsonValue = None


@dataclass(frozen=True)
class SpecShape:
    """The mutations a field's API admits beyond what its SDK type does."""

    kinds: tuple[MutationKind, ...]
    source: str


_LIST_CONTENT = SpecShape(
    ('text-as-parts',),
    'OpenAI-compatible servers send `content` as a list of parts: https://github.com/pydantic/pydantic-ai/pull/11885',
)

SPEC_SHAPES: dict[str, SpecShape] = {
    'openai.types.responses.response_function_web_search.ResponseFunctionWebSearch.action': SpecShape(
        ('drop',),
        'a completed `web_search_call` can come without `action`: https://github.com/openai/openai-openapi/issues/572',
    ),
    'openai.types.chat.chat_completion_chunk.ChoiceDelta.content': _LIST_CONTENT,
    'openai.types.chat.chat_completion_message.ChatCompletionMessage.content': _LIST_CONTENT,
}
"""Mutations a field admits although its SDK type doesn't, by site."""


@dataclass(frozen=True)
class OpenField:
    """A field whose values the API leaves open, so an unknown value must still make a `ModelResponse`."""

    source: str


_FINISH_REASON = OpenField(
    "OpenAI-compatible servers send values the SDK doesn't list, like Bedrock's `guardrail_intervened`: "
    'https://github.com/pydantic/pydantic-ai/pull/11119'
)

OPEN_FIELDS: dict[str, OpenField] = {
    'openai.types.chat.chat_completion.Choice.finish_reason': _FINISH_REASON,
    'openai.types.chat.chat_completion_chunk.Choice.finish_reason': _FINISH_REASON,
}
"""Fields where an `unknown-value` mutation must make a `ModelResponse`, by site."""


@dataclass(frozen=True)
class KnownCrash:
    """A mutation an adapter crashes on, until the fix `tracked_by` names lands."""

    raises: type[Exception]
    tracked_by: str
    repro: str
    """The recording, the path in its body, and what the crash raises from."""


KnownKey = tuple[str, str, MutationKind, str]
"""The model class name, the entry point, the mutation kind and the site."""

KNOWN: dict[KnownKey, KnownCrash] = {
    (
        'AnthropicModel',
        'request',
        'unknown-value',
        'anthropic.types.beta.beta_server_tool_use_block.BetaServerToolUseBlock.name',
    ): KnownCrash(
        AssertionError,
        'to open: anthropic-unknown-server-tool-name',
        '`cassettes/test_native_tool_search_vcr/test_anthropic_native_tool_search.yaml#0` with `content[0].name` an '
        'unknown server tool: `assert_never` in `_native_tools.py::map_server_tool_use_block`',
    ),
    ('GoogleModel', 'request', 'null', 'google.genai.types.FunctionCall.name'): KnownCrash(
        AssertionError,
        'to open: google-function-call-without-name',
        '`models/cassettes/test_google/test_google_tool_config_any_with_tool_without_args.yaml#0` with '
        '`candidates[0].content.parts[0].functionCall.name` null: `assert` in `google.py::_map_response_part`',
    ),
    ('GoogleModel', 'request', 'null', 'google.genai.types.Part.executable_code'): KnownCrash(
        AssertionError,
        'to open: google-code-execution-result-without-call',
        '`models/google/cassettes/test_code_execution/test_receive_history_from_another_provider.yaml#1` with '
        "`candidates[0].content.parts[0].executableCode` null, leaving the next part's `codeExecutionResult` without "
        'its call: `assert` in `google.py::_process_part`',
    ),
    ('GoogleModel', 'request_stream', 'null', 'google.genai.types.Part.text'): KnownCrash(
        AssertionError,
        'to open: google-stream-empty-part',
        "`models/cassettes/test_stream_close/test_cancel_recorded_provider_stream[google].yaml#0` with event 0's "
        '`candidates[0].content.parts[0].text` null, an empty part, which a complete response rejects with '
        '`UnexpectedModelBehavior`: `assert` in `google.py::GeminiStreamedResponse._map_part`',
    ),
    (
        'OpenAIResponsesModel',
        'request',
        'unknown-variant',
        'openai.types.responses.response.Response.output',
    ): KnownCrash(
        TypeError,
        'to open: openai-responses-unknown-output-item',
        '`models/cassettes/test_openai_responses/test_openai_responses_verbosity.yaml#0` with an output item of an '
        'unknown `type` first in `output`, which the SDK builds as a `ResponseOutputMessage` without `content`: '
        '`for content in item.content` in `_responses_model.py::_map_output_message`',
    ),
}
"""Mutations an adapter is known to crash on, each pinned to still raise `raises` until its fix lands."""


def mutations(body: JsonValue, annotation: object) -> Iterator[Mutation]:
    """Every mutation of a complete response `body` that its SDK type `annotation` admits, in document order."""
    yield from _walk(body, annotation, (), '')


def stream_mutations(events: Sequence[JsonValue], event_type: object) -> Iterator[Mutation]:
    """Every mutation of a stream's `events` that the SDK type of each event admits, in stream order.

    Where `event_type` is a union, an event of an unknown type goes before each event, and its site names the type of
    the event it precedes: how a stream is read can depend on where an event comes in it.
    """
    tags = _tags(_members(event_type))
    for index, event in enumerate(events):
        if tags and isinstance(event, dict):
            yield Mutation(
                'unknown-variant', f'{STREAM_SITE}[before {event.get("type")}]', (index,), 'insert', {'type': UNKNOWN}
            )
        yield from _walk(event, event_type, (index,), STREAM_SITE)


def _walk(value: JsonValue, annotation: object, path: JsonPath, site: str) -> Iterator[Mutation]:
    member = _member(value, _members(annotation))
    if isinstance(value, list):
        if member is not None:
            (item_type,) = get_args(member) or (object,)
            if value and isinstance(value[0], dict) and _tags(_members(item_type)):
                yield Mutation('unknown-variant', site, (*path, 0), 'insert', {'type': UNKNOWN})
            for index, item in enumerate(value):
                yield from _walk(item, item_type, (*path, index), site)
    elif isinstance(value, dict):
        if _is_model(member):
            yield from _walk_model(value, member, path)
        elif member is not None:
            *_, item_type = get_args(member) or (object,)
            for key, item in value.items():
                yield from _walk(item, item_type, (*path, key), site)


def _walk_model(value: dict[str, JsonValue], model: type[BaseModel], path: JsonPath) -> Iterator[Mutation]:
    fields = _fields(model)
    for key, item in value.items():
        if (named := fields.get(key)) is None:
            continue
        name, field = named
        site = f'{model.__module__}.{model.__qualname__}.{name}'
        yield from _field_mutations(item, field, (*path, key), site)
        yield from _walk(item, field.annotation, (*path, key), site)


def _field_mutations(item: JsonValue, field: FieldInfo, path: JsonPath, site: str) -> Iterator[Mutation]:
    members = _members(field.annotation)
    choices = _choices(members)
    if len(choices) == 1 and len(members) == 1:
        # A constant, like the `type` of a union member, which the API always sends as it is.
        return
    widened = shape.kinds if (shape := SPEC_SHAPES.get(site)) else ()
    if item is not None and (_admits_none(field.annotation) or 'null' in widened):
        yield Mutation('null', site, path, 'set')
    if (not field.is_required() and field.default is not None) or 'drop' in widened:
        yield Mutation('drop', site, path, 'delete')
    if isinstance(item, str):
        if len(choices) > 1:
            yield Mutation('unknown-value', site, path, 'set', UNKNOWN)
        if _admits_text_parts(members) or 'text-as-parts' in widened:
            yield Mutation('text-as-parts', site, path, 'set', [{'type': 'text', 'text': item}])
    elif isinstance(item, dict) and _tags(members):
        yield Mutation('unknown-variant', site, path, 'set', {'type': UNKNOWN})


def _expand(annotation: object) -> object:
    """`annotation` without the `Annotated` metadata or type aliases around it."""
    while True:
        annotation = unwrap_annotated(annotation)
        origin = get_origin(annotation)
        if typing_objects.is_typealiastype(annotation):
            annotation = annotation.__value__
        elif typing_objects.is_typealiastype(origin):
            annotation = origin.__value__[get_args(annotation)]
        else:
            return annotation


def _members(annotation: object) -> tuple[object, ...]:
    """The types other than `None` that `annotation` admits, with nested unions and aliases flattened."""
    annotation = _expand(annotation)
    if args := get_union_args(annotation):
        return tuple(member for arg in args for member in _members(arg))
    return () if annotation is NoneType else (annotation,)


def _admits_none(annotation: object) -> bool:
    annotation = _expand(annotation)
    if args := get_union_args(annotation):
        return any(_admits_none(arg) for arg in args)
    return annotation is NoneType


def _is_model(member: object) -> TypeIs[type[BaseModel]]:
    return isinstance(member, type) and issubclass(member, BaseModel)


@cache
def _fields(model: type[BaseModel]) -> dict[str, tuple[str, FieldInfo]]:
    """The model's fields and their names, by the key the provider sends each under."""
    return {field.alias or name: (name, field) for name, field in model.model_fields.items()}


def _choices(members: Iterable[object]) -> tuple[str, ...]:
    """The string values that the `Literal` and `Enum` types among `members` list."""
    values: list[str] = []
    for member in members:
        if typing_objects.is_literal(get_origin(member)):
            values.extend(value for value in get_args(member) if isinstance(value, str))
        elif isinstance(member, type) and issubclass(member, Enum):
            values.extend(item.value for item in member if isinstance(item.value, str))
    return tuple(values)


def _tags(members: Iterable[object]) -> dict[str, type[BaseModel]]:
    """The models among `members` by their `type` tags, if they are two or more and each has a tag."""
    tagged = {
        model: _choices(_members(named[1].annotation)) if (named := _fields(model).get('type')) else ()
        for model in members
        if _is_model(model) and model.model_fields
    }
    if len(tagged) < 2 or not all(tagged.values()):
        return {}
    # The first member to list a tag is the one the tag picks.
    return {value: model for model, values in reversed(tagged.items()) for value in values}


def _admits_text_parts(members: Iterable[object]) -> bool:
    """Whether `members` admits a list of parts whose `text` member has a `text` field."""
    for member in members:
        if get_origin(member) is list and (text := _tags(_members(get_args(member)[0])).get('text')) is not None:
            return 'text' in _fields(text)
    return False


def _member(value: JsonValue, members: Sequence[object]) -> object | None:
    """The type among `members` that `value` was sent as, if `value` is a list or an object."""
    if isinstance(value, list):
        return next((member for member in members if get_origin(member) in (list, Sequence)), None)
    if not isinstance(value, dict):
        return None
    if (tags := _tags(members)) and isinstance(tag := value.get('type'), str):
        return tags.get(tag)
    models = [member for member in members if _is_model(member)]
    best = max(models, key=lambda model: len(value.keys() & _fields(model).keys()), default=None)
    if best is not None and value.keys() & _fields(best).keys():
        return best
    return next((member for member in members if get_origin(member) in (dict, Mapping)), None)


def applied(body: JsonValue, mutations: Sequence[Mutation]) -> JsonValue:
    """`body` with `mutations` applied, copying only the containers on their paths.

    They apply from the last in document order to the first, so an insert doesn't move a place an earlier one is at.
    """
    for mutation in reversed(mutations):
        body = _edited(body, mutation.path, mutation)
    return body


def _edited(value: JsonValue, path: JsonPath, mutation: Mutation) -> JsonValue:
    key, *rest = path
    if isinstance(value, dict):
        assert isinstance(key, str), path
        edited = dict(value)
        if rest:
            edited[key] = _edited(value[key], tuple(rest), mutation)
        elif mutation.op == 'delete':
            del edited[key]
        else:
            edited[key] = mutation.value
        return edited
    assert isinstance(value, list) and isinstance(key, int), path
    items = list(value)
    if rest:
        items[key] = _edited(value[key], tuple(rest), mutation)
    else:
        # A list's item is only ever inserted before: a set or delete is at a field.
        assert mutation.op == 'insert', mutation
        items.insert(key, mutation.value)
    return items


_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class Recording:
    """A recorded 200 response: a JSON body, or the events of a stream."""

    name: str
    """`<cassette path relative to tests/>#<interaction index>`, as `recorded_interactions` names it."""
    body: JsonValue
    """The JSON body, or each streamed event's data: a JSON value, or the raw text of data that isn't JSON."""
    event_names: tuple[str | None, ...] | None = None
    """Each streamed event's `event:` name, or `None` for a JSON body."""

    @property
    def events(self) -> list[JsonValue]:
        assert isinstance(self.body, list), self.name
        return self.body

    def response(self, mutations: Sequence[Mutation] = ()) -> httpx2.Response:
        """The recorded response, with `mutations` applied."""
        body = applied(self.body, mutations)
        if self.event_names is None:
            return httpx2.Response(200, json=body)
        assert isinstance(body, list), self.name
        names = list(self.event_names)
        for mutation in reversed(mutations):
            if mutation.op == 'insert' and len(mutation.path) == 1:
                index = mutation.path[0]
                assert isinstance(index, int)
                names.insert(index, UNKNOWN if any(names) else None)
        content = ''.join(
            (f'event: {name}\n' if name else '') + f'data: {data if isinstance(data, str) else json.dumps(data)}\n\n'
            for name, data in zip(names, body, strict=True)
        )
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=content.encode())


def _parse_events(text: str) -> tuple[tuple[str | None, ...], list[JsonValue]]:
    """A recorded event stream's `event:` names and data."""
    names: list[str | None] = []
    events: list[JsonValue] = []
    for block in re.split(r'\r?\n\r?\n', text):
        lines = block.splitlines()
        if not (data_lines := [line[5:].removeprefix(' ') for line in lines if line.startswith('data:')]):
            continue
        names.append(next((line[6:].strip() for line in lines if line.startswith('event:')), None))
        raw = '\n'.join(data_lines)
        try:
            events.append(_JSON.validate_json(raw))
        except ValidationError:
            # Like the `[DONE]` that ends a Chat Completions stream.
            events.append(raw)
    return tuple(names), events


def _recording(name: str, interaction: dict[str, JsonValue], uri: re.Pattern[str], stream: bool) -> Recording | None:
    """The recorded response, if `interaction` answered a request at `uri` with a 200 that streams exactly if `stream`."""
    request, response = interaction.get('request'), interaction.get('response')
    if not (isinstance(request, dict) and isinstance(response, dict) and uri.search(str(request.get('uri')))):
        return None
    status = response.get('status')
    headers = response.get('headers')
    content_type = (
        next((str(value) for key, value in headers.items() if key.lower() == 'content-type'), '')
        if isinstance(headers, dict)
        else ''
    )
    if (status.get('code') if isinstance(status, dict) else status) != 200 or (
        'event-stream' in content_type
    ) != stream:
        return None
    body: JsonValue = recorded_request_body(response)
    if isinstance(body, dict) and isinstance(raw := body.get('string'), str):
        body = raw
    if stream:
        names, events = _parse_events(body) if isinstance(body, str) else ((), [])
        return Recording(name, events, names)
    # vcrpy records a JSON body it didn't parse as its text.
    body = _JSON.validate_json(body) if isinstance(body, str) else body
    return Recording(name, body) if isinstance(body, dict) else None


_MAX_CASSETTE_BYTES = 300_000
"""Larger cassettes are left out: they mostly carry image or audio data, and parsing one costs a second."""

_VOCABULARY = re.compile(
    r'"type"\s*:\s*"([\w.\-]+)"|^\s+-?\s*type: ([\w.\-]+)$|"([A-Za-z_]+)"\s*:|^\s+-?\s*([A-Za-z_]+):', re.MULTILINE
)
"""The field names and `type` values in a recorded response, whether a JSON body, an event stream or a YAML mapping."""


def sample_recordings(uris: Iterable[str]) -> dict[tuple[str, bool], list[Recording]]:
    """The recorded 200 responses to requests at each URI pattern, complete (`False`) and streamed (`True`).

    They come from the fewest cassettes whose responses there hold every field name and `type` value the responses
    there hold, so the mutations reach the shapes the suite recorded without parsing every cassette.

    Those holding the fewest field names come first, then the smallest. `distinct_mutations` mutates a field in the
    first recording that holds it, so the mutation meets the adapter's handling of the fields a response leaves out
    too, like the count it falls back to when no usage is reported, and replays as few events as it can.
    """
    patterns = {uri: re.compile(uri) for uri in uris}
    vocabularies: defaultdict[tuple[str, bool], dict[Path, frozenset[str]]] = defaultdict(dict)
    sizes: dict[Path, int] = {}
    for path in sorted(TESTS_DIR.rglob('*.yaml')):
        if (size := path.stat().st_size) > _MAX_CASSETTE_BYTES:
            continue
        text = path.read_text(encoding='utf-8')
        for uri, pattern in patterns.items():
            if pattern.search(text) is not None:
                sizes[path] = size
                for stream, words in _response_vocabularies(text, pattern).items():
                    vocabularies[uri, stream][path] = words
    sampled = {key: sorted(_fewest_covering(files, sizes)) for key, files in vocabularies.items()}
    interactions: dict[Path, list[dict[str, JsonValue]]] = {}
    for path in sorted({path for paths in sampled.values() for path in paths}):
        # YAML reads an unquoted timestamp as a `datetime`, which the recorded JSON had as a string, and binary data as
        # `bytes`.
        cassette = _JSON.validate_python(
            to_jsonable_python(load_cassette_yaml(path.read_text(encoding='utf-8')), bytes_mode='base64')
        )
        found = cassette.get('interactions') if isinstance(cassette, dict) else None
        interactions[path] = [item for item in found if isinstance(item, dict)] if isinstance(found, list) else []
    return {
        (uri, stream): sorted(
            (
                recording
                for path in paths
                for index, interaction in enumerate(interactions[path])
                if (
                    recording := _recording(
                        f'{path.relative_to(TESTS_DIR)}#{index}', interaction, patterns[uri], stream
                    )
                )
            ),
            key=lambda recording: (len(_field_names(recording.body)), len(json.dumps(recording.body)), recording.name),
        )
        for (uri, stream), paths in sampled.items()
    }


def _field_names(value: JsonValue) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_field_names(item) for item in value.values()))
    if isinstance(value, list):
        return set[str]().union(*(_field_names(item) for item in value))
    return set()


def _response_vocabularies(text: str, uri: re.Pattern[str]) -> dict[bool, frozenset[str]]:
    """The vocabulary of a cassette's responses to requests at `uri`, for its complete and its streamed ones."""
    words: defaultdict[bool, set[str]] = defaultdict(set)
    for interaction in text.split('\n- request:\n')[1:]:
        request, found, response = interaction.partition('\n  response:\n')
        if found and uri.search(request):
            words['event-stream' in response].update(
                word for match in _VOCABULARY.findall(response) for word in match if word
            )
    return {stream: frozenset(found) for stream, found in words.items()}


def _fewest_covering(files: dict[Path, frozenset[str]], sizes: dict[Path, int]) -> list[Path]:
    """Greedily, the fewest and smallest of `files` whose words together are all of theirs."""
    covered: set[str] = set()
    chosen: list[Path] = []
    while True:
        path = min(files, key=lambda path: (-len(files[path] - covered), sizes[path], path))
        if not files[path] - covered:
            break
        covered |= files[path]
        chosen.append(path)
    return chosen


def distinct_mutations(
    recordings: Iterable[Recording], response_type: object, event_type: object
) -> list[tuple[Recording, tuple[Mutation, ...]]]:
    """Each kind of mutation at each site, in the first of `recordings` to have it, at every place it has it.

    An API that sends a shape at a field sends it wherever the field comes, as in each event of a stream that repeats
    an item.
    """
    found: dict[tuple[MutationKind, str], tuple[Recording, tuple[Mutation, ...]]] = {}
    for recording in recordings:
        walk = (
            mutations(recording.body, response_type)
            if recording.event_names is None
            else stream_mutations(recording.events, event_type)
        )
        places: defaultdict[tuple[MutationKind, str], list[Mutation]] = defaultdict(list)
        for mutation in walk:
            places[mutation.kind, mutation.site].append(mutation)
        for key, here in places.items():
            found.setdefault(key, (recording, tuple(here)))
    return list(found.values())
