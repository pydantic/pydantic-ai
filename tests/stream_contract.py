"""The event contract of a streamed response, checked on every stream the suite iterates.

`fail_stream_contract_violations` in `tests/conftest.py` observes the events each `StreamedResponse` a test iterates
hands its consumer, through `StreamedResponse.__aiter__`, and fails the test when they break the contract:

- `delta-before-start`, `end-before-start`: a `PartDeltaEvent` or `PartEndEvent` names an index no `PartStartEvent`
  started.
- `delta-after-end`: a `PartDeltaEvent` arrives for an index after its `PartEndEvent`, which says the part is
  complete, without a new `PartStartEvent` restarting it.
- `tool-delta-without-id`: a `ToolCallPartDelta` carries no `tool_call_id`.
- `end-differs-from-stream`: a `PartEndEvent`'s part isn't the part its start and deltas built, so a consumer is told
  the part it streamed ended as something else. Replacing a streamed part with a `PartStartEvent` of the same index
  does this: `StreamedResponse.__aiter__` ends the part with its replacement.
- `native-args-not-json`: a `NativeToolCallPart`'s string args, which the adapter frames itself, don't parse.
- `stream-differs-from-get`: once the stream ends, the parts its events built aren't the parts `get()` returns.
- `replay-differs-from-get`: replaying the events through `ModelResponsePartsManager.apply_event`, as
  `CompletedStreamedResponse` does for events buffered by a durable execution engine, doesn't build what `get()`
  returns.

A start may name an index past the next one: an adapter can hold a part's slot and start the part once it's complete,
as `GeminiStreamedResponse` does for returns its grounding metadata fills. A resumed stream starts after the parts it
carries in from an earlier run, so the parts before its first start aren't compared. A part or delta of a class
defined outside `pydantic_ai` is not folded, so the check never builds a user's part subclass a test counts.

A violation an adapter is known to cause goes in `KNOWN`, with the issue that tracks its fix. A test that streams a
shape the contract forbids on purpose takes `@pytest.mark.breaks_stream_contract(reason=...)`.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import pytest
from pydantic_core import from_json

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    BaseToolCallPart,
    ModelResponsePart,
    ModelResponsePartDelta,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

StreamContractCode = Literal[
    'delta-before-start',
    'end-before-start',
    'delta-after-end',
    'tool-delta-without-id',
    'end-differs-from-stream',
    'native-args-not-json',
    'stream-differs-from-get',
    'replay-differs-from-get',
]


ViolationKey = tuple[str, StreamContractCode, str | None]
"""The stream class, the code and the part class a violation is about."""


@dataclass(frozen=True)
class StreamContractViolation:
    stream: str
    """The class of the stream whose events broke the contract: for a continuation, the segment it was streaming."""
    code: StreamContractCode
    part: str | None
    """The class of the part the violation is about, if it's about one."""
    detail: str

    @property
    def key(self) -> ViolationKey:
        return self.stream, self.code, self.part


@dataclass(frozen=True)
class Known:
    """A violation an adapter causes on `main`, kept out of the check until the issue that tracks it is fixed."""

    reason: str
    tracked_by: str
    """The issue tracking the fix."""
    repro: str
    """A test that streams the violation."""


KNOWN: dict[ViolationKey, Known] = {
    ('AnthropicStreamedResponse', 'end-differs-from-stream', 'NativeToolSearchCallPart'): Known(
        'A tool search call streams its raw input JSON, then is replaced at its block stop with the normalized '
        '`ToolSearchArgs`, so its end event carries args its deltas never built',
        tracked_by='to open: anthropic-tool-search-replaces-streamed-call',
        repro='tests/models/anthropic/test_tool_search.py::test_anthropic_native_tool_search_streaming',
    ),
    ('MistralStreamedResponse', 'end-differs-from-stream', 'ToolCallPart'): Known(
        'Each chunk replaces the tool call with the whole call so far, under a new generated id when the chunk has '
        'none, instead of streaming its args as deltas',
        tracked_by='to open: mistral-replaces-streamed-tool-call',
        repro='tests/models/test_mistral.py::test_stream_result_type_primitif_dict',
    ),
    ('MistralStreamedResponse', 'delta-after-end', 'ToolCallPart'): Known(
        'When the provider interleaves the fragments of parallel tool calls, a fragment of an earlier call arrives '
        'after the next call started, and so after the earlier call ended',
        tracked_by='to open: mistral-interleaved-tool-call-deltas',
        repro='tests/models/test_mistral.py::test_stream_tool_call_deltas',
    ),
    ('OpenAIResponsesStreamedResponse', 'delta-after-end', 'ThinkingPart'): Known(
        "A reasoning item's signature arrives with `response.output_item.done`, after its later summary parts "
        'started, so it lands on the first summary part after that part ended and the end event lacks it',
        tracked_by='to open: openai-responses-signature-after-part-end',
        repro='tests/models/test_openai_responses.py::test_openai_responses_thinking_part_iter',
    ),
    ('GeminiStreamedResponse', 'replay-differs-from-get', None): Known(
        '`ModelResponsePartsManager.apply_event` adds a newly started index at the end, so a return whose start '
        'Gemini held back until its grounding metadata arrived is replayed after the parts streamed meanwhile',
        tracked_by='to open: apply-event-replays-held-starts-out-of-order',
        repro='tests/models/google/test_native_tools.py::test_web_search_sources_go_on_the_last_web_search_return',
    ),
    ('XaiStreamedResponse', 'replay-differs-from-get', None): Known(
        'Citations and native tool results are set on parts already handed out, with no event, so a consumer that '
        'copies the events misses them',
        tracked_by='to open: xai-mutates-streamed-parts-in-place',
        repro='tests/test_citations.py::test_web_citations',
    ),
}


def comparable_part(part: ModelResponsePart) -> dict[str, object]:
    """The part's fields, without the time a return part stamps when it's built, and with tool call args parsed.

    Streamed tool call args are the JSON string the deltas built, where a complete part may hold a dict.
    """
    fields: dict[str, object] = {f.name: getattr(part, f.name) for f in dataclasses.fields(part)}
    fields.pop('timestamp', None)
    if isinstance(part, BaseToolCallPart):
        try:
            fields['args'] = part.args_as_dict()
        except ValueError:
            pass
    return {'type': type(part).__name__, **fields}


def _is_library_class(obj: object) -> bool:
    return type(obj).__module__.startswith('pydantic_ai.')


@dataclass
class StreamContractChecker:
    """Folds the events a consumer receives into the parts they build, and reports where they break the contract."""

    model_request_parameters: ModelRequestParameters
    report: Callable[[StreamContractCode, ModelResponsePart | None, str], None]
    _parts: dict[int, ModelResponsePart] = field(default_factory=dict[int, ModelResponsePart])
    _ended: set[int] = field(default_factory=set[int])
    # Indices with a part or delta of a class defined outside `pydantic_ai`, which the checker doesn't fold.
    _opaque: set[int] = field(default_factory=set[int])
    _replay: ModelResponsePartsManager | None = None

    def __post_init__(self) -> None:
        self._replay = ModelResponsePartsManager(model_request_parameters=self.model_request_parameters)

    def see(self, event: ModelResponseStreamEvent) -> None:
        if isinstance(event, PartStartEvent):
            self._see_start(event)
        elif isinstance(event, PartDeltaEvent):
            self._see_delta(event)
        elif isinstance(event, PartEndEvent):
            self._see_end(event)

    def _see_start(self, event: PartStartEvent) -> None:
        self._parts[event.index] = event.part
        self._ended.discard(event.index)
        if not _is_library_class(event.part):
            self._opaque.add(event.index)
            self._replay = None
        if self._replay is not None:
            self._replay.apply_event(event)

    def _see_delta(self, event: PartDeltaEvent) -> None:
        index, delta = event.index, event.delta
        part = self._parts.get(index)
        if part is None:
            self.report('delta-before-start', None, f'delta {index}: {delta!r}')
            self._replay = None
            return
        if index in self._ended:
            self.report('delta-after-end', part, f'delta {index}: {delta!r}')
        if isinstance(delta, ToolCallPartDelta) and not delta.tool_call_id:
            self.report('tool-delta-without-id', part, f'delta {index}: {delta!r}')
        if index in self._opaque or not _is_library_class(delta):
            self._opaque.add(index)
            self._replay = None
            return
        self._parts[index] = _apply(delta, part)
        if self._replay is not None:
            self._replay.apply_event(event)

    def _see_end(self, event: PartEndEvent) -> None:
        index = event.index
        part = self._parts.get(index)
        if part is None:
            self.report('end-before-start', None, f'end {index}: {event.part!r}')
            return
        self._ended.add(index)
        if index not in self._opaque and comparable_part(part) != comparable_part(event.part):
            self.report('end-differs-from-stream', part, f'{index}: streamed {part!r}, ended {event.part!r}')
        if isinstance(event.part, NativeToolCallPart) and isinstance(event.part.args, str) and event.part.args:
            try:
                from_json(event.part.args)
            except ValueError:
                self.report('native-args-not-json', event.part, f'{index}: {event.part.args!r}')

    def finish(self, built: Iterable[ModelResponsePart]) -> None:
        """Compare what the events built with what the stream's `get()` returns once it ends."""
        built = list(built)
        carried = min(self._parts, default=len(built))
        streamed = [
            'opaque' if index in self._opaque else _comparable_or_missing(self._parts.get(index))
            for index in range(carried, max(len(built), max(self._parts, default=-1) + 1))
        ]
        expected = [
            'opaque' if index in self._opaque else comparable_part(part)
            for index, part in enumerate(built)
            if index >= carried
        ]
        if streamed != expected:
            self.report('stream-differs-from-get', None, f'streamed {streamed!r}, get() {expected!r}')
        elif self._replay is not None and not self._opaque and carried == 0:
            replayed = [comparable_part(part) for part in self._replay.get_parts()]
            if replayed != expected:
                self.report('replay-differs-from-get', None, f'replayed {replayed!r}, get() {expected!r}')


def _apply(delta: ModelResponsePartDelta, part: ModelResponsePart) -> ModelResponsePart:
    applied = delta.apply(part)
    # A tool call delta applied to a part is a part; only a delta applied to a delta stays one.
    assert not isinstance(applied, ToolCallPartDelta)
    return applied


def _comparable_or_missing(part: ModelResponsePart | None) -> dict[str, object] | str:
    return 'never started' if part is None else comparable_part(part)


def check_stream_events(
    events: Iterable[ModelResponseStreamEvent],
    built: Iterable[ModelResponsePart],
    model_request_parameters: ModelRequestParameters | None = None,
) -> list[StreamContractCode]:
    """The codes of the violations a finished stream of `events` whose `get()` returned `built` commits, in order."""
    codes: list[StreamContractCode] = []
    checker = StreamContractChecker(
        model_request_parameters or ModelRequestParameters(), lambda code, _part, _detail: codes.append(code)
    )
    for event in events:
        checker.see(event)
    checker.finish(built)
    return codes


_OBSERVER_ATTRIBUTE = '_stream_contract_observer'


class _ObservedEvents:
    """The events a stream hands its consumer, passed through unchanged and checked on the way.

    A plain iterator rather than an async generator, so closing it closes the stream's own iterator at once. It holds
    the stream weakly and drops the stream's iterator once that's done, so it adds no reference cycle.
    """

    def __init__(
        self,
        stream: StreamedResponse,
        events: AsyncIterator[ModelResponseStreamEvent],
        violations: dict[ViolationKey, str],
    ) -> None:
        self._stream = weakref.ref(stream)
        self.events: AsyncIterator[ModelResponseStreamEvent] | None = events
        self._violations = violations
        self._checker = StreamContractChecker(stream.model_request_parameters, self._record)
        self._source = type(stream).__name__
        """The class the violations are attributed to: for a continuation, the last segment it streamed, which it
        no longer holds once it ends."""

    def _record(self, code: StreamContractCode, part: ModelResponsePart | None, detail: str) -> None:
        violation = StreamContractViolation(self._source, code, None if part is None else type(part).__name__, detail)
        self._violations.setdefault(violation.key, violation.detail)

    def __aiter__(self) -> _ObservedEvents:
        return self

    async def __anext__(self) -> ModelResponseStreamEvent:
        if self.events is None:
            raise StopAsyncIteration
        try:
            event = await self.events.__anext__()
        except StopAsyncIteration:
            self.events = None
            if (stream := self._stream()) is not None:
                self._checker.finish(stream.get().parts)
            raise
        # A continuation hands out the events of the segment it's streaming: a violation is that segment's.
        if isinstance(segment := getattr(self._stream(), '_current_sub', None), StreamedResponse):
            self._source = type(segment).__name__
        self._checker.see(event)
        return event

    async def aclose(self) -> None:
        events, self.events = self.events, None
        if isinstance(events, AsyncGenerator):
            await events.aclose()


def observe_stream_contract(monkeypatch: pytest.MonkeyPatch) -> dict[ViolationKey, str]:
    """Check the events of every stream iterated until `monkeypatch` is undone.

    Returns the violations found so far, the first detail of each, filled in as the streams are iterated.
    """
    violations: dict[ViolationKey, str] = {}
    for stream_class in (StreamedResponse, *_subclasses(StreamedResponse)):
        if (original := stream_class.__dict__.get('__aiter__')) is not None:
            monkeypatch.setattr(stream_class, '__aiter__', _observed_aiter(original, violations))
    return violations


def _observed_aiter(
    original: Callable[[StreamedResponse], AsyncIterator[ModelResponseStreamEvent]],
    violations: dict[ViolationKey, str],
) -> Callable[[StreamedResponse], AsyncIterator[ModelResponseStreamEvent]]:
    def __aiter__(self: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
        events = original(self)
        observer = vars(self).get(_OBSERVER_ATTRIBUTE)
        if isinstance(observer, _ObservedEvents):
            # A wrapper of the observed events, like the stream `FallbackModel` peeks into, already passes through it.
            return observer if events is observer.events else events
        if isinstance(events, _ObservedEvents):
            # A subclass's `__aiter__` that defers to the base class's.
            return events
        observer = _ObservedEvents(self, events, violations)
        vars(self)[_OBSERVER_ATTRIBUTE] = observer
        return observer

    return __aiter__


def _subclasses(cls: type[StreamedResponse]) -> list[type[StreamedResponse]]:
    found: list[type[StreamedResponse]] = []
    for subclass in cls.__subclasses__():
        found += [subclass, *_subclasses(subclass)]
    return found


def check_stream_contract(node: pytest.Item, violations: dict[ViolationKey, str]) -> None:
    """Fail `node` for each violation `KNOWN` doesn't hold, or for a `breaks_stream_contract` marker with none."""
    marker = node.get_closest_marker('breaks_stream_contract')
    if marker is not None:
        reason = marker.kwargs.get('reason')
        if not (isinstance(reason, str) and reason.strip()):
            pytest.fail('@pytest.mark.breaks_stream_contract requires reason=... explaining the shape it streams')
        if not violations:
            pytest.fail(
                '@pytest.mark.breaks_stream_contract is stale: this test streamed nothing the contract forbids, '
                'so remove the marker'
            )
        return
    unknown = {key: detail for key, detail in violations.items() if key not in KNOWN}
    if unknown:
        details = '\n'.join(f'{stream} [{code}] {detail}' for (stream, code, _), detail in unknown.items())
        pytest.fail(
            f'{details}\nThe events this test streamed break the contract in `tests/stream_contract.py`. Fix the '
            'stream. If an adapter causes it on `main` and an issue tracks the fix, add a `KNOWN` entry in '
            '`tests/stream_contract.py`; if this test streams such a shape on purpose, add '
            '@pytest.mark.breaks_stream_contract(reason=...)',
            pytrace=False,
        )
