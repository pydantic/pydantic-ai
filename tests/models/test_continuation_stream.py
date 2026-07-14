"""Unit tests for the streamed-continuation composite `_ContinuationStreamedResponse`.

These drive the composite directly with a scripted fake model (no HTTP, no cassettes) to
pin down the provider-agnostic stitching behavior: part-index reindexing across segments,
merged snapshots/usage, live state transitions, cancellation, and the continuation limit.

These are unit tests rather than VCR tests for two reasons: `FunctionModel` can't emit a
*suspended streaming* segment (the input a real continuation needs), and a cassette wouldn't
reliably protect this behavior anyway — our VCR matchers aren't sensitive to the reindex
payload, so a regression in the accumulate-vs-replace offset boundary could still match an
existing recording and pass green. Asserting the stitched indices directly is what catches it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models._continuation import _ContinuationStreamedResponse, merge_mode, merge_responses
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

pytestmark = pytest.mark.anyio

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)

# The exact framework-protocol keys the `FallbackModel` side stamps and this module honors. Spelled out
# here (rather than imported as module privates) since the test pins the wire contract itself.
_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
_REPLACE_PREVIOUS_RESPONSE_KEY = 'replace_previous_response'


@dataclass
class _Segment:
    """One scripted `request_stream` call: the events it yields and the response `get()` returns."""

    events: list[ModelResponseStreamEvent]
    response: ModelResponse


def _response(
    *,
    parts: list[str],
    provider_response_id: str,
    state: str,
    input_tokens: int,
    output_tokens: int,
    model_name: str = 'fake',
    metadata: dict[str, Any] | None = None,
    provider_details: dict[str, Any] | None = None,
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=p) for p in parts],
        model_name=model_name,
        provider_name='fake',
        provider_response_id=provider_response_id,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        state=state,  # type: ignore[arg-type]
        timestamp=_TIMESTAMP,
        metadata=metadata,
        provider_details=provider_details,
    )


def _starts(*parts: tuple[int, str]) -> list[ModelResponseStreamEvent]:
    return [PartStartEvent(index=index, part=TextPart(content=content)) for index, content in parts]


@dataclass
class _FakeStream(StreamedResponse):
    """A `StreamedResponse` that replays scripted events and returns a fixed `get()` response."""

    _events: list[ModelResponseStreamEvent]
    _response: ModelResponse
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Populate the id up-front so the composite can resolve replace-vs-accumulate on the
        # first event, mirroring a resumed provider stream that knows its id at the start.
        self.provider_response_id = self._response.provider_response_id

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for event in self._events:
            if self._closed:
                break
            yield event

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return self._get_event_iterator()

    def get(self) -> ModelResponse:
        return self._response

    async def close_stream(self) -> None:
        self._closed = True

    @property
    def model_name(self) -> str:
        return self._response.model_name or 'fake'

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp


class _FakeModel(Model):
    """A `Model` whose `request_stream` replays a queue of scripted segments."""

    def __init__(self, segments: list[_Segment]) -> None:
        super().__init__()
        self.segments = segments
        self.cancelled: list[ModelResponse] = []

    @property
    def model_name(self) -> str:
        return 'fake'

    @property
    def system(self) -> str:
        return 'fake'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise NotImplementedError

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        segment = self.segments.pop(0)
        yield _FakeStream(model_request_parameters, segment.events, segment.response)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response)

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if (response.provider_details or {}).get('background'):
            return 0.5
        return None


async def _no_sleep(delay: float) -> None:  # pragma: no cover - scripted responses have no delay
    return None


def _composite(
    model: _FakeModel,
    *,
    max_generation_continuations: int = 10,
    max_background_polls: int = 1000,
    sleep_func: Callable[[float], Awaitable[None]] = _no_sleep,
    check_usage: Callable[[RequestUsage], None] = lambda usage: None,
) -> _ContinuationStreamedResponse:
    return _ContinuationStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        model=model,
        model_settings=None,
        base_messages=[],
        run_context=None,
        max_generation_continuations=max_generation_continuations,
        max_background_polls=max_background_polls,
        sleep_func=sleep_func,
        check_usage=check_usage,
    )


def _poll_segments(*, count: int, provider_response_id: str) -> list[_Segment]:
    """`count` empty *replace*-style (same-id) suspended polls followed by a completing segment.

    Models a background job (OpenAI background mode) that stays pending across many polls under one
    `provider_response_id` and carries no new content while pending, then finally completes.
    """
    segments = [
        _Segment(
            events=[],
            response=_response(
                parts=[], provider_response_id=provider_response_id, state='suspended', input_tokens=1, output_tokens=0
            ),
        )
        for _ in range(count)
    ]
    segments.append(
        _Segment(
            events=_starts((0, 'done')),
            response=_response(
                parts=['done'],
                provider_response_id=provider_response_id,
                state='complete',
                input_tokens=1,
                output_tokens=1,
            ),
        )
    )
    return segments


def _part_start_indices(events: list[ModelResponseStreamEvent]) -> list[int]:
    return [event.index for event in events if isinstance(event, PartStartEvent)]


async def test_accumulate_offsets_indices_and_sums_usage() -> None:
    """Anthropic-style accumulate: the second segment's indices are offset past the first's parts."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a'), (1, 'b')),
                response=_response(
                    parts=['a', 'b'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'c')),
                response=_response(
                    parts=['c'], provider_response_id='r2', state='complete', input_tokens=2, output_tokens=3
                ),
            ),
        ]
    )
    stream = _composite(model)
    events = [event async for event in stream]

    # First segment keeps indices [0, 1]; second segment is offset by len(first.parts) == 2 → [2].
    assert _part_start_indices(events) == [0, 1, 2]

    merged = stream.get()
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['a', 'b', 'c']
    assert merged.usage == RequestUsage(input_tokens=3, output_tokens=4)
    assert merged.state == 'complete'


async def test_replace_reuses_indices_and_replaces_response() -> None:
    """OpenAI-style replace (same provider_response_id): the second segment reuses the index space."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'a'), (1, 'final')),
                response=_response(
                    parts=['a', 'final'],
                    provider_response_id='r1',
                    state='complete',
                    input_tokens=5,
                    output_tokens=6,
                ),
            ),
        ]
    )
    stream = _composite(model)
    events = [event async for event in stream]

    # Offset stays 0: the second segment reuses indices [0, 1] rather than appending.
    assert _part_start_indices(events) == [0, 0, 1]

    merged = stream.get()
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['a', 'final']
    assert merged.usage == RequestUsage(input_tokens=5, output_tokens=6)
    assert merged.state == 'complete'


def test_merge_accumulates_turn_scoped_provider_details() -> None:
    """`provider_details` accumulate across segments (latest-wins) so turn-scoped keys aren't lost.

    Turn-scoped identifiers a later segment may omit — OpenAI `background`/`conversation_id`, Anthropic
    `container_id` — must survive the merge, or e.g. `cancel_suspended_response` couldn't reach the
    server-side job. Driven at the merge level since the gap is a provider-detail race, not stitching.
    """
    existing = ModelResponse(
        parts=[TextPart('a')],
        provider_response_id='r1',
        provider_details={'background': True, 'conversation_id': 'conv_1', 'container_id': 'cont_1'},
    )

    # Replace (same id): the resumed segment carries its own `finish_reason` but none of the turn keys.
    replaced = merge_responses(
        existing,
        ModelResponse(
            parts=[TextPart('a2')], provider_response_id='r1', provider_details={'finish_reason': 'completed'}
        ),
    )
    assert replaced.provider_details == {
        'background': True,
        'conversation_id': 'conv_1',
        'container_id': 'cont_1',
        'finish_reason': 'completed',
    }

    # Accumulate (different id): the new segment has no `provider_details` at all — all keys carry over.
    accumulated = merge_responses(existing, ModelResponse(parts=[TextPart('b')], provider_response_id='r2'))
    assert accumulated.provider_details == {'background': True, 'conversation_id': 'conv_1', 'container_id': 'cont_1'}

    # `new` wins on conflicts (a refreshed value replaces the earlier one).
    updated = merge_responses(
        existing,
        ModelResponse(parts=[TextPart('c')], provider_response_id='r1', provider_details={'conversation_id': 'conv_2'}),
    )
    assert updated.provider_details == {'background': True, 'conversation_id': 'conv_2', 'container_id': 'cont_1'}


def test_merge_preserves_metadata_pin() -> None:
    """`metadata` accumulates across segments (latest-wins) so the `FallbackModel` pin isn't lost.

    The pin (`metadata['__pydantic_ai__']['fallback_model_id']`) is only stamped on a segment that
    *ends* suspended, so a later segment (e.g. an in-flight cancel snapshot) omits it. If the merge
    took `metadata` wholesale from `new`, the pin would be nuked and `FallbackModel.cancel_suspended_response`
    couldn't resolve the model that owns the server-side job. Driven at the merge level since the gap
    is a metadata race, not stitching.
    """
    pin = {'__pydantic_ai__': {'fallback_model_id': 'inner'}}
    existing = ModelResponse(parts=[TextPart('a')], provider_response_id='r1', metadata=pin)

    # Replace (same id): the resumed segment carries no `metadata` — the pin carries over.
    replaced = merge_responses(existing, ModelResponse(parts=[TextPart('a2')], provider_response_id='r1'))
    assert replaced.metadata == pin

    # Accumulate (different id): the new segment has no `metadata` at all — the pin carries over.
    accumulated = merge_responses(existing, ModelResponse(parts=[TextPart('b')], provider_response_id='r2'))
    assert accumulated.metadata == pin

    # `new` wins on conflicts (a refreshed pin replaces the earlier one).
    updated = merge_responses(
        existing,
        ModelResponse(
            parts=[TextPart('c')],
            provider_response_id='r1',
            metadata={'__pydantic_ai__': {'fallback_model_id': 'other'}},
        ),
    )
    assert updated.metadata == {'__pydantic_ai__': {'fallback_model_id': 'other'}}


async def test_snapshot_preserves_metadata_pin_across_inflight_segment() -> None:
    """A mid-flight `get()` snapshot keeps the prior suspended segment's `FallbackModel` pin.

    The in-flight segment hasn't stamped its own pin yet (that happens post-yield when it ends
    suspended), so its `metadata` is `None`; folding it into the accumulator must not drop the pin
    the earlier suspended segment carried, or the OpenAI background job would leak on cancel.
    """
    pin = {'__pydantic_ai__': {'fallback_model_id': 'inner'}}
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'],
                    provider_response_id='r1',
                    state='suspended',
                    input_tokens=1,
                    output_tokens=1,
                    metadata=pin,
                ),
            ),
            _Segment(
                events=_starts((0, 'b'), (1, 'c')),
                response=_response(
                    parts=['b', 'c'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()  # first (suspended) segment's only event
    await iterator.__anext__()  # second segment now in flight (its `get()` has no pin)

    snapshot = stream.get()
    assert snapshot.state == 'incomplete'  # a segment is still in flight → this is a live snapshot
    assert snapshot.metadata == pin


async def test_model_change_replaces_indices() -> None:
    """A segment from a different model *replaces* rather than accumulates, mirroring `merge_mode`.

    Unreachable through the public API today — every continuation pins to a single model (Anthropic and
    OpenAI reuse it, `FallbackModel` pins to the producing model) — so this drives the composite directly
    to pin the accumulate-vs-replace boundary that `_segment_offset` shares with `merge_mode`.
    """
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a'), (1, 'b')),
                response=_response(
                    parts=['a', 'b'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'x')),
                response=_response(
                    parts=['x'],
                    provider_response_id='r2',
                    state='complete',
                    input_tokens=2,
                    output_tokens=3,
                    model_name='other',
                ),
            ),
        ]
    )
    stream = _composite(model)
    events = [event async for event in stream]

    # A different `model_name` (with a fresh id) is a replace signal, so the second segment reuses the
    # index space (offset stays 0) rather than appending past the first segment's parts.
    assert _part_start_indices(events) == [0, 1, 0]

    merged = stream.get()
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['x']
    assert merged.model_name == 'other'
    assert merged.state == 'complete'


async def test_replace_new_after_accumulation_reindexes_from_zero() -> None:
    """A `replace-new` segment following *multiple* accumulate segments emits its parts from index 0.

    `merge_responses` drops all prior parts on a replace, so the merged response indexes the replacing
    segment's parts from 0. Its streamed events must match: reusing the *last* segment's offset (which is
    non-zero once at least two segments accumulated) would drift the live event indices past the final
    response's. A single prior segment can't catch this because the offset is still 0 — two are needed.
    """
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'x')),
                response=_response(
                    parts=['x'],
                    provider_response_id='r3',
                    state='complete',
                    input_tokens=1,
                    output_tokens=1,
                    model_name='other',
                ),
            ),
        ]
    )
    stream = _composite(model)
    events = [event async for event in stream]

    # Two accumulates put the second part at offset 1; the replacing third segment resets to offset 0
    # (not 1), matching the merged response that keeps only its part at index 0.
    assert _part_start_indices(events) == [0, 1, 0]

    merged = stream.get()
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['x']


async def test_get_state_transitions_incomplete_then_complete() -> None:
    """`get()` reports `incomplete` while a segment is in flight and `complete` once the loop exits."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()
    assert stream.get().state == 'incomplete'

    async for _ in iterator:
        pass
    assert stream.get().state == 'complete'


async def test_cancel_stops_loop_and_cancels_suspended_response() -> None:
    """Cancelling mid-segment cancels the server-side job once and issues no further continuation."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a'), (1, 'b')),
                response=_response(
                    parts=['a', 'b'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'c')),
                response=_response(
                    parts=['c'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()
    await stream.cancel()

    # Draining after cancel must not request the second (still-queued) segment.
    async for _ in iterator:
        pass

    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r1'
    assert len(model.segments) == 1  # second segment was never requested
    assert stream.get().state == 'interrupted'


async def test_close_stream_cancels_job_even_if_substream_teardown_fails() -> None:
    """A failing sub-stream connection teardown must not skip the server-side cancel (no job leak)."""

    class _RaisingCloseStream(_FakeStream):
        async def close_stream(self) -> None:
            raise RuntimeError('connection teardown failed')

    class _RaisingModel(_FakeModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: object | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.segments.pop(0)
            yield _RaisingCloseStream(model_request_parameters, segment.events, segment.response)

    model = _RaisingModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()
    await iterator.__anext__()

    # The teardown failure still surfaces, but the server-side job was cancelled all the same.
    with pytest.raises(RuntimeError, match='connection teardown failed'):
        await stream.cancel()
    assert len(model.cancelled) == 1


async def test_metadata_properties_track_current_segment_then_merged() -> None:
    """`model_name`/`provider_name`/`provider_url`/`timestamp` read the in-flight segment mid-stream,
    then fall back to the merged response once the continuation loop completes."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    assert model.model_name == 'fake'
    assert model.system == 'fake'

    stream = _composite(model)
    iterator = stream.__aiter__()
    # `__aiter__` is idempotent: a second call returns the already-built iterator.
    assert stream.__aiter__() is iterator

    await iterator.__anext__()  # first (suspended) segment now in flight → `_current_sub` is set
    assert stream.model_name == 'fake'
    assert stream.provider_name == 'fake'
    assert stream.provider_url is None
    assert stream.timestamp == _TIMESTAMP

    async for _ in iterator:
        pass
    # Loop done → `_current_sub` is None, so the properties read the merged response instead.
    assert stream.model_name == 'fake'
    assert stream.provider_name == 'fake'
    assert stream.provider_url is None
    assert stream.timestamp == _TIMESTAMP


async def test_segment_transport_error_propagates_when_not_cancelled() -> None:
    """A genuine transport error from a segment (not caused by `cancel()`) propagates out of the composite."""

    class _ExplodingStream(_FakeStream):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            raise httpx.ReadError('boom')
            yield  # pragma: no cover - unreachable, marks this a generator

    class _ExplodingModel(_FakeModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: object | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.segments[0]
            yield _ExplodingStream(model_request_parameters, segment.events, segment.response)

    model = _ExplodingModel(
        [
            _Segment(
                events=[],
                response=_response(
                    parts=[], provider_response_id='r1', state='complete', input_tokens=0, output_tokens=0
                ),
            ),
        ]
    )
    stream = _composite(model)

    with pytest.raises(httpx.ReadError, match='boom'):
        async for _ in stream:
            pass


async def test_cancel_suppresses_segment_transport_error() -> None:
    """When `cancel()` tears down an in-flight segment and the sub-stream raises a transport error, the guard suppresses it."""

    class _CancelRaisingStream(_FakeStream):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            yield self._events[0]
            # After the first event, `cancel()` has torn the connection down, so the next pull
            # raises a transport error — exactly what a real provider stream does mid-flight.
            raise httpx.StreamClosed()

    class _CancelRaisingModel(_FakeModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: object | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.segments.pop(0)
            yield _CancelRaisingStream(model_request_parameters, segment.events, segment.response)

    model = _CancelRaisingModel(
        [
            _Segment(
                events=_starts((0, 'a'), (1, 'b')),
                response=_response(
                    parts=['a', 'b'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()  # first segment in flight
    await stream.cancel()

    # Draining resumes the closed segment, which raises the transport error; the guard suppresses it
    # (composite is cancelled) rather than propagating.
    async for _ in iterator:
        pass

    assert stream.get().state == 'interrupted'
    assert len(model.cancelled) == 1


async def test_close_stream_after_completion_cancels_job() -> None:
    """`close_stream()` with no in-flight segment skips the sub teardown but still cancels the server-side job."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    async for _ in stream:
        pass

    # After the loop `_current_sub` is None, so `close_stream` goes straight to cancelling the job.
    await stream.close_stream()
    assert len(model.cancelled) == 1


async def test_aclose_reraises_unexpected_runtime_error() -> None:
    """`aclose()` swallows only the "async generator is already running" `RuntimeError`; any other propagates.

    This is a unit test rather than a VCR test because the defensive re-raise branch can't be reached
    through the public API: it fires only if the inner segment generator's own `aclose()` raises an
    unrelated `RuntimeError`, which no real provider stream produces. We pin it by assigning a stub
    iterator directly onto the instance — legitimate here since we're exercising exactly that branch.
    """

    class _BoomIterator:
        async def aclose(self) -> None:
            raise RuntimeError('boom')

    model = _FakeModel([])
    stream = _composite(model)
    stream._segment_iterator = _BoomIterator()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match='boom'):
        await stream.aclose()


async def test_exceeding_max_generation_continuations_raises() -> None:
    """A model past `max_generation_continuations` raises `UnexpectedModelBehavior`."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model, max_generation_continuations=1)

    with pytest.raises(UnexpectedModelBehavior, match='suspended more than the maximum'):
        async for _ in stream:
            pass


async def test_replace_poll_chain_runs_past_max_generation_continuations() -> None:
    """A same-id background poll chain is bounded by `max_background_polls`, not
    `max_generation_continuations`, so a healthy long-running job isn't killed after 10 continuations."""
    # Eight same-id polls, far past the strict cap of 2, then completion.
    model = _FakeModel(_poll_segments(count=8, provider_response_id='job1'))
    stream = _composite(model, max_generation_continuations=2, max_background_polls=1000)

    events = [event async for event in stream]

    # Only the completing segment carries a part; every poll was empty.
    assert _part_start_indices(events) == [0]
    merged = stream.get()
    assert merged.state == 'complete'
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['done']
    # The chain completed without raising and without cancelling the (healthy) job.
    assert model.cancelled == []
    assert model.segments == []


async def test_replace_poll_chain_bounded_by_replace_ceiling() -> None:
    """The replace backstop still exists: a job stuck returning the same suspended id forever eventually raises.

    Usage limits wouldn't catch this (a pending poll adds no tokens), so `max_background_polls` is
    the last-resort guard against a provider wedged on one `provider_response_id`.
    """
    # Never completes: every poll stays suspended under the same id.
    model = _FakeModel(
        [
            _Segment(
                events=[],
                response=_response(
                    parts=[], provider_response_id='job1', state='suspended', input_tokens=1, output_tokens=0
                ),
            )
            for _ in range(10)
        ]
    )
    stream = _composite(model, max_generation_continuations=2, max_background_polls=3)

    with pytest.raises(UnexpectedModelBehavior, match='remained suspended after polling the maximum'):
        async for _ in stream:
            pass

    # A stuck poll still gets its server-side job cancelled before the raise, so it doesn't leak.
    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'job1'


async def test_model_change_replace_chain_counts_against_strict_ceiling() -> None:
    """A model-change replace chain counts against `max_generation_continuations`, not the generous backstop.

    A replace caused by a model change (or a `FallbackModel` directive) is *fresh* generation, not a
    passive re-poll of one background job, so it must not inherit the generous `max_background_polls`
    ceiling — that would be the exact runaway (different models endlessly returning fresh suspensions) the
    strict cap guards against. Only a *same-id* replace gets the generous backstop.
    """
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'],
                    provider_response_id='r1',
                    state='suspended',
                    input_tokens=1,
                    output_tokens=1,
                    model_name='m1',
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'],
                    provider_response_id='r2',
                    state='suspended',
                    input_tokens=1,
                    output_tokens=1,
                    model_name='m2',
                ),
            ),
            _Segment(
                events=_starts((0, 'c')),
                response=_response(
                    parts=['c'],
                    provider_response_id='r3',
                    state='suspended',
                    input_tokens=1,
                    output_tokens=1,
                    model_name='m3',
                ),
            ),
        ]
    )
    # A generous replace ceiling that a same-id poll chain would sail past — but a model-change chain must
    # be bound by the strict `max_generation_continuations=2` instead and raise, naming the current job id.
    stream = _composite(model, max_generation_continuations=2, max_background_polls=1000)

    with pytest.raises(UnexpectedModelBehavior, match=r"Model response 'r3' was suspended more than the maximum"):
        async for _ in stream:
            pass


def test_replace_previous_marker_forces_replace_and_is_stripped() -> None:
    """A `new` response carrying the `FallbackModel` `replace_previous_response` marker merges as *replace*.

    The marker (`metadata['__pydantic_ai__']['replace_previous_response']`) means "this response supersedes
    the suspended turn, do not accumulate" — even when neither the id nor the model would otherwise signal a
    replace. It's transient: honored as a replace, then popped so it can't persist into history and wrongly
    force a later legitimate `pause_turn` continuation to replace. Other `__pydantic_ai__` keys survive.
    """
    existing = ModelResponse(parts=[TextPart('partial')], provider_response_id='r1')
    # Same model, fresh id → would normally *accumulate*; the marker flips it to replace.
    marked = ModelResponse(
        parts=[TextPart('fresh')],
        provider_response_id='r2',
        metadata={
            _PYDANTIC_AI_METADATA_KEY: {_REPLACE_PREVIOUS_RESPONSE_KEY: True, 'fallback_model_id': 'inner'},
        },
    )

    assert merge_mode(existing, marked) == 'replace-new'

    merged = merge_responses(existing, marked)
    # Replaced, not accumulated: the prior 'partial' part is gone.
    assert [p.content for p in merged.parts if isinstance(p, TextPart)] == ['fresh']
    # The marker was popped, but the sibling continuation pin under the same namespace survives.
    assert merged.metadata == {_PYDANTIC_AI_METADATA_KEY: {'fallback_model_id': 'inner'}}

    # A follow-up merge of the stripped result must not still see a replace directive.
    assert merge_mode(merged, ModelResponse(parts=[TextPart('more')], provider_response_id='r3')) == 'accumulate'


async def test_cancel_teardown_survives_scope_cancellation() -> None:
    """A cancellation injected while parked in the cancel-teardown still runs the job cancel to completion.

    When a workflow/task cancellation is what tears the run down, awaiting the (activity-wrapped) job cancel
    from inside the cancelled scope would otherwise raise `CancelledError` before the cancel runs, leaking the
    server-side job. The shielded teardown must let the cancel finish before the cancellation propagates.
    """

    class _SlowCancelModel(_FakeModel):
        def __init__(self, segments: list[_Segment]) -> None:
            super().__init__(segments)
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.completed = False

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            self.started.set()
            await self.release.wait()  # park mid-teardown so the outer scope can be cancelled
            self.cancelled.append(response)
            self.completed = True

    model = _SlowCancelModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()
    await iterator.__anext__()  # first (suspended) segment in flight

    close_task = asyncio.ensure_future(stream.close_stream())
    await model.started.wait()  # teardown has begun and is parked inside the shielded cancel
    close_task.cancel()  # inject the scope cancellation mid-teardown
    model.release.set()  # let the shielded cancel finish

    with pytest.raises(asyncio.CancelledError):
        await close_task

    # Despite the injected cancel, the server-side job cancel ran to completion (no leak).
    assert model.completed
    assert [r.provider_response_id for r in model.cancelled] == ['r1']


async def test_detach_records_suspended_when_job_pending() -> None:
    """Detaching (consumer stops iterating, `aclose()` without `cancel()`) with a still-pending suspended
    segment makes `get()` report `'suspended'` — resumable — without cancelling the server-side job."""
    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'partial')),
                response=_response(
                    parts=['partial'], provider_response_id='job1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'done')),
                response=_response(
                    parts=['done'], provider_response_id='job1', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()  # first (suspended) segment now in flight
    # Mid-stream, before any detach, this is a live snapshot → `'incomplete'`.
    assert stream.get().state == 'incomplete'

    await stream.aclose()  # detach: tear down the connection WITHOUT cancelling the job

    recorded = stream.get()
    assert recorded.state == 'suspended'  # resumable
    assert recorded.provider_response_id == 'job1'
    assert model.cancelled == []  # detach must NOT cancel the server-side job
    assert len(model.segments) == 1  # detaching did not poll again


async def test_later_segment_error_cancels_suspended_job() -> None:
    """A later segment failing cancels the server-side job in hand, mirroring the non-streaming loop.

    The first segment ends suspended (a live server-side job); if a subsequent segment then raises,
    the composite must best-effort cancel that job before the error propagates, or history records an
    unresumable, uncancellable `'interrupted'` turn and the job leaks.
    """

    class _ExplodingStream(_FakeStream):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            raise RuntimeError('segment 2 boom')
            yield  # pragma: no cover - unreachable, marks this a generator

    class _SecondSegmentExplodesModel(_FakeModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: object | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.segments.pop(0)
            stream_cls = _ExplodingStream if not self.segments else _FakeStream
            yield stream_cls(model_request_parameters, segment.events, segment.response)

    model = _SecondSegmentExplodesModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)

    with pytest.raises(RuntimeError, match='segment 2 boom'):
        async for _ in stream:
            pass

    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r1'


async def test_check_usage_error_cancels_suspended_job() -> None:
    """A `check_usage` failure between segments cancels the still-suspended server-side job."""

    def _reject_second(usage: RequestUsage) -> None:
        if usage.output_tokens > 1:  # first segment alone is 1; the merge with segment 2 trips this
            raise UnexpectedModelBehavior('usage limit exceeded')

    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=5
                ),
            ),
        ]
    )
    stream = _composite(model, check_usage=_reject_second)

    with pytest.raises(UnexpectedModelBehavior, match='usage limit exceeded'):
        async for _ in stream:
            pass

    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r1'


async def test_cancel_during_between_segment_sleep_skips_next_request() -> None:
    """A `cancel()` during the between-segment retry sleep must not open the next sub-stream.

    The loop re-checks `_cancelled`/`_stopped` after the sleep, so a cancel that lands while parked in
    `sleep_func` breaks cleanly instead of resuming a just-cancelled job (which for Anthropic `pause_turn`
    would actively continue generation and burn tokens). Unlike `test_cancel_stops_loop_...`, this uses a
    real (non-zero) delay so the cancel lands *during* the sleep, exercising the post-sleep re-check.
    """
    holder: list[_ContinuationStreamedResponse] = []

    async def _cancel_during_sleep(delay: float) -> None:
        await holder[0].cancel()

    model = _FakeModel(
        [
            _Segment(
                events=_starts((0, 'a')),
                response=_response(
                    parts=['a'],
                    provider_response_id='r1',
                    state='suspended',
                    input_tokens=1,
                    output_tokens=1,
                    provider_details={'background': True},
                ),
            ),
            _Segment(
                events=_starts((0, 'b')),
                response=_response(
                    parts=['b'], provider_response_id='r2', state='complete', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model, sleep_func=_cancel_during_sleep)
    holder.append(stream)

    async for _ in stream:
        pass

    assert len(model.segments) == 1  # the second segment was never requested
    assert len(model.cancelled) == 1
    assert model.cancelled[0].provider_response_id == 'r1'
    assert stream.get().state == 'interrupted'


async def test_cancel_suppresses_non_httpx_segment_error() -> None:
    """`cancel()` tearing down a non-httpx sub-stream suppresses the sub's own transport error type.

    The composite's cancel-guard defaults to httpx errors only; a sub on a different transport
    (Bedrock botocore, xAI grpc) reports its cancel errors via its own `get_stream_cancel_errors()`.
    The composite must consult the in-flight sub's error types, or the error escapes to the user
    instead of a clean `'interrupted'` stop.
    """

    class _BotoError(Exception):
        pass

    class _NonHttpxCancelStream(_FakeStream):
        def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
            return (_BotoError,)

        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            yield self._events[0]
            # After the first event, `cancel()` tore the connection down; the next pull raises the
            # transport's own (non-httpx) error, exactly what a real botocore/grpc stream does.
            raise _BotoError('connection torn down')

    class _NonHttpxModel(_FakeModel):
        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: object | None = None,
        ) -> AsyncGenerator[StreamedResponse]:
            segment = self.segments.pop(0)
            yield _NonHttpxCancelStream(model_request_parameters, segment.events, segment.response)

    model = _NonHttpxModel(
        [
            _Segment(
                events=_starts((0, 'a'), (1, 'b')),
                response=_response(
                    parts=['a', 'b'], provider_response_id='r1', state='suspended', input_tokens=1, output_tokens=1
                ),
            ),
        ]
    )
    stream = _composite(model)
    iterator = stream.__aiter__()

    await iterator.__anext__()  # first segment in flight
    await stream.cancel()

    # Draining resumes the closed segment, which raises the non-httpx error; the guard must suppress it.
    async for _ in iterator:
        pass

    assert stream.get().state == 'interrupted'
    assert len(model.cancelled) == 1
