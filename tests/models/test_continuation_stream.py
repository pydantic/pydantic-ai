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

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
from pydantic_ai.models._continuation import _ContinuationStreamedResponse
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

pytestmark = pytest.mark.anyio

_TIMESTAMP = datetime(2024, 1, 1, tzinfo=timezone.utc)


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
) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content=p) for p in parts],
        model_name='fake',
        provider_name='fake',
        provider_response_id=provider_response_id,
        usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        state=state,  # type: ignore[arg-type]
        timestamp=_TIMESTAMP,
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


async def _no_sleep(delay: float) -> None:  # pragma: no cover - scripted responses have no delay
    return None


def _composite(model: _FakeModel, *, max_continuations: int = 50) -> _ContinuationStreamedResponse:
    return _ContinuationStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        model=model,
        model_settings=None,
        base_messages=[],
        run_context=None,
        max_continuations=max_continuations,
        sleep_func=_no_sleep,
        check_usage=lambda usage: None,
    )


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


async def test_exceeding_max_continuations_raises() -> None:
    """A model that stays suspended past `max_continuations` raises `UnexpectedModelBehavior`."""
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
    stream = _composite(model, max_continuations=1)

    with pytest.raises(UnexpectedModelBehavior, match='suspended more than the maximum'):
        async for _ in stream:
            pass
