"""Model-continuation primitives shared by the agent graph.

A *continuation* happens when a model returns a `ModelResponse` with
`state == 'suspended'` (Anthropic `pause_turn`, OpenAI background mode, …): the
graph re-issues the request with the suspended response echoed back, and the
provider resumes the same logical turn. This module owns the provider-agnostic
glue for stitching those segments back into a single response/stream:

- [`merge_responses`][pydantic_ai.models._continuation.merge_responses] folds a
  continuation response into the one it continues.
- [`merge_mode`][pydantic_ai.models._continuation.merge_mode] reports whether a
  continuation *replaces* or *accumulates*, so the streamed composite can reindex
  parts consistently with the merge.
- [`_ContinuationStreamedResponse`][pydantic_ai.models._continuation._ContinuationStreamedResponse]
  drives the streamed loop, presenting every segment as one continuous stream.

This module is deliberately decoupled from `_agent_graph`: it imports only from
`models`, `messages`, `usage`, `exceptions`, and the stdlib. Pluggable timing is
injected as `sleep_func` so the loop stays free of `now_utc()`/RNG and replays
deterministically under durable executors (e.g. Temporal).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

from .. import _utils
from .._run_context import RunContext
from ..exceptions import UnexpectedModelBehavior
from ..messages import (
    FinalResultEvent,
    ModelMessage,
    ModelResponse,
    ModelResponseState,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
)
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import Model, StreamedResponse

__all__ = ['MAX_CONTINUATIONS', 'MergeMode', 'merge_mode', 'merge_responses', '_ContinuationStreamedResponse']

MAX_CONTINUATIONS = 50
"""Maximum number of continuation segments for a single model turn.

Guards against a provider that never leaves the `'suspended'` state from looping
forever; exceeding it raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
"""

MergeMode = Literal['replace', 'accumulate']

# Deterministic fallback for `timestamp` before any segment has streamed. This is
# never reached in practice (a segment is always in flight or finalized by the time
# `timestamp` is read), but keeps the loop free of `now_utc()` for durable replay.
_FALLBACK_TIMESTAMP = datetime.fromtimestamp(0, tz=timezone.utc)


def merge_mode(existing: ModelResponse, new: ModelResponse) -> MergeMode:
    """Whether `new` replaces `existing` wholesale or accumulates onto it.

    `'replace'` when both responses share a `provider_response_id` (e.g. an OpenAI
    background `retrieve`, which returns the full response so far) or when the model
    changed between them (accumulating parts from different models is always wrong).
    `'accumulate'` otherwise (e.g. Anthropic `pause_turn`, which appends new parts).
    """
    if existing.provider_response_id and existing.provider_response_id == new.provider_response_id:
        return 'replace'
    if existing.model_name and new.model_name and existing.model_name != new.model_name:
        return 'replace'
    return 'accumulate'


def merge_responses(existing: ModelResponse, new: ModelResponse) -> ModelResponse:
    """Merge a continuation response into the one it continues.

    If same `provider_response_id`, replace entirely with the new response.
    If the model changed between responses, replace entirely (incompatible responses should not be merged).
    Otherwise, accumulate parts, sum usage, and use other fields from the new response.
    """
    if merge_mode(existing, new) == 'replace':
        return new

    # Same model, different response → accumulate parts and sum usage.
    # Preserve existing provider response IDs when continuation responses omit them
    # (e.g. resumed OpenAI streams that start after a sequence number).
    return replace(
        new,
        parts=[*existing.parts, *new.parts],
        usage=existing.usage + new.usage,
        provider_response_id=new.provider_response_id or existing.provider_response_id,
    )


@dataclass
class _ContinuationStreamedResponse(StreamedResponse):
    """A [`StreamedResponse`][pydantic_ai.models.StreamedResponse] that stitches continuation segments into one stream.

    Each segment is an ordinary `model.request_stream(...)` sub-stream. Their events
    are re-emitted as a single continuous stream, with part indices offset so parts
    from accumulated segments (Anthropic `pause_turn`) don't collide, while replaced
    segments (OpenAI background `retrieve`) keep reusing the same index space.

    `get()` returns the live merged snapshot at any point; `usage` sums/replaces in
    lockstep with the merge so the graph accounts for it exactly once.
    """

    model: Model
    model_settings: ModelSettings | None
    base_messages: list[ModelMessage]
    run_context: RunContext[Any] | None
    max_continuations: int
    sleep_func: Callable[[float], Awaitable[None]]
    check_usage: Callable[[RequestUsage], None]
    initial_suspended_response: ModelResponse | None = None
    # Entered around each segment's `model.request_stream(...)`. The agent graph passes a factory
    # that re-attaches the ambient context (e.g. the OTel `chat` span opened by `wrap_model_request`
    # in a separate task) so span updates driven by `get_current_span()` land on the right span even
    # though segments are opened lazily in the consumer task. Opaque to this module (no OTel coupling).
    segment_context: Callable[[], AbstractContextManager[Any]] = nullcontext

    _merged_response: ModelResponse | None = field(default=None, init=False)
    _current_sub: StreamedResponse | None = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)
    # The inner segment-stitching generator, kept separately so `aclose()` can tear it down
    # directly: the outer cancel-guard's `async for … in iterator` does NOT forward `aclose()`
    # to it, and this generator owns each segment's `async with request_stream(...)`.
    _segment_iterator: AsyncGenerator[ModelResponseStreamEvent, None] | None = field(default=None, init=False)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream every segment as one continuous event stream.

        This intentionally bypasses the base `StreamedResponse.__aiter__`'s `iterator_with_final_event`
        / `iterator_with_part_end` wrappers: each sub-stream is already wrapped by them, so it emits
        fully-formed `PartStart`/`PartDelta`/`PartEnd` and `FinalResultEvent`s. The composite only
        applies reindexing + final-result capture (inside `_get_event_iterator`) and the cancel-guard
        (reproducing the base `_finished`/`_cancelled` transitions) on top.

        One minor semantic gap: `PartEndEvent.next_part_kind` is `None` at each sub-stream boundary
        (a segment can't see the next segment's first part), whereas a single-segment stream would
        populate it. This is acceptable because parts never merge across segment boundaries.
        """
        if self._event_iterator is None:
            self._segment_iterator = self._get_event_iterator()
            self._event_iterator = self._iterator_with_cancel_guard(self._segment_iterator)
        return self._event_iterator

    async def _iterator_with_cancel_guard(
        self, iterator: AsyncIterator[ModelResponseStreamEvent]
    ) -> AsyncIterator[ModelResponseStreamEvent]:
        # Mirror `StreamedResponse.__aiter__`'s cancel-guard: suppress transport
        # errors caused by `cancel()` tearing down an in-flight sub-stream, and only
        # flip `_finished` on a natural `StopAsyncIteration` of a stream that wasn't
        # cancelled, so an early `break`/`aclose()`/in-flight error — or a `cancel()`
        # that still drains to completion — leaves `get()` reporting `'incomplete'`/
        # `'interrupted'` rather than `'complete'`.
        try:
            async for event in iterator:
                yield event
        except self.get_stream_cancel_errors():
            if not self.cancelled:
                raise
        else:
            if not self._cancelled:
                self._finished = True

    async def _get_event_iterator(self) -> AsyncGenerator[ModelResponseStreamEvent, None]:
        iteration = 0
        response = self.initial_suspended_response
        # Index at which the most recent segment's parts began in the stitched stream. A replaced
        # segment reuses this (same parts under the same id); an accumulated segment appends after
        # all prior parts. See `_segment_offset`.
        last_segment_offset = 0
        while True:
            if self._cancelled or self._stopped:
                break

            if response is None:
                messages = self.base_messages
            elif response.state == 'suspended':
                iteration += 1
                if iteration > self.max_continuations:
                    raise UnexpectedModelBehavior(
                        f'Model response was suspended more than the maximum of {self.max_continuations} times'
                    )
                if delay := response.suspended_retry_delay:
                    await self.sleep_func(delay)
                messages = [*self.base_messages, response]
            else:
                break

            # While this sub is in flight, `_merged_response` holds the accumulator of
            # all prior segments (excluding the current sub) so `get()` can fold in the
            # live `sub.get()` snapshot without double-counting.
            self._merged_response = response
            # Resolved lazily on the first reindexable event, once `sub.provider_response_id`
            # is populated, so replace-vs-accumulate matches the eventual `merge_mode`.
            segment_offset: int | None = None
            with self.segment_context():
                async with self.model.request_stream(
                    messages, self.model_settings, self.model_request_parameters, self.run_context
                ) as sub:
                    self._current_sub = sub
                    async for event in sub:
                        if isinstance(event, FinalResultEvent):
                            self.final_result_event = event
                            yield event
                            continue
                        if segment_offset is None:
                            segment_offset = self._segment_offset(response, sub, last_segment_offset)
                        yield self._reindex(event, segment_offset)

            last_segment_offset = segment_offset or 0

            # Read `sub.get()` AFTER the `async with` exits so late-stamped metadata
            # (e.g. a `FallbackModel` continuation pin) is captured.
            sub_response = sub.get()
            merged = sub_response if response is None else merge_responses(response, sub_response)

            self._merged_response = merged
            self._current_sub = None
            self._usage = merged.usage
            self.check_usage(merged.usage)
            response = merged

        self._merged_response = response

    @staticmethod
    def _segment_offset(response: ModelResponse | None, sub: StreamedResponse, last_segment_offset: int) -> int:
        """Index at which the current segment's parts begin in the stitched stream.

        Mirrors [`merge_mode`][pydantic_ai.models._continuation.merge_mode]: a segment sharing the
        prior response's `provider_response_id` *replaces* it (reuse the replaced segment's index
        space), otherwise it *accumulates* (append after all prior parts).
        """
        if response is None:
            return 0
        if sub.provider_response_id and sub.provider_response_id == response.provider_response_id:
            return last_segment_offset
        return len(response.parts)

    def _reindex(self, event: ModelResponseStreamEvent, offset: int) -> ModelResponseStreamEvent:
        if offset and isinstance(event, (PartStartEvent, PartDeltaEvent, PartEndEvent)):
            return replace(event, index=event.index + offset)
        return event

    def _snapshot(self) -> ModelResponse | None:
        """The merged response so far, folding in any in-flight sub-stream."""
        merged = self._merged_response
        if (sub := self._current_sub) is not None:
            sub_response = sub.get()
            return sub_response if merged is None else merge_responses(merged, sub_response)
        return merged

    @property
    def usage(self) -> RequestUsage:
        """Live usage across all segments so far, including the in-flight sub-stream.

        The composite's `_usage` is only refreshed when a segment completes, so — unlike a
        plain segment, whose model updates `_usage` live during iteration — reading it mid
        segment would omit the in-flight sub's usage. Fold in the current sub's live snapshot
        so consumers (e.g. `AgentStream.usage`) see the running total at any point.
        """
        snapshot = self._snapshot()
        return snapshot.usage if snapshot is not None else self._usage

    def get(self) -> ModelResponse:
        """Build the live merged [`ModelResponse`][pydantic_ai.messages.ModelResponse] across all segments so far."""
        # The composite resolves the whole suspended → … → complete chain, so it never surfaces
        # `'suspended'`: it's `'complete'` once the loop exits, `'interrupted'` if cancelled, and
        # `'incomplete'` while a segment is still in flight.
        state: ModelResponseState
        if self._finished:
            state = 'complete'
        elif self._cancelled:
            state = 'interrupted'
        else:
            state = 'incomplete'

        snapshot = self._snapshot()
        if snapshot is None:
            return ModelResponse(parts=[], model_name=self.model_name, state=state)
        return replace(snapshot, state=state)

    async def close_stream(self) -> None:
        """Stop the continuation loop and cancel any server-side suspended/background job."""
        self._stopped = True
        if self._current_sub is not None:
            await self._current_sub.close_stream()
        await self.model.cancel_suspended_response(self.get())

    async def aclose(self) -> None:
        """Tear down the in-flight sub-stream (its HTTP connection) without cancelling any job.

        Each segment's `model.request_stream(...)` context manager lives inside the stitching
        async generator, so — unlike an ordinary `StreamedResponse` whose connection the agent
        graph closed via `async with request_stream(...)` — an early break from a consumer
        doesn't reliably propagate `aclose()` to it. Closing the generator here runs that
        context manager's teardown (mirroring the pre-stitching behavior), and is safe to call
        once the consumer has stopped iterating (including after a normal, fully-drained stream,
        where it's a no-op). Cancellation of a server-side job stays on the `close_stream()`
        path, driven by `AgentStream.cancel()`.

        Closes the inner segment generator directly: the outer cancel-guard's
        `async for … in iterator` does not forward `aclose()` to it, so closing the cancel-guard
        alone would leave each segment's `request_stream(...)` context manager (and its
        connection) open until garbage collection.
        """
        if self._segment_iterator is not None:
            try:
                await self._segment_iterator.aclose()
            except RuntimeError as exc:
                # A debounced consumer (`group_by_temporal`) can have a prefetch task parked mid-`__anext__`
                # inside the stitching generator, so `aclose()` raises `RuntimeError: aclose(): asynchronous
                # generator is already running`. That's exactly the case where a prefetch task exists, and its
                # own cancellation (when the consumer's debounce is torn down) unwinds the generator and runs
                # each segment's `request_stream(...)` teardown — so letting this pass still closes the
                # connection. Mirrors the model adapters' `close_stream()` handling of the same error.
                if not _utils.is_async_generator_already_running(exc):
                    raise

    @property
    def model_name(self) -> str:
        if self._current_sub is not None:
            return self._current_sub.model_name
        if self._merged_response is not None and self._merged_response.model_name:
            return self._merged_response.model_name
        return ''

    @property
    def provider_name(self) -> str | None:
        if self._current_sub is not None:
            return self._current_sub.provider_name
        return self._merged_response.provider_name if self._merged_response is not None else None

    @property
    def provider_url(self) -> str | None:
        if self._current_sub is not None:
            return self._current_sub.provider_url
        return self._merged_response.provider_url if self._merged_response is not None else None

    @property
    def timestamp(self) -> datetime:
        if self._current_sub is not None:
            return self._current_sub.timestamp
        return self._merged_response.timestamp if self._merged_response is not None else _FALLBACK_TIMESTAMP
