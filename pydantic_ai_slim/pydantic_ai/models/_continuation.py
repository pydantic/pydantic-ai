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

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext, suppress
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

__all__ = [
    'MAX_BACKGROUND_POLLS',
    'MAX_GENERATION_CONTINUATIONS',
    'MergeMode',
    'cancel_suspended_job',
    'merge_mode',
    'merge_responses',
    '_ContinuationStreamedResponse',
]

# Framework-protocol markers a model may stamp on a continuation response's `metadata`, namespaced
# under `__pydantic_ai__` so they never collide with a provider's own metadata. `FallbackModel` sets
# `replace_previous_response` on the fresh response it produces after a rewind-and-restart to signal
# "this response supersedes the suspended turn, do not accumulate onto it" — categorically different
# from a same-`provider_response_id` background poll. The `FallbackModel` side must use these exact
# keys. The marker is transient: `merge_responses` honors it as a replace, then pops it so it can't
# persist into history and wrongly force a later legitimate `pause_turn` continuation to replace.
_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
_REPLACE_PREVIOUS_RESPONSE_KEY = 'replace_previous_response'

MAX_GENERATION_CONTINUATIONS = 10
"""Maximum number of *fresh-generation* continuation segments for a single model turn.

Applies to every re-suspension that produces genuinely new generation: an *accumulate* (Anthropic
`pause_turn`, appending new parts under a new `provider_response_id`), a model change, or a
`FallbackModel` `replace_previous_response` directive. This is the guard against a model that never
leaves the `'suspended'` state, endlessly emitting new segments; exceeding it raises
[`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].

Only a *same-id* re-suspension — re-polling a single long-running job under the same
`provider_response_id` (OpenAI background mode) — is bounded by the far more generous
[`MAX_BACKGROUND_POLLS`][pydantic_ai.models._continuation.MAX_BACKGROUND_POLLS] instead,
so a healthy background job that legitimately runs for minutes isn't killed after ~10 continuations.
"""

MAX_BACKGROUND_POLLS = 1000
"""Backstop for *same-id* continuation polling of a single background job.

A same-id re-suspension re-fetches one long-running job under the same `provider_response_id`
(OpenAI background mode), so — unlike a fresh-generation re-suspension (see
[`MAX_GENERATION_CONTINUATIONS`][pydantic_ai.models._continuation.MAX_GENERATION_CONTINUATIONS]) — it carries no risk of an
unbounded model spawning endless new segments: it's the *same* job, and legitimate background jobs run
for minutes (hundreds of polls at a ~2s interval). The real bounds here are the run's usage limits and
explicit cancellation, which already work; this large ceiling is only a last-resort safety net against a
provider stuck returning `'suspended'` for the same id forever (which usage limits wouldn't catch, since
a pending poll adds no tokens). Exceeding it raises
[`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
"""

MergeMode = Literal['replace-same-id', 'replace-new', 'accumulate']
"""How a continuation response folds into the one it continues.

- `'replace-same-id'`: both responses share a `provider_response_id` — a passive re-poll of one
  long-running job (OpenAI background `retrieve`, which returns the full response so far). Replaces
  wholesale, and — being the *same* job rather than fresh generation — is bounded by the generous
  [`MAX_BACKGROUND_POLLS`][pydantic_ai.models._continuation.MAX_BACKGROUND_POLLS] ceiling.
- `'replace-new'`: the response supersedes the suspended turn with *fresh* generation — the model
  changed (accumulating parts from different models is always wrong) or a `FallbackModel` stamped the
  `replace_previous_response` marker after a rewind-and-restart. Replaces wholesale, but counts against
  the strict [`MAX_GENERATION_CONTINUATIONS`][pydantic_ai.models._continuation.MAX_GENERATION_CONTINUATIONS] ceiling, since a
  chain of fresh suspensions is the exact runaway that cap guards against.
- `'accumulate'`: appends new parts onto the prior response (Anthropic `pause_turn`). Strict ceiling.
"""

# Deterministic fallback for `timestamp` before any segment has streamed. This is
# never reached in practice (a segment is always in flight or finalized by the time
# `timestamp` is read), but keeps the loop free of `now_utc()` for durable replay.
_FALLBACK_TIMESTAMP = datetime.fromtimestamp(0, tz=timezone.utc)


def _has_replace_marker(response: ModelResponse) -> bool:
    """Whether `response` carries the `FallbackModel` `replace_previous_response` directive."""
    metadata = response.metadata
    if not _utils.is_str_dict(metadata):
        return False
    namespace = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    return bool(_utils.is_str_dict(namespace) and namespace.get(_REPLACE_PREVIOUS_RESPONSE_KEY))


def _strip_replace_marker(metadata: Any) -> dict[str, Any] | None:
    """Return `metadata` without the transient `replace_previous_response` marker (other keys intact).

    Copies before mutating so the caller's dicts (including the shared `__pydantic_ai__` namespace,
    which also holds the `FallbackModel` continuation pin) aren't touched.
    """
    if not _utils.is_str_dict(metadata):
        return metadata
    namespace = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    if not (_utils.is_str_dict(namespace) and _REPLACE_PREVIOUS_RESPONSE_KEY in namespace):
        return metadata
    namespace = {k: v for k, v in namespace.items() if k != _REPLACE_PREVIOUS_RESPONSE_KEY}
    metadata = {**metadata}
    if namespace:
        metadata[_PYDANTIC_AI_METADATA_KEY] = namespace
    else:
        del metadata[_PYDANTIC_AI_METADATA_KEY]
    return metadata


def merge_mode(existing: ModelResponse, new: ModelResponse) -> MergeMode:
    """Classify how `new` folds into `existing` — see [`MergeMode`][pydantic_ai.models._continuation.MergeMode].

    The single decision path shared by [`merge_responses`][pydantic_ai.models._continuation.merge_responses],
    the continuation-count ceilings, and the streamed composite's part-index reindexing.
    """
    # A `FallbackModel` rewind-and-restart marks its fresh response as superseding the suspended turn;
    # honor that first, since such a response may otherwise look like an accumulate.
    if _has_replace_marker(new):
        return 'replace-new'
    if existing.provider_response_id and existing.provider_response_id == new.provider_response_id:
        return 'replace-same-id'
    if existing.model_name and new.model_name and existing.model_name != new.model_name:
        return 'replace-new'
    return 'accumulate'


def merge_responses(existing: ModelResponse, new: ModelResponse) -> ModelResponse:
    """Merge a continuation response into the one it continues.

    On any `'replace-*'` mode (same `provider_response_id`, a model change, or a `FallbackModel`
    `replace_previous_response` directive), replace entirely with the new response. Otherwise
    accumulate parts, sum usage, and use other fields from the new response.

    Either way, `provider_details` and `metadata` accumulate across the turn's segments (latest-wins)
    so turn-scoped data a later segment omits isn't lost — see below.
    """
    if merge_mode(existing, new) != 'accumulate':
        merged = new
    else:
        # Same model, different response → accumulate parts and sum usage.
        # Preserve existing provider response IDs when continuation responses omit them
        # (e.g. resumed OpenAI streams that start after a sequence number).
        merged = replace(
            new,
            parts=[*existing.parts, *new.parts],
            usage=existing.usage + new.usage,
            provider_response_id=new.provider_response_id or existing.provider_response_id,
        )

    # A turn's provider metadata accumulates across its segments. Turn-scoped identifiers — OpenAI
    # `background`/`conversation_id`, Anthropic `container_id` in `provider_details`, and the
    # `FallbackModel` continuation pin in `metadata` — are only stamped on segments whose payload
    # carries them, so a resumed or interrupted segment (e.g. a mid-flight cancel snapshot, whose
    # in-flight segment hasn't stamped the pin yet) can omit one an earlier segment set. Merge
    # latest-wins (`new` overrides) so they survive into the merged response; e.g.
    # `cancel_suspended_response` relies on OpenAI's `background` marker and the `FallbackModel` pin
    # to reach the server-side job.
    if existing.provider_details:
        merged = replace(merged, provider_details={**existing.provider_details, **(merged.provider_details or {})})
    if existing.metadata:
        merged = replace(merged, metadata={**existing.metadata, **(merged.metadata or {})})

    # Pop the transient `replace_previous_response` marker now that it's been honored above, so it
    # doesn't persist into history where it would wrongly force a later legitimate `pause_turn`
    # continuation to replace rather than accumulate. Other `__pydantic_ai__` keys (the continuation
    # pin) survive.
    stripped = _strip_replace_marker(merged.metadata)
    if stripped is not merged.metadata:
        merged = replace(merged, metadata=stripped)
    return merged


async def cancel_suspended_job(model: Model, response: ModelResponse) -> None:
    """Best-effort teardown of a server-side suspended/background job that survives cancellation.

    When the trigger is a workflow/task cancellation (e.g. Temporal), awaiting the (activity-wrapped)
    cancel from inside an already-cancelled scope would raise `CancelledError` before the cancel runs,
    silently leaking the job. Shield the cancel so it completes before the cancellation propagates;
    Temporal's workflow loop respects `asyncio.shield`. Any error from the cancel itself is swallowed —
    a failing teardown must not replace the error (or cancellation) that aborted the run.
    """
    job = asyncio.ensure_future(model.cancel_suspended_response(response))
    try:
        await asyncio.shield(job)
    except asyncio.CancelledError:
        # Our scope was cancelled mid-teardown; let the shielded cancel finish before propagating.
        with suppress(Exception):
            await job
        raise
    except Exception:
        pass


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
    max_generation_continuations: int
    sleep_func: Callable[[float], Awaitable[None]]
    check_usage: Callable[[RequestUsage], None]
    initial_suspended_response: ModelResponse | None = None
    # Ceiling for *replace*-style (single-job background poll) re-suspensions, kept separate from
    # `max_generation_continuations` (which bounds fresh-generation re-suspensions). See `MAX_BACKGROUND_POLLS`.
    max_background_polls: int = MAX_BACKGROUND_POLLS
    # Entered around each segment's `model.request_stream(...)`. The agent graph passes a factory
    # that re-attaches the ambient context (e.g. the OTel `chat` span opened by `wrap_model_request`
    # in a separate task) so span updates driven by `get_current_span()` land on the right span even
    # though segments are opened lazily in the consumer task. Opaque to this module (no OTel coupling).
    segment_context: Callable[[], AbstractContextManager[Any]] = nullcontext

    _merged_response: ModelResponse | None = field(default=None, init=False)
    _current_sub: StreamedResponse | None = field(default=None, init=False)
    _stopped: bool = field(default=False, init=False)
    # Set by `aclose()`: the consumer stopped iterating and the stream was torn down *without* a
    # `cancel()`/`close_stream()` (which would flip `_stopped`/`_cancelled` and cancel the server-side
    # job). Lets `get()` distinguish a deliberate detach — where a still-pending suspended job survives
    # server-side and the run should be recorded as resumable `'suspended'` — from a live, still-streaming
    # snapshot (`'incomplete'`). See `get()`.
    _detached: bool = field(default=False, init=False)
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

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        """Cancel-teardown errors to suppress, extended with the in-flight sub-stream's own.

        The cancel-guard tears the current segment down via its `close_stream()`, so the transport
        error it raises is whatever that sub's transport produces — httpx for most providers, but
        botocore (Bedrock) or grpc (xAI) for others, which each report their own types via an
        override. Consult the in-flight sub (`_current_sub` is still set at exception time) on top of
        the httpx default, so a non-httpx teardown error is suppressed into a clean `'interrupted'`
        stop rather than escaping to the consumer.
        """
        errors = super().get_stream_cancel_errors()
        if self._current_sub is not None:
            errors = (*errors, *self._current_sub.get_stream_cancel_errors())
        return errors

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
                if self._first_chunk_monotonic is None:
                    # First event surfaced to the consumer: stamp the monotonic clock so
                    # `time_to_first_chunk` works, mirroring the base cancel-guard this replaces.
                    self._first_chunk_monotonic = time.perf_counter()
                yield event
        except self.get_stream_cancel_errors():
            if not self.cancelled:
                raise
        else:
            if not self._cancelled:
                self._finished = True

    def _count_continuation(
        self, response: ModelResponse, last_mode: MergeMode | None, accumulate_count: int, replace_count: int
    ) -> tuple[int, int]:
        """Count a suspended re-issue against its ceiling (same-id poll vs everything else), raising if exceeded.

        Only a `'replace-same-id'` re-suspension — a passive re-poll of one long-running background job —
        gets the generous `max_background_polls` ceiling. A model-change or `FallbackModel`-directed
        replace is *fresh* generation, not the same job, so it counts against the strict `max_generation_continuations`
        cap alongside accumulate re-suspensions — otherwise a chain of fresh suspensions is the exact
        runaway the strict cap guards against. The first re-issue has no prior merge to classify
        (`last_mode is None`) so it counts as strict, harmless since both ceilings allow at least one.
        See `MAX_BACKGROUND_POLLS`. Returns the updated `(accumulate_count, replace_count)`.
        """
        job_id = response.provider_response_id
        if last_mode == 'replace-same-id':
            replace_count += 1
            if replace_count > self.max_background_polls:
                raise UnexpectedModelBehavior(
                    f'Model response for job {job_id!r} remained suspended after polling the maximum of '
                    f'{self.max_background_polls} times'
                )
        else:
            accumulate_count += 1
            if accumulate_count > self.max_generation_continuations:
                raise UnexpectedModelBehavior(
                    f'Model response {job_id!r} was suspended more than the maximum of '
                    f'{self.max_generation_continuations} times'
                )
        return accumulate_count, replace_count

    async def _get_event_iterator(self) -> AsyncGenerator[ModelResponseStreamEvent, None]:
        # Two independent ceilings, distinguished by the generic `merge_mode` signal (the same one that
        # drives reindexing): every *fresh-generation* re-suspension (accumulate `pause_turn`, a model
        # change, or a `FallbackModel` replace directive) risks an unbounded model spawning new segments,
        # so it keeps the small `max_generation_continuations` cap; only a *same-id* re-suspension (OpenAI background
        # poll) re-fetches one long-running job under the same `provider_response_id`, so a healthy job
        # that legitimately runs for minutes must not be killed by the small cap — it gets the far more
        # generous `max_background_polls` backstop. Mirrors the non-streaming continuation loop in
        # `_agent_graph`. See `MAX_BACKGROUND_POLLS`.
        accumulate_count = 0
        replace_count = 0
        # Mode of the merge that produced the current suspended `response`, used to pick its ceiling. A
        # continuation chain is homogeneous in practice (a poll chain is all same-id, a `pause_turn` chain
        # all-accumulate), so the previous merge's mode reliably classifies the next re-issue.
        last_mode: MergeMode | None = None
        response = self.initial_suspended_response
        # Index at which the most recent segment's parts began in the stitched stream. A replaced
        # segment reuses this (same parts under the same id); an accumulated segment appends after
        # all prior parts. See `_segment_offset`.
        last_segment_offset = 0
        try:
            while True:
                if self._cancelled or self._stopped:
                    break

                if response is None:
                    messages = self.base_messages
                elif response.state == 'suspended':
                    accumulate_count, replace_count = self._count_continuation(
                        response, last_mode, accumulate_count, replace_count
                    )
                    if delay := self.model.continuation_delay(response):
                        await self.sleep_func(delay)
                        # A `cancel()`/`close_stream()` from another task during the inter-poll sleep
                        # already tore down the server-side job; don't open the next sub-stream, which
                        # for Anthropic `pause_turn` would actively resume generation and burn tokens.
                        if self._cancelled or self._stopped:
                            break
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
                if response is None:
                    merged = sub_response
                else:
                    # Classify this transition (replace vs accumulate) so the next re-issue is counted
                    # against the right ceiling.
                    last_mode = merge_mode(response, sub_response)
                    merged = merge_responses(response, sub_response)

                self._merged_response = merged
                self._current_sub = None
                self._usage = merged.usage
                self.check_usage(merged.usage)
                response = merged

            self._merged_response = response
        except GeneratorExit:
            # Deliberate `aclose()` detach: tear the connection down without cancelling the
            # server-side job (that stays on the `cancel()`/`close_stream()` path). Re-raise as-is.
            raise
        except BaseException:
            # A later segment failed (transport error, `check_usage` raising, or the max-continuations
            # raise) with a suspended job in hand. The non-streaming continuation loop cancels the
            # server-side job on exactly this class of failure; mirror it so streaming doesn't leak the
            # job (which history would otherwise record as unresumable and uncancellable). Skip when a
            # deliberate `cancel()`/`close_stream()` is already tearing things down — it cancels itself.
            if response is not None and response.state == 'suspended' and not (self._cancelled or self._stopped):
                await cancel_suspended_job(self.model, response)
            raise

    @staticmethod
    def _segment_offset(response: ModelResponse | None, sub: StreamedResponse, last_segment_offset: int) -> int:
        """Index at which the current segment's parts begin in the stitched stream.

        Shares [`merge_mode`][pydantic_ai.models._continuation.merge_mode]'s decision so reindexing
        matches the eventual merge:

        - `'accumulate'` appends after all prior parts (offset = number of prior parts).
        - `'replace-same-id'` (a background job re-polled under the same `provider_response_id`) re-emits
          the *same* parts in the *same* index space, so it reuses the replaced segment's offset.
        - `'replace-new'` (a model change, or a `FallbackModel` `replace_previous_response` directive)
          supersedes the whole prior response — `merge_responses` keeps only the new parts, indexed from
          0 — so its events must start at offset 0 too, or the live event indices would drift past the
          final response's (e.g. after one or more accumulated segments).
        """
        if response is None:
            return 0
        mode = merge_mode(response, sub.get())
        if mode == 'accumulate':
            return len(response.parts)
        if mode == 'replace-new':
            return 0
        return last_segment_offset

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
        """Build the live merged [`ModelResponse`][pydantic_ai.messages.ModelResponse] across all segments so far.

        The composite normally resolves the whole `suspended → … → complete` chain, so mid-run it's
        `'complete'` once the loop exits, `'interrupted'` if cancelled, and `'incomplete'` while a
        segment is still in flight. The one case it *does* surface `'suspended'` is a **detach**: the
        consumer stopped iterating and the stream was torn down via `aclose()` — not `cancel()` — while
        the current/last segment is itself a still-pending suspended job. The server-side job survives
        (detach doesn't cancel it), so recording `'suspended'` makes the run resumable later, matching
        the non-streaming path where a persisted suspended response can be resumed. A real `cancel()`
        (`_cancelled`) also cancels the server-side job, so it stays `'interrupted'` and non-resumable.
        """
        snapshot = self._snapshot()

        state: ModelResponseState
        if self._finished:
            state = 'complete'
        elif self._cancelled:
            # A real `cancel()` tore down the server-side job too, so this is not resumable.
            state = 'interrupted'
        elif self._detached and snapshot is not None and snapshot.state == 'suspended':
            # Detached with a still-pending suspended job in hand (and not cancelled): resumable.
            state = 'suspended'
        else:
            state = 'incomplete'

        if snapshot is None:
            return ModelResponse(parts=[], model_name=self.model_name, state=state)
        return replace(snapshot, state=state)

    async def close_stream(self) -> None:
        """Stop the continuation loop and cancel any server-side suspended/background job."""
        self._stopped = True
        try:
            if self._current_sub is not None:
                await self._current_sub.close_stream()
        finally:
            # Cancel the server-side job even if tearing down the sub-stream connection raised,
            # so a failed connection teardown can't leave the background job running. Shielded so the
            # cancel completes even when the run is being torn down by a workflow/task cancellation.
            await cancel_suspended_job(self.model, self.get())

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
        # Record that the consumer stopped and the stream was torn down without cancelling. `get()`
        # reads this to report a still-pending suspended job as resumable `'suspended'` rather than a
        # live `'incomplete'` snapshot; `_finished`/`_cancelled` take precedence, so this is a no-op
        # after a fully-drained (`'complete'`) or cancelled (`'interrupted'`) stream.
        self._detached = True
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
