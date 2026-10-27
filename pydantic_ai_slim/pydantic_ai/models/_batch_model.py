"""The batching behind [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests].

A `BatchPool` collects the requests of every model it wraps into batches, and checks on them. `BatchedModel` is the
wrapper the capability puts around the model serving each request, so requests from any agent or run that uses the
same capability instance share one pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, NotRequired

from opentelemetry import context as otel_context
from opentelemetry.trace import (
    Link,
    NonRecordingSpan,
    Span,
    SpanContext,
    SpanKind,
    Tracer,
    get_current_span,
    set_span_in_context,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from .._instrumentation import model_attributes, open_request_policy
from .._run_context import RunContext
from .._utils import cancel_and_drain
from ..exceptions import ModelAPIError, ModelHTTPError, SuspendedResponseExpired, UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from ._batch_codec import BatchJobError
from ._batch_settings import (
    DEFAULT_MAX_POLL_INTERVAL,
    DEFAULT_MAX_REQUESTS,
    DEFAULT_MAX_RESUBMISSIONS,
    DEFAULT_POLL_INTERVAL,
    batch_size,
    check_batch_intervals,
    check_failure_tolerated,
    missing_result_error,
    poll_delay,
    resubmittable,
    transient,
)
from .batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from .wrapper import WrapperModel

_TICKET_KEY = 'batch'
_RESULTS_CACHE_SIZE = 32
_RESULTS_CACHE_BYTES = 256 * 1024 * 1024
"""How many bytes of result bodies the results cache keeps of batches no request in this process is known to wait on."""
_SUBMITTED_BATCHES_SIZE = 1024
# Used directly rather than through the global propagator, which can be configured to propagate more, or to warn.
_TRACE_CONTEXT = TraceContextTextMapPropagator()
_EXITED_MESSAGE = '`BatchModelRequests` exited while the request was waiting for its batch.'

BATCH_ID_ATTRIBUTE = 'pydantic_ai.batch.id'
BATCH_STATUS_ATTRIBUTE = 'pydantic_ai.batch.status'
BATCH_REQUEST_COUNT_ATTRIBUTE = 'pydantic_ai.batch.request_count'


class _Ticket(TypedDict):
    """What a pending batch response records about its request, under `provider_details['batch']`."""

    custom_id: str
    batch: Batch
    polls: int
    failed_checks: NotRequired[int]
    """How many checks in a row failed in a way a live request is retried on, which bounds further checks."""
    resubmissions: NotRequired[int]
    """How many times the request was submitted again after a transient failure, which bounds further resubmissions."""
    continued_response_id: NotRequired[str]
    """The ID of the suspended response the request continued, which merging the pending response into it replaces."""
    trace_context: NotRequired[dict[str, str]]
    """The W3C trace context of the batch's span, so its checks join the batch's trace in any process."""


_TICKET_ADAPTER = TypeAdapter(_Ticket)


def _ticket_details(ticket: _Ticket) -> dict[str, Any]:
    """`ticket` as stored in `provider_details`, which must survive serializing the message history."""
    return {_TICKET_KEY: _TICKET_ADAPTER.dump_python(ticket, mode='json')}


def _response_id(batch_id: str, custom_id: str) -> str:
    return f'batch:{batch_id}:{custom_id}'


def _pending_ticket(message: ModelMessage | None) -> _Ticket | None:
    """The ticket of a response that is still waiting for its batch result, if `message` is one."""
    if not (isinstance(message, ModelResponse) and message.state == 'suspended' and message.provider_details):
        return None
    try:
        ticket = _TICKET_ADAPTER.validate_python(message.provider_details.get(_TICKET_KEY))
    except ValidationError:
        return None
    # A merged response can keep the ticket of a batch request it no longer waits for, so the ID decides.
    if message.provider_response_id != _response_id(ticket['batch'].id, ticket['custom_id']):
        return None
    return ticket


def check_collectable(model: Model, response: ModelResponse) -> None:
    """Raise if `response` waits on a batch request that `model` won't collect, as it isn't batched.

    The provider's model would otherwise send the history ending in it as a new, full-price request, and the batch's
    result, which is billed too, would never be read.
    """
    if (ticket := _pending_ticket(response)) is not None and model.continuation_delay(response) is None:
        raise UserError(
            f'The message history ends in a request waiting for its result from batch {ticket["batch"].id!r}, '
            'which only `BatchModelRequests` collects: add the capability to the agent or run that resumes it.'
        )


def check_resumable(model: Model, messages: Sequence[ModelMessage]) -> None:
    """Raise if `messages` end in a request waiting on a batch that `model` won't collect, like `check_collectable`."""
    if messages and isinstance(last := messages[-1], ModelResponse):
        check_collectable(model, last)


def refuse_pending_batch_request(messages: Sequence[ModelMessage]) -> None:
    """Raise if `messages` end in a request waiting on a batch, which a new batch request would send again."""
    if (ticket := _pending_ticket(messages[-1] if messages else None)) is not None:
        raise UserError(
            f'The messages end in a request waiting for its result from batch {ticket["batch"].id!r}, which a new '
            'batch request would send again: collect it with `BatchModelRequests`.'
        )


def _checked_again(pending: ModelResponse, ticket: _Ticket) -> ModelResponse:
    """`pending` with the `ticket` of a check that found its batch still running, or couldn't tell.

    Unchanged apart from the ticket: the same `provider_response_id` makes the graph replace the pending response with
    this one, so content the response accumulated before the batch is kept.
    """
    return replace(pending, provider_details={**(pending.provider_details or {}), **_ticket_details(ticket)})


def _span_context(trace_context: dict[str, str] | None) -> SpanContext | None:
    if not trace_context:
        return None
    span_context = get_current_span(_TRACE_CONTEXT.extract(trace_context)).get_span_context()
    return span_context if span_context.is_valid else None


@dataclass(frozen=True)
class _Requester:
    """The instrumented request a queued or checking request belongs to, if any, for the batch's spans."""

    tracer: Tracer
    span: Span

    @classmethod
    def current(cls) -> _Requester | None:
        policy = open_request_policy()
        return None if policy is None else cls(policy.tracer, policy.span)


@dataclass
class _Queued:
    request: EncodedBatchRequest
    future: asyncio.Future[tuple[Batch, dict[str, str] | None]]
    requester: _Requester | None
    size: int
    """The bytes the request adds to its batch."""
    withdrawn: bool = False
    """Whether the request was cancelled after its queue was taken for submission, so nobody waits on its result."""


@dataclass
class _Queue:
    model: Model
    """The model that submits the queue: any model whose requests share the queue's group can."""
    max_requests: int
    max_bytes: int | None
    requests: dict[str, _Queued] = field(default_factory=dict[str, _Queued])
    """The queued requests by their ID in the batch, in the order they were queued."""
    size: int = 0
    """The bytes the queued requests take in a batch."""
    duplicates: dict[str, int] = field(default_factory=dict[str, int])
    """How many requests with each body-derived ID were queued, to give the next one a unique ID in constant time."""
    timer: asyncio.TimerHandle | None = None

    def unique_id(self, base_id: str) -> str:
        """`base_id`, suffixed if an earlier request in the queue had the same one."""
        count = self.duplicates.get(base_id, 0)
        # A base ID is a hex digest, so a suffixed ID can't equal another request's base ID.
        return f'{base_id}-{count}' if count else base_id

    def fits(self, size: int) -> bool:
        """Whether a request of `size` bytes fits in the batch: an empty one takes any request `batch_encode` allowed."""
        return self.max_bytes is None or not self.requests or self.size + size <= self.max_bytes

    def add(self, queued: _Queued, base_id: str) -> None:
        self.duplicates[base_id] = self.duplicates.get(base_id, 0) + 1
        self.requests[queued.request.custom_id] = queued
        self.size += queued.size

    def remove(self, queued: _Queued) -> None:
        """Take a request out, which is still in the queue even when the queue was taken for submission."""
        del self.requests[queued.request.custom_id]
        self.size -= queued.size


_QueueKey = tuple[str, str]
"""The provider and `group` of a queue's requests: groups are only comparable within one provider."""

_BatchKey = tuple[str, str, str]
"""The provider, group and ID of a batch: IDs are only unique within one provider account."""

_CheckKey = tuple[str, str, str, str]
"""The credential a batch is checked with, and the batch's key.

Checks and results are only shared between requests checking a batch with one credential, so a request with another
one, which a message history's ticket can't vouch for, asks the provider, which decides whether it may see the batch.
"""


def _batch_key(batch: Batch) -> _BatchKey:
    return batch.provider_name, batch.group, batch.id


def _check_key(model: Model, batch: Batch) -> _CheckKey:
    return (model._batch_credential(), *_batch_key(batch))  # pyright: ignore[reportPrivateUsage]


def _take(queues: dict[_QueueKey, _Queue], key: _QueueKey) -> _Queue:
    """Take a queue off for submission, so later requests start a new one."""
    queue = queues.pop(key)
    if queue.timer is not None:
        queue.timer.cancel()
    return queue


@contextmanager
def _expired_after_retention(batch: Batch) -> Generator[None]:
    """Raise `SuspendedResponseExpired` for a batch the provider no longer has, as for an expired background response."""
    try:
        yield
    except ModelHTTPError as error:
        if error.status_code != 404:
            raise
        raise SuspendedResponseExpired(
            f'Batch {batch.id!r} could not be checked because {batch.provider_name!r} no longer has it or its results: '
            'providers keep them for a limited time, and only for the project or workspace that created them.'
        ) from error


@contextmanager
def _batch_span(
    requester: _Requester | None, name: str, *, parent: SpanContext | None, links: list[Link], model: Model
) -> Generator[Span | None]:
    """A span for work shared by the requests of a batch, in the batch's own trace rather than any one run's.

    The trace context of the run that started the work is set aside for its duration, so the provider's HTTP spans
    are this span's children, or, when no request it serves is instrumented and this yields `None`, start traces of
    their own. With no `parent`, the span starts a trace too.
    """
    token = otel_context.attach(otel_context.Context())
    try:
        if requester is None:
            yield None
            return
        context = otel_context.Context()
        if parent is not None:
            context = set_span_in_context(NonRecordingSpan(parent), context)
        with requester.tracer.start_as_current_span(
            name, context=context, links=links, kind=SpanKind.CLIENT, attributes=model_attributes(model)
        ) as span:
            yield span
    finally:
        otel_context.detach(token)


class BatchPool:
    """Queues, submissions and checks shared by every model a [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] wraps."""

    def __init__(
        self,
        *,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_wait: timedelta | None = timedelta(seconds=30),
        max_resubmissions: int = DEFAULT_MAX_RESUBMISSIONS,
        poll_interval: timedelta = DEFAULT_POLL_INTERVAL,
        max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL,
    ) -> None:
        check_batch_intervals(
            max_requests=max_requests,
            max_resubmissions=max_resubmissions,
            poll_interval=poll_interval,
            max_poll_interval=max_poll_interval,
        )
        if max_wait is not None and max_wait <= timedelta(0):
            raise UserError('`max_wait` must be positive.')
        self.max_requests = max_requests
        self.max_wait = max_wait
        self.max_resubmissions = max_resubmissions
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        # Queues hold futures, which belong to one event loop, so each loop gets its own.
        self._queues: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[_QueueKey, _Queue]] = (
            weakref.WeakKeyDictionary()
        )
        self._submissions: set[asyncio.Task[None]] = set()
        # Every request waiting on a batch checks on it, so concurrent checks and the results are shared between them.
        self._polls: dict[
            tuple[asyncio.AbstractEventLoop, _CheckKey], asyncio.Task[tuple[Batch, dict[str, BatchResult] | None]]
        ] = {}
        # The ended batch, as the provider last described it, and its results.
        self._results: OrderedDict[_CheckKey, tuple[Batch, dict[str, BatchResult]]] = OrderedDict()
        self._results_bytes: dict[_CheckKey, int] = {}
        # The IDs of the requests this pool submitted in each ended batch that are still waiting to read their result,
        # so the batch's results are dropped once they all have, and kept until then.
        self._unread: dict[_CheckKey, set[str]] = {}
        # The last status of each batch still in progress, with when it was checked, so requests resumed one after
        # another, like a loop over stored message histories, share a check per `poll_interval` rather than each
        # making one.
        self._statuses: OrderedDict[_CheckKey, tuple[float, Batch]] = OrderedDict()
        # The IDs of the requests in each batch this pool submitted that hasn't ended or been cancelled, as only these
        # batches are cancelled once they're abandoned: a ticket in a message history can't be trusted to say which
        # other requests share its batch. The oldest are forgotten first, which leaves their batches running.
        self._submitted: OrderedDict[_BatchKey, frozenset[str]] = OrderedDict()
        self._abandoned: dict[_BatchKey, set[str]] = {}
        # The requests of those batches that a run left for a later run to collect. Until a later run resumes one,
        # cancelling it explicitly still abandons it; a released request that isn't abandoned keeps its batch running.
        self._released: dict[_BatchKey, set[str]] = {}
        # The records above are shared by every event loop, and so every thread, the pool serves, unlike its queues
        # and checks. Their updates read and change several of them at once, so they hold this lock, never across an
        # `await`.
        self._lock = threading.Lock()
        # The batches each request span is already linked to, so a span waiting on several checks links to one once.
        self._linked: weakref.WeakKeyDictionary[Span, set[_BatchKey]] = weakref.WeakKeyDictionary()

    def wrap(self, model: Model) -> BatchedModel:
        """Wrap `model` so its requests go through this pool, checking that it has a batch API."""
        batch_size(model, self.max_requests)
        return BatchedModel(model, pool=self)

    async def close(self) -> None:
        """Fail the requests still queued on the running event loop, and cancel its submissions and checks."""
        loop = asyncio.get_running_loop()
        queues = self._queues.pop(loop, {})
        for key in list(queues):
            for queued in _take(queues, key).requests.values():
                queued.future.set_exception(UserError(_EXITED_MESSAGE))
        await cancel_and_drain(
            *(task for task in self._submissions if task.get_loop() is loop),
            *(task for (task_loop, _), task in self._polls.items() if task_loop is loop),
        )

    async def flush(self) -> None:
        """Submit every queued request now instead of waiting for a full batch or `max_wait`."""
        loop = asyncio.get_running_loop()
        queues = self._queues.get(loop, {})
        if submissions := [self._start_submission(loop, queues, key) for key in list(queues)]:
            # Waits without cancelling the submissions, so cancelling the caller doesn't fail the queued requests.
            await asyncio.wait(submissions)

    def link(self, requester: _Requester | None, batch: Batch, trace_context: dict[str, str] | None) -> None:
        """Tie a request's span to the batch it waits on: the batch's ID, and a link to the batch's span."""
        if requester is None or not requester.span.is_recording():
            return
        linked = self._linked.setdefault(requester.span, set())
        if _batch_key(batch) in linked:
            return
        linked.add(_batch_key(batch))
        requester.span.set_attribute(BATCH_ID_ATTRIBUTE, batch.id)
        if (span_context := _span_context(trace_context)) is not None:
            requester.span.add_link(span_context)

    def _start_submission(
        self, loop: asyncio.AbstractEventLoop, queues: dict[_QueueKey, _Queue], key: _QueueKey
    ) -> asyncio.Task[None]:
        """Submit a queue in a task of its own, so cancelling whoever triggered it doesn't cancel the others.

        The queue is taken off `queues` right away, so a request made before the submission runs starts the next batch.
        """
        queue = _take(queues, key)
        task = loop.create_task(self._submit(queue))
        self._submissions.add(task)
        task.add_done_callback(self._submissions.discard)
        requests = list(queue.requests.values())

        def settle(_: asyncio.Task[None]) -> None:
            # A submission that was cancelled, for example by `close()` or as the event loop shuts down, leaves nothing
            # to wait for, even one cancelled before it started.
            for queued in requests:
                queued.future.cancel()

        task.add_done_callback(settle)
        return task

    async def enqueue(
        self, model: Model, encoded: EncodedBatchRequest, requester: _Requester | None
    ) -> tuple[str, Batch, dict[str, str] | None]:
        """Queue a request, and return the ID it was given in its batch, the batch, and the batch span's context."""
        loop = asyncio.get_running_loop()
        queues = self._queues.setdefault(loop, {})
        key = (model.system, encoded.group)

        def submit() -> None:
            self._start_submission(loop, queues, key)

        def queued_in(queue: _Queue) -> _Queued:
            # Derived from the request, so a rerun of the same requests gets the same IDs; unique within the batch.
            request = replace(encoded, custom_id=queue.unique_id(base_id))
            return _Queued(request, future, requester, size=model.batch_request_size(request))

        base_id = hashlib.sha256(encoded.body.encode()).hexdigest()[:32]
        future: asyncio.Future[tuple[Batch, dict[str, str] | None]] = loop.create_future()
        queue = queues.get(key) or self._new_queue(queues, key, model)
        queued = queued_in(queue)
        if not queue.fits(queued.size):
            # The batch would exceed the provider's size limit with this request, so it's submitted without it.
            submit()
            queue = self._new_queue(queues, key, model)
            queued = queued_in(queue)
        queue.add(queued, base_id)
        custom_id = queued.request.custom_id

        if len(queue.requests) >= queue.max_requests:
            submit()
        elif queue.timer is None and self.max_wait is not None:
            queue.timer = loop.call_later(self.max_wait.total_seconds(), submit)
        try:
            # Waits without cancelling the future on cancellation, as the submission resolves it.
            await asyncio.wait((queued.future,))
        except asyncio.CancelledError:
            # Leaves the batch if it's still queued. A submission has already taken the queue's requests, so the
            # request is in its batch: it's abandoned once the batch exists, which can cancel the batch.
            queue.remove(queued)
            queued.withdrawn = True
            if queued.future.done() and not queued.future.cancelled() and queued.future.exception() is None:
                self._withdraw(model, queued.future.result()[0], custom_id)
            raise
        if queued.future.cancelled():
            # The submission was cancelled, so there's no batch to wait for.
            raise UserError(_EXITED_MESSAGE)
        batch, trace_context = queued.future.result()
        return custom_id, batch, trace_context

    def _new_queue(self, queues: dict[_QueueKey, _Queue], key: _QueueKey, model: Model) -> _Queue:
        queue = queues[key] = _Queue(
            model, max_requests=batch_size(model, self.max_requests), max_bytes=model.batch_max_bytes
        )
        return queue

    async def _submit(self, queue: _Queue) -> None:
        if not (requests := list(queue.requests.values())):
            # Every queued request was cancelled before the deadline.
            return
        requester = next((queued.requester for queued in requests if queued.requester is not None), None)
        links = [Link(queued.requester.span.get_span_context()) for queued in requests if queued.requester is not None]
        try:
            with _batch_span(
                requester, f'batch {queue.model.model_name}', parent=None, links=links, model=queue.model
            ) as span:
                if span is not None:
                    span.set_attribute(BATCH_REQUEST_COUNT_ATTRIBUTE, len(requests))
                batch = await queue.model.batch_create([queued.request for queued in requests])
                trace_context: dict[str, str] | None = None
                if span is not None:
                    span.set_attributes({BATCH_ID_ATTRIBUTE: batch.id, BATCH_STATUS_ATTRIBUTE: batch.status})
                    carrier: dict[str, str] = {}
                    _TRACE_CONTEXT.inject(carrier)
                    trace_context = carrier or None
        except Exception as error:
            for queued in requests:
                queued.future.set_exception(error)
        else:
            with self._lock:
                self._submitted[_batch_key(batch)] = frozenset(queued.request.custom_id for queued in requests)
                while len(self._submitted) > _SUBMITTED_BATCHES_SIZE:
                    self._untrack(self._submitted.popitem(last=False)[0])
            for queued in requests:
                queued.future.set_result((batch, trace_context))
            for queued in requests:
                if queued.withdrawn:
                    self._withdraw(queue.model, batch, queued.request.custom_id)

    def _withdraw(self, model: Model, batch: Batch, custom_id: str) -> None:
        """Abandon a request that was submitted without anyone left to wait on it, cancelling the batch if it's the last."""
        if self.abandon(batch, custom_id):
            # In a task of its own, as the request's own task is being cancelled.
            task = asyncio.get_running_loop().create_task(self._cancel(model, batch))
            self._submissions.add(task)
            task.add_done_callback(self._submissions.discard)

    async def _cancel(self, model: Model, batch: Batch) -> None:
        # Best effort, like cancelling an abandoned request's batch: nobody is left to report a failure to.
        with suppress(Exception):
            await model.batch_cancel(batch)

    def may_resubmit(self, ticket: _Ticket) -> bool:
        """Whether a request that failed in a way a new batch may not repeat has resubmissions left."""
        return ticket.get('resubmissions', 0) < self.max_resubmissions

    def abandon(self, batch: Batch, custom_id: str) -> bool:
        """Record that a request no longer waits on its batch, and return whether to cancel the batch.

        A provider cancels whole batches, so a batch is only cancelled once every request this pool submitted in it
        has been abandoned. A batch this pool didn't submit, like one recorded in a message history from another
        process, is left running, as is one that has ended. Returns `True` at most once per batch.
        """
        key = _batch_key(batch)
        with self._lock:
            self._forget_reader(key, custom_id)
            if (submitted := self._submitted.get(key)) is None or custom_id not in submitted:
                return False
            abandoned = self._abandoned.setdefault(key, set())
            abandoned.add(custom_id)
            if abandoned < submitted:
                return False
            self._untrack(key)
            return True

    def release(self, batch: Batch, custom_id: str) -> None:
        """Record that a run left a request waiting on `batch` for a later run to collect.

        The request keeps the batch running until it's abandoned: its caller can still cancel it explicitly, as the
        Harness does for a sub-agent run its delegating run can't resume. Nothing in this process waits to read the
        request's result any more.
        """
        key = _batch_key(batch)
        with self._lock:
            if (submitted := self._submitted.get(key)) is not None and custom_id in submitted:
                self._released.setdefault(key, set()).add(custom_id)
            self._forget_reader(key, custom_id)

    def resumed(self, batch: Batch, custom_id: str) -> None:
        """Record that a run resumed waiting on a request an earlier run left for it, so the batch is never cancelled.

        The resuming run can't be told apart from one that resumed a history this pool didn't submit, whose abandoning
        can't cancel the batch either: the same history may be resumed again elsewhere.
        """
        key = _batch_key(batch)
        with self._lock:
            if custom_id in self._released.get(key, ()):
                self._untrack(key)

    def _untrack(self, key: _BatchKey) -> None:
        """Forget a batch this pool submitted, which leaves it running unless it's being cancelled."""
        self._submitted.pop(key, None)
        self._abandoned.pop(key, None)
        self._released.pop(key, None)

    def _forget_reader(self, key: _BatchKey, custom_id: str) -> None:
        for check_key in [check_key for check_key in self._unread if check_key[1:] == key]:
            self._read(check_key, custom_id)
        self._evict_results()

    def collected(self, model: Model, batch: Batch, custom_id: str) -> None:
        """Record that a request read its result, dropping the batch's results once every request waiting on it has."""
        with self._lock:
            self._read(_check_key(model, batch), custom_id)

    def _read(self, check_key: _CheckKey, custom_id: str) -> None:
        if (unread := self._unread.get(check_key)) is None:
            return
        unread.discard(custom_id)
        if not unread:
            del self._unread[check_key]
            self._drop_results(check_key)

    def _drop_results(self, check_key: _CheckKey) -> None:
        self._results.pop(check_key, None)
        self._results_bytes.pop(check_key, None)

    def _evict_results(self) -> None:
        """Drop the least recently used results that no request this pool submitted still waits to read."""
        while len(self._results) > _RESULTS_CACHE_SIZE or sum(self._results_bytes.values()) > _RESULTS_CACHE_BYTES:
            if (key := next((key for key in self._results if key not in self._unread), None)) is None:
                return
            self._drop_results(key)

    async def poll(
        self, model: Model, ticket: _Ticket, requester: _Requester | None
    ) -> tuple[Batch, dict[str, BatchResult] | None]:
        """The batch's current state, and its results once it has ended, shared by concurrent checks of the batch."""
        batch = ticket['batch']
        check_key = _check_key(model, batch)
        with self._lock:
            if (ended := self._results.get(check_key)) is not None:
                self._results.move_to_end(check_key)
                return ended
            status = self._statuses.get(check_key)
        if status is not None:
            checked_at, checked = status
            if time.monotonic() - checked_at < self.poll_interval.total_seconds():
                return checked, None
        loop = asyncio.get_running_loop()
        key = (loop, check_key)
        if (poll := self._polls.get(key)) is None:
            poll = self._polls[key] = loop.create_task(
                self._fetch(model, batch, check_key, ticket.get('trace_context'), requester)
            )
            poll.add_done_callback(lambda done: self._poll_done(key, done))
        # Waits without cancelling the shared poll when this request is cancelled.
        await asyncio.wait((poll,))
        if poll.cancelled():
            raise UserError(_EXITED_MESSAGE)
        return poll.result()

    def _poll_done(
        self,
        key: tuple[asyncio.AbstractEventLoop, _CheckKey],
        poll: asyncio.Task[tuple[Batch, dict[str, BatchResult] | None]],
    ) -> None:
        self._polls.pop(key, None)
        # Marks a failure retrieved even when every request waiting on the check was cancelled; the others still
        # raise it from the task's result.
        if not poll.cancelled():
            poll.exception()

    async def _fetch(
        self,
        model: Model,
        batch: Batch,
        check_key: _CheckKey,
        trace_context: dict[str, str] | None,
        requester: _Requester | None,
    ) -> tuple[Batch, dict[str, BatchResult] | None]:
        links = [] if requester is None else [Link(requester.span.get_span_context())]
        with _batch_span(
            requester, f'batch check {model.model_name}', parent=_span_context(trace_context), links=links, model=model
        ) as span:
            if span is not None:
                span.set_attribute(BATCH_ID_ATTRIBUTE, batch.id)
            with _expired_after_retention(batch):
                batch = await model.batch_status(batch)
            if span is not None:
                span.set_attribute(BATCH_STATUS_ATTRIBUTE, batch.status)
            if batch.status != 'ended':
                with self._lock:
                    self._statuses[check_key] = (time.monotonic(), batch)
                    self._statuses.move_to_end(check_key)
                    while len(self._statuses) > _SUBMITTED_BATCHES_SIZE:
                        self._statuses.popitem(last=False)
                return batch, None
            with _expired_after_retention(batch):
                results = {result.custom_id: result for result in await model.batch_results(batch)}
        with self._lock:
            self._statuses.pop(check_key, None)
            self._results[check_key] = batch, results
            self._results_bytes[check_key] = sum(len((result.body or '').encode()) for result in results.values())
            # The requests this pool submitted and still waits on read the results, which are kept until they have.
            # Requests abandoned while it ran no longer matter: an ended batch can't be cancelled.
            key = _batch_key(batch)
            submitted = self._submitted.get(key, frozenset[str]())
            if unread := submitted - self._abandoned.get(key, set[str]()) - self._released.get(key, set[str]()):
                self._unread[check_key] = set(unread)
            self._untrack(key)
            self._evict_results()
        return batch, results


@dataclass(init=False)
class BatchedModel(WrapperModel):
    """Sends the wrapped model's requests through its provider's batch API, in a pool's batches."""

    pool: BatchPool

    def __init__(self, wrapped: Model, *, pool: BatchPool) -> None:
        super().__init__(wrapped)
        self.pool = pool

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        requester = _Requester.current()
        last_message = messages[-1] if messages else None
        if (ticket := _pending_ticket(last_message)) is not None:
            assert isinstance(last_message, ModelResponse)
            return await self._check(
                ticket, last_message, messages[:-1], model_settings, model_request_parameters, requester
            )
        return await self._submit(messages, model_settings, model_request_parameters, requester, resubmissions=0)

    async def _submit(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        requester: _Requester | None,
        *,
        resubmissions: int,
    ) -> ModelResponse:
        encoded = await self.wrapped.batch_encode(
            BatchRequest(
                custom_id='',
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
        )
        custom_id, batch, trace_context = await self.pool.enqueue(self.wrapped, encoded, requester)
        self.pool.link(requester, batch, trace_context)
        ticket = _Ticket(custom_id=custom_id, batch=batch, polls=0)
        if resubmissions:
            ticket['resubmissions'] = resubmissions
        if (
            isinstance(continued := messages[-1], ModelResponse)
            and continued.state == 'suspended'
            and continued.provider_response_id
        ):
            ticket['continued_response_id'] = continued.provider_response_id
        if trace_context is not None:
            ticket['trace_context'] = trace_context
        return ModelResponse(
            parts=[],
            # No model name: the result may name a dated model, and differing names would make the graph treat the
            # result as a different model's, dropping content this turn accumulated before it.
            provider_name=self.wrapped.system,
            provider_url=self.wrapped.base_url,
            provider_response_id=_response_id(batch.id, custom_id),
            provider_details=_ticket_details(ticket),
            state='suspended',
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        # A batch result arrives all at once, so it's waited for as `request` does, and its parts are then streamed as
        # one event each, as a durable engine replays a recorded response.
        response = await self.request(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters, replay_events=True)

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if (ticket := _pending_ticket(response)) is None:
            return self.wrapped.continuation_delay(response)
        return poll_delay(ticket['polls'], self.pool.poll_interval, self.pool.max_poll_interval)

    def _limits_polls(self, response: ModelResponse) -> bool:
        # A batch ends within the provider's batch window, as it expires once that has passed.
        return _pending_ticket(response) is None and self.wrapped._limits_polls(response)

    def _release_suspended_response(self, response: ModelResponse) -> None:
        if (ticket := _pending_ticket(response)) is None:
            return self.wrapped._release_suspended_response(response)
        self.pool.release(ticket['batch'], ticket['custom_id'])

    def _withdraws_rejected_suspended_response(self, response: ModelResponse) -> bool:
        # Nothing else collects a request whose pending response a hook rejected, so it's abandoned, which cancels its
        # batch once no other request waits on it.
        return _pending_ticket(response) is not None or self.wrapped._withdraws_rejected_suspended_response(response)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        if (ticket := _pending_ticket(response)) is None:
            return await self.wrapped.cancel_suspended_response(response)
        if self.pool.abandon(ticket['batch'], ticket['custom_id']):
            await self.wrapped.batch_cancel(ticket['batch'])

    async def _check(
        self,
        ticket: _Ticket,
        pending: ModelResponse,
        history: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        requester: _Requester | None,
    ) -> ModelResponse:
        if (provider := ticket['batch'].provider_name) != self.wrapped.system:
            raise UserError(
                f'The message history waits on a batch submitted to {provider!r}, '
                f"which the {self.wrapped.system!r} model serving this request can't check."
            )
        self.pool.resumed(ticket['batch'], ticket['custom_id'])
        trace_context = ticket.get('trace_context')
        self.pool.link(requester, ticket['batch'], trace_context)
        try:
            batch, results = await self.pool.poll(self.wrapped, ticket, requester)
        except BatchJobError as error:
            # The job ended without running its requests, which a new batch may well do.
            if transient(error) and self.pool.may_resubmit(ticket):
                return await self._resubmit(
                    ticket, pending, history, model_settings, model_request_parameters, requester
                )
            raise
        except ModelAPIError as error:
            failed_checks = ticket.get('failed_checks', 0) + 1
            if not check_failure_tolerated(error, failed_checks):
                raise
            checked = ticket.copy()
            checked['polls'] += 1
            checked['failed_checks'] = failed_checks
            return _checked_again(pending, checked)
        if results is None:
            checked = ticket.copy()
            checked['batch'] = batch
            checked['polls'] += 1
            checked.pop('failed_checks', None)
            return _checked_again(pending, checked)
        if (result := results.get(ticket['custom_id'])) is None and (
            unlisted := self.wrapped._batch_unlisted_status(batch)
        ) is not None:
            result = BatchResult(custom_id=ticket['custom_id'], status=unlisted)
        self.pool.collected(self.wrapped, batch, ticket['custom_id'])
        try:
            if result is None:
                raise missing_result_error(self.wrapped.model_name, batch.id)
            response = self.wrapped.batch_decode(
                result,
                BatchRequest(
                    custom_id=ticket['custom_id'],
                    messages=history,
                    model_settings=model_settings,
                    model_request_parameters=model_request_parameters,
                ),
            )
        except ModelAPIError as error:
            # A request the batch didn't run, or that failed in a way a live request is retried on, is submitted again,
            # as a provider SDK retries a live request: re-reading the same result would fail the same way every time.
            if (result is None or resubmittable(result, error)) and self.pool.may_resubmit(ticket):
                return await self._resubmit(
                    ticket, pending, history, model_settings, model_request_parameters, requester
                )
            raise
        # Replaces the internal ticket that merging the pending response would otherwise carry into the final one.
        details: dict[str, Any] = {'id': batch.id, 'custom_id': ticket['custom_id']}
        if resubmissions := ticket.get('resubmissions'):
            details['resubmissions'] = resubmissions
        response.provider_details = {**(response.provider_details or {}), _TICKET_KEY: details}
        return response

    async def _resubmit(
        self,
        ticket: _Ticket,
        pending: ModelResponse,
        history: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        requester: _Requester | None,
    ) -> ModelResponse:
        messages = history
        if pending.parts:
            # The pending response holds what a provider continuation, like Anthropic's `pause_turn`, generated before
            # the request, which the request sent as the turn to continue, so it's sent again without the ticket.
            details = {key: value for key, value in (pending.provider_details or {}).items() if key != _TICKET_KEY}
            continued = replace(
                pending, provider_response_id=ticket.get('continued_response_id'), provider_details=details or None
            )
            messages = [*history, continued]
        return await self._submit(
            messages,
            model_settings,
            model_request_parameters,
            requester,
            resubmissions=ticket.get('resubmissions', 0) + 1,
        )
