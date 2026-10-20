"""The batching behind [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests].

A `BatchPool` collects the requests of every model it wraps into batches, and checks on them. `BatchedModel` is the
wrapper the capability puts around the model serving each request, so requests from any agent or run that uses the
same capability instance share one pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, Generator
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
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import Model, ModelRequestParameters, StreamedResponse
from ._batch_settings import DEFAULT_MAX_POLL_INTERVAL, DEFAULT_POLL_INTERVAL, batch_size, check_batch_intervals
from .batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from .wrapper import WrapperModel

_TICKET_KEY = 'batch'
_RESULTS_CACHE_SIZE = 32
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
    withdrawn: bool = False
    """Whether the request was cancelled after its queue was taken for submission, so nobody waits on its result."""


@dataclass
class _Queue:
    model: Model
    """The model that submits the queue: any model whose requests share the queue's group can."""
    max_requests: int
    requests: list[_Queued] = field(default_factory=list[_Queued])
    timer: asyncio.TimerHandle | None = None


_QueueKey = tuple[str, str]
"""The provider and `group` of a queue's requests: groups are only comparable within one provider."""

_BatchKey = tuple[str, str, str]
"""The provider, group and ID of a batch: IDs are only unique within one provider account."""


def _batch_key(batch: Batch) -> _BatchKey:
    return batch.provider_name, batch.group, batch.id


def _take(queues: dict[_QueueKey, _Queue], key: _QueueKey) -> _Queue:
    """Take a queue off for submission, so later requests start a new one."""
    queue = queues.pop(key)
    if queue.timer is not None:
        queue.timer.cancel()
    return queue


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
        max_requests: int = 100,
        max_wait: timedelta | None = timedelta(seconds=30),
        poll_interval: timedelta = DEFAULT_POLL_INTERVAL,
        max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL,
    ) -> None:
        check_batch_intervals(
            max_requests=max_requests, poll_interval=poll_interval, max_poll_interval=max_poll_interval
        )
        if max_wait is not None and max_wait <= timedelta(0):
            raise UserError('`max_wait` must be positive.')
        self.max_requests = max_requests
        self.max_wait = max_wait
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        # Queues hold futures, which belong to one event loop, so each loop gets its own.
        self._queues: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[_QueueKey, _Queue]] = (
            weakref.WeakKeyDictionary()
        )
        self._submissions: set[asyncio.Task[None]] = set()
        # Every request waiting on a batch checks on it, so concurrent checks and the results are shared between them.
        self._polls: dict[
            tuple[asyncio.AbstractEventLoop, _BatchKey], asyncio.Task[tuple[Batch, dict[str, BatchResult] | None]]
        ] = {}
        self._results: OrderedDict[_BatchKey, dict[str, BatchResult]] = OrderedDict()
        self._abandoned: dict[_BatchKey, set[str]] = {}
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
            for queued in _take(queues, key).requests:
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

    def poll_delay(self, polls: int) -> float:
        delay = self.poll_interval.total_seconds() * 1.5**polls
        return min(delay, self.max_poll_interval.total_seconds())

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
        task = loop.create_task(self._submit(_take(queues, key)))
        self._submissions.add(task)
        task.add_done_callback(self._submissions.discard)
        return task

    async def enqueue(
        self, model: Model, encoded: EncodedBatchRequest, requester: _Requester | None
    ) -> tuple[str, Batch, dict[str, str] | None]:
        """Queue a request, and return the ID it was given in its batch, the batch, and the batch span's context."""
        loop = asyncio.get_running_loop()
        queues = self._queues.setdefault(loop, {})
        key = (model.system, encoded.group)
        queue = queues.get(key)
        if queue is None:
            queue = queues[key] = _Queue(model, max_requests=batch_size(model, self.max_requests))
        # Derived from the request, so a rerun of the same requests gets the same IDs; unique within the batch.
        base_id = custom_id = hashlib.sha256(encoded.body.encode()).hexdigest()[:32]
        taken = {queued.request.custom_id for queued in queue.requests}
        duplicates = itertools.count(1)
        while custom_id in taken:
            custom_id = f'{base_id}-{next(duplicates)}'
        queued = _Queued(replace(encoded, custom_id=custom_id), loop.create_future(), requester)
        queue.requests.append(queued)

        def submit() -> None:
            self._start_submission(loop, queues, key)

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
            queue.requests = [other for other in queue.requests if other is not queued]
            queued.withdrawn = True
            if queued.future.done() and not queued.future.cancelled() and queued.future.exception() is None:
                self._withdraw(model, queued.future.result()[0], custom_id)
            raise
        if queued.future.cancelled():
            # The submission was cancelled, so there's no batch to wait for.
            raise UserError(_EXITED_MESSAGE)
        batch, trace_context = queued.future.result()
        return custom_id, batch, trace_context

    async def _submit(self, queue: _Queue) -> None:
        if not (requests := queue.requests):
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
            for queued in requests:
                queued.future.set_result((batch, trace_context))
            for queued in requests:
                if queued.withdrawn:
                    self._withdraw(queue.model, batch, queued.request.custom_id)
        finally:
            # A submission that was itself cancelled, for example as the event loop shuts down, leaves nothing to wait for.
            for queued in requests:
                queued.future.cancel()

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

    def abandon(self, batch: Batch, custom_id: str) -> bool:
        """Record that a request no longer waits on its batch, and return whether none still does."""
        abandoned = self._abandoned.setdefault(_batch_key(batch), set())
        abandoned.add(custom_id)
        # A provider cancels whole batches, so only cancel once no request in it is still waiting. This process only
        # knows about the requests abandoned here; any others keep the batch running.
        if len(abandoned) < batch.request_count:
            return False
        del self._abandoned[_batch_key(batch)]
        return True

    async def poll(
        self, model: Model, ticket: _Ticket, requester: _Requester | None
    ) -> tuple[Batch, dict[str, BatchResult] | None]:
        """The batch's current state, and its results once it has ended, shared by concurrent checks of the batch."""
        batch = ticket['batch']
        if (results := self._results.get(_batch_key(batch))) is not None:
            self._results.move_to_end(_batch_key(batch))
            return batch, results
        loop = asyncio.get_running_loop()
        key = (loop, _batch_key(batch))
        if (poll := self._polls.get(key)) is None:
            poll = self._polls[key] = loop.create_task(
                self._fetch(model, batch, ticket.get('trace_context'), requester)
            )
            poll.add_done_callback(lambda _: self._polls.pop(key, None))
        # Waits without cancelling the shared poll when this request is cancelled.
        await asyncio.wait((poll,))
        if poll.cancelled():
            raise UserError(_EXITED_MESSAGE)
        return poll.result()

    async def _fetch(
        self, model: Model, batch: Batch, trace_context: dict[str, str] | None, requester: _Requester | None
    ) -> tuple[Batch, dict[str, BatchResult] | None]:
        links = [] if requester is None else [Link(requester.span.get_span_context())]
        with _batch_span(
            requester, f'batch check {model.model_name}', parent=_span_context(trace_context), links=links, model=model
        ) as span:
            if span is not None:
                span.set_attribute(BATCH_ID_ATTRIBUTE, batch.id)
            batch = await model.batch_status(batch)
            if span is not None:
                span.set_attribute(BATCH_STATUS_ATTRIBUTE, batch.status)
            if batch.status != 'ended':
                return batch, None
            results = {result.custom_id: result for result in await model.batch_results(batch)}
        self._results[_batch_key(batch)] = results
        while len(self._results) > _RESULTS_CACHE_SIZE:
            self._results.popitem(last=False)
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
                ticket,
                last_message,
                BatchRequest(
                    custom_id=ticket['custom_id'],
                    messages=messages[:-1],
                    model_settings=model_settings,
                    model_request_parameters=model_request_parameters,
                ),
                requester,
            )

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
        raise UserError('`BatchModelRequests` does not support streaming: a batch result arrives all at once.')
        # Required to type the context manager; streaming always raises.
        yield  # pragma: no cover

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if (ticket := _pending_ticket(response)) is None:
            return self.wrapped.continuation_delay(response)
        return self.pool.poll_delay(ticket['polls'])

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        if (ticket := _pending_ticket(response)) is None:
            return await self.wrapped.cancel_suspended_response(response)
        if self.pool.abandon(ticket['batch'], ticket['custom_id']):
            await self.wrapped.batch_cancel(ticket['batch'])

    async def _check(
        self, ticket: _Ticket, pending: ModelResponse, request: BatchRequest, requester: _Requester | None
    ) -> ModelResponse:
        trace_context = ticket.get('trace_context')
        self.pool.link(requester, ticket['batch'], trace_context)
        batch, results = await self.pool.poll(self.wrapped, ticket, requester)
        if results is None:
            # Unchanged apart from the poll count: the same `provider_response_id` makes the graph replace the pending
            # response with this one, so content the response accumulated before the batch is kept.
            checked = ticket.copy()
            checked['batch'] = batch
            checked['polls'] += 1
            return replace(pending, provider_details={**(pending.provider_details or {}), **_ticket_details(checked)})
        result = results.get(ticket['custom_id']) or BatchResult(custom_id=ticket['custom_id'], status='expired')
        response = self.wrapped.batch_decode(result, request)
        # Replaces the internal ticket that merging the pending response would otherwise carry into the final one.
        response.provider_details = {
            **(response.provider_details or {}),
            _TICKET_KEY: {'id': batch.id, 'custom_id': ticket['custom_id']},
        }
        return response
