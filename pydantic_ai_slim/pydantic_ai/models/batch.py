"""Send model requests through a provider's batch API, for batch pricing in exchange for latency.

[`BatchModel`][pydantic_ai.models.batch.BatchModel] lets an agent use a batch API without changing how it runs, and
[`model_request_batch`][pydantic_ai.direct.model_request_batch] sends a list of requests as one batch. The `batch_*`
methods on [`Model`][pydantic_ai.models.Model] are the provider-level primitives both are built on.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, Self

from pydantic import TypeAdapter, ValidationError
from typing_extensions import Literal, TypeAlias, TypedDict  # noqa: UP035

from .._run_context import RunContext
from .._utils import cancel_and_drain
from ..exceptions import UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse
from ._batch_settings import DEFAULT_MAX_POLL_INTERVAL, DEFAULT_POLL_INTERVAL, check_batch_settings
from .wrapper import WrapperModel

__all__ = (
    'Batch',
    'BatchModel',
    'BatchRequest',
    'BatchResult',
    'BatchResultStatus',
    'BatchStatus',
    'EncodedBatchRequest',
)

BatchStatus: TypeAlias = Literal['in_progress', 'canceling', 'ended']
"""The processing state of a batch job. An ended job can hold a mix of per-request outcomes."""

BatchResultStatus: TypeAlias = Literal['succeeded', 'errored', 'expired', 'canceled']
"""How the provider finished one request of a batch."""


@dataclass(kw_only=True)
class BatchRequest:
    """One model request to send as part of a batch.

    Every aspect of the request can differ between the requests of a batch, as long as the provider accepts them in
    one job: [`Model.batch_encode`][pydantic_ai.models.Model.batch_encode] reports which ones it can combine.
    """

    custom_id: str
    """The ID that matches the request to its result. It must be unique within a batch."""

    messages: list[ModelMessage]
    """The messages to send to the model."""

    model_settings: ModelSettings | None = None
    """Settings for this request."""

    model_request_parameters: ModelRequestParameters = field(default_factory=ModelRequestParameters)
    """Tools, output schema and other request parameters for this request."""


@dataclass(frozen=True, kw_only=True)
class EncodedBatchRequest:
    """A [`BatchRequest`][pydantic_ai.models.batch.BatchRequest] encoded for one provider's batch API."""

    custom_id: str
    """The ID that matches the request to its result within the batch."""

    group: str
    """Everything requests must share to be submitted in one batch, such as the endpoint, credentials and model.

    It is stored on the [`Batch`][pydantic_ai.models.batch.Batch] handle, so it never contains secrets.
    """

    body: str
    """The provider-native request, as JSON."""


@dataclass(kw_only=True)
class Batch:
    """A handle to a submitted batch job.

    It holds only plain data, so you can store it, for example with a Pydantic `TypeAdapter`, and use it from another
    process to check on the job, retrieve its results or cancel it.
    """

    id: str
    """The provider's ID for the batch job."""

    model_name: str
    """The name of the model the batch was submitted to."""

    provider_name: str
    """The name of the provider the batch was submitted to."""

    group: str
    """The [`group`][pydantic_ai.models.batch.EncodedBatchRequest.group] the batch's requests share.

    Used to check that a later call uses a compatible client configuration.
    """

    status: BatchStatus
    """The job's processing state when this handle was last updated."""

    created_at: datetime
    """When the provider created the job."""

    request_count: int
    """The number of requests in the batch."""

    provider_details: dict[str, Any] | None = None
    """Provider-specific job details, such as the files a provider writes results to."""


@dataclass(frozen=True, kw_only=True)
class BatchResult:
    """The provider's result for one request of a batch.

    Turn it into a [`ModelResponse`][pydantic_ai.messages.ModelResponse] with
    [`Model.batch_decode`][pydantic_ai.models.Model.batch_decode].
    """

    custom_id: str
    """The ID of the request this result belongs to."""

    status: BatchResultStatus
    """How the provider finished the request."""

    body: str | None = None
    """The provider-native result or error, as JSON, when the request `'succeeded'` or `'errored'`."""


_TICKET_KEY = 'batch'
_RESULTS_CACHE_SIZE = 32
_EXITED_MESSAGE = '`BatchModel` exited while the request was waiting for its batch.'


class _Ticket(TypedDict):
    """What a pending batch response records about its request, under `provider_details['batch']`."""

    custom_id: str
    batch: Batch
    polls: int


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


@dataclass
class _Queue:
    requests: list[tuple[EncodedBatchRequest, asyncio.Future[Batch]]] = field(
        default_factory=list[tuple[EncodedBatchRequest, asyncio.Future[Batch]]]
    )
    timer: asyncio.TimerHandle | None = None


def _take(queues: dict[str, _Queue], group: str) -> _Queue:
    """Take a queue off for submission, so later requests start a new one."""
    queue = queues.pop(group)
    if queue.timer is not None:
        queue.timer.cancel()
    return queue


@dataclass(init=False)
class BatchModel(WrapperModel):
    """Send a model's requests through its provider's batch API.

    Batch APIs cost less, typically half, but a result can take up to a day. The agent runs as usual: each request is
    queued with other requests made in the same process, submitted in a batch, and checked on until its result arrives.
    Meanwhile the run's message history ends in a suspended response that records the batch, so a run resumed from
    that history, in this process or another, waits for the same batch instead of submitting the request again. To
    end the run instead of waiting, add [`PendingModelResponse`][pydantic_ai.output.PendingModelResponse] to the
    agent's `output_type`. See [Batch processing](../../batch.md) for a guide.

    Under [durable execution](../../durable_execution/overview.md), submitting a request and every check on its batch run
    as separate durable steps, so an interrupted run continues from the same batch when it's recovered.

    Requests can't be streamed, and the model needs an `asyncio` event loop.

    Submissions and checks on a batch run as tasks shared by the requests waiting on them. Using the model as an async
    context manager, for example through `async with agent:`, cancels any that are still running when it exits.
    """

    max_requests: int
    """The number of queued requests that triggers submission."""

    max_wait: timedelta | None
    """The longest the oldest queued request waits before submission, or `None` to wait for a full batch or `flush()`."""

    poll_interval: timedelta
    """The wait before the first check of a submitted batch."""

    max_poll_interval: timedelta
    """The longest wait between checks of a submitted batch."""

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        *,
        max_requests: int = 100,
        max_wait: timedelta | None = timedelta(seconds=30),
        poll_interval: timedelta = DEFAULT_POLL_INTERVAL,
        max_poll_interval: timedelta = DEFAULT_MAX_POLL_INTERVAL,
    ) -> None:
        """Wrap a model so its requests go through its provider's batch API.

        A batch is submitted when it holds `max_requests` requests, when its oldest request has waited `max_wait`, or
        when [`flush()`][pydantic_ai.models.batch.BatchModel.flush] is called, whichever comes first.

        Args:
            wrapped: The model to send batched requests to.
            max_requests: The most requests one batch holds. The provider's own limit caps it.
            max_wait: How long the oldest queued request waits for others before its batch is submitted. Later
                requests never extend it. `None` waits until the batch is full or `flush()` is called.
            poll_interval: How long to wait before first checking a submitted batch for its result. The interval grows
                by half after every check, up to `max_poll_interval`.
            max_poll_interval: The longest wait between checks of a submitted batch.
        """
        super().__init__(wrapped)
        if max_wait is not None and max_wait <= timedelta(0):
            raise UserError('`max_wait` must be positive.')
        self.max_requests = check_batch_settings(
            self.wrapped, max_requests=max_requests, poll_interval=poll_interval, max_poll_interval=max_poll_interval
        )
        self.max_wait = max_wait
        self.poll_interval = poll_interval
        self.max_poll_interval = max_poll_interval
        # Queues hold futures, which belong to one event loop, so each loop gets its own.
        self._queues: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, _Queue]] = (
            weakref.WeakKeyDictionary()
        )
        self._submissions: set[asyncio.Task[None]] = set()
        # Every request waiting on a batch checks on it, so concurrent checks and the results are shared between them.
        self._polls: dict[
            tuple[asyncio.AbstractEventLoop, str], asyncio.Task[tuple[Batch, dict[str, BatchResult] | None]]
        ] = {}
        self._results: OrderedDict[str, dict[str, BatchResult]] = OrderedDict()
        self._abandoned: dict[str, set[str]] = {}
        self._entered_count = 0

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        self._entered_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self._entered_count -= 1
        if self._entered_count == 0:
            loop = asyncio.get_running_loop()
            queues = self._queues.pop(loop, {})
            for group in list(queues):
                for _, future in _take(queues, group).requests:
                    future.set_exception(UserError(_EXITED_MESSAGE))
            await cancel_and_drain(
                *(task for task in self._submissions if task.get_loop() is loop),
                *(task for (task_loop, _), task in self._polls.items() if task_loop is loop),
            )
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
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
            )

        encoded = await self.wrapped.batch_encode(
            BatchRequest(
                custom_id='',
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
        )
        custom_id, batch = await self._enqueue(encoded)
        return ModelResponse(
            parts=[],
            # No model name: the result may name a dated model, and differing names would make the graph treat the
            # result as a different model's, dropping content this turn accumulated before it.
            provider_name=self.wrapped.system,
            provider_url=self.wrapped.base_url,
            provider_response_id=_response_id(batch.id, custom_id),
            provider_details=_ticket_details(_Ticket(custom_id=custom_id, batch=batch, polls=0)),
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
        raise UserError('`BatchModel` does not support streaming: a batch result arrives all at once.')
        # Required to type the context manager; streaming always raises.
        yield  # pragma: no cover

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if (ticket := _pending_ticket(response)) is None:
            return self.wrapped.continuation_delay(response)
        delay = self.poll_interval.total_seconds() * 1.5 ** ticket['polls']
        return min(delay, self.max_poll_interval.total_seconds())

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        if (ticket := _pending_ticket(response)) is None:
            return await self.wrapped.cancel_suspended_response(response)
        batch = ticket['batch']
        abandoned = self._abandoned.setdefault(batch.id, set())
        abandoned.add(ticket['custom_id'])
        # A provider cancels whole batches, so only cancel once no request in it is still waiting. This process only
        # knows about the requests abandoned here; any others keep the batch running.
        if len(abandoned) == batch.request_count:
            del self._abandoned[batch.id]
            await self.wrapped.batch_cancel(batch)

    async def flush(self) -> None:
        """Submit every queued request now instead of waiting for a full batch or `max_wait`."""
        loop = asyncio.get_running_loop()
        queues = self._queues.get(loop, {})
        if submissions := [self._start_submission(loop, queues, group) for group in list(queues)]:
            # Waits without cancelling the submissions, so cancelling the caller doesn't fail the queued requests.
            await asyncio.wait(submissions)

    def _start_submission(
        self, loop: asyncio.AbstractEventLoop, queues: dict[str, _Queue], group: str
    ) -> asyncio.Task[None]:
        """Submit a group's queue in a task of its own, so cancelling whoever triggered it doesn't cancel the others.

        The queue is taken off `queues` right away, so a request made before the submission runs starts the next batch.
        """
        task = loop.create_task(self._submit(_take(queues, group)))
        self._submissions.add(task)
        task.add_done_callback(self._submissions.discard)
        return task

    async def _enqueue(self, encoded: EncodedBatchRequest) -> tuple[str, Batch]:
        """Queue a request, and return the ID it was given in the batch it was submitted in."""
        loop = asyncio.get_running_loop()
        queues = self._queues.setdefault(loop, {})
        queue = queues.setdefault(encoded.group, _Queue())
        # Derived from the request, so a rerun of the same requests gets the same IDs; unique within the batch.
        base_id = custom_id = hashlib.sha256(encoded.body.encode()).hexdigest()[:32]
        taken = {request.custom_id for request, _ in queue.requests}
        duplicates = itertools.count(1)
        while custom_id in taken:
            custom_id = f'{base_id}-{next(duplicates)}'
        encoded = replace(encoded, custom_id=custom_id)
        future: asyncio.Future[Batch] = loop.create_future()
        queue.requests.append((encoded, future))

        def submit() -> None:
            self._start_submission(loop, queues, encoded.group)

        if len(queue.requests) >= self.max_requests:
            submit()
        elif queue.timer is None and self.max_wait is not None:
            queue.timer = loop.call_later(self.max_wait.total_seconds(), submit)
        try:
            # Waits without cancelling the future on cancellation, as the submission resolves it.
            await asyncio.wait((future,))
        except asyncio.CancelledError:
            # Leaves the batch if it's still queued; a submitted batch has already taken the queue's requests.
            queue.requests = [(request, f) for request, f in queue.requests if f is not future]
            raise
        if future.cancelled():
            # The submission was cancelled, so there's no batch to wait for.
            raise UserError(_EXITED_MESSAGE)
        return custom_id, future.result()

    async def _submit(self, queue: _Queue) -> None:
        if not (requests := queue.requests):
            # Every queued request was cancelled before the deadline.
            return
        try:
            batch = await self.wrapped.batch_create([request for request, _ in requests])
        except Exception as error:
            for _, future in requests:
                future.set_exception(error)
        else:
            for _, future in requests:
                future.set_result(batch)
        finally:
            # A submission that was itself cancelled, for example as the event loop shuts down, leaves nothing to wait for.
            for _, future in requests:
                future.cancel()

    async def _check(self, ticket: _Ticket, pending: ModelResponse, request: BatchRequest) -> ModelResponse:
        batch, results = await self._poll(ticket['batch'])
        if results is None:
            # Unchanged apart from the poll count: the same `provider_response_id` makes the graph replace the pending
            # response with this one, so content the response accumulated before the batch is kept.
            return replace(
                pending,
                provider_details={
                    **(pending.provider_details or {}),
                    **_ticket_details(_Ticket(custom_id=ticket['custom_id'], batch=batch, polls=ticket['polls'] + 1)),
                },
            )
        result = results.get(ticket['custom_id']) or BatchResult(custom_id=ticket['custom_id'], status='expired')
        response = self.wrapped.batch_decode(result, request)
        # Replaces the internal ticket that merging the pending response would otherwise carry into the final one.
        response.provider_details = {
            **(response.provider_details or {}),
            _TICKET_KEY: {'id': batch.id, 'custom_id': ticket['custom_id']},
        }
        return response

    async def _poll(self, batch: Batch) -> tuple[Batch, dict[str, BatchResult] | None]:
        """The batch's current state, and its results once it has ended, shared by concurrent checks of the batch."""
        if (results := self._results.get(batch.id)) is not None:
            self._results.move_to_end(batch.id)
            return batch, results
        key = (asyncio.get_running_loop(), batch.id)
        if (poll := self._polls.get(key)) is None:
            poll = self._polls[key] = asyncio.create_task(self._fetch(batch))
            poll.add_done_callback(lambda _: self._polls.pop(key, None))
        # Waits without cancelling the shared poll when this request is cancelled.
        await asyncio.wait((poll,))
        if poll.cancelled():
            raise UserError(_EXITED_MESSAGE)
        return poll.result()

    async def _fetch(self, batch: Batch) -> tuple[Batch, dict[str, BatchResult] | None]:
        batch = await self.wrapped.batch_status(batch)
        if batch.status != 'ended':
            return batch, None
        results = {result.custom_id: result for result in await self.wrapped.batch_results(batch)}
        self._results[batch.id] = results
        while len(self._results) > _RESULTS_CACHE_SIZE:
            self._results.popitem(last=False)
        return batch, results
