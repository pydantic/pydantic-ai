"""The Temporal workflow that collects `BatchModel` requests into provider batches.

One queue workflow runs per agent, registered model, request group and task queue, with an ID derived from them,
so every worker feeds the same queue. Requests arrive as updates, which the server records before
acknowledging them, so an accepted request survives any crash. The queue submits a batch at the
policy's request count, at its byte limit, or when its oldest request has waited `max_wait`, then
polls the batch. Results don't pass through the queue: the poll activity signals each one straight to
the workflow that asked for it, so the queue's history only ever holds requests waiting to be submitted.

Everything the queue holds is bounded to fit Temporal's 2 MB payload limit, since its state is carried
across continue-as-new: requests waiting to be submitted, and a tombstone per finished request kept for
`_RETENTION` so a retried or cancelled enqueue is recognized.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from pydantic_core import to_json
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from typing_extensions import Literal  # noqa: UP035

from pydantic_ai.models.batch import BatchRequest, BatchResult

RESULT_SIGNAL = 'pydantic_ai__batch_result'
"""The signal that delivers a request's result to the workflow that enqueued it."""

MAX_REQUEST_BYTES = 512 * 1024
"""The largest encoded request a queue accepts, as it is held inline in workflow history."""

_MAX_QUEUE_BYTES = 1024 * 1024
"""The most encoded request bytes a queue holds before backpressure makes new requests wait."""

_MAX_ITEMS = 500
"""The most requests waiting to be submitted; serialized state also bounds submitted and finished records."""

_RETENTION = timedelta(hours=1)
"""How long a finished request's tombstone is kept, covering retries of the activity that enqueued it."""

_MAX_ACCEPTED_BYTES = 3 * 1024 * 1024
_MAX_OPERATIONS = 500

_MAX_POLL_FAILURES = 10

_SUBMIT_TIMEOUT = timedelta(minutes=5)
_POLL_TIMEOUT = timedelta(minutes=10)

ItemState = Literal['queued', 'submitting', 'submitted', 'done', 'canceled']


@dataclass(frozen=True, kw_only=True)
class QueuePolicy:
    """Everything that decides how a queue groups and submits its requests; every worker feeding it must agree."""

    agent_name: str
    group: str
    model_id: str | None
    max_requests: int
    max_wait: timedelta
    poll_interval: timedelta

    def activity_name(self, operation: str) -> str:
        return f'agent__{self.agent_name}__batch_{operation}'


@dataclass(frozen=True, kw_only=True)
class Consumer:
    """A request, and the workflow run waiting for its result."""

    item_id: str
    workflow_id: str
    run_id: str


@dataclass(frozen=True, kw_only=True)
class Outcome:
    """What became of a request: its provider result, or why it has none."""

    item_id: str
    batch_id: str | None = None
    result: BatchResult | None = None
    error: str | None = None


@dataclass(kw_only=True)
class QueueItem:
    consumer: Consumer
    accepted_at: datetime
    size: int = 0
    request: BatchRequest | None = None
    state: ItemState = 'queued'
    batch_id: str | None = None
    error: str | None = None
    finished_at: datetime | None = None


@dataclass(kw_only=True)
class QueueState:
    policy: QueuePolicy
    items: dict[str, QueueItem] = field(default_factory=dict[str, QueueItem])
    next_polls: dict[str, datetime] = field(default_factory=dict[str, datetime])
    poll_failures: dict[str, int] = field(default_factory=dict[str, int])


@dataclass(frozen=True, kw_only=True)
class EnqueueRequest:
    consumer: Consumer
    request: BatchRequest
    size: int
    policy: QueuePolicy


@dataclass(frozen=True, kw_only=True)
class ItemStatus:
    state: ItemState
    batch_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, kw_only=True)
class SubmitParams:
    model_id: str | None
    requests: list[BatchRequest]


@dataclass(frozen=True, kw_only=True)
class PollParams:
    group: str
    model_id: str | None
    batch_id: str
    consumers: list[Consumer]


@dataclass(frozen=True, kw_only=True)
class PollResult:
    ended: bool
    missing: list[str] = field(default_factory=list[str])


@dataclass(frozen=True, kw_only=True)
class CancelJobParams:
    group: str
    model_id: str | None
    batch_id: str


@workflow.defn(name='pydantic_ai__batch_queue')
class BatchQueueWorkflow:
    @workflow.init
    def __init__(self, state: QueueState) -> None:
        self._state = state
        self._changed = False
        self._updates = 0
        self._accepted_bytes = 0
        self._operations = 0

    @workflow.run
    async def run(self, state: QueueState) -> None:
        while True:
            self._purge()
            info = workflow.info()
            if (
                self._updates >= 500
                or info.is_continue_as_new_suggested()
                or self._accepted_bytes >= _MAX_ACCEPTED_BYTES
                or self._operations >= _MAX_OPERATIONS
            ):
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(self._state)
            if (batch := self._ready_batch()) is not None:
                await self._submit(batch)
                continue
            if (batch_id := self._due_poll()) is not None:
                await self._poll(batch_id)
                continue
            wake = self._next_wake()
            if wake is None:
                await workflow.wait_condition(workflow.all_handlers_finished)
                return
            self._changed = False
            # Every pending deadline lies ahead after the checks above; the floor keeps the timer positive.
            timeout = max(wake - workflow.now(), timedelta(milliseconds=1))
            try:
                await workflow.wait_condition(lambda: self._changed, timeout=timeout)
            except TimeoutError:
                pass

    @workflow.update
    def enqueue(self, request: EnqueueRequest) -> datetime:
        self._updates += 1
        self._accepted_bytes += len(to_json(request))
        item_id = request.consumer.item_id
        item = self._state.items.get(item_id)
        if item is None:
            item = QueueItem(
                consumer=request.consumer, accepted_at=workflow.now(), size=request.size, request=request.request
            )
            self._state.items[item_id] = item
            self._changed = True
        return item.accepted_at

    @enqueue.validator
    def _validate_enqueue(self, request: EnqueueRequest) -> None:
        if request.policy != self._state.policy:
            raise ApplicationError(
                f'Workers disagree on the batch policy of queue {workflow.info().workflow_id!r}: it runs with '
                f'{self._state.policy}, this request came with {request.policy}. Deploy one `BatchSettings` per '
                'agent and model; a changed policy takes effect once the queue has drained.',
                type='BatchPolicyConflict',
                non_retryable=True,
            )
        item = self._state.items.get(request.consumer.item_id)
        if item is not None:
            if item.state == 'canceled':
                raise ApplicationError(
                    'The batch request was cancelled before the queue accepted it.',
                    type='BatchRequestCancelled',
                    non_retryable=True,
                )
            return
        candidate = replace(
            self._state,
            items={
                **self._state.items,
                request.consumer.item_id: QueueItem(
                    consumer=request.consumer, accepted_at=workflow.now(), size=request.size, request=request.request
                ),
            },
        )
        if (
            len(to_json(candidate)) > _MAX_QUEUE_BYTES
            or sum(item.state == 'queued' for item in self._state.items.values()) >= _MAX_ITEMS
        ):
            raise ApplicationError(
                'The batch queue is full; the request is retried once it has submitted a batch.', type='BatchQueueFull'
            )

    @workflow.update
    def cancel(self, item_id: str) -> None:
        self._updates += 1
        now = workflow.now()
        item = self._state.items.get(item_id)
        if item is None:
            # A tombstone, so the enqueue this cancel overtook is rejected when it arrives.
            self._state.items[item_id] = QueueItem(
                consumer=Consumer(item_id=item_id, workflow_id='', run_id=''),
                accepted_at=now,
                state='canceled',
                finished_at=now,
            )
        elif item.state in ('queued', 'submitting', 'submitted'):
            if item.state == 'queued':
                item.request = None
            item.state = 'canceled'
            item.finished_at = now
        self._changed = True

    @workflow.query
    def status(self, item_id: str) -> ItemStatus | None:
        item = self._state.items.get(item_id)
        return None if item is None else ItemStatus(state=item.state, batch_id=item.batch_id, error=item.error)

    @cancel.validator
    def _validate_cancel(self, item_id: str) -> None:
        if item_id in self._state.items:
            return
        now = workflow.now()
        candidate = replace(
            self._state,
            items={
                **self._state.items,
                item_id: QueueItem(
                    consumer=Consumer(item_id=item_id, workflow_id='', run_id=''),
                    accepted_at=now,
                    state='canceled',
                    finished_at=now,
                ),
            },
        )
        if len(to_json(candidate)) > _MAX_QUEUE_BYTES:
            raise ApplicationError(
                'The batch queue is full; cancellation is retried after retention.', type='BatchQueueFull'
            )

    def _purge(self) -> None:
        cutoff = workflow.now() - _RETENTION
        expired = [
            item_id
            for item_id, item in self._state.items.items()
            if item.finished_at is not None and item.finished_at <= cutoff
        ]
        for item_id in expired:
            del self._state.items[item_id]

    def _ready_batch(self) -> list[QueueItem] | None:
        policy = self._state.policy
        queued = [item for item in self._state.items.values() if item.state == 'queued']
        if not queued:
            return None
        held = sum(item.size for item in queued)
        # Dicts keep insertion order, so the first queued item is the oldest.
        if (
            len(queued) < min(policy.max_requests, _MAX_ITEMS)
            and held + MAX_REQUEST_BYTES <= _MAX_QUEUE_BYTES
            and workflow.now() < queued[0].accepted_at + policy.max_wait
        ):
            return None
        return queued[: policy.max_requests]

    def _due_poll(self) -> str | None:
        now = workflow.now()
        return next((batch_id for batch_id, due in self._state.next_polls.items() if due <= now), None)

    def _next_wake(self) -> datetime | None:
        policy = self._state.policy
        wakes = [*self._state.next_polls.values()]
        wakes.extend(item.finished_at + _RETENTION for item in self._state.items.values() if item.finished_at)
        if queued := next((item for item in self._state.items.values() if item.state == 'queued'), None):
            wakes.append(queued.accepted_at + policy.max_wait)
        return min(wakes, default=None)

    def _finish(self, item: QueueItem, *, error: str | None = None) -> None:
        if item.state in ('submitting', 'submitted'):
            item.state = 'done'
            item.error = error
            item.consumer = Consumer(item_id=item.consumer.item_id, workflow_id='', run_id='')
            item.size = 0
            item.finished_at = workflow.now()

    async def _submit(self, batch: list[QueueItem]) -> None:
        self._operations += 1
        policy = self._state.policy
        requests: list[BatchRequest] = []
        for item in batch:
            assert item.request is not None, 'a queued item holds its request'
            requests.append(item.request)
            item.state = 'submitting'
        try:
            batch_id = await workflow.execute_activity(
                policy.activity_name('submit'),
                SubmitParams(model_id=policy.model_id, requests=requests),
                result_type=str,
                start_to_close_timeout=_SUBMIT_TIMEOUT,
                # The provider may have created the batch before a failure reached us, and its API takes no
                # idempotency key, so a retry could submit and bill it twice.
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        except ActivityError as error:
            message = (
                'Submitting the batch failed, and the provider may or may not have created it, so it was not '
                f'resubmitted: {error.cause or error}'
            )
            for item in batch:
                item.request = None
                self._finish(item, error=message)
            return
        for item in batch:
            item.request = None
            item.batch_id = batch_id
            if item.state == 'submitting':
                item.state = 'submitted'
        self._state.next_polls[batch_id] = workflow.now() + policy.poll_interval

    async def _poll(self, batch_id: str) -> None:
        self._operations += 1
        policy = self._state.policy
        waiting = [
            item for item in self._state.items.values() if item.batch_id == batch_id and item.state == 'submitted'
        ]
        if not waiting:
            del self._state.next_polls[batch_id]
            self._state.poll_failures.pop(batch_id, None)
            await self._cancel_job(batch_id)
            return
        try:
            result = await workflow.execute_activity(
                policy.activity_name('poll'),
                PollParams(
                    group=policy.group,
                    model_id=policy.model_id,
                    batch_id=batch_id,
                    consumers=[item.consumer for item in waiting],
                ),
                result_type=PollResult,
                start_to_close_timeout=_POLL_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except ActivityError as error:
            failures = self._state.poll_failures.get(batch_id, 0) + 1
            if failures < _MAX_POLL_FAILURES:
                self._state.poll_failures[batch_id] = failures
                self._state.next_polls[batch_id] = workflow.now() + policy.poll_interval
                return
            del self._state.next_polls[batch_id]
            del self._state.poll_failures[batch_id]
            for item in waiting:
                self._finish(item, error=f'Collecting batch {batch_id!r} failed: {error.cause or error}')
            return
        if not result.ended:
            self._state.next_polls[batch_id] = workflow.now() + policy.poll_interval
            return
        del self._state.next_polls[batch_id]
        self._state.poll_failures.pop(batch_id, None)
        for item in waiting:
            missing = item.consumer.item_id in result.missing
            self._finish(item, error=f'Batch {batch_id!r} ended without a result for it.' if missing else None)

    async def _cancel_job(self, batch_id: str) -> None:
        """Cancel a batch whose every request was abandoned; requests it already ran may still be billed."""
        policy = self._state.policy
        try:
            await workflow.execute_activity(
                policy.activity_name('cancel_job'),
                CancelJobParams(group=policy.group, model_id=policy.model_id, batch_id=batch_id),
                start_to_close_timeout=_SUBMIT_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except ActivityError:
            # Best effort: the batch ends on its own, and nothing waits for it.
            pass
