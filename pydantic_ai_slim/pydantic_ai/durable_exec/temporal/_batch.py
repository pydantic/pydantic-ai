"""Route `BatchModel` requests through a `BatchQueueWorkflow`, from the agent's workflow and its activities.

The agent's workflow draws a deterministic item ID, enqueues the request in a short activity, and then
waits in workflow code for the result the queue's poll activity signals back, keyed by that ID. Each wait
is bounded: when it runs out, a collect activity asks the queue directly, which recovers a result whose
signal never arrived and reports a request the queue lost, instead of waiting forever.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, cast

import anyio
from pydantic import ConfigDict, with_config
from pydantic_core import to_json
from temporalio import activity, workflow
from temporalio.client import WithStartWorkflowOperation, WorkflowUpdateFailedError
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.workflow import ActivityConfig
from typing_extensions import TypeVar

from pydantic_ai._utils import is_str_dict
from pydantic_ai.durable_exec._batch import BatchTicket
from pydantic_ai.durable_exec._operation import ModelRequestParams
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import BatchModel, BatchProvider
from pydantic_ai.settings import ModelSettings

from ._activity_execution import execute_activity
from ._batch_queue import (
    MAX_REQUEST_BYTES,
    RESULT_SIGNAL,
    BatchQueueWorkflow,
    CancelJobParams,
    Consumer,
    EnqueueRequest,
    Outcome,
    PollParams,
    PollResult,
    QueuePolicy,
    QueueState,
    SubmitParams,
)
from ._model_errors import model_errors_as_application_errors, rebuilt_model_errors
from ._toolset import heartbeating

_ENQUEUE_TIMEOUT = timedelta(minutes=30)

_WAIT_ROUND = timedelta(minutes=10)
"""How long one wait for a signalled result lasts before the queue is asked directly.

The continuation loop allows 1,000 polls of one request, so this covers about a week: a provider's
processing window plus queueing.
"""


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class EnqueueParams:
    item_id: str
    model_id: str | None
    messages: list[ModelMessage]
    # Not `ModelSettings`, which would drop the fields only its subclasses define.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CollectParams:
    ticket: BatchTicket
    outcome: Outcome | None
    model_id: str | None
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters


_Result = TypeVar('_Result')


class _ResultInbox:
    """The agent workflow's signal handler for batch results, keyed by item ID."""

    def __init__(self) -> None:
        self.outcomes: dict[str, Outcome] = {}
        self.pending: set[str] = set()

    def __call__(self, outcome: Outcome) -> None:
        if outcome.item_id in self.pending:
            self.outcomes.setdefault(outcome.item_id, outcome)


def _inbox() -> _ResultInbox:
    """The current workflow's inbox, installed on first use; signals that arrived earlier are buffered until then."""
    handler = workflow.get_signal_handler(RESULT_SIGNAL)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(handler, _ResultInbox):
        return handler
    inbox = _ResultInbox()
    workflow.set_signal_handler(RESULT_SIGNAL, inbox)  # pyright: ignore[reportUnknownMemberType]
    return inbox


def _queue_id(policy: QueuePolicy, group: str) -> str:
    group_hash = hashlib.sha256(group.encode()).hexdigest()[:16]
    return f'pydantic-ai-batch:{policy.agent_name}:{policy.model_id or "default"}:{group_hash}'


async def _shielded(cleanup: Coroutine[Any, Any, object]) -> None:
    """Run cleanup to completion even when the workflow is being cancelled; its own failure is dropped."""
    with anyio.CancelScope(shield=True):
        with suppress(ActivityError, RPCError):
            await cleanup


class TemporalBatchBinding:
    """The `DurableBatchBinding` for Temporal, with the activities the agent's worker registers for it."""

    def __init__(
        self,
        *,
        agent_name: str,
        resolve_model: Callable[[str | None], BatchModel],
        activity_config: ActivityConfig,
    ) -> None:
        self._agent_name = agent_name
        self._resolve_model = resolve_model
        self._activity_config = activity_config
        prefix = f'agent__{agent_name}__batch_'

        @activity.defn(name=f'{prefix}enqueue')
        async def enqueue(params: EnqueueParams) -> BatchTicket:
            async with heartbeating():
                return await self._enqueue_activity(params)

        @activity.defn(name=f'{prefix}withdraw')
        async def withdraw(params: EnqueueParams) -> None:
            async with heartbeating():
                await self._withdraw_activity(params)

        @activity.defn(name=f'{prefix}collect')
        async def collect(params: CollectParams) -> ModelResponse | None:
            async with heartbeating():
                return await self._collect_activity(params)

        @activity.defn(name=f'{prefix}cancel')
        async def cancel(ticket: BatchTicket) -> None:
            async with heartbeating():
                await self._cancel_activity(ticket)

        @activity.defn(name=f'{prefix}submit')
        async def submit(params: SubmitParams) -> str:
            async with heartbeating():
                return await self._resolve_model(params.model_id).batch_provider.create(params.requests)

        @activity.defn(name=f'{prefix}poll')
        async def poll(params: PollParams) -> PollResult:
            async with heartbeating():
                return await self._poll_activity(params)

        @activity.defn(name=f'{prefix}cancel_job')
        async def cancel_job(params: CancelJobParams) -> None:
            async with heartbeating():
                await self._resolve_model(params.model_id).batch_provider.cancel(params.batch_id, group=params.group)

        self._enqueue = enqueue
        self._withdraw = withdraw
        self._collect = collect
        self._cancel = cancel
        self.registrations: list[Callable[..., Any]] = [enqueue, withdraw, collect, cancel, submit, poll, cancel_job]

    # --- Workflow side ---

    async def enqueue(self, params: ModelRequestParams) -> BatchTicket:
        _inbox()
        wire = EnqueueParams(
            item_id=workflow.uuid4().hex,
            model_id=params.model_id,
            messages=params.messages,
            model_settings=cast(dict[str, Any] | None, params.model_settings),
            model_request_parameters=params.model_request_parameters,
        )
        _inbox().pending.add(wire.item_id)
        try:
            return await self._execute(self._enqueue, wire, bounded_enqueue=True)
        except BaseException as error:
            if isinstance(error, (Exception, asyncio.CancelledError)):
                _inbox().pending.discard(wire.item_id)
                _inbox().outcomes.pop(wire.item_id, None)
                await _shielded(self._execute(self._withdraw, wire, bounded_enqueue=True))
            raise

    async def wait(self, ticket: BatchTicket, params: ModelRequestParams) -> ModelResponse | None:
        inbox = _inbox()
        wait = asyncio.create_task(
            workflow.wait_condition(lambda: ticket.item_id in inbox.outcomes, timeout=_WAIT_ROUND)
        )
        try:
            with suppress(asyncio.TimeoutError):
                await asyncio.shield(wait)
        except asyncio.CancelledError:
            wait.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait([wait])
            raise
        response = await self._execute(
            self._collect,
            CollectParams(
                ticket=ticket,
                outcome=inbox.outcomes.get(ticket.item_id),
                model_id=params.model_id,
                model_settings=cast(dict[str, Any] | None, params.model_settings),
                model_request_parameters=params.model_request_parameters,
            ),
        )
        if response is not None:
            inbox.pending.discard(ticket.item_id)
            inbox.outcomes.pop(ticket.item_id, None)
        return response

    async def cancel(self, ticket: BatchTicket) -> None:
        inbox = _inbox()
        inbox.pending.discard(ticket.item_id)
        inbox.outcomes.pop(ticket.item_id, None)
        await _shielded(self._execute(self._cancel, ticket))

    async def _execute(
        self, registration: Callable[..., Awaitable[_Result]], arg: object, *, bounded_enqueue: bool = False
    ) -> _Result:
        config = self._activity_config.copy()
        if bounded_enqueue:
            timeout = config.get('schedule_to_close_timeout') or _ENQUEUE_TIMEOUT
            config['schedule_to_close_timeout'] = min(timeout, _ENQUEUE_TIMEOUT)
        with rebuilt_model_errors():
            return cast(_Result, await execute_activity(activity=registration, args=[arg], **config))

    # --- Activities ---

    def _policy(self, model_id: str | None, model: BatchModel, group: str) -> QueuePolicy:
        settings = model.batch_settings
        return QueuePolicy(
            agent_name=self._agent_name,
            group=group,
            model_id=model_id,
            max_requests=min(settings.max_requests, model.batch_provider.max_requests),
            max_wait=settings.max_wait,
            poll_interval=settings.poll_interval,
        )

    async def _enqueue_activity(self, params: EnqueueParams) -> BatchTicket:
        model = self._resolve_model(params.model_id)
        request = await model.batch_provider.encode(
            params.messages,
            cast(ModelSettings | None, params.model_settings),
            params.model_request_parameters,
            custom_id=params.item_id,
        )
        size = len(to_json(request))
        if size > MAX_REQUEST_BYTES:
            raise UserError(
                f'This `BatchModel` request is {size} bytes, over the {MAX_REQUEST_BYTES}-byte limit of a Temporal '
                'batch queue, which holds requests in its workflow history. Pass large files by reference, such as '
                'an uploaded file, rather than inline.'
            )
        policy = self._policy(params.model_id, model, request.group)
        queue_id = _queue_id(policy, request.group)
        info = activity.info()
        assert info.workflow_id is not None and info.workflow_run_id is not None, 'run by an agent workflow'
        consumer = Consumer(item_id=params.item_id, workflow_id=info.workflow_id, run_id=info.workflow_run_id)
        try:
            accepted_at = await activity.client().execute_update_with_start_workflow(  # pyright: ignore[reportUnknownMemberType]
                BatchQueueWorkflow.enqueue,
                EnqueueRequest(consumer=consumer, request=request, size=size, policy=policy),
                start_workflow_operation=WithStartWorkflowOperation(
                    BatchQueueWorkflow.run,
                    QueueState(policy=policy),
                    id=queue_id,
                    task_queue=info.task_queue,
                    id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
                ),
            )
        except WorkflowUpdateFailedError as error:
            cause = error.cause
            if isinstance(cause, ApplicationError) and cause.non_retryable:
                raise UserError(cause.message) from error
            raise
        return BatchTicket(item_id=params.item_id, queue_id=queue_id, group=request.group, accepted_at=accepted_at)

    async def _withdraw_activity(self, params: EnqueueParams) -> None:
        model = self._resolve_model(params.model_id)
        # Encoded again only to find the queue, which the request's group decides.
        request = await model.batch_provider.encode(
            params.messages,
            cast(ModelSettings | None, params.model_settings),
            params.model_request_parameters,
            custom_id=params.item_id,
        )
        policy = self._policy(params.model_id, model, request.group)
        await activity.client().execute_update_with_start_workflow(  # pyright: ignore[reportUnknownMemberType]
            BatchQueueWorkflow.cancel,
            params.item_id,
            start_workflow_operation=WithStartWorkflowOperation(
                BatchQueueWorkflow.run,
                QueueState(policy=policy),
                id=_queue_id(policy, request.group),
                task_queue=activity.info().task_queue,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
            ),
        )

    async def _cancel_activity(self, ticket: BatchTicket) -> None:
        handle = activity.client().get_workflow_handle_for(BatchQueueWorkflow.run, ticket.queue_id)
        try:
            await handle.execute_update(BatchQueueWorkflow.cancel, ticket.item_id)  # pyright: ignore[reportUnknownMemberType]
        except RPCError as error:
            # A queue that's gone finished with the request long ago.
            if error.status != RPCStatusCode.NOT_FOUND:
                raise

    async def _collect_activity(self, params: CollectParams) -> ModelResponse | None:
        model = self._resolve_model(params.model_id)
        provider = model.batch_provider
        outcome = params.outcome or await self._recover_outcome(params.ticket, provider)
        if outcome is None:
            return None
        if outcome.result is None and outcome.error is None and outcome.batch_id is not None:
            results = await provider.results(outcome.batch_id, group=params.ticket.group) or []
            result = next((item for item in results if item.custom_id == params.ticket.item_id), None)
            outcome = replace(outcome, result=result)
        # A request's own error reproduces on every attempt; the run's model retry handling decides what's next.
        with model_errors_as_application_errors(non_retryable=True):
            if outcome.result is None or outcome.batch_id is None:
                raise ModelAPIError(model.model_name, outcome.error or 'The batch request has no result.')
            response = provider.decode(
                outcome.result, cast(ModelSettings | None, params.model_settings), params.model_request_parameters
            )
        batch_details = (response.provider_details or {}).get('batch')
        return replace(
            response,
            provider_details={
                **(response.provider_details or {}),
                'batch': {
                    **(batch_details if is_str_dict(batch_details) else {}),
                    'state': 'completed',
                    'item_id': params.ticket.item_id,
                    'batch_id': outcome.batch_id,
                },
            },
        )

    async def _recover_outcome(self, ticket: BatchTicket, provider: BatchProvider) -> Outcome | None:
        """Ask the queue what became of a request whose result hasn't been signalled, or `None` if it's still pending."""
        handle = activity.client().get_workflow_handle_for(BatchQueueWorkflow.run, ticket.queue_id)
        try:
            status = await handle.query(BatchQueueWorkflow.status, ticket.item_id)
        except RPCError as error:
            if error.status != RPCStatusCode.NOT_FOUND:
                raise
            status = None
        item_id = ticket.item_id
        if status is None:
            return Outcome(item_id=item_id, error='The batch queue has no record of the request, so it was lost.')
        if status.state in ('queued', 'submitting', 'submitted'):
            return None
        if status.state == 'canceled':
            return Outcome(item_id=item_id, error='The batch request was cancelled.')
        if status.error is not None or status.batch_id is None:
            return Outcome(item_id=item_id, batch_id=status.batch_id, error=status.error)
        # The result was delivered but never reached this workflow, so fetch it again.
        results = await provider.results(status.batch_id, group=ticket.group) or []
        result = next((result for result in results if result.custom_id == item_id), None)
        return Outcome(item_id=item_id, batch_id=status.batch_id, result=result)

    async def _poll_activity(self, params: PollParams) -> PollResult:
        results = await self._resolve_model(params.model_id).batch_provider.results(params.batch_id, group=params.group)
        if results is None:
            return PollResult(ended=False)
        by_id = {result.custom_id: result for result in results}
        client = activity.client()
        missing: list[str] = []
        for consumer in params.consumers:
            if (result := by_id.get(consumer.item_id)) is None:
                missing.append(consumer.item_id)
                continue
            outcome = Outcome(item_id=consumer.item_id, batch_id=params.batch_id, result=result)
            if len(to_json(outcome)) > MAX_REQUEST_BYTES:
                outcome = Outcome(item_id=consumer.item_id, batch_id=params.batch_id)
            handle = client.get_workflow_handle(consumer.workflow_id, run_id=consumer.run_id)
            try:
                await handle.signal(RESULT_SIGNAL, outcome)
            except RPCError as error:
                # The workflow ended without waiting for this result.
                if error.status != RPCStatusCode.NOT_FOUND:
                    raise
        return PollResult(ended=True, missing=missing)
