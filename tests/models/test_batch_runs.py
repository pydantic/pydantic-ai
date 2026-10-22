"""How batched agent runs and `model_request_batch` handle failed requests, scale, limits and event loops.

Uses the in-memory batch API of the test models, scripted to end batches the ways a provider can: a request that
expired, was canceled or is missing from the results, an error per request, or a job that failed as a whole.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from pydantic_core import to_json

from pydantic_ai import (
    Agent,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRetry,
    PendingModelResponse,
    TextPart,
    ToolCallPart,
    UsageLimitExceeded,
)
from pydantic_ai.capabilities import BatchModelRequests, SelectModel
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
    SuspendedResponseExpired,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models._batch_codec import BatchJobError
from pydantic_ai.models._batch_model import BatchedModel, BatchPool
from pydantic_ai.models._in_memory_batch import _State  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from ..batch_utils import FAST_BATCH_SETTINGS, FAST_POLLING, SlowBatchTestModel

Outcome = Literal['ran', 'expired', 'canceled', 'missing'] | int | ModelAPIError
"""How a scripted batch ends each of its requests: as they ran, unrun, missing, with an HTTP status, with an error, or
as a failed job (a `BatchJobError`)."""


class ScriptedBatchModel(SlowBatchTestModel):
    """Ends the batches it creates, in order, with `outcomes`, and then as their requests ran."""

    def __init__(self, *outcomes: Outcome, pending_checks: int = 0) -> None:
        super().__init__(pending_checks=pending_checks)
        self.outcomes = list(outcomes)
        self.ended: dict[str, Outcome] = {}
        self._errors: dict[str, ModelAPIError] = {}

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        batch = await super().batch_create(requests)
        self.ended[batch.id] = outcome = self.outcomes.pop(0) if self.outcomes else 'ran'
        if isinstance(outcome, ModelAPIError) and not isinstance(outcome, BatchJobError):
            self._errors.update(dict.fromkeys((request.custom_id for request in requests), outcome))
        return batch

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        results = await super().batch_results(batch)
        outcome = self.ended[batch.id]
        if isinstance(outcome, BatchJobError):
            raise outcome
        if isinstance(outcome, int | ModelAPIError):
            status_code = outcome if isinstance(outcome, int) else 0
            body = to_json({'status_code': status_code, 'body': {'error': 'scripted'}}).decode()
            return [BatchResult(custom_id=result.custom_id, status='errored', body=body) for result in results]
        if outcome == 'missing':
            return []
        if outcome == 'ran':
            return results
        return [BatchResult(custom_id=result.custom_id, status=outcome) for result in results]

    def batch_decode(self, result: BatchResult, request: BatchRequest) -> ModelResponse:
        if result.status == 'errored' and (error := self._errors.get(result.custom_id)) is not None:
            raise error
        return super().batch_decode(result, request)


def _categorized(
    category: type[
        ModelRateLimitError
        | ModelQuotaExceededError
        | ModelUnavailableError
        | ModelServerError
        | ModelContextWindowExceededError
    ],
    status_code: int,
) -> ModelHTTPError:
    return ModelHTTPError.for_category(category, status_code=status_code, model_name='test', body={'error': 'scripted'})


def _job_error(status_code: int) -> BatchJobError:
    return BatchJobError(status_code=status_code, model_name='test', body={'error': 'job failed'})


@pytest.mark.parametrize(
    'outcome',
    [
        pytest.param('expired', id='expired'),
        pytest.param('canceled', id='canceled-elsewhere'),
        pytest.param('missing', id='missing'),
        pytest.param(408, id='timeout'),
        pytest.param(409, id='conflict'),
        pytest.param(429, id='rate-limit'),
        pytest.param(500, id='server-error'),
        pytest.param(529, id='overloaded'),
        pytest.param(_job_error(500), id='failed-job'),
        pytest.param(_categorized(ModelRateLimitError, 429), id='rate-limit-category'),
        pytest.param(_categorized(ModelOverloadedError, 529), id='overloaded-category'),
        pytest.param(_categorized(ModelUnavailableError, 503), id='unavailable-category'),
        pytest.param(_categorized(ModelServerError, 500), id='server-error-category'),
        pytest.param(ModelRateLimitError('test', 'Slow down'), id='rate-limit-without-status'),
        pytest.param(_job_error(499), id='canceled-job'),
    ],
)
async def test_a_transient_failure_is_submitted_again(outcome: Outcome) -> None:
    wrapped = ScriptedBatchModel(outcome)
    result = await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')

    assert result.output == 'success (no tool calls)'
    assert len(wrapped.created) == 2
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse) and response.provider_details is not None
    assert response.provider_details['batch']['resubmissions'] == 1
    # The run's usage counts the one response it acted on.
    assert result.usage.requests == 1


@pytest.mark.parametrize(
    'outcome,error',
    [
        pytest.param(400, ModelHTTPError, id='invalid-request'),
        pytest.param(401, ModelHTTPError, id='unauthorized'),
        pytest.param(_categorized(ModelQuotaExceededError, 429), ModelQuotaExceededError, id='quota'),
        pytest.param(
            _categorized(ModelContextWindowExceededError, 400), ModelContextWindowExceededError, id='context-window'
        ),
        pytest.param(ModelAPIError('test', 'Unknown'), ModelAPIError, id='uncategorized-without-status'),
        pytest.param(_job_error(400), BatchJobError, id='failed-job'),
    ],
)
async def test_a_request_that_would_fail_again_is_not_submitted_again(
    outcome: Outcome, error: type[ModelAPIError]
) -> None:
    wrapped = ScriptedBatchModel(outcome)
    with pytest.raises(error):
        await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert len(wrapped.created) == 1


async def test_resubmissions_are_bounded() -> None:
    wrapped = ScriptedBatchModel('expired', 'expired', 'expired', 'expired')
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS, max_resubmissions=3)])
    with pytest.raises(ModelAPIError, match='The batch request was expired before the model ran it'):
        await agent.run('Hello')
    assert len(wrapped.created) == 4

    wrapped = ScriptedBatchModel(503)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS, max_resubmissions=0)])
    with pytest.raises(ModelHTTPError, match='status_code: 503'):
        await agent.run('Hello')
    assert len(wrapped.created) == 1


def test_max_resubmissions_is_validated() -> None:
    with pytest.raises(UserError, match='`max_resubmissions` must not be negative'):
        BatchModelRequests(max_resubmissions=-1)


async def test_a_batch_this_pool_cancelled_is_not_submitted_again() -> None:
    """Canceled requests are resubmitted when someone else cancelled the batch, not when the pool did."""
    wrapped = ScriptedBatchModel('canceled')
    model = BatchedModel(wrapped, pool=BatchPool(**FAST_BATCH_SETTINGS))
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    pending = await model.request(messages, None, ModelRequestParameters())
    await model.cancel_suspended_response(pending)
    assert wrapped.canceled == [wrapped.created[0].id]

    with pytest.raises(ModelAPIError, match='was canceled before the model ran it'):
        await model.request([*messages, pending], None, ModelRequestParameters())
    assert len(wrapped.created) == 1


async def test_a_resubmission_resends_the_content_a_continuation_generated() -> None:
    """A paused turn's content was part of the request, so the resubmitted request carries it too."""
    calls: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart('The answer ')], state='suspended', provider_response_id='seg1')
        last = messages[-1]
        # The same continuation as first sent.
        assert isinstance(last, ModelResponse) and last.text == 'The answer '
        assert (last.state, last.provider_response_id) == ('suspended', 'seg1')
        return ModelResponse(parts=[TextPart('is 42.')], provider_response_id='seg2')

    class ExpiringFunctionModel(FunctionModel):
        expire_next = 1

        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            results = await super().batch_results(batch)
            if len(calls) == 2 and self.expire_next:
                # The continuation's first batch expires; its result is dropped.
                self.expire_next = 0
                calls.pop()
                return [BatchResult(custom_id=result.custom_id, status='expired') for result in results]
            return results

    result = await Agent(ExpiringFunctionModel(respond), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run(
        'What is the answer?'
    )

    assert result.output == 'The answer is 42.'
    assert len(calls) == 2


async def test_model_request_batch_submits_transient_failures_again() -> None:
    wrapped = ScriptedBatchModel('expired', 'missing')
    requests = [BatchRequest(custom_id=key, messages=[ModelRequest.user_text_prompt(key)]) for key in ('a', 'b')]

    responses = await model_request_batch(wrapped, requests, **FAST_POLLING)

    assert all(isinstance(response, ModelResponse) for response in responses)
    assert [batch.request_count for batch in wrapped.created] == [2, 2, 2]


async def test_model_request_batch_returns_errors_that_resubmitting_would_not_fix() -> None:
    wrapped = ScriptedBatchModel(400, 'missing', _job_error(503), _job_error(503))
    requests = [BatchRequest(custom_id='a', messages=[ModelRequest.user_text_prompt('a')])]

    (invalid,) = await model_request_batch(wrapped, requests, **FAST_POLLING)
    assert isinstance(invalid, ModelHTTPError) and invalid.status_code == 400

    (failed,) = await model_request_batch(wrapped, requests, max_resubmissions=2, **FAST_POLLING)
    assert isinstance(failed, BatchJobError) and failed.status_code == 503
    assert len(wrapped.created) == 4

    wrapped = ScriptedBatchModel('missing')
    (missing,) = await model_request_batch(wrapped, requests, max_resubmissions=0, **FAST_POLLING)
    assert isinstance(missing, ModelAPIError)
    assert str(missing) == f'Batch {wrapped.created[0].id!r} ended without a result for the request.'


async def test_model_request_batch_validates_max_resubmissions() -> None:
    with pytest.raises(UserError, match='`max_resubmissions` must not be negative'):
        await model_request_batch(TestModel(), [], max_resubmissions=-1)


@pytest.mark.parametrize('method', ['batch_status', 'batch_results'])
async def test_a_batch_the_provider_no_longer_has_is_expired(method: str) -> None:
    class ForgettingModel(SlowBatchTestModel):
        async def batch_status(self, batch: Batch) -> Batch:
            if method == 'batch_status':
                raise ModelHTTPError(status_code=404, model_name='test', body={'error': 'not found'})
            return await super().batch_status(batch)

        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            raise ModelHTTPError(status_code=404, model_name='test', body={'error': 'not found'})

    wrapped = ForgettingModel()
    with pytest.raises(SuspendedResponseExpired) as exc_info:
        await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert isinstance(exc_info.value.__cause__, ModelHTTPError)
    assert len(wrapped.created) == 1


async def test_other_errors_checking_a_batch_reach_the_run() -> None:
    class FailingStatusModel(SlowBatchTestModel):
        async def batch_status(self, batch: Batch) -> Batch:
            raise ModelHTTPError(status_code=500, model_name='test', body={'error': 'boom'})

    with pytest.raises(ModelHTTPError, match='status_code: 500'):
        await Agent(FailingStatusModel(), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')


async def test_a_history_waiting_on_another_providers_batch_is_rejected() -> None:
    pending = (
        await Agent(
            SlowBatchTestModel(pending_checks=1),
            output_type=[str, PendingModelResponse],
            capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
        ).run('Hello')
    ).all_messages()
    response = pending[-1]
    assert isinstance(response, ModelResponse) and response.provider_details is not None
    ticket = response.provider_details['batch']
    other = replace(
        response, provider_details={'batch': {**ticket, 'batch': {**ticket['batch'], 'provider_name': 'anthropic'}}}
    )

    agent = Agent(TestModel(), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    with pytest.raises(UserError, match="waits on a batch submitted to 'anthropic'"):
        await agent.run(message_history=[*pending[:-1], other])


async def test_checks_of_a_batch_within_the_poll_interval_share_its_status() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[BatchModelRequests(max_wait=timedelta(milliseconds=1))],
    )
    stored = (await agent.run('Hello')).all_messages()

    for _ in range(3):
        result = await agent.run(message_history=stored)
        assert isinstance(result.output, PendingModelResponse)
        stored = result.all_messages()
    assert wrapped.status_checks == 1


async def test_a_pending_history_round_trips_through_json_into_a_new_model_instance() -> None:
    """Batches that outlive the model instance, as a provider's do, are collected by a new instance and capability."""
    store = _State(group='shared')

    def agent(pending_checks: int = 0) -> Agent[None, str | PendingModelResponse]:
        model = SlowBatchTestModel(pending_checks=pending_checks)
        vars(model)['_batch_state'] = store
        return Agent(
            model, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
        )

    stored = (await agent(pending_checks=1).run('Hello')).all_messages_json()
    pending = await agent(pending_checks=1).run(message_history=ModelMessagesTypeAdapter.validate_json(stored))
    assert isinstance(pending.output, PendingModelResponse)
    finished = await agent().run(message_history=ModelMessagesTypeAdapter.validate_json(pending.all_messages_json()))
    assert finished.output == 'success (no tool calls)'
    assert len(store.batches) == 1


async def test_an_output_validation_retry_is_another_batch_request() -> None:
    """The validator never sees the pending output, and its retry is submitted in a new batch."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    validated: list[str] = []

    def agent() -> Agent[None, str | PendingModelResponse]:
        # A new capability for each run, as in a new process, so each run checks the batch itself.
        agent = Agent(
            wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
        )

        @agent.output_validator
        def check(output: str | PendingModelResponse) -> str | PendingModelResponse:
            assert isinstance(output, str)
            validated.append(output)
            if len(validated) == 1:
                raise ModelRetry('Try again')
            return output

        return agent

    result = await agent().run('Hello')
    pending_outputs = 0
    while isinstance(result.output, PendingModelResponse):
        pending_outputs += 1
        result = await agent().run(message_history=result.all_messages())

    assert result.output == 'success (no tool calls)'
    assert validated == ['success (no tool calls)'] * 2
    assert len(wrapped.created) == 2
    # Submitted, checked while in progress, then each batch's result arrived at the next check.
    assert (pending_outputs, wrapped.status_checks) == (3, 3)
    assert [type(part).__name__ for part in result.all_messages()[2].parts] == ['RetryFeedbackPart']


async def test_usage_limits_apply_to_batched_runs() -> None:
    wrapped = SlowBatchTestModel()
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    with pytest.raises(UsageLimitExceeded, match='request_limit of 1'):
        await agent.run('Where?', usage_limits=UsageLimits(request_limit=1))
    assert len(wrapped.created) == 1

    result = await agent.run('Where?', usage_limits=UsageLimits(request_limit=2))
    assert result.usage.requests == 2
    assert len(wrapped.created) == 3


async def test_each_run_of_a_pending_cycle_counts_the_response_it_acted_on() -> None:
    """A run ending in, or resuming from, a pending response counts it once; its tokens are only counted once."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )

    runs = [await agent.run('Hello')]
    while isinstance(runs[-1].output, PendingModelResponse):
        runs.append(
            await Agent(
                wrapped,
                output_type=[str, PendingModelResponse],
                capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
            ).run(message_history=runs[-1].all_messages())
        )

    assert [run.usage.requests for run in runs] == [1, 1, 1]
    assert [run.usage.total_tokens for run in runs] == [0, 0, 55]


async def test_a_tool_loop_takes_a_batch_per_model_request() -> None:
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls < 4:
            return ModelResponse(parts=[ToolCallPart('step', {'n': calls})])
        return ModelResponse(parts=[TextPart('done')])

    wrapped = FunctionModel(respond)
    created: list[int] = []
    original = wrapped.batch_create

    async def batch_create(requests: Sequence[EncodedBatchRequest]) -> Batch:
        created.append(len(requests))
        return await original(requests)

    wrapped.batch_create = batch_create
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    @agent.tool_plain
    def step(n: int) -> str:
        return f'did {n}'

    result = await agent.run('Go')
    assert result.output == 'done'
    assert created == [1, 1, 1, 1]


async def test_many_concurrent_runs_fill_batches_of_the_default_size() -> None:
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    assert batching.max_requests == 1_000
    agent = Agent(wrapped, capabilities=[batching])

    runs = [asyncio.create_task(agent.run(f'Question {i}')) for i in range(1_050)]
    await wrapped.wait_until(lambda m: m.encoded == 1_050)
    await batching.flush()
    results = await asyncio.gather(*runs)

    assert {result.output for result in results} == {'success (no tool calls)'}
    assert [batch.request_count for batch in wrapped.created] == [1_000, 50]


async def test_a_batch_waits_up_to_max_wait_for_more_requests() -> None:
    wrapped = SlowBatchTestModel()
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_wait=timedelta(milliseconds=200))])

    first = asyncio.create_task(agent.run('one'))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    second = asyncio.create_task(agent.run('two'))
    await asyncio.gather(first, second)

    # The second request joined the batch of the first, which waited for it rather than leaving straight away.
    assert [batch.request_count for batch in wrapped.created] == [2]


async def test_queueing_many_identical_requests_takes_linear_time() -> None:
    """Every request gets a unique ID in constant time, so a provider-sized batch queues without stalling the loop."""

    class FastModel(TestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return 100_000

        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            self.custom_ids = [request.custom_id for request in requests]
            return Batch(
                id='batch',
                model_name='test',
                provider_name='test',
                group='group',
                status='in_progress',
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                request_count=len(requests),
            )

    model = FastModel()
    pool = BatchPool(max_requests=50_000, max_wait=None)
    encoded = EncodedBatchRequest(custom_id='', group='group', body='same')
    await asyncio.gather(*(pool.enqueue(model, encoded, None) for _ in range(50_000)))

    assert len(set(model.custom_ids)) == 50_000


def test_one_capability_serves_runs_on_several_event_loops_and_threads() -> None:
    """Each event loop has its own queues, so runs on different threads never share, or wait on, each other's batches."""
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_BATCH_SETTINGS)
    agent = Agent(wrapped, capabilities=[batching])
    outputs: list[str] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            outputs.extend(asyncio.run(_gather(agent, 3)))
        except BaseException as error:  # pragma: no cover
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert outputs == ['success (no tool calls)'] * 12
    assert sum(batch.request_count for batch in wrapped.created) == 12
    # Sequential runs on new event loops work too.
    assert agent.run_sync('Hello').output == agent.run_sync('Hello').output == 'success (no tool calls)'


async def _gather(agent: Agent[None, str], runs: int) -> list[str]:
    return [result.output for result in await asyncio.gather(*(agent.run(f'Hello {i}') for i in range(runs)))]


async def test_the_pool_remembers_a_bounded_number_of_statuses_and_cancellations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    pool = BatchPool(
        max_wait=timedelta(milliseconds=1), poll_interval=timedelta(hours=1), max_poll_interval=timedelta(hours=1)
    )
    model = BatchedModel(wrapped, pool=pool)
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    first = await model.request(messages, None, ModelRequestParameters())
    second = await model.request([ModelRequest.user_text_prompt('Bye')], None, ModelRequestParameters())
    monkeypatch.setattr('pydantic_ai.models._batch_model._SUBMITTED_BATCHES_SIZE', 1)

    await model.request([*messages, first], None, ModelRequestParameters())
    await model.request([*messages, second], None, ModelRequestParameters())
    # The first batch's status was forgotten, so checking it again asks the provider.
    await model.request([*messages, first], None, ModelRequestParameters())
    assert wrapped.status_checks == 3

    await model.cancel_suspended_response(first)
    await model.cancel_suspended_response(second)
    assert wrapped.canceled == [wrapped.created[0].id, wrapped.created[1].id]
    assert len(pool._canceled) == 1  # pyright: ignore[reportPrivateUsage]


class SizedBatchModel(SlowBatchTestModel):
    """Limits a batch to 100 bytes, with each request taking 40."""

    @property
    def batch_max_bytes(self) -> int | None:
        return 100

    def batch_request_size(self, request: EncodedBatchRequest) -> int:
        return 40


async def test_a_batch_is_submitted_before_a_request_would_take_it_past_the_byte_limit() -> None:
    wrapped = SizedBatchModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    agent = Agent(wrapped, capabilities=[batching])

    runs = [asyncio.create_task(agent.run(f'Question {i}')) for i in range(5)]
    await wrapped.wait_until(lambda m: m.encoded == 5)
    await batching.flush()
    await asyncio.gather(*runs)

    assert [batch.request_count for batch in wrapped.created] == [2, 2, 1]


async def test_a_withdrawn_request_frees_its_bytes_in_the_batch() -> None:
    wrapped = SizedBatchModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    agent = Agent(wrapped, capabilities=[batching])

    first = asyncio.create_task(agent.run('one'))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    runs = [asyncio.create_task(agent.run(f'Question {i}')) for i in range(2)]
    await wrapped.wait_until(lambda m: m.encoded == 3)
    await batching.flush()
    await asyncio.gather(*runs)

    assert [batch.request_count for batch in wrapped.created] == [2]


async def test_model_request_batch_splits_batches_by_bytes_and_count() -> None:
    wrapped = SizedBatchModel()
    requests = [BatchRequest(custom_id=str(i), messages=[ModelRequest.user_text_prompt(str(i))]) for i in range(5)]

    responses = await model_request_batch(wrapped, requests, **FAST_POLLING)
    assert all(isinstance(response, ModelResponse) for response in responses)
    assert [batch.request_count for batch in wrapped.created] == [2, 2, 1]

    wrapped = SizedBatchModel()
    await model_request_batch(wrapped, requests, max_requests=1, **FAST_POLLING)
    assert [batch.request_count for batch in wrapped.created] == [1] * 5


@dataclass(init=False)
class _NoBatchModel(TestModel):
    @property
    def batch_max_requests(self) -> int | None:
        return None


async def test_a_model_without_a_batch_api_selected_mid_run_raises_at_its_request() -> None:
    """Each request is batched with the model serving it, so switching mid-run to one without a batch API raises then."""
    batched = SlowBatchTestModel()

    def select(ctx: ModelSelectionContext[object]) -> Model:
        return batched if ctx.run_step <= 1 else _NoBatchModel()

    agent = Agent(batched, capabilities=[SelectModel(select), BatchModelRequests(**FAST_BATCH_SETTINGS)])

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    with pytest.raises(UserError, match=r"`_NoBatchModel` 'test' has no batch API\."):
        await agent.run('Where?')
    assert len(batched.created) == 1


def test_a_fallback_model_is_not_batched() -> None:
    agent = Agent(
        FallbackModel(SlowBatchTestModel(), SlowBatchTestModel()),
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    with pytest.raises(UserError, match=r'`FallbackModel` .* has no batch API\.'):
        agent.run_sync('Hello')
