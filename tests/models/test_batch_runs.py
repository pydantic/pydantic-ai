"""How batched agent runs and `model_request_batch` handle failed requests, scale, limits and event loops.

Uses the in-memory batch API of the test models, scripted to end batches the ways a provider can: a request that
expired, was canceled or is missing from the results, an error per request, or a job that failed as a whole.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, get_args

import pytest
from inline_snapshot import snapshot
from pydantic_core import to_json

from pydantic_ai import (
    Agent,
    AgentRunResult,
    CallToolsNode,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestNode,
    ModelRetry,
    PendingModelResponse,
    TextPart,
    ToolCallPart,
    UsageLimitExceeded,
)
from pydantic_ai.capabilities import AbstractCapability, BatchModelRequests, SelectModel
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.direct import model_request, model_request_batch, model_request_stream
from pydantic_ai.exceptions import (
    AgentRunError,
    ModelAPIError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestTimeout,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    RunCancelled,
    SuspendedResponseExpired,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models._batch_codec import BatchJobError
from pydantic_ai.models._batch_model import BatchedModel, BatchPool
from pydantic_ai.models._in_memory_batch import _State  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import UsageLimits

from ..batch_utils import FAST_BATCH_SETTINGS, FAST_POLLING, UNRUN_MESSAGES, SlowBatchTestModel
from ..continuation_utils import ScriptedContinuationModel, scripted_response

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
        pytest.param('canceled', ModelAPIError, id='canceled'),
        pytest.param(_job_error(499), BatchJobError, id='canceled-job'),
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


async def test_a_canceled_request_is_not_submitted_again() -> None:
    """A canceled request raises, saying its batch was canceled, rather than being submitted again."""
    wrapped = ScriptedBatchModel('canceled')
    model = BatchedModel(wrapped, pool=BatchPool(**FAST_BATCH_SETTINGS))
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    pending = await model.request(messages, None, ModelRequestParameters())
    await model.cancel_suspended_response(pending)
    assert wrapped.canceled == [wrapped.created[0].id]

    with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES['canceled']):
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

    wrapped = ScriptedBatchModel('canceled')
    (canceled,) = await model_request_batch(wrapped, requests, **FAST_POLLING)
    assert isinstance(canceled, ModelAPIError)
    assert str(canceled) == 'The batch was canceled before the model ran the request.'
    assert len(wrapped.created) == 1

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


class FlakyCheckModel(SlowBatchTestModel):
    """Fails the status checks and results downloads of its batches with `errors`, in order, before answering them."""

    def __init__(self, *errors: ModelAPIError, method: str = 'batch_status', pending_checks: int = 0) -> None:
        super().__init__(pending_checks=pending_checks)
        self.errors = list(errors)
        self.method = method
        self.failed = 0

    def _fail(self, method: str) -> None:
        if method == self.method and self.errors:
            self.failed += 1
            raise self.errors.pop(0)

    async def batch_status(self, batch: Batch) -> Batch:
        self._fail('batch_status')
        return await super().batch_status(batch)

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        self._fail('batch_results')
        return await super().batch_results(batch)


def _server_error() -> ModelHTTPError:
    return ModelHTTPError(status_code=502, model_name='test', body='<html>Bad Gateway</html>')


@pytest.mark.parametrize('method', ['batch_status', 'batch_results'])
@pytest.mark.parametrize(
    'make_error',
    [
        pytest.param(_server_error, id='server-error'),
        pytest.param(lambda: ModelHTTPError(status_code=429, model_name='test', body=None), id='rate-limit'),
        pytest.param(lambda: _categorized(ModelOverloadedError, 529), id='overloaded-category'),
        pytest.param(lambda: ModelConnectionError('test', 'Connection reset'), id='connection'),
        pytest.param(lambda: ModelTimeoutError('test', 'Read timed out'), id='timeout'),
    ],
)
async def test_a_check_that_fails_transiently_is_tried_again_at_the_next_poll(
    method: str, make_error: Callable[[], ModelAPIError]
) -> None:
    """A check that fails the way a live request is retried on says nothing about the batch, which keeps the request."""
    wrapped = FlakyCheckModel(make_error(), make_error(), method=method)
    result = await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')

    assert result.output == 'success (no tool calls)'
    assert wrapped.failed == 2
    assert len(wrapped.created) == 1


async def test_a_pending_response_counts_failed_checks_until_a_check_succeeds() -> None:
    wrapped = FlakyCheckModel(_server_error(), _server_error(), pending_checks=3)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    stored = (await agent.run('Hello')).all_messages()

    failed_checks: list[int | None] = []
    for _ in range(3):
        stored = (await agent.run(message_history=stored)).all_messages()
        response = stored[-1]
        assert isinstance(response, ModelResponse) and response.provider_details is not None
        failed_checks.append(response.provider_details['batch'].get('failed_checks'))
    assert failed_checks == [1, 2, None]
    assert len(wrapped.created) == 1


async def test_checks_that_keep_failing_reach_the_run() -> None:
    wrapped = FlakyCheckModel(*(_server_error() for _ in range(10)))

    with pytest.raises(ModelHTTPError, match='status_code: 502'):
        await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert wrapped.failed == 4
    assert len(wrapped.created) == 1


@pytest.mark.parametrize(
    'error',
    [
        pytest.param(ModelHTTPError(status_code=401, model_name='test', body={'error': 'denied'}), id='unauthorized'),
        pytest.param(_categorized(ModelQuotaExceededError, 429), id='quota'),
        pytest.param(ModelAPIError('test', 'Unreadable status'), id='uncategorized'),
    ],
)
async def test_a_check_that_would_fail_again_reaches_the_run(error: ModelAPIError) -> None:
    wrapped = FlakyCheckModel(error)

    with pytest.raises(type(error)) as exc_info:
        await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert exc_info.value is error
    assert wrapped.failed == 1


def _one_request() -> list[BatchRequest]:
    return [BatchRequest(custom_id='one', messages=[ModelRequest.user_text_prompt('Hello')])]


async def test_model_request_batch_tries_a_check_that_fails_transiently_again() -> None:
    """Like `BatchModelRequests`, a transient check failure doesn't fail the call or cancel its batches."""
    wrapped = FlakyCheckModel(
        _server_error(), ModelConnectionError('test', 'Connection reset'), _server_error(), pending_checks=1
    )

    (response,) = await model_request_batch(wrapped, _one_request(), **FAST_POLLING)

    assert isinstance(response, ModelResponse)
    assert wrapped.failed == 3
    assert (len(wrapped.created), wrapped.canceled) == (1, [])


async def test_model_request_batch_raises_after_too_many_failed_checks_in_a_row() -> None:
    wrapped = FlakyCheckModel(*(_server_error() for _ in range(10)), pending_checks=10)

    with pytest.raises(ModelHTTPError, match='status_code: 502'):
        await model_request_batch(wrapped, _one_request(), **FAST_POLLING)
    assert wrapped.failed == 4
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_model_request_batch_raises_a_check_failure_that_would_repeat() -> None:
    error = ModelHTTPError(status_code=401, model_name='test', body={'error': 'denied'})
    wrapped = FlakyCheckModel(error, pending_checks=10)

    with pytest.raises(ModelHTTPError) as exc_info:
        await model_request_batch(wrapped, _one_request(), **FAST_POLLING)
    assert exc_info.value is error
    assert (wrapped.failed, wrapped.canceled) == (1, [wrapped.created[0].id])


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


@pytest.mark.parametrize('stream', [False, True], ids=['run', 'run_stream'])
async def test_resuming_a_pending_batch_request_without_batching_raises(stream: bool) -> None:
    """The model would otherwise send the history as a new, full-price request, and the batch's result would be lost."""
    pending = (
        await Agent(
            SlowBatchTestModel(pending_checks=1),
            output_type=[str, PendingModelResponse],
            capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
        ).run('Hello')
    ).all_messages()
    response = pending[-1]
    assert isinstance(response, ModelResponse) and response.provider_details is not None
    batch_id = response.provider_details['batch']['batch']['id']

    unbatched = TestModel()
    agent = Agent(unbatched)
    with pytest.raises(UserError) as exc_info:
        if stream:
            async with agent.run_stream(message_history=pending):
                pass  # pragma: no cover
        else:
            await agent.run(message_history=pending)
    assert str(exc_info.value) == (
        f'The message history ends in a request waiting for its result from batch {batch_id!r}, which only '
        '`BatchModelRequests` collects: add the capability to the agent or run that resumes it.'
    )
    assert unbatched.last_model_request_parameters is None


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
        vars(model)['_in_memory_batch_state'] = store
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


async def test_a_batch_waits_up_to_max_wait_for_more_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    max_wait = timedelta(hours=1)
    loop = asyncio.get_running_loop()
    call_later = loop.call_later
    deadlines: list[Callable[[], object]] = []

    def record_deadline(
        delay: float, callback: Callable[..., object], *args: object, **kwargs: Any
    ) -> asyncio.TimerHandle:
        # The `max_wait` deadline is fired by the test below instead of the clock.
        if delay == max_wait.total_seconds():
            deadlines.append(callback)
        return call_later(delay, callback, *args, **kwargs)

    monkeypatch.setattr(loop, 'call_later', record_deadline)
    wrapped = SlowBatchTestModel()
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_wait=max_wait)])

    first = asyncio.create_task(agent.run('one'))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    second = asyncio.create_task(agent.run('two'))
    await wrapped.wait_until(lambda m: m.encoded == 2)
    # The first request's batch is still waiting, on one deadline for both requests.
    assert wrapped.created == []
    assert len(deadlines) == 1

    deadlines[0]()
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

    encoded = EncodedBatchRequest(custom_id='', group='group', body='same')

    async def queue(count: int) -> float:
        model = FastModel()
        pool = BatchPool(max_requests=count, max_wait=None)
        # Collections of the test process's whole heap would add pauses that have nothing to do with queueing.
        gc.disable()
        try:
            start = time.perf_counter()
            await asyncio.gather(*(pool.enqueue(model, encoded, None) for _ in range(count)))
            elapsed = time.perf_counter() - start
        finally:
            gc.enable()
        assert len(set(model.custom_ids)) == count
        return elapsed

    # Eight times the requests take about eight times as long in linear time, and 64 times as long in quadratic time.
    # Each round times both sizes back to back, so a busy machine slows both, and the best round is kept.
    ratios = [await queue(8_000) / await queue(1_000) for _ in range(5)]
    assert min(ratios) < 24


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


async def test_the_pool_remembers_a_bounded_number_of_statuses(
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


@pytest.mark.parametrize('batching_first', [False, True], ids=['select-model-first', 'batching-first'])
async def test_a_run_switching_models_batches_each_request_with_the_model_serving_it(batching_first: bool) -> None:
    first, second = SlowBatchTestModel(), SlowBatchTestModel()
    selection = SelectModel(lambda ctx: first if ctx.run_step <= 1 else second)
    batching = BatchModelRequests(**FAST_BATCH_SETTINGS)
    agent = Agent(first, capabilities=[batching, selection] if batching_first else [selection, batching])

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    result = await agent.run('Where?')
    assert result.output == snapshot('{"get_city":"Paris"}')
    assert (len(first.created), len(second.created)) == (1, 1)


async def test_a_model_a_capability_selects_is_batched() -> None:
    @dataclass
    class PickModel(AbstractCapability[object]):
        model: Model

        def get_model(self) -> Model:
            return self.model

    agent_model, picked = SlowBatchTestModel(), SlowBatchTestModel()
    agent = Agent(agent_model, capabilities=[PickModel(picked), BatchModelRequests(**FAST_BATCH_SETTINGS)])
    await agent.run('Hello')
    assert (len(agent_model.created), len(picked.created)) == (0, 1)


def test_a_fallback_model_is_not_batched() -> None:
    agent = Agent(
        FallbackModel(SlowBatchTestModel(), SlowBatchTestModel()),
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    with pytest.raises(UserError, match=r'`FallbackModel` .* has no batch API\.'):
        agent.run_sync('Hello')


def test_a_model_used_from_several_threads_at_once_has_one_batch_store_and_identity() -> None:
    """The first use of a model's in-memory batches, or of its batch identity, can happen on several threads at once."""
    for _ in range(50):
        model = TestModel()
        barrier = threading.Barrier(4)
        seen: list[tuple[int, str]] = []

        def first_use() -> None:
            barrier.wait()
            seen.append((id(model._batch_state), model._batch_credential()))  # pyright: ignore[reportPrivateUsage]

        threads = [threading.Thread(target=first_use) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(set(seen)) == 1


class _HangingStatus(SlowBatchTestModel):
    """Never answers a status check, like a slow provider."""

    async def batch_status(self, batch: Batch) -> Batch:
        self.status_checks += 1
        self.notify_waiters()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')  # pragma: no cover


@pytest.mark.parametrize('stop', ['cancel', 'request_timeout'])
async def test_a_batch_left_for_a_later_run_is_never_cancelled_by_that_run(stop: str) -> None:
    """A run that ends pending leaves its batch for a later run, which can't tell it apart from a stored history's."""
    wrapped = _HangingStatus(pending_checks=1)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    stored = (await agent.run('Hello')).all_messages()

    collecting = asyncio.create_task(
        agent.run(
            message_history=stored, model_settings={'request_timeout': 0.05} if stop == 'request_timeout' else None
        )
    )
    if stop == 'cancel':
        await wrapped.wait_until(lambda model: model.status_checks >= 1)
        collecting.cancel()
    with pytest.raises((asyncio.CancelledError, ModelRequestTimeout)):
        await collecting
    await asyncio.sleep(0.01)

    assert wrapped.canceled == []


async def test_a_run_waiting_in_process_still_cancels_its_batch_when_cancelled() -> None:
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    run = asyncio.create_task(agent.run('Hello'))
    await wrapped.wait_until(lambda model: model.status_checks >= 1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert wrapped.canceled == [wrapped.created[0].id]


@dataclass
class _RejectPendingOnce(AbstractCapability[object]):
    """Rejects the first pending response it sees, the way a response handler or a fallback would."""

    reject: Literal['retry', 'replace', 'mutate', 'strip', 'raise']
    rejected: int = 0

    async def after_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if response.state != 'suspended' or self.rejected:
            return response
        self.rejected += 1
        if self.reject == 'retry':
            raise ModelRetry('Not this one.')
        if self.reject == 'replace':
            return ModelResponse(parts=[TextPart('replaced')])
        if self.reject == 'mutate':
            response.parts = [TextPart('replaced')]
            response.state = 'complete'
            response.provider_details = None
            return response
        if self.reject == 'strip':
            # Still pending, but without what a later run would resume it from.
            return replace(response, provider_details=None)
        raise RuntimeError('rejected')


@pytest.mark.parametrize('stream', [False, True], ids=['run', 'run_stream'])
@pytest.mark.parametrize('reject', ['retry', 'replace', 'mutate', 'strip', 'raise'])
async def test_a_pending_response_a_hook_rejects_is_withdrawn_from_its_batch(
    reject: Literal['retry', 'replace', 'mutate', 'strip', 'raise'], stream: bool
) -> None:
    """Nothing collects a request whose pending response a hook rejected, so its batch is cancelled."""
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[_RejectPendingOnce(reject), BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    output: object
    try:
        if stream:
            async with agent.run_stream('Hello') as result:
                output = await result.get_output()
        else:
            output = (await agent.run('Hello')).output
    except RuntimeError as error:
        output = error

    if reject == 'raise':
        assert isinstance(output, RuntimeError)
        assert len(wrapped.created) == 1
    elif reject == 'strip':
        assert isinstance(output, PendingModelResponse)
        assert len(wrapped.created) == 1
    elif reject == 'retry':
        # The retry is submitted in a new batch, which the run ends pending on and leaves running.
        assert isinstance(output, PendingModelResponse)
        assert len(wrapped.created) == 2
    else:
        assert output == 'replaced'
        assert len(wrapped.created) == 1
    assert wrapped.canceled == [wrapped.created[0].id]


class _FailingExit(WrapperModel):
    async def __aexit__(self, *args: object) -> bool | None:
        await super().__aexit__(*args)  # pyright: ignore[reportArgumentType]
        raise RuntimeError('teardown failed')


@dataclass
class _WrapWithFailingExit(AbstractCapability[object]):
    def get_wrapper_model(self, model: Model) -> Model | None:
        return _FailingExit(model)


@pytest.mark.parametrize('stream', [False, True], ids=['run', 'run_stream'])
async def test_a_pending_response_is_withdrawn_when_exiting_a_wrapper_around_it_fails(stream: bool) -> None:
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[_WrapWithFailingExit(), BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    with pytest.raises(RuntimeError, match='teardown failed'):
        if stream:
            # The stream fails before it's handed over, as the wrapper is exited before the run ends.
            async with agent.run_stream('Hello'):
                raise AssertionError('unreachable')  # pragma: no cover
        else:
            await agent.run('Hello')
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_pending_response_a_hook_rejects_leaves_a_shared_batch_running_for_the_others() -> None:
    wrapped = SlowBatchTestModel(pending_checks=2)
    batching = BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=timedelta(seconds=5))
    rejecting = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[_RejectPendingOnce('raise'), batching]
    )
    waiting = Agent(wrapped, capabilities=[batching])

    rejected, waited = await asyncio.gather(rejecting.run('Hello'), waiting.run('Hello'), return_exceptions=True)

    assert isinstance(rejected, RuntimeError)
    assert not isinstance(waited, BaseException) and waited.output == 'success (no tool calls)'
    assert len(wrapped.created) == 1
    assert wrapped.canceled == []


class _PolledScriptedModel(ScriptedContinuationModel):
    """Polls its suspended responses, as OpenAI background mode does."""

    def continuation_delay(self, response: ModelResponse) -> float | None:
        return 0.0 if response.state == 'suspended' else None


async def test_another_models_pending_response_a_hook_rejects_keeps_running() -> None:
    """Only a batch request is withdrawn when a hook rejects its pending response; other jobs are left as before."""
    model = _PolledScriptedModel(
        responses=[
            scripted_response(texts=[], state='suspended', provider_response_id=job, input_tokens=1, output_tokens=0)
            for job in ('job1', 'job2')
        ]
    )
    agent = Agent(model, output_type=[str, PendingModelResponse], capabilities=[_RejectPendingOnce('retry')])

    result = await agent.run('Hello')

    assert isinstance(result.output, PendingModelResponse)
    assert result.output.response.provider_response_id == 'job2'
    assert model.cancelled == []


async def test_waiting_on_a_batch_is_not_limited_to_a_number_of_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch ends within the provider's batch window, so a run waits however many checks that takes."""
    monkeypatch.setattr('pydantic_ai._agent_graph.MAX_BACKGROUND_POLLS', 2)
    wrapped = SlowBatchTestModel(pending_checks=5)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    result = await agent.run('Hello')
    assert result.output == 'success (no tool calls)'
    assert wrapped.status_checks == 6


@pytest.mark.parametrize('stream', [False, True], ids=['model_request', 'model_request_stream'])
async def test_a_direct_request_resuming_a_pending_batch_request_without_batching_raises(stream: bool) -> None:
    pending = (
        await Agent(
            SlowBatchTestModel(pending_checks=1),
            output_type=[str, PendingModelResponse],
            capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
        ).run('Hello')
    ).all_messages()
    unbatched = TestModel()
    with pytest.raises(UserError, match='which only `BatchModelRequests` collects'):
        if stream:
            async with model_request_stream(unbatched, pending):
                pass  # pragma: no cover
        else:
            await model_request(unbatched, pending)
    assert unbatched.last_model_request_parameters is None


async def test_model_request_batch_refuses_messages_waiting_on_a_batch() -> None:
    """A new batch request would send the waiting request again, at a second price."""
    pending = (
        await Agent(
            SlowBatchTestModel(pending_checks=1),
            output_type=[str, PendingModelResponse],
            capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)],
        ).run('Hello')
    ).all_messages()
    model = SlowBatchTestModel()
    with pytest.raises(UserError, match='which a new batch request would send again'):
        await model_request_batch(model, [BatchRequest(custom_id='one', messages=pending)], **FAST_POLLING)
    assert model.created == []


class _CountingResults(SlowBatchTestModel):
    def __init__(self, pending_checks: int = 0) -> None:
        super().__init__(pending_checks)
        self.downloads = 0

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        self.downloads += 1
        return await super().batch_results(batch)


async def test_a_batchs_results_are_dropped_once_every_run_waiting_on_it_has_read_them() -> None:
    wrapped = _CountingResults()
    batching = BatchModelRequests(**FAST_BATCH_SETTINGS)
    agent = Agent(wrapped, capabilities=[batching])
    await asyncio.gather(*(agent.run(f'Question {index}') for index in range(5)))

    assert len(wrapped.created) == 1
    assert wrapped.downloads == 1
    assert batching._pool._results == {}  # pyright: ignore[reportPrivateUsage]


async def test_the_results_of_batches_nobody_here_waits_on_are_bounded_by_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai.models._batch_model._RESULTS_CACHE_BYTES', 1)
    wrapped = _CountingResults()
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    stored = [(await agent.run(f'Question {index}')).all_messages() for index in range(2)]

    collecting = BatchModelRequests(**FAST_BATCH_SETTINGS)
    collector = Agent(wrapped, output_type=[str, PendingModelResponse], capabilities=[collecting])
    for history in stored:
        assert (await collector.run(message_history=history)).output == 'success (no tool calls)'
    # Each batch's results were too large to keep, so collecting the first again downloads them again.
    assert collecting._pool._results == {}  # pyright: ignore[reportPrivateUsage]
    assert (await collector.run(message_history=stored[0])).output == 'success (no tool calls)'
    assert wrapped.downloads == 3


async def test_an_error_one_in_memory_batch_request_raises_fails_only_its_run() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if messages[-1].parts[-1].content == 'bad':  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
            raise ValueError('Not this one')
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(respond), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    good, bad = await asyncio.gather(agent.run('good'), agent.run('bad'), return_exceptions=True)
    assert not isinstance(good, BaseException) and good.output == 'ok'
    assert isinstance(bad, ValueError) and str(bad) == 'Not this one'


def test_exiting_on_one_event_loop_closes_only_that_loops_batching() -> None:
    """Each event loop counts its own `async with` blocks, so exiting one doesn't wait on another loop's block."""
    batching = BatchModelRequests(max_wait=None, **FAST_POLLING)
    wrapped = SlowBatchTestModel()
    agent = Agent(wrapped, capabilities=[batching])
    entered = threading.Event()
    done = threading.Event()

    async def hold_open() -> None:
        async with batching:
            entered.set()
            while not done.is_set():
                await asyncio.sleep(0.001)

    async def run_and_exit() -> None:
        async with batching:
            run = asyncio.create_task(agent.run('Hello'))
            await wrapped.wait_until(lambda model: model.encoded == 1)
        with pytest.raises(UserError, match='exited while the request was waiting'):
            await run

    holder = threading.Thread(target=lambda: asyncio.run(hold_open()))
    holder.start()
    try:
        assert entered.wait(10)
        asyncio.run(run_and_exit())
    finally:
        done.set()
        holder.join()


EndPendingRun = Literal[
    'ends_pending',
    'after_run_raises',
    'after_run_replaces',
    'wrap_run_raises',
    'on_run_error_raises',
    'on_run_error_recovers',
    'cancelled',
    'ctx_cancel',
]


@dataclass
class _EndPendingRun(AbstractCapability[object]):
    """Ends a run whose model request step accepted a pending response, as `end` says."""

    end: EndPendingRun
    reached: asyncio.Event = field(default_factory=asyncio.Event)

    async def after_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if self.end == 'ctx_cancel':
            ctx.cancel()
        return response

    async def after_node_run(
        self, ctx: RunContext[object], *, node: AgentNode[object], result: NodeResult[object]
    ) -> NodeResult[object]:
        if isinstance(node, CallToolsNode):
            if self.end in ('on_run_error_raises', 'on_run_error_recovers'):
                raise RuntimeError('the run body failed')
            if self.end == 'cancelled':
                self.reached.set()
                await asyncio.Event().wait()
        return result

    async def on_run_error(self, ctx: RunContext[object], *, error: BaseException) -> AgentRunResult[Any]:
        if self.end == 'on_run_error_recovers':
            return AgentRunResult('recovered')
        if self.end == 'on_run_error_raises':
            raise RuntimeError('on_run_error failed') from error
        raise error

    async def after_run(self, ctx: RunContext[object], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        if self.end == 'after_run_raises':
            raise RuntimeError('after_run failed')
        if self.end == 'after_run_replaces':
            return replace(result, output='replaced')
        return result

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        result = await handler()
        if self.end == 'wrap_run_raises':
            raise RuntimeError('wrap_run failed')
        return result


_ENDED_WITH: dict[EndPendingRun, object] = {
    'after_run_raises': RuntimeError('after_run failed'),
    'after_run_replaces': 'replaced',
    'wrap_run_raises': RuntimeError('wrap_run failed'),
    'on_run_error_raises': RuntimeError('on_run_error failed'),
    'on_run_error_recovers': 'recovered',
    'cancelled': asyncio.CancelledError(),
    'ctx_cancel': RunCancelled('The agent run was cancelled.'),
}
"""What each way of ending a run other than with its pending response ends `agent.run()` with."""

_STREAM_BLOCK_ENDED_WITH: dict[EndPendingRun, object] = {
    # The block gets the stream's output, while `after_run` replaces the run's once the block has exited.
    'after_run_replaces': PendingModelResponse,
    # The block can't get the output `on_run_error` recovers the run with, as the run had none when it failed.
    'on_run_error_recovers': AgentRunError('Agent run finished without producing a final result'),
}
"""Where the `run_stream()` block ends differently from `agent.run()`."""


async def _assert_left_for_a_later_run(
    agent: Agent[None, Any], wrapped: SlowBatchTestModel, stored: list[ModelMessage]
) -> None:
    """Resume `stored` and cancel the resuming run, which only leaves the batch running if the first run released it."""
    checks = wrapped.status_checks
    resuming = asyncio.create_task(agent.run(message_history=stored))
    await wrapped.wait_until(lambda model: model.status_checks > checks)
    resuming.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resuming
    await asyncio.sleep(0.01)
    assert wrapped.canceled == []


@pytest.mark.parametrize('stream', [False, True], ids=['run', 'run_stream'])
@pytest.mark.parametrize('end', get_args(EndPendingRun))
async def test_a_pending_response_is_withdrawn_unless_the_run_ends_with_it(end: EndPendingRun, stream: bool) -> None:
    """A run's model request step accepting a pending response doesn't leave it for a later run if the run then fails."""
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    ending = _EndPendingRun(end)
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[ending, BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    stored: list[ModelMessage] = []

    async def run() -> object:
        if stream:
            async with agent.run_stream('Hello') as result:
                output = await result.get_output()
                stored.extend(result.all_messages())
                return output
        result = await agent.run('Hello')
        stored.extend(result.all_messages())
        return result.output

    task = asyncio.create_task(run())
    if end == 'cancelled':
        await ending.reached.wait()
        task.cancel()
    try:
        output = await task
    except (Exception, asyncio.CancelledError) as error:
        output = error

    assert len(wrapped.created) == 1
    if end == 'ends_pending':
        assert isinstance(output, PendingModelResponse)
        assert wrapped.canceled == []
        await _assert_left_for_a_later_run(agent, wrapped, stored)
        return

    expected = (_STREAM_BLOCK_ENDED_WITH.get(end) if stream else None) or _ENDED_WITH[end]
    if expected is PendingModelResponse:
        assert isinstance(output, PendingModelResponse)
    elif isinstance(expected, BaseException):
        assert type(output) is type(expected) and str(output) == str(expected)
    else:
        assert output == expected
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_pending_response_is_withdrawn_when_the_caller_stops_iterating_before_the_run_ends() -> None:
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    async with agent.iter('Hello') as agent_run:
        async for node in agent_run:
            if isinstance(node, CallToolsNode):
                break
    assert agent_run.result is None
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_pending_response_is_left_for_a_later_run_when_an_iterated_run_ends_with_it() -> None:
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    async with agent.iter('Hello') as agent_run:
        async for _ in agent_run:
            pass
    assert agent_run.result is not None and isinstance(agent_run.result.output, PendingModelResponse)
    assert wrapped.canceled == []
    await _assert_left_for_a_later_run(agent, wrapped, agent_run.result.all_messages())


@dataclass
class _AskAgainOnce(AbstractCapability[object]):
    """Sends the run back to the model once it would have ended, as a hook redirecting the graph can."""

    asked: bool = False

    async def after_node_run(
        self, ctx: RunContext[object], *, node: AgentNode[object], result: NodeResult[object]
    ) -> NodeResult[object]:
        if isinstance(node, CallToolsNode) and not self.asked:
            self.asked = True
            return ModelRequestNode(request=ModelRequest.user_text_prompt('Again'))
        return result


async def test_a_pending_response_a_later_request_supersedes_is_withdrawn() -> None:
    wrapped = SlowBatchTestModel(pending_checks=10**6)
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[_AskAgainOnce(), BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    result = await agent.run('Hello')
    assert isinstance(result.output, PendingModelResponse)
    assert len(wrapped.created) == 2
    assert wrapped.canceled == [wrapped.created[0].id]


class _ClosingBatchModel(SlowBatchTestModel):
    """Records whether it had been exited when it was asked to cancel a batch, as a closed HTTP client couldn't."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.closed = False
        self.closed_when_canceling: list[bool] = []

    async def __aexit__(self, *args: Any) -> bool | None:
        self.closed = True
        return await super().__aexit__(*args)

    async def batch_cancel(self, batch: Batch) -> Batch:
        self.closed_when_canceling.append(self.closed)
        return await super().batch_cancel(batch)


async def test_a_pending_response_of_a_model_selected_later_in_the_run_is_withdrawn_before_the_model_is_exited() -> (
    None
):
    first, second = SlowBatchTestModel(pending_checks=10**6), _ClosingBatchModel(pending_checks=10**6)
    agent = Agent(
        first,
        output_type=[str, PendingModelResponse],
        capabilities=[
            SelectModel(lambda ctx: first if ctx.run_step <= 1 else second),
            # The first step's pending response is retried, so the second step's request is the one the run accepts.
            _RejectPendingOnce('retry'),
            _EndPendingRun('after_run_raises'),
            BatchModelRequests(**FAST_BATCH_SETTINGS),
        ],
    )
    with pytest.raises(RuntimeError, match='after_run failed'):
        await agent.run('Hello')
    assert first.canceled == [first.created[0].id]
    assert second.closed
    assert second.canceled == [second.created[0].id]
    assert second.closed_when_canceling == [False]
