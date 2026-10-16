"""`BatchModel`, `model_request_batch` and the batch contract, using the in-memory batch API of the test models.

Provider codecs and their recorded batches are tested per provider (`test_anthropic_batch*.py`, ...); these tests pin the
provider-independent behavior: grouping requests into batches, waiting through the continuation loop, resuming from
history, and how results and errors reach a run. They aren't VCR tests because they exercise timing and failure
orderings a recording can't produce on demand.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, TextPart, UserPromptPart
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import Batch, BatchModel, BatchRequest, BatchResult, BatchSettings, EncodedBatchRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from ..batch_utils import FAST_BATCH_SETTINGS, SlowBatchTestModel


@pytest.mark.parametrize(
    'settings',
    [
        {'max_requests': 0},
        {'max_wait': timedelta(0)},
        {'poll_interval': timedelta(seconds=-1)},
        {'poll_interval': timedelta(minutes=20), 'max_poll_interval': timedelta(minutes=10)},
    ],
)
def test_batch_settings_are_validated(settings: BatchSettings) -> None:
    with pytest.raises(UserError, match='must be positive'):
        BatchModel(TestModel(), batch_settings=settings)


def test_batch_model_needs_a_model_with_a_batch_api() -> None:
    @dataclass(init=False)
    class NoBatchModel(TestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return None

    with pytest.raises(UserError, match=r"`NoBatchModel` 'test' has no batch API\."):
        BatchModel(NoBatchModel())


def test_batch_size_is_capped_by_the_provider_limit() -> None:
    assert BatchModel(TestModel(), batch_settings={'max_requests': 1_000_000}).max_requests == 10_000


async def test_concurrent_runs_share_a_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=2)
    agent = Agent(BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 3, 'max_wait': None}))

    results = await asyncio.gather(*(agent.run(f'Hello {i}') for i in range(3)))

    assert [result.output for result in results] == ['success (no tool calls)'] * 3
    assert [batch.request_count for batch in wrapped.created] == [3]
    for result in results:
        assert [type(message).__name__ for message in result.all_messages()] == ['ModelRequest', 'ModelResponse']
        response = result.all_messages()[-1]
        assert isinstance(response, ModelResponse)
        assert response.state == 'complete'
        assert result.usage.requests == 1


async def test_a_partial_batch_is_submitted_after_max_wait() -> None:
    wrapped = SlowBatchTestModel()
    agent = Agent(BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 10}))

    result = await agent.run('Hello')

    assert result.output == 'success (no tool calls)'
    assert [batch.request_count for batch in wrapped.created] == [1]


async def test_flush_submits_without_waiting() -> None:
    wrapped = SlowBatchTestModel()
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_wait': None})
    agent = Agent(model)

    runs = [asyncio.create_task(agent.run(f'Hello {i}')) for i in range(2)]
    await wrapped.wait_until(lambda m: m.encoded == 2)
    assert wrapped.created == []
    await model.flush()

    assert [result.output for result in await asyncio.gather(*runs)] == ['success (no tool calls)'] * 2
    assert [batch.request_count for batch in wrapped.created] == [2]


async def test_a_cancelled_request_leaves_the_queue_before_submission() -> None:
    wrapped = SlowBatchTestModel()
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_wait': None})
    agent = Agent(model)

    kept = asyncio.create_task(agent.run('Keep me'))
    dropped = asyncio.create_task(agent.run('Drop me'))
    await wrapped.wait_until(lambda m: m.encoded == 2)
    dropped.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dropped
    await model.flush()

    assert (await kept).output == 'success (no tool calls)'
    assert [batch.request_count for batch in wrapped.created] == [1]


async def test_checks_back_off_up_to_the_maximum_interval() -> None:
    model = BatchModel(
        TestModel(),
        batch_settings={
            'max_wait': timedelta(milliseconds=1),
            'poll_interval': timedelta(seconds=10),
            'max_poll_interval': timedelta(seconds=30),
        },
    )
    pending = await _pending_response(model)

    delays = [model.continuation_delay(_with_polls(pending, polls)) for polls in range(5)]
    assert delays == snapshot([10.0, 15.0, 22.5, 30.0, 30.0])
    # A response that isn't waiting for a batch gets the wrapped model's delay.
    assert model.continuation_delay(ModelResponse(parts=[TextPart('done')])) is None


async def test_a_run_resumes_waiting_for_the_same_batch_from_history() -> None:
    """A history ending in a pending batch response picks the batch back up instead of submitting the request again."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    model = BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)
    assert pending.state == 'suspended'
    assert [batch.request_count for batch in wrapped.created] == [1]

    # A fresh `BatchModel` over the same provider model, as in a new process.
    agent = Agent(BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS))
    result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart('Hello')]), pending])

    assert result.output == 'success (no tool calls)'
    assert len(wrapped.created) == 1
    assert [message.state for message in result.all_messages() if isinstance(message, ModelResponse)] == ['complete']


async def test_a_provider_continuation_keeps_its_content_across_batch_rounds() -> None:
    """A suspended batch result is continued in a new batch request, and the turn keeps both segments."""
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart('The answer ')], state='suspended', provider_response_id='seg1')
        assert isinstance(messages[-1], ModelResponse) and messages[-1].text == 'The answer '
        return ModelResponse(parts=[TextPart('is 42.')], provider_response_id='seg2')

    result = await Agent(BatchModel(FunctionModel(respond), batch_settings=FAST_BATCH_SETTINGS)).run(
        'What is the answer?'
    )

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    assert calls == 2


async def test_a_batch_error_reaches_the_run() -> None:
    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name='function', body={'error': 'bad request'})

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(BatchModel(FunctionModel(fail), batch_settings=FAST_BATCH_SETTINGS)).run('Hello')
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {'error': 'bad request'}


async def test_a_result_missing_from_the_batch_is_a_model_error() -> None:
    @dataclass(init=False)
    class LosingModel(TestModel):
        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            return []

    with pytest.raises(ModelAPIError, match='expired before the model ran it'):
        await Agent(BatchModel(LosingModel(), batch_settings=FAST_BATCH_SETTINGS)).run('Hello')


async def test_abandoning_every_request_cancels_the_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)

    await model.cancel_suspended_response(pending)

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_cancelling_a_run_cancels_its_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    run = asyncio.create_task(Agent(BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS)).run('Hello'))
    await wrapped.wait_until(lambda m: m.status_checks > 0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_cancelling_a_run_from_an_anyio_scope_cancels_its_batch() -> None:
    """The provider cancel makes a request, so it only completes if the re-delivered `anyio` cancellation can't stop it."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    agent = Agent(BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS))

    async def run() -> None:
        await agent.run('Hello')

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        await wrapped.wait_until(lambda m: m.status_checks > 0)
        tg.cancel_scope.cancel()

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_response_that_is_not_waiting_is_cancelled_by_the_wrapped_model() -> None:
    model = BatchModel(TestModel(), batch_settings=FAST_BATCH_SETTINGS)
    await model.cancel_suspended_response(ModelResponse(parts=[], state='suspended'))


async def test_streaming_is_rejected() -> None:
    agent = Agent(BatchModel(TestModel()))
    with pytest.raises(UserError, match='does not support streaming'):
        async with agent.run_stream('Hello'):
            pass  # pragma: no cover


async def test_model_request_batch_returns_responses_and_errors_in_request_order() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = messages[-1].parts[-1]
        assert isinstance(prompt, UserPromptPart)
        if prompt.content == 'fail':
            raise ModelHTTPError(status_code=429, model_name='function', body='slow down')
        return ModelResponse(parts=[TextPart(f'echo: {prompt.content}')])

    responses = await model_request_batch(
        FunctionModel(respond),
        [
            BatchRequest(custom_id=name, messages=[ModelRequest(parts=[UserPromptPart(name)])])
            for name in ('one', 'fail', 'two')
        ],
        batch_settings=FAST_BATCH_SETTINGS,
    )

    one, failed, two = responses
    assert isinstance(one, ModelResponse) and one.text == 'echo: one'
    assert isinstance(two, ModelResponse) and two.text == 'echo: two'
    assert isinstance(failed, ModelHTTPError)
    assert (failed.status_code, failed.body) == (429, 'slow down')


async def test_model_request_batch_splits_requests_by_the_provider_limit() -> None:
    @dataclass(init=False)
    class SmallBatchModel(SlowBatchTestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return 2

    model = SmallBatchModel(pending_checks=1)
    responses = await model_request_batch(
        model,
        [BatchRequest(custom_id=str(i), messages=[ModelRequest(parts=[UserPromptPart('Hello')])]) for i in range(5)],
        batch_settings=FAST_BATCH_SETTINGS,
    )

    assert len(responses) == 5
    assert [batch.request_count for batch in model.created] == [2, 2, 1]


async def test_model_request_batch_cancels_unfinished_batches_when_cancelled() -> None:
    model = SlowBatchTestModel(pending_checks=1000)
    task = asyncio.create_task(
        model_request_batch(
            model,
            [BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])],
            batch_settings=FAST_BATCH_SETTINGS,
        )
    )
    await model.wait_until(lambda m: m.status_checks > 0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert model.canceled == [model.created[0].id]


async def test_model_request_batch_cancels_unfinished_batches_under_an_anyio_cancel() -> None:
    model = SlowBatchTestModel(pending_checks=1000)

    async def send() -> None:
        await model_request_batch(
            model,
            [BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])],
            batch_settings=FAST_BATCH_SETTINGS,
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(send)
        await model.wait_until(lambda m: m.status_checks > 0)
        tg.cancel_scope.cancel()

    assert model.canceled == [model.created[0].id]


async def test_exiting_the_model_ends_its_queued_and_running_work() -> None:
    """Leaving `async with model:` fails queued requests and cancels in-flight submissions and checks."""
    never = anyio.Event()

    class HangingModel(SlowBatchTestModel):
        submissions = 0

        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            self.submissions += 1
            self.notify_waiters()
            if self.submissions == 2:
                await never.wait()
            return await super().batch_create(requests)

        async def batch_status(self, batch: Batch) -> Batch:
            await super().batch_status(batch)
            await never.wait()
            raise AssertionError('unreachable')  # pragma: no cover

    tasks_before = asyncio.all_tasks()
    wrapped = HangingModel(pending_checks=1)
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_wait': None, 'max_requests': 1})
    agent = Agent(model)
    async with model:
        checked = asyncio.create_task(agent.run('Check on me'))
        await wrapped.wait_until(lambda m: m.status_checks == 1)
        submitting = asyncio.create_task(agent.run('Hang on submission'))
        await wrapped.wait_until(lambda m: isinstance(m, HangingModel) and m.submissions == 2)
        model.max_requests = 2
        queued = asyncio.create_task(agent.run('Queue me'))
        await wrapped.wait_until(lambda m: m.encoded == 3)

    for run in (queued, submitting, checked):
        with pytest.raises(UserError, match='exited while the request was waiting for its batch'):
            await run
    assert asyncio.all_tasks() - tasks_before == {asyncio.current_task()} - tasks_before


async def test_only_the_outermost_exit_ends_queued_work() -> None:
    wrapped = SlowBatchTestModel()
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_wait': None})
    async with model:
        async with model:
            run = asyncio.create_task(Agent(model).run('Hello'))
            await wrapped.wait_until(lambda m: m.encoded == 1)
        await model.flush()
        assert (await run).output == 'success (no tool calls)'


async def test_model_request_batch_validates_its_input() -> None:
    @dataclass(init=False)
    class NoBatchModel(TestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return None

    request = BatchRequest(custom_id='same', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])
    with pytest.raises(UserError, match=r"`NoBatchModel` 'test' has no batch API\."):
        await model_request_batch(NoBatchModel(), [request])
    with pytest.raises(UserError, match='unique `custom_id`'):
        await model_request_batch(TestModel(), [request, request])


async def test_in_memory_batches_belong_to_their_model_instance() -> None:
    model = TestModel()
    batch = await model.batch_create(
        [await model.batch_encode(BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hi')])]))]
    )
    assert (await model.batch_cancel(batch)).status == 'ended'
    with pytest.raises(UserError, match='was not created by this model instance'):
        await TestModel().batch_status(batch)


async def _pending_response(model: BatchModel) -> ModelResponse:
    return await model.request([ModelRequest(parts=[UserPromptPart('Hello')])], None, ModelRequestParameters())


def _with_polls(response: ModelResponse, polls: int) -> ModelResponse:
    assert response.provider_details is not None
    ticket = {**response.provider_details['batch'], 'polls': polls}
    return ModelResponse(
        parts=response.parts,
        state=response.state,
        provider_response_id=response.provider_response_id,
        provider_details={**response.provider_details, 'batch': ticket},
    )


async def test_a_failed_submission_reaches_every_request_in_the_batch() -> None:
    class FailingBatchModel(TestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            raise ModelHTTPError(status_code=500, model_name='test', body='batch creation failed')

    agent = Agent(
        BatchModel(FailingBatchModel(), batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 2, 'max_wait': None})
    )
    results = await asyncio.gather(agent.run('one'), agent.run('two'), return_exceptions=True)

    assert [type(result).__name__ for result in results] == ['ModelHTTPError', 'ModelHTTPError']


async def test_identical_requests_get_distinct_ids_in_a_batch() -> None:
    """IDs are derived from the encoded request, so they repeat across reruns, but never within one batch."""

    class ConstantBodyModel(SlowBatchTestModel):
        def __init__(self) -> None:
            super().__init__()
            self.tokens: list[str] = []
            self.custom_ids: list[str] = []

        async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
            encoded = await super().batch_encode(request)
            self.tokens.append(encoded.body)
            return EncodedBatchRequest(custom_id=encoded.custom_id, group=encoded.group, body='same')

        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            self.custom_ids = [request.custom_id for request in requests]
            tokens = iter(self.tokens)
            return await super().batch_create([replace(request, body=next(tokens)) for request in requests])

    wrapped = ConstantBodyModel()
    agent = Agent(BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 3, 'max_wait': None}))

    results = await asyncio.gather(*(agent.run('Hello') for _ in range(3)))

    assert [result.output for result in results] == ['success (no tool calls)'] * 3
    base_id = wrapped.custom_ids[0]
    assert wrapped.custom_ids == [base_id, f'{base_id}-1', f'{base_id}-2']


async def test_requests_cancelled_before_submission_submit_nothing() -> None:
    wrapped = SlowBatchTestModel()
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_wait': None})
    run = asyncio.create_task(Agent(model).run('Hello'))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await model.flush()

    assert wrapped.created == []


async def test_abandoning_some_requests_keeps_the_batch_running() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 2, 'max_wait': None})
    first, _ = await asyncio.gather(_pending_response(model), _pending_response(model))

    await model.cancel_suspended_response(first)

    assert wrapped.canceled == []


async def test_results_are_shared_by_the_requests_of_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request of a batch reads the results fetched once, and only the most recent batches' are kept."""
    monkeypatch.setattr('pydantic_ai.models.batch._RESULTS_CACHE_SIZE', 1)

    @dataclass(init=False)
    class CountingResultsModel(TestModel):
        result_fetches: int = 0

        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            self.result_fetches += 1
            return await super().batch_results(batch)

    wrapped = CountingResultsModel()
    wrapped.result_fetches = 0
    agent = Agent(BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 2, 'max_wait': None}))

    await asyncio.gather(agent.run('one'), agent.run('two'))
    await asyncio.gather(agent.run('three'), agent.run('four'))

    assert wrapped.result_fetches == 2


def test_a_malformed_batch_ticket_is_not_waited_on() -> None:
    model = BatchModel(TestModel())
    response = ModelResponse(parts=[], state='suspended', provider_details={'batch': 'not a ticket'})
    assert model.continuation_delay(response) is None


async def test_a_request_checked_after_the_results_arrived_reads_them_from_memory() -> None:
    @dataclass(init=False)
    class CountingResultsModel(TestModel):
        result_fetches: int = 0

        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            self.result_fetches += 1
            return await super().batch_results(batch)

    wrapped = CountingResultsModel()
    wrapped.result_fetches = 0
    model = BatchModel(wrapped, batch_settings={**FAST_BATCH_SETTINGS, 'max_requests': 2, 'max_wait': None})
    first, second = await asyncio.gather(_pending_response(model), _pending_response(model))
    prompt = ModelRequest(parts=[UserPromptPart('Hello')])

    for pending in (first, second):
        response = await model.request([prompt, pending], None, ModelRequestParameters())
        assert response.state == 'complete'

    assert wrapped.result_fetches == 1


async def test_a_paused_turn_keeps_its_content_when_the_result_names_a_dated_model() -> None:
    """Results often name a dated model, which must not make the graph drop what the turn accumulated before."""

    @dataclass(init=False)
    class DatedModel(TestModel):
        calls: int = 0

        async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            self.calls += 1
            return ModelResponse(
                parts=[TextPart(f'segment {self.calls} ')],
                model_name=f'{self.model_name}-2026-01-01',
                state='suspended' if self.calls == 1 else 'complete',
                provider_response_id=str(self.calls),
            )

    wrapped = DatedModel()
    wrapped.calls = 0
    result = await Agent(BatchModel(wrapped, batch_settings=FAST_BATCH_SETTINGS)).run('Hello')

    assert result.output == 'segment 1 segment 2 '


async def test_wrapper_models_forward_the_batch_api() -> None:
    model = WrapperModel(TestModel())
    request = BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])

    assert model.batch_max_requests == 10_000
    batch = await model.batch_create([await model.batch_encode(request)])
    assert (await model.batch_status(batch)).status == 'ended'
    assert (await model.batch_cancel(batch)).status == 'ended'
    [result] = await model.batch_results(batch)
    assert model.batch_decode(result, request).text == 'success (no tool calls)'
