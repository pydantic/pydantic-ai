"""`BatchModelRequests`, `model_request_batch` and the batch contract, using the in-memory batch API of the test models.

Provider codecs and their recorded batches are tested per provider (`test_anthropic_batch*.py`, ...); these tests pin the
provider-independent behavior: grouping requests into batches, waiting through the continuation loop, resuming from
history, and how results and errors reach a run. They aren't VCR tests because they exercise timing and failure
orderings a recording can't produce on demand.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterable, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    ModelRequest,
    NativeOutput,
    PendingModelResponse,
    PromptedOutput,
    RunContext,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import BatchModelRequests, Hooks, Instrumentation
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelRequestTimeout, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models._batch_model import BatchedModel, BatchPool
from pydantic_ai.models._batch_settings import poll_delay
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from ..batch_utils import FAST_BATCH_SETTINGS, FAST_POLLING, READINESS_WAIT_TIMEOUT, SlowBatchTestModel
from ..conftest import try_import

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


@pytest.mark.parametrize(
    'settings',
    [
        {'max_requests': 0},
        {'max_wait': timedelta(0)},
        {'poll_interval': timedelta(seconds=-1)},
        {'poll_interval': timedelta(minutes=20), 'max_poll_interval': timedelta(minutes=10)},
    ],
)
def test_batch_settings_are_validated(settings: dict[str, Any]) -> None:
    with pytest.raises(UserError, match='must be positive'):
        BatchModelRequests(**settings)


def test_batch_model_needs_a_model_with_a_batch_api() -> None:
    @dataclass(init=False)
    class NoBatchModel(TestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return None

    agent = Agent(NoBatchModel(), capabilities=[BatchModelRequests()])
    with pytest.raises(UserError, match=r"`NoBatchModel` 'test' has no batch API\."):
        agent.run_sync('Hello')


async def test_batch_size_is_capped_by_the_provider_limit() -> None:
    @dataclass(init=False)
    class SmallBatchModel(SlowBatchTestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return 2

    wrapped = SmallBatchModel()
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=1_000, max_wait=None)])

    await asyncio.gather(*(agent.run(f'Hello {i}') for i in range(4)))

    assert [batch.request_count for batch in wrapped.created] == [2, 2]


async def test_agents_sharing_the_capability_share_batches() -> None:
    """One `BatchModelRequests` is one pool: added to several agents, or to runs, their requests share a batch."""
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_requests=3, max_wait=None)
    writer = Agent(wrapped, instructions='Write.', capabilities=[batching])
    reviewer = Agent(wrapped, instructions='Review.', capabilities=[batching])
    unbatched = Agent(wrapped)

    results = await asyncio.gather(
        writer.run('Hello'), reviewer.run('Hello'), unbatched.run('Hello', capabilities=[batching])
    )

    assert [result.output for result in results] == ['success (no tool calls)'] * 3
    assert [batch.request_count for batch in wrapped.created] == [3]


async def test_separate_capabilities_have_separate_batches() -> None:
    wrapped = SlowBatchTestModel()
    first = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    second = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    await asyncio.gather(first.run('Hello'), second.run('Hello'))

    assert [batch.request_count for batch in wrapped.created] == [1, 1]


async def test_the_model_serving_the_request_is_batched() -> None:
    """The run's `model=` and the model a `before_model_request` hook swaps in are batched, each in its own batch."""
    agent_model = SlowBatchTestModel()
    run_model = SlowBatchTestModel()
    hook_model = SlowBatchTestModel()
    hooks = Hooks[Any]()

    @hooks.on.before_model_request
    async def swap(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        if ctx.prompt == 'swap':
            request_context.model = hook_model
        return request_context

    agent = Agent(agent_model, capabilities=[hooks, BatchModelRequests(**FAST_BATCH_SETTINGS)])

    await agent.run('Hello', model=run_model)
    await agent.run('swap')

    assert [len(model.created) for model in (agent_model, run_model, hook_model)] == [0, 1, 1]


async def test_a_model_already_batched_is_not_batched_again() -> None:
    wrapped = SlowBatchTestModel()
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS), BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )

    assert (await agent.run('Hello')).output == 'success (no tool calls)'
    assert [batch.request_count for batch in wrapped.created] == [1]


async def test_concurrent_runs_share_a_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=2)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=3, max_wait=None)])

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
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS, max_requests=10)])

    result = await agent.run('Hello')

    assert result.output == 'success (no tool calls)'
    assert [batch.request_count for batch in wrapped.created] == [1]


async def test_flush_submits_without_waiting() -> None:
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    agent = Agent(wrapped, capabilities=[batching])

    runs = [asyncio.create_task(agent.run(f'Hello {i}')) for i in range(2)]
    await wrapped.wait_until(lambda m: m.encoded == 2)
    assert wrapped.created == []
    await batching.flush()

    assert [result.output for result in await asyncio.gather(*runs)] == ['success (no tool calls)'] * 2
    assert [batch.request_count for batch in wrapped.created] == [2]


async def test_cancelling_a_flush_only_stops_its_caller_waiting() -> None:
    submitting = asyncio.Event()
    release = asyncio.Event()

    class GatedBatchModel(SlowBatchTestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            submitting.set()
            await release.wait()
            return await super().batch_create(requests)

    wrapped = GatedBatchModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    agent = Agent(wrapped, capabilities=[batching])

    runs = [asyncio.create_task(agent.run(f'Hello {i}')) for i in range(2)]
    await wrapped.wait_until(lambda m: m.encoded == 2)
    flush = asyncio.create_task(batching.flush())
    await submitting.wait()
    flush.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flush
    release.set()

    assert [result.output for result in await asyncio.gather(*runs)] == ['success (no tool calls)'] * 2
    assert [batch.request_count for batch in wrapped.created] == [2]


async def test_a_cancelled_request_leaves_the_queue_before_submission() -> None:
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    agent = Agent(wrapped, capabilities=[batching])

    kept = asyncio.create_task(agent.run('Keep me'))
    dropped = asyncio.create_task(agent.run('Drop me'))
    await wrapped.wait_until(lambda m: m.encoded == 2)
    dropped.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dropped
    await batching.flush()

    assert (await kept).output == 'success (no tool calls)'
    assert [batch.request_count for batch in wrapped.created] == [1]


async def test_checks_back_off_up_to_the_maximum_interval() -> None:
    model = _batched(
        TestModel(),
        max_wait=timedelta(milliseconds=1),
        poll_interval=timedelta(seconds=10),
        max_poll_interval=timedelta(seconds=30),
    )
    pending = await _pending_response(model)

    delays = [model.continuation_delay(_with_polls(pending, polls)) for polls in range(5)]
    assert delays == snapshot([10.0, 15.0, 22.5, 30.0, 30.0])
    # A response that isn't waiting for a batch gets the wrapped model's delay.
    assert model.continuation_delay(ModelResponse(parts=[TextPart('done')])) is None


async def test_a_run_resumes_waiting_for_the_same_batch_from_history() -> None:
    """A history ending in a pending batch response picks the batch back up instead of submitting the request again."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)
    assert pending.state == 'suspended'
    assert [batch.request_count for batch in wrapped.created] == [1]

    # A fresh `BatchModelRequests` over the same provider model, as in a new process.
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])
    result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart('Hello')]), pending])

    assert result.output == 'success (no tool calls)'
    assert len(wrapped.created) == 1
    assert [message.state for message in result.all_messages() if isinstance(message, ModelResponse)] == ['complete']


async def test_a_run_with_pending_output_ends_once_its_batch_is_submitted() -> None:
    """With `PendingModelResponse` in the output type, a run ends instead of waiting, and resuming checks once."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)], output_type=[str, PendingModelResponse]
    )

    result = await agent.run('Hello')
    assert isinstance(result.output, PendingModelResponse)
    assert result.output.response == result.all_messages()[-1]
    assert result.output.response.state == 'suspended'
    assert (len(wrapped.created), wrapped.status_checks) == (1, 0)

    # A fresh `BatchModelRequests` over the same provider model, as in a new process.
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)], output_type=[str, PendingModelResponse]
    )
    still_pending = await agent.run(message_history=result.all_messages())
    assert isinstance(still_pending.output, PendingModelResponse)
    assert (len(wrapped.created), wrapped.status_checks) == (1, 1)
    # The new check replaces the pending response, rather than being added after it.
    assert len(still_pending.all_messages()) == len(result.all_messages())

    # Another process again: a check in the same pool within `poll_interval` would share the last status.
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)], output_type=[str, PendingModelResponse]
    )
    finished = await agent.run(message_history=still_pending.all_messages())
    assert finished.output == 'success (no tool calls)'
    assert (len(wrapped.created), wrapped.status_checks, wrapped.canceled) == (1, 2, [])
    assert [message.state for message in finished.all_messages() if isinstance(message, ModelResponse)] == ['complete']


async def test_a_run_with_pending_output_ends_for_each_request_it_makes() -> None:
    """Each model request of a run that calls a tool ends the run with a `PendingModelResponse` until it's answered."""
    wrapped = SlowBatchTestModel()
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)], output_type=[str, PendingModelResponse]
    )

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    result = await agent.run('Where?')
    pending_outputs = 0
    while isinstance(result.output, PendingModelResponse):
        pending_outputs += 1
        result = await agent.run(message_history=result.all_messages())

    assert result.output == snapshot('{"get_city":"Paris"}')
    assert pending_outputs == 2
    assert len(wrapped.created) == 2


def test_pending_model_response_needs_another_output_type() -> None:
    with pytest.raises(UserError, match='other than `PendingModelResponse`'):
        Agent(TestModel(), output_type=PendingModelResponse)
    with pytest.raises(UserError, match='`NativeOutput` cannot contain `PendingModelResponse`'):
        Agent(TestModel(), output_type=NativeOutput([str, PendingModelResponse]))
    with pytest.raises(UserError, match='`PromptedOutput` cannot contain `PendingModelResponse`'):
        Agent(TestModel(), output_type=PromptedOutput([str, PendingModelResponse]))


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

    result = await Agent(FunctionModel(respond), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run(
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
        await Agent(FunctionModel(fail), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {'error': 'bad request'}


async def test_a_result_missing_from_the_batch_is_submitted_again_and_then_a_model_error() -> None:
    """A batch that ended without the request's result didn't run it: it's resubmitted, then reported as missing."""

    class LosingModel(SlowBatchTestModel):
        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            return []

    wrapped = LosingModel()
    with pytest.raises(ModelAPIError, match=r"Batch 'batch_\w+' ended without a result for the request\.") as exc_info:
        await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')
    assert type(exc_info.value) is ModelAPIError
    assert len(wrapped.created) == 3


async def test_abandoning_every_request_cancels_the_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)

    await model.cancel_suspended_response(pending)

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_cancelling_a_run_cancels_its_batch() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    run = asyncio.create_task(Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello'))
    await wrapped.wait_until(lambda m: m.status_checks > 0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_cancelling_a_run_from_an_anyio_scope_cancels_its_batch() -> None:
    """The provider cancel makes a request, so it only completes if the re-delivered `anyio` cancellation can't stop it."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    async def run() -> None:
        await agent.run('Hello')

    async with anyio.create_task_group() as tg:
        tg.start_soon(run)
        await wrapped.wait_until(lambda m: m.status_checks > 0)
        tg.cancel_scope.cancel()

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_request_timeout_covers_the_wait_for_the_batch() -> None:
    """The deadline spans the submission and every check, so the run stops waiting at it and cancels the batch."""
    wrapped = SlowBatchTestModel(pending_checks=100_000)
    agent = Agent(
        wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)], model_settings={'request_timeout': 0.5}
    )

    with pytest.raises(ModelRequestTimeout):
        await agent.run('Hello')

    assert wrapped.status_checks > 0
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_response_that_is_not_waiting_is_cancelled_by_the_wrapped_model() -> None:
    model = _batched(TestModel(), **FAST_BATCH_SETTINGS)
    await model.cancel_suspended_response(ModelResponse(parts=[], state='suspended'))


def test_a_response_that_is_not_waiting_is_released_and_limited_by_the_wrapped_model() -> None:
    """A suspended response that isn't a batch request's, like an OpenAI background response, is the wrapped model's."""
    response = ModelResponse(parts=[], state='suspended')
    model = _batched(WrapperModel(TestModel()), **FAST_BATCH_SETTINGS)
    model._release_suspended_response(response)  # pyright: ignore[reportPrivateUsage]
    assert model._limits_polls(response)  # pyright: ignore[reportPrivateUsage]
    assert not model._withdraws_rejected_suspended_response(response)  # pyright: ignore[reportPrivateUsage]


async def test_an_ended_batchs_results_are_kept_until_every_request_waiting_on_it_has_read_or_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results too large to cache are still kept for the requests waiting on them, and dropped once none are."""
    monkeypatch.setattr('pydantic_ai.models._batch_model._RESULTS_CACHE_BYTES', 1)
    wrapped = SlowBatchTestModel()
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    prompts = [ModelRequest(parts=[UserPromptPart(prompt)]) for prompt in ('one', 'two')]
    first, second = await asyncio.gather(
        *(model.request([prompt], None, ModelRequestParameters()) for prompt in prompts)
    )
    assert len(wrapped.created) == 1

    response = await model.request([prompts[0], first], None, ModelRequestParameters())
    assert response.state == 'complete'
    assert len(model.pool._results) == 1  # pyright: ignore[reportPrivateUsage]

    await model.cancel_suspended_response(second)
    assert model.pool._results == {}  # pyright: ignore[reportPrivateUsage]
    assert wrapped.canceled == []


async def test_a_request_left_for_a_later_run_no_longer_keeps_its_batchs_results() -> None:
    wrapped = SlowBatchTestModel()
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    prompts = [ModelRequest(parts=[UserPromptPart(prompt)]) for prompt in ('one', 'two')]
    first, second = await asyncio.gather(
        *(model.request([prompt], None, ModelRequestParameters()) for prompt in prompts)
    )
    await model.request([prompts[0], first], None, ModelRequestParameters())
    assert len(model.pool._results) == 1  # pyright: ignore[reportPrivateUsage]

    model._release_suspended_response(second)  # pyright: ignore[reportPrivateUsage]
    assert model.pool._results == {}  # pyright: ignore[reportPrivateUsage]


async def test_a_request_left_for_a_later_run_before_its_batch_ended_doesnt_keep_its_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Results too large to cache are kept only for requests still waiting on them, not one a run left pending."""
    monkeypatch.setattr('pydantic_ai.models._batch_model._RESULTS_CACHE_BYTES', 1)
    wrapped = SlowBatchTestModel()
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    prompts = [ModelRequest(parts=[UserPromptPart(prompt)]) for prompt in ('one', 'two')]
    first, second = await asyncio.gather(
        *(model.request([prompt], None, ModelRequestParameters()) for prompt in prompts)
    )
    model._release_suspended_response(second)  # pyright: ignore[reportPrivateUsage]

    response = await model.request([prompts[0], first], None, ModelRequestParameters())

    assert response.state == 'complete'
    assert model.pool._results == {}  # pyright: ignore[reportPrivateUsage]


async def test_the_results_cache_counts_the_bytes_of_result_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    text = 'é' * 1_000
    wrapped = SlowBatchTestModel()
    wrapped.custom_output_text = text
    pending = await _pending_response(_batched(wrapped, **FAST_BATCH_SETTINGS))
    # Between the characters and the UTF-8 bytes of the result body, which holds the text.
    monkeypatch.setattr('pydantic_ai.models._batch_model._RESULTS_CACHE_BYTES', 2_000)
    collecting = _batched(wrapped, **FAST_BATCH_SETTINGS)
    response = await collecting.request(
        [ModelRequest(parts=[UserPromptPart('Hello')]), pending], None, ModelRequestParameters()
    )
    assert response.parts == [TextPart(text)]
    assert collecting.pool._results == {}  # pyright: ignore[reportPrivateUsage]


async def test_a_streamed_run_gets_the_batch_result_all_at_once() -> None:
    """A streamed run waits for its batch like any run, and then streams each part of the result in one event."""
    wrapped = SlowBatchTestModel(pending_checks=2)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    async with agent.run_stream('Hello') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True)]
        assert await result.get_output() == 'success (no tool calls)'
    assert chunks == ['success (no tool calls)']
    assert (len(wrapped.created), wrapped.status_checks) == (1, 3)


async def test_event_stream_handlers_and_run_stream_events_get_a_batched_runs_events() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    @agent.tool_plain
    def get_city() -> str:
        return 'Paris'

    handled: list[AgentStreamEvent] = []

    async def handle(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        handled.extend([event async for event in events])

    result = await agent.run('Where?', event_stream_handler=handle)
    assert result.output == snapshot('{"get_city":"Paris"}')
    async with agent.run_stream_events('Where?') as stream:
        streamed = [event async for event in stream]
    assert isinstance(streamed[-1], AgentRunResultEvent)
    assert streamed[-1].result.output == result.output
    for events in (handled, streamed):
        assert [type(event).__name__ for event in events if not isinstance(event, AgentRunResultEvent)] == snapshot(
            [
                'PartStartEvent',
                'PartEndEvent',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                'PartStartEvent',
                'FinalResultEvent',
                'PartEndEvent',
            ]
        )
    assert len(wrapped.created) == 4


async def test_a_streamed_run_can_end_pending_and_be_collected_later() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(
        wrapped, output_type=[str, PendingModelResponse], capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]
    )
    async with agent.run_stream('Hello') as result:
        pending = await result.get_output()
    assert isinstance(pending, PendingModelResponse)
    history = result.all_messages()

    async with agent.run_stream(message_history=history) as result:
        assert isinstance(await result.get_output(), PendingModelResponse)
        history = result.all_messages()
    async with agent.run_stream(message_history=history) as result:
        assert await result.get_output() == 'success (no tool calls)'
    assert (len(wrapped.created), wrapped.canceled) == (1, [])


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
        **FAST_POLLING,
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
        **FAST_POLLING,
    )

    assert len(responses) == 5
    assert [batch.request_count for batch in model.created] == [2, 2, 1]


async def test_model_request_batch_cancels_unfinished_batches_when_cancelled() -> None:
    model = SlowBatchTestModel(pending_checks=1000)
    task = asyncio.create_task(
        model_request_batch(
            model,
            [BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])],
            **FAST_POLLING,
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
            **FAST_POLLING,
        )

    async with anyio.create_task_group() as tg:
        tg.start_soon(send)
        await model.wait_until(lambda m: m.status_checks > 0)
        tg.cancel_scope.cancel()

    assert model.canceled == [model.created[0].id]


async def test_model_request_batch_does_not_cancel_batches_that_ended() -> None:
    class FailingResultsModel(SlowBatchTestModel):
        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            raise ModelHTTPError(status_code=500, model_name='test', body='results unavailable')

    model = FailingResultsModel()
    with pytest.raises(ModelHTTPError, match='results unavailable'):
        await model_request_batch(
            model,
            [BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])],
            **FAST_POLLING,
        )

    assert len(model.created) == 1
    assert model.canceled == []


async def test_exiting_the_capability_ends_its_queued_and_running_work() -> None:
    """Leaving `async with batching:` fails queued requests and cancels in-flight submissions and checks."""
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
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None, max_requests=1)
    agent = Agent(wrapped, capabilities=[batching])
    async with batching:
        checked = asyncio.create_task(agent.run('Check on me'))
        await wrapped.wait_until(lambda m: m.status_checks == 1)
        submitting = asyncio.create_task(agent.run('Hang on submission'))
        await wrapped.wait_until(lambda m: isinstance(m, HangingModel) and m.submissions == 2)
        batching._pool.max_requests = 2  # pyright: ignore[reportPrivateUsage]
        queued = asyncio.create_task(agent.run('Queue me'))
        await wrapped.wait_until(lambda m: m.encoded == 3)

    for run in (queued, submitting, checked):
        with pytest.raises(UserError, match='exited while the request was waiting for its batch'):
            await run
    assert asyncio.all_tasks() - tasks_before == {asyncio.current_task()} - tasks_before


async def test_closing_before_a_submission_starts_fails_its_requests() -> None:
    """A submission cancelled before its first step still fails the requests whose queue it took, so none hangs."""
    wrapped = SlowBatchTestModel()
    model = _batched(wrapped, **FAST_POLLING, max_wait=None)
    request = asyncio.create_task(_pending_response(model))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    await asyncio.sleep(0)
    flush = asyncio.create_task(model.pool.flush())
    # The flush takes the queue and schedules its submission, which hasn't started when the pool closes.
    await asyncio.sleep(0)
    await model.pool.close()

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        with pytest.raises(UserError, match='exited while the request was waiting for its batch'):
            await request
        await flush
    assert wrapped.created == []


async def test_only_the_outermost_exit_ends_queued_work() -> None:
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    async with batching:
        async with batching:
            run = asyncio.create_task(Agent(wrapped, capabilities=[batching]).run('Hello'))
            await wrapped.wait_until(lambda m: m.encoded == 1)
        await batching.flush()
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
    timed_request = BatchRequest(custom_id='timed', messages=request.messages, model_settings={'request_timeout': 60})
    with pytest.raises(UserError, match='does not support `request_timeout`'):
        await model_request_batch(TestModel(), [timed_request])
    with pytest.raises(UserError, match='does not support `request_timeout`'):
        await model_request_batch(TestModel(settings={'request_timeout': 60}), [request])


async def test_in_memory_batches_belong_to_their_model_instance() -> None:
    model = TestModel()
    batch = await model.batch_create(
        [await model.batch_encode(BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hi')])]))]
    )
    assert (await model.batch_cancel(batch)).status == 'ended'
    with pytest.raises(UserError, match='was not created by this model instance'):
        await TestModel().batch_status(batch)


def _batched(wrapped: Model, **settings: Any) -> BatchedModel:
    """`wrapped` in a pool of its own, as `BatchModelRequests` wraps the model serving a request."""
    return BatchPool(**settings).wrap(wrapped)


async def _pending_response(model: BatchedModel) -> ModelResponse:
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

    agent = Agent(FailingBatchModel(), capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=None)])
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
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=3, max_wait=None)])

    results = await asyncio.gather(*(agent.run('Hello') for _ in range(3)))

    assert [result.output for result in results] == ['success (no tool calls)'] * 3
    base_id = wrapped.custom_ids[0]
    assert wrapped.custom_ids == [base_id, f'{base_id}-1', f'{base_id}-2']


async def test_requests_cancelled_before_submission_submit_nothing() -> None:
    wrapped = SlowBatchTestModel()
    batching = BatchModelRequests(**FAST_POLLING, max_wait=None)
    run = asyncio.create_task(Agent(wrapped, capabilities=[batching]).run('Hello'))
    await wrapped.wait_until(lambda m: m.encoded == 1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    await batching.flush()

    assert wrapped.created == []


async def test_abandoning_some_requests_keeps_the_batch_running() -> None:
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
    first, _ = await asyncio.gather(_pending_response(model), _pending_response(model))

    await model.cancel_suspended_response(first)

    assert wrapped.canceled == []


def _forged(response: ModelResponse, *, request_count: int, custom_id: str | None = None) -> ModelResponse:
    """`response` with its ticket edited, as a caller who controls the message history could."""
    assert response.provider_details is not None
    ticket = response.provider_details['batch']
    batch = {**ticket['batch'], 'request_count': request_count}
    custom_id = custom_id or ticket['custom_id']
    return replace(
        response,
        provider_response_id=f'batch:{batch["id"]}:{custom_id}',
        provider_details={**response.provider_details, 'batch': {**ticket, 'custom_id': custom_id, 'batch': batch}},
    )


async def test_a_forged_ticket_cannot_cancel_a_batch_other_requests_wait_on() -> None:
    """Only the requests the pool submitted in a batch count towards cancelling it, whatever a ticket claims."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
    first, second = await asyncio.gather(_pending_response(model), _pending_response(model))

    # Claiming to be the batch's only request doesn't make the caller its last participant.
    await model.cancel_suspended_response(_forged(first, request_count=1))
    assert wrapped.canceled == []
    # Nor does abandoning, again and again, requests the batch doesn't hold.
    for custom_id in ('other', 'another'):
        await model.cancel_suspended_response(_forged(first, custom_id=custom_id, request_count=1))
    assert wrapped.canceled == []

    await model.cancel_suspended_response(second)
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_abandoning_a_batch_another_pool_submitted_leaves_it_running() -> None:
    """A ticket resumed from a message history can't say who else waits on its batch, so the batch isn't cancelled."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    pending = await _pending_response(_batched(wrapped, **FAST_BATCH_SETTINGS))

    await _batched(wrapped, **FAST_BATCH_SETTINGS).cancel_suspended_response(pending)

    assert wrapped.canceled == []


async def test_a_batch_is_cancelled_only_once() -> None:
    """Abandoning a request of a batch that was already cancelled, like a resumed history cancelled again, is a no-op."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)

    await model.cancel_suspended_response(pending)
    await model.cancel_suspended_response(pending)

    assert wrapped.canceled == [wrapped.created[0].id]
    assert (model.pool._submitted, model.pool._abandoned) == ({}, {})  # pyright: ignore[reportPrivateUsage]


def _serving_model_recorder() -> tuple[Hooks[Any], list[Model]]:
    """A capability recording the model that served each request, which a caller cancels a pending request through."""
    hooks = Hooks[Any]()
    served: list[Model] = []

    @hooks.on.after_model_request
    async def record(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        served.append(request_context.model)
        return response

    return hooks, served


async def test_cancelling_a_request_a_run_left_pending_cancels_its_batch() -> None:
    """A run that ends pending releases its request, but its caller can still cancel it, as the Harness does."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    recorder, served = _serving_model_recorder()
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS), recorder],
    )

    result = await agent.run('Hello')
    assert isinstance(result.output, PendingModelResponse)
    await served[-1].cancel_suspended_response(result.output.response)

    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_request_left_pending_keeps_its_batch_running_until_it_is_cancelled() -> None:
    """Released requests count as abandoned only once cancelled, so one a later run may collect keeps the batch."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    recorder, served = _serving_model_recorder()
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=None), recorder],
    )

    first, second = await asyncio.gather(agent.run('one'), agent.run('two'))
    assert len(wrapped.created) == 1
    assert isinstance(first.output, PendingModelResponse) and isinstance(second.output, PendingModelResponse)
    await served[0].cancel_suspended_response(first.output.response)
    assert wrapped.canceled == []

    await served[0].cancel_suspended_response(second.output.response)
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_request_resumed_by_a_later_run_can_no_longer_cancel_its_batch() -> None:
    """Once a later run resumed a released request, its history may be resumed again elsewhere, so it's left running."""
    wrapped = SlowBatchTestModel(pending_checks=1000)
    recorder, served = _serving_model_recorder()
    agent = Agent(
        wrapped,
        output_type=[str, PendingModelResponse],
        capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS), recorder],
    )
    result = await agent.run('Hello')
    resumed = await agent.run(message_history=result.all_messages())
    assert isinstance(resumed.output, PendingModelResponse)

    await served[-1].cancel_suspended_response(resumed.output.response)

    assert wrapped.canceled == []


async def test_the_oldest_submitted_batches_are_forgotten_and_left_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pool remembers a bounded number of batches it submitted; abandoning a forgotten one doesn't cancel it."""
    monkeypatch.setattr('pydantic_ai.models._batch_model._SUBMITTED_BATCHES_SIZE', 1)
    wrapped = SlowBatchTestModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
    first, second = await asyncio.gather(_pending_response(model), _pending_response(model))
    await model.cancel_suspended_response(first)

    await asyncio.gather(_pending_response(model), _pending_response(model))
    assert model.pool._abandoned == {}  # pyright: ignore[reportPrivateUsage]
    await model.cancel_suspended_response(second)

    assert len(wrapped.created) == 2
    assert wrapped.canceled == []


async def test_results_are_shared_by_the_requests_of_a_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request of a batch reads the results fetched once, and only the most recent batches' are kept."""
    monkeypatch.setattr('pydantic_ai.models._batch_model._RESULTS_CACHE_SIZE', 1)

    @dataclass(init=False)
    class CountingResultsModel(TestModel):
        result_fetches: int = 0

        async def batch_results(self, batch: Batch) -> list[BatchResult]:
            self.result_fetches += 1
            return await super().batch_results(batch)

    wrapped = CountingResultsModel()
    wrapped.result_fetches = 0
    agent = Agent(wrapped, capabilities=[BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=None)])

    await asyncio.gather(agent.run('one'), agent.run('two'))
    await asyncio.gather(agent.run('three'), agent.run('four'))

    assert wrapped.result_fetches == 2


async def test_flushing_an_empty_queue_does_nothing() -> None:
    wrapped = SlowBatchTestModel()
    await BatchModelRequests().flush()
    assert wrapped.created == []


async def test_a_ticket_for_another_response_is_not_waited_on() -> None:
    model = _batched(TestModel(), **FAST_BATCH_SETTINGS)
    pending = await _pending_response(model)
    assert model.continuation_delay(pending) is not None
    assert model.continuation_delay(replace(pending, provider_response_id='resp_other')) is None


def test_a_malformed_batch_ticket_is_not_waited_on() -> None:
    model = _batched(TestModel())
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
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
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
    result = await Agent(wrapped, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)]).run('Hello')

    assert result.output == 'segment 1 segment 2 '


async def test_wrapper_models_forward_the_batch_api() -> None:
    inner = TestModel()
    model = WrapperModel(inner)
    # A model's batch checks are shared with the model it wraps, and not with another instance.
    assert model._batch_credential() == inner._batch_credential() != TestModel()._batch_credential()  # pyright: ignore[reportPrivateUsage]
    request = BatchRequest(custom_id='one', messages=[ModelRequest(parts=[UserPromptPart('Hello')])])

    assert model.batch_max_requests == 10_000
    batch = await model.batch_create([await model.batch_encode(request)])
    assert (await model.batch_status(batch)).status == 'ended'
    assert model._batch_unlisted_status(batch) is None  # pyright: ignore[reportPrivateUsage]
    assert (await model.batch_cancel(batch)).status == 'ended'
    [result] = await model.batch_results(batch)
    assert model.batch_decode(result, request).text == 'success (no tool calls)'


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_a_batch_has_its_own_trace_linked_to_the_requests_in_it(capfire: CaptureLogfire) -> None:
    """Shared batch work is traced once, in the batch's own trace, rather than under whichever run started it.

    The submission is a `batch` span that links to the `chat` span of every request in the batch, and every check is
    a child of it. Each `chat` span records the batch's ID and links back to the `batch` span.
    """
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(
        wrapped,
        capabilities=[Instrumentation(), BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=None)],
    )

    await asyncio.gather(agent.run('one'), agent.run('two'))

    spans = [
        span
        for span in capfire.exporter.exported_spans
        if (span.attributes or {}).get('logfire.span_type') != 'pending_span'
    ]
    [batch] = [span for span in spans if span.name == 'batch test']
    checks = [span for span in spans if span.name == 'batch check test']
    chats = [span for span in spans if span.name == 'chat test']
    assert batch.context is not None
    batch_id = wrapped.created[0].id

    assert batch.parent is None
    assert dict(batch.attributes or {}) == snapshot(
        {
            'gen_ai.provider.name': 'test',
            'gen_ai.system': 'test',
            'gen_ai.request.model': 'test',
            'logfire.span_type': 'span',
            'logfire.msg': 'batch test',
            'pydantic_ai.batch.request_count': 2,
            'pydantic_ai.batch.id': batch_id,
            'pydantic_ai.batch.status': 'in_progress',
            'gen_ai.response.model': 'test',
        }
    )
    assert {link.context.span_id for link in batch.links} == {chat.context.span_id for chat in chats if chat.context}
    assert [
        (check.parent and check.parent.span_id, (check.attributes or {}).get('pydantic_ai.batch.status'))
        for check in checks
    ] == [
        (batch.context.span_id, 'in_progress'),
        (batch.context.span_id, 'ended'),
    ]
    assert not [span for span in spans if 'propagated trace context' in span.name]
    for chat in chats:
        assert (chat.attributes or {})['pydantic_ai.batch.id'] == batch_id
        assert [link.context.span_id for link in chat.links] == [batch.context.span_id]


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_a_resumed_run_checks_the_batch_in_its_trace(capfire: CaptureLogfire) -> None:
    """The pending response records the `batch` span's trace context, so a later run's checks join the batch's trace."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    agent = Agent(
        wrapped,
        capabilities=[Instrumentation(), BatchModelRequests(**FAST_BATCH_SETTINGS)],
        output_type=[str, PendingModelResponse],
    )
    submitted = await agent.run('Hello')
    capfire.exporter.clear()

    # A fresh capability, as in another process.
    agent = Agent(
        wrapped,
        capabilities=[Instrumentation(), BatchModelRequests(**FAST_BATCH_SETTINGS)],
        output_type=[str, PendingModelResponse],
    )
    resumed = await agent.run(message_history=submitted.all_messages())

    assert isinstance(resumed.output, PendingModelResponse)
    ticket = resumed.output.response.provider_details
    assert ticket is not None
    traceparent = ticket['batch']['trace_context']['traceparent']
    batch_trace_id, batch_span_id = (int(part, 16) for part in traceparent.split('-')[1:3])
    spans = [
        span
        for span in capfire.exporter.exported_spans
        if (span.attributes or {}).get('logfire.span_type') != 'pending_span'
    ]
    [check] = [span for span in spans if span.name == 'batch check test']
    [chat] = [span for span in spans if span.name == 'chat test']
    assert check.context is not None and check.parent is not None
    assert (check.context.trace_id, check.parent.span_id) == (batch_trace_id, batch_span_id)
    assert [link.context.span_id for link in chat.links] == [batch_span_id]
    assert [link.context.span_id for link in check.links] == [chat.context and chat.context.span_id]


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_a_batch_submitted_without_tracing_is_still_recorded_on_a_traced_check(capfire: CaptureLogfire) -> None:
    wrapped = SlowBatchTestModel(pending_checks=1)
    pending = await _pending_response(_batched(wrapped, **FAST_BATCH_SETTINGS))
    agent = Agent(wrapped, capabilities=[Instrumentation(), BatchModelRequests(**FAST_BATCH_SETTINGS)])

    await agent.run(message_history=[ModelRequest(parts=[UserPromptPart('Hello')]), pending])

    [chat] = [
        span
        for span in capfire.exporter.exported_spans
        if span.name == 'chat test' and (span.attributes or {}).get('logfire.span_type') == 'span'
    ]
    assert (chat.attributes or {})['pydantic_ai.batch.id'] == wrapped.created[0].id
    assert list(chat.links) == []


async def test_providers_with_the_same_group_and_batch_ids_are_kept_apart() -> None:
    """Groups and batch IDs are only unique within a provider, and one capability can batch several providers."""

    class SameIdsModel(TestModel):
        def __init__(self, system: str, text: str) -> None:
            super().__init__(custom_output_text=text)
            self._system = system

        async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
            return replace(await super().batch_encode(request), group='shared')

        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            batch = await super().batch_create(
                [replace(request, group=self._batch_state.group) for request in requests]
            )
            stored = self._batch_state.batches.pop(batch.id)
            batch = replace(batch, id='batch_1', group='shared')
            self._batch_state.batches[batch.id] = (batch, stored[1])
            return batch

    batching = BatchModelRequests(**FAST_BATCH_SETTINGS)
    first = Agent(SameIdsModel('first', 'from first'), capabilities=[batching])
    second = Agent(SameIdsModel('second', 'from second'), capabilities=[batching])

    results = await asyncio.gather(first.run('Hello'), second.run('Hello'))

    assert [result.output for result in results] == ['from first', 'from second']


@pytest.mark.parametrize('cancelled', [1, 2])
async def test_requests_cancelled_during_submission_are_abandoned_in_their_batch(cancelled: int) -> None:
    """A request cancelled while its batch is being created is in the batch anyway, with nobody waiting on it.

    It's abandoned once the batch exists, so the batch is cancelled once every request in it has been abandoned.
    """
    submitting = asyncio.Event()
    release = asyncio.Event()

    class GatedBatchModel(SlowBatchTestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            submitting.set()
            await release.wait()
            return await super().batch_create(requests)

    wrapped = GatedBatchModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
    requests = [asyncio.create_task(_pending_response(model)) for _ in range(2)]
    await submitting.wait()
    for request in requests[:cancelled]:
        request.cancel()
    release.set()
    for request in requests[:cancelled]:
        with pytest.raises(asyncio.CancelledError):
            await request
    if cancelled == 1:
        # The other request still waits on the batch, until it's abandoned too.
        pending = await requests[1]
        assert wrapped.canceled == []
        await model.cancel_suspended_response(pending)
    else:
        # The cancellation runs in a task of its own.
        with anyio.fail_after(5):
            while not wrapped.canceled:
                await asyncio.sleep(0)
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_request_cancelled_as_its_batch_is_created_is_abandoned_in_it() -> None:
    """Cancelled after the submission handed it its batch, but before it saw it, a request abandons the batch."""
    tasks: list[asyncio.Task[ModelResponse]] = []

    class CancellingBatchModel(SlowBatchTestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            batch = await super().batch_create(requests)
            # Runs before the requests wake up to the batch the submission is about to hand them.
            asyncio.get_running_loop().call_soon(tasks[0].cancel)
            return batch

    wrapped = CancellingBatchModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=1, max_wait=None)
    tasks.append(asyncio.create_task(_pending_response(model)))
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    with anyio.fail_after(5):
        while not wrapped.canceled:
            await asyncio.sleep(0)
    assert wrapped.canceled == [wrapped.created[0].id]


async def test_a_failing_cancel_of_an_abandoned_batch_is_ignored() -> None:
    tasks: list[asyncio.Task[ModelResponse]] = []
    cancel_attempts: list[str] = []
    cancel_attempted = asyncio.Event()

    class FailingCancelModel(SlowBatchTestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            batch = await super().batch_create(requests)
            asyncio.get_running_loop().call_soon(tasks[0].cancel)
            return batch

        async def batch_cancel(self, batch: Batch) -> Batch:
            cancel_attempts.append(batch.id)
            cancel_attempted.set()
            raise ModelHTTPError(status_code=500, model_name='test', body='cancel failed')

    wrapped = FailingCancelModel(pending_checks=1000)
    model = _batched(wrapped, **FAST_POLLING, max_requests=1, max_wait=None)
    tasks.append(asyncio.create_task(_pending_response(model)))
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    with anyio.fail_after(5):
        await cancel_attempted.wait()
    # Let the cancel task finish; an unretrieved exception would fail the test as an unraisable warning.
    for _ in range(5):
        await asyncio.sleep(0)
    assert cancel_attempts == [wrapped.created[0].id]


def test_checks_back_off_without_overflowing_after_many_polls() -> None:
    """A batch resumed from history over and over keeps polling at the maximum interval."""
    delays = [poll_delay(polls, timedelta(seconds=30), timedelta(minutes=10)) for polls in (0, 7, 8, 2_000, 1_000_000)]
    assert delays == snapshot([30.0, 512.578125, 600.0, 600.0, 600.0])


async def test_a_batch_forgets_abandoned_requests_once_it_has_ended() -> None:
    """A request abandoned while the others still wait leaves no bookkeeping behind once the batch's results arrive."""
    wrapped = SlowBatchTestModel(pending_checks=1)
    model = _batched(wrapped, **FAST_POLLING, max_requests=2, max_wait=None)
    first, second = await asyncio.gather(_pending_response(model), _pending_response(model))
    await model.cancel_suspended_response(first)
    assert model.pool._abandoned  # pyright: ignore[reportPrivateUsage]

    prompt = ModelRequest(parts=[UserPromptPart('Hello')])
    response = await model.request([prompt, second], None, ModelRequestParameters())
    while response.state == 'suspended':
        response = await model.request([prompt, response], None, ModelRequestParameters())

    assert model.pool._abandoned == {}  # pyright: ignore[reportPrivateUsage]
    # Abandoning a request of an ended batch records nothing, and cancels nothing.
    await model.cancel_suspended_response(first)
    assert model.pool._abandoned == {}  # pyright: ignore[reportPrivateUsage]
    assert wrapped.canceled == []


@pytest.mark.parametrize('batched', [False, True])
async def test_batch_waits_dont_count_as_generation_continuations(batched: bool) -> None:
    """A batched turn of several provider pauses gets as many continuations as the same turn sent directly."""
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls <= 5:
            return ModelResponse(parts=[TextPart(f'{calls} ')], state='suspended', provider_response_id=f'seg{calls}')
        return ModelResponse(parts=[TextPart('done')], provider_response_id='final')

    capabilities = [BatchModelRequests(**FAST_BATCH_SETTINGS)] if batched else []
    result = await Agent(FunctionModel(respond), capabilities=capabilities).run('Hello')

    assert result.output == '1 2 3 4 5 done'


async def test_shared_failures_with_nobody_waiting_are_retrieved() -> None:
    """A check that fails after every request waiting on it was cancelled doesn't log an unretrieved error."""
    unretrieved: list[str] = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _, context: unretrieved.append(context['message']))
    checking = asyncio.Event()
    submitting = asyncio.Event()
    fail = asyncio.Event()

    class FailingModel(SlowBatchTestModel):
        async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
            if self.created:
                submitting.set()
                await fail.wait()
                raise ModelHTTPError(status_code=500, model_name='test', body='submission failed')
            return await super().batch_create(requests)

        async def batch_status(self, batch: Batch) -> Batch:
            checking.set()
            await fail.wait()
            raise ModelHTTPError(status_code=500, model_name='test', body='check failed')

    try:
        wrapped = FailingModel(pending_checks=1)
        model = _batched(wrapped, **FAST_POLLING, max_requests=1, max_wait=None)
        pending = await _pending_response(model)
        prompt = ModelRequest(parts=[UserPromptPart('Hello')])
        check = asyncio.create_task(model.request([prompt, pending], None, ModelRequestParameters()))
        submission = asyncio.create_task(_pending_response(model))
        await checking.wait()
        await submitting.wait()
        for task in (check, submission):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        fail.set()
        for _ in range(10):
            await asyncio.sleep(0)
        # Their tracebacks would keep the failed futures alive past the collection.
        del check, submission
        gc.collect()
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(None)
    assert unretrieved == []


async def test_model_request_batch_checks_later_batches_without_waiting_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batches after the first ran while it was waited for, so each is checked before waiting on it."""
    sleeps = 0
    real_sleep = anyio.sleep

    async def counting_sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        await real_sleep(0)

    monkeypatch.setattr('pydantic_ai.direct.anyio.sleep', counting_sleep)

    @dataclass(init=False)
    class OneRequestBatchModel(SlowBatchTestModel):
        @property
        def batch_max_requests(self) -> int | None:
            return 1

    model = OneRequestBatchModel(pending_checks=2)
    await model_request_batch(
        model,
        [BatchRequest(custom_id=str(i), messages=[ModelRequest(parts=[UserPromptPart('Hello')])]) for i in range(3)],
        **FAST_POLLING,
    )

    # Only the first batch was waited for: two checks found it in progress, the third ended it, and by then the
    # others had ended too.
    assert (sleeps, model.status_checks) == (3, 5)
