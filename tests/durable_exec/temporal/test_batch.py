"""`BatchModel` requests through Temporal's batch queue, end to end through real workflows.

The provider here is a scripted implementation of the public `BatchProvider` protocol, so these tests
pin the engine contract (grouping, deadlines, delivery, errors, cancellation, replay) deterministically.
The Anthropic codec against the real Message Batches API is covered by recorded tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import (
    BatchModel,
    BatchRequest,
    BatchResult,
    BatchResultStatus,
    BatchSettings,
    BatchStatus,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage
from tests.temporal_utils import temporal_dev_server_cache_dir

try:
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import (
        Client,
        Interceptor,
        OutboundInterceptor,
        QueryWorkflowInput,
        SignalWorkflowInput,
        StartWorkflowUpdateInput,
        StartWorkflowUpdateWithStartInput,
        WorkflowFailureError,
        WorkflowHandle,
        WorkflowUpdateFailedError,
        WorkflowUpdateHandle,
    )
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ApplicationError
    from temporalio.service import RPCError, RPCStatusCode
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import (
        Interceptor as WorkerInterceptor,
        Replayer,
        StartActivityInput,
        Worker,
        WorkflowInboundInterceptor,
        WorkflowInterceptorClassInput,
        WorkflowOutboundInterceptor,
    )
    from temporalio.workflow import ActivityConfig, ActivityHandle

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14 '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

pytestmark = pytest.mark.xdist_group(name='temporal-batch')

TASK_QUEUE = 'pydantic-ai-batch-task-queue'
OTHER_TASK_QUEUE = 'pydantic-ai-batch-other-task-queue'
READINESS_WAIT_TIMEOUT = 30
ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60), retry_policy=RetryPolicy(maximum_attempts=1)
)
FAST_POLL = timedelta(milliseconds=50)


@dataclass
class ClientFaults:
    drop_signals: int = 0
    signal_error: RPCStatusCode | None = None
    signal_failures: int = 0
    query_error: RPCStatusCode | None = None
    cancel_error: RPCStatusCode | None = None
    capture: bool = False
    updates: list[tuple[str, str, object]] = field(default_factory=list[tuple[str, str, object]])


faults = ClientFaults()


class FaultInterceptor(Interceptor, WorkerInterceptor):
    def intercept_client(self, next: OutboundInterceptor) -> OutboundInterceptor:
        return FaultOutbound(next)

    def workflow_interceptor_class(self, input: WorkflowInterceptorClassInput) -> type[WorkflowInboundInterceptor]:
        return FastRetryInbound


class FastRetryInbound(WorkflowInboundInterceptor):
    def init(self, outbound: WorkflowOutboundInterceptor) -> None:
        super().init(FastRetryOutbound(outbound))


class FastRetryOutbound(WorkflowOutboundInterceptor):
    def start_activity(self, input: StartActivityInput) -> ActivityHandle[Any]:
        if input.retry_policy is not None:
            input.retry_policy = replace(
                input.retry_policy,
                initial_interval=timedelta(milliseconds=1),
                maximum_interval=timedelta(milliseconds=1),
            )
        return super().start_activity(input)


class FaultOutbound(OutboundInterceptor):
    async def signal_workflow(self, input: SignalWorkflowInput) -> None:
        if input.signal == 'pydantic_ai__batch_result':
            if faults.drop_signals:
                faults.drop_signals -= 1
                return
            if faults.signal_error is not None and faults.signal_failures:
                faults.signal_failures -= 1
                raise RPCError('Injected signal failure', faults.signal_error, b'')
        await super().signal_workflow(input)

    async def query_workflow(self, input: QueryWorkflowInput) -> Any:
        if input.query == 'status' and faults.query_error is not None:
            error, faults.query_error = faults.query_error, None
            raise RPCError('Injected query failure', error, b'')
        return await super().query_workflow(input)

    async def start_workflow_update(self, input: StartWorkflowUpdateInput) -> WorkflowUpdateHandle[Any]:
        if input.update == 'cancel' and faults.cancel_error is not None:
            error, faults.cancel_error = faults.cancel_error, None
            raise RPCError('Injected cancellation failure', error, b'')
        return await super().start_workflow_update(input)

    async def start_update_with_start_workflow(
        self, input: StartWorkflowUpdateWithStartInput
    ) -> WorkflowUpdateHandle[Any]:
        if faults.capture:
            faults.updates.append(
                (input.start_workflow_input.id, input.update_workflow_input.update, input.update_workflow_input.args[0])
            )
        return await super().start_update_with_start_workflow(input)


@dataclass
class ScriptedBatchProvider:
    """A batch API for engine contract tests: it answers each request with an echo of its prompt.

    `hold` keeps every batch processing; `errored_prompts` makes those requests fail at the provider.
    """

    hold: bool = False
    ambiguous_create: bool = False
    hold_create: bool = False
    create_release: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_failures: int = 0
    omit_results: bool = False
    poll_failures: int = 0
    response_status: BatchResultStatus = 'succeeded'
    oversized_results: bool = False
    errored_prompts: set[str] = field(default_factory=set[str])
    batches: dict[str, list[BatchRequest]] = field(default_factory=dict[str, list[BatchRequest]])
    canceled: list[str] = field(default_factory=list[str])
    created: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled_job: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def max_requests(self) -> int:
        return 100

    async def encode(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        *,
        custom_id: str,
    ) -> BatchRequest:
        parts = [part for message in messages if isinstance(message, ModelRequest) for part in message.parts]
        prompt = next(part.content for part in parts if isinstance(part, UserPromptPart))
        tool_returns = sum(isinstance(part, ToolReturnPart) for part in parts)
        return BatchRequest(
            custom_id=custom_id, group='scripted', body=json.dumps({'prompt': prompt, 'tool_returns': tool_returns})
        )

    async def create(self, requests: Sequence[BatchRequest]) -> str:
        batch_id = f'batch_{uuid.uuid4().hex}'
        self.batches[batch_id] = list(requests)
        self.created.set()
        if self.hold_create:
            await self.create_release.wait()
        if self.ambiguous_create:
            raise ModelAPIError('scripted', 'Connection lost after creating the batch')
        return batch_id

    async def status(self, batch_id: str, *, group: str | None = None) -> BatchStatus:
        return 'in_progress' if self.hold else 'ended'

    async def results(self, batch_id: str, *, group: str | None = None) -> list[BatchResult] | None:
        if self.poll_failures:
            self.poll_failures -= 1
            raise ModelAPIError('scripted', 'Transient result fetch failure')
        if self.hold:
            return None
        results: list[BatchResult] = []
        if self.omit_results:
            return results
        for request in self.batches[batch_id]:
            status: BatchResultStatus = (
                'errored' if json.loads(request.body)['prompt'] in self.errored_prompts else self.response_status
            )
            body = request.body if not self.oversized_results else 'x' * 1024
            results.append(BatchResult(custom_id=request.custom_id, status=status, body=body))
        return results

    async def cancel(self, batch_id: str, *, group: str | None = None) -> None:
        self.canceled.append(batch_id)
        self.cancelled_job.set()
        if self.cancel_failures:
            self.cancel_failures -= 1
            raise ModelAPIError('scripted', 'Cancellation endpoint unavailable')

    def decode(
        self,
        result: BatchResult,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        assert result.body is not None
        if result.status in ('expired', 'canceled'):
            raise ModelAPIError('scripted', f'The request was {result.status}')
        body = json.loads(result.body)
        if result.status == 'errored':
            raise ModelHTTPError(status_code=400, model_name='scripted', body={'prompt': body['prompt']})
        if model_request_parameters.function_tools and not body['tool_returns']:
            parts = [ToolCallPart(model_request_parameters.function_tools[0].name, {})]
        else:
            parts = [TextPart(f'Echo: {body["prompt"]}')]
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=10, output_tokens=5), model_name='scripted')

    def batch_ids(self, *prompts: str) -> list[str]:
        """The batches holding any of `prompts`, in submission order."""
        return [
            batch_id
            for batch_id, requests in self.batches.items()
            if any(json.loads(request.body)['prompt'] in prompts for request in requests)
        ]

    def prompts_by_batch(self, *prompts: str) -> list[list[str]]:
        """The prompts of each batch holding any of `prompts`, sorted."""
        return sorted(
            sorted(json.loads(request.body)['prompt'] for request in self.batches[batch_id])
            for batch_id in self.batch_ids(*prompts)
        )


provider = ScriptedBatchProvider()


class ScriptedModel(TestModel):
    @property
    def batch_provider(self) -> ScriptedBatchProvider:
        return provider


def get_weather() -> str:
    return 'sunny'


def batch_agent(name: str, settings: BatchSettings, *, tools: Sequence[Callable[[], str]] = ()) -> Agent[None, str]:
    return Agent(
        BatchModel(ScriptedModel(), batch_settings=settings),
        name=name,
        tools=tools,
        capabilities=[TemporalDurability(activity_config=ACTIVITY_CONFIG)],
    )


SHORT_WAIT = timedelta(milliseconds=100)

pair_agent = batch_agent(
    'batch_pair_agent', BatchSettings(max_requests=2, max_wait=timedelta(hours=1), poll_interval=FAST_POLL)
)
deadline_agent = batch_agent(
    'batch_deadline_agent',
    BatchSettings(max_requests=100, max_wait=SHORT_WAIT, poll_interval=FAST_POLL),
    tools=[get_weather],
)
policy_agent = batch_agent(
    'batch_policy_agent', BatchSettings(max_requests=100, max_wait=SHORT_WAIT, poll_interval=FAST_POLL)
)
# Same agent and model as `policy_agent`, so the same queue, deployed with a different policy.
conflicting_policy_agent = batch_agent(
    'batch_policy_agent', BatchSettings(max_requests=99, max_wait=SHORT_WAIT, poll_interval=FAST_POLL)
)


restart_agent = batch_agent('batch_restart_agent', BatchSettings(max_requests=1, poll_interval=FAST_POLL))


plain_agent = Agent(
    TestModel(), name='batch_plain_agent', capabilities=[TemporalDurability(activity_config=ACTIVITY_CONFIG)]
)


@workflow.defn(failure_exception_types=[ModelAPIError, UserError])
class PairWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        return (await pair_agent.run(prompt)).all_messages()


@workflow.defn(failure_exception_types=[ModelAPIError, UserError])
class DeadlineWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        return (await deadline_agent.run(prompt)).all_messages()


@workflow.defn(failure_exception_types=[ModelAPIError, UserError])
class PolicyWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await policy_agent.run(prompt)).output


@workflow.defn(failure_exception_types=[ModelAPIError, UserError])
class ConflictingPolicyWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await conflicting_policy_agent.run(prompt)).output


@workflow.defn(failure_exception_types=[ModelAPIError, UserError])
class RestartWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await restart_agent.run(prompt)).output


@pytest.fixture
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(download_dest_dir=temporal_dev_server_cache_dir()) as env:
        yield env


@pytest.fixture
def client(temporal_env: WorkflowEnvironment) -> Client:
    config = temporal_env.client.config()
    config['plugins'] = [*config['plugins'], PydanticAIPlugin()]
    config['interceptors'] = [*config['interceptors'], FaultInterceptor()]
    return Client(**config)


@pytest.fixture(autouse=True)
async def workers(client: Client) -> AsyncIterator[None]:
    async with (
        Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[PairWorkflow, DeadlineWorkflow, PolicyWorkflow],
            plugins=[AgentPlugin(pair_agent), AgentPlugin(deadline_agent), AgentPlugin(policy_agent)],
        ),
        Worker(
            client,
            task_queue=OTHER_TASK_QUEUE,
            workflows=[PairWorkflow, ConflictingPolicyWorkflow],
            plugins=[AgentPlugin(pair_agent), AgentPlugin(conflicting_policy_agent)],
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def reset_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch._WAIT_ROUND', timedelta(milliseconds=100))
    faults.capture = False
    faults.updates.clear()
    faults.drop_signals = 0
    faults.signal_error = None
    faults.signal_failures = 0
    faults.query_error = None
    faults.cancel_error = None
    provider.hold = False
    provider.ambiguous_create = False
    provider.hold_create = False
    provider.create_release.clear()
    provider.cancel_failures = 0
    provider.omit_results = False
    provider.poll_failures = 0
    provider.response_status = 'succeeded'
    provider.oversized_results = False
    provider.errored_prompts.clear()
    provider.created.clear()
    provider.cancelled_job.clear()
    provider.canceled.clear()
    yield
    provider.hold = False


def failure_causes(error: BaseException) -> list[BaseException]:
    causes: list[BaseException] = []
    cause: BaseException | None = error
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    return causes


async def test_batch_requests_from_two_workers_share_one_batch(client: Client) -> None:
    """Two runs on two workers fill one batch of `max_requests`, each gets its own result, and usage counts once."""
    # Two agents that batch share one queue workflow class, which each worker must register only once.
    handles = [
        await client.start_workflow(PairWorkflow.run, prompt, id=f'batch-pair-{uuid.uuid4()}', task_queue=task_queue)
        for prompt, task_queue in (('pair one', TASK_QUEUE), ('pair two', OTHER_TASK_QUEUE))
    ]
    first, second = [await handle.result() for handle in handles]

    assert provider.prompts_by_batch('pair one', 'pair two') == [['pair one', 'pair two']]
    for prompt, messages in (('pair one', first), ('pair two', second)):
        # The pending placeholder never reaches history: one request, one merged response.
        assert len(messages) == 2
        response = messages[-1]
        assert isinstance(response, ModelResponse)
        assert response.text == f'Echo: {prompt}'
        assert response.usage == RequestUsage(input_tokens=10, output_tokens=5)
        assert response.state == 'complete'
        assert response.provider_details is not None
        details = response.provider_details['batch']
        assert details['state'] == 'completed'
        assert details['batch_id'] in provider.batches


async def test_batch_is_submitted_when_its_oldest_request_has_waited_max_wait(client: Client) -> None:
    """A lone request is submitted at its deadline, and the tool turn after it goes into a batch round of its own."""
    messages = await client.execute_workflow(
        DeadlineWorkflow.run, 'deadline', id=f'batch-deadline-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )

    assert provider.prompts_by_batch('deadline') == [['deadline'], ['deadline']]
    assert [type(message).__name__ for message in messages] == [
        'ModelRequest',
        'ModelResponse',
        'ModelRequest',
        'ModelResponse',
    ]
    assert isinstance(messages[1], ModelResponse) and isinstance(messages[1].parts[0], ToolCallPart)
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == 'Echo: deadline'


async def test_batch_request_error_is_raised_as_the_model_error(client: Client) -> None:
    provider.errored_prompts.add('fail')
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'fail', id=f'batch-error-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )

    cause = exc_info.value.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == 'ModelHTTPError'
    assert 'status_code: 400' in cause.message


async def test_cancelling_one_run_leaves_the_rest_of_its_batch_alone(
    client: Client, temporal_env: WorkflowEnvironment
) -> None:
    provider.hold = True
    cancelled = await client.start_workflow(
        PairWorkflow.run, 'cancel me', id=f'batch-cancel-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    kept = await client.start_workflow(
        PairWorkflow.run, 'keep me', id=f'batch-keep-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    await cancelled.cancel()
    with temporal_env.auto_time_skipping_disabled():
        with pytest.raises(WorkflowFailureError):
            await cancelled.result()
    provider.hold = False
    messages = await kept.result()

    assert provider.prompts_by_batch('cancel me', 'keep me') == [['cancel me', 'keep me']]
    response = messages[-1]
    assert isinstance(response, ModelResponse) and response.text == 'Echo: keep me'
    assert provider.canceled == []


async def test_cancelling_every_run_of_a_batch_cancels_the_batch(
    client: Client, temporal_env: WorkflowEnvironment
) -> None:
    provider.hold = True
    handle: WorkflowHandle[DeadlineWorkflow, list[ModelMessage]] = await client.start_workflow(
        DeadlineWorkflow.run, 'abandoned', id=f'batch-abandon-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    await handle.cancel()
    with temporal_env.auto_time_skipping_disabled():
        with pytest.raises(WorkflowFailureError):
            await handle.result()
    await asyncio.wait_for(provider.cancelled_job.wait(), READINESS_WAIT_TIMEOUT)

    assert provider.canceled == provider.batch_ids('abandoned')


async def test_workers_with_conflicting_batch_policies_fail_instead_of_splitting_the_queue(client: Client) -> None:
    provider.hold = True
    first = await client.start_workflow(
        PolicyWorkflow.run, 'policy', id=f'batch-policy-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            ConflictingPolicyWorkflow.run,
            'conflict',
            id=f'batch-conflict-{uuid.uuid4()}',
            task_queue=OTHER_TASK_QUEUE,
        )
    provider.hold = False
    assert await first.result() == 'Echo: policy'

    assert any(
        isinstance(cause, ApplicationError) and cause.type == 'UserError' and 'Workers disagree' in cause.message
        for cause in failure_causes(exc_info.value)
    )


async def test_batch_workflow_replays_deterministically(client: Client) -> None:
    handle = await client.start_workflow(
        DeadlineWorkflow.run, 'replay', id=f'batch-replay-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await handle.result()
    history = await handle.fetch_history()

    replay = await Replayer(workflows=[DeadlineWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    assert replay.replay_failure is None


def test_batch_registrations_only_for_agents_that_batch() -> None:
    """Only an agent with a `BatchModel` gains the queue workflow and batch activities, whose names are persisted."""
    batching = TemporalDurability.from_agent(pair_agent)
    plain = TemporalDurability.from_agent(plain_agent)
    assert batching is not None and plain is not None

    batch_activity_names = [
        name
        for activity in batching.temporal_activities
        if (name := ActivityDefinition.must_from_callable(activity).name or '').startswith(  # pyright: ignore[reportUnknownMemberType]
            'agent__batch_pair_agent__batch_'
        )
    ]
    assert batch_activity_names == [
        'agent__batch_pair_agent__batch_enqueue',
        'agent__batch_pair_agent__batch_withdraw',
        'agent__batch_pair_agent__batch_collect',
        'agent__batch_pair_agent__batch_cancel',
        'agent__batch_pair_agent__batch_submit',
        'agent__batch_pair_agent__batch_poll',
        'agent__batch_pair_agent__batch_cancel_job',
    ]
    assert [workflow_class.__name__ for workflow_class in batching.temporal_workflows] == ['BatchQueueWorkflow']

    assert plain.temporal_workflows == []
    assert not any(
        (ActivityDefinition.must_from_callable(activity).name or '').startswith('agent__batch_plain_agent__batch_')  # pyright: ignore[reportUnknownMemberType]
        for activity in plain.temporal_activities
    )


async def test_batch_model_fails_outside_a_workflow_before_queueing() -> None:
    """Outside a workflow the durability capability is transparent, so the `BatchModel` itself refuses the request."""
    with pytest.raises(UserError, match='need a durable execution context with batch support'):
        await pair_agent.run('outside')
    assert provider.batch_ids('outside') == []


async def test_batch_resumes_after_workers_are_replaced(client: Client) -> None:
    provider.hold = True
    queue = 'batch-restart-task-queue'
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RestartWorkflow],
        plugins=[AgentPlugin(restart_agent)],
    ):
        handle = await client.start_workflow(
            RestartWorkflow.run, 'restart', id=f'batch-restart-{uuid.uuid4()}', task_queue=queue
        )
        await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    provider.hold = False
    async with Worker(
        client,
        task_queue=queue,
        workflows=[RestartWorkflow],
        plugins=[AgentPlugin(restart_agent)],
    ):
        assert await handle.result() == 'Echo: restart'
    assert len(provider.batch_ids('restart')) == 1


async def test_uncertain_submission_is_not_retried(client: Client) -> None:
    provider.ambiguous_create = True
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'uncertain', id=f'batch-uncertain-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert len(provider.batch_ids('uncertain')) == 1
    assert any('not resubmitted' in str(error) for error in failure_causes(exc_info.value))


async def test_a_transient_collection_error_does_not_submit_again(client: Client) -> None:
    provider.poll_failures = 1
    messages = await client.execute_workflow(
        DeadlineWorkflow.run, 'retry-poll', id=f'batch-retry-poll-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == 'Echo: retry-poll'
    assert len(provider.batch_ids('retry-poll')) == 2


@pytest.mark.parametrize('status', ['expired', 'canceled'])
async def test_provider_terminal_status_is_reported(client: Client, status: BatchResultStatus) -> None:
    provider.response_status = status
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, status, id=f'batch-terminal-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert any(f'was {status}' in str(error) for error in failure_causes(exc_info.value))


async def test_large_result_is_signalled_by_reference(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch.MAX_REQUEST_BYTES', 200)
    prompt = 'x' * 30
    handle = await client.start_workflow(
        DeadlineWorkflow.run, prompt, id=f'batch-large-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    messages = await handle.result()
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == f'Echo: {prompt}'
    history = await handle.fetch_history()
    signals = [
        event.workflow_execution_signaled_event_attributes
        for event in history.events
        if event.HasField('workflow_execution_signaled_event_attributes')
    ]
    assert signals
    assert all(json.loads(signal.input.payloads[0].data)['result'] is None for signal in signals)


@pytest.mark.parametrize('failure', ['dropped', 'not-found', 'unavailable'])
async def test_result_delivery_recovers_from_signal_failures(client: Client, failure: str) -> None:
    if failure == 'dropped':
        faults.drop_signals = 1
    else:
        faults.signal_error = RPCStatusCode.NOT_FOUND if failure == 'not-found' else RPCStatusCode.UNAVAILABLE
        faults.signal_failures = 1 if failure == 'not-found' else 3
    messages = await client.execute_workflow(
        DeadlineWorkflow.run, failure, id=f'batch-delivery-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == f'Echo: {failure}'
    assert len(provider.batch_ids(failure)) == 2


@pytest.mark.parametrize('status', [RPCStatusCode.NOT_FOUND, RPCStatusCode.UNAVAILABLE])
async def test_collection_reports_an_unreachable_queue(client: Client, status: RPCStatusCode) -> None:
    faults.drop_signals = 1
    faults.query_error = status
    faults.cancel_error = status
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'missing-queue', id=f'batch-missing-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    expected = 'lost' if status == RPCStatusCode.NOT_FOUND else 'Injected query failure'
    assert any(expected in str(error) for error in failure_causes(exc_info.value))


async def test_completed_records_do_not_consume_the_queued_request_limit(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch_queue._MAX_ITEMS', 2)
    for number in range(3):
        handles = [
            await client.start_workflow(
                PairWorkflow.run,
                f'capacity-{number}-{item}',
                id=f'batch-capacity-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )
            for item in range(2)
        ]
        for handle in handles:
            messages = await handle.result()
            assert isinstance(messages[-1], ModelResponse) and (messages[-1].text or '').startswith('Echo: capacity-')


async def test_queue_rejects_an_oversized_request_before_submission(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch.MAX_REQUEST_BYTES', 100)
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'too-large', id=f'batch-size-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert any('over the 100-byte limit' in str(error) for error in failure_causes(exc_info.value))
    assert provider.batch_ids('too-large') == []


async def test_a_full_queue_does_not_submit_a_rejected_request(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch_queue._MAX_QUEUE_BYTES', 100)
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'queue-full', id=f'batch-full-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert any('batch queue is full' in str(error) for error in failure_causes(exc_info.value))
    assert provider.batch_ids('queue-full') == []


async def test_duplicate_enqueue_and_late_result_delivery_are_idempotent(
    client: Client, temporal_env: WorkflowEnvironment
) -> None:
    from pydantic_core import to_json

    provider.hold = True
    faults.capture = True
    first = await client.start_workflow(
        PairWorkflow.run, 'idempotent-first', id=f'idem-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    second = await client.start_workflow(
        PairWorkflow.run, 'idempotent-second', id=f'idem-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    enqueue = [(queue_id, json.loads(to_json(arg))) for queue_id, name, arg in faults.updates if name == 'enqueue']
    assert len(enqueue) == 2
    request = next(arg for _, arg in enqueue if json.loads(arg['request']['body'])['prompt'] == 'idempotent-first')
    manager = client.get_workflow_handle(enqueue[0][0])
    await manager.execute_update('enqueue', request)  # pyright: ignore[reportUnknownMemberType]
    await first.cancel()
    with temporal_env.auto_time_skipping_disabled():
        with pytest.raises(WorkflowFailureError):
            await first.result()
    with pytest.raises(WorkflowUpdateFailedError) as exc_info:
        await manager.execute_update('enqueue', request)  # pyright: ignore[reportUnknownMemberType]
    assert isinstance(exc_info.value.cause, ApplicationError) and exc_info.value.cause.type == 'BatchRequestCancelled'
    await second.signal('pydantic_ai__batch_result', {'item_id': 'unknown'})
    provider.hold = False
    messages = await second.result()
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == 'Echo: idempotent-second'
    assert len(provider.batch_ids('idempotent-first', 'idempotent-second')) == 1


async def test_manager_rotates_history_with_pending_jobs(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch_queue._MAX_HISTORY_EVENTS', 10)
    faults.capture = True
    handle = await client.start_workflow(
        DeadlineWorkflow.run, 'rotate', id=f'rotate-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    messages = await handle.result()
    assert isinstance(messages[-1], ModelResponse) and messages[-1].text == 'Echo: rotate'
    queue_id = next(queue_id for queue_id, name, _ in faults.updates if name == 'enqueue')
    latest = await client.get_workflow_handle(queue_id).fetch_history()
    original_id = latest.events[0].workflow_execution_started_event_attributes.first_execution_run_id
    original = await client.get_workflow_handle(queue_id, run_id=original_id).fetch_history()
    assert any(event.HasField('workflow_execution_continued_as_new_event_attributes') for event in original.events)
    durability = TemporalDurability.from_agent(deadline_agent)
    assert durability is not None
    replay = await Replayer(workflows=durability.temporal_workflows, plugins=[PydanticAIPlugin()]).replay_workflow(
        original
    )
    assert replay.replay_failure is None


async def test_missing_provider_item_is_reported(client: Client) -> None:
    provider.omit_results = True
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'missing-item', id=f'missing-item-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert any('ended without a result' in str(error) for error in failure_causes(exc_info.value))


async def test_repeated_poll_failures_terminate_the_wait(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._batch_queue._MAX_POLL_FAILURES', 2)
    provider.poll_failures = 6
    with pytest.raises(WorkflowFailureError) as exc_info:
        await client.execute_workflow(
            DeadlineWorkflow.run, 'poll-failure', id=f'poll-failure-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
    assert any('Collecting batch' in str(error) for error in failure_causes(exc_info.value))


async def test_cancellation_during_provider_submission_cancels_the_created_job(
    client: Client, temporal_env: WorkflowEnvironment
) -> None:
    provider.hold_create = True
    handle = await client.start_workflow(
        DeadlineWorkflow.run, 'cancel-submit', id=f'cancel-submit-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await temporal_env.sleep(timedelta(seconds=1))
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    await handle.cancel()
    with temporal_env.auto_time_skipping_disabled():
        with pytest.raises(WorkflowFailureError):
            await handle.result()
    provider.create_release.set()
    await temporal_env.sleep(timedelta(seconds=1))
    await asyncio.wait_for(provider.cancelled_job.wait(), READINESS_WAIT_TIMEOUT)
    assert provider.canceled == provider.batch_ids('cancel-submit')


async def test_failed_provider_cancellation_does_not_leave_a_run_waiting(
    client: Client, temporal_env: WorkflowEnvironment
) -> None:
    provider.hold = True
    provider.cancel_failures = 3
    handle = await client.start_workflow(
        DeadlineWorkflow.run, 'cancel-failure', id=f'cancel-failure-{uuid.uuid4()}', task_queue=TASK_QUEUE
    )
    await temporal_env.sleep(timedelta(seconds=1))
    await asyncio.wait_for(provider.created.wait(), READINESS_WAIT_TIMEOUT)
    await handle.cancel()
    with temporal_env.auto_time_skipping_disabled():
        with pytest.raises(WorkflowFailureError):
            await handle.result()
    await temporal_env.sleep(timedelta(seconds=30))
    assert len(provider.canceled) == 3
