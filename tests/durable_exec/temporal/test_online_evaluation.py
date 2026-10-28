"""`OnlineEvaluation` under Temporal: each run's evaluators are dispatched once, however often the workflow replays.

Not VCR tests: the behavior under test is what Temporal records, replays and retries, and what crosses its
data converter, which a provider recording could not show.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from datetime import timedelta

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

# The 3.14 durable-exec CI leg takes this skip; every other leg falls through.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14 '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    try:
        from pydantic_evals.evaluators import Evaluator, EvaluatorContext
        from pydantic_evals.online import OnlineEvaluator, SamplingContext, wait_for_evaluations
        from pydantic_evals.online_capability import OnlineEvaluation
        from pydantic_evals.otel import SpanTreeRecordingError
    except ImportError:  # pragma: lax no cover
        pytest.skip('pydantic_evals not installed', allow_module_level=True)

    from ._shared import workflow_activity_raises

pytestmark = [pytest.mark.xdist_group(name='temporal-online-evaluation')]

TASK_QUEUE = 'online-evaluation-task-queue'


class Answer(BaseModel):
    text: str


_evaluated: list[tuple[str, str | None, object]] = []
_span_tree_errors: list[str] = []


@dataclass
class RecordRun(Evaluator):
    label: str

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = ctx.output
        if isinstance(output, BaseException):
            # The message's first sentence; the rest links to the docs.
            output = f'{type(output).__name__}: {output}'.split('.')[0]
        _evaluated.append((self.label, ctx.name, output))
        try:
            ctx.span_tree
        except SpanTreeRecordingError as e:
            _span_tree_errors.append(str(e))
        return True


_agent = Agent(
    TestModel(),
    name='online_evaluation_agent',
    output_type=Answer,
    capabilities=[
        TemporalDurability(),
        OnlineEvaluation(
            id='online_evaluation',
            evaluators=[
                OnlineEvaluator(evaluator=RecordRun('always'), run_on_errors=True),
                OnlineEvaluator(evaluator=RecordRun('sampled'), sample_rate=0.5),
            ],
        ),
    ],
)


@workflow.defn
class OnlineEvaluationWorkflow:
    @workflow.run
    async def run(self) -> list[str]:
        run_ids: list[str] = []
        for prompt in ['one', 'two']:
            result = await _agent.run(prompt)
            run_ids.append(result.run_id)
        try:
            await _agent.run('three', usage_limits=UsageLimits(request_limit=0))
        except UsageLimitExceeded:
            pass
        return run_ids


async def test_online_evaluation_dispatches_once_across_replays(client: Client):
    """With workflow caching off, every workflow task replays the runs before it, and the `Replayer` replays them all.

    Before the dispatch was a durable operation, each replay ran the earlier runs' evaluators again.
    """
    _evaluated.clear()
    _span_tree_errors.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[OnlineEvaluationWorkflow],
        plugins=[AgentPlugin(_agent)],
        # Unsandboxed so evaluators that (before the fix) ran in workflow code record into this module's list.
        workflow_runner=UnsandboxedWorkflowRunner(),
        max_cached_workflows=0,
    ):
        handle = await client.start_workflow(
            OnlineEvaluationWorkflow.run, id=f'online-evaluation-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
        run_ids = await handle.result()
        await wait_for_evaluations()

    always = [(name, output) for label, name, output in _evaluated if label == 'always']
    # The output crossed Temporal's data converter, so the evaluator sees its JSON form, and the
    # exception that failed the third run as an `Exception` naming its type.
    assert [output for _, output in always] == [
        {'text': 'a'},
        {'text': 'a'},
        'Exception: UsageLimitExceeded: The next request would exceed the request_limit of 0',
    ]
    assert [name for name, _ in always[:2]] == run_ids
    # The span tree doesn't cross the data converter either.
    assert set(_span_tree_errors) == {
        "The span tree is not available because the run's evaluation context crossed a durable "
        'execution boundary that cannot carry it.'
    }
    assert len(_span_tree_errors) == len(_evaluated)
    sampled_names = [name for label, name, _ in _evaluated if label == 'sampled']
    assert len(sampled_names) == len(set(sampled_names))
    assert set(sampled_names) <= set(run_ids)

    evaluated_before_replay = list(_evaluated)
    await Replayer(
        workflows=[OnlineEvaluationWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())
    await wait_for_evaluations()
    assert _evaluated == evaluated_before_replay


_sampling_attempts: list[str] = []


def _failing_sample_rate(ctx: SamplingContext) -> bool:
    _sampling_attempts.append(str(ctx.inputs))
    raise RuntimeError('sampling failed')


_failing_agent = Agent(
    TestModel(),
    name='online_evaluation_failing_agent',
    capabilities=[
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(milliseconds=1)),
            )
        ),
        OnlineEvaluation(
            id='online_evaluation',
            evaluators=[
                OnlineEvaluator(evaluator=RecordRun('failing'), sample_rate=_failing_sample_rate, run_on_errors=True)
            ],
        ),
    ],
)


@workflow.defn
class FailingDispatchWorkflow:
    @workflow.run
    async def run(self) -> None:
        try:
            await _failing_agent.run('failed run', usage_limits=UsageLimits(request_limit=0))
        except UsageLimitExceeded:
            pass
        await _failing_agent.run('completed run')


async def test_online_evaluation_dispatch_failure_is_retried_per_activity_retry_policy(client: Client):
    """A dispatch whose sampling raises is retried up to the activity's `maximum_attempts`.

    A failed run keeps its own exception, with a warning; a completed run fails with the dispatch error.
    """
    _evaluated.clear()
    _sampling_attempts.clear()
    with pytest.warns(UserWarning, match='Online evaluation of the failed agent run was not dispatched'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[FailingDispatchWorkflow],
            plugins=[AgentPlugin(_failing_agent)],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            with workflow_activity_raises(RuntimeError, 'sampling failed'):
                await client.execute_workflow(
                    FailingDispatchWorkflow.run, id=f'online-evaluation-failing-{uuid.uuid4()}', task_queue=TASK_QUEUE
                )
    assert _sampling_attempts == ['failed run'] * 3 + ['completed run'] * 3
    assert _evaluated == []
