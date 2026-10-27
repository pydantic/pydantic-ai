"""`OnlineEvaluation` under `DBOSDurability`, against a real DBOS runtime (sqlite).

Not VCR tests: the behavior under test is which dispatch steps a workflow records and what a forked
re-execution replays, which only the DBOS system database can show.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass

import logfire
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

from ..conftest import detach_dbos_logging

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)

try:
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import OnlineEvaluator, SamplingContext, wait_for_evaluations
    from pydantic_evals.online_capability import OnlineEvaluation
    from pydantic_evals.otel.span_tree import SpanTree
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic_evals not installed', allow_module_level=True)


pytestmark = pytest.mark.xdist_group(name='dbos')


@pytest.fixture(scope='module')
async def dbos(tmp_path_factory: pytest.TempPathFactory) -> AsyncGenerator[DBOS]:
    # An async module-scoped fixture keeps one event loop for the module: DBOS binds its async
    # machinery to the loop it is launched under, and a per-test loop would leave it on a closed one.
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_dbos_online_eval',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        # DBOS steps record spans, which the evaluators' span tree shows.
        'enable_otlp': True,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()
        detach_dbos_logging()


@pytest.fixture(autouse=True, scope='module')
def setup_logfire_instrumentation() -> Iterator[None]:
    # Recording the evaluators' span tree needs a configured tracer provider.
    logfire.configure(metrics=False)
    yield


class Answer(BaseModel):
    text: str


def _failing_sample_rate(ctx: SamplingContext) -> bool:
    raise RuntimeError('sampling failed')


async def test_dbos_online_evaluation_dispatches_once_on_replay(dbos: DBOS) -> None:
    """Replaying the workflow returns the recorded dispatch step, so the run's evaluators don't run again.

    Forking from past the last step re-executes the workflow function against every recorded step,
    the way recovery does. Before the dispatch was a step, each replay ran the evaluators again.
    """
    evaluated: list[tuple[str | None, object]] = []
    span_trees: list[SpanTree] = []

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            evaluated.append((ctx.name, ctx.output))
            span_trees.append(ctx.span_tree)
            return True

    agent = Agent(
        TestModel(),
        name='dbos_online_evaluation',
        output_type=Answer,
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(id='online_evaluation', evaluators=[RecordRun()]),
        ],
    )

    @DBOS.workflow()
    async def run_workflow() -> str:
        return (await agent.run('run')).run_id

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        run_id = await run_workflow()
    await wait_for_evaluations()
    # DBOS steps run in the workflow's process, so evaluators see the live output and span tree.
    assert evaluated == [(run_id, Answer(text='a'))]
    [span_tree] = span_trees
    assert span_tree.any({'name_equals': 'dbos_online_evaluation__model.request'})

    steps = await dbos.list_workflow_steps_async(workflow_id)
    fork = await dbos.fork_workflow_async(workflow_id, max(step['function_id'] for step in steps) + 1)
    await fork.get_result()
    await wait_for_evaluations()
    assert evaluated == [(run_id, Answer(text='a'))]


async def test_dbos_online_evaluation_dispatch_step_only_when_an_evaluator_may_be_sampled(dbos: DBOS) -> None:
    """Sampling happens inside the dispatch step, which a run skips only when no evaluator can ever be sampled."""
    evaluated: list[str | None] = []

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            evaluated.append(ctx.name)  # pragma: no cover
            return True  # pragma: no cover

    never_sampled = Agent(
        TestModel(),
        name='dbos_online_evaluation_never_sampled',
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(
                id='online_evaluation', evaluators=[OnlineEvaluator(evaluator=RecordRun(), sample_rate=0)]
            ),
        ],
    )
    not_sampled = Agent(
        TestModel(),
        name='dbos_online_evaluation_not_sampled',
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(
                id='online_evaluation',
                evaluators=[OnlineEvaluator(evaluator=RecordRun(), sample_rate=lambda _: False)],
            ),
        ],
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await never_sampled.run('run')
        await not_sampled.run('run')

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        await run_workflow()
    await wait_for_evaluations()
    assert evaluated == []
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps if step['function_name'].endswith('.dispatch')] == [
        'dbos_online_evaluation_not_sampled__capability__online_evaluation.dispatch'
    ]


async def test_dbos_online_evaluation_dispatch_failure_keeps_run_error(dbos: DBOS) -> None:
    """When dispatching a failed run's evaluation fails, the run's own exception still reaches the caller."""

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return True  # pragma: no cover

    agent = Agent(
        TestModel(),
        name='dbos_online_evaluation_dispatch_failure',
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(
                id='online_evaluation',
                evaluators=[
                    OnlineEvaluator(evaluator=RecordRun(), sample_rate=_failing_sample_rate, run_on_errors=True)
                ],
            ),
        ],
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await agent.run('run', usage_limits=UsageLimits(request_limit=0))

    with SetWorkflowID(str(uuid.uuid4())):
        with (
            pytest.warns(UserWarning, match='Online evaluation of the failed agent run was not dispatched'),
            pytest.raises(UsageLimitExceeded),
        ):
            await run_workflow()


async def test_dbos_online_evaluation_dispatch_failure_fails_completed_run(dbos: DBOS) -> None:
    """When dispatching a completed run's evaluation fails, the run fails with the dispatch error, without a warning."""

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return True  # pragma: no cover

    agent = Agent(
        TestModel(),
        name='dbos_online_evaluation_completed_run_dispatch_failure',
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(
                id='online_evaluation',
                evaluators=[OnlineEvaluator(evaluator=RecordRun(), sample_rate=_failing_sample_rate)],
            ),
        ],
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await agent.run('run')

    with SetWorkflowID(str(uuid.uuid4())):
        with pytest.raises(RuntimeError, match='sampling failed'):
            await run_workflow()


async def test_dbos_online_evaluation_failed_run_without_run_on_errors_records_no_step(dbos: DBOS) -> None:
    """A failed run whose evaluators don't run on errors records no dispatch step and keeps its own error."""

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return True  # pragma: no cover

    agent = Agent(
        TestModel(),
        name='dbos_online_evaluation_failed_run',
        capabilities=[DBOSDurability[object](), OnlineEvaluation(id='online_evaluation', evaluators=[RecordRun()])],
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await agent.run('run', usage_limits=UsageLimits(request_limit=0))

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        with pytest.raises(UsageLimitExceeded):
            await run_workflow()
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps if step['function_name'].endswith('.dispatch')] == []


async def test_dbos_online_evaluation_dispatches_each_capability_in_its_own_step(dbos: DBOS) -> None:
    """Several `OnlineEvaluation` capabilities on one durable agent each record their own dispatch step."""
    evaluated: list[tuple[str, str | None]] = []

    @dataclass
    class RecordRun(Evaluator):
        label: str

        def evaluate(self, ctx: EvaluatorContext) -> bool:
            evaluated.append((self.label, ctx.name))
            return True

    agent = Agent(
        TestModel(),
        name='dbos_online_evaluation_several',
        capabilities=[
            DBOSDurability[object](),
            OnlineEvaluation(id='outer_evaluation', evaluators=[RecordRun('outer')]),
            OnlineEvaluation(id='inner_evaluation', evaluators=[RecordRun('inner')]),
        ],
    )

    @DBOS.workflow()
    async def run_workflow() -> str:
        return (await agent.run('run')).run_id

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        run_id = await run_workflow()
    await wait_for_evaluations()
    assert sorted(evaluated) == [('inner', run_id), ('outer', run_id)]
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert sorted(step['function_name'] for step in steps if step['function_name'].endswith('.dispatch')) == [
        'dbos_online_evaluation_several__capability__inner_evaluation.dispatch',
        'dbos_online_evaluation_several__capability__outer_evaluation.dispatch',
    ]

    fork = await dbos.fork_workflow_async(workflow_id, max(step['function_id'] for step in steps) + 1)
    await fork.get_result()
    await wait_for_evaluations()
    assert len(evaluated) == 2
