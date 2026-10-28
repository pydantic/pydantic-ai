"""`OnlineEvaluation` under `PrefectDurability`, against the Prefect test harness.

Not VCR tests: the behavior under test is what the dispatch task hands evaluators and how its failure
reaches the flow, which a provider recording could not show.
"""

from __future__ import annotations

from collections.abc import Iterator

import logfire
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits

try:
    from prefect import flow

    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)

try:
    from pydantic_evals.online import OnlineEvaluator, wait_for_evaluations
    from pydantic_evals.online_capability import OnlineEvaluation

    from .online_evaluation_helpers import Answer, RecordRun, evaluated, failing_sample_rate, seen
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic_evals not installed', allow_module_level=True)


pytestmark = [pytest.mark.xdist_group(name='prefect'), pytest.mark.usefixtures('prefect_test_server')]


@pytest.fixture(autouse=True)
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """Prefect's `@flow` constructor synchronously inspects its decorated function's source."""
    return ('pydantic_ai.durable_exec.prefect',)


@pytest.fixture(autouse=True)
def setup_logfire_instrumentation() -> Iterator[None]:
    # Recording the evaluators' span tree needs a configured tracer provider. Prefect sets the
    # `traceparent` header, so distributed tracing is enabled explicitly to avoid the warning.
    logfire.configure(metrics=False, distributed_tracing=False)
    yield


@pytest.fixture(autouse=True)
def clear_evaluated() -> None:
    evaluated.clear()


async def test_prefect_online_evaluation_sees_live_values() -> None:
    """Prefect tasks run in the flow's process, so the dispatch task hands evaluators the live output and span tree, once."""
    agent = Agent(
        TestModel(),
        name='prefect_online_evaluation_agent',
        output_type=Answer,
        capabilities=[PrefectDurability[object](), OnlineEvaluation(id='online_evaluation', evaluators=[RecordRun()])],
    )

    @flow
    async def run_flow() -> str:
        return (await agent.run('go')).run_id

    run_id = await run_flow()
    await wait_for_evaluations()
    assert seen() == [('record', run_id, Answer(text='a'))]
    [(_, context)] = evaluated
    # The model request ran as a Prefect task inside the run, so the span tree shows it.
    assert context.span_tree.any({'name_contains': 'Model Request'})


async def test_prefect_online_evaluation_dispatch_failure() -> None:
    """A failed run whose dispatch task fails keeps its own error, with a warning; a completed run fails with it."""
    agent = Agent(
        TestModel(),
        name='prefect_online_evaluation_failing',
        capabilities=[
            PrefectDurability[object](),
            OnlineEvaluation(
                id='online_evaluation',
                evaluators=[
                    OnlineEvaluator(evaluator=RecordRun(), sample_rate=failing_sample_rate, run_on_errors=True)
                ],
            ),
        ],
    )

    @flow
    async def failed_run_flow() -> None:
        await agent.run('go', usage_limits=UsageLimits(request_limit=0))

    @flow
    async def completed_run_flow() -> None:
        await agent.run('go')

    with (
        pytest.warns(UserWarning, match='Online evaluation of the failed agent run was not dispatched'),
        pytest.raises(UsageLimitExceeded),
    ):
        await failed_run_flow()
    with pytest.raises(RuntimeError, match='sampling failed'):
        await completed_run_flow()
    assert evaluated == []


async def test_prefect_online_evaluation_failed_run_hands_evaluators_the_live_exception() -> None:
    """A failed run's `run_on_errors` evaluators get the run's own exception, as the task runs in the flow's process."""
    agent = Agent(
        TestModel(),
        name='prefect_online_evaluation_failed_run_evaluated',
        capabilities=[
            PrefectDurability[object](),
            OnlineEvaluation(
                id='online_evaluation', evaluators=[OnlineEvaluator(evaluator=RecordRun(), run_on_errors=True)]
            ),
        ],
    )

    @flow
    async def failed_run_flow() -> None:
        await agent.run('go', usage_limits=UsageLimits(request_limit=0))

    with pytest.raises(UsageLimitExceeded) as exc_info:
        await failed_run_flow()
    await wait_for_evaluations()
    [(_, context)] = evaluated
    assert isinstance(context.output, UsageLimitExceeded)
    assert str(context.output) == str(exc_info.value)
