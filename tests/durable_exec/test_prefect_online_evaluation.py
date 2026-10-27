"""`OnlineEvaluation` under `PrefectDurability`, against the Prefect test harness."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

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
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import OnlineEvaluator, SamplingContext, wait_for_evaluations
    from pydantic_evals.online_capability import OnlineEvaluation
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic_evals not installed', allow_module_level=True)


pytestmark = [pytest.mark.xdist_group(name='prefect'), pytest.mark.usefixtures('prefect_test_server')]


@pytest.fixture(autouse=True)
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """Prefect's `@flow` constructor synchronously inspects its decorated function's source."""
    return ('pydantic_ai.durable_exec.prefect',)


class Answer(BaseModel):
    text: str


async def test_prefect_online_evaluation_sees_live_output() -> None:
    """Prefect tasks run in the flow's process, so the dispatch task hands evaluators the live output, once."""
    evaluated: list[tuple[str | None, object]] = []

    @dataclass
    class RecordRun(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            evaluated.append((ctx.name, ctx.output))
            return True

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
    assert evaluated == [(run_id, Answer(text='a'))]


def _failing_sample_rate(ctx: SamplingContext) -> bool:
    raise RuntimeError('sampling failed')


@dataclass
class NeverRun(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        raise AssertionError('not sampled')  # pragma: no cover


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
                    OnlineEvaluator(evaluator=NeverRun(), sample_rate=_failing_sample_rate, run_on_errors=True)
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
