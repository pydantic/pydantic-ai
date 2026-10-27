"""`OnlineEvaluation` under `PrefectDurability`, against the Prefect test harness."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

try:
    from prefect import flow

    from pydantic_ai.durable_exec.prefect import PrefectDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)

try:
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import wait_for_evaluations
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
