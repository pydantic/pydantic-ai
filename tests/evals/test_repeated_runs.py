from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.reporting import ReportCase, ReportCaseMultiRun

    from .utils import render_table


pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


class Input(BaseModel):
    val: int


class Output(BaseModel):
    val: int


@dataclass
class RandomScore(Evaluator[Input, Output, Any]):
    """Evaluator that returns a score based on output."""

    def evaluate(self, ctx: EvaluatorContext[Input, Output, Any]) -> float:
        return float(ctx.output.val)


async def test_repeated_runs_success():
    """Test running a dataset with runs > 1."""

    # Task that returns the input value
    async def task(input: Input) -> Output:
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[
            Case(name='c1', inputs=Input(val=10)),
            Case(name='c2', inputs=Input(val=20)),
        ],
        evaluators=[RandomScore()],
    )

    report = await dataset.evaluate(task, runs=3)

    assert len(report.cases) == 2

    # Check first case
    c1 = report.cases[0]
    assert isinstance(c1, ReportCaseMultiRun)
    assert c1.name == 'c1'
    assert len(c1.runs) == 3
    # Check that individual runs are correct
    for run in c1.runs:
        assert isinstance(run, ReportCase)
        assert run.output.val == 10
        assert run.scores['RandomScore'].value == 10.0

    # Check aggregate
    assert c1.aggregate.scores['RandomScore'] == 10.0

    # Check second case
    c2 = report.cases[1]
    assert isinstance(c2, ReportCaseMultiRun)
    assert len(c2.runs) == 3
    assert c2.aggregate.scores['RandomScore'] == 20.0


async def test_repeated_runs_mixed_failures():
    """Test repeated runs where some fail."""

    count = 0

    async def task(input: Input) -> Output:
        nonlocal count
        count += 1
        # Fail on every 2nd call
        if count % 2 == 0:
            raise ValueError('Flaky failure')
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[
            Case(name='c1', inputs=Input(val=10)),
        ],
        evaluators=[RandomScore()],
    )

    # Run 4 times: 1(ok), 2(fail), 3(ok), 4(fail)
    report = await dataset.evaluate(task, runs=4)

    # We expect 2 failures and 1 MultiRun (with 2 successes)
    assert len(report.failures) == 2
    assert len(report.cases) == 1

    c1 = report.cases[0]
    assert isinstance(c1, ReportCaseMultiRun)
    assert c1.name == 'c1'
    assert len(c1.runs) == 2  # 2 successful runs

    # Check aggregates are based on successes
    assert c1.aggregate.scores['RandomScore'] == 10.0


async def test_repeated_runs_all_failures():
    """Test repeated runs where all fail."""

    async def task(input: Input) -> Output:
        raise ValueError('Always fail')

    dataset = Dataset[Input, Output, Any](
        cases=[
            Case(name='c1', inputs=Input(val=10)),
        ],
    )

    report = await dataset.evaluate(task, runs=3)

    assert len(report.cases) == 0
    assert len(report.failures) == 3
    assert report.failures[0].error_message == 'ValueError: Always fail'


async def test_single_run_compat():
    """Test that runs=1 (default) still produces ReportCase."""

    async def task(input: Input) -> Output:
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[
            Case(name='c1', inputs=Input(val=10)),
        ],
    )

    report = await dataset.evaluate(task)  # Default runs=1

    assert len(report.cases) == 1
    assert isinstance(report.cases[0], ReportCase)
    # Should NOT be ReportCaseMultiRun
    assert not isinstance(report.cases[0], ReportCaseMultiRun)


def test_evaluate_sync_repeated():
    """Test evaluate_sync with runs > 1."""

    def task(input: Input) -> Output:
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[Case(name='c1', inputs=Input(val=10))],
        evaluators=[RandomScore()],
    )

    report = dataset.evaluate_sync(task, runs=2)

    assert len(report.cases) == 1
    assert isinstance(report.cases[0], ReportCaseMultiRun)
    assert len(report.cases[0].runs) == 2


async def test_rendering_repeated_runs():
    """Test rendering of a report with repeated runs."""

    async def task(input: Input) -> Output:
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[Case(name='c1', inputs=Input(val=10))],
        evaluators=[RandomScore()],
    )

    report = await dataset.evaluate(task, runs=2, name='repeated_test')

    table = report.console_table()
    rendered = render_table(table)

    # Check for the multi-run indicator in the Case ID column
    assert 'c1 (2 runs)' in rendered
    # Check for score
    assert 'RandomScore: 10.0' in rendered


async def test_rendering_diff_single_vs_multi():
    """Test rendering a diff between a single run baseline and multi-run new report."""

    async def task(input: Input) -> Output:
        return Output(val=input.val)

    dataset = Dataset[Input, Output, Any](
        cases=[Case(name='c1', inputs=Input(val=10))],
        evaluators=[RandomScore()],
    )

    # Baseline: single run
    baseline = await dataset.evaluate(task, runs=1, name='baseline')

    # New: 2 runs
    new_report = await dataset.evaluate(task, runs=2, name='new')

    table = new_report.console_table(baseline=baseline)
    rendered = render_table(table)

    # Check that it renders a diff
    # The score should be identical (10.0 -> 10.0)
    assert 'RandomScore: 10.0' in rendered
