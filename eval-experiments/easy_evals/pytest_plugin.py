"""Pytest integration for easy_evals.

`as_pytest(suite)` turns an `EvalSuite` into a parametrized pytest test:

    # test_my_agent.py
    from easy_evals import eval_suite
    from easy_evals.pytest_plugin import as_pytest

    suite = eval_suite(agent)
    suite.case('Capital of France?', expect='Paris')
    suite.case('Write a haiku', judge='is a haiku')

    test_agent = as_pytest(suite)   # -> test_agent[1. Capital of France?], test_agent[2. ...]

The whole suite runs ONCE per test session (concurrently, one `Dataset.evaluate`)
via a real session-scoped fixture, and you get one green/red pytest item per case.
"""

from __future__ import annotations

import inspect
from itertools import count
from typing import Callable, TypeVar

import pytest
from pydantic_evals.reporting import EvaluationReport

from .core import EvalSuite, case_failures

_counter = count(1)
OutputT = TypeVar('OutputT')


def as_pytest(suite: EvalSuite[OutputT], *, name: str | None = None, max_concurrency: int | None = None) -> Callable[..., None]:
    """Build a parametrized pytest test (one item per case) backed by a single session run."""
    fixture_name = f'_easy_evals_report_{name or next(_counter)}'

    @pytest.fixture(scope='session', name=fixture_name)
    def report_fixture() -> EvaluationReport[str, OutputT, None]:
        return suite.report(max_concurrency=max_concurrency)

    @pytest.mark.parametrize('case_name', suite.case_names)
    def test_eval(case_name: str, request: pytest.FixtureRequest) -> None:
        report: EvaluationReport[str, OutputT, None] = request.getfixturevalue(fixture_name)
        rc = next(c for c in report.cases if c.name == case_name)
        if failures := case_failures(rc):
            pytest.fail(f'{case_name} (answer: {rc.output!r})\n' + '\n'.join(failures), pytrace=False)

    # Register the session fixture in the caller's test module so pytest discovers it.
    caller = inspect.stack()[1].frame
    caller.f_globals[fixture_name] = report_fixture
    return test_eval
