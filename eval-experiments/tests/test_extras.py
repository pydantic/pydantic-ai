"""Tests for the extra features: repeat, baseline compare, and LLM case generation."""

from __future__ import annotations

import pytest

from easy_evals import eval_suite
from fake_models import qa_agent
from pydantic_evals import Case, Dataset


def test_repeat_runs_each_case_n_times() -> None:
    suite = eval_suite(qa_agent())
    suite.case('What is the capital of France?', expect='Paris')
    suite.case('What is 2 + 2?', contains='4')
    assert len(suite.report(repeat=1).cases) == 2
    assert len(suite.report(repeat=3).cases) == 6  # 2 cases x 3 repeats


def test_baseline_compare_runs_without_error() -> None:
    suite = eval_suite(qa_agent())
    suite.case('What is the capital of France?', expect='Paris')
    before = suite.report()
    # Diffing against a baseline report should run and still return overall pass/fail.
    assert suite.run(baseline=before) is True


def test_generate_adds_mapped_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate(
        *, dataset_type: object, n_examples: int, extra_instructions: str | None = None, **_: object
    ) -> Dataset[str, str, None]:
        cases = [Case(name=f'gen-{i}', inputs=f'q{i}?', expected_output=f'a{i}') for i in range(n_examples)]
        return Dataset(name='generated', cases=cases)

    monkeypatch.setattr('pydantic_evals.generation.generate_dataset', fake_generate)

    suite = eval_suite(qa_agent())
    suite.case('seed prompt', expect='x')
    suite.generate(n=3, about='trivia questions')

    assert suite.case_names == ['1. seed prompt', 'gen-0', 'gen-1', 'gen-2']
    # prompts map from the generated `inputs` (verified via the public dataset)
    assert [c.inputs for c in suite.to_dataset().cases] == ['seed prompt', 'q0?', 'q1?', 'q2?']


def test_generated_cases_are_runnable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_generate(
        *, dataset_type: object, n_examples: int, extra_instructions: str | None = None, **_: object
    ) -> Dataset[str, str, None]:
        return Dataset(
            name='generated',
            cases=[Case(name='france', inputs='What is the capital of France?', expected_output='Paris')],
        )

    monkeypatch.setattr('pydantic_evals.generation.generate_dataset', fake_generate)

    suite = eval_suite(qa_agent())
    suite.generate(n=1)
    assert suite.run() is True  # the generated case actually evaluates
