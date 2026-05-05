from __future__ import annotations as _annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_examples.config import ExamplesConfig as BaseExamplesConfig


@dataclass
class ExamplesConfig(BaseExamplesConfig):
    known_first_party: list[str] = field(default_factory=list[str])

    def ruff_config(self) -> tuple[str, ...]:
        config = super().ruff_config()
        if self.known_first_party:  # pragma: no branch
            config = (*config, '--config', f'lint.isort.known-first-party = {self.known_first_party}')
        return config


def find_skill_examples() -> Iterable[ParameterSet]:
    # Skill examples are package assets for agents, not executable docs pages.
    # Lint them to catch stale Python snippets without running model/file-system examples.
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    for ex in find_examples('pydantic_ai_slim/pydantic_ai/.agents'):
        try:
            path = ex.path.relative_to(root_dir)
        except ValueError:
            path = ex.path
        yield pytest.param(ex, id=f'{path}:{ex.start_line}')


@pytest.mark.parametrize('example', find_skill_examples())
def test_skill_examples_lint(example: CodeExample, eval_example: EvalExample):
    eval_example.config = ExamplesConfig(
        ruff_ignore=['D', 'Q001'],
        target_version='py310',
        line_length=120,
        isort=True,
        upgrade=True,
        quotes='single',
        known_first_party=['pydantic_ai', 'pydantic_evals', 'pydantic_graph'],
    )
    eval_example.lint_ruff(example)
