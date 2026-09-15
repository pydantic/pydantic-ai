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
    # Genuinely illustrative fragments (e.g. sandbox-side code the model would generate)
    # opt out with a `lint="skip"` fence directive.
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)

    # `find_examples` yields paths relative to the cwd we just set, so use them as-is.
    for ex in find_examples('pydantic_ai_harness/.agents'):
        yield pytest.param(ex, id=f'{ex.path}:{ex.start_line}')


def test_migration_skill_examples_are_executable():
    root_dir = Path(__file__).parent.parent
    skill_dir = root_dir / 'pydantic_ai_harness/.agents/skills/migrating-deep-agents-to-pydantic-ai-harness'
    examples = list(find_examples(skill_dir))
    fence_count = 0

    for path in skill_dir.rglob('*.md'):
        open_fence = False
        for line in path.read_text().splitlines():
            if not line.startswith('```'):
                continue
            if open_fence:
                open_fence = False
            else:
                assert line == '```python', f'{path} contains an untested non-Python example'
                fence_count += 1
                open_fence = True
        assert not open_fence, f'{path} contains an unclosed example'

    assert len(examples) == fence_count > 0
    for example in examples:
        prefix = example.prefix_settings()
        assert not prefix.get('lint', '').startswith('skip')
        assert not prefix.get('test', '').startswith('skip')


@pytest.mark.parametrize('example', find_skill_examples())
def test_skill_examples(example: CodeExample, eval_example: EvalExample):
    # Lint every snippet to catch stale imports/syntax, and additionally execute the ones
    # that need no live model, network, or external file -- those exercise the real
    # constructor/decorator signatures at runtime. Snippets that need a model, network, or
    # file (or are illustrative fragments) opt out with `test="skip"` / `lint="skip"`
    # fence directives; model-backed flows are covered by `test_readme_quick_start.py`.
    # Run with `--update-examples` to reformat snippets and regenerate their printed output.
    prefix = example.prefix_settings()

    # Snippets default to black's 88-column width (matching pydantic-ai's docs examples);
    # a snippet can widen this with a `line_length="..."` fence directive.
    line_length = int(prefix.get('line_length', '88'))

    eval_example.config = ExamplesConfig(
        ruff_ignore=['D', 'Q001'],
        target_version='py310',
        line_length=line_length,
        isort=True,
        upgrade=True,
        quotes='single',
        known_first_party=['pydantic_ai_harness'],
    )

    if not prefix.get('lint', '').startswith('skip'):
        if eval_example.update_examples:  # pragma: lax no cover
            eval_example.format_ruff(example)
        else:
            eval_example.lint_ruff(example)

    if prefix.get('test', '').startswith('skip'):
        pytest.skip('running skipped for this example')

    if eval_example.update_examples:  # pragma: lax no cover
        eval_example.run_print_update(example)
    else:
        eval_example.run_print_check(example)
