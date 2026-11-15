from __future__ import annotations as _annotations

import ast
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_examples.config import ExamplesConfig as BaseExamplesConfig

import pydantic_ai.capabilities
import pydantic_ai.durable_exec
from pydantic_ai.capabilities import AbstractCapability


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
    root_dir = Path(__file__).parents[2]
    os.chdir(root_dir)

    # `find_examples` yields paths relative to the cwd we just set, so use them as-is.
    for ex in find_examples('src/pydantic_ai_harness/pydantic_ai_harness/.agents'):
        yield pytest.param(ex, id=f'{ex.path}:{ex.start_line}')


def test_migration_skill_examples_are_executable():
    root_dir = Path(__file__).parents[2]
    skill_dir = (
        root_dir
        / 'src/pydantic_ai_harness/pydantic_ai_harness/.agents/skills/migrating-deep-agents-to-pydantic-ai-harness'
    )
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


def _squash(text: str) -> str:
    return re.sub(r'[\s_-]', '', text).lower()


def _section_lines(markdown: str) -> list[str]:
    """Headings and detail-table rows, skipping code blocks and `| ... | Use |` selection tables."""
    lines: list[str] = []
    in_code = False
    table_header: str | None = None
    for line in markdown.splitlines():
        if line.startswith('```'):
            in_code = not in_code
        elif in_code:
            continue
        elif line.startswith('|'):
            table_header = table_header or line
            if not table_header.rstrip(' |').endswith('| Use'):
                lines.append(line)
        else:
            table_header = None
            if line.startswith('#'):
                lines.append(line)
    return lines


def _capability_exports(package_dir: Path) -> dict[str, list[str]]:
    """Capability classes each submodule exports, found from source so optional dependencies need not be installed.

    A class counts when its bases lead, within the package, to a core capability class.
    """
    bases: dict[str, list[str]] = {}
    for path in package_dir.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                bases.setdefault(node.name, []).extend(_base_name(base) for base in node.bases)

    def is_capability(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name in bases and name not in seen:
            return any(is_capability(base, seen | {name}) for base in bases[name])
        core = getattr(pydantic_ai.capabilities, name, None) or getattr(pydantic_ai.durable_exec, name, None)
        return isinstance(core, type) and issubclass(core, AbstractCapability)

    exports: dict[str, list[str]] = {}
    for init in [*package_dir.glob('*/__init__.py'), *package_dir.glob('experimental/*/__init__.py')]:
        module = '.'.join(init.parent.relative_to(package_dir).parts)
        names = [
            element.value
            for node in ast.parse(init.read_text()).body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets)
            and isinstance(node.value, ast.List | ast.Tuple)
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        exports[module] = [name for name in names if is_capability(name)]
    return exports


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ''


def test_harness_skill_covers_every_capability_module():
    # Every public capability submodule, and every capability class it exports, needs an entry in the
    # `Task-Family References` table of the `pydantic-ai-harness` skill, and each class needs its own section in
    # the linked reference. Deprecated rename shims are skipped. Rows look like
    # `| [Title](./references/FILE.md) | `Class` (`.module`, `[extra]`); ... |`.
    package_dir = Path(__file__).parents[2] / 'src/pydantic_ai_harness/pydantic_ai_harness'
    skill_dir = package_dir / '.agents/skills/pydantic-ai-harness'
    table = (skill_dir / 'SKILL.md').read_text().split('## Task-Family References', 1)[1]
    rows = [line for line in table.splitlines() if line.startswith('| [')]

    modules: list[str] = []
    for init in [*package_dir.glob('[!_.]*/__init__.py'), *package_dir.glob('experimental/*/__init__.py')]:
        module = '.'.join(init.parent.relative_to(package_dir).parts)
        docstring = ast.get_docstring(ast.parse(init.read_text())) or ''
        if module != 'experimental' and not docstring.startswith('Deprecated import location'):
            modules.append(module)
    assert 'coder' in modules

    entries = {
        module: (row, entry)
        for module in modules
        for row in rows
        for entry in row.split('|')[2].split(';')
        if f'(`.{module}`' in entry
    }
    missing_entries = sorted(set(modules) - set(entries))

    links = {module: re.search(r'\]\(\./(references/[A-Z-]+\.md)\)', row) for module, (row, _) in entries.items()}
    references = {
        module: skill_dir / (link.group(1) if link else 'references/MISSING.md') for module, link in links.items()
    }
    missing_references = sorted(module for module, reference in references.items() if not reference.exists())

    listed = {module: re.findall(r'`([A-Z]\w+)`', entry) for module, (_, entry) in entries.items()}
    capability_exports = _capability_exports(package_dir)
    unlisted = sorted(
        f'{module}.{name}'
        for module in entries
        for name in capability_exports.get(module, ())
        if name not in listed[module]
    )

    # Each class (or, for entries without one, the module) needs a heading or detail-table row in the reference, not
    # just a mention in passing prose. Compare loosely so `CodeMode` matches a `# Code Mode` heading.
    sections = {
        module: [_squash(line) for line in _section_lines(reference.read_text())]
        for module, reference in references.items()
        if reference.exists()
    }
    without_section = sorted(
        f'{module}.{name}'
        for module, lines in sections.items()
        for name in listed[module] or [module.rsplit('.', 1)[-1]]
        if not any(_squash(name) in line for line in lines)
    )

    assert (missing_entries, missing_references, unlisted, without_section) == ([], [], [], [])


@pytest.mark.parametrize('example', list(find_skill_examples()))
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
        target_version='py311',
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
