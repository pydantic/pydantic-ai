"""Exercise pytest lifecycle behavior with no provider requests for VCR to record."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from tests.benchmarks import conftest


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Keep BlockBuster's unused CLI and harness imports out of the dependency audit."""
    return False


@pytest.mark.parametrize(
    'argument',
    [
        'tests/benchmarks',
        'tests/benchmarks/',
        str(Path(__file__).resolve().parent),
        'tests/benchmarks/test_agent.py',
        'tests/benchmarks/test_agent.py::test_agent_run_without_capabilities',
    ],
)
@pytest.mark.parametrize(
    ('dependency', 'included'),
    [
        ('pydantic_ai_slim/pydantic_ai/capabilities/foo.py', True),
        ('pydantic_ai_slim/pydantic_ai/models/function.py', True),
        ('pydantic_ai_slim/pydantic_ai/providers/openai.py', False),
    ],
)
def test_dependency_guard_checks_loaded_files_against_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    argument: str,
    dependency: str,
    included: bool,
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    module = ModuleType('loaded_dependency')
    module.__file__ = str(root / dependency)
    external = ModuleType('external_dependency')
    external.__file__ = str(tmp_path / 'external.py')
    environment = ModuleType('environment_dependency')
    environment.__file__ = str(root / '.venv/package/module.py')
    modules: dict[str, ModuleType] = {
        'loaded': module,
        'originless': ModuleType('originless'),
        'external': external,
        'environment': environment,
    }
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(modules=modules))
    monkeypatch.setattr(request.config, 'args', [argument])
    if included:
        conftest.pytest_sessionfinish(request.session)
    else:
        with pytest.raises(pytest.exit.Exception, match='Loaded benchmark dependencies would skip CodSpeed') as error:
            conftest.pytest_sessionfinish(request.session)
        assert str(error.value).endswith(f'{dependency}\n')


@pytest.mark.parametrize(
    'arguments',
    [[], ['tests'], ['.github/scripts/test_ci_content_classifier.py'], ['tests/benchmarks', 'tests/test_tools.py']],
)
def test_dependency_guard_ignores_mixed_and_unrelated_suites(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, arguments: list[str]
) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    module = ModuleType('excluded_dependency')
    module.__file__ = str(root / 'pydantic_ai_slim/pydantic_ai/providers/openai.py')
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(modules={'excluded': module}))
    monkeypatch.setattr(request.config, 'args', arguments)
    conftest.pytest_sessionfinish(request.session)


def test_dependency_guard_accepts_equivalent_workflow_indentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / '.github/workflows/benchmark.yml').read_text()
    reindented = '\n'.join(f'  {line}' for line in workflow.splitlines()) + '\n'
    original: object = yaml.safe_load(workflow)
    reformatted: object = yaml.safe_load(reindented)
    assert original == reformatted

    workflow_path = tmp_path / '.github/workflows/benchmark.yml'
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(reindented)
    benchmark_dir = tmp_path / 'tests/benchmarks'
    benchmark_dir.mkdir(parents=True)
    monkeypatch.setattr(conftest, '__file__', str(benchmark_dir / 'conftest.py'))
    monkeypatch.setattr(request.config, 'args', [str(benchmark_dir)])
    module = ModuleType('loaded_dependency')
    module.__file__ = str(tmp_path / 'pydantic_ai_slim/pydantic_ai/capabilities/foo.py')
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(modules={'loaded': module}))
    conftest.pytest_sessionfinish(request.session)


@pytest.mark.parametrize(
    ('classifier', 'expected_error'),
    [
        pytest.param('affects_benchmarks() {\n  return 0\n}\n', False, id='first-line'),
        pytest.param(
            'old_affects_benchmarks() {\n  return 1\n}\naffects_benchmarks() {\n  return 0\n}\n',
            False,
            id='unused-prefixed-function',
        ),
        pytest.param(
            "affects_benchmarks() {\n  cat <<'END' >/dev/null\n}ignored\nEND\n  return 0\n}\n",
            False,
            id='brace-prefix-in-heredoc',
        ),
        pytest.param('renamed_classifier() {\n  return 0\n}\n', True, id='missing-function'),
        pytest.param('affects_benchmarks() {\n  return 0\n', True, id='missing-closing-brace'),
    ],
)
def test_dependency_guard_extracts_whole_function_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    classifier: str,
    expected_error: bool,
) -> None:
    workflow: conftest.Workflow = {'jobs': {'classify': {'steps': [{'run': classifier}]}}}
    workflow_path = tmp_path / '.github/workflows/benchmark.yml'
    workflow_path.parent.mkdir(parents=True)
    workflow_path.write_text(yaml.safe_dump(workflow))
    benchmark_dir = tmp_path / 'tests/benchmarks'
    benchmark_dir.mkdir(parents=True)
    monkeypatch.setattr(conftest, '__file__', str(benchmark_dir / 'conftest.py'))
    monkeypatch.setattr(request.config, 'args', [str(benchmark_dir)])
    module = ModuleType('loaded_dependency')
    module.__file__ = str(tmp_path / 'pydantic_ai_slim/pydantic_ai/models/function.py')
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(modules={'loaded': module}))
    if expected_error:
        with pytest.raises(pytest.exit.Exception, match=r'Cannot extract.*affects_benchmarks.*classify'):
            conftest.pytest_sessionfinish(request.session)
    else:
        conftest.pytest_sessionfinish(request.session)


@pytest.mark.parametrize('workflow', [None, '[', '{}', 'jobs: {}', 'jobs: {classify: {steps: []}}'])
def test_dependency_guard_reports_workflow_extraction_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    workflow: str | None,
) -> None:
    if workflow is not None:
        workflow_path = tmp_path / '.github/workflows/benchmark.yml'
        workflow_path.parent.mkdir(parents=True)
        workflow_path.write_text(workflow)
    benchmark_dir = tmp_path / 'tests/benchmarks'
    benchmark_dir.mkdir(parents=True)
    monkeypatch.setattr(conftest, '__file__', str(benchmark_dir / 'conftest.py'))
    monkeypatch.setattr(request.config, 'args', [str(benchmark_dir)])
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(modules={}))
    with pytest.raises(pytest.exit.Exception, match=r'Cannot extract.*affects_benchmarks.*classify') as error:
        conftest.pytest_sessionfinish(request.session)
    assert '.github/workflows/benchmark.yml' in str(error.value)
    assert 'tests/benchmarks/conftest.py' in str(error.value)
