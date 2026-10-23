from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.benchmarks import conftest


@pytest.fixture
def blockbuster_enabled() -> bool:
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
