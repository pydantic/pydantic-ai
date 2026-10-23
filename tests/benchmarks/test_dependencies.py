from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from tests.benchmarks import conftest


@pytest.fixture
def blockbuster_enabled() -> bool:
    return False


@pytest.mark.parametrize('mode', ['missing', 'present', 'write'])
def test_dependency_guard_detects_new_runtime_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    mode: Literal['missing', 'present', 'write'],
) -> None:
    benchmark_dir = tmp_path / 'tests' / 'benchmarks'
    benchmark_dir.mkdir(parents=True)
    manifest = benchmark_dir / 'dependencies.json'
    expected: dict[str, list[str]] = {
        'loaded': ['pydantic_ai_slim/pydantic_ai/providers/new.py', 'tests/benchmarks/conftest.py']
    }
    manifest.write_text(json.dumps(expected if mode == 'present' else {'loaded': []}))
    monkeypatch.setattr(conftest, '__file__', str(benchmark_dir / 'conftest.py'))
    module = ModuleType('new_benchmark_dependency')
    module.__file__ = str(tmp_path / 'pydantic_ai_slim' / 'pydantic_ai' / 'providers' / 'new.py')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(request.config, 'args', ['tests/benchmarks'])
    monkeypatch.setattr(request.config.option, 'benchmark_write_dependencies', mode == 'write')
    if mode == 'write':
        conftest.pytest_sessionfinish(request.session)
        assert json.loads(manifest.read_text()) == expected
    elif mode == 'present':
        conftest.pytest_sessionfinish(request.session)
    else:
        with pytest.raises(pytest.exit.Exception, match='Benchmark dependencies changed'):
            conftest.pytest_sessionfinish(request.session)


def test_dependency_guard_ignores_other_suites(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    monkeypatch.setattr(request.config, 'args', ['tests'])
    conftest.pytest_sessionfinish(request.session)
