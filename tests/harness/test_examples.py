"""Construction tests for the `examples/` scripts.

Each example exposes `build_agent()`. Building the agent exercises every
capability constructor and the Agent wiring without any model calls, so these
tests catch API drift between the examples and the library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob('*.py'))


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f'examples_{path.stem}', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_examples_present():
    assert [path.name for path in EXAMPLE_FILES] == [
        'coding_agent.py',
        'research_agent.py',
    ]


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=lambda p: p.stem)
@pytest.mark.parametrize(
    'optional_dependency_error',
    [
        None,
        ImportError("Install 'pydantic-ai-harness[example]'"),
        UserError("Install 'pydantic-ai-slim[example]'"),
    ],
)
def test_example_builds_agent(
    path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optional_dependency_error: ImportError | UserError | None,
):
    # Keep any filesystem-scoped capabilities and memory stores inside tmp_path.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SUPPORT_MEMORY_DIR', str(tmp_path / 'memory'))
    try:
        module = _load(path)
        if optional_dependency_error is not None:

            def raise_optional_dependency_error(*, model: TestModel) -> None:
                raise optional_dependency_error

            monkeypatch.setattr(module, 'build_agent', raise_optional_dependency_error)
        agent = module.build_agent(model=TestModel())
    except (ImportError, UserError) as exc:
        assert 'pydantic-ai-harness[' in str(exc) or 'pydantic-ai-slim[' in str(exc)
        pytest.skip(str(exc))
    assert isinstance(agent, Agent)
