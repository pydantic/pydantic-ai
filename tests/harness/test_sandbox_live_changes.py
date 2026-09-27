"""Which changes start the billed sandbox live tests (`scripts/sandbox_live_changes.py`)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'sandbox_live_changes.py'


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location('sandbox_live_changes', _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()
ALL = list(script.PROVIDERS)


def _lock(**packages: str) -> str:
    tables = [
        f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = {{ registry = "https://pypi.org/simple" }}\n'
        for name, version in packages.items()
    ]
    return 'version = 1\n\n' + '\n'.join(tables)


BASE_LOCK = _lock(**{'pydantic-ai-slim': '2.0.0', 'modal': '1.5.2', 'e2b': '2.48.0', 'httpx': '0.28.0'})


@pytest.mark.parametrize(
    ('changed', 'expected'),
    [
        pytest.param(['pydantic_ai_harness/modal_sandbox/_backend.py'], ['modal'], id='provider package'),
        pytest.param(['tests/e2b_sandbox/test_e2b_live.py'], ['e2b'], id='provider tests'),
        pytest.param(['tests/filesystem/test_search_live.py'], ['e2b'], id='shared filesystem E2B test'),
        pytest.param(['pydantic_ai_harness/shell/_toolset.py'], ALL, id='shared capability'),
        pytest.param(['pydantic_ai_harness/_workspace.py'], ALL, id='shared module'),
        pytest.param(['scripts/sandbox_live_changes.py'], ALL, id='this script'),
        pytest.param(['docs/modal-sandbox.md'], ['modal'], id='provider docs page'),
        pytest.param(['pydantic_ai_harness/sprites_sandbox/README.md'], ['sprites'], id='provider readme'),
        pytest.param(['tests/_docs_examples.py'], ALL, id='docs example runner'),
        pytest.param(['docs/shell.md'], [], id='other docs'),
        pytest.param(['pyproject.toml', '.github/workflows/main.yml'], [], id='project and ci config'),
        pytest.param(['pydantic_ai_harness/memory/_store.py'], [], id='unrelated capability'),
        pytest.param(['pydantic_ai_harness/modal_sandboxes.py'], [], id='name that only shares a prefix'),
    ],
)
def test_changed_paths_select_providers(changed: list[str], expected: list[str]) -> None:
    assert script.providers_for(changed, BASE_LOCK, BASE_LOCK) == expected


@pytest.mark.parametrize(
    ('moved', 'expected'),
    [
        pytest.param({'pydantic-ai-slim': '2.1.0'}, ALL, id='core pin'),
        pytest.param({'modal': '1.6.0'}, ['modal'], id='provider sdk'),
        pytest.param({'httpx': '0.28.1'}, [], id='unrelated dependency'),
        pytest.param({'sprites-py': '0.1.0'}, ['sprites'], id='provider sdk added'),
    ],
)
def test_moved_lock_entries_select_providers(moved: dict[str, str], expected: list[str]) -> None:
    head = _lock(**{'pydantic-ai-slim': '2.0.0', 'modal': '1.5.2', 'e2b': '2.48.0', 'httpx': '0.28.0', **moved})
    assert script.providers_for(['uv.lock'], BASE_LOCK, head) == expected


def test_a_source_change_counts_as_a_move() -> None:
    # A git pin keeps its version string until the package is rebuilt, so the pin itself is compared.
    base = '[[package]]\nname = "pydantic-ai-slim"\nversion = "2.0.0"\nsource = { git = "https://x?rev=aaa#aaa" }\n'
    head = base.replace('aaa', 'bbb')
    assert script.moved_lock_packages(base, head) == {'pydantic-ai-slim'}


def test_lock_entries_are_ignored_when_the_lock_did_not_change() -> None:
    head = _lock(**{'pydantic-ai-slim': '9.9.9'})
    assert script.providers_for(['README.md'], BASE_LOCK, head) == []


def test_only_providers_in_the_tree_are_selected() -> None:
    assert script.providers_for(['pydantic_ai_harness/_warn.py'], BASE_LOCK, BASE_LOCK, present=['modal']) == ['modal']


def test_present_providers_follow_the_package_layout(tmp_path: Path) -> None:
    (tmp_path / 'pydantic_ai_harness' / 'sprites_sandbox').mkdir(parents=True)
    (tmp_path / 'pydantic_ai_harness' / 'e2b_sandbox').mkdir()
    assert script.present_providers(tmp_path) == ['e2b', 'sprites']


def test_the_uv_lock_in_this_repo_parses() -> None:
    lock = (Path(__file__).parents[1] / 'uv.lock').read_text()
    assert 'pydantic-ai-slim' in script.moved_lock_packages('', lock)


def test_e2b_live_workflow_selects_shared_filesystem_test() -> None:
    workflow = (Path(__file__).parents[1] / '.github/workflows/sandbox-live.yml').read_text()
    assert 'tests/filesystem' in workflow


def test_all_prints_the_providers_in_this_tree() -> None:
    result = subprocess.run([sys.executable, str(_SCRIPT), 'all'], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == script.present_providers()
