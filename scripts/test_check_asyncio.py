from __future__ import annotations

import io
import json
import os
import runpy
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name('check_asyncio.py')
NATIVE_SOURCE = 'import asyncio  # noqa: TID251 - native boundary\nasync def worker():\n    await asyncio.sleep(1)\n'
NATIVE_SITES = {
    'import:import asyncio': 1,
    'worker:asyncio.sleep': 1,
    'suppression:# noqa: TID251 - native boundary': 1,
}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in tuple(os.environ):
        if name.startswith('GIT_'):
            monkeypatch.delenv(name)
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True, capture_output=True)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/asyncio_exceptions.json').write_text('{}', encoding='utf-8')
    (tmp_path / 'pyproject.toml').write_text('', encoding='utf-8')
    return tmp_path


def run_check(project: Path) -> subprocess.CompletedProcess[str]:
    arguments = [str(SCRIPT), '--root', str(project)]
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(stdout), redirect_stderr(stderr):
        patch.setattr(sys, 'argv', arguments)
        patch.setattr(sys, 'path', [str(SCRIPT.parent), *sys.path])
        with pytest.raises(SystemExit) as outcome:
            runpy.run_path(str(SCRIPT), run_name='__main__')
    assert isinstance(outcome.value.code, int)
    return subprocess.CompletedProcess(arguments, outcome.value.code, stdout.getvalue(), stderr.getvalue())


@pytest.fixture
def native_project(project: Path) -> Path:
    (project / 'native.py').write_text(NATIVE_SOURCE, encoding='utf-8')
    inventory = {'native.py': {'reason': 'Preserve native runtime cancellation.', 'sites': NATIVE_SITES}}
    (project / 'scripts/asyncio_exceptions.json').write_text(json.dumps(inventory), encoding='utf-8')
    return project


@pytest.mark.parametrize(
    'source',
    [
        'import asyncio',
        'import asyncio as aio',
        'from asyncio import sleep',
        'import asyncio.tasks',
        'from asyncio.tasks import create_task',
        'from asyncio import *',
        "import importlib\naio = importlib.import_module('asyncio')\naio.sleep(0)",
        "import importlib as loader\nloader.import_module(name='asyncio.tasks')",
        "from importlib import import_module as load\nload('asyncio')",
        "__import__('asyncio')",
        'import asyncio  # noqa: TID251',
        '# ruff: noqa: TID251\nimport asyncio',
        '# noqa\nimport anyio',
        '# flake8: noqa\nimport anyio',
        '# NOQA\nimport anyio',
        'import anyio  # type: ignore  # noqa: TID251',
    ],
)
def test_reject_unapproved_source(project: Path, source: str) -> None:
    (project / 'new.py').write_text(source, encoding='utf-8')
    result = run_check(project)
    assert result.returncode == 1
    assert 'new.py: unapproved asyncio usage or lint suppression' in result.stderr


@pytest.mark.parametrize('suffix', ['.py', '.pyi'])
def test_check_tracked_and_untracked_files(project: Path, suffix: str) -> None:
    path = project / f'new{suffix}'
    path.write_text('import asyncio', encoding='utf-8')
    assert run_check(project).returncode == 1
    subprocess.run(['git', 'add', path.name], cwd=project, check=True, capture_output=True)
    assert run_check(project).returncode == 1


def test_allow_anyio_and_ignore_strings(project: Path) -> None:
    (project / 'ok.py').write_text(
        'import anyio\nimport importlib\nimportlib.import_module("anyio")\n'
        'module_name = "anyio"\nimportlib.import_module(module_name)\n'
        'text = "import asyncio # noqa"\nasync def run():\n    await anyio.sleep(0)\n',
        encoding='utf-8',
    )
    result = run_check(project)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('filename', ['new.py', 'ruff.toml'])
def test_reject_symlinks_without_reading_the_target(project: Path, filename: str) -> None:
    target = project / 'external.txt'
    target.write_text('private target contents that are not Python or TOML', encoding='utf-8')
    (project / filename).symlink_to(target)
    result = run_check(project)
    assert result.returncode == 1
    assert f'{filename}: symlinked source/config files are not supported' in result.stderr
    assert target.read_text(encoding='utf-8') not in result.stderr


def test_allow_documented_native_boundary(native_project: Path) -> None:
    result = run_check(native_project)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ('source', 'sites'),
    [
        (
            'from asyncio import sleep as pause\nasync def worker():\n    await pause(0)\n',
            {'import:from asyncio import sleep as pause': 1, 'worker:asyncio.sleep': 1},
        ),
        (
            'import asyncio.tasks as tasks\nclass Worker:\n    def run(self):\n        return tasks.Task\n',
            {'import:import asyncio.tasks as tasks': 1, 'Worker.run:asyncio.tasks.Task': 1},
        ),
    ],
)
def test_inventory_resolves_aliases(project: Path, source: str, sites: dict[str, int]) -> None:
    (project / 'native.py').write_text(source, encoding='utf-8')
    inventory = {'native.py': {'reason': 'Native compatibility', 'sites': sites}}
    (project / 'scripts/asyncio_exceptions.json').write_text(json.dumps(inventory), encoding='utf-8')
    assert run_check(project).returncode == 0


@pytest.mark.parametrize(
    'source',
    [
        NATIVE_SOURCE + '    await asyncio.sleep(1)\n',
        NATIVE_SOURCE + '    await asyncio.Event().wait()\n',
        NATIVE_SOURCE.replace('worker', 'new_worker'),
        NATIVE_SOURCE.replace('noqa: TID251 - native boundary', 'noqa'),
        NATIVE_SOURCE.replace('await asyncio.sleep(1)', 'pass'),
        NATIVE_SOURCE.replace('import asyncio', 'import asyncio as aio').replace('asyncio.sleep', 'aio.sleep'),
        '',
    ],
)
def test_inventory_does_not_allow_growth_or_stale_entries(native_project: Path, source: str) -> None:
    (native_project / 'native.py').write_text(source, encoding='utf-8')
    result = run_check(native_project)
    assert result.returncode == 1
    assert 'native.py:' in result.stderr


def test_removed_file_requires_removing_exception(native_project: Path) -> None:
    subprocess.run(['git', 'add', 'native.py'], cwd=native_project, check=True, capture_output=True)
    (native_project / 'native.py').unlink()
    result = run_check(native_project)
    assert result.returncode == 1
    assert 'remove the exception for a missing file' in result.stderr


def test_exception_requires_a_reason(native_project: Path) -> None:
    inventory = {'native.py': {'reason': ' ', 'sites': NATIVE_SITES}}
    (native_project / 'scripts/asyncio_exceptions.json').write_text(json.dumps(inventory), encoding='utf-8')
    result = run_check(native_project)
    assert result.returncode == 1
    assert 'exception needs a reason' in result.stderr


def test_line_movement_does_not_change_exception(native_project: Path) -> None:
    (native_project / 'native.py').write_text('# An unrelated edit\n\n' + NATIVE_SOURCE, encoding='utf-8')
    result = run_check(native_project)
    assert result.returncode == 0, result.stderr


def test_empty_project_passes(project: Path) -> None:
    result = run_check(project)
    assert result.returncode == 0, result.stderr


def test_cli_from_another_directory(project: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--root', str(project)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')},
    )
    assert result.returncode == 0, result.stderr


def test_git_hook_environment_does_not_modify_parent_repository(project: Path) -> None:
    config = project / 'pytest.ini'
    config.write_text('[pytest]\n', encoding='utf-8')
    git_config = (project / '.git/config').read_bytes()
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-c', str(config), f'{__file__}::test_empty_project_passes', '-q'],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')},
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'GIT_DIR': str(project / '.git'),
            'GIT_WORK_TREE': str(project),
            'GIT_INDEX_FILE': str(project / '.git/index'),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (project / '.git/config').read_bytes() == git_config
    assert not (project / '.git/index').exists()


@pytest.mark.parametrize('filename', ['pyproject.toml', 'ruff.toml', '.ruff.toml'])
@pytest.mark.parametrize(
    'setting',
    [
        'per-file-ignores = {"*.py" = ["TID251"]}',
        'extend-per-file-ignores = {"*.py" = ["TID"]}',
        'ignore = ["ALL"]',
        'extend-ignore = ["TID251"]',
    ],
)
def test_reject_config_suppressions(project: Path, filename: str, setting: str) -> None:
    table = 'tool.ruff.lint' if filename == 'pyproject.toml' else 'lint'
    (project / filename).write_text(f'[{table}]\n{setting}\n', encoding='utf-8')
    result = run_check(project)
    assert result.returncode == 1
    assert f'{filename}: unapproved asyncio usage or lint suppression' in result.stderr


def test_allow_unrelated_config_and_suppressions(project: Path) -> None:
    (project / 'pyproject.toml').write_text(
        '[tool.ruff.lint]\nignore = ["D"]\nper-file-ignores = {"tests/*" = ["F401"]}\n', encoding='utf-8'
    )
    (project / 'other.toml').write_text('not a Ruff config', encoding='utf-8')
    (project / 'ok.py').write_text('import anyio  # noqa: F401\n', encoding='utf-8')
    assert run_check(project).returncode == 0
