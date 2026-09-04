from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import typecheck_changed
from typecheck_changed import CHECKPOINT_NAME

_MODULES = {
    'pkg_src/pkg/__init__.py': '',
    'pkg_src/pkg/leaf.py': 'VALUE = 1\n',
    'pkg_src/pkg/middle.py': 'from pkg.leaf import VALUE\n\nDOUBLED: int = VALUE * 2\n',
    'pkg_src/pkg/top.py': 'from .middle import DOUBLED\n\nTRIPLED = DOUBLED * 3\n',
    'pkg_src/pkg/aside.py': 'ASIDE = 1\n',
    **{f'pkg_src/pkg/spare{index}.py': f'SPARE = {index}\n' for index in range(4)},
}

_PYPROJECT = """\
[tool.pyright]
include = ["pkg_src"]

[tool.uv.workspace]
members = ["pkg_src"]
"""

# Pyright resolves `pkg` through the workspace package's own directory, which the real
# project gets from the editable installs in its virtual environment.
_EXECUTION_ENVIRONMENT = """
[[tool.pyright.executionEnvironments]]
root = "."
extraPaths = ["pkg_src"]
"""

# `middle.py` sits between `leaf.py` and `top.py`, so excluding it puts a file Pyright
# reports nothing about in the middle of an import chain.
_EXCLUDING_PYPROJECT = _PYPROJECT.replace(
    'include = ["pkg_src"]', 'include = ["pkg_src"]\nexclude = ["pkg_src/pkg/middle.py"]'
)

_FULL_RUN = [['make', 'typecheck-pyright']]


class _Recorder:
    """Stands in for Pyright, recording what it would have been asked to check."""

    def __init__(self, code: int) -> None:
        self.code = code
        self.commands: list[list[str]] = []
        self.exit_code = -1

    def __call__(self, command: Sequence[str]) -> int:
        self.commands.append(list(command))
        return self.code

    @property
    def checked(self) -> list[str]:
        command = self.commands[-1]
        assert command[:3] == [sys.executable, '-m', 'pyright'], command
        return command[3:]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, source in _MODULES.items():
        _write(tmp_path, name, source)
    _write(tmp_path, 'pyproject.toml', _PYPROJECT)
    _write(tmp_path, 'uv.lock', 'version = 1\n')
    _write(tmp_path, 'Makefile', f'typecheck-pyright:\n\t{sys.executable} -m pyright pkg_src\n')
    subprocess.run(['git', 'init', '--quiet'], cwd=tmp_path, check=True, capture_output=True)
    _stage(tmp_path)

    monkeypatch.chdir(tmp_path)
    # Both are read from the environment, and both are set while this suite runs in CI.
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.delenv('PYRIGHT_PYTHON', raising=False)
    return tmp_path


def _write(project: Path, name: str, source: str) -> None:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')


def _edit(project: Path, name: str) -> None:
    # A comment is the one edit that stays valid in a module, a TOML file and a Makefile.
    path = project / name
    path.write_text(f'{path.read_text(encoding="utf-8")}# edited\n', encoding='utf-8')


def _stage(project: Path) -> None:
    subprocess.run(['git', 'add', '--all'], cwd=project, check=True, capture_output=True)


def _typecheck(*, fails: bool = False) -> _Recorder:
    recorder = _Recorder(1 if fails else 0)
    recorder.exit_code = typecheck_changed.main(recorder)
    return recorder


def _checkpoint(project: Path) -> Path:
    return project / '.git' / CHECKPOINT_NAME


def test_the_first_run_checks_every_tracked_file(project: Path):
    recorder = _typecheck()

    assert recorder.commands == _FULL_RUN
    assert recorder.exit_code == 0
    checkpoint = json.loads(_checkpoint(project).read_text(encoding='utf-8'))
    assert sorted(checkpoint['files']) == sorted(_MODULES)


def test_a_run_on_unchanged_files_checks_nothing(project: Path):
    _typecheck()

    recorder = _typecheck()

    assert recorder.commands == []
    assert recorder.exit_code == 0


def test_an_edit_no_one_imports_checks_one_file(project: Path):
    _typecheck()
    _edit(project, 'pkg_src/pkg/aside.py')

    assert _typecheck().checked == ['pkg_src/pkg/aside.py']


def test_an_edit_checks_everything_that_imports_it(project: Path):
    _typecheck()
    _edit(project, 'pkg_src/pkg/leaf.py')

    assert _typecheck().checked == ['pkg_src/pkg/leaf.py', 'pkg_src/pkg/middle.py', 'pkg_src/pkg/top.py']


@pytest.mark.parametrize('staged', [True, False], ids=['staged', 'unstaged'])
def test_a_deleted_module_checks_what_used_to_import_it(project: Path, staged: bool):
    _typecheck()
    (project / 'pkg_src/pkg/middle.py').unlink()
    if staged:
        _stage(project)

    assert _typecheck().checked == ['pkg_src/pkg/top.py']


def test_a_new_module_inside_a_package_checks_only_itself(project: Path):
    _typecheck()
    _write(project, 'pkg_src/pkg/extra.py', 'EXTRA = 1\n')
    _stage(project)

    assert _typecheck().checked == ['pkg_src/pkg/extra.py']


def test_the_requested_python_version_reaches_pyright(project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYRIGHT_PYTHON', '3.10')
    _typecheck()
    _edit(project, 'pkg_src/pkg/aside.py')

    assert _typecheck().commands[-1][-3:] == ['--pythonversion', '3.10', 'pkg_src/pkg/aside.py']


def test_every_import_shape_resolves_to_the_module_it_names(project: Path):
    _write(
        project,
        'pkg_src/pkg/shapes.py',
        'import pkg.leaf\nfrom . import aside\nfrom pkg.spare0 import *\nfrom ... import nowhere\n',
    )
    _stage(project)
    _typecheck()

    checkpoint = json.loads(_checkpoint(project).read_text(encoding='utf-8'))
    assert checkpoint['files']['pkg_src/pkg/shapes.py']['imports'] == [
        'pkg_src/pkg/__init__.py',
        'pkg_src/pkg/aside.py',
        'pkg_src/pkg/leaf.py',
        'pkg_src/pkg/spare0.py',
    ]


def test_a_stub_stands_in_for_the_module_beside_it(project: Path):
    _write(project, 'pkg_src/pkg/stub.py', 'STUB = 1\n')
    _write(project, 'pkg_src/pkg/stub.pyi', 'STUB: int\n')
    _write(project, 'pkg_src/pkg/reader.py', 'from pkg.stub import STUB\n')
    _stage(project)
    _typecheck()
    _edit(project, 'pkg_src/pkg/stub.pyi')

    assert _typecheck().checked == ['pkg_src/pkg/reader.py', 'pkg_src/pkg/stub.pyi']


def test_a_dot_directory_is_checked_when_exclude_is_set(project: Path):
    # Pyright's built-in `**/.*` exclusion only applies while `exclude` is unset.
    _write(project, 'pyproject.toml', _PYPROJECT.replace('["pkg_src"]', '["pkg_src"]\nexclude = ["nothing.py"]', 1))
    _write(project, 'pkg_src/pkg/.skill/helper.py', 'HELPER = 1\n')
    _stage(project)
    _typecheck()
    _edit(project, 'pkg_src/pkg/.skill/helper.py')

    assert _typecheck().checked == ['pkg_src/pkg/.skill/helper.py']


def test_a_dot_directory_is_skipped_when_exclude_is_unset(project: Path):
    _write(project, 'pkg_src/pkg/.skill/helper.py', 'HELPER = 1\n')
    _stage(project)
    _typecheck()
    _edit(project, 'pkg_src/pkg/.skill/helper.py')

    assert _typecheck().commands == []


def test_a_file_that_does_not_parse_is_still_checked(project: Path):
    _typecheck()
    _write(project, 'pkg_src/pkg/middle.py', 'from pkg.leaf import (\n')

    assert _typecheck().checked == ['pkg_src/pkg/middle.py', 'pkg_src/pkg/top.py']


def test_a_new_top_level_module_checks_everything(project: Path):
    _typecheck()
    _write(project, 'pkg_src/shadow.py', 'SHADOW = 1\n')
    _stage(project)

    assert _typecheck().commands == _FULL_RUN


def test_a_module_that_becomes_a_package_checks_everything(project: Path):
    _typecheck()
    (project / 'pkg_src/pkg/leaf.py').unlink()
    _write(project, 'pkg_src/pkg/leaf/__init__.py', 'VALUE = 1\n')
    _stage(project)

    assert _typecheck().commands == _FULL_RUN


@pytest.mark.parametrize('name', ['pyproject.toml', 'uv.lock', 'Makefile'])
def test_a_configuration_change_checks_everything(project: Path, name: str):
    _typecheck()
    _edit(project, name)

    assert _typecheck().commands == _FULL_RUN


def test_a_new_interpreter_checks_everything(project: Path, monkeypatch: pytest.MonkeyPatch):
    _typecheck()
    monkeypatch.setattr(typecheck_changed.platform, 'python_version', lambda: '9.9.9')

    assert _typecheck().commands == _FULL_RUN


def test_asking_pyright_for_another_python_version_checks_everything(project: Path, monkeypatch: pytest.MonkeyPatch):
    # `PYRIGHT_PYTHON` becomes `--pythonversion`, so what passed under one value says
    # nothing about another.
    _typecheck()
    monkeypatch.setenv('PYRIGHT_PYTHON', '3.10')

    assert _typecheck().commands == _FULL_RUN


def test_a_change_reaching_most_of_the_project_checks_everything(project: Path):
    _typecheck()
    for name in ['aside', 'spare0', 'spare1', 'spare2', 'spare3']:
        _edit(project, f'pkg_src/pkg/{name}.py')

    assert _typecheck().commands == _FULL_RUN


def test_a_change_reaching_half_the_project_stays_narrowed(project: Path):
    _typecheck()
    for name in ['aside', 'spare0', 'spare1', 'spare2']:
        _edit(project, f'pkg_src/pkg/{name}.py')

    assert len(_typecheck().checked) == 4


def test_an_excluded_file_still_carries_the_closure_through_it(project: Path):
    _write(project, 'pyproject.toml', _EXCLUDING_PYPROJECT)
    _typecheck()
    _edit(project, 'pkg_src/pkg/leaf.py')

    # `top.py` imports `leaf.py` only through the excluded `middle.py`, and is reached anyway.
    assert _typecheck().checked == ['pkg_src/pkg/leaf.py', 'pkg_src/pkg/top.py']
    assert 'pkg_src/pkg/middle.py' in json.loads(_checkpoint(project).read_text(encoding='utf-8'))['files']


def test_editing_an_excluded_file_checks_what_imports_it(project: Path):
    _write(project, 'pyproject.toml', _EXCLUDING_PYPROJECT)
    _typecheck()
    _edit(project, 'pkg_src/pkg/middle.py')

    assert _typecheck().checked == ['pkg_src/pkg/top.py']


@pytest.mark.parametrize(
    'pyproject',
    [_PYPROJECT.replace('["pkg_src"]', '["pkg_src/**"]'), _PYPROJECT.replace('include = ["pkg_src"]\n', '')],
    ids=['glob', 'no-include'],
)
def test_a_file_list_this_script_cannot_reproduce_checks_everything(project: Path, pyproject: str):
    _write(project, 'pyproject.toml', pyproject)

    recorder = _typecheck()

    assert recorder.commands == _FULL_RUN
    # Without a file list there is nothing to record, so every run stays a full one.
    assert not _checkpoint(project).exists()


def test_an_unreadable_checkpoint_checks_everything(project: Path):
    _typecheck()
    _checkpoint(project).write_text('not a checkpoint', encoding='utf-8')

    assert _typecheck().commands == _FULL_RUN


def test_ci_checks_everything_and_records_nothing(project: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CI', 'true')

    recorder = _typecheck()

    assert recorder.commands == _FULL_RUN
    assert not _checkpoint(project).exists()


def test_a_failing_run_leaves_the_checkpoint_alone(project: Path):
    _typecheck()
    recorded = _checkpoint(project).read_bytes()
    _edit(project, 'pkg_src/pkg/aside.py')

    recorder = _typecheck(fails=True)

    assert recorder.exit_code == 1
    assert _checkpoint(project).read_bytes() == recorded


def test_pyright_reports_an_error_only_the_closure_reveals(project: Path, capsys: pytest.CaptureFixture[str]):
    _write(project, 'pyproject.toml', _PYPROJECT + _EXECUTION_ENVIRONMENT)

    assert typecheck_changed.main() == 0

    # `leaf.py` itself still type-checks. The error is in `middle.py`, which did not change
    # and is only reached by following the import graph.
    _write(project, 'pkg_src/pkg/leaf.py', "VALUE = 'one'\n")

    assert typecheck_changed.main() == 1
    assert 'Type-checking 3 of 9 files' in capsys.readouterr().out
