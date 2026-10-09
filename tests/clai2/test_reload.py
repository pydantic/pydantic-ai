"""Development reloads replace running shell code without replacing the process or conversation."""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / 'src' / 'pydantic_clai2' / 'pydantic_clai2'


@dataclass
class ReloadServer:
    """`reload_script.py`, which imports the shell once and forks a fresh child per scenario."""

    process: subprocess.Popen[str]

    def run(self, root: Path, mode: str) -> None:
        assert self.process.stdin is not None and self.process.stdout is not None
        # The server started before this test isolated its environment; the child takes the isolated one.
        request = {'root': str(root), 'mode': mode, 'environment': dict(os.environ)}
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        exit_code = self.process.stdout.readline().strip()
        assert exit_code == '0', (root / 'output.log').read_text()


@pytest.fixture(scope='module')
def reload_server() -> Iterator[ReloadServer]:
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name('reload_script.py'))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    yield ReloadServer(process)
    process.communicate(timeout=30)
    assert process.returncode == 0


@pytest.mark.parametrize(
    'mode',
    ['unchanged', 'success', 'custom', 'new_imports', 'stock', 'syntax', 'import', 'build', 'harness', 'transcript'],
)
def test_reload_running_shell(tmp_path: Path, reload_server: ReloadServer, mode: str) -> None:
    copy_package(tmp_path)
    reload_server.run(tmp_path, mode)


def copy_package(tmp_path: Path) -> None:
    shutil.copytree(PACKAGE, tmp_path / 'pydantic_clai2', ignore=shutil.ignore_patterns('__pycache__'))


def run_script(tmp_path: Path, script: str, mode: str) -> None:
    copy_package(tmp_path)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script)), str(tmp_path), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    'mode',
    [
        'relative',
        'annotated',
        'augmented',
        'branch_alias',
        'agreed_alias',
        'unknown_guards',
        'invalid_guard',
        *(
            f'guard:{name}'
            for name in (
                'platform',
                'platform_alias',
                'os_alias',
                'os_dotted',
                'annotation_only',
                'version',
                'version_lt',
                'version_le',
                'version_gt',
                'constant',
                'main',
                'module_name',
                'package_name',
                'false',
                'not',
            )
        ),
        'absolute',
        'module',
        'relative_module',
        'class',
        'class_scope',
        'reverse',
        'lazy',
        'inactive',
        'new_package',
        'import_error',
        'build_error',
        'cycle',
        'syntax',
    ],
)
def test_reload_changed_import_graph(tmp_path: Path, mode: str) -> None:
    run_script(tmp_path, 'reload_import_script.py', mode)
