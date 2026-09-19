"""Development reloads replace running shell code without replacing the process or conversation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('mode', ['unchanged', 'success', 'custom', 'new_imports', 'syntax', 'import', 'build'])
def test_reload_running_shell(tmp_path: Path, mode: str) -> None:
    run_script(tmp_path, 'reload_script.py', mode)


def run_script(tmp_path: Path, script: str, mode: str) -> None:
    package = Path(__file__).parents[1] / 'src' / 'pydantic_clai2'
    shutil.copytree(package, tmp_path / 'pydantic_clai2', ignore=shutil.ignore_patterns('__pycache__'))
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
