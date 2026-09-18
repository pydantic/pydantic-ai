"""Development reloads replace running shell code without replacing the process or conversation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('mode', ['unchanged', 'success', 'custom', 'syntax', 'import', 'build'])
def test_reload_running_shell(tmp_path: Path, mode: str) -> None:
    package = Path(__file__).parents[1] / 'src' / 'pydantic_clai2'
    shutil.copytree(package, tmp_path / 'pydantic_clai2', ignore=shutil.ignore_patterns('__pycache__'))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name('reload_script.py')), str(tmp_path), mode],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
