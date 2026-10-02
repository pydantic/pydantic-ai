"""`pydantic-evals` documented imports succeed on a clean install.

The repository environment always contains `sniffio` transitively, so a
missing base dependency only surfaces when the built wheel is installed into
a fresh virtual environment. These checks install the built wheel with its
real dependency metadata and import the documented public API.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EVALS_IMPORT_CODE = """\
import pydantic_evals
from pydantic_evals import Dataset
from pydantic_evals.online import evaluate
"""


@pytest.fixture(scope='module')
def wheelhouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the workspace wheels with a fixed version so exact sibling pins resolve from the wheelhouse."""
    wheelhouse = tmp_path_factory.mktemp('wheelhouse')
    env = {**os.environ, 'UV_DYNAMIC_VERSIONING_BYPASS': '0.0.0'}
    subprocess.run(
        ['uv', 'build', '--all-packages', '--no-sources', '--out-dir', str(wheelhouse)],
        cwd=REPO_ROOT,
        check=True,
        env=env,
    )
    return wheelhouse


def _clean_import(wheelhouse: Path, tmp_path: Path, requirement: str) -> None:
    venv_dir = tmp_path / 'venv'
    subprocess.run(['uv', 'venv', '--python', sys.executable, str(venv_dir)], check=True)
    venv_python = venv_dir / 'bin' / 'python'
    subprocess.run(
        ['uv', 'pip', 'install', '--python', str(venv_python), '--find-links', str(wheelhouse), requirement],
        check=True,
    )
    result = subprocess.run([str(venv_python), '-c', EVALS_IMPORT_CODE], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    'extra',
    [
        pytest.param(None, id='base'),
        pytest.param('logfire', id='logfire'),
    ],
)
def test_pydantic_evals_documented_imports_succeed(wheelhouse: Path, tmp_path: Path, extra: str | None) -> None:
    wheels = sorted(wheelhouse.glob('pydantic_evals-*.whl'))
    assert wheels, f'no pydantic_evals wheel in {wheelhouse}'
    requirement = str(wheels[0]) if extra is None else f'{wheels[0]}[{extra}]'
    _clean_import(wheelhouse, tmp_path, requirement)
