"""`pydantic-ai-harness` documented imports succeed on a clean install.

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

BASE_IMPORT_CODE = """\
import pydantic_ai_harness
from pydantic_ai_harness import BubblewrapSandbox, SSHWorkspace
"""
E2B_IMPORT_CODE = 'from pydantic_ai_harness.e2b_sandbox import E2BSandbox\n'
MODAL_IMPORT_CODE = 'from pydantic_ai_harness.modal_sandbox import ModalSandbox\n'
SPRITES_IMPORT_CODE = 'from pydantic_ai_harness.sprites_sandbox import SpritesSandbox\n'


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


def _clean_import(wheelhouse: Path, tmp_path: Path, requirement: str, import_code: str) -> None:
    venv_dir = tmp_path / 'venv'
    subprocess.run(['uv', 'venv', '--python', sys.executable, str(venv_dir)], check=True)
    venv_python = venv_dir / 'bin' / 'python'
    subprocess.run(
        ['uv', 'pip', 'install', '--python', str(venv_python), '--find-links', str(wheelhouse), requirement],
        check=True,
    )
    result = subprocess.run([str(venv_python), '-c', import_code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ('extra', 'import_code'),
    [
        pytest.param(None, BASE_IMPORT_CODE, id='base'),
        pytest.param('e2b', E2B_IMPORT_CODE, id='e2b'),
        pytest.param('modal', MODAL_IMPORT_CODE, id='modal'),
        pytest.param('sprites', SPRITES_IMPORT_CODE, id='sprites'),
    ],
)
def test_harness_documented_imports_succeed(
    wheelhouse: Path, tmp_path: Path, extra: str | None, import_code: str
) -> None:
    wheels = sorted(wheelhouse.glob('pydantic_ai_harness-*.whl'))
    assert wheels, f'no pydantic_ai_harness wheel in {wheelhouse}'
    requirement = str(wheels[0]) if extra is None else f'{wheels[0]}[{extra}]'
    _clean_import(wheelhouse, tmp_path, requirement, import_code)
