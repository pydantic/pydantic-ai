from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from check_pyright_sources import main


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository whose Pyright configuration names four tracked files, one of them under a dot directory."""
    (tmp_path / 'pyproject.toml').write_text(
        '[tool.pyright]\ninclude = ["pkg", "tool.py", ".github/scripts/ci.py"]\nexclude = ["pkg/skipped.py"]\n'
    )
    for path in ('pkg/__init__.py', 'pkg/types.pyi', 'pkg/skipped.py', 'tool.py', '.github/scripts/ci.py', 'other.py'):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text('')
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ('found', 'code'), [('Found 4 source files', 0), ('Found 5 source files', 0), ('Found 3 source files', 1)]
)
def test_fails_when_pyright_reads_fewer_files_than_configured(project: Path, found: str, code: int):
    (project / 'pyright.log').write_text(f'Auto-excluding **/.*\n{found}\npyright 1.1.414\n')

    assert main('pyright.log') == code


def test_fails_without_a_count(project: Path):
    (project / 'pyright.log').write_text('0 errors, 0 warnings\n')

    assert main('pyright.log') == 1
