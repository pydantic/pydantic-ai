from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot

sys.path.insert(0, str(Path(__file__).parent))

from check_pyright_sources import check_count, print_report


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


def _report(files: int, diagnostics: list[dict[str, object]]) -> str:
    severities = [diagnostic['severity'] for diagnostic in diagnostics]
    summary = {
        'filesAnalyzed': files,
        'errorCount': severities.count('error'),
        'warningCount': severities.count('warning'),
        'informationCount': severities.count('information'),
    }
    return json.dumps({'version': '1.1.414', 'generalDiagnostics': diagnostics, 'summary': summary})


@pytest.mark.parametrize(('files', 'code'), [(4, 0), (5, 0), (3, 1)])
def test_fails_when_pyright_reads_fewer_files_than_configured(project: Path, files: int, code: int):
    (project / 'pyright.json').write_text(_report(files, []))

    assert check_count('pyright.json') == code


def test_prints_the_report_as_pyright_does(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Each file's diagnostics in Pyright's own format: 1-based positions, continuation lines indented, rule last."""
    diagnostics: list[dict[str, object]] = [
        {
            'file': '/r/a.py',
            'severity': 'error',
            'message': 'Type "int" is not assignable to return type "str"\n  "int" is not assignable to "str"',
            'range': {'start': {'line': 7, 'character': 11}, 'end': {'line': 7, 'character': 17}},
            'rule': 'reportReturnType',
        },
        {'file': '/r/a.py', 'severity': 'information', 'message': 'Type of "y" is "int"'},
    ]
    (tmp_path / 'pyright.json').write_text(_report(1, diagnostics))

    assert print_report(str(tmp_path / 'pyright.json')) == 1
    assert capsys.readouterr().out == snapshot("""\
/r/a.py
  /r/a.py:8:12 - error: Type "int" is not assignable to return type "str"
    "int" is not assignable to "str" (reportReturnType)
  /r/a.py: information: Type of "y" is "int"
1 error, 0 warnings, 1 information
""")


def test_a_clean_report_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / 'pyright.json').write_text(_report(2, []))

    assert print_report(str(tmp_path / 'pyright.json')) == 0
    assert capsys.readouterr().out == '0 errors, 0 warnings, 0 informations\n'  # codespell:ignore informations


def test_output_that_is_not_a_report_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Pyright prints a configuration error instead of a report: it is shown, and the run fails."""
    (tmp_path / 'pyright.json').write_text('Config file could not be parsed\n')

    assert print_report(str(tmp_path / 'pyright.json')) == 1
    assert 'Config file could not be parsed' in capsys.readouterr().out
