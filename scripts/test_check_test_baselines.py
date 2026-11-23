from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_test_baselines

_BASELINE = 'tests/leak_guards.py'


def _source(*entries: str) -> str:
    rows = ''.join(f'        {entry!r},\n' for entry in entries)
    return f'GC_FINALIZED_BASELINE: frozenset[str] = frozenset(\n    {{\n{rows}    }}\n)\n'


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, _source('a.py:first', 'a.py:second'))
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, source: str) -> None:
    path = repo / _BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')


def _git(repo: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


def _commit(repo: Path) -> None:
    _git(repo, 'add', '--all')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '--quiet', '-m', 'commit')


def _check(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = check_test_baselines.main(list(args))
    return code, capsys.readouterr().out


def test_a_baseline_that_shrinks_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _source('a.py:first'))

    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_creating_a_baseline_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, '')
    _commit(repo)
    _write(repo, _source('a.py:first'))

    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_an_added_entry_fails_and_names_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _source('a.py:first', 'a.py:third'))

    assert _check(capsys, '--base', 'HEAD', '--format', 'github') == (
        1,
        '::error file=tests/leak_guards.py::`GC_FINALIZED_BASELINE` gained `a.py:third`: fix the violation instead; '
        'a baseline only shrinks\n',
    )


def test_staged_mode_reads_the_index(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, _source('a.py:first', 'a.py:second', 'a.py:third'))
    _git(repo, 'add', '--all')
    _write(repo, _source('a.py:first'))

    assert _check(capsys, '--staged') == (
        1,
        'tests/leak_guards.py: `GC_FINALIZED_BASELINE` gained `a.py:third`: fix the violation instead; a baseline only '
        'shrinks\n',
    )


def test_a_merge_may_bring_in_the_entries_the_merged_branch_has(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, _source('a.py:first', 'a.py:second', 'a.py:theirs'))
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    (repo / 'mine.py').write_text('', encoding='utf-8')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')

    assert _check(capsys, '--staged') == (0, '')
