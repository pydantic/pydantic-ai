from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_pragma_comments

_REPO = Path(__file__).parents[1]

_DEBT = """\
def debt(value: int) -> int:
    if value:  # pragma: no cover - never zero
        return 1
    return 2
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, 'pkg/debt.py', _DEBT)
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, name: str, source: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True).stdout


def _commit(repo: Path) -> None:
    _git(repo, 'add', '--all')
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '--quiet', '-m', 'commit')


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    code = check_pragma_comments.main(['--staged'])
    return code, capsys.readouterr().out


@pytest.mark.parametrize(
    'comment',
    [
        '# pragma: no cover - never zero',
        '# pragma: lax no cover  (covered on some platforms)',
        '# pragma: no branch  # the loop always breaks',
        '# pragma: no cover — reason',
        '# pragma: no cover  # pyright: ignore[reportUnknownVariableType]  # and why',
    ],
)
def test_text_after_a_pragma_fails_and_says_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str], comment: str):
    code, out = _staged(repo, capsys, 'tests/test_new.py', f'if True:  {comment}\n    pass\n')

    assert code == 1
    assert f'    tests/test_new.py:1: if True:  {comment}' in out
    assert 'tests/test_new.py: pragma-prose 0 -> 1' in out
    assert f'See {check_pragma_comments.GUIDE}.' in out


@pytest.mark.parametrize(
    'comment',
    [
        '# pragma: no cover',
        '# pragma: lax no cover',
        '# pragma: no branch  # pyright: ignore[reportUnnecessaryComparison]',
        '# pragma: no cover  # pyright: ignore[reportPrivateUsage]  # noqa: BLE001',
    ],
)
def test_a_bare_pragma_or_one_followed_by_tooling_directives_passes(
    repo: Path, capsys: pytest.CaptureFixture[str], comment: str
):
    source = f'# why the branch is excluded\nif True:  {comment}\n    pass\n'

    assert _staged(repo, capsys, 'tests/test_new.py', source) == (0, '')


def test_a_pragma_inside_a_string_is_not_a_comment(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'tests/test_new.py', "TEXT = 'x  # pragma: no cover - not a comment'\n") == (0, '')


def test_existing_lines_pass_until_a_change_adds_one(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'pkg/debt.py', _DEBT + '\n\nVALUE = 1\n') == (0, '')

    code, out = _staged(repo, capsys, 'pkg/debt.py', _DEBT + '\n\nVALUE = 1  # pragma: no cover - unused\n')

    assert code == 1
    assert 'pkg/debt.py: pragma-prose 1 -> 2' in out


def test_the_guide_exists():
    assert (_REPO / check_pragma_comments.GUIDE).is_file()
