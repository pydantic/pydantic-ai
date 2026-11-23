from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_any_ratchet

_REPO = Path(__file__).parents[1]

_DEBT = """\
from typing import Any, cast


def debt(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value)
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py', _DEBT)
    _write(tmp_path, 'pydantic_ai_slim/pydantic_ai/elsewhere.py', 'from typing import Any\n')
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


def _check(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = check_any_ratchet.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def _flagged(out: str) -> list[str]:
    """The `path:line: source` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_any_and_cast_in_a_harness_test_fail_and_say_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(
        repo,
        capsys,
        'tests/harness/test_new.py',
        'import typing\n\n\ndef test(value: typing.Any) -> None:\n    typing.cast(int, value)\n',
    )

    assert code == 1
    assert _flagged(out) == [
        'tests/harness/test_new.py:4: def test(value: typing.Any) -> None:',
        'tests/harness/test_new.py:5: typing.cast(int, value)',
    ]
    assert 'tests/harness/test_new.py: any 0 -> 1' in out
    assert 'tests/harness/test_new.py: cast 0 -> 1' in out
    assert f'See {check_any_ratchet.GUIDE}.' in out


def test_existing_uses_pass_until_a_change_adds_one(repo: Path, capsys: pytest.CaptureFixture[str]):
    path = 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py'
    assert _staged(repo, capsys, path, _DEBT + '\n\ndef more() -> None:\n    pass\n') == (0, '')

    code, out = _staged(repo, capsys, path, _DEBT + '\n\ndef more(other: Any) -> None:\n    pass\n')

    assert code == 1
    assert _flagged(out) == [f'{path}:8: def more(other: Any) -> None:']


def test_removing_a_use_makes_room_for_a_new_one_in_the_same_file(repo: Path, capsys: pytest.CaptureFixture[str]):
    path = 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py'
    replaced = _DEBT.replace('def debt(value: Any)', 'def debt(value: object)')

    assert _staged(repo, capsys, path, replaced + '\n\ndef more(other: Any) -> None:\n    pass\n') == (0, '')


def test_a_function_moved_to_another_changed_file_is_not_counted(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py', 'from typing import Any, cast\n')

    assert _staged(repo, capsys, 'tests/harness/moved.py', _DEBT) == (0, '')


def test_a_renamed_file_keeps_its_count(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(
        repo,
        'mv',
        'src/pydantic_ai_harness/pydantic_ai_harness/debt.py',
        'src/pydantic_ai_harness/pydantic_ai_harness/renamed.py',
    )

    assert _check(capsys, '--staged') == (0, '')


def test_files_outside_the_harness_and_its_tests_are_not_checked(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'pydantic_ai_slim/pydantic_ai/elsewhere.py', _DEBT) == (0, '')
    assert _staged(repo, capsys, 'tests/test_core.py', _DEBT) == (0, '')


def test_another_modules_cast_and_a_string_annotation_are_not_counted(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'import numpy\n\n\ndef test(value: "Any") -> None:\n    numpy.cast(value)\n'

    assert _staged(repo, capsys, 'tests/harness/test_new.py', source) == (0, '')


def _merge_in(repo: Path, theirs: str) -> None:
    """Start a merge of a branch that adds `theirs` to the debt file, leaving it uncommitted."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py', theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'tests/harness/test_mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + '\n\nEXTRA: Any = None\n')

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _DEBT + '\n\nEXTRA: Any = None\n')

    code, out = _staged(
        repo,
        capsys,
        'src/pydantic_ai_harness/pydantic_ai_harness/debt.py',
        _DEBT + '\n\nEXTRA: Any = None\nMORE: Any = None\n',
    )

    assert code == 1
    assert 'src/pydantic_ai_harness/pydantic_ai_harness/debt.py: any 4 -> 5' in out


def test_github_format_annotates_each_added_finding(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'tests/harness/test_new.py', 'from typing import Any\n\nVALUE: Any = None\n')

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith('::error file=tests/harness/test_new.py,line=3,title=Any ratchet%3A any 0 -> 1::')


def test_the_guide_and_roots_exist():
    for path in [check_any_ratchet.GUIDE, *check_any_ratchet.ROOTS]:
        assert (_REPO / path).exists(), path
