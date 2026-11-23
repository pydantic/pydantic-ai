from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_confine
from check_confine import GUIDE

_HARNESS = 'src/pydantic_ai_harness/pydantic_ai_harness'

_DEBT = """\
async def existing(workspace, root, path):
    real = await workspace.realpath(path)
    return real.startswith(root + '/')
"""

_NEW_CHECK = """\
from pathlib import Path


def new(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, f'{_HARNESS}/store.py', _DEBT)
    _write(tmp_path, 'tests/test_loose.py', _NEW_CHECK)
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
    code = check_confine.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_check_fails_and_names_the_primitive(repo: Path, capsys: pytest.CaptureFixture[str]):
    code, out = _staged(repo, capsys, f'{_HARNESS}/loader.py', _NEW_CHECK)

    assert code == 1
    assert f'{_HARNESS}/loader.py: hand-written confine checks 0 -> 1' in out
    assert f'{_HARNESS}/loader.py:4: `new` resolves a path and compares it with a prefix' in out
    assert GUIDE in out


@pytest.mark.parametrize(
    ('resolve', 'compare'),
    [
        ('os.path.realpath(path)', 'real.startswith(root)'),
        ('path.resolve()', 'real.is_relative_to(root)'),
        ('os.path.realpath(path)', 'os.path.commonpath([root, real]) == root'),
        ('os.path.realpath(path)', 'commonpath([root, real]) == root'),
    ],
)
def test_each_resolve_and_compare_pair_counts(
    repo: Path, capsys: pytest.CaptureFixture[str], resolve: str, compare: str
):
    source = f'import os\nfrom os.path import commonpath\n\n\ndef f(root, path):\n    real = {resolve}\n    return {compare}\n'

    assert _staged(repo, capsys, f'{_HARNESS}/new.py', source)[0] == 1


def test_resolving_or_comparing_alone_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'def a(path):\n    return path.resolve()\n\n\ndef b(name):\n    return name.startswith("x")\n'

    assert _staged(repo, capsys, f'{_HARNESS}/new.py', source) == (0, '')


def test_a_nested_function_is_its_own_scope(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = 'def outer(path):\n    real = path.resolve()\n\n    def inner(name):\n        return name.startswith("x")\n\n    return inner(real)\n'

    assert _staged(repo, capsys, f'{_HARNESS}/new.py', source) == (0, '')


def test_existing_checks_pass_until_a_change_adds_one(repo: Path, capsys: pytest.CaptureFixture[str]):
    edited = _DEBT + '\n\nasync def unrelated():\n    return 1\n'
    assert _staged(repo, capsys, f'{_HARNESS}/store.py', edited) == (0, '')

    code, out = _staged(repo, capsys, f'{_HARNESS}/store.py', edited + '\n\n' + _NEW_CHECK)
    assert code == 1
    assert 'hand-written confine checks 1 -> 2' in out


def test_removing_a_check_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, f'{_HARNESS}/store.py', 'X = 1\n') == (0, '')


def test_a_check_moved_between_files_is_not_counted(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, f'{_HARNESS}/store.py', 'X = 1\n')

    assert _staged(repo, capsys, f'{_HARNESS}/moved.py', _DEBT) == (0, '')


def test_code_outside_the_library_roots_is_not_checked(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, 'tests/test_other.py', _NEW_CHECK) == (0, '')


def test_the_primitive_itself_is_exempt(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, GUIDE, _NEW_CHECK) == (0, '')


def test_an_escape_with_a_reason_is_reported_and_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _NEW_CHECK.replace('def new', '# not-confinement: compares for display only\ndef new')

    code, out = _staged(repo, capsys, f'{_HARNESS}/loader.py', source)

    assert code == 0
    assert f'{_HARNESS}/loader.py:5: not confinement: compares for display only, see {GUIDE}' in out


def test_an_escape_on_the_def_line_counts(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _NEW_CHECK.replace('-> bool:', '-> bool:  # not-confinement: a sort key')

    assert _staged(repo, capsys, f'{_HARNESS}/loader.py', source)[0] == 0


def test_an_escape_without_a_reason_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    source = _NEW_CHECK.replace('def new', '# not-confinement:\ndef new')

    code, out = _staged(repo, capsys, f'{_HARNESS}/loader.py', source)

    assert code == 1
    assert 'give a reason' in out


def test_github_format_annotates_the_function(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, f'{_HARNESS}/loader.py', _NEW_CHECK)

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    assert f'::error file={_HARNESS}/loader.py,line=4,title=hand-written confine::' in out


def test_github_format_annotates_an_escape(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, f'{_HARNESS}/loader.py', _NEW_CHECK.replace('def new', '# not-confinement:\ndef new'))

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    assert f'::error file={_HARNESS}/loader.py,line=5,title=not-confinement::give a reason' in out


def test_unparsable_source_is_skipped(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _staged(repo, capsys, f'{_HARNESS}/broken.py', 'def (:\n') == (0, '')


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as the library's `theirs.py` on another branch, then start merging it into this one."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, f'{_HARNESS}/theirs.py', theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'tests/test_mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _NEW_CHECK)

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _NEW_CHECK)

    code, out = _staged(repo, capsys, f'{_HARNESS}/theirs.py', _NEW_CHECK + '\n\n' + _NEW_CHECK.replace('new', 'other'))

    assert code == 1
    assert f'{_HARNESS}/theirs.py: hand-written confine checks 1 -> 2' in out
