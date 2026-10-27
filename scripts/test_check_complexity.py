from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_complexity
from check_complexity import GUIDE

_GUIDE_PATH = Path(__file__).parents[1] / GUIDE

# A limit of 3 keeps every function in these tests a few lines long.
_PYPROJECT = """\
[tool.complexipy]
paths = ["pkg"]
max-complexity-allowed = 3
"""


def _ifs(name: str, count: int, indent: str = '', marker: str = '') -> str:
    """A function of `count` sequential `if`s, which complexipy scores at exactly `count`."""
    branches = ''.join(f'{indent}    if x == {i}:\n{indent}        return {i}\n' for i in range(count))
    return f'{indent}def {name}(x):{marker}\n{branches}{indent}    return -1\n'


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, 'pyproject.toml', _PYPROJECT)
    _write(tmp_path, 'pkg/__init__.py', '')
    _write(tmp_path, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 2))
    _write(tmp_path, 'outside/loose.py', _ifs('loose', 2))
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
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
    code = check_complexity.main(list(args))
    return code, capsys.readouterr().out


def _stage(repo: Path) -> None:
    _git(repo, 'add', '--all')


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_function_over_the_limit_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/new.py', _ifs('added', 4) + '\n\n' + _ifs('fine', 3))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert '1 function(s) got worse' in out
    assert f'pkg/new.py:1: added 4: new function over the limit ({GUIDE}#new-function)' in out
    assert 'fine' not in out


def test_a_function_taken_over_the_limit_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 4))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert f'small 2 -> 4: this change took it over the limit ({GUIDE}#crossed-the-limit)' in out


def test_debt_that_grows_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert f'debt 5 -> 6: already over the limit, and this change made it worse ({GUIDE}#existing-debt)' in out


@pytest.mark.parametrize('count', [5, 4])
def test_debt_that_holds_or_shrinks_passes(repo: Path, capsys: pytest.CaptureFixture[str], count: int):
    _write(repo, 'pkg/debt.py', _ifs('debt', count) + '\n\n' + _ifs('small', 3))
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_renaming_a_function_without_growing_it_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('renamed', 5) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 0
    assert 'note: pkg/debt.py renamed replaces pkg/debt.py debt, so it is not counted' in out


def test_moving_a_file_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(repo, 'mv', 'pkg/debt.py', 'pkg/moved.py')

    assert _check(capsys, '--staged') == (0, '')


def test_moving_a_function_into_a_class_without_growing_it_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('small', 2))
    _write(repo, 'pkg/owner.py', 'class Owner:\n' + _ifs('debt', 5, indent='    '))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 0
    assert 'note: pkg/owner.py Owner::debt replaces pkg/debt.py debt, so it is not counted' in out


def test_a_move_keeps_its_predecessor_when_a_rename_could_take_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    # `alpha.py` is checked first, and its new function has the removed function's score.
    _write(repo, 'pkg/debt.py', _ifs('small', 2))
    _write(repo, 'pkg/alpha.py', _ifs('unrelated', 5))
    _write(repo, 'pkg/beta.py', _ifs('debt', 5))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'note: pkg/beta.py debt replaces pkg/debt.py debt, so it is not counted' in out
    assert 'pkg/alpha.py:1: unrelated 5: new function over the limit' in out


def test_a_moved_function_that_grows_is_judged_against_its_predecessor(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('small', 2))
    _write(repo, 'pkg/other.py', _ifs('debt', 6))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'pkg/other.py:1: debt 5 -> 6: already over the limit, and this change made it worse' in out


def test_an_overload_stub_does_not_hide_its_implementation(repo: Path, capsys: pytest.CaptureFixture[str]):
    stub = 'from typing import overload\n\n\n@overload\ndef debt(x: int) -> int: ...\n\n\n'
    _write(repo, 'pkg/debt.py', stub + _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'debt 5 -> 6' in out


def test_a_simpler_definition_of_the_same_name_does_not_hide_a_complex_one(
    repo: Path, capsys: pytest.CaptureFixture[str]
):
    getter = _ifs('value', 2, indent='    ').replace('    def value', '    @property\n    def value')
    _write(repo, 'pkg/prop.py', 'class Box:\n' + getter)
    _commit(repo)
    grown = _ifs('value', 4, indent='    ').replace('    def value', '    @property\n    def value')
    setter = '\n    @value.setter\n    def value(self, x):\n        self.x = x\n'
    _write(repo, 'pkg/prop.py', 'class Box:\n' + grown + setter)
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'Box::value 2 -> 4' in out


def test_a_complex_getter_does_not_hide_its_setter(repo: Path, capsys: pytest.CaptureFixture[str]):
    getter = _ifs('value', 5, indent='    ').replace('    def value', '    @property\n    def value')
    setter = '\n' + _ifs('value', 2, indent='    ').replace('    def value', '    @value.setter\n    def value')
    _write(repo, 'pkg/prop.py', 'class Box:\n' + getter + setter)
    _commit(repo)
    grown = '\n' + _ifs('value', 4, indent='    ').replace('    def value', '    @value.setter\n    def value')
    _write(repo, 'pkg/prop.py', 'class Box:\n' + getter + grown)
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'Box::value 2 -> 4: this change took it over the limit' in out
    assert 'Box::value 5' not in out


def test_a_redefinition_keeps_its_most_complex_definition(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/added.py', _ifs('added', 5) + '\n\n' + _ifs('added', 1))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'pkg/added.py:1: added 5: new function over the limit' in out


def test_deleting_a_file_or_a_function_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('small', 2))
    (repo / 'pkg/__init__.py').unlink()
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_files_outside_the_roots_and_other_files_are_not_checked(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'outside/loose.py', _ifs('loose', 9))
    _write(repo, 'pkg/notes.txt', _ifs('text', 9))
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_a_file_that_does_not_parse_is_skipped(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', 'def broken(:\n')
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_staged_mode_reads_the_index_not_the_working_tree(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _stage(repo)
    _write(repo, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 2))

    assert _check(capsys, '--staged')[0] == 1
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_base_mode_reads_the_working_tree_and_untracked_files(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/untracked.py', _ifs('added', 4))
    _write(repo, 'pkg/notes.md', 'not python\n')

    code, out = _check(capsys, '--base', 'HEAD')

    assert code == 1
    assert 'pkg/untracked.py:1: added 4: new function over the limit' in out


def test_named_files_limit_the_check(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _write(repo, 'pkg/untracked.py', _ifs('added', 4))

    code, out = _check(capsys, '--base', 'HEAD', str(repo / 'pkg/untracked.py'))

    assert code == 1
    assert 'pkg/untracked.py' in out
    assert 'debt 5 -> 6' not in out


def test_head_mode_compares_two_commits(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _commit(repo)
    # The working tree is ignored, so restoring it does not change the verdict.
    _write(repo, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 2))

    code, out = _check(capsys, '--base', 'HEAD~1', '--head', 'HEAD')

    assert code == 1
    assert 'debt 5 -> 6' in out


def test_head_needs_a_base(repo: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exc_info:
        check_complexity.main(['--staged', '--head', 'HEAD'])

    assert exc_info.value.code == 2
    assert '--head needs --base' in capsys.readouterr().err


def test_an_ignore_without_a_reason_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6, marker='  # complexipy: ignore') + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert out == f'pkg/debt.py:1: complexipy give the ignore a reason, see {GUIDE}#ignoring-a-function\n'


@pytest.mark.parametrize('marker', ['  # NOQA: complexipy', '  # noqa: complexipy, C901', '  # Complexipy : ignore ()'])
def test_every_spelling_complexipy_honors_needs_a_reason(repo: Path, capsys: pytest.CaptureFixture[str], marker: str):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6, marker=marker) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'give the ignore a reason' in out


@pytest.mark.parametrize(
    'marker', ['  # complexipy: ignore (one branch per value)', '  # complexipy: ignore(one branch per value)']
)
def test_an_ignore_with_a_reason_passes_and_is_reported(repo: Path, capsys: pytest.CaptureFixture[str], marker: str):
    _write(repo, 'pkg/debt.py', _ifs('debt', 6, marker=marker) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged', '--format', 'github')

    assert code == 0
    assert out == (
        f'pkg/debt.py:1: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n'
        '::warning file=pkg/debt.py,line=1,title=complexipy ignore::ignored: one branch per value\n'
    )


@pytest.mark.parametrize('other', ['# complexipy: ignored', '# complexipy: ignore[C003]'])
def test_a_comment_complexipy_does_not_honor_is_not_read_as_the_ignore(
    repo: Path, capsys: pytest.CaptureFixture[str], other: str
):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/debt.py', f'{other}\n' + _ifs('debt', 6, marker=marker) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    assert _check(capsys, '--staged') == (
        0,
        f'pkg/debt.py:2: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n',
    )


def test_a_marker_outside_a_comment_is_not_an_ignore(repo: Path, capsys: pytest.CaptureFixture[str]):
    docstring = 'def documented():\n    """Mark a function with `# complexipy: ignore`."""\n'
    _write(repo, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 2) + '\n\n' + docstring)
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_a_marker_in_a_file_that_does_not_tokenize_is_skipped(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', 'def broken(x:  # complexipy: ignore\n')
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_a_marker_in_a_function_body_is_not_an_ignore(repo: Path, capsys: pytest.CaptureFixture[str]):
    # complexipy still scores the function, so the marker ignores nothing.
    body_marker = _ifs('debt', 5).replace('    if x == 0:', '    if x == 0:  # complexipy: ignore')
    _write(repo, 'pkg/debt.py', body_marker + '\n\n' + _ifs('small', 2))
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_a_copied_marker_on_a_new_function_is_reported(repo: Path, capsys: pytest.CaptureFixture[str]):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/debt.py', _ifs('debt', 5, marker=marker) + '\n\n' + _ifs('small', 2))
    _commit(repo)
    _write(
        repo,
        'pkg/debt.py',
        _ifs('debt', 5, marker=marker) + '\n\n' + _ifs('small', 2) + '\n\n' + _ifs('added', 5, marker=marker),
    )
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 0
    assert out == f'pkg/debt.py:23: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n'


def test_a_marker_above_the_decorators_is_reported_on_its_own_line(repo: Path, capsys: pytest.CaptureFixture[str]):
    decorated = '# complexipy: ignore (one branch per value)\n@staticmethod\n' + _ifs('added', 5)
    _write(repo, 'pkg/added.py', decorated)
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 0
    assert out == f'pkg/added.py:1: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n'


def test_removing_the_reason_from_an_existing_ignore_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'pkg/debt.py', _ifs('debt', 5, marker='  # complexipy: ignore (one branch per value)'))
    _commit(repo)
    _write(repo, 'pkg/debt.py', _ifs('debt', 5, marker='  # complexipy: ignore'))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert out == f'pkg/debt.py:1: complexipy give the ignore a reason, see {GUIDE}#ignoring-a-function\n'


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as `pkg/debt.py` on another branch, then start merging it into this one, which changed another file."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, 'pkg/debt.py', theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'pkg/mine.py', _ifs('mine', 2))
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_made_worse(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _ifs('debt', 6) + '\n\n' + _ifs('small', 2))

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_makes_a_function_worse_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    _merge_in(repo, _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _write(repo, 'pkg/debt.py', _ifs('debt', 7) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert f'debt 6 -> 7: already over the limit, and this change made it worse ({GUIDE}#existing-debt)' in out


def test_an_ignored_method_is_reported_at_its_marker(repo: Path, capsys: pytest.CaptureFixture[str]):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/box.py', 'class Box:\n    size = 1\n\n' + _ifs('value', 6, indent='    ', marker=marker))
    _stage(repo)

    assert _check(capsys, '--staged') == (
        0,
        f'pkg/box.py:4: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n',
    )


@pytest.mark.parametrize(
    ('scope', 'indent', 'line'),
    [('class Box:\n    class Lid:\n', '        ', 3), ('if TYPE_CHECKING:\n    pass\nelse:\n', '    ', 4)],
)
def test_an_ignore_in_a_nested_scope_is_reported_at_its_marker(
    repo: Path, capsys: pytest.CaptureFixture[str], scope: str, indent: str, line: int
):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/box.py', scope + _ifs('value', 6, indent=indent, marker=marker))
    _stage(repo)

    assert _check(capsys, '--staged') == (
        0,
        f'pkg/box.py:{line}: complexipy ignored: one branch per value, see {GUIDE}#ignoring-a-function\n',
    )


def test_an_ignore_that_was_already_there_is_not_reported(repo: Path, capsys: pytest.CaptureFixture[str]):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/debt.py', _ifs('debt', 6, marker=marker) + '\n\n' + _ifs('small', 2))
    _commit(repo)
    _write(repo, 'pkg/debt.py', _ifs('debt', 7, marker=marker) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


def test_an_ignored_function_is_not_a_predecessor_for_a_new_one(repo: Path, capsys: pytest.CaptureFixture[str]):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/debt.py', _ifs('debt', 5, marker=marker) + '\n\n' + _ifs('small', 2))
    _write(repo, 'pkg/owner.py', 'class Owner:\n' + _ifs('debt', 4, indent='    '))
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert 'pkg/owner.py:2: Owner::debt 4: new function over the limit' in out


def test_removing_an_ignore_from_unchanged_debt_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    marker = '  # complexipy: ignore (one branch per value)'
    _write(repo, 'pkg/debt.py', _ifs('debt', 5, marker=marker) + '\n\n' + _ifs('small', 2))
    _commit(repo)
    _write(repo, 'pkg/debt.py', _ifs('debt', 5) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    assert _check(capsys, '--staged') == (0, '')


@pytest.mark.parametrize(
    'config',
    [
        '[tool.complexipy]\npaths = "pkg"\nmax-complexity-allowed = 3\n',
        '[tool.complexipy]\npaths = ["pkg"]\nmax-complexity-allowed = "3"\n',
        '[tool.complexipy]\npaths = ["pkg", 1]\nmax-complexity-allowed = 3\n',
    ],
)
def test_a_malformed_config_stops_the_check(repo: Path, config: str):
    _write(repo, 'pyproject.toml', config)

    with pytest.raises(SystemExit, match=r'\[tool\.complexipy\]'):
        check_complexity.main(['--staged'])


def test_github_format_annotates_and_summarizes(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    summary = repo / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    _write(repo, 'pkg/debt.py', _ifs('debt', 6) + '\n\n' + _ifs('small', 2))
    _stage(repo)

    code, out = _check(capsys, '--staged', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith('::error file=pkg/debt.py,line=1,title=Cognitive complexity 5 -> 6 (limit 3)::')
    assert annotation.endswith(f'({GUIDE}#cases)')
    assert summary.read_text(encoding='utf-8') == (
        '### Cognitive complexity (limit 3)\n'
        '\n'
        '| Function | Change | Failure |\n'
        '|---|---|---|\n'
        '| `pkg/debt.py:1` `debt` | 5 -> 6 | already over the limit, and this change made it worse |\n'
    )


def test_a_failure_lists_complexipy_suggestions(repo: Path, capsys: pytest.CaptureFixture[str]):
    nested = (
        'def deep(x):\n    if x:\n        if x > 1:\n            if x > 2:\n                return 1\n    return 0\n'
    )
    _write(repo, 'pkg/deep.py', nested)
    _stage(repo)

    code, out = _check(capsys, '--staged')

    assert code == 1
    assert re.search(rf'C00\d at line \d+, about -\d+: .+ \({re.escape(GUIDE)}#[a-z-]+\)', out)


def test_an_unmapped_rule_falls_back_to_complexipy_text(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(check_complexity, '_RULES', {})
    offender = _offender(
        'def deep(x):\n    if x:\n        if x > 1:\n            if x > 2:\n                return 1\n'
    )

    advice = check_complexity._advice(offender)

    assert advice[0].endswith(f'({GUIDE}#fixes)')


def _offender(source: str) -> check_complexity._Offender:
    [function] = check_complexity._functions(source).values()
    return check_complexity._Offender('pkg/x.py', function, None, 'new')


def _guide_anchors() -> set[str]:
    headings = re.findall(r'^#+ (.+)$', _GUIDE_PATH.read_text(encoding='utf-8'), flags=re.MULTILINE)
    return {re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-') for heading in headings}


def test_every_anchor_the_report_links_is_a_heading_in_the_guide():
    linked = {
        *(anchor for anchor, _ in check_complexity._FAILURES.values()),
        *(anchor for anchor, _ in check_complexity._RULES.values()),
        'cases',
        'fixes',
        'ignoring-a-function',
    }

    assert linked - _guide_anchors() == set()


def test_the_repository_config_names_existing_roots():
    root = Path(__file__).parents[1]
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['complexipy']

    assert [path for path in config['paths'] if not (root / path).is_dir()] == []
    assert config['max-complexity-allowed'] == 15
