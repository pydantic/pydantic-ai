from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_test_baselines

_BASELINE = 'tests/leak_guards.py'


def _source(*entries: str, name: str = 'GC_FINALIZED_BASELINE') -> str:
    rows = ''.join(f'        {entry!r},\n' for entry in entries)
    return f'{name}: frozenset[str] = frozenset(\n    {{\n{rows}    }}\n)\n'


def _register(repo: Path, baselines: dict[str, str]) -> None:
    path = repo / check_test_baselines.CHECKER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'BASELINES: dict[str, str] = {baselines!r}\n', encoding='utf-8')


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, _source('a.py:first', 'a.py:second'))
    _register(tmp_path, {'GC_FINALIZED_BASELINE': _BASELINE})
    _git(tmp_path, 'init', '--quiet')
    _commit(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(repo: Path, source: str, baseline: str = _BASELINE) -> None:
    path = repo / baseline
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


def test_the_change_that_introduces_baselines_creates_them(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, '')
    _register(repo, {})
    _commit(repo)
    _write(repo, _source('a.py:first'))
    _register(repo, {'GC_FINALIZED_BASELINE': _BASELINE})

    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_moved_baseline_is_compared_where_the_base_kept_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    (repo / _BASELINE).unlink()
    _write(repo, '# rewritten\n' * 40 + _source('a.py:first', 'a.py:second', 'a.py:third'), 'tests/leaks.py')
    _register(repo, {'GC_FINALIZED_BASELINE': 'tests/leaks.py'})

    assert _check(capsys, '--base', 'HEAD') == (
        1,
        'tests/leaks.py: `GC_FINALIZED_BASELINE` gained `a.py:third`: fix the violation instead; a baseline only '
        'shrinks\n',
    )


@pytest.mark.parametrize(
    ('source', 'baselines', 'gained'),
    [
        pytest.param(
            _source('a.py:first', 'a.py:second') + _source('a.py:third', name='SECOND_BASELINE'),
            {'GC_FINALIZED_BASELINE': _BASELINE, 'SECOND_BASELINE': _BASELINE},
            [('SECOND_BASELINE', 'a.py:third')],
            id='second-baseline',
        ),
        pytest.param(
            _source() + _source('a.py:first', 'a.py:second', name='RENAMED_BASELINE'),
            {'GC_FINALIZED_BASELINE': _BASELINE, 'RENAMED_BASELINE': _BASELINE},
            [('RENAMED_BASELINE', 'a.py:first'), ('RENAMED_BASELINE', 'a.py:second')],
            id='renamed',
        ),
    ],
)
def test_once_the_base_registers_a_baseline_a_change_cannot_create_one(
    repo: Path,
    capsys: pytest.CaptureFixture[str],
    source: str,
    baselines: dict[str, str],
    gained: list[tuple[str, str]],
):
    _write(repo, source)
    _register(repo, baselines)

    assert _check(capsys, '--base', 'HEAD') == (
        1,
        ''.join(
            f'{_BASELINE}: `{name}` gained `{entry}`: fix the violation instead; a baseline only shrinks\n'
            for name, entry in gained
        ),
    )


@pytest.mark.parametrize(
    ('source', 'problem'),
    [
        pytest.param(
            _source('a.py:first') + "GC_FINALIZED_BASELINE = GC_FINALIZED_BASELINE | {'a.py:third'}\n",
            'is assigned more than once: assign it once',
            id='reassigned',
        ),
        pytest.param(
            _source('a.py:first') + "GC_FINALIZED_BASELINE |= {'a.py:third'}\n",
            'is assigned more than once: assign it once',
            id='augmented',
        ),
        pytest.param(
            "GC_FINALIZED_BASELINE = frozenset({'a.py:first'}) | {'a.py:third'}\n",
            'is not `frozenset()` of a set literal of strings',
            id='expression',
        ),
        pytest.param(
            "EXTRA = ('a.py:third',)\nGC_FINALIZED_BASELINE = frozenset({'a.py:first', *EXTRA})\n",
            'is not `frozenset()` of a set literal of strings',
            id='unpacked',
        ),
        pytest.param(
            "GC_FINALIZED_BASELINE = {'a.py:first'}\nGC_FINALIZED_BASELINE.add('a.py:third')\n",
            'is not `frozenset()` of a set literal of strings',
            id='mutable-set',
        ),
        pytest.param(
            "GC_FINALIZED_BASELINE = {'a.py:first': 'reason'}\n",
            'is not `frozenset()` of a set literal of strings',
            id='mutable-dict',
        ),
    ],
)
def test_a_baseline_that_is_not_one_literal_fails(
    repo: Path, capsys: pytest.CaptureFixture[str], source: str, problem: str
):
    _write(repo, source)

    assert _check(capsys, '--base', 'HEAD') == (1, f'{_BASELINE}: `GC_FINALIZED_BASELINE` {problem}\n')


def test_an_emptied_baseline_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'GC_FINALIZED_BASELINE: frozenset[str] = frozenset()\n')

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


def test_a_merge_of_a_commit_without_the_baseline_still_checks_it(repo: Path, capsys: pytest.CaptureFixture[str]):
    _git(repo, 'branch', '-m', 'mine')
    _git(repo, 'checkout', '--quiet', '--orphan', 'other')
    _git(repo, 'rm', '-r', '--quiet', '--cached', '.')
    (repo / _BASELINE).unlink()
    (repo / 'theirs.py').write_text('', encoding='utf-8')
    _commit(repo)
    _git(repo, 'checkout', '--quiet', 'mine')
    _git(
        repo,
        *('-c', 'user.name=test', '-c', 'user.email=test@example.com'),
        *('merge', '--no-commit', '--no-ff', '--allow-unrelated-histories', 'other'),
    )
    _write(repo, _source('a.py:first', 'a.py:second', 'a.py:third'))
    _git(repo, 'add', '--all')

    assert _check(capsys, '--staged') == (
        1,
        'tests/leak_guards.py: `GC_FINALIZED_BASELINE` gained `a.py:third`: fix the violation instead; a baseline only '
        'shrinks\n',
    )


@pytest.mark.parametrize(
    ('baselines', 'error'),
    [
        pytest.param(
            {},
            '`BASELINES` drops `GC_FINALIZED_BASELINE`, which the base holds entries in: empty it first',
            id='dropped',
        ),
        pytest.param(
            {'GC_FINALIZED_BASELINE': 'tests/other.py'},
            '`BASELINES` reads `GC_FINALIZED_BASELINE` from `tests/other.py`, which assigns no such baseline',
            id='unassigned',
        ),
    ],
)
def test_a_registry_that_loses_sight_of_a_baseline_fails(
    repo: Path, capsys: pytest.CaptureFixture[str], baselines: dict[str, str], error: str
):
    _write(repo, _source('a.py:first', 'a.py:second', 'a.py:third'))
    _register(repo, baselines)

    assert _check(capsys, '--base', 'HEAD') == (1, f'{check_test_baselines.CHECKER}: {error}\n')


def test_a_registry_that_is_not_one_literal_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    (repo / check_test_baselines.CHECKER).write_text("BASELINES = {'GC_FINALIZED_BASELINE': PATH}\n", encoding='utf-8')

    assert _check(capsys, '--base', 'HEAD') == (
        1,
        f'{check_test_baselines.CHECKER}: `BASELINES` is not a dict literal of strings\n',
    )


def test_a_baseline_the_base_holds_empty_may_leave_the_registry(repo: Path, capsys: pytest.CaptureFixture[str]):
    _write(repo, 'GC_FINALIZED_BASELINE: frozenset[str] = frozenset()\n')
    _commit(repo)
    (repo / _BASELINE).unlink()
    _register(repo, {})

    assert _check(capsys, '--base', 'HEAD') == (0, '')
