from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import check_tool_manager_state
from check_tool_manager_state import OWNER

_REPO = Path(__file__).parents[1]

_CAPABILITY = 'src/pkg/capability.py'

_DEBT = """\
def settle(manager, name):
    manager.succeeded_tools.add(name)
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch repository whose capability module already writes one record, at its first commit."""
    # Keep the developer's git config, such as commit signing or a global hooks path, out of the scratch repo.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    _write(tmp_path, _CAPABILITY, _DEBT)
    _write(tmp_path, OWNER, 'def charge(self, name):\n    self.failed_tools.add(name)\n')
    _write(tmp_path, 'scripts/loose.py', 'manager.failed_tools.add(name)\n')
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
    code = check_tool_manager_state.main(list(args))
    return code, capsys.readouterr().out


def _staged(repo: Path, capsys: pytest.CaptureFixture[str], name: str, source: str) -> tuple[int, str]:
    _write(repo, name, source)
    _git(repo, 'add', '--all')
    return _check(capsys, '--staged')


def _flagged(out: str) -> list[str]:
    """The `path:line: source` rows the report lists."""
    return [line.strip() for line in out.splitlines() if line.startswith('    ')]


def test_an_unchanged_tree_passes_silently(repo: Path, capsys: pytest.CaptureFixture[str]):
    """Nothing changed, so nothing is reported."""
    assert _check(capsys, '--staged') == (0, '')
    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_new_write_fails_and_says_what_to_do(repo: Path, capsys: pytest.CaptureFixture[str]):
    """A write added to a file fails, names the line, and points at the supported operation."""
    code, out = _staged(repo, capsys, _CAPABILITY, _DEBT + '    manager.failed_tools.discard(name)\n')

    assert code == 1
    assert f'{_CAPABILITY}: tool-manager-state 1 -> 2' in out
    assert _flagged(out) == [f'{_CAPABILITY}:3: manager.failed_tools.discard(name)']
    assert '`ToolManager.for_nested_calls`' in out


@pytest.mark.parametrize(
    'line',
    [
        'manager.retries.update({name: 1})',
        'ctx.retries[name] = 1',
        'ctx.retries[name] += 1',
        'del ctx.retries[name]',
        'manager.availability_refused = set()',
        'manager.succeeded_tools: set[str] = set()',
        'ToolManager(toolset, failed_tools=parent.failed_tools)',
        'ToolManager[None](toolset, succeeded_tools=parent.succeeded_tools)',
        'replace(ctx, retries={})',
        'dataclasses.replace(manager, availability_refused=set())',
        'RunContext(deps=None, retries={})',
    ],
)
def test_each_kind_of_write_counts(repo: Path, capsys: pytest.CaptureFixture[str], line: str):
    """Each form of write to a retry record counts."""
    code, out = _staged(repo, capsys, 'src/pkg/new.py', f'{line}\n')

    assert code == 1
    assert _flagged(out) == [f'src/pkg/new.py:1: {line}']


@pytest.mark.parametrize(
    'line',
    [
        'count = ctx.retries.get(name, 0)',
        'failed = name in manager.failed_tools',
        'manager.failed_tools.copy()',
        'manager.tools.update(tools)',
        'Report(failed_tools=names)',
        'replace(ctx, max_retries=2)',
    ],
)
def test_reads_and_other_records_pass(repo: Path, capsys: pytest.CaptureFixture[str], line: str):
    """Reading a record, or writing a field of the same name on another type, is not a write."""
    assert _staged(repo, capsys, 'src/pkg/new.py', f'{line}\n') == (0, '')


def test_the_tool_manager_module_may_write_its_records(repo: Path, capsys: pytest.CaptureFixture[str]):
    """The module that owns the records writes them freely."""
    source = 'def charge(self, name):\n    self.failed_tools.add(name)\n    self.succeeded_tools.discard(name)\n'

    assert _staged(repo, capsys, OWNER, source) == (0, '')


def test_files_outside_the_roots_pass(repo: Path, capsys: pytest.CaptureFixture[str]):
    """Only the library sources are checked."""
    source = 'manager.failed_tools.add(name)\nmanager.retries.clear()\n'

    assert _staged(repo, capsys, 'scripts/loose.py', source) == (0, '')


def test_existing_writes_and_removals_pass(repo: Path, capsys: pytest.CaptureFixture[str]):
    """Writes the base already has pass, and removing one passes."""
    assert _staged(repo, capsys, _CAPABILITY, _DEBT + '\n\ndef unrelated():\n    pass\n') == (0, '')
    assert _staged(repo, capsys, _CAPABILITY, 'def settle(manager, name):\n    pass\n') == (0, '')


def test_a_write_moved_to_another_file_passes(repo: Path, capsys: pytest.CaptureFixture[str]):
    """A write moved between changed files is not new."""
    _write(repo, _CAPABILITY, '')
    _write(repo, 'src/pkg/settle.py', _DEBT)

    assert _check(capsys, '--base', 'HEAD') == (0, '')


def test_a_write_moved_out_of_the_tool_manager_module_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    """A write that leaves the owning module is new where it lands."""
    _write(repo, OWNER, '')
    _write(repo, 'src/pkg/charge.py', 'def charge(self, name):\n    self.failed_tools.add(name)\n')

    code, out = _check(capsys, '--base', 'HEAD')

    assert code == 1
    assert _flagged(out) == ['src/pkg/charge.py:2: self.failed_tools.add(name)']


def test_an_unparsable_file_counts_nothing(repo: Path, capsys: pytest.CaptureFixture[str]):
    """A file that does not parse counts no writes."""
    assert _staged(repo, capsys, 'src/pkg/broken.py', 'manager.failed_tools.add(\n') == (0, '')


def _merge_in(repo: Path, theirs: str) -> None:
    """Commit `theirs` as the capability module on another branch, then start merging it into this one."""
    _git(repo, 'checkout', '--quiet', '-b', 'other')
    _write(repo, _CAPABILITY, theirs)
    _commit(repo)
    _git(repo, 'checkout', '--quiet', '-')
    _write(repo, 'src/pkg/mine.py', '')
    _commit(repo)
    _git(repo, '-c', 'user.name=test', '-c', 'user.email=test@example.com', 'merge', '--no-commit', '--no-ff', 'other')


def test_a_merge_is_not_blamed_for_what_the_merged_branch_added(repo: Path, capsys: pytest.CaptureFixture[str]):
    """During a merge, writes the merged branch brings in are not blamed on the merge."""
    _merge_in(repo, _DEBT + '    manager.failed_tools.add(name)\n')

    assert _check(capsys, '--staged') == (0, '')


def test_a_merge_that_adds_more_than_both_parents_fails(repo: Path, capsys: pytest.CaptureFixture[str]):
    """A merge that adds beyond both parents fails."""
    _merge_in(repo, _DEBT + '    manager.failed_tools.add(name)\n')

    code, out = _staged(
        repo,
        capsys,
        _CAPABILITY,
        _DEBT + '    manager.failed_tools.add(name)\n    manager.failed_tools.discard(name)\n',
    )

    assert code == 1
    assert f'{_CAPABILITY}: tool-manager-state 2 -> 3' in out


def test_github_format_annotates_each_added_write(repo: Path, capsys: pytest.CaptureFixture[str]):
    """The GitHub format annotates each added write."""
    _write(repo, 'src/pkg/new.py', 'import os\n\nmanager.failed_tools.add(name)\n')

    code, out = _check(capsys, '--base', 'HEAD', '--format', 'github')

    assert code == 1
    annotation = next(line for line in out.splitlines() if line.startswith('::error'))
    assert annotation.startswith('::error file=src/pkg/new.py,line=3,title=Tool manager state%3A 0 -> 1::')


def test_the_owner_and_roots_exist():
    """The module and roots the check names exist in this repository."""
    for path in [*check_tool_manager_state.ROOTS, OWNER]:
        assert (_REPO / path).exists(), path
