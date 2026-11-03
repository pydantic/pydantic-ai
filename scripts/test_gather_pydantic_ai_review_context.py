"""Regression tests for the review-context gather scripts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from typing_extensions import TypedDict

SCRIPT = Path(__file__).with_name('gather-pydantic-ai-review-context.sh')
DOUWEBOT_SCRIPT = Path(__file__).with_name('gather-review-context.sh')
# Answers the gatherer's GitHub calls for a PR with no comments, threads or linked issues.
FAKE_GH = """#!/usr/bin/env bash
case "$*" in
  *graphql*) echo '{"data":{"repository":{"pullRequest":{"reviewThreads":{"nodes":[],"pageInfo":{"hasNextPage":false}}}}}}' ;;
  *--jq*) ;;
  'pr view'*) echo '{"baseRefName":"main"}' ;;
  *) echo '[]' ;;
esac
"""


class FileChange(TypedDict):
    """Changed-file metadata passed to the partitioning script."""

    filename: str
    additions: int
    deletions: int


def _partition(tmp_path: Path, files: list[FileChange]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    files_path = tmp_path / 'files.json'
    files_path.write_text(json.dumps(files), encoding='utf-8')
    output_dir = tmp_path / 'file-orderings'
    result = subprocess.run(
        ['bash', str(SCRIPT), '--partition-files', str(files_path), str(output_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output_dir


def _assignments(output_dir: Path) -> list[list[str]]:
    count = int((output_dir / 'delegate-count.txt').read_text(encoding='utf-8'))
    return [
        (output_dir / f'delegate-{index}.txt').read_text(encoding='utf-8').splitlines() for index in range(1, count + 1)
    ]


def test_assignments_cover_changed_files_once_with_empty_intersections(tmp_path: Path):
    """Each changed file belongs to exactly one review assignment."""
    files: list[FileChange] = [
        {'filename': f'src/file_{index:02}.py', 'additions': 100, 'deletions': 0} for index in range(11)
    ]

    assignments = _assignments(_partition(tmp_path, files))

    assert len(assignments) == 3
    assert sorted(path for assignment in assignments for path in assignment) == sorted(
        file['filename'] for file in files
    )
    assert all(
        set(assignments[left]).isdisjoint(assignments[right]) for left in range(3) for right in range(left + 1, 3)
    )


def test_assignments_are_deterministic_and_balance_descending_change_sizes(tmp_path: Path):
    """Assignments remain stable and distribute similarly sized changes evenly."""
    weights = [1000, 900, 800, 700, 600, 500]
    files: list[FileChange] = [
        {'filename': f'src/file_{index}.py', 'additions': weight, 'deletions': 0}
        for index, weight in enumerate(weights)
    ]
    first = _partition(tmp_path / 'first', files)
    second = _partition(tmp_path / 'second', list(reversed(files)))

    assignments = _assignments(first)
    assert assignments == _assignments(second)
    changes = {file['filename']: file['additions'] for file in files}
    assert [sum(changes[path] for path in assignment) for assignment in assignments] == [1500, 1500, 1500]
    assert (first / 'largest.txt').read_text(encoding='utf-8').splitlines() == [
        'src/file_0.py',
        'src/file_1.py',
        'src/file_2.py',
        'src/file_3.py',
        'src/file_4.py',
        'src/file_5.py',
    ]


def test_zero_line_files_are_evenly_partitioned_deterministically(tmp_path: Path):
    """Zero-line files spread across every delegate and stay balanced in reverse order."""
    files: list[FileChange] = [
        {'filename': f'src/rename_{index:02}.py', 'additions': 0, 'deletions': 0} for index in range(11)
    ]
    first = _partition(tmp_path / 'first', files)
    second = _partition(tmp_path / 'second', list(reversed(files)))

    assignments = _assignments(first)
    assert assignments == _assignments(second)
    assert len(assignments) == 3
    assert all(assignments)
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1
    assert sorted(path for assignment in assignments for path in assignment) == sorted(
        file['filename'] for file in files
    )


def test_generated_files_are_excluded_from_orderings_and_assignments(tmp_path: Path):
    """Generated files do not appear in reviewer assignments."""
    files: list[FileChange] = [
        {'filename': 'src/one.py', 'additions': 20, 'deletions': 0},
        {'filename': 'src/two.py', 'additions': 10, 'deletions': 0},
        {'filename': 'tests/three.py', 'additions': 5, 'deletions': 0},
        {'filename': 'docs/four.md', 'additions': 1, 'deletions': 0},
        {'filename': 'uv.lock', 'additions': 500, 'deletions': 0},
        {'filename': 'tests/cassettes/review.yaml', 'additions': 500, 'deletions': 0},
    ]

    output_dir = _partition(tmp_path, files)
    ordered_paths = [
        path
        for filename in ['az.txt', 'za.txt', 'largest.txt']
        for path in (output_dir / filename).read_text(encoding='utf-8').splitlines()
    ]
    assigned_paths = [path for assignment in _assignments(output_dir) for path in assignment]

    assert 'uv.lock' not in ordered_paths + assigned_paths
    assert 'tests/cassettes/review.yaml' not in ordered_paths + assigned_paths
    assert sorted(assigned_paths) == ['docs/four.md', 'src/one.py', 'src/two.py', 'tests/three.py']


@pytest.mark.parametrize(
    ('weights', 'expected_delegates'),
    [
        ([100, 50, 50], 0),
        ([100, 51, 50], 2),
        ([100] * 10, 2),
        ([100] * 11, 3),
        ([1001, 0, 0, 0], 3),
    ],
)
def test_existing_size_thresholds_select_delegate_count(tmp_path: Path, weights: list[int], expected_delegates: int):
    """Existing thresholds choose the expected number of delegates."""
    files: list[FileChange] = [
        {'filename': f'src/file_{index}.py', 'additions': weight, 'deletions': 0}
        for index, weight in enumerate(weights)
    ]

    output_dir = _partition(tmp_path, files)

    assert int((output_dir / 'delegate-count.txt').read_text(encoding='utf-8')) == expected_delegates


def _git(repo: Path, env: dict[str, str], *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, env=env, check=True, capture_output=True)


def _douwebot_python_diff(tmp_path: Path, *, gitattributes: str, in_ci: bool) -> tuple[Path, str]:
    """Gather douwebot's context for a PR that changes one of two methods; return the head and its diff."""
    head = tmp_path / 'head'
    head.mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'gh').write_text(FAKE_GH, encoding='utf-8')
    (bin_dir / 'gh').chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != 'GITHUB_ACTIONS'} | {
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 'test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
        # The gatherer fetches the base branch from GitHub; serve it from the head repository.
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': f'url.{head}.insteadOf',
        'GIT_CONFIG_VALUE_0': 'https://github.com/owner/repo.git',
    }
    module = head / 'family.py'
    module.write_text(
        'class Model:\n'
        '    def first(self):\n'
        "        marker = 'first'\n"
        '        return marker, 1, 2, 3\n'
        '\n'
        '\n'
        '\n'
        '\n'
        '    def second(self):\n'
        '        return 2\n',
        encoding='utf-8',
    )
    _git(head, env, 'init', '-q', '-b', 'main')
    _git(head, env, 'add', '.')
    _git(head, env, 'commit', '-qm', 'base')
    _git(head, env, 'checkout', '-qb', 'pr')
    module.write_text(module.read_text(encoding='utf-8').replace('return 2', 'return 3'), encoding='utf-8')
    (head / '.gitattributes').write_text(gitattributes, encoding='utf-8')
    _git(head, env, 'add', '.')
    _git(head, env, 'commit', '-qm', 'pr')

    work = tmp_path / 'work'
    work.mkdir()
    result = subprocess.run(
        ['bash', str(DOUWEBOT_SCRIPT), '1', 'owner/repo'],
        cwd=work,
        env=env | {'PR_HEAD_DIR': str(head)} | ({'GITHUB_ACTIONS': 'true'} if in_ci else {}),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return head, (work / '.github/.review-context/diff/family.py.diff').read_text(encoding='utf-8')


def test_douwebot_diff_in_ci_ignores_a_pr_marking_python_as_binary(tmp_path: Path):
    """In CI, a PR's `*.py -diff` cannot hide its Python changes, and the hunk stops at the changed method."""
    _, diff = _douwebot_python_diff(tmp_path, gitattributes='*.py -diff\n', in_ci=True)

    assert 'Binary files' not in diff
    assert 'return 3' in diff
    assert "marker = 'first'" not in diff


def test_douwebot_diff_outside_ci_leaves_repository_attributes_untouched(tmp_path: Path):
    """A local run gets the same Python hunks without writing the repository's `info/attributes`."""
    head, diff = _douwebot_python_diff(tmp_path, gitattributes='', in_ci=False)

    assert 'return 3' in diff
    assert "marker = 'first'" not in diff
    assert not (head / '.git' / 'info' / 'attributes').exists()
