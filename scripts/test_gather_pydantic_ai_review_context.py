"""Regression tests for the review-context gather scripts."""

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from typing_extensions import TypedDict

SCRIPT = Path(__file__).with_name('gather-pydantic-ai-review-context.sh')
DOUWEBOT_SCRIPT = Path(__file__).with_name('gather-review-context.sh')
# Answers the gatherer's GitHub calls for a PR with no comments, threads or linked issues,
# listing its changed files from `$PR_FILES` and serving its diff from `$PR_DIFF`.
FAKE_GH = """#!/usr/bin/env bash
case "$*" in
  *graphql*) echo '{"data":{"repository":{"pullRequest":{"reviewThreads":{"nodes":[],"pageInfo":{"hasNextPage":false}}}}}}' ;;
  */files*--jq*) jq -r "${@: -1}" "$PR_FILES" ;;
  */files*) cat "$PR_FILES" ;;
  *--jq*) ;;
  'pr view'*) echo '{"baseRefName":"main"}' ;;
  'pr diff'*) cat "$PR_DIFF" ;;
  *) echo '[]' ;;
esac
"""


@dataclass
class Symlink:
    """A symbolic link to write in place of a file's content."""

    target: str


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
    """Generated files do not appear in reviewer assignments, but a file whose path only contains a generated name does."""
    files: list[FileChange] = [
        {'filename': 'src/one.py', 'additions': 20, 'deletions': 0},
        {'filename': 'src/two.py', 'additions': 10, 'deletions': 0},
        {'filename': 'tests/three.py', 'additions': 5, 'deletions': 0},
        {'filename': 'docs/four.md', 'additions': 1, 'deletions': 0},
        {'filename': 'scripts/uv.lock.sh', 'additions': 1, 'deletions': 0},
        {'filename': 'tests/cassettes/conftest.py', 'additions': 1, 'deletions': 0},
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
    assert sorted(assigned_paths) == [
        'docs/four.md',
        'scripts/uv.lock.sh',
        'src/one.py',
        'src/two.py',
        'tests/cassettes/conftest.py',
        'tests/three.py',
    ]


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


def _write_files(root: Path, files: Mapping[str, str | bytes | Symlink | None]) -> None:
    """Write each file, replace it with a symlink, or delete it when its content is `None`."""
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            path.unlink()
        elif isinstance(content, Symlink):
            path.unlink(missing_ok=True)
            path.symlink_to(content.target)
        elif isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding='utf-8')


def _gather(
    tmp_path: Path,
    *,
    script: Path,
    base: Mapping[str, str | bytes],
    pr: Mapping[str, str | bytes | Symlink | None],
    in_ci: bool,
    api_diff: bool = False,
) -> tuple[Path, Path]:
    """Gather review context for a PR that writes `pr` over a `main` branch holding `base`.

    With `api_diff`, the base branch can't be fetched, so the gatherer splits the PR's diff from the API.
    Returns the head repository and the review-context directory.
    """
    head = tmp_path / 'head'
    head.mkdir()
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'gh').write_text(FAKE_GH, encoding='utf-8')
    (bin_dir / 'gh').chmod(0o755)
    files_json = tmp_path / 'files.json'
    pr_diff = tmp_path / 'pr.diff'
    env = {key: value for key, value in os.environ.items() if key != 'GITHUB_ACTIONS'} | {
        'PATH': f'{bin_dir}{os.pathsep}{os.environ["PATH"]}',
        'PR_FILES': str(files_json),
        'PR_DIFF': str(pr_diff),
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 'test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
        # The gatherer fetches the base branch from GitHub; serve it from the head repository,
        # or from a missing one for `api_diff`.
        'GIT_CONFIG_COUNT': '1',
        'GIT_CONFIG_KEY_0': f'url.{tmp_path / "missing" if api_diff else head}.insteadOf',
        'GIT_CONFIG_VALUE_0': 'https://github.com/owner/repo.git',
    }
    _write_files(head, base)
    _git(head, env, 'init', '-q', '-b', 'main')
    _git(head, env, 'add', '.')
    _git(head, env, 'commit', '-qm', 'base')
    _git(head, env, 'checkout', '-qb', 'pr')
    _write_files(head, pr)
    _git(head, env, 'add', '.')
    _git(head, env, 'commit', '-qm', 'pr')
    changed = subprocess.run(
        ['git', 'diff', '--name-only', '-z', 'main', 'HEAD'], cwd=head, env=env, check=True, capture_output=True
    ).stdout.decode()
    files: list[FileChange] = [
        {'filename': name, 'additions': 1, 'deletions': 0} for name in changed.split('\0') if name
    ]
    files_json.write_text(json.dumps(files), encoding='utf-8')
    if api_diff:
        # GitHub quotes paths as git does by default, octal-escaping each byte of a non-ASCII character.
        pr_diff.write_bytes(
            subprocess.run(
                ['git', 'diff', '--no-color', 'main', 'HEAD'], cwd=head, env=env, check=True, capture_output=True
            ).stdout
        )

    work = tmp_path / 'work'
    work.mkdir()
    # douwebot's gatherer runs from the base checkout; CI Review's runs inside the PR checkout.
    if script == DOUWEBOT_SCRIPT:
        cwd, location, context = work, {'PR_HEAD_DIR': str(head)}, work / '.github/.review-context'
    else:
        cwd, location, context = head, {'GITHUB_WORKSPACE': str(work)}, work / '.review-context'
    result = subprocess.run(
        ['bash', str(script), '1', 'owner/repo'],
        cwd=cwd,
        env=env | location | ({'GITHUB_ACTIONS': 'true'} if in_ci else {}),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return head, context


def _gathered_diffs(tmp_path: Path, *, script: Path, gitattributes: str, in_ci: bool) -> tuple[Path, Path]:
    """Gather review context for a PR that changes a method, a Markdown file and a binary.

    Returns the head repository and the per-file diff directory.
    """
    family = (
        'class Model:\n'
        '    def first(self):\n'
        "        marker = 'first'\n"
        '        return marker, 1, 2, 3\n'
        '\n'
        '\n'
        '\n'
        '\n'
        '    def second(self):\n'
        '        return 2\n'
    )
    head, context = _gather(
        tmp_path,
        script=script,
        base={'family.py': family, 'NOTES.md': 'first line\n', 'logo.png': b'\x89PNG\x00\x01'},
        pr={
            'family.py': family.replace('return 2', 'return 3'),
            'NOTES.md': 'first line\nsecond line\n',
            'logo.png': b'\x89PNG\x00\x02',
            '.gitattributes': gitattributes,
        },
        in_ci=in_ci,
    )
    return head, context / 'diff'


def test_douwebot_diff_in_ci_ignores_a_pr_marking_python_as_binary(tmp_path: Path):
    """In CI, a PR's `*.py -diff` cannot hide its Python changes, and the hunk stops at the changed method."""
    _, diffs = _gathered_diffs(tmp_path, script=DOUWEBOT_SCRIPT, gitattributes='*.py -diff\n', in_ci=True)
    diff = (diffs / 'family.py.diff').read_text(encoding='utf-8')

    assert 'Binary files' not in diff
    assert 'return 3' in diff
    assert "marker = 'first'" not in diff


def test_douwebot_diff_in_ci_ignores_a_pr_marking_markdown_as_binary(tmp_path: Path):
    """In CI, a PR's `*.md -diff` cannot hide its Markdown changes, while a real binary stays binary."""
    _, diffs = _gathered_diffs(tmp_path, script=DOUWEBOT_SCRIPT, gitattributes='*.md -diff\n', in_ci=True)
    notes = (diffs / 'NOTES.md.diff').read_text(encoding='utf-8')

    assert 'Binary files' not in notes
    assert 'second line' in notes
    assert 'Binary files' in (diffs / 'logo.png.diff').read_text(encoding='utf-8')


def test_douwebot_diff_outside_ci_leaves_repository_attributes_untouched(tmp_path: Path):
    """A local run gets the same Python hunks without writing the repository's `info/attributes`."""
    head, diffs = _gathered_diffs(tmp_path, script=DOUWEBOT_SCRIPT, gitattributes='', in_ci=False)
    diff = (diffs / 'family.py.diff').read_text(encoding='utf-8')

    assert 'return 3' in diff
    assert "marker = 'first'" not in diff
    assert not (head / '.git' / 'info' / 'attributes').exists()


def test_ci_review_diff_stops_at_the_changed_method(tmp_path: Path):
    """CI Review's gatherer applies the same `python` driver, so its hunk stops at the changed method."""
    _, diffs = _gathered_diffs(tmp_path, script=SCRIPT, gitattributes='', in_ci=False)
    diff = (diffs / 'family.py.diff').read_text(encoding='utf-8')

    assert 'return 3' in diff
    assert "marker = 'first'" not in diff


@pytest.mark.parametrize('api_diff', [False, True], ids=['git-diff', 'api-diff'])
@pytest.mark.parametrize('script', [DOUWEBOT_SCRIPT, SCRIPT], ids=['douwebot', 'ci-review'])
def test_a_crafted_path_cannot_replace_another_files_diff(tmp_path: Path, script: Path, api_diff: bool):
    """Every changed file gets its own readable diff name, so no path can overwrite another file's diff."""
    added = [
        'pydantic_ai_slim/pydantic_ai/_ssrf.py',
        # Each pair below shared one diff name when `/` became `__` and leading dots were stripped,
        # or would share one if `%` were left unescaped.
        'pydantic_ai_slim__pydantic_ai___ssrf.py',
        '.github/ci.yml',
        'github/ci.yml',
        'docs/a.md',
        'docs%2Fa.md',
        # These read as `src.py` and `new.py` when the path was cut after the last ` b/` of `diff --git a/... b/...`.
        'src.py',
        'z b/src.py',
        'new.py',
        # Git quotes these paths in `diff --git`, and `@tsv` escapes their backslash and tab in the index.
        # The API's diff also quotes `é`, as the octal escapes of its two bytes.
        'docs/café.md',
        'docs/a"b\\c\td.md',
        'docs/x\\y.md',
        'docs/x\\\\y.md',
        # Before the index escaped it, this path's tab and newline forged a `src.py` row naming `evil`'s diff.
        'src.py\t+1 -0\tdiff/evil.diff\nzz',
        'evil',
    ]
    renamed = {'q b/new.py': 'old.py'}
    kept = 'kept\n' * 5
    _, context = _gather(
        tmp_path,
        script=script,
        base={'README.md': '', 'old.py': kept},
        pr={path: f'{path}\n' for path in added} | {'old.py': None, 'q b/new.py': f'{kept}q b/new.py\n'},
        in_ci=False,
        api_diff=api_diff,
    )

    rows = [
        row.split('\t', 2) for row in (context / 'changed-files.txt').read_text(encoding='utf-8').split('\n') if row
    ]
    names = {
        'pydantic_ai_slim/pydantic_ai/_ssrf.py': 'diff/pydantic_ai_slim%2Fpydantic_ai%2F_ssrf.py.diff',
        'pydantic_ai_slim__pydantic_ai___ssrf.py': 'diff/pydantic_ai_slim__pydantic_ai___ssrf.py.diff',
        '.github/ci.yml': 'diff/%2Egithub%2Fci.yml.diff',
        'github/ci.yml': 'diff/github%2Fci.yml.diff',
        'docs/a.md': 'diff/docs%2Fa.md.diff',
        'docs%2Fa.md': 'diff/docs%252Fa.md.diff',
        'src.py': 'diff/src.py.diff',
        'z b/src.py': 'diff/z b%2Fsrc.py.diff',
        'new.py': 'diff/new.py.diff',
        'q b/new.py': 'diff/q b%2Fnew.py.diff',
        'docs/café.md': 'diff/docs%2Fcafé.md.diff',
        'docs/a"b\\c\td.md': 'diff/docs%2Fa"b%5Cc%09d.md.diff',
        'docs/x\\y.md': 'diff/docs%2Fx%5Cy.md.diff',
        'docs/x\\\\y.md': 'diff/docs%2Fx%5C%5Cy.md.diff',
        'src.py\t+1 -0\tdiff/evil.diff\nzz': 'diff/src.py%09+1 -0%09diff%2Fevil.diff%0Azz.diff',
        'evil': 'diff/evil.diff',
    }
    # The index is `@tsv`, which escapes a backslash, tab and newline in the path column.
    assert sorted((path, diff) for path, _, diff in rows) == sorted(
        (path.replace('\\', '\\\\').replace('\t', '\\t').replace('\n', '\\n'), diff) for path, diff in names.items()
    )
    # Each file's content is its own path, so the lines a diff adds show whose diff it is.
    assert {
        diff: [
            line
            for line in (context / diff).read_text(encoding='utf-8').split('\n')
            if line.startswith('NL:') and ' +' in line
        ]
        for diff in names.values()
    } == {
        names[path]: [
            f'NL:{6 if path in renamed else number} +{line}' for number, line in enumerate(path.split('\n'), 1)
        ]
        for path in [*added, *renamed]
    }


@pytest.mark.parametrize('script', [DOUWEBOT_SCRIPT, SCRIPT], ids=['douwebot', 'ci-review'])
def test_only_lockfiles_and_cassettes_are_left_out_of_the_review(tmp_path: Path, script: Path):
    """A path that only contains `uv.lock` or `/cassettes/` still gets its diff, so its name can't hide it."""
    generated = ['uv.lock', 'examples/uv.lock', 'tests/models/cassettes/test_model/test_request.yaml']
    reviewed = {
        # pytest loads a `conftest.py` from every directory it collects, cassette directories included.
        'tests/models/cassettes/conftest.py': 'diff/tests%2Fmodels%2Fcassettes%2Fconftest.py.diff',
        'scripts/uv.lock.sh': 'diff/scripts%2Fuv.lock.sh.diff',
        'docs/uv.lock-notes.md': 'diff/docs%2Fuv.lock-notes.md.diff',
        # Cassettes live in a `cassettes/` directory below the root, as the skip's docs say.
        'cassettes/notes.yaml': 'diff/cassettes%2Fnotes.yaml.diff',
    }
    _, context = _gather(
        tmp_path,
        script=script,
        base={'README.md': ''},
        pr={path: f'{path}\n' for path in [*generated, *reviewed]},
        in_ci=False,
    )

    rows = [row.split('\t') for row in (context / 'changed-files.txt').read_text(encoding='utf-8').splitlines()]
    assert sorted(rows) == sorted(
        [[path, '+1 -0'] for path in generated] + [[path, '+1 -0', diff] for path, diff in reviewed.items()]
    )
    assert sorted(path.name for path in (context / 'diff').iterdir()) == sorted(
        diff.removeprefix('diff/') for diff in reviewed.values()
    )
    assert all(f'NL:1 +{path}' in (context / diff).read_text(encoding='utf-8') for path, diff in reviewed.items())
    if script == SCRIPT:
        # CI Review sizes its review from the files it reviews.
        assert (context / 'pr-size.txt').read_text(encoding='utf-8') == (
            '4 files, 4 diff lines (excluding generated files)\n'
        )


@pytest.mark.parametrize('script', [DOUWEBOT_SCRIPT, SCRIPT], ids=['douwebot', 'ci-review'])
def test_a_typechange_keeps_both_halves_of_its_diff(tmp_path: Path, script: Path):
    """Git diffs a file replaced by a symlink as a deletion and then an addition of the same path; both are kept."""
    _, context = _gather(
        tmp_path,
        script=script,
        base={'link': 'regular file\n'},
        pr={'link': Symlink('target.txt')},
        in_ci=False,
    )
    diff = (context / 'diff/link.diff').read_text(encoding='utf-8')

    assert 'OL:1 -regular file' in diff
    assert 'NL:1 +target.txt' in diff
