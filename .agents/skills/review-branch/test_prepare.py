"""Tests for prepare.py against temporary Git histories.

Run: uv run pytest .agents/skills/review-branch/test_prepare.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare  # noqa: E402


@dataclass
class GitRepo:
    """A repository with one `base` commit on `main` and a `feature` branch checked out."""

    path: Path
    output: Path
    base: str = ''

    def git(self, *args: str) -> str:
        return subprocess.run(
            ['git', '-C', str(self.path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def write(self, name: str, text: str) -> None:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> str:
        self.git('add', '--all')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def snapshot(self, base: str = 'main', head: str = 'HEAD') -> str:
        prepare.prepare(self.path, base, head, self.output)
        return (self.output / 'code.blame-map.txt').read_text()


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    repo = GitRepo(tmp_path, tmp_path / 'review')
    repo.git('init', '-q', '--initial-branch=main')
    repo.git('config', 'user.name', 'Review Test')
    repo.git('config', 'user.email', 'review@example.invalid')
    repo.git('config', 'commit.gpgSign', 'false')
    repo.git('config', 'core.hooksPath', '/dev/null')
    repo.write('file.txt', 'base\n')
    repo.base = repo.commit('base')
    repo.git('checkout', '-qb', 'feature')
    return repo


def test_metadata_like_source_and_multiple_hunks(repo: GitRepo) -> None:
    repo.write('file.txt', ''.join(f'line {i}\n' for i in range(30)))
    repo.commit('long file')
    repo.git('branch', '-f', 'main')
    lines = (repo.path / 'file.txt').read_text().splitlines()
    lines[1] = 'new import'
    lines[20] = 'VERSION = (0, 1, 2)'
    lines[25] = '++ b/fake'
    repo.write('file.txt', '\n'.join(lines) + '\n')
    head = repo.commit('change')
    assert repo.snapshot() == ''.join(f'file.txt:{number} NEW {head}\n' for number in (2, 21, 26))


def test_no_newline_and_quoted_paths(repo: GitRepo) -> None:
    repo.write('file.txt', 'old')
    repo.commit('no newline')
    repo.git('branch', '-f', 'main')
    repo.write('file.txt', 'new\nsecond')
    names = ['é.txt', 'a b.txt', 'tab\tquote"\\.txt', 'new\nline.txt']
    for name in names:
        repo.write(name, 'one\ntwo\n')
    head = repo.commit('paths')
    mapping = repo.snapshot()
    for name in names + ['file.txt']:
        label = json.dumps(name) if name != 'é.txt' and name != 'file.txt' else name
        for number in (1, 2):
            assert f'{label}:{number} NEW {head}\n' in mapping
    assert len(mapping.splitlines()) == 10


def test_dirty_tree_and_moving_head_do_not_change_pinned_snapshot(repo: GitRepo) -> None:
    repo.write('file.txt', 'base\nnew\n')
    head = repo.commit('change')
    expected = repo.snapshot(head=head)
    repo.write('file.txt', 'inserted\nbase\nnew\n')
    assert repo.snapshot(head=head) == expected
    repo.git('add', 'file.txt')
    repo.git('commit', '-qm', 'later change')
    assert repo.snapshot(head=head) == expected
    assert (repo.output / 'head-sha.txt').read_text().strip() == head


def test_committed_merge_resolution_is_new(repo: GitRepo) -> None:
    repo.write('file.txt', 'branch\n')
    repo.commit('branch')
    repo.git('checkout', 'main')
    repo.write('file.txt', 'upstream\n')
    repo.commit('upstream')
    repo.git('checkout', 'feature')
    with pytest.raises(subprocess.CalledProcessError):
        repo.git('merge', 'main')
    with pytest.raises(ValueError, match='unmerged index'):
        repo.snapshot()
    repo.write('file.txt', 'resolved\n')
    repo.git('add', 'file.txt')
    with pytest.raises(ValueError, match='MERGE_HEAD'):
        repo.snapshot()
    head = repo.commit('resolve')
    assert repo.snapshot() == f'file.txt:1 NEW {head}\n'


def test_conflicting_base_fails_before_writing_artifacts(repo: GitRepo) -> None:
    repo.write('file.txt', 'branch\n')
    repo.commit('branch')
    repo.git('checkout', 'main')
    repo.write('file.txt', 'upstream\n')
    repo.commit('upstream')
    repo.git('checkout', 'feature')
    with pytest.raises(subprocess.CalledProcessError):
        repo.snapshot()
    assert not repo.output.exists()


def test_renames_binary_deletions_and_cassettes(repo: GitRepo) -> None:
    repo.write('old.txt', 'line\n' * 20)
    repo.write('deleted.txt', 'gone\n')
    repo.commit('fixtures')
    repo.git('branch', '-f', 'main')
    repo.git('mv', 'old.txt', 'renamed.txt')
    repo.write('renamed.txt', 'line\n' * 20 + 'added\n')
    repo.git('rm', 'deleted.txt')
    (repo.path / 'binary').write_bytes(b'\x00\xff')
    repo.write('tests/cassettes/request.yaml', 'recording\n')
    repo.write('other.cassette.yaml', 'recording\n')
    head = repo.commit('change')
    assert repo.snapshot() == f'renamed.txt:21 NEW {head}\n'
    diff = (repo.output / 'code.diff').read_text()
    assert 'rename from old.txt' in diff
    assert 'recording' not in diff
    assert 'tests/cassettes/request.yaml' in (repo.output / 'cassettes.list').read_text()
    assert 'binary' in (repo.output / 'branch-files.txt').read_text()


def test_bad_ref_does_not_publish_success(repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    assert prepare.main(['--repo', str(repo.path), '--base', 'missing', '--output', str(repo.output)]) != 0
    assert not repo.output.exists()
    assert capsys.readouterr().out == ''


def test_symlinks_and_gitlinks(repo: GitRepo) -> None:
    (repo.path / 'link').symlink_to('file.txt')
    repo.git('update-index', '--add', '--cacheinfo', f'160000,{repo.base},module')
    repo.git('add', 'link')
    repo.git('commit', '-qm', 'links')
    head = repo.git('rev-parse', 'HEAD')
    assert repo.snapshot() == f'link:1 NEW {head}\n'
    assert 'Subproject commit' in (repo.output / 'code.diff').read_text()


def test_blame_error_is_not_preexisting(repo: GitRepo) -> None:
    diff = b'diff --git a/missing b/missing\n+++ b/missing\n@@ -0,0 +1 @@\n+value\n'
    with pytest.raises(subprocess.CalledProcessError):
        prepare.blame_map(repo.path, repo.base, repo.base, diff)


def test_empty_diff(repo: GitRepo) -> None:
    assert repo.snapshot() == ''


def test_blame_distinguishes_base_and_branch_origins(repo: GitRepo) -> None:
    repo.write('file.txt', 'base\nnew\n')
    head = repo.commit('change')
    # Exercise provenance for both origins independently of diff selection.
    diff = b'+++ b/file.txt\n@@ -0,0 +1,2 @@\n+base\n+new\n'
    mapping, counts = prepare.blame_map(repo.path, repo.base, head, diff)
    assert mapping.decode() == f'file.txt:1 PRE-EXISTING {repo.base}\nfile.txt:2 NEW {head}\n'
    assert counts == {'PRE-EXISTING': 1, 'NEW': 1}


def test_truncated_hunk_fails() -> None:
    with pytest.raises(ValueError, match='truncated hunk'):
        prepare.added_lines(b'+++ b/a\n@@ -0,0 +1,2 @@\n+one\n')
