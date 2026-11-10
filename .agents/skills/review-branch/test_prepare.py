"""Tests for prepare.py against temporary Git histories (stdlib unittest).

Run: uv run pytest .agents/skills/review-branch/test_prepare.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare  # noqa: E402


class ReviewSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.output = self.repo / 'review'
        self.git('init', '-q', '--initial-branch=main')
        self.git('config', 'user.name', 'Review Test')
        self.git('config', 'user.email', 'review@example.invalid')
        self.git('config', 'commit.gpgSign', 'false')
        self.git('config', 'core.hooksPath', '/dev/null')
        self.write('file.txt', 'base\n')
        self.base = self.commit('base')
        self.git('checkout', '-qb', 'feature')

    def git(self, *args: str) -> str:
        return subprocess.run(
            ['git', '-C', str(self.repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write(self, name: str, text: str) -> None:
        target = self.repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> str:
        self.git('add', '--all')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def snapshot(self, base: str = 'main', head: str = 'HEAD') -> str:
        prepare.prepare(self.repo, base, head, self.output)
        return (self.output / 'code.blame-map.txt').read_text()

    def test_metadata_like_source_and_multiple_hunks(self) -> None:
        self.write('file.txt', ''.join(f'line {i}\n' for i in range(30)))
        self.commit('long file')
        self.git('branch', '-f', 'main')
        lines = (self.repo / 'file.txt').read_text().splitlines()
        lines[1] = 'new import'
        lines[20] = 'VERSION = (0, 1, 2)'
        lines[25] = '++ b/fake'
        self.write('file.txt', '\n'.join(lines) + '\n')
        head = self.commit('change')
        self.assertEqual(
            self.snapshot(),
            ''.join(f'file.txt:{number} NEW {head}\n' for number in (2, 21, 26)),
        )

    def test_no_newline_and_quoted_paths(self) -> None:
        self.write('file.txt', 'old')
        self.commit('no newline')
        self.git('branch', '-f', 'main')
        self.write('file.txt', 'new\nsecond')
        names = ['é.txt', 'a b.txt', 'tab\tquote"\\.txt', 'new\nline.txt']
        for name in names:
            self.write(name, 'one\ntwo\n')
        head = self.commit('paths')
        mapping = self.snapshot()
        for name in names + ['file.txt']:
            label = json.dumps(name) if name != 'é.txt' and name != 'file.txt' else name
            for number in (1, 2):
                self.assertIn(f'{label}:{number} NEW {head}\n', mapping)
        self.assertEqual(len(mapping.splitlines()), 10)

    def test_dirty_tree_and_moving_head_do_not_change_pinned_snapshot(self) -> None:
        self.write('file.txt', 'base\nnew\n')
        head = self.commit('change')
        expected = self.snapshot(head=head)
        self.write('file.txt', 'inserted\nbase\nnew\n')
        self.assertEqual(self.snapshot(head=head), expected)
        self.git('add', 'file.txt')
        self.git('commit', '-qm', 'later change')
        self.assertEqual(self.snapshot(head=head), expected)
        self.assertEqual((self.output / 'head-sha.txt').read_text().strip(), head)

    def test_committed_merge_resolution_is_new(self) -> None:
        self.write('file.txt', 'branch\n')
        self.commit('branch')
        self.git('checkout', 'main')
        self.write('file.txt', 'upstream\n')
        self.commit('upstream')
        self.git('checkout', 'feature')
        with self.assertRaises(subprocess.CalledProcessError):
            self.git('merge', 'main')
        with self.assertRaisesRegex(ValueError, 'unmerged index'):
            self.snapshot()
        self.write('file.txt', 'resolved\n')
        self.git('add', 'file.txt')
        with self.assertRaisesRegex(ValueError, 'MERGE_HEAD'):
            self.snapshot()
        head = self.commit('resolve')
        self.assertEqual(self.snapshot(), f'file.txt:1 NEW {head}\n')

    def test_conflicting_base_fails_before_writing_artifacts(self) -> None:
        self.write('file.txt', 'branch\n')
        self.commit('branch')
        self.git('checkout', 'main')
        self.write('file.txt', 'upstream\n')
        self.commit('upstream')
        self.git('checkout', 'feature')
        with self.assertRaises(subprocess.CalledProcessError):
            self.snapshot()
        self.assertFalse(self.output.exists())

    def test_renames_binary_deletions_and_cassettes(self) -> None:
        self.write('old.txt', 'line\n' * 20)
        self.write('deleted.txt', 'gone\n')
        self.commit('fixtures')
        self.git('branch', '-f', 'main')
        self.git('mv', 'old.txt', 'renamed.txt')
        self.write('renamed.txt', 'line\n' * 20 + 'added\n')
        self.git('rm', 'deleted.txt')
        (self.repo / 'binary').write_bytes(b'\x00\xff')
        self.write('tests/cassettes/request.yaml', 'recording\n')
        self.write('other.cassette.yaml', 'recording\n')
        head = self.commit('change')
        self.assertEqual(self.snapshot(), f'renamed.txt:21 NEW {head}\n')
        diff = (self.output / 'code.diff').read_text()
        self.assertIn('rename from old.txt', diff)
        self.assertNotIn('recording', diff)
        self.assertIn('tests/cassettes/request.yaml', (self.output / 'cassettes.list').read_text())
        self.assertIn('binary', (self.output / 'branch-files.txt').read_text())

    def test_bad_ref_does_not_publish_success(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(HERE / 'prepare.py'),
                '--repo',
                str(self.repo),
                '--base',
                'missing',
                '--output',
                str(self.output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(result.stdout, '')

    def test_symlinks_and_gitlinks(self) -> None:
        (self.repo / 'link').symlink_to('file.txt')
        self.git('update-index', '--add', '--cacheinfo', f'160000,{self.base},module')
        self.git('add', 'link')
        self.git('commit', '-qm', 'links')
        head = self.git('rev-parse', 'HEAD')
        self.assertEqual(self.snapshot(), f'link:1 NEW {head}\n')
        self.assertIn('Subproject commit', (self.output / 'code.diff').read_text())

    def test_blame_error_is_not_preexisting(self) -> None:
        diff = b'diff --git a/missing b/missing\n+++ b/missing\n@@ -0,0 +1 @@\n+value\n'
        with self.assertRaises(subprocess.CalledProcessError):
            prepare.blame_map(self.repo, self.base, self.base, diff)

    def test_empty_diff(self) -> None:
        self.assertEqual(self.snapshot(), '')

    def test_blame_distinguishes_base_and_branch_origins(self) -> None:
        self.write('file.txt', 'base\nnew\n')
        head = self.commit('change')
        # Exercise provenance for both origins independently of diff selection.
        diff = b'+++ b/file.txt\n@@ -0,0 +1,2 @@\n+base\n+new\n'
        mapping, counts = prepare.blame_map(self.repo, self.base, head, diff)
        self.assertEqual(
            mapping.decode(),
            f'file.txt:1 PRE-EXISTING {self.base}\nfile.txt:2 NEW {head}\n',
        )
        self.assertEqual(counts, {'PRE-EXISTING': 1, 'NEW': 1})

    def test_truncated_hunk_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, 'truncated hunk'):
            prepare.added_lines(b'+++ b/a\n@@ -0,0 +1,2 @@\n+one\n')


if __name__ == '__main__':
    unittest.main()
