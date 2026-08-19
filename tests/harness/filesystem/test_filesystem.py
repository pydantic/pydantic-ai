"""Tests for the FileSystem capability and FileSystemToolset."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.filesystem import READ_ONLY_TOOL_NAMES, FileSystem
from pydantic_ai_harness.filesystem._toolset import (
    _NOT_A_PATH,
    _OUTSIDE_WORKSPACE,
    FileSystemToolset,
    _content_hash,
    _format_lines,
    _is_binary,
    _sanitize_recoverable_error,
)


class TestFormatLines:
    def test_basic_formatting(self) -> None:
        text = 'line1\nline2\nline3\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert '     1\tline1\n' in result
        assert '     2\tline2\n' in result
        assert '     3\tline3\n' in result

    def test_offset(self) -> None:
        text = 'a\nb\nc\nd\ne\n'
        result = _format_lines(text.splitlines(keepends=True), 2, 2)
        assert '     3\tc\n' in result
        assert '     4\td\n' in result
        assert '... (1 more lines. Use offset=4 to continue reading.)' in result

    def test_offset_exceeds_length(self) -> None:
        text = 'a\nb\n'
        with pytest.raises(ValueError, match='Offset 5 exceeds file length'):
            _format_lines(text.splitlines(keepends=True), 5, 10)

    def test_empty_file(self) -> None:
        result = _format_lines([], 0, 10)
        assert result == '(empty file)\n'

    def test_no_trailing_newline(self) -> None:
        text = 'no newline'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert result.endswith('\n')

    def test_continuation_hint(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(10))
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        assert '... (7 more lines. Use offset=3 to continue reading.)' in result


class TestIsBinary:
    def test_text_content(self) -> None:
        assert _is_binary(b'hello world\n') is False

    def test_binary_content(self) -> None:
        assert _is_binary(b'hello\x00world') is True

    def test_null_after_sample(self) -> None:
        data = b'x' * 9000 + b'\x00'
        assert _is_binary(data) is False

    def test_null_at_boundary(self) -> None:
        data = b'x' * 8191 + b'\x00'
        assert _is_binary(data) is True

    def test_empty(self) -> None:
        assert _is_binary(b'') is False


class TestContentHash:
    def test_deterministic(self) -> None:
        assert _content_hash('hello') == _content_hash('hello')

    def test_different_content(self) -> None:
        assert _content_hash('hello') != _content_hash('world')

    def test_length(self) -> None:
        assert len(_content_hash('test')) == 12


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    (tmp_path / 'hello.txt').write_text('Hello, world!\n')
    (tmp_path / 'multi.txt').write_text('line1\nline2\nline3\nline4\nline5\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.py').write_text('print("nested")\n')
    (tmp_path / '.hidden').write_text('secret\n')
    (tmp_path / 'binary.bin').write_bytes(b'\x00\x01\x02\x03')
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git' / 'config').write_text('[core]\n')
    (tmp_path / '.env').write_text('SECRET_KEY=abc123\n')
    return tmp_path


@pytest.fixture
def toolset(fs_root: Path) -> FileSystemToolset[None]:
    return FileSystemToolset(
        root_dir=fs_root,
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=['.git/*', '.env', '.env.*'],
        max_read_lines=2000,
        max_list_results=1000,
        max_search_results=1000,
        max_find_results=1000,
    )


class TestPathSecurity:
    async def test_traversal_with_dotdot(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('../../../etc/passwd')

    async def test_traversal_absolute_path(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('/etc/passwd')

    async def test_traversal_encoded(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('subdir/../../..')

    async def test_symlink_escape(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """Symlink pointing outside root is rejected."""
        target = fs_root.parent / 'symlink-escape-target'
        target.write_text('escaped!\n')
        try:
            link = fs_root / 'escape_link'
            link.symlink_to(target)
            with pytest.raises(PermissionError, match='resolves outside'):
                toolset._resolve_path('escape_link')
        finally:
            target.unlink(missing_ok=True)

    async def test_valid_path_resolves(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = toolset._resolve_path('hello.txt')
        assert result == (fs_root / 'hello.txt').resolve()

    def test_first_matching_pattern_match(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('secret.key', ['*.txt', '*.key'])
        assert result == '*.key'

    def test_first_matching_pattern_no_match(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('readme.md', ['*.txt', '*.key'])
        assert result is None

    def test_first_matching_pattern_empty(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('anything.py', [])
        assert result is None

    async def test_nested_path_resolves(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._resolve_path('subdir/nested.py')
        assert result.name == 'nested.py'


class TestAccessPatterns:
    async def test_denied_pattern_blocks(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(PermissionError, match='denied by pattern'):
            ts._check_access('data.secret')

    async def test_denied_pattern_passes_non_matching(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # Path that doesn't match any denied pattern should pass
        ts._check_access('data.txt')

    async def test_allowed_pattern_permits(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # Should not raise for .py files
        ts._check_access('test.py')

    async def test_allowed_pattern_blocks_non_matching(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(PermissionError, match='does not match any allowed'):
            ts._check_access('data.txt')

    async def test_protected_pattern_blocks_write(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.git/config', write=True)

    async def test_protected_pattern_allows_read(self, toolset: FileSystemToolset[None]) -> None:
        # Should not raise for read
        toolset._check_access('.git/config', write=False)

    async def test_env_file_protected(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.env', write=True)

    async def test_write_non_protected_with_patterns_configured(self, toolset: FileSystemToolset[None]) -> None:
        # write=True on a path that doesn't match any protected pattern should pass
        toolset._check_access('hello.txt', write=True)

    async def test_access_with_no_denied_patterns(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # No denied, no protected, no allowed → should pass for any path
        ts._check_access('anything.txt', write=True)

    async def test_no_patterns_allows_reads_and_writes(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'Hello, world!' in await ts.read_file('hello.txt')
        assert 'Wrote' in await ts.write_file('hello.txt', 'rewritten\n')

    async def test_protected_path_reads_but_rejects_writes(self, fs_root: Path) -> None:
        (fs_root / 'keys.pem').write_text('PRIVATE\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=['*.pem'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'PRIVATE' in await ts.read_file('keys.pem')
        with pytest.raises(ModelRetry, match='protected'):
            await ts.write_file('keys.pem', 'HACKED\n')

    async def test_denied_pattern_rejects_reads(self, fs_root: Path) -> None:
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'Hello, world!' in await ts.read_file('hello.txt')
        with pytest.raises(ModelRetry, match='denied by pattern'):
            await ts.read_file('creds.secret')

    async def test_allowed_patterns_reject_non_matching_reads(self, fs_root: Path) -> None:
        (fs_root / 'main.py').write_text('print("hi")\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'print' in await ts.read_file('main.py')
        with pytest.raises(ModelRetry, match='does not match any allowed'):
            await ts.read_file('hello.txt')


class TestReadFile:
    async def test_read_basic(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.read_file('hello.txt')
        assert 'Hello, world!' in result
        assert 'hash:' in result
        assert '1 lines' in result

    async def test_read_with_offset(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.read_file('multi.txt', offset=2)
        assert 'line3' in result
        assert 'line1' not in result

    async def test_read_with_limit(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.read_file('multi.txt', limit=2)
        assert 'line1' in result
        assert 'line2' in result
        assert '... (3 more lines' in result

    async def test_read_directory_raises(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='is a directory'):
            await toolset.read_file('subdir')

    async def test_read_missing_raises(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='File not found'):
            await toolset.read_file('nonexistent.txt')

    async def test_read_binary_file(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.read_file('binary.bin')
        assert 'Binary file' in result
        assert '4 bytes' in result

    async def test_read_traversal_blocked(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry):
            await toolset.read_file('../../../etc/passwd')


class TestWriteFile:
    async def test_write_new_file(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = await toolset.write_file('new.txt', 'new content\n')
        assert 'Wrote' in result
        assert (fs_root / 'new.txt').read_text() == 'new content\n'

    async def test_write_nonexistent_parent_raises(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match="Parent directory 'deep/nested' does not exist"):
            await toolset.write_file('deep/nested/file.txt', 'deep\n')

    async def test_write_overwrite(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        await toolset.write_file('hello.txt', 'overwritten\n')
        assert (fs_root / 'hello.txt').read_text() == 'overwritten\n'

    async def test_write_conflict_detection(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # Get current hash
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)

        # Write with correct hash succeeds
        await toolset.write_file('hello.txt', 'updated\n', expected_hash=current_hash)
        assert (fs_root / 'hello.txt').read_text() == 'updated\n'

    async def test_write_conflict_rejection(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        with pytest.raises(ModelRetry, match='Conflict'):
            await toolset.write_file('hello.txt', 'bad\n', expected_hash='wrong_hash_x')

    async def test_write_protected_blocked(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.write_file('.env', 'HACKED=true\n')

    async def test_write_returns_hash(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.write_file('hashed.txt', 'content\n')
        assert 'hash:' in result


class TestEditFile:
    async def test_edit_basic(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hello, universe!')
        assert 'Edited' in result
        assert (fs_root / 'hello.txt').read_text() == 'Hello, universe!\n'

    async def test_edit_not_found_text(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'NONEXISTENT', 'replacement')

    async def test_edit_ambiguous_match(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        (fs_root / 'repeat.txt').write_text('foo bar foo\n')
        with pytest.raises(ModelRetry, match='found 2 times'):
            await toolset.edit_file('repeat.txt', 'foo', 'baz')

    async def test_edit_missing_file(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='File not found'):
            await toolset.edit_file('ghost.txt', 'x', 'y')

    async def test_edit_conflict_detection(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)
        result = await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash=current_hash)
        assert 'hash:' in result

    async def test_edit_conflict_rejection(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='Conflict'):
            await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash='stale_hash_')

    async def test_edit_protected_blocked(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.edit_file('.env', 'SECRET', 'HACKED')

    async def test_edit_returns_new_hash(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Goodbye!')
        assert 'hash:' in result


class TestListDirectory:
    async def test_list_root(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('.')
        assert 'hello.txt' in result
        assert 'subdir/' in result

    async def test_list_subdir(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('subdir')
        assert 'nested.py' in result

    async def test_list_skips_symlink_escaping_root(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """An out-of-root symlink target isn't listed, so its size stays hidden."""
        target = fs_root.parent / 'list-escape-target'
        target.write_text('escaped!\n')
        try:
            (fs_root / 'escape_link.txt').symlink_to(target)
            result = await toolset.list_directory('.')
            assert 'escape_link.txt' not in result
        finally:
            target.unlink(missing_ok=True)

    async def test_list_skips_dangling_symlink(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """A link to a missing in-root target has no size to report, so it isn't listed."""
        (fs_root / 'dangling.txt').symlink_to(fs_root / 'gone.txt')
        result = await toolset.list_directory('.')
        assert 'dangling.txt' not in result
        assert 'hello.txt' in result

    async def test_list_not_a_dir(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry):
            await toolset.list_directory('hello.txt')

    async def test_list_skips_hidden(self, toolset: FileSystemToolset[None]) -> None:
        # Dotfiles/dot-directories are hidden, matching find_files/search_files.
        result = await toolset.list_directory('.')
        assert 'hello.txt' in result
        assert '.hidden' not in result
        assert '.git' not in result

    async def test_list_shows_sizes(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('.')
        assert 'bytes' in result

    async def test_list_shows_dir_indicator(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('.')
        assert 'subdir/' in result

    async def test_list_empty_directory(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        (fs_root / 'empty').mkdir()
        result = await toolset.list_directory('empty')
        assert result == '(empty directory)'

    async def test_list_shows_protected_entries(self, fs_root: Path) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt' in await ts.list_directory('.')

    async def test_list_root_allowed_patterns_filters_entries(self, fs_root: Path) -> None:
        # A file-shaped allowed pattern must not make the root unlistable: '.'
        # is always listed, and entries are filtered against the pattern.
        (fs_root / 'keep.py').write_text('ok\n')
        (fs_root / 'skip.md').write_text('ok\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.')
        assert 'keep.py' in result
        assert 'skip.md' not in result

    async def test_list_hides_denied_entries(self, fs_root: Path) -> None:
        (fs_root / 'visible.txt').write_text('ok\n')
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.')
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_list_truncation(self, fs_root: Path) -> None:
        for i in range(20):
            (fs_root / f'file{i}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=5,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.')
        lines = result.splitlines()
        assert len(lines) == 6
        assert lines[-1] == '[... truncated at 5 entries]'

    async def test_list_at_cap_is_not_marked_truncated(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f'cap{i}.dat').write_text('x\n')
        ts = FileSystemToolset(
            root_dir=tmp_path,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=3,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.')
        assert [line.split(' ')[0] for line in result.splitlines()] == ['cap0.dat', 'cap1.dat', 'cap2.dat']


class TestSearchFiles:
    async def test_search_basic(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('Hello')
        assert 'hello.txt:1:Hello, world!' in result

    async def test_search_regex(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files(r'line\d')
        assert 'multi.txt' in result

    async def test_search_no_matches(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('ZZZZNOTHERE')
        assert result == 'No matches found.'

    async def test_search_skips_hidden(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('secret')
        assert '.hidden' not in result

    async def test_search_skips_binary(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('.')
        assert 'binary.bin' not in result

    async def test_search_invalid_regex(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='Invalid regex'):
            await toolset.search_files('[invalid')

    async def test_search_include_glob(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('print', include_glob='*.py')
        assert 'nested.py' in result

    async def test_search_include_glob_excludes(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('Hello', include_glob='*.py')
        assert result == 'No matches found.'

    async def test_search_in_specific_file(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files('line', path='multi.txt')
        assert 'multi.txt' in result

    async def test_search_truncation(self, fs_root: Path) -> None:
        # Create many matching files
        for i in range(20):
            (fs_root / f'match{i}.txt').write_text('findme\n' * 100)
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=50,
            max_find_results=1000,
        )
        result = await ts.search_files('findme')
        assert 'truncated at 50 matches' in result

    async def test_search_includes_protected_files(self, fs_root: Path) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt:1:protected marker' in await ts.search_files('protected marker')

    async def test_search_skips_denied_files(self, fs_root: Path) -> None:
        (fs_root / 'visible.txt').write_text('lookhere\n')
        (fs_root / 'creds.secret').write_text('lookhere\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.search_files('lookhere')
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_search_only_matches_allowed_files(self, fs_root: Path) -> None:
        # The search root ('.') isn't required to match allowed_patterns; only
        # the matched files are filtered against it per-entry.
        (fs_root / 'keep.py').write_text('findme\n')
        (fs_root / 'skip.md').write_text('findme\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.search_files('findme')
        assert 'keep.py' in result
        assert 'skip.md' not in result

    async def test_search_truncates_within_a_single_file(self, fs_root: Path) -> None:
        (fs_root / 'many.txt').write_text('findme\n' * 100)
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=5,
            max_find_results=1000,
        )
        result = await ts.search_files('findme')
        lines = result.splitlines()
        assert len(lines) == 6
        assert lines[-1] == '[... truncated at 5 matches]'

    async def test_search_at_cap_is_not_marked_truncated(self, fs_root: Path) -> None:
        (fs_root / 'exact.txt').write_text('capmarker\ncapmarker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=2,
            max_find_results=1000,
        )
        result = await ts.search_files('capmarker')
        assert result.splitlines() == ['exact.txt:1:capmarker', 'exact.txt:2:capmarker']

    async def test_search_at_cap_across_files_is_not_marked_truncated(self, fs_root: Path) -> None:
        (fs_root / 'one.txt').write_text('capmarker\n')
        (fs_root / 'two.txt').write_text('capmarker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=2,
            max_find_results=1000,
        )
        result = await ts.search_files('capmarker')
        assert result.splitlines() == ['one.txt:1:capmarker', 'two.txt:1:capmarker']

    async def test_search_with_zero_cap_returns_no_matches(self, fs_root: Path) -> None:
        # FileSystemToolset is public, so it can be built without the capability's
        # positive-integer validation.
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=0,
            max_find_results=1000,
        )
        result = await ts.search_files('Hello')
        assert result == '[... truncated at 0 matches]'

    async def test_search_skips_symlink_escaping_root(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """A symlink to an out-of-root file is skipped, matching `read_file`."""
        target = fs_root.parent / 'search-escape-target'
        target.write_text('escaped marker\n')
        try:
            (fs_root / 'escape_link.txt').symlink_to(target)
            result = await toolset.search_files('escaped marker')
            assert result == 'No matches found.'
        finally:
            target.unlink(missing_ok=True)


class TestFindFiles:
    async def test_find_glob(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*.txt')
        assert 'hello.txt' in result
        assert 'multi.txt' in result

    async def test_find_recursive(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('**/*.py')
        assert 'nested.py' in result

    async def test_find_no_matches(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*.xyz')
        assert result == 'No matches found.'

    async def test_find_skips_hidden(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*')
        assert '.hidden' not in result
        assert '.git' not in result

    async def test_find_skips_symlink_escaping_root(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        target = fs_root.parent / 'find-escape-target'
        target.write_text('escaped!\n')
        try:
            (fs_root / 'escape_link.txt').symlink_to(target)
            result = await toolset.find_files('*.txt')
            assert 'escape_link.txt' not in result
            assert 'hello.txt' in result
        finally:
            target.unlink(missing_ok=True)

    async def test_find_skips_dangling_symlink(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        (fs_root / 'dangling.txt').symlink_to(fs_root / 'gone.txt')
        result = await toolset.find_files('*.txt')
        assert 'dangling.txt' not in result
        assert 'hello.txt' in result

    async def test_find_absolute_pattern_rejected(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='must be relative to the search path'):
            await toolset.find_files('/etc/*')

    async def test_find_at_cap_is_not_marked_truncated(self, fs_root: Path) -> None:
        for i in range(3):
            (fs_root / f'cap{i}.dat').write_text('x\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=3,
        )
        result = await ts.find_files('*.dat')
        assert result.splitlines() == ['cap0.dat', 'cap1.dat', 'cap2.dat']

    async def test_find_not_a_dir(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry):
            await toolset.find_files('*.txt', path='hello.txt')

    async def test_find_in_subdir(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*.py', path='subdir')
        assert 'nested.py' in result

    async def test_find_directories(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('sub*')
        assert 'subdir/' in result

    async def test_find_truncation(self, fs_root: Path) -> None:
        for i in range(20):
            (fs_root / f'file{i}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=5,
        )
        result = await ts.find_files('*.dat')
        assert 'truncated at 5 matches' in result

    async def test_find_includes_protected_files(self, fs_root: Path) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt' in await ts.find_files('*.txt')

    async def test_find_hides_denied_entries(self, fs_root: Path) -> None:
        (fs_root / 'visible.txt').write_text('ok\n')
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.find_files('*')
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_find_only_shows_allowed_entries(self, fs_root: Path) -> None:
        # The find root ('.') isn't required to match allowed_patterns; only
        # the matched entries are filtered against it per-entry.
        (fs_root / 'keep.py').write_text('ok\n')
        (fs_root / 'skip.md').write_text('ok\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.find_files('*')
        assert 'keep.py' in result
        assert 'skip.md' not in result


class TestWalkerEntryResolution:
    """Walkers authorize the entry's resolved target, matching `read_file`."""

    async def test_denied_target_alias_hidden_from_walkers(self, fs_root: Path) -> None:
        (fs_root / 'creds.secret').write_text('hunter2\n')
        (fs_root / 'alias.txt').symlink_to(fs_root / 'creds.secret')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(ModelRetry, match='denied by pattern'):
            await ts.read_file('alias.txt')
        assert 'alias.txt' not in await ts.list_directory('.')
        assert 'alias.txt' not in await ts.find_files('*')
        assert await ts.search_files('hunter2') == 'No matches found.'

    async def test_alias_does_not_smuggle_a_file_past_allowed_patterns(self, fs_root: Path) -> None:
        (fs_root / 'notes.md').write_text('hunter2\n')
        (fs_root / 'alias.py').symlink_to(fs_root / 'notes.md')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(ModelRetry, match='does not match any allowed'):
            await ts.read_file('alias.py')
        assert 'alias.py' not in await ts.list_directory('.')
        assert 'alias.py' not in await ts.find_files('*')
        assert await ts.search_files('hunter2') == 'No matches found.'

    async def test_in_root_alias_to_allowed_target_stays_visible(self, fs_root: Path) -> None:
        (fs_root / 'alias.txt').symlink_to(fs_root / 'hello.txt')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'alias.txt' in await ts.list_directory('.')
        assert 'alias.txt' in await ts.find_files('*.txt')
        assert 'alias.txt:1:Hello, world!' in await ts.search_files('Hello')


class TestCreateDirectory:
    async def test_create_basic(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = await toolset.create_directory('newdir')
        assert 'Created directory' in result
        assert (fs_root / 'newdir').is_dir()

    async def test_create_nested(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        await toolset.create_directory('a/b/c')
        assert (fs_root / 'a' / 'b' / 'c').is_dir()

    async def test_create_existing_ok(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.create_directory('subdir')
        assert 'Created directory' in result

    async def test_create_protected_blocked(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.create_directory('.git/hooks')


class TestFileInfo:
    async def test_info_file(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('hello.txt')
        assert 'type: file' in result
        assert 'size:' in result
        assert 'lines:' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_info_directory(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('subdir')
        assert 'type: directory' in result

    async def test_info_binary(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('binary.bin')
        assert 'binary: True' in result
        assert 'lines:' not in result

    async def test_info_not_found(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='Path not found'):
            await toolset.file_info('nonexistent')

    async def test_info_symlink(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # The link target is stored as an absolute path; the tool must report
        # it relative to the root, never as the absolute host path.
        link = fs_root / 'link.txt'
        link.symlink_to(fs_root / 'hello.txt')
        result = await toolset.file_info('link.txt')
        assert 'type: file' in result
        assert 'symlink_target: hello.txt' in result
        _assert_no_host_root(result, fs_root)


class TestMutationKillers:
    async def test_format_lines_offset_equals_total(self) -> None:
        text = 'a\nb\n'  # 2 lines
        with pytest.raises(ValueError, match='Offset 2 exceeds file length'):
            _format_lines(text.splitlines(keepends=True), 2, 10)

    async def test_format_lines_exact_fit_no_continuation(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_exact_fit_from_offset(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 1, 2)  # lines 2-3, 0 remaining
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_one_line_remaining(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 0, 2)
        assert '... (1 more lines. Use offset=2 to continue reading.)' in result

    async def test_format_lines_line_number_starts_at_one(self) -> None:
        text = 'first\nsecond\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert '     1\tfirst\n' in result
        assert '     0\t' not in result

    async def test_format_lines_offset_line_numbering(self) -> None:
        text = 'a\nb\nc\n'
        result = _format_lines(text.splitlines(keepends=True), 1, 2)
        assert '     2\tb\n' in result
        assert '     3\tc\n' in result

    async def test_is_binary_exactly_at_sample_boundary(self) -> None:
        # Null byte at position 8191 (index 8191, within first 8192 bytes)
        data = b'x' * 8191 + b'\x00'
        assert _is_binary(data) is True
        # Null byte at position 8192 (outside the sample)
        data2 = b'x' * 8192 + b'\x00'
        assert _is_binary(data2) is False

    async def test_content_hash_returns_exactly_12_chars(self) -> None:
        h = _content_hash('test content')
        assert len(h) == 12
        # Verify it's hex characters
        assert all(c in '0123456789abcdef' for c in h)

    async def test_write_file_with_hash_on_new_file(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """When a file doesn't exist, expected_hash should be ignored and the write should succeed."""
        result = await toolset.write_file('brand_new.txt', 'new content\n', expected_hash='any_hash_val')
        assert 'Wrote' in result
        assert (fs_root / 'brand_new.txt').read_text() == 'new content\n'

    async def test_edit_file_single_match_succeeds(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        (fs_root / 'unique.txt').write_text('unique text here\n')
        result = await toolset.edit_file('unique.txt', 'unique text', 'replaced text')
        assert 'Edited' in result
        assert (fs_root / 'unique.txt').read_text() == 'replaced text here\n'

    async def test_edit_file_zero_matches_raises(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'DEFINITELY NOT IN FILE', 'x')

    async def test_search_truncation_stops_after_limit(self, fs_root: Path) -> None:
        # Create many files with 1 match each so truncation is per-file
        for i in range(10):
            (fs_root / f'searchable{i}.txt').write_text(f'match_this_{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=5,
            max_find_results=1000,
        )
        result = await ts.search_files('match_this')
        lines = result.strip().split('\n')
        # Truncation check is after each file, so 5 matches + truncation msg
        # Ensure we don't get all 10 matches
        match_lines = [ln for ln in lines if ln.startswith('searchable')]
        assert len(match_lines) <= 5
        assert 'truncated at 5 matches' in lines[-1]

    async def test_find_truncation_stops_after_limit(self, fs_root: Path) -> None:
        for i in range(10):
            (fs_root / f'findme{i:02d}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=3,
        )
        result = await ts.find_files('*.dat')
        lines = result.strip().split('\n')
        # Should have exactly 4 lines: 3 matches + 1 truncation message
        assert len(lines) == 4
        assert 'truncated at 3 matches' in lines[-1]

    async def test_read_file_default_limit_used(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # Create file with more lines than we'd see with limit=0
        (fs_root / 'big.txt').write_text('\n'.join(f'line{i}' for i in range(100)) + '\n')
        result = await toolset.read_file('big.txt')
        # All 100 lines should be present since max_read_lines is 2000
        assert 'line99' in result

    async def test_list_directory_with_files_not_empty(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('subdir')
        assert result != '(empty directory)'
        assert 'nested.py' in result

    async def test_search_in_file_returns_only_that_file(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # Both files contain 'Hello' / 'hello' but searching a specific file should only return from that file
        (fs_root / 'other.txt').write_text('Hello from other\n')
        result = await toolset.search_files('Hello', path='hello.txt')
        assert 'hello.txt' in result
        assert 'other.txt' not in result

    async def test_file_info_non_binary_shows_lines_and_hash(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('hello.txt')
        assert 'lines: 1' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_file_info_binary_no_lines_no_hash(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('binary.bin')
        assert 'binary: True' in result
        assert 'lines:' not in result
        assert 'hash:' not in result

    async def test_safe_resolve_passes_write_flag(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # Protected patterns block writes but allow reads
        (fs_root / '.env.local').write_text('SECRET=x\n')
        # Read should work (write=False internally)
        result = await toolset.read_file('.env.local')
        assert 'SECRET=x' in result
        # Write should be blocked (write=True internally)
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.write_file('.env.local', 'HACKED\n')

    async def test_format_lines_join_separator(self) -> None:
        """Verify the result doesn't contain garbage between lines."""
        text = 'a\nb\nc\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        # Lines should be directly adjacent (no separator between them)
        assert '     1\ta\n     2\tb\n     3\tc\n' in result

    async def test_format_lines_no_trailing_newline_preserves_content(self) -> None:
        text = 'no newline'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        # The content must still be present
        assert 'no newline' in result
        assert result.endswith('\n')

    async def test_read_file_hash_is_real_hash(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.read_file('hello.txt')
        # The actual hash should be a hex string, not 'None'
        assert 'hash:None' not in result
        # Verify the hash matches what we'd compute
        expected_hash = _content_hash('Hello, world!\n')
        assert f'hash:{expected_hash}' in result

    async def test_read_file_non_ascii_content(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """With invalid UTF-8 bytes, the tool should not crash -- it should use replacement chars."""
        # Write raw bytes that are invalid UTF-8
        (fs_root / 'broken_utf8.txt').write_bytes(b'hello \xff\xfe world\n')
        result = await toolset.read_file('broken_utf8.txt')
        # Should not crash, content should contain replacement characters
        assert 'hello' in result
        assert 'world' in result

    async def test_read_file_default_offset_starts_at_first_line(self, toolset: FileSystemToolset[None]) -> None:
        """The first line must be included when no offset is specified."""
        result = await toolset.read_file('multi.txt')
        # First line must be present (line1)
        assert '     1\tline1' in result
        # Verify line numbering starts at 1
        assert '     0\t' not in result

    async def test_toolset_tool_names(self, toolset: FileSystemToolset[None]) -> None:
        """Verify tools are registered with correct names."""
        tool_names = set(toolset.tools.keys())
        assert 'read_file' in tool_names
        assert 'write_file' in tool_names
        assert 'edit_file' in tool_names
        assert 'list_directory' in tool_names
        assert 'search_files' in tool_names
        assert 'find_files' in tool_names
        assert 'create_directory' in tool_names
        assert 'file_info' in tool_names

    async def test_write_file_output_format(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = await toolset.write_file('fmt.txt', 'ab\ncd\n')
        # Verify specific format: chars, lines, path, hash
        assert 'Wrote 6 chars (2 lines) to fmt.txt.' in result
        assert 'hash:' in result
        # Verify hash is a real hex hash not None
        assert 'hash:None' not in result

    async def test_edit_file_output_format(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hi')
        assert result.startswith('Edited hello.txt.')
        assert 'hash:' in result
        assert 'hash:None' not in result

    def test_format_lines_no_double_trailing_newline(self) -> None:
        """Text that already ends with newline must NOT get a second one appended."""
        text = 'hello\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        # Exact match: no trailing double newline
        assert result == '     1\thello\n'

    def test_safe_resolve_write_default_is_false(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """Protected files should be readable via _safe_resolve's default (write=False)."""
        (fs_root / '.env.local').write_text('SECRET=x\n')
        # _safe_resolve without write= uses default write=False → read is allowed
        resolved = toolset._safe_resolve('.env.local')
        assert resolved.name == '.env.local'
        # But with write=True, it should raise. `_safe_resolve` is an internal
        # helper, so it raises the native PermissionError; the `ModelRetry`
        # conversion happens in the public tool methods that wrap it.
        with pytest.raises(PermissionError, match='protected'):
            toolset._safe_resolve('.env.local', write=True)

    async def test_list_directory_exact_size(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('.')
        # hello.txt has 'Hello, world!\n' = 14 bytes
        assert '14 bytes' in result

    async def test_list_directory_no_garbage_separator(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.list_directory('.')
        assert 'XX' not in result

    async def test_list_directory_error_message(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='Not a directory'):
            await toolset.list_directory('hello.txt')

    async def test_find_files_error_message(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(ModelRetry, match='Not a directory'):
            await toolset.find_files('*.txt', path='hello.txt')

    async def test_find_files_no_suffix_on_files(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*')
        for line in result.splitlines():
            if not line.endswith('/'):
                assert 'XXXX' not in line

    async def test_find_files_no_garbage_separator(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.find_files('*.txt')
        assert 'XX' not in result

    async def test_search_files_no_garbage_separator(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.search_files(r'line\d')
        assert 'XX' not in result

    async def test_file_info_exact_size(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('hello.txt')
        assert '14 bytes' in result

    async def test_file_info_no_garbage_separator(self, toolset: FileSystemToolset[None]) -> None:
        result = await toolset.file_info('hello.txt')
        assert 'XX' not in result

    async def test_search_with_invalid_utf8_file(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        """A file with invalid UTF-8 (but no null bytes = not binary) should be searchable."""
        # Write a file with invalid UTF-8 but no null bytes (not detected as binary)
        (fs_root / 'bad_encoding.txt').write_bytes(b'marker_text \xff\xfe end\n')
        result = await toolset.search_files('marker_text')
        # Should find the file even with broken encoding
        assert 'bad_encoding.txt' in result

    async def test_search_binary_skip_does_not_stop_iteration(self, toolset: FileSystemToolset[None]) -> None:
        """A binary file must be skipped, but subsequent text files must still be searched."""
        # binary.bin exists in the fixture and comes before 'hello.txt' alphabetically
        result = await toolset.search_files('Hello')
        # hello.txt must still be found (binary.bin didn't break the loop)
        assert 'hello.txt' in result

    async def test_find_hidden_skip_does_not_stop_iteration(self, toolset: FileSystemToolset[None]) -> None:
        """Hidden files must be skipped, but subsequent visible files must still appear."""
        # .hidden comes before hello.txt alphabetically -- skipping must not break the loop
        result = await toolset.find_files('*')
        assert 'hello.txt' in result
        assert 'multi.txt' in result


class TestFileSystemCapability:
    def test_default_construction(self) -> None:
        fs = FileSystem()
        assert fs.root_dir == '.'
        assert fs.max_read_lines == 2000

    def test_custom_construction(self, tmp_path: Path) -> None:
        fs = FileSystem(
            root_dir=tmp_path,
            allowed_patterns=['*.py'],
            denied_patterns=['test_*'],
            max_read_lines=500,
        )
        assert fs.max_read_lines == 500

    def test_get_toolset_returns_toolset(self, tmp_path: Path) -> None:
        fs = FileSystem(root_dir=tmp_path)
        toolset = fs.get_toolset()
        assert isinstance(toolset, FileSystemToolset)

    async def test_read_only_exposes_exactly_read_only_tools(self, tmp_path: Path) -> None:
        filesystem = FileSystem[None](root_dir=tmp_path, read_only=True)
        context = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='test')

        tools = await filesystem.get_toolset().get_tools(context)

        assert set(tools) == READ_ONLY_TOOL_NAMES
        assert 'write_file' not in tools

    def test_search_files_description_has_string_return_type(self) -> None:
        toolset = FileSystem().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        description = toolset.tools['search_files'].description

        assert description == (
            '<summary>Search file contents using a regular expression.</summary>\n'
            '<returns>\n'
            '<type>str</type>\n'
            '<description>Matching lines formatted as file:line_number:text.</description>\n'
            '</returns>'
        )

    def test_protected_defaults(self) -> None:
        fs = FileSystem()
        assert '.git/*' in fs.protected_patterns
        assert '.env' in fs.protected_patterns

    def test_non_positive_max_read_lines_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines=0)
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines=-1)

    def test_non_positive_max_list_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_list_results must be a positive integer'):
            FileSystem(max_list_results=0)

    def test_non_positive_max_search_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_search_results must be a positive integer'):
            FileSystem(max_search_results=0)

    def test_non_positive_max_find_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_find_results must be a positive integer'):
            FileSystem(max_find_results=-1)

    def test_non_integer_max_read_lines_rejected(self) -> None:
        # Runtime validation: dataclass annotations are advisory, so a string
        # slipped in from a config must be rejected, not propagated.
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines='1000')  # type: ignore[arg-type]

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_integration(self, tmp_path: Path, anyio_backend: object) -> None:
        if str(anyio_backend) != 'asyncio':
            pytest.skip('Agent.run requires asyncio event loop')
        (tmp_path / 'test.txt').write_text('hello agent\n')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[FileSystem(root_dir=tmp_path)])
        result = await agent.run('read test.txt')
        assert result.output == 'done'


class TestPatternCanonicalization:
    """Sec#3: patterns match the canonical path, and a leading `**/` also
    covers the repository root."""

    async def test_denied_pattern_not_bypassed_by_dot_segment(self, fs_root: Path) -> None:
        (fs_root / 'config').mkdir()
        (fs_root / 'config' / 'secret.txt').write_text('token\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['config/secret.txt'],
            protected_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # A './' segment must not slip the file past its deny rule.
        with pytest.raises(ModelRetry, match='denied'):
            await ts.read_file('config/./secret.txt')

    async def test_root_level_secrets_readable_but_protected_from_write(self, fs_root: Path) -> None:
        (fs_root / 'secrets.yaml').write_text('api: PRIVATE KEY material\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=['**/secrets*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'secrets.yaml:1:api: PRIVATE KEY material' in await ts.search_files('PRIVATE KEY')
        with pytest.raises(ModelRetry, match='protected'):
            await ts.write_file('secrets.yaml', 'changed\n')


def _assert_no_host_root(message: str, root: Path) -> None:
    """Fail if `message` contains the absolute workspace root."""
    assert str(root) not in message
    assert str(root.resolve()) not in message


class TestModelSafeRecoverableErrors:
    """OS-raised filesystem errors must not leak absolute host paths into `ModelRetry`."""

    async def test_write_through_file_hides_host_path(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # The parent path 'hello.txt' exists as a file, so the OS itself raises
        # the error, with the absolute host path as its filename.
        with pytest.raises(ModelRetry) as exc_info:
            await toolset.write_file('hello.txt/nested', 'x')
        message = str(exc_info.value)
        _assert_no_host_root(message, fs_root)
        assert "'hello.txt/nested'" in message
        assert 'Errno' in message

    def test_outside_root_path_is_redacted(self, fs_root: Path) -> None:
        real_root = Path(os.path.realpath(fs_root))
        outside = fs_root.parent / 'private' / 'secret.txt'
        error = FileNotFoundError(errno.ENOENT, 'No such file or directory', str(outside))
        message = _sanitize_recoverable_error(error, real_root)
        assert str(outside) not in message
        assert _OUTSIDE_WORKSPACE in message

    def test_relative_filename_is_preserved(self, fs_root: Path) -> None:
        real_root = Path(os.path.realpath(fs_root))
        error = PermissionError(errno.EACCES, 'Permission denied', 'hello.txt')
        message = _sanitize_recoverable_error(error, real_root)
        assert message == f"[Errno {errno.EACCES}] Permission denied: 'hello.txt'"

    def test_non_path_filename_is_labeled(self, fs_root: Path) -> None:
        real_root = Path(os.path.realpath(fs_root))
        error = OSError(errno.ENOENT, 'No such file or directory')
        error.filename = object()
        message = _sanitize_recoverable_error(error, real_root)
        assert _NOT_A_PATH in message

    def test_symlink_alias_is_normalized(self, fs_root: Path) -> None:
        # macOS reports tmp paths through a symlink alias (`/var` vs `/private/var`);
        # `realpath` maps such an alias back under the real root.
        real_root = Path(os.path.realpath(fs_root))
        alias = fs_root.parent / 'alias-root'
        alias.symlink_to(fs_root)
        error = FileNotFoundError(errno.ENOENT, 'No such file or directory', str(alias / 'hello.txt'))
        message = _sanitize_recoverable_error(error, real_root)
        assert str(alias) not in message
        assert "'hello.txt'" in message
