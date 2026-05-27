"""Exhaustive tests for the FileSystem capability and FileSystemToolset.

Covers:
- Path traversal prevention (relative .., absolute, symlink escapes)
- Allow/deny/protected pattern enforcement
- All tool operations (read, write, edit, list, search, find, mkdir, info)
- Binary file detection
- Optimistic concurrency (hash-based conflict detection)
- Edge cases (empty files, encoding, large files, hidden files)
- Agent-level integration via TestModel
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.filesystem._toolset import FileSystemToolset, _content_hash, _format_lines, _is_binary

# ============================================================================
# Unit tests for helper functions
# ============================================================================


class TestFormatLines:
    def test_basic_formatting(self) -> None:
        text = 'line1\nline2\nline3\n'
        result = _format_lines(text, 0, 10)
        assert '     1\tline1\n' in result
        assert '     2\tline2\n' in result
        assert '     3\tline3\n' in result

    def test_offset(self) -> None:
        text = 'a\nb\nc\nd\ne\n'
        result = _format_lines(text, 2, 2)
        assert '     3\tc\n' in result
        assert '     4\td\n' in result
        assert '... (1 more lines. Use offset=4 to continue reading.)' in result

    def test_offset_exceeds_length(self) -> None:
        text = 'a\nb\n'
        with pytest.raises(ValueError, match='Offset 5 exceeds file length'):
            _format_lines(text, 5, 10)

    def test_empty_file(self) -> None:
        result = _format_lines('', 0, 10)
        assert result == '(empty file)\n'

    def test_no_trailing_newline(self) -> None:
        text = 'no newline'
        result = _format_lines(text, 0, 10)
        assert result.endswith('\n')

    def test_continuation_hint(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(10))
        result = _format_lines(text, 0, 3)
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


# ============================================================================
# FileSystemToolset tests
# ============================================================================


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    """Create a temporary directory with test files."""
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
def toolset(fs_root: Path) -> FileSystemToolset:
    """Create a FileSystemToolset for the test root."""
    return FileSystemToolset(
        root_dir=fs_root,
        allowed_patterns=[],
        denied_patterns=[],
        protected_patterns=['.git/*', '.env', '.env.*'],
        max_read_lines=2000,
        max_search_results=1000,
        max_find_results=1000,
    )


class TestPathSecurity:
    async def test_traversal_with_dotdot(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('../../../etc/passwd')

    async def test_traversal_absolute_path(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('/etc/passwd')

    async def test_traversal_encoded(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='resolves outside'):
            toolset._resolve_path('subdir/../../..')

    async def test_symlink_escape(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Symlink pointing outside root is rejected."""
        target = Path('/tmp/symlink-escape-target')
        target.write_text('escaped!\n')
        try:
            link = fs_root / 'escape_link'
            link.symlink_to(target)
            with pytest.raises(PermissionError, match='resolves outside'):
                toolset._resolve_path('escape_link')
        finally:
            target.unlink(missing_ok=True)

    async def test_valid_path_resolves(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        result = toolset._resolve_path('hello.txt')
        assert result == (fs_root / 'hello.txt').resolve()

    async def test_nested_path_resolves(self, toolset: FileSystemToolset) -> None:
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
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(PermissionError, match='does not match any allowed'):
            ts._check_access('data.txt')

    async def test_protected_pattern_blocks_write(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.git/config', write=True)

    async def test_protected_pattern_allows_read(self, toolset: FileSystemToolset) -> None:
        # Should not raise for read
        toolset._check_access('.git/config', write=False)

    async def test_env_file_protected(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.env', write=True)

    async def test_write_non_protected_with_patterns_configured(self, toolset: FileSystemToolset) -> None:
        # write=True on a path that doesn't match any protected pattern should pass
        toolset._check_access('hello.txt', write=True)

    async def test_access_with_no_denied_patterns(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # No denied, no protected, no allowed → should pass for any path
        ts._check_access('anything.txt', write=True)


class TestReadFile:
    async def test_read_basic(self, toolset: FileSystemToolset) -> None:
        result = await toolset.read_file('hello.txt')
        assert 'Hello, world!' in result
        assert 'hash:' in result
        assert '1 lines' in result

    async def test_read_with_offset(self, toolset: FileSystemToolset) -> None:
        result = await toolset.read_file('multi.txt', offset=2)
        assert 'line3' in result
        assert 'line1' not in result

    async def test_read_with_limit(self, toolset: FileSystemToolset) -> None:
        result = await toolset.read_file('multi.txt', limit=2)
        assert 'line1' in result
        assert 'line2' in result
        assert '... (3 more lines' in result

    async def test_read_directory_raises(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(FileNotFoundError, match='is a directory'):
            await toolset.read_file('subdir')

    async def test_read_missing_raises(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(FileNotFoundError, match='File not found'):
            await toolset.read_file('nonexistent.txt')

    async def test_read_binary_file(self, toolset: FileSystemToolset) -> None:
        result = await toolset.read_file('binary.bin')
        assert 'Binary file' in result
        assert '4 bytes' in result

    async def test_read_traversal_blocked(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError):
            await toolset.read_file('../../../etc/passwd')


class TestWriteFile:
    async def test_write_new_file(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        result = await toolset.write_file('new.txt', 'new content\n')
        assert 'Wrote' in result
        assert (fs_root / 'new.txt').read_text() == 'new content\n'

    async def test_write_creates_parents(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        result = await toolset.write_file('deep/nested/file.txt', 'deep\n')
        assert 'Wrote' in result
        assert (fs_root / 'deep' / 'nested' / 'file.txt').read_text() == 'deep\n'

    async def test_write_overwrite(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        await toolset.write_file('hello.txt', 'overwritten\n')
        assert (fs_root / 'hello.txt').read_text() == 'overwritten\n'

    async def test_write_conflict_detection(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        # Get current hash
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)

        # Write with correct hash succeeds
        await toolset.write_file('hello.txt', 'updated\n', expected_hash=current_hash)
        assert (fs_root / 'hello.txt').read_text() == 'updated\n'

    async def test_write_conflict_rejection(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        with pytest.raises(ValueError, match='Conflict'):
            await toolset.write_file('hello.txt', 'bad\n', expected_hash='wrong_hash_x')

    async def test_write_protected_blocked(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='protected'):
            await toolset.write_file('.env', 'HACKED=true\n')

    async def test_write_returns_hash(self, toolset: FileSystemToolset) -> None:
        result = await toolset.write_file('hashed.txt', 'content\n')
        assert 'hash:' in result


class TestEditFile:
    async def test_edit_basic(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hello, universe!')
        assert 'Edited' in result
        assert (fs_root / 'hello.txt').read_text() == 'Hello, universe!\n'

    async def test_edit_not_found_text(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(ValueError, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'NONEXISTENT', 'replacement')

    async def test_edit_ambiguous_match(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        (fs_root / 'repeat.txt').write_text('foo bar foo\n')
        with pytest.raises(ValueError, match='found 2 times'):
            await toolset.edit_file('repeat.txt', 'foo', 'baz')

    async def test_edit_missing_file(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(FileNotFoundError, match='File not found'):
            await toolset.edit_file('ghost.txt', 'x', 'y')

    async def test_edit_conflict_detection(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)
        result = await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash=current_hash)
        assert 'hash:' in result

    async def test_edit_conflict_rejection(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(ValueError, match='Conflict'):
            await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash='stale_hash_')

    async def test_edit_protected_blocked(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='protected'):
            await toolset.edit_file('.env', 'SECRET', 'HACKED')

    async def test_edit_returns_new_hash(self, toolset: FileSystemToolset) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Goodbye!')
        assert 'hash:' in result


class TestListDirectory:
    async def test_list_root(self, toolset: FileSystemToolset) -> None:
        result = await toolset.list_directory('.')
        assert 'hello.txt' in result
        assert 'subdir/' in result

    async def test_list_subdir(self, toolset: FileSystemToolset) -> None:
        result = await toolset.list_directory('subdir')
        assert 'nested.py' in result

    async def test_list_not_a_dir(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(NotADirectoryError):
            await toolset.list_directory('hello.txt')

    async def test_list_shows_sizes(self, toolset: FileSystemToolset) -> None:
        result = await toolset.list_directory('.')
        assert 'bytes' in result

    async def test_list_shows_dir_indicator(self, toolset: FileSystemToolset) -> None:
        result = await toolset.list_directory('.')
        assert 'subdir/' in result

    async def test_list_empty_directory(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        (fs_root / 'empty').mkdir()
        result = await toolset.list_directory('empty')
        assert result == '(empty directory)'


class TestSearchFiles:
    async def test_search_basic(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('Hello')
        assert 'hello.txt:1:Hello, world!' in result

    async def test_search_regex(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files(r'line\d')
        assert 'multi.txt' in result

    async def test_search_no_matches(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('ZZZZNOTHERE')
        assert result == 'No matches found.'

    async def test_search_skips_hidden(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('secret')
        assert '.hidden' not in result

    async def test_search_skips_binary(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('.')
        assert 'binary.bin' not in result

    async def test_search_invalid_regex(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(ValueError, match='Invalid regex'):
            await toolset.search_files('[invalid')

    async def test_search_include_glob(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('print', include_glob='*.py')
        assert 'nested.py' in result

    async def test_search_include_glob_excludes(self, toolset: FileSystemToolset) -> None:
        result = await toolset.search_files('Hello', include_glob='*.py')
        assert result == 'No matches found.'

    async def test_search_in_specific_file(self, toolset: FileSystemToolset) -> None:
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
            max_search_results=50,
            max_find_results=1000,
        )
        result = await ts.search_files('findme')
        assert 'truncated at 50 matches' in result


class TestFindFiles:
    async def test_find_glob(self, toolset: FileSystemToolset) -> None:
        result = await toolset.find_files('*.txt')
        assert 'hello.txt' in result
        assert 'multi.txt' in result

    async def test_find_recursive(self, toolset: FileSystemToolset) -> None:
        result = await toolset.find_files('**/*.py')
        assert 'nested.py' in result

    async def test_find_no_matches(self, toolset: FileSystemToolset) -> None:
        result = await toolset.find_files('*.xyz')
        assert result == 'No matches found.'

    async def test_find_skips_hidden(self, toolset: FileSystemToolset) -> None:
        result = await toolset.find_files('*')
        assert '.hidden' not in result
        assert '.git' not in result

    async def test_find_not_a_dir(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(NotADirectoryError):
            await toolset.find_files('*.txt', path='hello.txt')

    async def test_find_in_subdir(self, toolset: FileSystemToolset) -> None:
        result = await toolset.find_files('*.py', path='subdir')
        assert 'nested.py' in result

    async def test_find_directories(self, toolset: FileSystemToolset) -> None:
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
            max_search_results=1000,
            max_find_results=5,
        )
        result = await ts.find_files('*.dat')
        assert 'truncated at 5 matches' in result


class TestCreateDirectory:
    async def test_create_basic(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        result = await toolset.create_directory('newdir')
        assert 'Created directory' in result
        assert (fs_root / 'newdir').is_dir()

    async def test_create_nested(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        await toolset.create_directory('a/b/c')
        assert (fs_root / 'a' / 'b' / 'c').is_dir()

    async def test_create_existing_ok(self, toolset: FileSystemToolset) -> None:
        result = await toolset.create_directory('subdir')
        assert 'Created directory' in result

    async def test_create_protected_blocked(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(PermissionError, match='protected'):
            await toolset.create_directory('.git/hooks')


class TestFileInfo:
    async def test_info_file(self, toolset: FileSystemToolset) -> None:
        result = await toolset.file_info('hello.txt')
        assert 'type: file' in result
        assert 'size:' in result
        assert 'lines:' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_info_directory(self, toolset: FileSystemToolset) -> None:
        result = await toolset.file_info('subdir')
        assert 'type: directory' in result

    async def test_info_binary(self, toolset: FileSystemToolset) -> None:
        result = await toolset.file_info('binary.bin')
        assert 'binary: True' in result
        assert 'lines:' not in result

    async def test_info_not_found(self, toolset: FileSystemToolset) -> None:
        with pytest.raises(FileNotFoundError, match='Path not found'):
            await toolset.file_info('nonexistent')

    async def test_info_symlink(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        link = fs_root / 'link.txt'
        link.symlink_to(fs_root / 'hello.txt')
        result = await toolset.file_info('link.txt')
        assert 'type: file' in result
        assert 'symlink_target:' in result


# ============================================================================
# Capability integration tests
# ============================================================================


# ============================================================================
# Mutation-killing tests (boundary conditions, operator swaps, negation)
# ============================================================================


class TestMutationKillers:
    """Tests targeting specific mutations that might survive."""

    async def test_format_lines_offset_equals_total(self) -> None:
        """Kill: offset >= total → offset > total."""
        text = 'a\nb\n'  # 2 lines
        with pytest.raises(ValueError, match='Offset 2 exceeds file length'):
            _format_lines(text, 2, 10)

    async def test_format_lines_exact_fit_no_continuation(self) -> None:
        """Kill: remaining > 0 → remaining >= 0."""
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text, 0, 3)
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_exact_fit_from_offset(self) -> None:
        """Kill: remaining > 0 → remaining >= 0 with offset."""
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text, 1, 2)  # lines 2-3, 0 remaining
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_one_line_remaining(self) -> None:
        """Kill: remaining > 0 → remaining > 1."""
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text, 0, 2)
        assert '... (1 more lines. Use offset=2 to continue reading.)' in result

    async def test_format_lines_line_number_starts_at_one(self) -> None:
        """Kill: start=offset + 1 → start=offset."""
        text = 'first\nsecond\n'
        result = _format_lines(text, 0, 10)
        assert '     1\tfirst\n' in result
        assert '     0\t' not in result

    async def test_format_lines_offset_line_numbering(self) -> None:
        """Kill: start=offset + 1 → start=offset + 2."""
        text = 'a\nb\nc\n'
        result = _format_lines(text, 1, 2)
        assert '     2\tb\n' in result
        assert '     3\tc\n' in result

    async def test_is_binary_exactly_at_sample_boundary(self) -> None:
        """Kill: sample_size mutations at the exact boundary."""
        # Null byte at position 8191 (index 8191, within first 8192 bytes)
        data = b'x' * 8191 + b'\x00'
        assert _is_binary(data) is True
        # Null byte at position 8192 (outside the sample)
        data2 = b'x' * 8192 + b'\x00'
        assert _is_binary(data2) is False

    async def test_content_hash_returns_exactly_12_chars(self) -> None:
        """Kill: [:12] → [:11] or [:13]."""
        h = _content_hash('test content')
        assert len(h) == 12
        # Verify it's hex characters
        assert all(c in '0123456789abcdef' for c in h)

    async def test_write_file_with_hash_on_new_file(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: expected_hash is not None and resolved.is_file() → expected_hash is not None.

        When a file doesn't exist, expected_hash should be ignored and the write should succeed.
        """
        result = await toolset.write_file('brand_new.txt', 'new content\n', expected_hash='any_hash_val')
        assert 'Wrote' in result
        assert (fs_root / 'brand_new.txt').read_text() == 'new content\n'

    async def test_edit_file_single_match_succeeds(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: count > 1 → count >= 1 (single match must not raise)."""
        (fs_root / 'unique.txt').write_text('unique text here\n')
        result = await toolset.edit_file('unique.txt', 'unique text', 'replaced text')
        assert 'Edited' in result
        assert (fs_root / 'unique.txt').read_text() == 'replaced text here\n'

    async def test_edit_file_zero_matches_raises(self, toolset: FileSystemToolset) -> None:
        """Kill: count == 0 → count != 0 or count == 1."""
        with pytest.raises(ValueError, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'DEFINITELY NOT IN FILE', 'x')

    async def test_search_truncation_stops_after_limit(self, fs_root: Path) -> None:
        """Kill: removing the 'break' after truncation message."""
        # Create many files with 1 match each so truncation is per-file
        for i in range(10):
            (fs_root / f'searchable{i}.txt').write_text(f'match_this_{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
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
        """Kill: removing the 'break' after truncation in find_files."""
        for i in range(10):
            (fs_root / f'findme{i:02d}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            protected_patterns=[],
            max_read_lines=2000,
            max_search_results=1000,
            max_find_results=3,
        )
        result = await ts.find_files('*.dat')
        lines = result.strip().split('\n')
        # Should have exactly 4 lines: 3 matches + 1 truncation message
        assert len(lines) == 4
        assert 'truncated at 3 matches' in lines[-1]

    async def test_read_file_default_limit_used(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: if limit is None: limit = self._max_read_lines → removing this."""
        # Create file with more lines than we'd see with limit=0
        (fs_root / 'big.txt').write_text('\n'.join(f'line{i}' for i in range(100)) + '\n')
        result = await toolset.read_file('big.txt')
        # All 100 lines should be present since max_read_lines is 2000
        assert 'line99' in result

    async def test_list_directory_with_files_not_empty(self, toolset: FileSystemToolset) -> None:
        """Kill: 'entries' being falsy check — ensure non-empty dirs return actual content."""
        result = await toolset.list_directory('subdir')
        assert result != '(empty directory)'
        assert 'nested.py' in result

    async def test_search_in_file_returns_only_that_file(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: if resolved.is_file(): files = [resolved] → files = sorted(resolved.rglob('*'))."""
        # Both files contain 'Hello' / 'hello' but searching a specific file should only return from that file
        (fs_root / 'other.txt').write_text('Hello from other\n')
        result = await toolset.search_files('Hello', path='hello.txt')
        assert 'hello.txt' in result
        assert 'other.txt' not in result

    async def test_file_info_non_binary_shows_lines_and_hash(self, toolset: FileSystemToolset) -> None:
        """Kill: not is_bin → is_bin (negation of binary check in file_info)."""
        result = await toolset.file_info('hello.txt')
        assert 'lines: 1' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_file_info_binary_no_lines_no_hash(self, toolset: FileSystemToolset) -> None:
        """Kill: not is_bin → is_bin (ensure binary files DON'T get lines/hash)."""
        result = await toolset.file_info('binary.bin')
        assert 'binary: True' in result
        assert 'lines:' not in result
        assert 'hash:' not in result

    async def test_safe_resolve_passes_write_flag(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: _safe_resolve not passing write= to _check_access."""
        # Protected patterns block writes but allow reads
        (fs_root / '.env.local').write_text('SECRET=x\n')
        # Read should work (write=False internally)
        result = await toolset.read_file('.env.local')
        assert 'SECRET=x' in result
        # Write should be blocked (write=True internally)
        with pytest.raises(PermissionError, match='protected'):
            await toolset.write_file('.env.local', 'HACKED\n')

    async def test_format_lines_join_separator(self) -> None:
        """Kill: ''.join(numbered) → 'XXXX'.join(numbered).

        Verify the result doesn't contain garbage between lines.
        """
        text = 'a\nb\nc\n'
        result = _format_lines(text, 0, 3)
        # Lines should be directly adjacent (no separator between them)
        assert '     1\ta\n     2\tb\n     3\tc\n' in result

    async def test_format_lines_no_trailing_newline_preserves_content(self) -> None:
        """Kill: result += '\\n' → result = '\\n' (content destroyed)."""
        text = 'no newline'
        result = _format_lines(text, 0, 10)
        # The content must still be present
        assert 'no newline' in result
        assert result.endswith('\n')

    async def test_read_file_hash_is_real_hash(self, toolset: FileSystemToolset) -> None:
        """Kill: content_hash = _content_hash(text) → content_hash = None."""
        result = await toolset.read_file('hello.txt')
        # The actual hash should be a hex string, not 'None'
        assert 'hash:None' not in result
        # Verify the hash matches what we'd compute
        expected_hash = _content_hash('Hello, world!\n')
        assert f'hash:{expected_hash}' in result

    async def test_read_file_non_ascii_content(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: errors='replace' removal and errors='XXreplaceXX'.

        With invalid UTF-8 bytes, the tool should not crash — it should use replacement chars.
        """
        # Write raw bytes that are invalid UTF-8
        (fs_root / 'broken_utf8.txt').write_bytes(b'hello \xff\xfe world\n')
        result = await toolset.read_file('broken_utf8.txt')
        # Should not crash, content should contain replacement characters
        assert 'hello' in result
        assert 'world' in result

    async def test_read_file_default_offset_starts_at_first_line(self, toolset: FileSystemToolset) -> None:
        """Kill: offset: int = 0 → offset: int = 1 (default param change).

        The first line must be included when no offset is specified.
        """
        result = await toolset.read_file('multi.txt')
        # First line must be present (line1)
        assert '     1\tline1' in result
        # Verify line numbering starts at 1
        assert '     0\t' not in result

    async def test_toolset_tool_names(self, toolset: FileSystemToolset) -> None:
        """Kill: name='read_file' → name=None / name='XXread_fileXX'.

        Verify tools are registered with correct names.
        """
        tool_names = set(toolset.tools.keys())
        assert 'read_file' in tool_names
        assert 'write_file' in tool_names
        assert 'edit_file' in tool_names
        assert 'list_directory' in tool_names
        assert 'search_files' in tool_names
        assert 'find_files' in tool_names
        assert 'create_directory' in tool_names
        assert 'file_info' in tool_names

    async def test_write_file_output_format(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: write return string mutations."""
        result = await toolset.write_file('fmt.txt', 'ab\ncd\n')
        # Verify specific format: chars, lines, path, hash
        assert 'Wrote 6 chars (2 lines) to fmt.txt.' in result
        assert 'hash:' in result
        # Verify hash is a real hex hash not None
        assert 'hash:None' not in result

    async def test_edit_file_output_format(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: edit return string mutations."""
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hi')
        assert result.startswith('Edited hello.txt.')
        assert 'hash:' in result
        assert 'hash:None' not in result

    def test_format_lines_no_double_trailing_newline(self) -> None:
        """Kill: result.endswith('\\n') → result.endswith('XX\\nXX').

        Text that already ends with newline must NOT get a second one appended.
        """
        text = 'hello\n'
        result = _format_lines(text, 0, 10)
        # Exact match: no trailing double newline
        assert result == '     1\thello\n'

    def test_safe_resolve_write_default_is_false(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: _safe_resolve write: bool = False → True.

        Synchronous test to avoid trio crash confusing mutmut.
        Protected files should be READABLE via _safe_resolve's default (write=False).
        """
        (fs_root / '.env.local').write_text('SECRET=x\n')
        # _safe_resolve without write= uses default write=False → read is allowed
        resolved = toolset._safe_resolve('.env.local')
        assert resolved.name == '.env.local'
        # But with write=True, it should raise
        with pytest.raises(PermissionError, match='protected'):
            toolset._safe_resolve('.env.local', write=True)

    async def test_list_directory_exact_size(self, toolset: FileSystemToolset) -> None:
        """Kill: size = stat.st_size → size = None."""
        result = await toolset.list_directory('.')
        # hello.txt has 'Hello, world!\n' = 14 bytes
        assert '14 bytes' in result

    async def test_list_directory_no_garbage_separator(self, toolset: FileSystemToolset) -> None:
        """Kill: '\\n'.join(entries) → 'XX\\nXX'.join(entries)."""
        result = await toolset.list_directory('.')
        assert 'XX' not in result

    async def test_list_directory_error_message(self, toolset: FileSystemToolset) -> None:
        """Kill: NotADirectoryError(f'...') → NotADirectoryError(None)."""
        with pytest.raises(NotADirectoryError, match='Not a directory'):
            await toolset.list_directory('hello.txt')

    async def test_find_files_error_message(self, toolset: FileSystemToolset) -> None:
        """Kill: NotADirectoryError(f'...') → NotADirectoryError(None)."""
        with pytest.raises(NotADirectoryError, match='Not a directory'):
            await toolset.find_files('*.txt', path='hello.txt')

    async def test_find_files_no_suffix_on_files(self, toolset: FileSystemToolset) -> None:
        """Kill: suffix '' → 'XXXX' for non-directory entries."""
        result = await toolset.find_files('*.txt')
        for line in result.splitlines():
            if not line.endswith('/'):
                assert 'XXXX' not in line

    async def test_find_files_no_garbage_separator(self, toolset: FileSystemToolset) -> None:
        """Kill: '\\n'.join(matches) → 'XX\\nXX'.join(matches)."""
        result = await toolset.find_files('*.txt')
        assert 'XX' not in result

    async def test_search_files_no_garbage_separator(self, toolset: FileSystemToolset) -> None:
        """Kill: '\\n'.join(results) → 'XX\\nXX'.join(results)."""
        result = await toolset.search_files(r'line\d')
        assert 'XX' not in result

    async def test_file_info_exact_size(self, toolset: FileSystemToolset) -> None:
        """Kill: size = stat.st_size → size = None."""
        result = await toolset.file_info('hello.txt')
        assert '14 bytes' in result

    async def test_file_info_no_garbage_separator(self, toolset: FileSystemToolset) -> None:
        """Kill: '\\n'.join(parts) → 'XX\\nXX'.join(parts)."""
        result = await toolset.file_info('hello.txt')
        assert 'XX' not in result

    async def test_search_with_invalid_utf8_file(self, toolset: FileSystemToolset, fs_root: Path) -> None:
        """Kill: errors='replace' removal and errors='XXreplaceXX'.

        A file with invalid UTF-8 (but no null bytes = not binary) should be searchable.
        """
        # Write a file with invalid UTF-8 but no null bytes (not detected as binary)
        (fs_root / 'bad_encoding.txt').write_bytes(b'marker_text \xff\xfe end\n')
        result = await toolset.search_files('marker_text')
        # Should find the file even with broken encoding
        assert 'bad_encoding.txt' in result

    async def test_search_binary_skip_does_not_stop_iteration(self, toolset: FileSystemToolset) -> None:
        """Kill: if _is_binary(raw): continue → break.

        A binary file must be skipped, but subsequent text files must still be searched.
        """
        # binary.bin exists in the fixture and comes before 'hello.txt' alphabetically
        result = await toolset.search_files('Hello')
        # hello.txt must still be found (binary.bin didn't break the loop)
        assert 'hello.txt' in result

    async def test_find_hidden_skip_does_not_stop_iteration(self, toolset: FileSystemToolset) -> None:
        """Kill: if any(part.startswith('.')): continue → break.

        Hidden files must be skipped, but subsequent visible files must still appear.
        """
        # .hidden comes before hello.txt alphabetically — skipping must not break the loop
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

    def test_protected_defaults(self) -> None:
        fs = FileSystem()
        assert '.git/*' in fs.protected_patterns
        assert '.env' in fs.protected_patterns

    @pytest.mark.anyio(backends=['asyncio'])
    async def test_agent_integration(self, tmp_path: Path, anyio_backend: object) -> None:
        if str(anyio_backend) != 'asyncio':
            pytest.skip('Agent.run requires asyncio event loop')
        (tmp_path / 'test.txt').write_text('hello agent\n')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[FileSystem(root_dir=tmp_path)])
        result = await agent.run('read test.txt')
        assert result.output == 'done'
