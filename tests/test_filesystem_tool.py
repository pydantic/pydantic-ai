"""Tests for FileSystemToolset and FileSystemTool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import FileSystemTool, FileSystemToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context(deps: Any = None, run_step: int = 0) -> RunContext[Any]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


class TestFileSystemToolsetPathValidation:
    """Tests for path validation in FileSystemToolset."""

    async def test_access_within_allowed_paths_succeeds(self, tmp_path: Path):
        """Test that accessing files within allowed paths succeeds."""
        file = tmp_path / 'test.txt'
        file.write_text('Hello, World!')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)
        assert 'file_system' in tools

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Hello, World!'

    async def test_access_outside_allowed_paths_raises_permission_error(self, tmp_path: Path):
        """Test that accessing files outside allowed paths raises PermissionError."""
        # Create a file outside allowed paths
        other_dir = tmp_path / 'other'
        other_dir.mkdir()
        file = other_dir / 'test.txt'
        file.write_text('Secret content')

        # Only allow access to a different directory
        allowed_dir = tmp_path / 'allowed'
        allowed_dir.mkdir()

        toolset = FileSystemToolset(allowed_paths=[str(allowed_dir)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='outside allowed directories'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(file)},
                ctx,
                tools['file_system'],
            )

    async def test_path_traversal_blocked(self, tmp_path: Path):
        """Test that path traversal attempts are blocked."""
        allowed_dir = tmp_path / 'allowed'
        allowed_dir.mkdir()

        # Try to escape using ..
        toolset = FileSystemToolset(allowed_paths=[str(allowed_dir)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='outside allowed directories'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(allowed_dir / '..' / 'secret.txt')},
                ctx,
                tools['file_system'],
            )

    async def test_default_cwd_restriction(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that default restriction to current directory works."""
        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        file = tmp_path / 'test.txt'
        file.write_text('Hello')

        # No allowed_paths means only cwd is allowed
        toolset = FileSystemToolset()
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        # Access within cwd should work
        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Hello'

    async def test_access_outside_cwd_when_no_allowed_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test that accessing outside cwd raises error when no allowed_paths configured."""
        cwd = tmp_path / 'cwd'
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        outside_file = tmp_path / 'outside.txt'
        outside_file.write_text('Outside content')

        toolset = FileSystemToolset()
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='outside current directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(outside_file)},
                ctx,
                tools['file_system'],
            )


class TestFileSystemToolsetExtensionFiltering:
    """Tests for extension filtering in FileSystemToolset."""

    async def test_allowed_extensions_can_be_accessed(self, tmp_path: Path):
        """Test that files with allowed extensions can be accessed."""
        file = tmp_path / 'test.txt'
        file.write_text('Text content')

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allowed_extensions=['.txt', '.json'],
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Text content'

    async def test_disallowed_extensions_raise_error(self, tmp_path: Path):
        """Test that files with disallowed extensions raise PermissionError."""
        file = tmp_path / 'script.py'
        file.write_text('print("hello")')

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allowed_extensions=['.txt', '.json'],
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='not in allowed extensions'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(file)},
                ctx,
                tools['file_system'],
            )

    async def test_extension_checking_case_insensitive(self, tmp_path: Path):
        """Test that extension checking is case insensitive."""
        file = tmp_path / 'test.TXT'
        file.write_text('Content')

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allowed_extensions=['.txt'],
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Content'


class TestFileSystemToolsetOperationPermissions:
    """Tests for operation permission validation."""

    async def test_read_operations_work_when_allowed(self, tmp_path: Path):
        """Test that read operations work by default."""
        file = tmp_path / 'test.txt'
        file.write_text('Content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Content'

    async def test_write_operations_blocked_by_default(self, tmp_path: Path):
        """Test that write operations are blocked when allow_write=False."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='Write operations are not allowed'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'write', 'path': str(tmp_path / 'new.txt'), 'content': 'test'},
                ctx,
                tools['file_system'],
            )

    async def test_write_operations_allowed_when_enabled(self, tmp_path: Path):
        """Test that write operations work when allow_write=True."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        file = tmp_path / 'new.txt'
        result = await toolset.call_tool(
            'file_system',
            {'operation': 'write', 'path': str(file), 'content': 'New content'},
            ctx,
            tools['file_system'],
        )
        assert 'Successfully wrote' in result
        assert file.read_text() == 'New content'

    async def test_delete_operations_blocked_by_default(self, tmp_path: Path):
        """Test that delete operations are blocked when allow_delete=False."""
        file = tmp_path / 'test.txt'
        file.write_text('Content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='Delete operations are not allowed'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'delete', 'path': str(file)},
                ctx,
                tools['file_system'],
            )

    async def test_delete_requires_allow_write(self, tmp_path: Path):
        """Test that delete requires allow_write=True."""
        file = tmp_path / 'test.txt'
        file.write_text('Content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_delete=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='Delete operations require allow_write=True'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'delete', 'path': str(file)},
                ctx,
                tools['file_system'],
            )

    async def test_delete_operations_allowed_when_enabled(self, tmp_path: Path):
        """Test that delete operations work when both allow_write and allow_delete are True."""
        file = tmp_path / 'test.txt'
        file.write_text('Content')

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'delete', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert 'Successfully deleted' in result
        assert not file.exists()


class TestFileSystemToolsetFileSizeLimit:
    """Tests for file size limit validation."""

    async def test_files_under_limit_can_be_read(self, tmp_path: Path):
        """Test that files under the size limit can be read."""
        file = tmp_path / 'small.txt'
        file.write_text('Small content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], max_file_size=1000)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'Small content'

    async def test_files_over_limit_raise_error(self, tmp_path: Path):
        """Test that files over the size limit raise PermissionError."""
        file = tmp_path / 'large.txt'
        file.write_text('x' * 1000)  # 1000 bytes

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], max_file_size=100)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(PermissionError, match='exceeds maximum allowed size'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(file)},
                ctx,
                tools['file_system'],
            )


class TestFileSystemToolsetOperations:
    """Tests for individual file system operations."""

    async def test_read_file(self, tmp_path: Path):
        """Test read operation."""
        file = tmp_path / 'test.txt'
        file.write_text('File content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == 'File content'

    async def test_read_nonexistent_file(self, tmp_path: Path):
        """Test reading a nonexistent file."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(FileNotFoundError, match='File not found'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(tmp_path / 'nonexistent.txt')},
                ctx,
                tools['file_system'],
            )

    async def test_read_directory_as_file(self, tmp_path: Path):
        """Test reading a directory as a file raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(IsADirectoryError, match='is a directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read', 'path': str(tmp_path)},
                ctx,
                tools['file_system'],
            )

    async def test_write_new_file(self, tmp_path: Path):
        """Test writing a new file."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        file = tmp_path / 'new.txt'
        result = await toolset.call_tool(
            'file_system',
            {'operation': 'write', 'path': str(file), 'content': 'New content'},
            ctx,
            tools['file_system'],
        )
        assert 'Successfully wrote' in result
        assert file.read_text() == 'New content'

    async def test_write_creates_parent_directories(self, tmp_path: Path):
        """Test that write creates parent directories if needed."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        file = tmp_path / 'subdir' / 'nested' / 'file.txt'
        await toolset.call_tool(
            'file_system',
            {'operation': 'write', 'path': str(file), 'content': 'Nested content'},
            ctx,
            tools['file_system'],
        )
        assert file.exists()
        assert file.read_text() == 'Nested content'

    async def test_write_overwrites_existing(self, tmp_path: Path):
        """Test that write overwrites existing files."""
        file = tmp_path / 'existing.txt'
        file.write_text('Old content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        await toolset.call_tool(
            'file_system',
            {'operation': 'write', 'path': str(file), 'content': 'New content'},
            ctx,
            tools['file_system'],
        )
        assert file.read_text() == 'New content'

    async def test_write_to_directory_raises_error(self, tmp_path: Path):
        """Test writing to a directory path raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(IsADirectoryError, match='is a directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'write', 'path': str(tmp_path), 'content': 'test'},
                ctx,
                tools['file_system'],
            )

    async def test_list_directory(self, tmp_path: Path):
        """Test listing directory contents."""
        (tmp_path / 'file1.txt').write_text('content1')
        (tmp_path / 'file2.txt').write_text('content2')
        (tmp_path / 'subdir').mkdir()

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'list', 'path': str(tmp_path)},
            ctx,
            tools['file_system'],
        )

        names = {entry['name'] for entry in result}
        assert names == {'file1.txt', 'file2.txt', 'subdir'}

        for entry in result:
            if entry['name'] == 'subdir':
                assert entry['type'] == 'directory'
            else:
                assert entry['type'] == 'file'

    async def test_list_nonexistent_directory(self, tmp_path: Path):
        """Test listing a nonexistent directory."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(FileNotFoundError, match='Directory not found'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'list', 'path': str(tmp_path / 'nonexistent')},
                ctx,
                tools['file_system'],
            )

    async def test_list_file_as_directory(self, tmp_path: Path):
        """Test listing a file as a directory raises error."""
        file = tmp_path / 'file.txt'
        file.write_text('content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(NotADirectoryError, match='not a directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'list', 'path': str(file)},
                ctx,
                tools['file_system'],
            )

    async def test_file_info(self, tmp_path: Path):
        """Test getting file info."""
        file = tmp_path / 'test.txt'
        file.write_text('content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'info', 'path': str(file)},
            ctx,
            tools['file_system'],
        )

        assert result['name'] == 'test.txt'
        assert result['type'] == 'file'
        assert result['size'] == 7  # len('content')
        assert 'modified' in result
        assert 'created' in result
        assert result['is_symlink'] is False

    async def test_directory_info(self, tmp_path: Path):
        """Test getting directory info."""
        subdir = tmp_path / 'subdir'
        subdir.mkdir()

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'info', 'path': str(subdir)},
            ctx,
            tools['file_system'],
        )

        assert result['name'] == 'subdir'
        assert result['type'] == 'directory'
        assert result['size'] is None

    async def test_info_nonexistent_path(self, tmp_path: Path):
        """Test info on nonexistent path."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(FileNotFoundError, match='Path not found'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'info', 'path': str(tmp_path / 'nonexistent')},
                ctx,
                tools['file_system'],
            )

    async def test_search_files(self, tmp_path: Path):
        """Test searching for files."""
        (tmp_path / 'file1.txt').write_text('content1')
        (tmp_path / 'file2.txt').write_text('content2')
        (tmp_path / 'file3.py').write_text('code')
        subdir = tmp_path / 'subdir'
        subdir.mkdir()
        (subdir / 'nested.txt').write_text('nested')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'search', 'path': str(tmp_path), 'pattern': '*.txt'},
            ctx,
            tools['file_system'],
        )

        assert len(result) == 3  # file1.txt, file2.txt, nested.txt

    async def test_search_nonexistent_directory(self, tmp_path: Path):
        """Test searching in nonexistent directory."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(FileNotFoundError, match='Directory not found'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'search', 'path': str(tmp_path / 'nonexistent'), 'pattern': '*.txt'},
                ctx,
                tools['file_system'],
            )

    async def test_search_file_as_directory(self, tmp_path: Path):
        """Test searching with file path."""
        file = tmp_path / 'file.txt'
        file.write_text('content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(NotADirectoryError, match='not a directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'search', 'path': str(file), 'pattern': '*.txt'},
                ctx,
                tools['file_system'],
            )

    async def test_delete_file(self, tmp_path: Path):
        """Test deleting a file."""
        file = tmp_path / 'test.txt'
        file.write_text('content')

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'delete', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert 'Successfully deleted' in result
        assert not file.exists()

    async def test_delete_nonexistent_file(self, tmp_path: Path):
        """Test deleting a nonexistent file."""
        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(FileNotFoundError, match='File not found'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'delete', 'path': str(tmp_path / 'nonexistent.txt')},
                ctx,
                tools['file_system'],
            )

    async def test_delete_directory_raises_error(self, tmp_path: Path):
        """Test deleting a directory raises error."""
        subdir = tmp_path / 'subdir'
        subdir.mkdir()

        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(IsADirectoryError, match='is a directory'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'delete', 'path': str(subdir)},
                ctx,
                tools['file_system'],
            )


class TestFileSystemToolsetToolDefinition:
    """Tests for the tool definition generation."""

    async def test_tool_definition_read_only(self, tmp_path: Path):
        """Test tool definition for read-only access."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['file_system'].tool_def
        assert tool_def.name == 'file_system'
        assert tool_def.description is not None
        assert 'read' in tool_def.description
        assert 'write' not in tool_def.description or 'allow_write' in tool_def.description

        schema = tool_def.parameters_json_schema
        assert schema is not None
        assert 'operation' in schema['properties']
        assert 'write' not in schema['properties']['operation']['enum']
        assert 'delete' not in schema['properties']['operation']['enum']

    async def test_tool_definition_with_write(self, tmp_path: Path):
        """Test tool definition with write enabled."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['file_system'].tool_def
        schema = tool_def.parameters_json_schema
        assert 'write' in schema['properties']['operation']['enum']
        assert 'delete' not in schema['properties']['operation']['enum']

    async def test_tool_definition_with_delete(self, tmp_path: Path):
        """Test tool definition with delete enabled."""
        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['file_system'].tool_def
        schema = tool_def.parameters_json_schema
        assert 'write' in schema['properties']['operation']['enum']
        assert 'delete' in schema['properties']['operation']['enum']

    async def test_tool_definition_includes_allowed_paths(self, tmp_path: Path):
        """Test that tool definition includes allowed paths in description."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['file_system'].tool_def
        assert tool_def.description is not None
        assert str(tmp_path) in tool_def.description

    async def test_tool_definition_includes_allowed_extensions(self, tmp_path: Path):
        """Test that tool definition includes allowed extensions in description."""
        toolset = FileSystemToolset(
            allowed_paths=[str(tmp_path)],
            allowed_extensions=['.txt', '.json'],
        )
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        tool_def = tools['file_system'].tool_def
        assert tool_def.description is not None
        assert '.txt' in tool_def.description
        assert '.json' in tool_def.description


class TestFileSystemToolsetEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_unknown_operation(self, tmp_path: Path):
        """Test unknown operation raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(ValueError, match='Unknown operation'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'unknown', 'path': str(tmp_path)},
                ctx,
                tools['file_system'],
            )

    async def test_missing_operation(self, tmp_path: Path):
        """Test missing operation raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(ValueError, match='Missing required argument: operation'):
            await toolset.call_tool(
                'file_system',
                {'path': str(tmp_path)},
                ctx,
                tools['file_system'],
            )

    async def test_missing_path(self, tmp_path: Path):
        """Test missing path raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(ValueError, match='Missing required argument: path'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'read'},
                ctx,
                tools['file_system'],
            )

    async def test_write_missing_content(self, tmp_path: Path):
        """Test write without content raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)], allow_write=True)
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(ValueError, match='Missing required argument for write operation: content'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'write', 'path': str(tmp_path / 'test.txt')},
                ctx,
                tools['file_system'],
            )

    async def test_search_missing_pattern(self, tmp_path: Path):
        """Test search without pattern raises error."""
        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        with pytest.raises(ValueError, match='Missing required argument for search operation: pattern'):
            await toolset.call_tool(
                'file_system',
                {'operation': 'search', 'path': str(tmp_path)},
                ctx,
                tools['file_system'],
            )

    async def test_empty_file(self, tmp_path: Path):
        """Test reading an empty file."""
        file = tmp_path / 'empty.txt'
        file.write_text('')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()
        tools = await toolset.get_tools(ctx)

        result = await toolset.call_tool(
            'file_system',
            {'operation': 'read', 'path': str(file)},
            ctx,
            tools['file_system'],
        )
        assert result == ''

    async def test_toolset_id(self):
        """Test toolset ID property."""
        toolset_no_id = FileSystemToolset()
        assert toolset_no_id.id is None

        toolset_with_id = FileSystemToolset(_id='my-fs-toolset')
        assert toolset_with_id.id == 'my-fs-toolset'


class TestFileSystemToolBuiltinTool:
    """Tests for the FileSystemTool builtin tool configuration."""

    def test_file_system_tool_kind(self):
        """Test that FileSystemTool has correct kind."""
        tool = FileSystemTool()
        assert tool.kind == 'file_system'

    def test_file_system_tool_default_values(self):
        """Test default values for FileSystemTool."""
        tool = FileSystemTool()
        assert tool.allowed_paths is None
        assert tool.allow_write is False
        assert tool.allow_delete is False
        assert tool.max_file_size == 10 * 1024 * 1024
        assert tool.allowed_extensions is None

    def test_file_system_tool_custom_values(self, tmp_path: Path):
        """Test custom values for FileSystemTool."""
        tool = FileSystemTool(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
            max_file_size=1000,
            allowed_extensions=['.txt'],
        )
        assert tool.allowed_paths == [str(tmp_path)]
        assert tool.allow_write is True
        assert tool.allow_delete is True
        assert tool.max_file_size == 1000
        assert tool.allowed_extensions == ['.txt']

    def test_to_toolset(self, tmp_path: Path):
        """Test converting FileSystemTool to FileSystemToolset."""
        tool = FileSystemTool(
            allowed_paths=[str(tmp_path)],
            allow_write=True,
            allow_delete=True,
            max_file_size=1000,
            allowed_extensions=['.txt'],
        )
        toolset = tool.to_toolset()

        assert isinstance(toolset, FileSystemToolset)
        assert toolset.allowed_paths == [str(tmp_path)]
        assert toolset.allow_write is True
        assert toolset.allow_delete is True
        assert toolset.max_file_size == 1000
        assert toolset.allowed_extensions == ['.txt']

    def test_to_toolset_empty_allowed_paths(self):
        """Test converting FileSystemTool with no allowed paths."""
        tool = FileSystemTool()
        toolset = tool.to_toolset()

        assert isinstance(toolset, FileSystemToolset)
        assert toolset.allowed_paths == []


class TestFileSystemToolsetWithToolManager:
    """Tests for FileSystemToolset integration with ToolManager."""

    async def test_tool_manager_integration(self, tmp_path: Path):
        """Test that FileSystemToolset works with ToolManager."""
        file = tmp_path / 'test.txt'
        file.write_text('Content')

        toolset = FileSystemToolset(allowed_paths=[str(tmp_path)])
        ctx = build_run_context()

        tool_manager = await ToolManager(toolset).for_run_step(ctx)

        tool_defs = tool_manager.tool_defs
        assert len(tool_defs) == 1
        assert tool_defs[0].name == 'file_system'

        result = await tool_manager.handle_call(
            ToolCallPart(
                tool_name='file_system',
                args={'operation': 'read', 'path': str(file)},
            )
        )
        assert result == 'Content'
