"""FileSystemToolset for providing file system access to agents."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pydantic import TypeAdapter

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .abstract import AbstractToolset, SchemaValidatorProt, ToolsetTool

__all__ = ('FileSystemToolset',)


@dataclass(kw_only=True)
class FileSystemToolset(AbstractToolset[AgentDepsT]):
    """A toolset that provides file system operations to agents.

    This toolset allows agents to read, write, list, search, and delete files
    within configured allowed directories.

    Example usage:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.toolsets import FileSystemToolset

        agent = Agent(
            'openai:gpt-4',
            toolsets=[
                FileSystemToolset(
                    allowed_paths=['/home/user/data'],
                    allow_write=False,
                ),
            ],
        )
        ```
    """

    allowed_paths: list[str | Path] = field(default_factory=list)
    """List of paths the agent is allowed to access.
    If empty, allows access to current working directory only."""

    allow_write: bool = False
    """Whether to allow write operations (create, modify)."""

    allow_delete: bool = False
    """Whether to allow delete operations. Only applies if allow_write is True."""

    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    """Maximum file size to read in bytes."""

    allowed_extensions: list[str] | None = None
    """If provided, only files with these extensions can be accessed.
    Example: ['.txt', '.json', '.py']"""

    _id: str | None = None
    """Optional unique ID for the toolset."""

    @property
    def id(self) -> str | None:
        """Return the toolset ID."""
        return self._id

    def _validate_path(self, path: str | Path) -> Path:
        """Validate that path is within allowed directories.

        Args:
            path: The path to validate.

        Returns:
            The resolved path.

        Raises:
            PermissionError: If the path is outside allowed directories or has a disallowed extension.
        """
        resolved = Path(path).resolve()

        # Check allowed paths
        if self.allowed_paths:
            allowed = False
            for allowed_path in self.allowed_paths:
                allowed_resolved = Path(allowed_path).resolve()
                try:
                    resolved.relative_to(allowed_resolved)
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise PermissionError(f'Path {path} is outside allowed directories: {self.allowed_paths}')
        else:
            # Default: only allow current directory
            cwd = Path.cwd().resolve()
            try:
                resolved.relative_to(cwd)
            except ValueError:
                raise PermissionError(
                    f'Path {path} is outside current directory. Configure allowed_paths to access other directories.'
                )

        # Check extension for files (only if path exists and is a file, or if it's a new file being written)
        if self.allowed_extensions:
            suffix = resolved.suffix.lower()
            allowed_lower = [ext.lower() for ext in self.allowed_extensions]
            if suffix and suffix not in allowed_lower:
                raise PermissionError(
                    f'File extension {resolved.suffix} not in allowed extensions: {self.allowed_extensions}'
                )

        return resolved

    def _validate_operation(self, operation: str) -> None:
        """Validate that the operation is allowed.

        Args:
            operation: The operation to validate (read, write, list, info, search, delete).

        Raises:
            PermissionError: If the operation is not allowed.
        """
        if operation == 'write' and not self.allow_write:
            raise PermissionError('Write operations are not allowed')
        if operation == 'delete':
            if not self.allow_write:
                raise PermissionError('Delete operations require allow_write=True')
            if not self.allow_delete:
                raise PermissionError('Delete operations are not allowed')

    async def _read_file(self, path: str) -> str:
        """Read contents of a file.

        Args:
            path: Path to the file to read.

        Returns:
            The contents of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the path is not allowed or file is too large.
            IsADirectoryError: If the path is a directory.
        """
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f'File not found: {path}')

        if resolved.is_dir():
            raise IsADirectoryError(f'Path is a directory, not a file: {path}')

        # Check file size
        file_size = resolved.stat().st_size
        if file_size > self.max_file_size:
            raise PermissionError(
                f'File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)'
            )

        return resolved.read_text(encoding='utf-8')

    async def _write_file(self, path: str, content: str) -> str:
        """Write content to a file.

        Args:
            path: Path to the file to write.
            content: Content to write to the file.

        Returns:
            A success message.

        Raises:
            PermissionError: If the path is not allowed or write is disabled.
            IsADirectoryError: If the path is a directory.
        """
        self._validate_operation('write')
        resolved = self._validate_path(path)

        if resolved.exists() and resolved.is_dir():
            raise IsADirectoryError(f'Path is a directory, not a file: {path}')

        # Create parent directories if they don't exist
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding='utf-8')
        return f'Successfully wrote {len(content)} characters to {path}'

    async def _list_directory(self, path: str) -> list[dict[str, Any]]:
        """List contents of a directory.

        Args:
            path: Path to the directory to list.

        Returns:
            List of file/directory information dictionaries.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
            PermissionError: If the path is not allowed.
        """
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f'Directory not found: {path}')

        if not resolved.is_dir():
            raise NotADirectoryError(f'Path is not a directory: {path}')

        entries: list[dict[str, Any]] = []
        for entry in resolved.iterdir():
            try:
                stat = entry.stat()
                entries.append(
                    {
                        'name': entry.name,
                        'type': 'directory' if entry.is_dir() else 'file',
                        'size': stat.st_size if entry.is_file() else None,
                        'modified': stat.st_mtime,
                    }
                )
            except (PermissionError, OSError):  # pragma: lax no cover
                # Skip entries we can't access
                entries.append(
                    {
                        'name': entry.name,
                        'type': 'unknown',
                        'size': None,
                        'modified': None,
                    }
                )

        return entries

    async def _file_info(self, path: str) -> dict[str, Any]:
        """Get information about a file or directory.

        Args:
            path: Path to the file or directory.

        Returns:
            Dictionary with file/directory information.

        Raises:
            FileNotFoundError: If the path does not exist.
            PermissionError: If the path is not allowed.
        """
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f'Path not found: {path}')

        stat = resolved.stat()
        return {
            'name': resolved.name,
            'path': str(resolved),
            'type': 'directory' if resolved.is_dir() else 'file',
            'size': stat.st_size if resolved.is_file() else None,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_symlink': resolved.is_symlink(),
        }

    async def _search_files(self, directory: str, pattern: str) -> list[str]:
        """Search for files matching a pattern.

        Args:
            directory: Directory to search in.
            pattern: Glob pattern to match files against (e.g., '*.txt', '**/*.py').

        Returns:
            List of matching file paths.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
            PermissionError: If the path is not allowed.
        """
        resolved = self._validate_path(directory)

        if not resolved.exists():
            raise FileNotFoundError(f'Directory not found: {directory}')

        if not resolved.is_dir():
            raise NotADirectoryError(f'Path is not a directory: {directory}')

        matches: list[str] = []
        for entry in resolved.rglob('*'):
            if fnmatch.fnmatch(entry.name, pattern):
                # Validate each matched path is within allowed directories
                try:
                    self._validate_path(entry)
                    matches.append(str(entry))
                except PermissionError:  # pragma: lax no cover
                    # Skip files outside allowed paths
                    pass

        return matches

    async def _delete_file(self, path: str) -> str:
        """Delete a file.

        Args:
            path: Path to the file to delete.

        Returns:
            A success message.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path is a directory (use rmdir for directories).
            PermissionError: If the path is not allowed or delete is disabled.
        """
        self._validate_operation('delete')
        resolved = self._validate_path(path)

        if not resolved.exists():
            raise FileNotFoundError(f'File not found: {path}')

        if resolved.is_dir():
            raise IsADirectoryError(f'Path is a directory, not a file: {path}. Use rmdir for directories.')

        resolved.unlink()
        return f'Successfully deleted {path}'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Get the file system tool definition.

        Args:
            ctx: The run context.

        Returns:
            Dictionary mapping tool name to tool definition.
        """
        # Build allowed operations list based on configuration
        allowed_ops: list[str] = ['read', 'list', 'info', 'search']
        if self.allow_write:
            allowed_ops.append('write')
            if self.allow_delete:
                allowed_ops.append('delete')

        description = (
            f'Access the local file system to read and manage files. Allowed operations: {", ".join(allowed_ops)}.'
        )
        if self.allowed_paths:
            paths_str = ', '.join(str(p) for p in self.allowed_paths)
            description += f' Allowed paths: {paths_str}.'
        if self.allowed_extensions:
            description += f' Allowed extensions: {", ".join(self.allowed_extensions)}.'

        # Build the parameters schema
        operation_enum = allowed_ops
        properties: dict[str, Any] = {
            'operation': {
                'type': 'string',
                'enum': operation_enum,
                'description': 'The operation to perform',
            },
            'path': {
                'type': 'string',
                'description': 'Path to the file or directory',
            },
        }

        if self.allow_write:
            properties['content'] = {
                'type': 'string',
                'description': 'Content to write (required for write operation)',
            }

        properties['pattern'] = {
            'type': 'string',
            'description': 'Search pattern for glob matching (required for search operation)',
        }

        parameters_schema = {
            'type': 'object',
            'properties': properties,
            'required': ['operation', 'path'],
            'additionalProperties': False,
        }

        tool_def = ToolDefinition(
            name='file_system',
            description=description,
            parameters_json_schema=parameters_schema,
        )

        # Create a TypeAdapter for validation that returns a dict
        from typing_extensions import TypedDict

        class FileSystemArgsRequired(TypedDict):
            operation: str
            path: str

        class FileSystemArgs(FileSystemArgsRequired, total=False):
            content: str
            pattern: str

        validator = cast(SchemaValidatorProt, TypeAdapter(FileSystemArgs).validator)

        return {
            'file_system': ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=1,
                args_validator=validator,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        """Call the file system tool with the given arguments.

        Args:
            name: The name of the tool (should be 'file_system').
            tool_args: The arguments for the tool call.
            ctx: The run context.
            tool: The tool definition.

        Returns:
            The result of the file system operation.

        Raises:
            ValueError: If the operation is unknown or required arguments are missing.
        """
        operation = tool_args.get('operation')
        path = tool_args.get('path')

        if not operation:
            raise ValueError('Missing required argument: operation')
        if not path:
            raise ValueError('Missing required argument: path')

        if operation == 'read':
            return await self._read_file(path)
        elif operation == 'write':
            content = tool_args.get('content')
            if content is None:
                raise ValueError('Missing required argument for write operation: content')
            return await self._write_file(path, content)
        elif operation == 'list':
            return await self._list_directory(path)
        elif operation == 'info':
            return await self._file_info(path)
        elif operation == 'search':
            pattern = tool_args.get('pattern')
            if not pattern:
                raise ValueError('Missing required argument for search operation: pattern')
            return await self._search_files(path, pattern)
        elif operation == 'delete':
            return await self._delete_file(path)
        else:
            raise ValueError(f'Unknown operation: {operation}')
