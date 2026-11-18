"""Regression for unknown file sizes in https://github.com/pydantic/pydantic-ai/issues/13532."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from pydantic_ai.workspaces import FileEntry, LocalWorkspaceBackend
from pydantic_ai_harness.file_system import FileSystem, FileSystemToolset


class NoSizeBackend:
    """The issue's files-only workspace, whose stat reports no size."""

    def __init__(self, root: Path) -> None:
        self._local = LocalWorkspaceBackend(root)

    async def working_dir(self) -> str:
        return await self._local.working_dir()

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._local.list_dir(path)

    async def realpath(self, path: str) -> str:
        return await self._local.realpath(path)

    async def read_bytes(self, path: str) -> bytes:
        return await self._local.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._local.write_bytes(path, data)

    async def exists(self, path: str) -> bool:
        return await self._local.exists(path)

    async def make_dir(self, path: str) -> None:
        await self._local.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._local.remove(path)

    async def stat(self, path: str) -> FileEntry:
        return replace(await self._local.stat(path), size=None)


class TestFileSystem:
    async def test_search_skips_oversized_file_with_unknown_size(self, tmp_path: Path) -> None:
        (tmp_path / 'large.txt').write_bytes(b'needle\n' + b'x' * (10 * 1024 * 1024))
        tools = FileSystem[None]().get_toolset()
        assert isinstance(tools, FileSystemToolset)

        result = await tools.search_files('needle', path='large.txt', workspace=NoSizeBackend(tmp_path))

        assert result == '[1 files skipped (too large or unreadable): large.txt]'
