from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CompositeFilesystem,
    SupportsCommands,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceFileEntry,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None


class _MemoryFilesystem(SupportsFilesystem):
    """A filesystem whose public namespace is rooted at `/`, like an object store adapter."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}
        self.directories = {'/', *(posixpath.dirname(path) for path in self.files)}
        self.calls: list[tuple[str, str]] = []

    async def read_bytes(self, path: str) -> bytes:
        self.calls.append(('read', path))
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.calls.append(('write', path))
        parent = posixpath.dirname(path)
        while parent not in self.directories:
            self.directories.add(parent)
            parent = posixpath.dirname(parent)
        self.files[path] = data

    async def stat(self, path: str) -> WorkspaceFileEntry:
        self.calls.append(('stat', path))
        if path in self.directories:
            return _Entry(posixpath.basename(path) or '/', path, True, None)
        try:
            data = self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None
        return _Entry(posixpath.basename(path), path, False, len(data))

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        self.calls.append(('list', path))
        if path not in self.directories:
            raise FileNotFoundError(path)
        prefix = '/' if path == '/' else path + '/'
        children: dict[str, _Entry] = {}
        for directory in self.directories:
            if directory == path or not directory.startswith(prefix):
                continue
            relative = directory[len(prefix) :]
            if '/' not in relative:
                children[relative] = _Entry(relative, prefix + relative, True, None)
        for file_path, data in self.files.items():
            if not file_path.startswith(prefix):
                continue
            relative = file_path[len(prefix) :]
            if '/' not in relative:
                children[relative] = _Entry(relative, prefix + relative, False, len(data))
        return tuple(children[name] for name in sorted(children))

    async def make_dir(self, path: str) -> None:
        self.calls.append(('make_dir', path))
        self.directories.add(path)

    async def remove(self, path: str) -> None:
        self.calls.append(('remove', path))
        if path in self.files:
            del self.files[path]
        elif path in self.directories:
            self.directories.remove(path)
        else:
            raise FileNotFoundError(path)

    async def exists(self, path: str) -> bool:
        self.calls.append(('exists', path))
        return path in self.files or path in self.directories


async def test_composite_is_a_filesystem_only_workspace() -> None:
    data = _MemoryFilesystem({'/input.csv': b'a,b\n1,2\n'})
    skills = _MemoryFilesystem({'/guide.md': b'# Guide\n'})
    backend = CompositeFilesystem({'/data': data, '/skills': skills})
    workspace = Workspace(backend)

    assert isinstance(backend, WorkspaceBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert not isinstance(backend, SupportsCommands)
    assert await workspace.read_text('/data/input.csv') == 'a,b\n1,2\n'
    assert await workspace.read_text('/skills/guide.md') == '# Guide\n'
    assert data.calls[-1] == ('read', '/input.csv')
    assert skills.calls[-1] == ('read', '/guide.md')
    with pytest.raises(UserError, match='does not support command execution'):
        await workspace.run(['cat', '/data/input.csv'])


async def test_root_and_nested_mounts_use_longest_prefix_and_shadow_entries() -> None:
    root = _MemoryFilesystem({'/README.md': b'root', '/skills/old.md': b'old'})
    skills = _MemoryFilesystem({'/guide.md': b'new'})
    workspace = Workspace(CompositeFilesystem({'/': root, '/skills': skills}))

    assert await workspace.read_text('/README.md') == 'root'
    assert await workspace.read_text('/skills/guide.md') == 'new'
    assert not await workspace.exists('/skills/old.md')
    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/')] == [
        ('README.md', '/README.md'),
        ('skills', '/skills'),
    ]


async def test_nested_mount_routes_more_specific_paths() -> None:
    data = _MemoryFilesystem({'/current.txt': b'current'})
    archive = _MemoryFilesystem({'/old.txt': b'old'})
    workspace = Workspace(CompositeFilesystem({'/data': data, '/data/archive': archive}))

    assert await workspace.read_text('/data/current.txt') == 'current'
    assert await workspace.read_text('/data/archive/old.txt') == 'old'
    assert data.calls[-1] == ('read', '/current.txt')
    assert archive.calls[-1] == ('read', '/old.txt')
    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/data')] == [
        ('archive', '/data/archive'),
        ('current.txt', '/data/current.txt'),
    ]


async def test_composite_routes_writes_and_rewrites_metadata_paths() -> None:
    data = _MemoryFilesystem()
    workspace = Workspace(CompositeFilesystem({'/data': data}))

    await workspace.write_text('/data/nested/output.txt', 'result')

    assert data.files['/nested/output.txt'] == b'result'
    entry = await workspace.stat('/data/nested/output.txt')
    assert entry.name == 'output.txt'
    assert entry.path == '/data/nested/output.txt'
    assert entry.size == 6


async def test_composite_synthesizes_mounts_and_virtual_parents() -> None:
    backend = CompositeFilesystem({'/team/data': _MemoryFilesystem(), '/team/skills': _MemoryFilesystem()})
    workspace = Workspace(backend)

    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/')] == [('team', '/team')]
    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/team')] == [
        ('data', '/team/data'),
        ('skills', '/team/skills'),
    ]
    assert (await workspace.stat('/team')).is_dir
    assert (await workspace.stat('/team/data')).is_dir
    assert await workspace.exists('/team/skills')
    await workspace.make_dir('/team')


async def test_composite_rejects_operations_outside_mounts() -> None:
    workspace = Workspace(CompositeFilesystem({'/data': _MemoryFilesystem()}))

    assert not await workspace.exists('/other/file.txt')
    with pytest.raises(FileNotFoundError, match=r'/other/file\.txt'):
        await workspace.read_bytes('/other/file.txt')
    with pytest.raises(IsADirectoryError, match='/data'):
        await workspace.read_bytes('/data')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await workspace.remove('/data')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await workspace.remove('/')


async def test_composite_protects_ancestors_of_nested_mounts() -> None:
    workspace = Workspace(CompositeFilesystem({'/data/archive': _MemoryFilesystem()}))

    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await workspace.remove('/data')


@pytest.mark.parametrize('mounts', [{}, {'data': _MemoryFilesystem()}, {'/data/../other': _MemoryFilesystem()}])
def test_composite_rejects_invalid_mount_tables(mounts: dict[str, SupportsFilesystem]) -> None:
    with pytest.raises(ValueError):
        CompositeFilesystem(mounts)


def test_composite_exposes_an_immutable_mount_table() -> None:
    mounts: dict[str, SupportsFilesystem] = {'/data': _MemoryFilesystem()}
    composite = CompositeFilesystem(mounts)
    mounts['/other'] = _MemoryFilesystem()

    assert tuple(composite.mounts) == ('/data',)
    with pytest.raises(TypeError):
        composite.mounts['/other'] = _MemoryFilesystem()  # type: ignore[index]
