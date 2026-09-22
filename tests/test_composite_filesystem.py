from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai.workspaces import CompositeFilesystem, SupportsFilesystem, WorkspaceBackend, WorkspaceFileEntry

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None


class _MemoryFilesystem(SupportsFilesystem):
    """A filesystem whose public namespace is rooted at `/`, like an object-store adapter."""

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


async def test_composite_selects_filesystems_and_rebases_paths() -> None:
    data = _MemoryFilesystem({'/input.csv': b'a,b\n1,2\n'})
    skills = _MemoryFilesystem({'/guide.md': b'# Guide\n'})
    filesystem = CompositeFilesystem({'/data': data, '/skills': skills})

    assert isinstance(filesystem, SupportsFilesystem)
    assert not isinstance(filesystem, WorkspaceBackend)
    assert await filesystem.read_bytes('/data/input.csv') == b'a,b\n1,2\n'
    assert await filesystem.read_bytes('/skills/guide.md') == b'# Guide\n'
    assert data.calls[-1] == ('read', '/input.csv')
    assert skills.calls[-1] == ('read', '/guide.md')


async def test_root_and_nested_mounts_use_longest_prefix_and_shadow_entries() -> None:
    root = _MemoryFilesystem({'/README.md': b'root', '/skills/old.md': b'old'})
    skills = _MemoryFilesystem({'/guide.md': b'new'})
    filesystem = CompositeFilesystem({'/': root, '/skills': skills})

    assert await filesystem.read_bytes('/README.md') == b'root'
    assert await filesystem.read_bytes('/skills/guide.md') == b'new'
    assert not await filesystem.exists('/skills/old.md')
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/')] == [
        ('README.md', '/README.md'),
        ('skills', '/skills'),
    ]


async def test_nested_mount_routes_more_specific_paths() -> None:
    data = _MemoryFilesystem({'/current.txt': b'current'})
    archive = _MemoryFilesystem({'/old.txt': b'old'})
    filesystem = CompositeFilesystem({'/data': data, '/data/archive': archive})

    assert await filesystem.read_bytes('/data/current.txt') == b'current'
    assert await filesystem.read_bytes('/data/archive/old.txt') == b'old'
    assert data.calls[-1] == ('read', '/current.txt')
    assert archive.calls[-1] == ('read', '/old.txt')
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/data')] == [
        ('archive', '/data/archive'),
        ('current.txt', '/data/current.txt'),
    ]


async def test_composite_can_be_mounted_in_another_composite() -> None:
    team = _MemoryFilesystem({'/review.md': b'team'})
    skills = CompositeFilesystem({'/team': team})
    filesystem = CompositeFilesystem({'/skills': skills})

    assert await filesystem.read_bytes('/skills/team/review.md') == b'team'
    assert team.calls[-1] == ('read', '/review.md')
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/')] == [('skills', '/skills')]
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/skills')] == [('team', '/skills/team')]


async def test_composite_routes_mutations_and_rewrites_metadata_paths() -> None:
    data = _MemoryFilesystem()
    filesystem = CompositeFilesystem({'/data': data})

    await filesystem.make_dir('/data/nested')
    await filesystem.write_bytes('/data/nested/output.txt', b'result')

    assert ('make_dir', '/nested') in data.calls
    assert data.files['/nested/output.txt'] == b'result'
    entry = await filesystem.stat('/data/nested/output.txt')
    assert entry.name == 'output.txt'
    assert entry.path == '/data/nested/output.txt'
    assert entry.size == 6

    await filesystem.remove('/data/nested/output.txt')
    assert not await filesystem.exists('/data/nested/output.txt')


async def test_composite_synthesizes_mounts_and_virtual_parents() -> None:
    filesystem = CompositeFilesystem({'/team/data': _MemoryFilesystem(), '/team/skills': _MemoryFilesystem()})

    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/')] == [('team', '/team')]
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/team')] == [
        ('data', '/team/data'),
        ('skills', '/team/skills'),
    ]
    assert (await filesystem.stat('/team')).is_dir
    assert (await filesystem.stat('/team/data')).is_dir
    assert await filesystem.exists('/team/skills')
    await filesystem.make_dir('/team')


async def test_list_dir_synthesizes_a_mount_whose_child_root_is_missing() -> None:
    data = _MemoryFilesystem()
    data.directories.clear()
    filesystem = CompositeFilesystem({'/data': data})

    assert await filesystem.list_dir('/data') == ()
    assert [(entry.name, entry.path) for entry in await filesystem.list_dir('/')] == [('data', '/data')]


async def test_composite_rejects_operations_outside_mounts() -> None:
    filesystem = CompositeFilesystem({'/data': _MemoryFilesystem()})

    assert not await filesystem.exists('/other/file.txt')
    with pytest.raises(FileNotFoundError, match=r'/other/file\.txt'):
        await filesystem.read_bytes('/other/file.txt')
    with pytest.raises(IsADirectoryError, match='/data'):
        await filesystem.read_bytes('/data')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await filesystem.remove('/data')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await filesystem.remove('/')


async def test_composite_protects_ancestors_of_nested_mounts() -> None:
    filesystem = CompositeFilesystem({'/data/archive': _MemoryFilesystem()})

    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await filesystem.remove('/data')


async def test_composite_normalizes_operation_paths() -> None:
    data = _MemoryFilesystem({'/input.txt': b'data'})
    filesystem = CompositeFilesystem({'/data': data})

    assert await filesystem.read_bytes('/data/./nested/../input.txt') == b'data'
    with pytest.raises(ValueError, match='path must be absolute'):
        await filesystem.read_bytes('data/input.txt')


@pytest.mark.parametrize('mounts', [{}, {'data': _MemoryFilesystem()}, {'/data/../other': _MemoryFilesystem()}])
def test_composite_rejects_invalid_mount_tables(mounts: dict[str, SupportsFilesystem]) -> None:
    with pytest.raises(ValueError):
        CompositeFilesystem(mounts)


async def test_composite_copies_the_mount_table() -> None:
    mounts: dict[str, SupportsFilesystem] = {'/data': _MemoryFilesystem()}
    filesystem = CompositeFilesystem(mounts)
    mounts['/other'] = _MemoryFilesystem({'/file.txt': b'other'})

    assert not await filesystem.exists('/other/file.txt')
