from pathlib import Path

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CompositeFilesystem,
    FilesystemMount,
    LocalWorkspace,
    SupportsCommands,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _mount(path: Path) -> FilesystemMount:
    return FilesystemMount(filesystem=LocalWorkspace(path), source_root=str(path))


async def test_composite_is_a_filesystem_only_workspace(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    skills = tmp_path / 'skills'
    data.mkdir()
    skills.mkdir()
    (data / 'input.csv').write_text('a,b\n1,2\n')
    (skills / 'guide.md').write_text('# Guide\n')

    backend = CompositeFilesystem({'/data': _mount(data), '/skills': _mount(skills)})
    workspace = Workspace(backend)

    assert isinstance(backend, WorkspaceBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert not isinstance(backend, SupportsCommands)
    assert await workspace.read_text('/data/input.csv') == 'a,b\n1,2\n'
    assert await workspace.read_text('/skills/guide.md') == '# Guide\n'
    with pytest.raises(UserError, match='does not support command execution'):
        await workspace.run(['cat', '/data/input.csv'])


async def test_composite_routes_writes_and_rewrites_metadata_paths(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    backend = CompositeFilesystem({'/data': _mount(data)}, working_dir='/data')
    workspace = Workspace(backend)

    await workspace.write_text('nested/output.txt', 'result')

    assert (data / 'nested' / 'output.txt').read_text() == 'result'
    entry = await workspace.stat('nested/output.txt')
    assert entry.name == 'output.txt'
    assert entry.path == '/data/nested/output.txt'
    assert entry.size == 6


async def test_composite_synthesizes_mounts_and_virtual_parents(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    skills = tmp_path / 'skills'
    data.mkdir()
    skills.mkdir()
    backend = CompositeFilesystem({'/team/data': _mount(data), '/team/skills': _mount(skills)})
    workspace = Workspace(backend)

    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/')] == [('team', '/team')]
    assert [(entry.name, entry.path) for entry in await workspace.list_dir('/team')] == [
        ('data', '/team/data'),
        ('skills', '/team/skills'),
    ]
    assert (await workspace.stat('/team')).is_dir
    assert (await workspace.stat('/team/data')).is_dir
    assert await workspace.exists('/team/skills')
    await workspace.make_dir('/team')  # Existing virtual parents have mkdir -p semantics.


async def test_composite_rejects_operations_outside_mounts(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()
    workspace = Workspace(CompositeFilesystem({'/data': _mount(data)}))

    assert not await workspace.exists('/other/file.txt')
    with pytest.raises(FileNotFoundError, match=r'/other/file\.txt'):
        await workspace.read_bytes('/other/file.txt')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await workspace.remove('/data')
    with pytest.raises(PermissionError, match='Cannot remove composite mount'):
        await workspace.remove('/')


@pytest.mark.parametrize(
    'mounts',
    [
        {},
        {'data': None},
        {'/data/../other': None},
    ],
)
def test_composite_rejects_invalid_mount_tables(mounts: dict[str, None]) -> None:
    with pytest.raises(ValueError):
        CompositeFilesystem(mounts)  # type: ignore[arg-type]


def test_composite_rejects_overlapping_mounts(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    archive = tmp_path / 'archive'
    data.mkdir()
    archive.mkdir()

    with pytest.raises(ValueError, match='must not overlap'):
        CompositeFilesystem({'/data': _mount(data), '/data/archive': _mount(archive)})


def test_composite_rejects_a_working_directory_outside_its_namespace(tmp_path: Path) -> None:
    data = tmp_path / 'data'
    data.mkdir()

    with pytest.raises(ValueError, match='must be the virtual root, a mount point, or a virtual parent'):
        CompositeFilesystem({'/data': _mount(data)}, working_dir='/workspace')
