from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    Mount,
    MountError,
    ReadOnlyWorkspace,
    SupportsMounts,
    Workspace,
    WorkspaceError,
    WorkspaceReadOnlyError,
)

from .durable_exec.test_durable_workspace import FakeDurability
from .workspace_fakes import FakeWorkspace, MemoryFilesystem

pytestmark = pytest.mark.anyio


class MountingWorkspace(FakeWorkspace, SupportsMounts):
    """A fake backend that can mount a plain `MemoryFilesystem` for its commands."""

    def __init__(self, *, keys_last: timedelta | None = None) -> None:
        super().__init__('mounting')
        self.ensured: list[tuple[list[str], str]] = []
        self.keys_last = keys_last

    def check_mount(self, mount: Mount) -> None:
        if type(mount.filesystem) is not MemoryFilesystem:
            raise MountError(f'cannot mount {type(mount.filesystem).__name__}')

    async def ensure_mounts(self, mounts: Sequence[Mount], working_dir: str) -> datetime | None:
        # A real mount awaits the sandbox; this lets the parallel calls in a test overlap it.
        await anyio.sleep(0)
        self.ensured.append(([mount.path for mount in mounts], working_dir))
        return None if self.keys_last is None else datetime.now(UTC) + self.keys_last


async def test_file_operations_under_a_mount_reach_its_filesystem() -> None:
    storage = MemoryFilesystem({'/report.csv': b'a,b'})
    backend = FakeWorkspace('box')
    workspace = Workspace(backend, mounts=[Mount('data', storage)])

    assert await workspace.read_bytes('data/report.csv') == b'a,b'
    await workspace.write_bytes('data/out/summary.md', b'# done')
    await workspace.make_dir('data/empty')

    assert storage.files['/out/summary.md'] == b'# done'
    assert '/empty' in storage.directories
    assert not any(path.startswith('/workspace/data') for path in backend.files)
    assert await workspace.exists('/workspace/data/out')
    assert [entry.path for entry in await workspace.list_dir('data')] == [
        '/workspace/data/empty',
        '/workspace/data/out',
        '/workspace/data/report.csv',
    ]
    assert (await workspace.stat('data/report.csv')).path == '/workspace/data/report.csv'
    assert await workspace.realpath('data/../data/report.csv') == '/workspace/data/report.csv'

    await workspace.remove('data/out')
    assert '/out/summary.md' not in storage.files


async def test_a_mount_shows_as_a_folder_where_it_is_mounted() -> None:
    backend = FakeWorkspace('box', {'/workspace/notes.txt': b''})
    workspace = Workspace(
        backend, mounts=[Mount('data', MemoryFilesystem()), Mount('/mnt/shared/team', MemoryFilesystem())]
    )

    listing = await workspace.list_dir('.')
    assert [(entry.name, entry.is_dir) for entry in listing] == [('data', True), ('notes.txt', False)]
    # `/mnt/shared` is not in the environment, but it holds a mount.
    assert (await workspace.stat('/mnt/shared')).is_dir
    assert await workspace.exists('/mnt')
    assert [entry.path for entry in await workspace.list_dir('/mnt/shared')] == ['/mnt/shared/team']
    with pytest.raises(FileNotFoundError):
        await workspace.list_dir('/elsewhere')


async def test_a_read_only_mount_refuses_writes() -> None:
    storage = MemoryFilesystem({'/a.txt': b'a'})
    workspace = Workspace(FakeWorkspace('box'), mounts=[Mount('data', storage, read_only=True)])

    assert await workspace.read_text('data/a.txt') == 'a'
    for write in (
        workspace.write_bytes('data/a.txt', b'b'),
        workspace.make_dir('data/new'),
        workspace.remove('data/a.txt'),
    ):
        with pytest.raises(WorkspaceReadOnlyError):
            await write
    assert storage.files == {'/a.txt': b'a'}


async def test_a_mount_or_a_folder_holding_one_cannot_be_removed() -> None:
    workspace = Workspace(
        MountingWorkspace(),
        mounts=[Mount('data', MemoryFilesystem()), Mount('out/seen', MemoryFilesystem(), shell=True)],
    )

    for path in ('data', 'out/seen', 'out'):
        with pytest.raises(ValueError, match='which is or holds a mount'):
            await workspace.remove(path)


@pytest.mark.parametrize('path', ['.', '', '/', '//', '..', '../up'])
def test_a_mount_names_a_folder_inside_the_workspace(path: str) -> None:
    with pytest.raises(ValueError, match='must name a folder'):
        Mount(path, MemoryFilesystem())


def test_a_mount_takes_a_filesystem_that_runs_no_commands(tmp_path: Path) -> None:
    # A host directory or a sandbox is an environment, and a `Workspace` brings mounts of its own.
    for backend in (LocalWorkspaceBackend(tmp_path), FakeWorkspace('box'), Workspace(MemoryFilesystem())):
        with pytest.raises(TypeError, match='runs no commands'):
            Mount('data', backend)


async def test_a_mount_shows_the_filesystems_working_directory() -> None:
    exports = MemoryFilesystem({'/exports/a.csv': b'a', '/private.txt': b'p'}, home='/exports')
    workspace = Workspace(FakeWorkspace('box'), mounts=[Mount('data', exports)])

    assert await workspace.read_bytes('data/a.csv') == b'a'
    assert [entry.name for entry in await workspace.list_dir('data')] == ['a.csv']
    assert (await workspace.stat('data')).path == '/workspace/data'
    assert (await workspace.stat('data')).name == 'data'
    await workspace.write_bytes('data/b.csv', b'b')
    assert exports.files['/exports/b.csv'] == b'b'
    # `..` is collapsed against the workspace's path, so it leaves the mount rather than climbing in the filesystem.
    assert not await workspace.exists('data/../private.txt')
    with pytest.raises(ValueError, match='which is or holds a mount'):
        await workspace.remove('data')


async def test_a_filesystem_is_a_workspace_without_a_shell() -> None:
    root = MemoryFilesystem({'/notes.md': b'n'})
    data = MemoryFilesystem({'/a.csv': b'a'})
    workspace = Workspace(root, mounts=[Mount('data', data)])

    assert [entry.name for entry in await workspace.list_dir('/')] == ['data', 'notes.md']
    assert await workspace.read_bytes('/data/a.csv') == b'a'
    with pytest.raises(UserError, match='does not support command execution'):
        await workspace.run('ls')


async def test_an_agent_runs_in_a_filesystem() -> None:
    root = MemoryFilesystem({'/notes.md': b'n'})
    agent = Agent(TestModel())

    result = await agent.run('go', workspace=root)

    assert await result.workspace.read_bytes('notes.md') == b'n'


async def test_overlapping_mounts_are_refused() -> None:
    with pytest.raises(ValueError, match='overlap'):
        Workspace(
            FakeWorkspace('box'), mounts=[Mount('data', MemoryFilesystem()), Mount('data/inner', MemoryFilesystem())]
        )
    # A relative and an absolute path can only be compared once the working directory is known.
    workspace = Workspace(
        FakeWorkspace('box'), mounts=[Mount('data', MemoryFilesystem()), Mount('/workspace/data', MemoryFilesystem())]
    )
    with pytest.raises(ValueError, match='overlap'):
        await workspace.read_bytes('data/a.txt')


async def test_only_shell_mounts_are_mounted() -> None:
    backend = MountingWorkspace()
    seen = Mount('seen', MemoryFilesystem(), description='Shared files', shell=True)
    hidden = Mount('hidden', MemoryFilesystem())
    workspace = ReadOnlyWorkspace(Workspace(backend, mounts=[seen, hidden]))

    assert workspace.mounts == (seen, hidden)
    # Asking only about a mount the shell doesn't see needs nothing mounted.
    assert not await workspace.exists('hidden/x')
    assert backend.ensured == []


async def test_the_backend_mounts_once_before_commands_or_file_operations_need_it() -> None:
    backend = MountingWorkspace()
    workspace = Workspace(
        backend, mounts=[Mount('seen', MemoryFilesystem(), shell=True), Mount('hidden', MemoryFilesystem())]
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(workspace.run, 'ls seen')
        tg.start_soon(workspace.write_bytes, 'seen/a.txt', b'a')
        tg.start_soon(workspace.list_dir, '.')
    await workspace.run('ls')

    assert backend.ensured == [(['seen'], '/workspace')]
    # Once mounted, the folder is an ordinary path in the environment.
    assert backend.files['/workspace/seen/a.txt'] == b'a'


async def test_shell_true_is_refused_where_it_cannot_work(tmp_path: Path) -> None:
    mount = Mount('data', MemoryFilesystem(), shell=True)
    for root in (MemoryFilesystem(), Workspace(MemoryFilesystem())):
        with pytest.raises(UserError, match='this workspace has no shell'):
            Workspace(root, mounts=[mount])
    with pytest.raises(UserError, match='FakeWorkspace cannot mount filesystems for shell commands'):
        Workspace(FakeWorkspace('box'), mounts=[mount])
    with pytest.raises(UserError, match=r'but cannot mount LinkedFilesystem\. Leave'):
        Workspace(MountingWorkspace(), mounts=[Mount('data', LinkedFilesystem(), shell=True)])
    Workspace(MountingWorkspace(), mounts=[mount])

    # Before the run starts, not on the first command.
    agent = Agent(TestModel(call_tools=[]), capabilities=[LocalWorkspace(tmp_path, mounts=[mount])])
    with pytest.raises(UserError, match='but LocalWorkspace cannot mount'):
        await agent.run('go')


async def test_local_workspace_mounts_reach_storage(tmp_path: Path) -> None:
    storage = MemoryFilesystem({'/a.txt': b'a'})
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path, mounts=[Mount('data', storage)], read_only=True)])

    result = await agent.run('go')

    assert await result.workspace.read_text('data/a.txt') == 'a'
    assert not (tmp_path / 'data').exists()


async def test_durable_runs_reject_a_workspace_with_other_mounts(tmp_path: Path) -> None:
    storage = MemoryFilesystem()
    agent = Agent(
        TestModel(),
        name='ws',
        capabilities=[LocalWorkspace(tmp_path, mounts=[Mount('data', storage)]), FakeDurability()],
    )
    backend = LocalWorkspaceBackend(tmp_path)

    await agent.run('go', workspace=Workspace(backend, mounts=[Mount('data', storage)]))
    with pytest.raises(UserError, match='policy would be lost'):
        await agent.run('go', workspace=Workspace(backend, mounts=[Mount('other', storage)]))
    # A workspace with no policy of its own takes the capability's mounts.
    result = await agent.run('go', workspace=Workspace(backend))
    assert result.workspace.mounts == (Mount('data', storage),)


class LinkedFilesystem(MemoryFilesystem):
    """A filesystem at `/files` with two symlinks: `latest` -> `2026`, and `escape` -> `/elsewhere`."""

    def __init__(self) -> None:
        super().__init__(home='/files')

    async def realpath(self, path: str) -> str:
        return path.replace('/files/latest', '/files/2026', 1).replace('/files/escape', '/elsewhere', 1)


async def test_realpath_follows_links_the_filesystem_resolves() -> None:
    workspace = Workspace(FakeWorkspace('box'), mounts=[Mount('data', LinkedFilesystem())])

    assert await workspace.realpath('data/latest/a.csv') == '/workspace/data/2026/a.csv'
    assert await workspace.realpath('data') == '/workspace/data'
    with pytest.raises(WorkspaceError, match='links outside its mount'):
        await workspace.realpath('data/escape/a.csv')


async def test_commands_clear_mounts_an_earlier_configuration_left() -> None:
    backend = MountingWorkspace()

    await Workspace(backend).run('ls')
    assert backend.ensured == []
    # A workspace that manages mounts but has none for commands still clears what an earlier one left.
    await Workspace(backend, mounts=[]).run('ls')
    await Workspace(backend, mounts=[Mount('hidden', MemoryFilesystem())]).run('ls')
    assert backend.ensured == [([], '/workspace'), ([], '/workspace')]


async def test_mounts_are_ensured_again_before_their_keys_expire() -> None:
    lasting = MountingWorkspace(keys_last=timedelta(hours=1))
    expiring = MountingWorkspace(keys_last=timedelta(seconds=30))

    for backend in (lasting, expiring):
        workspace = Workspace(backend, mounts=[Mount('seen', MemoryFilesystem(), shell=True)])
        await workspace.run('ls')
        await workspace.run('ls')

    assert len(lasting.ensured) == 1
    assert len(expiring.ensured) == 2
