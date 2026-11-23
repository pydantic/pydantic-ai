"""`SmolvmCliBackend`, which drives the `smolvm` CLI where smolmachines has no wheel, such as on Windows.

Over the fake CLI in `fake_smolvm.py`, whose machines run each command on this machine.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness._smolvm_cli import SmolvmCliBackend, delete_machine, find_cli

from .fake_smolvm import FakeSmolvm, put_smolvm_in


async def _delete(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await delete_machine(backend.ref.id)


class TestFakeSmolvmCliBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def working_dir(self, fake_smolvm: FakeSmolvm, tmp_path: Path) -> str:
        # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
        return str(tmp_path.resolve() / 'workspace')

    @pytest.fixture
    def backend(self, working_dir: str) -> SmolvmCliBackend:
        return SmolvmCliBackend(working_dir=working_dir)

    @pytest.mark.skip(reason='shell paging at a host process per 64 KiB; smolvm itself runs it in a VM')
    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None: ...

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # The fake CLI's process ends once its command's output pipes close.
        return False

    @pytest.fixture
    def fresh_backend(self, working_dir: str) -> Callable[[], WorkspaceBackend]:
        return lambda: SmolvmCliBackend(working_dir=working_dir)

    @pytest.fixture
    def attach_backend(self, working_dir: str) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: SmolvmCliBackend(ref=ref, working_dir=working_dir)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


def _delete_now(ref: WorkspaceRef | None) -> None:  # pragma: no cover - live tier
    # Blocking, because a class-scoped fixture outlives each test's event loop.
    cli = find_cli()
    if ref is not None and cli is not None:
        subprocess.run([cli, 'machine', 'delete', '--name', ref.id, '--force'], check=False, capture_output=True)


# The live tier runs without coverage; in CI only its gate runs, to skip it. It needs the real `smolvm` on `PATH`.
@pytest.mark.smol_live
class TestLiveSmolvmCliBackend(WorkspaceBackendSuite):  # pragma: lax no cover
    # Class-scoped so the rules share one VM instead of booting one each.
    @pytest.fixture(scope='class')
    @classmethod
    def backend(cls) -> Iterator[SmolvmCliBackend]:
        backend = SmolvmCliBackend()
        yield backend
        _delete_now(backend.ref)

    @pytest.fixture
    async def fresh_backend(self) -> AsyncIterator[Callable[[], WorkspaceBackend]]:
        created: list[SmolvmCliBackend] = []

        def factory() -> SmolvmCliBackend:
            backend = SmolvmCliBackend()
            created.append(backend)
            return backend

        yield factory
        for backend in created:
            if backend.ref is not None:
                await delete_machine(backend.ref.id)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: SmolvmCliBackend(ref=ref)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _delete


@pytest.fixture
def working_dir(fake_smolvm: FakeSmolvm, tmp_path: Path) -> str:
    return str(tmp_path.resolve() / 'workspace')


async def _booted(working_dir: str) -> SmolvmCliBackend:
    backend = SmolvmCliBackend(working_dir=working_dir, image='python:3.13', allow_internet_access=True)
    await backend.working_dir()
    return backend


class TestSmolvmCliBackend:
    async def test_commands_go_on_stdin(self, fake_smolvm: FakeSmolvm, working_dir: str) -> None:
        backend = await _booted(working_dir)
        assert backend.ref is not None
        assert fake_smolvm.created == [['--name', backend.ref.id, '--net', '--image', 'python:3.13']]
        # Neither arguments nor the environment reach a command line here, which Windows caps at 32,767 characters.
        long = 'x' * 40_000
        result = await backend.run(
            ['sh', '-c', 'printf %s "$1" | wc -c; printf %s "$KEY"', 'sh', long], env={'KEY': 'v=1'}
        )
        assert result == CommandResult(exit_code=0, stdout='40000\nv=1', stderr='')
        assert any(long in script for script in fake_smolvm.scripts)
        arguments = [argument for command in fake_smolvm.commands for argument in command]
        assert not any(long in argument or 'v=1' in argument for argument in arguments)
        await delete_machine(backend.ref.id)

    async def test_timeout(self, fake_smolvm: FakeSmolvm, working_dir: str) -> None:
        backend = await _booted(working_dir)
        with pytest.raises(WorkspaceTimeoutError, match=r'timed out after 0\.1 seconds'):
            await backend.run(['sleep', '10'], timeout=0.1)
        # smolvm also kills the command in the VM, a little after this machine gives up on it.
        sleep = fake_smolvm.commands[-1]
        assert sleep[sleep.index('--timeout') + 1] == '5100ms'

    async def test_output_limit(self, fake_smolvm: FakeSmolvm, working_dir: str) -> None:
        backend = await _booted(working_dir)
        # Killed once its output passes 10 MiB, rather than held in memory however much it prints.
        with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as raised:
            await backend.run(['yes'])
        assert raised.value.stdout.startswith('y\ny\n')
        assert (await backend.run(['true'])).exit_code == 0

    @pytest.mark.parametrize('state', ['stopped', 'created', 'paused'])
    async def test_a_stopped_vm_is_started(self, fake_smolvm: FakeSmolvm, working_dir: str, state: str) -> None:
        backend = await _booted(working_dir)
        assert backend.ref is not None
        # As after this machine restarts.
        fake_smolvm.machines[backend.ref.id] = state
        assert (await SmolvmCliBackend(ref=backend.ref, working_dir=working_dir).run(['true'])).exit_code == 0
        assert fake_smolvm.machines[backend.ref.id] == 'running'
        fake_smolvm.machines[backend.ref.id] = 'error'
        with pytest.raises(WorkspaceUnavailableError, match=r'is error\.$'):
            await SmolvmCliBackend(ref=backend.ref, working_dir=working_dir).run(['true'])

    async def test_stop(self, fake_smolvm: FakeSmolvm, working_dir: str) -> None:
        backend = await _booted(working_dir)
        assert backend.ref is not None
        # One that never reached the VM stops nothing.
        await SmolvmCliBackend(ref=backend.ref, working_dir=working_dir).stop()
        assert fake_smolvm.machines[backend.ref.id] == 'running'
        await backend.stop()
        assert fake_smolvm.machines[backend.ref.id] == 'stopped'
        assert (await backend.run(['true'])).exit_code == 0
        assert fake_smolvm.machines[backend.ref.id] == 'running'
        # Stopped by another backend, as another CLAI closing it: the command fails without running again.
        other = SmolvmCliBackend(ref=backend.ref, working_dir=working_dir)
        await other.run(['true'])
        await other.stop()
        with pytest.raises(WorkspaceError, match='was stopped before the command ended; run it again'):
            await backend.run(['true'])
        assert (await backend.run(['true'])).exit_code == 0
        fake_smolvm.failures['stop'] = ('SMOLVM_ERROR', 'stuck')
        with pytest.raises(WorkspaceError, match='smolvm machine stop failed: stuck'):
            await backend.stop()
        # A failed stop can be tried again.
        del fake_smolvm.failures['stop']
        await backend.stop()
        assert fake_smolvm.machines[backend.ref.id] == 'stopped'
        # One another backend stopped meanwhile is down, even if smolvm says the stop failed.
        assert (await backend.run(['true'])).exit_code == 0
        await other.run(['true'])
        await other.stop()
        fake_smolvm.failures['stop'] = ('SMOLVM_ERROR', 'machine is not running')
        await backend.stop()
        del fake_smolvm.failures['stop']
        # A VM deleted already has nothing to stop.
        assert (await backend.run(['true'])).exit_code == 0
        await delete_machine(backend.ref.id)
        await backend.stop()

    async def test_smolvm_failures(self, fake_smolvm: FakeSmolvm, working_dir: str) -> None:
        backend = await _booted(working_dir)
        assert backend.ref is not None
        # A listing under `machines`, as some smolvm versions print it, names the VM too.
        fake_smolvm.wrapped_listing = True
        assert (await SmolvmCliBackend(ref=backend.ref, working_dir=working_dir).run(['true'])).exit_code == 0
        # Output that looks like smolvm's own failure is the command's while the VM runs.
        report = json.dumps({'error': {'code': 'NOT_FOUND', 'message': 'no such thing'}})
        result = await backend.run(['sh', '-c', f"echo '{report}' >&2; exit 1"])
        assert (result.exit_code, result.stderr) == (1, f'{report}\n')
        fake_smolvm.machines[backend.ref.id] = 'error'
        with pytest.raises(WorkspaceUnavailableError, match='is error'):
            await backend.run(['true'])
        fake_smolvm.failures['delete'] = ('SMOLVM_ERROR', 'machine is busy')
        with pytest.raises(WorkspaceError, match='smolvm machine delete failed: machine is busy'):
            await delete_machine(backend.ref.id)
        del fake_smolvm.failures['delete']
        await delete_machine(backend.ref.id)
        await delete_machine(backend.ref.id)  # One already gone is fine.
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await backend.run(['true'])

    async def test_failed_boots(self, fake_smolvm: FakeSmolvm, working_dir: str, tmp_path: Path) -> None:
        fake_smolvm.failures['start'] = ('HYPERVISOR_UNAVAILABLE', 'the Windows Hypervisor Platform is not enabled')
        backend = SmolvmCliBackend(working_dir=working_dir)
        with pytest.raises(WorkspaceUnavailableError, match=r'cannot run a local smol VM \(HYPERVISOR_UNAVAILABLE\)'):
            await backend.working_dir()
        # The machine exists, so its ref stays for whoever deletes it.
        assert backend.ref is not None and backend.ref.id in fake_smolvm.machines
        fake_smolvm.failures['start'] = ('SMOLVM_ERROR', 'disk full')
        with pytest.raises(WorkspaceError, match='smolvm machine start failed: disk full'):
            await SmolvmCliBackend(working_dir=working_dir).working_dir()
        del fake_smolvm.failures['start']
        (tmp_path / 'file').touch()
        with pytest.raises(WorkspaceError, match='Could not create working_dir'):
            await SmolvmCliBackend(working_dir=str(tmp_path.resolve() / 'file/workspace')).working_dir()
        fake_smolvm.failures['create'] = ('SMOLVM_ERROR', 'no space left')
        backend = SmolvmCliBackend(working_dir=working_dir)
        with pytest.raises(WorkspaceError, match='smolvm machine create failed: no space left'):
            await backend.working_dir()
        # smolvm made no machine, so a retry makes one under a new name.
        assert backend.ref is None

    async def test_only_its_own_vms(self) -> None:
        # A ref from history could otherwise open any VM of this user, such as one with this machine's folders.
        with pytest.raises(ValueError, match='only uses VMs it created'):
            SmolvmCliBackend(ref=WorkspaceRef(provider='smol', id='dev'))
        with pytest.raises(ValueError, match='only uses VMs it created'):
            await delete_machine('dev')


class TestFindCli:
    def test_only_on_absolute_path_entries(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        installed = put_smolvm_in(tmp_path / 'bin')
        monkeypatch.setenv('PATH', f'{tmp_path / "nowhere"}{os.pathsep}{installed}')
        assert find_cli() == str(installed / 'smolvm')
        # Windows would otherwise run one in the current directory, such as one a checkout holds.
        monkeypatch.chdir(installed)
        monkeypatch.setenv('PATH', f'.{os.pathsep}{tmp_path / "nowhere"}')
        assert find_cli() is None

    def test_windows_programs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        installed = put_smolvm_in(tmp_path / 'bin')
        monkeypatch.setattr('pydantic_ai_harness._smolvm_cli.sys.platform', 'win32')
        monkeypatch.setenv('PATH', str(installed))
        assert find_cli() == str(installed / 'smolvm.exe')
        # Never a batch file, which Windows runs through `cmd.exe`.
        (installed / 'smolvm.exe').unlink()
        (installed / 'smolvm.cmd').touch(mode=0o755)
        assert find_cli() is None

    async def test_without_smolvm(
        self, fake_smolvm: FakeSmolvm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv('PATH', str(tmp_path / 'nowhere'))
        with pytest.raises(WorkspaceUnavailableError, match='smolvm was not found on PATH'):
            await SmolvmCliBackend().working_dir()
        assert fake_smolvm.commands == []
