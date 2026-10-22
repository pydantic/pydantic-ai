"""Public smolvm workspace capability tests, over the fake SDK in `fake_smol.py`."""

from __future__ import annotations

import asyncio
import re
import threading
import uuid
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
import smol

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend, _backend

from .._tool_calls import call_tools
from .fake_smol import FakeMachine, FakeSmol, FakeVM

GONE = WorkspaceRef(provider='smol', id='pydantic-ai-' + '0' * 32)


@pytest.fixture
def working_dir(tmp_path: Path) -> str:
    # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
    return str(tmp_path.resolve() / 'workspace')


def test_env_is_not_shown_in_repr() -> None:
    secret = 'sensitive-credential-value'
    assert secret not in repr(SmolSandbox[None](env={'TOKEN': secret}))


def test_root_package_exports_the_capability_and_backend() -> None:
    import pydantic_ai_harness

    assert pydantic_ai_harness.SmolSandbox is SmolSandbox
    assert pydantic_ai_harness.SmolSandboxBackend is SmolSandboxBackend
    assert {'SmolSandbox', 'SmolSandboxBackend'} <= set(pydantic_ai_harness.__all__)


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'cpus': 0}, 'cpus must be an integer of at least 1'),
        ({'memory_mb': True}, 'memory_mb must be an integer of at least 1'),
        ({'working_dir': 'relative'}, 'working_dir must be an absolute POSIX path'),
        ({'defer_loading': True}, 'does not support `defer_loading=True`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, object], message: str) -> None:
    with pytest.raises(UserError, match=message):
        SmolSandbox[None](**settings)  # pyright: ignore[reportArgumentType]


async def test_invalid_backend_arguments_fail_before_any_vm(fake_smol: FakeSmol) -> None:
    with pytest.raises(ValueError, match="expected 'smol'"):
        SmolSandboxBackend(ref=WorkspaceRef(provider='e2b', id='sbx'))
    with pytest.raises(ValueError, match='pass either `sandbox` or `ref`, not both'):
        SmolSandboxBackend(sandbox=cast(Any, object()), ref=WorkspaceRef(provider='smol', id='dev'))
    with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
        SmolSandboxBackend(working_dir='relative')
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        SmolSandboxBackend(env={'COUNT': 1})  # pyright: ignore[reportArgumentType]
    for env in ({'': 'x'}, {'A=B': 'x'}, {'A\0': 'x'}, {'PATH': '/bin\0/evil'}):
        with pytest.raises(ValueError, match='env keys must be non-empty without `=` or NUL'):
            SmolSandboxBackend(env=env)
    backend = SmolSandboxBackend()
    with pytest.raises(ValueError, match='env keys must be non-empty'):
        await backend.run(['true'], env={'A=B': 'x'})
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        await backend.run(['true'], env={'COUNT': 1})  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match='timeout must be a positive finite number'):
        await backend.run(['true'], timeout=0)
    assert backend.ref is None and fake_smol.vms == {}


def test_capability_declines_foreign_reference() -> None:
    capability = SmolSandbox[None]()
    assert capability.get_workspace(None, ref=WorkspaceRef(provider='e2b', id='sbx')) is None  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('machine_id', ['dev', 'smol-1a2b3c4d', '../../etc/passwd', 'pydantic-ai-' + 'A' * 32])
async def test_the_capability_refuses_refs_to_vms_it_did_not_create(fake_smol: FakeSmol, machine_id: str) -> None:
    # A local VM's id is its name, so a ref from history could otherwise reach a VM made by another
    # tool, with host mounts or network.
    ref = WorkspaceRef(provider='smol', id=machine_id)
    capability = SmolSandbox[None]()
    message = 'SmolSandbox only uses VMs it created'
    with pytest.raises(ValueError, match=message):
        capability.get_workspace(None, ref=ref)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=message):
        capability.backend(ref)
    with pytest.raises(ValueError, match=message):
        await capability.destroy(ref)
    assert fake_smol.connections == []


async def test_capability_forwards_creation_options_on_the_local_target(
    fake_smol: FakeSmol, working_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit credential would otherwise make the SDK pick Smol Cloud.
    monkeypatch.setenv('SMOL_CLOUD_TOKEN', 'token')
    capability = SmolSandbox[None](
        image='python:3.12-alpine', cpus=2, memory_mb=2048, allow_internet_access=True, working_dir=working_dir
    )
    backend = capability.get_workspace(None, ref=None)  # pyright: ignore[reportArgumentType]
    assert isinstance(backend, SmolSandboxBackend)
    await backend.run(['true'])
    (vm,) = fake_smol.vms.values()
    assert backend.ref == WorkspaceRef(provider='smol', id=vm.id)
    assert re.fullmatch('pydantic-ai-[0-9a-f]{32}', vm.id) and vm.config.name == vm.id
    assert vm.config.image == 'python:3.12-alpine' and vm.config.persistent
    assert vm.config.resources == smol.ResourceSpec(cpus=2, memory_mb=2048, network=True)
    assert [conn.target if conn is not None else None for _, conn in fake_smol.connections] == ['local']
    assert Path(working_dir).is_dir()


async def test_default_vm_has_no_network_and_its_own_resources(fake_smol: FakeSmol, working_dir: str) -> None:
    await SmolSandboxBackend(working_dir=working_dir).get_sandbox()
    (vm,) = fake_smol.vms.values()
    assert vm.config.image is None
    assert vm.config.resources == smol.ResourceSpec(network=False)


async def test_commands_get_the_working_dir_and_layered_env(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir, env={'BASE': 'base', 'OVERRIDE': 'base'})
    result = await backend.run('printf "%s %s" "$BASE" "$OVERRIDE"; pwd -P >&2', shell=True, env={'OVERRIDE': 'run'})
    assert (result.exit_code, result.stdout, result.stderr) == (0, 'base run', f'{working_dir}\n')
    argv, options = fake_smol.commands[-1]
    assert argv[:2] == ['/bin/sh', '-c'] and argv[3:] == [
        'sh',
        '/bin/sh',
        '-c',
        'printf "%s %s" "$BASE" "$OVERRIDE"; pwd -P >&2',
    ]
    assert options is not None and options.workdir == working_dir and options.timeout is None


async def test_no_working_dir_uses_the_vms_own(fake_smol: FakeSmol) -> None:
    backend = SmolSandboxBackend(working_dir=None)
    assert await backend.working_dir() == str(fake_smol.root)
    assert [options.workdir if options else None for _, options in fake_smol.commands] == [None]


async def test_unresolvable_working_dir_is_an_error(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    Path(working_dir).rmdir()
    with pytest.raises(WorkspaceError, match=f'The working_dir {working_dir!r} or `/bin/sh` is missing'):
        await backend.working_dir()


async def test_failed_working_dir_creation_keeps_the_ref(fake_smol: FakeSmol, tmp_path: Path) -> None:
    (tmp_path / 'file').write_text('not a directory')
    backend = SmolSandboxBackend(working_dir=str(tmp_path / 'file' / 'workspace'))
    with pytest.raises(WorkspaceError, match='Could not create working_dir'):
        await backend.get_sandbox()
    (vm,) = fake_smol.vms.values()
    # The VM exists, so its ref stays available to delete it; a retry attaches instead of booting another.
    assert backend.ref == WorkspaceRef(provider='smol', id=vm.id)
    with pytest.raises(WorkspaceError, match='Could not create working_dir'):
        await backend.get_sandbox()
    assert [machine_id for machine_id, _ in fake_smol.connections] == [None, vm.id]


async def test_hung_setup_commands_time_out(
    fake_smol: FakeSmol, working_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A custom image, or root in the guest, can make the backend's own `mkdir` and `pwd` hang.
    hanging = tmp_path / 'bin'
    hanging.mkdir()
    for program in ('mkdir', 'pwd'):
        (hanging / program).write_text('#!/bin/sh\nexec /bin/sleep 30\n')
        (hanging / program).chmod(0o755)
    monkeypatch.setattr(_backend, '_SETUP_TIMEOUT', 0.2)
    monkeypatch.setenv('PATH', str(hanging))
    backend = SmolSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceTimeoutError, match=r'timed out after 0\.2 seconds'):
        await backend.get_sandbox()
    assert backend.ref is not None and len(fake_smol.vms) == 1
    no_dir = SmolSandboxBackend(working_dir=None, env={'PATH': str(hanging)})
    with pytest.raises(WorkspaceTimeoutError, match=r'timed out after 0\.2 seconds'):
        await no_dir.working_dir()


async def test_pwd_failure_is_an_error(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir, env={'PATH': '/nonexistent'})
    with pytest.raises(WorkspaceError, match=r'`pwd -P` exited 127'):
        await backend.working_dir()


async def test_concurrent_first_use_boots_once(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    first, second = await asyncio.gather(backend.get_sandbox(), backend.get_sandbox())
    assert first is second
    assert len(fake_smol.vms) == 1
    assert await backend.get_sandbox() is first


async def test_cancelled_creation_keeps_the_vm_it_made(fake_smol: FakeSmol, working_dir: str) -> None:
    fake_smol.create_gate = gate = threading.Event()
    backend = SmolSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    assert await asyncio.to_thread(fake_smol.create_started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    gate.set()
    machine = await backend.get_sandbox()
    assert backend.ref == WorkspaceRef(provider='smol', id=machine.id)
    assert len(fake_smol.vms) == 1


async def test_cleanup_of_a_cancelled_boot_deletes_the_vm(fake_smol: FakeSmol, working_dir: str) -> None:
    # The documented cleanup: whatever ends the run, delete the ref if there is one.
    fake_smol.create_gate = gate = threading.Event()
    backend = SmolSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    assert await asyncio.to_thread(fake_smol.create_started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    ref = backend.ref
    assert ref is not None and fake_smol.vms == {}
    cleanup = asyncio.create_task(SmolSandbox[None]().destroy(ref))
    await asyncio.sleep(0)
    gate.set()
    with anyio.fail_after(5):
        await cleanup
    assert fake_smol.vms == {}
    assert [machine_id for machine_id, _ in fake_smol.connections] == [None, ref.id]


async def test_cancelled_destroy_still_deletes(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    # `connect` boots a stopped VM; a cancelled caller must not leave it running undeleted.
    fake_smol.connect_gate = gate = threading.Event()
    caller = asyncio.create_task(SmolSandbox[None]().destroy(backend.ref))
    assert await asyncio.to_thread(fake_smol.connect_started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    gate.set()
    with anyio.fail_after(5):
        await SmolSandbox[None]().destroy(backend.ref)
    assert fake_smol.vms == {}


async def test_destroy_during_a_boot_and_a_failed_attach_still_deletes(fake_smol: FakeSmol, working_dir: str) -> None:
    # An attach to a VM still booting waits for the boot, so it can't forget it for the deletion behind it.
    fake_smol.create_gate = gate = threading.Event()
    booting = SmolSandboxBackend(working_dir=working_dir)
    boot = asyncio.create_task(booting.get_sandbox())
    assert await asyncio.to_thread(fake_smol.create_started.wait, 5)
    assert booting.ref is not None
    attach = asyncio.create_task(SmolSandboxBackend(ref=booting.ref, working_dir=working_dir).get_sandbox())
    await asyncio.sleep(0)
    cleanup = asyncio.create_task(SmolSandbox[None]().destroy(booting.ref))
    await asyncio.sleep(0)
    gate.set()
    with anyio.fail_after(5):
        await asyncio.gather(boot, attach, cleanup)
    assert fake_smol.vms == {}
    assert [machine_id for machine_id, _ in fake_smol.connections] == [None, booting.ref.id, booting.ref.id]


@pytest.mark.parametrize('opened', [True, False])
async def test_a_command_opened_after_its_caller_was_cancelled_is_killed(
    fake_smol: FakeSmol, working_dir: str, opened: bool
) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_smol.open_started.clear()
    fake_smol.open_gate = gate = threading.Event()
    if not opened:
        fake_smol.open_error = smol.SmolError('SMOLVM_ERROR', 'engine busy')
    caller = asyncio.create_task(backend.run(['sleep', '30']))
    assert await asyncio.to_thread(fake_smol.open_started.wait, 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    started = len(fake_smol.spawned)
    gate.set()
    if opened:
        with anyio.fail_after(5):
            while len(fake_smol.spawned) == started:
                await asyncio.sleep(0.01)
            # Without a timeout, nothing but the backend closing the late stream stops the command.
            assert await asyncio.to_thread(fake_smol.spawned[-1].wait) == -9
    else:
        await asyncio.sleep(0.05)
        assert len(fake_smol.spawned) == started


async def test_a_program_named_like_an_option_runs(fake_smol: FakeSmol, working_dir: str, tmp_path: Path) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / '-c').write_text('#!/bin/sh\necho "ran $*"\n')
    (bin_dir / '-c').chmod(0o755)
    backend = SmolSandboxBackend(working_dir=working_dir, env={'PATH': f'{bin_dir}:/usr/bin:/bin'})
    assert (await backend.run(['-c', '/bin/echo', 'WRONG'])).stdout == 'ran /bin/echo WRONG\n'
    missing = await backend.run(['-n', 'x'])
    assert (missing.exit_code, missing.stderr) == (127, 'sh: -n: not found\n')
    await backend.run(['mkdir', '--', '-d'])
    await backend.run(['cp', '--', str(bin_dir / '-c'), '-d/run'])
    assert (await backend.run(['-d/run', 'here'])).stdout == 'ran here\n'


@pytest.mark.smol_live
async def test_live_program_named_like_an_option_runs() -> None:  # pragma: lax no cover
    # BusyBox's `exec` reads a program named `-c` as its own option.
    backend = SmolSandboxBackend(env={'PATH': '/workspace/bin:/usr/bin:/bin'})
    try:
        await backend.run(
            'mkdir bin && printf \'#!/bin/sh\\necho "ran $*"\\n\' > bin/-c && chmod +x bin/-c', shell=True
        )
        assert (await backend.run(['-c', '/bin/echo', 'WRONG'])).stdout == 'ran /bin/echo WRONG\n'
        assert (await backend.run(['-n'])).exit_code == 127
    finally:
        assert backend.ref is not None
        await SmolSandbox[None]().destroy(backend.ref)


@pytest.mark.smol_live
async def test_live_vm_passed_in_is_used() -> None:  # pragma: lax no cover
    # The backend opens its own handle, by id, to a VM the application already has open, made with the
    # SDK's default, non-persistent config.
    config = smol.MachineConfig(name=f'byo-{uuid.uuid4().hex[:12]}')
    machine = await asyncio.to_thread(smol.Machine.create, config, smol.ConnectOptions(target='local'))
    try:
        backend = SmolSandboxBackend(sandbox=smol.AsyncMachine(machine))
        assert backend.ref == WorkspaceRef(provider='smol', id=machine.id)
        await backend.run(['touch', 'made-by-the-agent'])
        assert machine.exec(['ls', '/workspace']).stdout == 'made-by-the-agent\n'
    finally:
        await asyncio.to_thread(machine.delete)


async def test_failed_creation_can_retry(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    fake_smol.create_error = smol.SmolError('SMOLVM_ERROR', 'image pull failed')
    with pytest.raises(smol.SmolError, match='image pull failed'):
        await backend.get_sandbox()
    assert backend.ref is None
    fake_smol.create_error = None
    await backend.get_sandbox()
    assert backend.ref is not None


@pytest.mark.parametrize(
    'code', ['UNSUPPORTED_PLATFORM', 'RUNTIME_NOT_INSTALLED', 'KVM_UNAVAILABLE', 'HYPERVISOR_UNAVAILABLE']
)
async def test_host_that_cannot_run_vms_is_unavailable(fake_smol: FakeSmol, code: str) -> None:
    fake_smol.create_error = smol.SmolError(code, 'no hypervisor')
    with pytest.raises(WorkspaceUnavailableError, match=f'This host cannot run a local smol VM \\({code}\\)'):
        await SmolSandboxBackend().run(['true'])


async def test_the_backend_attaches_to_a_vm_of_any_name(fake_smol: FakeSmol, working_dir: str) -> None:
    fake_smol.module.Machine.create(smol.MachineConfig(name='dev'))
    backend = SmolSandboxBackend(ref=WorkspaceRef(provider='smol', id='dev'), working_dir=working_dir)
    assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'
    assert backend.ref == WorkspaceRef(provider='smol', id='dev')


@pytest.mark.parametrize('wrap', [False, True], ids=['Machine', 'AsyncMachine'])
async def test_a_vm_passed_in_is_used_and_kept(fake_smol: FakeSmol, working_dir: str, wrap: bool) -> None:
    machine = fake_smol.module.Machine.create(smol.MachineConfig(name='dev'))
    backend = SmolSandboxBackend(sandbox=smol.AsyncMachine(machine) if wrap else machine, working_dir=working_dir)
    # On the local target a VM's id is its name, so the backend opens its own handle to it by id.
    assert backend.ref == WorkspaceRef(provider='smol', id='dev')
    assert (await backend.run(['pwd'])).stdout == f'{working_dir}\n'
    assert fake_smol.connections[-1] == ('dev', smol.ConnectOptions(target='local'))
    assert list(fake_smol.vms) == ['dev']


async def test_a_vm_not_on_this_machine_is_unavailable(fake_smol: FakeSmol, working_dir: str) -> None:
    # Like a Smol Cloud machine: the local target doesn't know its id.
    elsewhere = FakeMachine(FakeVM(id='mach-0a1b2c', config=smol.MachineConfig()))
    backend = SmolSandboxBackend(sandbox=cast(Any, elsewhere), working_dir=working_dir)
    with pytest.raises(
        WorkspaceUnavailableError, match="'mach-0a1b2c' passed as `sandbox` is not, or is no longer, a VM on this"
    ):
        await backend.run(['true'])
    assert fake_smol.commands == []


async def test_a_deleted_vm_passed_in_is_unavailable(fake_smol: FakeSmol, working_dir: str) -> None:
    machine = fake_smol.module.Machine.create(smol.MachineConfig(name='dev'))
    backend = SmolSandboxBackend(sandbox=machine, working_dir=working_dir)
    del fake_smol.vms['dev']
    with pytest.raises(WorkspaceUnavailableError, match="'dev' passed as `sandbox` is not, or is no longer, a VM"):
        await backend.run(['true'])


async def test_attaching_to_a_deleted_vm_is_unavailable(fake_smol: FakeSmol) -> None:
    backend = SmolSandbox[None]().backend(GONE)
    with pytest.raises(WorkspaceUnavailableError, match=f"The smol VM '{GONE.id}' no longer exists"):
        await backend.run(['true'])
    assert backend.ref == GONE
    assert fake_smol.vms == {}


async def test_attach_uses_the_ref_not_the_creation_options(fake_smol: FakeSmol, working_dir: str) -> None:
    created = SmolSandboxBackend(working_dir=working_dir)
    await created.run(['touch', 'marker'])
    assert created.ref is not None
    attached = SmolSandbox[None](image='ignored', working_dir=working_dir).backend(created.ref)
    assert (await attached.run(['ls'])).stdout == 'marker\n'
    assert len(fake_smol.vms) == 1


async def test_destroy_deletes_by_ref_and_ignores_a_missing_vm(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    capability = SmolSandbox[None]()
    await capability.destroy(backend.ref)
    assert fake_smol.vms == {}
    await capability.destroy(backend.ref)
    with pytest.raises(ValueError, match="expected 'smol'"):
        await capability.destroy(WorkspaceRef(provider='e2b', id='sbx'))


async def test_destroy_surfaces_other_failures(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    fake_smol.connect_error = smol.SmolError('SMOLVM_ERROR', 'engine busy')
    with pytest.raises(smol.SmolError, match='engine busy'):
        await SmolSandbox[None]().destroy(backend.ref)
    with pytest.raises(smol.SmolError, match='engine busy'):
        await SmolSandboxBackend(ref=backend.ref, working_dir=working_dir).run(['true'])
    assert len(fake_smol.vms) == 1


async def test_timeout_kills_the_command_and_keeps_partial_output(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceTimeoutError, match=r'timed out after 0\.5 seconds') as caught:
        await backend.run('echo started; exec sleep 30', shell=True, timeout=0.5)
    assert caught.value.stdout == 'started\n'
    (vm,) = fake_smol.vms.values()
    assert vm.processes == set()
    # smolvm kills it itself a little later, in case this event loop is too busy to.
    _, options = fake_smol.commands[-1]
    assert options is not None and options.timeout == 5.5


async def test_output_past_the_limit_stops_the_command(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as caught:
        await backend.run('head -c 11000000 /dev/zero | tr "\\0" x; sleep 30', shell=True)
    assert caught.value.limit == 10 * 1024 * 1024
    assert caught.value.stdout == 'x' * 64 * 1024 and caught.value.stderr == ''


async def test_one_huge_output_event_is_cut_to_the_preview(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_smol.exec_events = [{'kind': 'stderr', 'data': 'x' * (11 * 1024 * 1024)}]
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['true'])
    assert caught.value.stdout == '' and caught.value.stderr == 'x' * 64 * 1024


@pytest.mark.parametrize(
    ('events', 'message'),
    [
        ([{'kind': 'stdout', 'data': 'partial'}], 'ended without an exit status'),
        ([{'kind': 'exit', 'exit_code': '0'}], "exit status that is not an integer: '0'"),
        ([{'kind': 'exit', 'exit_code': True}], 'exit status that is not an integer: True'),
        ([{'kind': 'stdout', 'data': b'bytes'}], 'stdout output that is not text: bytes'),
    ],
)
async def test_malformed_command_streams_are_workspace_errors(
    fake_smol: FakeSmol, working_dir: str, events: list[dict[str, object]], message: str
) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_smol.exec_events = events
    with pytest.raises(WorkspaceError, match=message) as caught:
        await backend.run(['true'])
    assert type(caught.value) is WorkspaceError


async def test_engine_errors_are_classified(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    # A live VM's engine failure propagates for durable engines to retry.
    fake_smol.exec_error = '[NETWORK_ERROR] vsock reset'
    with pytest.raises(smol.SmolError, match='vsock reset') as caught:
        await backend.run(['true'])
    assert caught.value.code == 'NETWORK_ERROR'
    # So does one when the liveness probe fails too.
    fake_smol.state_error = smol.SmolError('SMOLVM_ERROR', 'probe failed')
    with pytest.raises(smol.SmolError, match='vsock reset'):
        await backend.run(['true'])
    # A probe that can't find the VM means it is gone.
    fake_smol.state_error = smol.SmolError('NOT_FOUND', 'VM not found')
    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await backend.run(['true'])
    # An error that says the VM is gone while it still runs is not believed: its code comes from its text.
    fake_smol.state_error = None
    fake_smol.exec_error = '[NOT_FOUND] VM not found'
    with pytest.raises(smol.SmolError, match='VM not found'):
        await backend.run(['true'])


async def test_deleted_vm_is_unavailable_to_its_backend(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    await SmolSandbox[None]().destroy(backend.ref)
    with pytest.raises(WorkspaceUnavailableError, match='it was stopped or deleted'):
        await backend.run(['true'])


async def test_stopped_vm_is_unavailable(fake_smol: FakeSmol, working_dir: str) -> None:
    backend = SmolSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    (vm,) = fake_smol.vms.values()
    vm.state = 'stopped'
    fake_smol.exec_error = 'io operation failed: connection closed'
    with pytest.raises(WorkspaceUnavailableError, match=f"The smol VM '{vm.id}' is stopped"):
        await backend.run(['true'])


async def test_agent_without_workspace_use_does_not_boot(fake_smol: FakeSmol) -> None:
    result = await Agent(TestModel(), capabilities=[SmolSandbox()]).run('go')
    assert result.workspace.ref is None
    assert fake_smol.vms == {}


async def test_agent_history_attaches_same_vm(fake_smol: FakeSmol, working_dir: str) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='use_vm', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[SmolSandbox(working_dir=working_dir)])
    tool_results: list[str] = []

    @agent.tool
    async def use_vm(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker.txt'):
            result = await ctx.workspace.read_text('marker.txt')
        else:
            await ctx.workspace.write_text('marker.txt', 'persisted')
            result = 'created'
        tool_results.append(result)
        return result

    first = await agent.run('go')
    second = await agent.run('go', message_history=first.all_messages())
    assert tool_results == ['created', 'persisted']
    (vm,) = fake_smol.vms.values()
    assert first.workspace.ref == second.workspace.ref == WorkspaceRef(provider='smol', id=vm.id)
    assert [machine_id for machine_id, _ in fake_smol.connections] == [None, vm.id]


async def test_history_does_not_reattach_a_vm_the_application_passed_in(fake_smol: FakeSmol, working_dir: str) -> None:
    machine = fake_smol.module.Machine.create(smol.MachineConfig(name='dev'))
    agent = Agent(TestModel(), capabilities=[SmolSandbox(working_dir=working_dir)])
    first = await agent.run('go', workspace=SmolSandboxBackend(sandbox=machine, working_dir=working_dir))
    assert first.workspace.ref == WorkspaceRef(provider='smol', id='dev')
    # The history names `dev`, but only the application that chose it can open it again.
    with pytest.raises(ValueError, match=re.escape("got 'dev'. To use another VM, pass `workspace=SmolSandboxBackend")):
        await agent.run('go', message_history=first.all_messages())


async def test_coder_works_in_the_vm(fake_smol: FakeSmol, working_dir: str) -> None:
    shell_output, read_output = await call_tools(
        [SmolSandbox(working_dir=working_dir), Coder(repo_context=False)],
        [('shell', {'command': 'echo made-in-vm > note.txt'}), ('read_file', {'path': 'note.txt'})],
    )
    assert '"exit_code": 0' in shell_output
    assert 'made-in-vm' in read_output
    assert (Path(working_dir) / 'note.txt').read_text() == 'made-in-vm\n'
    assert len(fake_smol.vms) == 1


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_vms_need_asyncio(fake_smol: FakeSmol, anyio_backend: str) -> None:
    # The smol SDK runs its blocking calls with `asyncio.to_thread`, so Trio gets a clear error, not a crash.
    with pytest.raises(UserError, match='needs the asyncio event loop'):
        await SmolSandboxBackend().run(['true'])
    with pytest.raises(UserError, match='needs the asyncio event loop'):
        await SmolSandbox[None]().destroy(GONE)
    assert fake_smol.vms == {}
