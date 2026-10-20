"""Public BoxLite workspace capability tests, over the fake SDK in `fake_boxlite.py`."""

from __future__ import annotations

import asyncio
import re
import signal
from pathlib import Path
from typing import Any

import anyio
import boxlite
import pytest

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
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend, _backend
from pydantic_ai_harness.coder import Coder

from .._tool_calls import call_tools
from .fake_boxlite import FakeBoxLite, PanicException

GONE = WorkspaceRef(provider='boxlite', id='pydantic-ai-' + '0' * 32)


@pytest.fixture
def working_dir(tmp_path: Path) -> str:
    # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
    return str(tmp_path.resolve() / 'workspace')


class _Abort(BaseException):
    """A `BaseException` that asyncio stores on a task, unlike `KeyboardInterrupt`."""


async def _wrapper_started(fake_boxlite: FakeBoxLite, count: int) -> None:
    """Wait until the fake has spawned `count` processes: creating a box runs two, `mkdir` and its read."""
    with anyio.fail_after(5):
        while len(fake_boxlite.spawned) < count:
            await asyncio.sleep(0.01)


def test_env_and_box_options_are_not_shown_in_repr(fake_boxlite: FakeBoxLite) -> None:
    secret = 'sensitive-credential-value'
    shown = repr(BoxLiteSandbox[None](env={'TOKEN': secret}, box_options={'disk_size_gb': 1, 'user': secret}))
    assert secret not in shown


def test_root_package_exports_the_capability_and_backend() -> None:
    import pydantic_ai_harness

    assert pydantic_ai_harness.BoxLiteSandbox is BoxLiteSandbox
    assert pydantic_ai_harness.BoxLiteSandboxBackend is BoxLiteSandboxBackend
    assert {'BoxLiteSandbox', 'BoxLiteSandboxBackend'} <= set(pydantic_ai_harness.__all__)


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'cpus': 0}, 'cpus must be an integer of at least 1'),
        ({'memory_mb': True}, 'memory_mb must be an integer of at least 1'),
        ({'working_dir': 'relative'}, 'working_dir must be an absolute POSIX path'),
        ({'defer_loading': True}, 'does not support `defer_loading=True`'),
        (
            {'box_options': {'image': 'x', 'env': []}},
            'box_options must not set env, image; use the named argument instead',
        ),
        ({'box_options': {'bogus': 1}}, "unexpected keyword argument 'bogus'"),
        ({'allowed_hosts': ['pypi.org']}, 'allowed_hosts needs allow_internet_access=True'),
        ({'allow_internet_access': True, 'allowed_hosts': []}, 'and at least one host'),
        ({'allow_internet_access': True, 'allowed_hosts': 'pypi.org'}, 'allowed_hosts must be a sequence'),
        ({'allow_internet_access': True, 'allowed_hosts': ['']}, 'allowed_hosts must be a sequence'),
    ],
)
def test_invalid_settings_fail_at_construction(
    fake_boxlite: FakeBoxLite, settings: dict[str, object], message: str
) -> None:
    with pytest.raises(UserError, match=message):
        BoxLiteSandbox[None](**settings)  # pyright: ignore[reportArgumentType]


async def test_invalid_backend_arguments_fail_before_any_box(fake_boxlite: FakeBoxLite) -> None:
    with pytest.raises(ValueError, match="expected 'boxlite'"):
        BoxLiteSandboxBackend(ref=WorkspaceRef(provider='e2b', id='sbx'))
    with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
        BoxLiteSandboxBackend(working_dir='relative')
    with pytest.raises(ValueError, match='box_options must not set auto_delete; use the named argument instead'):
        BoxLiteSandboxBackend(box_options={'auto_delete': 60})
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        BoxLiteSandboxBackend(env={'COUNT': 1})  # pyright: ignore[reportArgumentType]
    backend = BoxLiteSandboxBackend()
    with pytest.raises(ValueError, match='env keys must be non-empty'):
        await backend.run(['true'], env={'A=B': 'x'})
    with pytest.raises(ValueError, match='timeout must be a positive finite number'):
        await backend.run(['true'], timeout=0)
    assert backend.ref is None and fake_boxlite.boxes == {}


def test_capability_declines_foreign_reference(fake_boxlite: FakeBoxLite) -> None:
    capability = BoxLiteSandbox[None]()
    assert capability.get_workspace(None, ref=WorkspaceRef(provider='e2b', id='sbx')) is None  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('box_id', ['dev', 'ItPP9M8DStN7', '../../etc/passwd', 'pydantic-ai-' + 'A' * 32])
async def test_refs_to_boxes_it_did_not_create_are_refused(fake_boxlite: FakeBoxLite, box_id: str) -> None:
    # `get` opens any box of this BoxLite home by name or id, so a ref from history could otherwise
    # reach a box made by another tool, with host mounts or published ports.
    ref = WorkspaceRef(provider='boxlite', id=box_id)
    capability = BoxLiteSandbox[None]()
    message = 'BoxLiteSandbox only uses boxes it created'
    with pytest.raises(ValueError, match=message):
        capability.get_workspace(None, ref=ref)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match=message):
        capability.backend(ref)
    with pytest.raises(ValueError, match=message):
        await capability.destroy(ref)
    assert fake_boxlite.commands == []


async def test_capability_forwards_creation_options(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    capability = BoxLiteSandbox[None](
        image='python:3.12-slim',
        cpus=2,
        memory_mb=2048,
        allow_internet_access=True,
        allowed_hosts=['pypi.org', '10.0.0.0/8'],
        working_dir=working_dir,
        box_options={'disk_size_gb': 4, 'user': 'root'},
    )
    backend = capability.get_workspace(None, ref=None)  # pyright: ignore[reportArgumentType]
    assert isinstance(backend, BoxLiteSandboxBackend)
    await backend.run(['true'])
    (box,) = fake_boxlite.boxes.values()
    assert backend.ref == WorkspaceRef(provider='boxlite', id=box.name)
    assert re.fullmatch('pydantic-ai-[0-9a-f]{32}', box.name)
    options = box.options
    assert (options.image, options.cpus, options.memory_mib, options.disk_size_gb, options.user) == (
        'python:3.12-slim',
        2,
        2048,
        4,
        'root',
    )
    # Kept after it stops, so the ref can boot it again.
    assert options.auto_delete == 0
    assert (options.network.outbound.mode, options.network.outbound.allow_net) == (
        'enabled',
        ['pypi.org', '10.0.0.0/8'],
    )
    assert box.status == 'running' and Path(working_dir).is_dir()


async def test_default_box_has_no_network_and_boxlites_own_image(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    await BoxLiteSandboxBackend(working_dir=working_dir).get_sandbox()
    (box,) = fake_boxlite.boxes.values()
    assert box.options.image is None and box.options.cpus is None
    assert box.options.network.outbound.mode == 'disabled'


async def test_unrestricted_network_has_no_allow_list(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    await BoxLiteSandboxBackend(allow_internet_access=True, working_dir=working_dir).get_sandbox()
    (box,) = fake_boxlite.boxes.values()
    assert (box.options.network.outbound.mode, box.options.network.outbound.allow_net) == ('enabled', [])


async def test_commands_get_the_working_dir_and_layered_env(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir, env={'BASE': 'base', 'OVERRIDE': 'base'})
    result = await backend.run('printf "%s %s" "$BASE" "$OVERRIDE"; pwd -P >&2', shell=True, env={'OVERRIDE': 'run'})
    assert (result.exit_code, result.stdout, result.stderr) == (0, 'base run', f'{working_dir}\n')
    program, args, env, cwd = fake_boxlite.commands[-2]
    assert program == '/bin/sh' and args[:2] == ['-c', _backend._EXEC_SCRIPT]  # pyright: ignore[reportPrivateUsage]
    assert args[4:] == ['/bin/sh', '-c', 'printf "%s %s" "$BASE" "$OVERRIDE"; pwd -P >&2']
    assert env == {'BASE': 'base', 'OVERRIDE': 'run'} and cwd == working_dir
    # The output is read back with the box's own environment, not the command's.
    _, read_args, read_env, read_cwd = fake_boxlite.commands[-1]
    assert read_args[3] == args[3] and read_env == {} and read_cwd is None


async def test_the_command_env_does_not_break_the_backend(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The script runs with the command's environment, so it must not depend on `PATH` or `IFS`.
    backend = BoxLiteSandboxBackend(working_dir=working_dir, env={'PATH': '/nonexistent', 'IFS': ':'})
    assert (await backend.run(['ls'])).exit_code == 127
    assert (await backend.run(['/bin/echo', 'a b'])).stdout == 'a b\n'


async def test_no_working_dir_uses_the_boxs_own(fake_boxlite: FakeBoxLite) -> None:
    backend = BoxLiteSandboxBackend(working_dir=None)
    assert await backend.working_dir() == str(fake_boxlite.root)
    assert await backend.working_dir() == str(fake_boxlite.root)
    assert [cwd for _, _, _, cwd in fake_boxlite.commands] == [None, None]


async def test_missing_working_dir_is_an_error(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    Path(working_dir).rmdir()
    with pytest.raises(WorkspaceError, match=f'The working_dir {working_dir!r} or `/bin/sh` is missing'):
        await backend.working_dir()


async def test_failed_working_dir_creation_keeps_the_ref(fake_boxlite: FakeBoxLite, tmp_path: Path) -> None:
    (tmp_path / 'file').write_text('not a directory')
    backend = BoxLiteSandboxBackend(working_dir=str(tmp_path / 'file' / 'workspace'))
    with pytest.raises(WorkspaceError, match='Could not create working_dir'):
        await backend.get_sandbox()
    (box,) = fake_boxlite.boxes.values()
    # The box exists, so its ref stays available to delete it; a retry attaches instead of creating another.
    assert backend.ref == WorkspaceRef(provider='boxlite', id=box.name)
    with pytest.raises(WorkspaceError, match='Could not create working_dir'):
        await backend.get_sandbox()
    assert len(fake_boxlite.boxes) == 1


async def test_unexpected_pwd_output_is_an_error(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.read_chunks = ['relative\n']
    with pytest.raises(WorkspaceError, match=r"`pwd -P` exited 0 and printed 'relative\\n'"):
        await backend.working_dir()


async def test_concurrent_first_use_creates_once(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    first, second = await asyncio.gather(backend.get_sandbox(), backend.get_sandbox())
    assert first is second
    assert len(fake_boxlite.boxes) == 1
    assert await backend.get_sandbox() is first


async def test_cancelled_creation_keeps_the_box_it_made(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    fake_boxlite.start_gate = gate = asyncio.Event()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.start_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    gate.set()
    box = await backend.get_sandbox()
    assert backend.ref == WorkspaceRef(provider='boxlite', id=box.name)
    assert len(fake_boxlite.boxes) == 1


async def test_cleanup_of_a_cancelled_creation_deletes_the_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The documented cleanup: whatever ends the run, delete the ref if there is one.
    fake_boxlite.start_gate = gate = asyncio.Event()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.start_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    ref = backend.ref
    assert ref is not None
    cleanup = asyncio.create_task(BoxLiteSandbox[None]().destroy(ref))
    await asyncio.sleep(0)
    # The deletion waits for the creation, so it can't overtake it and leave the box behind.
    assert len(fake_boxlite.boxes) == 1
    gate.set()
    with anyio.fail_after(5):
        await cleanup
    assert fake_boxlite.boxes == {}


async def test_cancelled_destroy_still_deletes(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    fake_boxlite.remove_gate = gate = asyncio.Event()
    caller = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    await fake_boxlite.remove_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    # A second deletion of the same box waits for the first, which goes on without its caller.
    second = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    gate.set()
    with anyio.fail_after(5):
        await second
    assert fake_boxlite.boxes == {} and _backend.lifecycle == {}


async def test_failed_start_deletes_the_box_and_can_retry(
    fake_boxlite: FakeBoxLite, working_dir: str, caplog: pytest.LogCaptureFixture
) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    fake_boxlite.start_error = boxlite.StorageError("storage error: Failed to pull image 'missing:tag'")
    with pytest.raises(boxlite.StorageError, match='Failed to pull image'):
        await backend.get_sandbox()
    # A box that never booted is left to no one, so it is deleted and the next use creates another.
    assert backend.ref is None and fake_boxlite.boxes == {}
    fake_boxlite.remove_error = boxlite.DatabaseError('database locked')
    with pytest.raises(boxlite.StorageError):
        await backend.get_sandbox()
    # The box couldn't be deleted, so its ref stays for the caller to delete it, and a retry attaches to it.
    (box,) = fake_boxlite.boxes.values()
    assert 'Could not delete BoxLite box' in caplog.text and backend.ref == WorkspaceRef(
        provider='boxlite', id=box.name
    )
    fake_boxlite.start_error = fake_boxlite.remove_error = None
    await backend.get_sandbox()
    assert len(fake_boxlite.boxes) == 1 and box.status == 'running'


@pytest.mark.parametrize('opened', [True, False])
async def test_a_command_started_after_its_caller_was_cancelled_is_stopped(
    fake_boxlite: FakeBoxLite, working_dir: str, opened: bool
) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.exec_started.clear()
    fake_boxlite.exec_gate = gate = asyncio.Event()
    if not opened:
        fake_boxlite.exec_error = boxlite.NetworkError('vsock reset')
    caller = asyncio.create_task(backend.run(['sleep', '30']))
    await fake_boxlite.exec_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    started = len(fake_boxlite.spawned)
    gate.set()
    if opened:
        with anyio.fail_after(5):
            while len(fake_boxlite.spawned) == started:
                await asyncio.sleep(0.01)
            # Without a timeout, nothing but the backend stopping the late command ends it.
            assert await asyncio.to_thread(fake_boxlite.spawned[started].wait) != 0
    else:
        await asyncio.sleep(0)
        assert len(fake_boxlite.spawned) == started


async def test_a_timeout_while_the_command_starts_still_stops_it(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The deadline covers starting the command, and a command that starts after it is stopped.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.exec_gate = gate = asyncio.Event()
    started = len(fake_boxlite.spawned)
    with anyio.fail_after(5), pytest.raises(WorkspaceTimeoutError) as caught:
        await backend.run(['sleep', '30'], timeout=0.2)
    assert (caught.value.stdout, caught.value.stderr) == ('', '')
    gate.set()
    with anyio.fail_after(5):
        while len(fake_boxlite.spawned) == started:
            await asyncio.sleep(0.01)
        assert await asyncio.to_thread(fake_boxlite.spawned[started].wait) != 0


async def test_a_box_stopped_while_starting_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    created = BoxLiteSandboxBackend(working_dir=working_dir)
    await created.get_sandbox()
    assert created.ref is not None
    fake_boxlite.start_started.clear()
    fake_boxlite.start_gate = gate = asyncio.Event()
    attaching = asyncio.create_task(BoxLiteSandboxBackend(ref=created.ref, working_dir=working_dir).get_sandbox())
    await fake_boxlite.start_started.wait()
    # Stopped between booting and the backend's own `mkdir`, so the handle is no longer valid.
    fake_boxlite.stop(created.ref.id)
    gate.set()
    with pytest.raises(WorkspaceUnavailableError, match='was stopped while starting'):
        await attaching


async def test_failed_creation_can_retry(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    fake_boxlite.create_error = boxlite.InvalidArgumentError('invalid argument: cpus')
    with pytest.raises(boxlite.InvalidArgumentError):
        await backend.get_sandbox()
    assert backend.ref is None
    fake_boxlite.create_error = None
    await backend.get_sandbox()
    assert backend.ref is not None


async def test_a_runtime_that_cannot_open_is_unavailable(fake_boxlite: FakeBoxLite) -> None:
    # Another process holding the BoxLite home directory makes the runtime panic.
    fake_boxlite.runtime_error = PanicException('Failed to acquire runtime lock at /home/.boxlite')
    with pytest.raises(WorkspaceUnavailableError, match='Could not open the BoxLite runtime: Failed to acquire'):
        await BoxLiteSandboxBackend().run(['true'])
    fake_boxlite.runtime_error = _Abort()
    with pytest.raises(_Abort):
        await BoxLiteSandboxBackend().run(['true'])


async def test_attaching_to_a_deleted_box_is_unavailable(fake_boxlite: FakeBoxLite) -> None:
    backend = BoxLiteSandbox[None]().backend(GONE)
    with pytest.raises(WorkspaceUnavailableError, match=f"The BoxLite box '{GONE.id}' no longer exists"):
        await backend.run(['true'])
    assert backend.ref == GONE
    assert fake_boxlite.boxes == {}


async def test_attach_uses_the_ref_not_the_creation_options(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    created = BoxLiteSandboxBackend(working_dir=working_dir)
    await created.run(['touch', 'marker'])
    assert created.ref is not None
    (box,) = fake_boxlite.boxes.values()
    box.status = 'stopped'
    attached = BoxLiteSandbox[None](image='ignored', working_dir=working_dir).backend(created.ref)
    assert (await attached.run(['ls'])).stdout == 'marker\n'
    # Attaching boots a stopped box again.
    assert len(fake_boxlite.boxes) == 1 and box.status == 'running'


async def test_destroy_deletes_by_ref_and_ignores_a_missing_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    capability = BoxLiteSandbox[None]()
    await capability.destroy(backend.ref)
    assert fake_boxlite.boxes == {}
    await capability.destroy(backend.ref)
    with pytest.raises(ValueError, match="expected 'boxlite'"):
        await capability.destroy(WorkspaceRef(provider='e2b', id='sbx'))


async def test_destroy_surfaces_other_failures(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    fake_boxlite.remove_error = boxlite.DatabaseError('database locked')
    with pytest.raises(boxlite.DatabaseError, match='database locked'):
        await BoxLiteSandbox[None]().destroy(backend.ref)
    assert len(fake_boxlite.boxes) == 1


async def test_timeout_stops_the_command_and_keeps_partial_output(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceTimeoutError, match=r'timed out after 0\.5 seconds') as caught:
        await backend.run('echo started; echo oops >&2; exec sleep 30', shell=True, timeout=0.5)
    assert (caught.value.stdout, caught.value.stderr) == ('started\n', 'oops\n')
    (box,) = fake_boxlite.boxes.values()
    assert box.processes == set()
    # The output files are removed with the rest.
    output = fake_boxlite.commands[-2][1][3]
    assert not Path(f'{output}.stdout').exists()


async def test_a_command_that_kills_its_script_reports_the_signal(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    result = await backend.run(['sh', '-c', 'echo before; kill -KILL $PPID; sleep 1'])
    assert (result.exit_code, result.stdout) == (128 + signal.SIGKILL, 'before\n')


async def test_output_past_the_limit_is_an_error(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as caught:
        await backend.run('head -c 11000000 /dev/zero | tr "\\0" x', shell=True)
    assert caught.value.limit == 10 * 1024 * 1024
    assert caught.value.stdout == 'x' * 64 * 1024 and caught.value.stderr == ''


async def test_one_huge_output_chunk_is_cut_to_the_preview(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.read_chunks = ['x' * (11 * 1024 * 1024)]
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['true'])
    assert caught.value.stdout == 'x' * 64 * 1024 and caught.value.stderr == ''


async def test_output_boxlite_dropped_is_an_error(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.read_chunks = ['start', '[boxlite] stdout output dropped 4096 bytes\r\n', 'end']
    with pytest.raises(WorkspaceError, match=r'BoxLite dropped part of the command output: .* dropped 4096 bytes'):
        await backend.run(['true'])
    # The same text inside a chunk of the command's own output is just output.
    fake_boxlite.read_chunks = ['x [boxlite] stdout output dropped 1 bytes\r\n']
    assert (await backend.run(['true'])).stdout == 'x [boxlite] stdout output dropped 1 bytes\r\n'


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
async def test_lost_output_is_an_error(fake_boxlite: FakeBoxLite, working_dir: str, stream: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    gate = Path(working_dir).parent / 'gate'
    command = asyncio.create_task(
        backend.run(['sh', '-c', 'while [ ! -f "$1" ]; do sleep 0.01; done', 'sh', str(gate)])
    )
    await _wrapper_started(fake_boxlite, 3)
    output = Path(f'{fake_boxlite.commands[-1][1][3]}.{stream}')
    with anyio.fail_after(5):
        while not output.exists():
            await asyncio.sleep(0.01)
    output.unlink()
    gate.touch()
    with pytest.raises(WorkspaceError, match=r'Could not read the command output in BoxLite box .* \(exit code [12]\)'):
        await command


async def test_engine_errors_are_classified(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    # A running box's engine failure propagates for durable engines to retry.
    fake_boxlite.exec_error = boxlite.NetworkError('vsock reset')
    with pytest.raises(boxlite.NetworkError, match='vsock reset'):
        await backend.run(['true'])
    # So does one when checking the box fails too.
    fake_boxlite.get_info_error = boxlite.DatabaseError('database locked')
    with pytest.raises(boxlite.NetworkError, match='vsock reset'):
        await backend.run(['true'])
    # So does a runtime that can no longer open.
    fake_boxlite.get_info_error = None
    fake_boxlite.runtime_error = PanicException('runtime lock')
    with pytest.raises(boxlite.NetworkError, match='vsock reset'):
        await backend.run(['true'])


async def test_deleted_box_is_unavailable_to_its_backend(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    await BoxLiteSandbox[None]().destroy(backend.ref)
    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await backend.run(['true'])


async def test_a_deletion_ending_a_command_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The box reports itself running while the deletion ends its commands; the command waits for it.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    command = asyncio.create_task(backend.run(['sleep', '3']))
    await _wrapper_started(fake_boxlite, 3)
    script = fake_boxlite.spawned[-1]
    fake_boxlite.remove_gate = gate = asyncio.Event()
    deletion = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    with anyio.fail_after(5):
        await asyncio.to_thread(script.wait)
    # Ended by the deletion, the command waits for it rather than reading output from a box being removed,
    # which would fail as `stopped during the command`.
    gate.set()
    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await command
    await deletion


async def test_a_box_stopped_during_a_command_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    (box,) = fake_boxlite.boxes.values()
    command = asyncio.create_task(backend.run(['sleep', '3']))
    await _wrapper_started(fake_boxlite, 3)
    fake_boxlite.stop(box.name)
    with pytest.raises(WorkspaceUnavailableError, match=f"The BoxLite box '{box.name}' is stopped"):
        await command
    # Its handle is no longer valid, so the next command boots the box through a new one.
    assert (await backend.run(['true'])).exit_code == 0
    assert box.status == 'running' and len(fake_boxlite.boxes) == 1


async def test_a_box_stopped_elsewhere_while_still_reported_running_is_unavailable(
    fake_boxlite: FakeBoxLite, working_dir: str
) -> None:
    # Stopped from another handle, the box can still report itself running when the command ends.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    command = asyncio.create_task(backend.run(['sleep', '3']))
    await _wrapper_started(fake_boxlite, 3)
    fake_boxlite.exec_error = boxlite.UpstreamUnavailableError('gRPC/tonic error: status: Unavailable')
    fake_boxlite.spawned[-1].send_signal(signal.SIGTERM)
    with pytest.raises(WorkspaceUnavailableError, match='stopped during the command: gRPC'):
        await command


@pytest.mark.parametrize('status', ['stopped', 'running'])
async def test_a_failed_output_read_is_classified(fake_boxlite: FakeBoxLite, working_dir: str, status: str) -> None:
    # The command finished, but reading its output failed: unavailable if the box went away, else the SDK error.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    gate = Path(working_dir).parent / 'gate'
    command = asyncio.create_task(
        backend.run(['sh', '-c', 'while [ ! -f "$1" ]; do sleep 0.01; done', 'sh', str(gate)])
    )
    await _wrapper_started(fake_boxlite, 3)
    (box,) = fake_boxlite.boxes.values()
    fake_boxlite.exec_error = boxlite.UpstreamUnavailableError('gRPC/tonic error: status: Unavailable')
    box.status = status
    gate.touch()
    if status == 'stopped':
        with pytest.raises(WorkspaceUnavailableError, match='is stopped'):
            await command
    else:
        with pytest.raises(boxlite.UpstreamUnavailableError):
            await command


async def test_agent_without_workspace_use_does_not_create(fake_boxlite: FakeBoxLite) -> None:
    result = await Agent(TestModel(), capabilities=[BoxLiteSandbox()]).run('go')
    assert result.workspace.ref is None
    assert fake_boxlite.boxes == {}


async def test_agent_history_attaches_same_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='use_box', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[BoxLiteSandbox(working_dir=working_dir)])
    tool_results: list[str] = []

    @agent.tool
    async def use_box(ctx: RunContext[object]) -> str:
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
    (box,) = fake_boxlite.boxes.values()
    assert first.workspace.ref == second.workspace.ref == WorkspaceRef(provider='boxlite', id=box.name)


async def test_coder_works_in_the_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    shell_output, read_output = await call_tools(
        [BoxLiteSandbox(working_dir=working_dir), Coder(repo_context=False)],
        [('shell', {'command': 'echo made-in-box > note.txt'}), ('read_file', {'path': 'note.txt'})],
    )
    assert '"exit_code": 0' in shell_output
    assert 'made-in-box' in read_output
    assert (Path(working_dir) / 'note.txt').read_text() == 'made-in-box\n'
    assert len(fake_boxlite.boxes) == 1


async def test_get_sandbox_returns_the_sdk_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    box: Any = await backend.get_sandbox()
    assert backend.ref is not None and box.name == backend.ref.id


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_boxes_need_asyncio(fake_boxlite: FakeBoxLite, anyio_backend: str) -> None:
    # The BoxLite SDK returns asyncio futures, so Trio gets a clear error, not a crash.
    with pytest.raises(UserError, match='needs the asyncio event loop'):
        await BoxLiteSandboxBackend().run(['true'])
    with pytest.raises(UserError, match='needs the asyncio event loop'):
        await BoxLiteSandbox[None]().destroy(GONE)
    assert fake_boxlite.boxes == {}
