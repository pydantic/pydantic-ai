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
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend
from pydantic_ai_harness.coder import Coder

from .._tool_calls import call_tools
from .fake_boxlite import FakeBox, FakeBoxLite, FakeHandle, PanicException

GONE = WorkspaceRef(provider='boxlite', id='pydantic-ai-' + '0' * 32)


@pytest.fixture
def working_dir(tmp_path: Path) -> str:
    # Resolved so the directory `pwd -P` reports matches it on hosts whose temp directory is a symlink.
    return str(tmp_path.resolve() / 'workspace')


class _Abort(BaseException):
    """A `BaseException` that asyncio stores on a task, unlike `KeyboardInterrupt`."""


async def _command_started(fake_boxlite: FakeBoxLite) -> None:
    """Wait until the command has started: creating the box runs one process, its `mkdir`, and the command a second."""
    with anyio.fail_after(5):
        while len(fake_boxlite.spawned) < 2:
            await asyncio.sleep(0.01)


def _awaiting(task: asyncio.Task[Any]) -> list[str]:
    """The names of the coroutines `task` is suspended in, outermost first."""
    names: list[str] = []
    awaited: Any = task.get_coro()
    while hasattr(awaited, 'cr_code'):
        names.append(awaited.cr_code.co_name)
        awaited = awaited.cr_await
    return names


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
    with pytest.raises(ValueError, match='pass either `sandbox` or `ref`, not both'):
        BoxLiteSandboxBackend(sandbox=object(), ref=WorkspaceRef(provider='boxlite', id='dev'))
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
async def test_the_capability_refuses_refs_to_boxes_it_did_not_create(fake_boxlite: FakeBoxLite, box_id: str) -> None:
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
    ((name, box),) = fake_boxlite.boxes.items()
    assert backend.ref == WorkspaceRef(provider='boxlite', id=name)
    assert re.fullmatch('pydantic-ai-[0-9a-f]{32}', name)
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
    program, args, env, cwd = fake_boxlite.commands[-1]
    assert program == '/bin/sh' and args[-3:] == ['/bin/sh', '-c', 'printf "%s %s" "$BASE" "$OVERRIDE"; pwd -P >&2']
    assert env == {'BASE': 'base', 'OVERRIDE': 'run'} and cwd == working_dir


async def test_the_command_env_does_not_break_the_backend(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The script runs with the command's environment, so it must not depend on `PATH` or `IFS`.
    backend = BoxLiteSandboxBackend(working_dir=working_dir, env={'PATH': '/nonexistent', 'IFS': ':'})
    assert (await backend.run(['ls'])).exit_code == 127
    assert (await backend.run(['/bin/echo', 'a b'])).stdout == 'a b\n'
    # The script's own variables start empty whatever the command's environment holds.
    assert (await backend.run(['/bin/echo', 'ran'], env={'t': '1', 'c': '1', 'r': '9'})).stdout == 'ran\n'


async def test_no_working_dir_uses_the_boxs_own(fake_boxlite: FakeBoxLite) -> None:
    backend = BoxLiteSandboxBackend(working_dir=None)
    assert await backend.working_dir() == str(fake_boxlite.root)
    assert await backend.working_dir() == str(fake_boxlite.root)
    assert [cwd for _, _, _, cwd in fake_boxlite.commands] == [None]


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
    (name,) = fake_boxlite.boxes
    # The box exists, so its ref stays available to delete it; a retry attaches instead of creating another.
    assert backend.ref == WorkspaceRef(provider='boxlite', id=name)
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


async def test_a_creation_cancelled_mid_create_waits_for_it_and_deletes_the_box(
    fake_boxlite: FakeBoxLite, working_dir: str
) -> None:
    # BoxLite finishes a create after its caller is cancelled, so the backend waits for it, then deletes the box.
    fake_boxlite.create_gate = gate = asyncio.Event()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    tasks = asyncio.all_tasks()
    caller = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.create_started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    gate.set()
    with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
        await caller
    assert len(fake_boxlite.removed) == 1 and fake_boxlite.boxes == {}
    # Nothing is left for cleanup to delete, and nothing runs on without its caller.
    assert backend.ref is None
    assert asyncio.all_tasks() == tasks


@pytest.mark.parametrize('cancelled', [False, True])
async def test_a_create_boxlite_never_answers_is_given_up_on(
    fake_boxlite: FakeBoxLite,
    working_dir: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cancelled: bool,
) -> None:
    # Waited for even by a cancelled caller, but not forever: a hung BoxLite can't hold the run.
    monkeypatch.setattr('pydantic_ai_harness.boxlite_sandbox._backend._SDK_TIMEOUT', 0.1)
    fake_boxlite.create_gate = asyncio.Event()  # Never set.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.create_started.wait()
    if cancelled:
        caller.cancel()
    with anyio.fail_after(5), pytest.raises(asyncio.CancelledError if cancelled else WorkspaceUnavailableError):
        await caller
    assert 'did not finish creating box' in caplog.text and 'it may still finish' in caplog.text
    assert backend.ref is None


async def test_a_create_the_sdk_cancels_itself_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    fake_boxlite.create_error = asyncio.CancelledError()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceUnavailableError, match='BoxLite cancelled creating box'):
        await backend.get_sandbox()
    assert backend.ref is None and fake_boxlite.boxes == {}


async def test_a_creation_cancelled_mid_boot_deletes_the_box(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # Never set: a cancelled boot stops, as BoxLite's does.
    fake_boxlite.start_gate = asyncio.Event()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    caller = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.start_started.wait()
    caller.cancel()
    with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
        await caller
    assert backend.ref is None and fake_boxlite.boxes == {}
    fake_boxlite.start_gate = None
    await backend.get_sandbox()
    assert len(fake_boxlite.boxes) == 1


async def test_a_cancelled_destroy_waits_for_the_deletion(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    assert backend.ref is not None
    fake_boxlite.remove_gate = gate = asyncio.Event()
    caller = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    await fake_boxlite.remove_started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    gate.set()
    with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
        await caller
    assert fake_boxlite.boxes == {}
    # Nothing is left waiting: a new box with the same capability is created and deleted as usual.
    fresh = BoxLiteSandboxBackend(working_dir=working_dir)
    await fresh.get_sandbox()
    assert fresh.ref is not None
    await BoxLiteSandbox[None]().destroy(fresh.ref)
    assert fake_boxlite.boxes == {}


async def test_two_deletions_of_one_box_both_return(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    ref = backend.ref
    assert ref is not None
    fake_boxlite.remove_gate = gate = asyncio.Event()
    first = asyncio.create_task(BoxLiteSandbox[None]().destroy(ref))
    await fake_boxlite.remove_started.wait()
    fake_boxlite.remove_started.clear()
    second = asyncio.create_task(BoxLiteSandbox[None]().destroy(ref))
    await fake_boxlite.remove_started.wait()
    gate.set()
    with anyio.fail_after(5):
        await first
        await second
    assert fake_boxlite.removed == [ref.id] and fake_boxlite.boxes == {}


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
    ((name, box),) = fake_boxlite.boxes.items()
    assert 'Could not delete BoxLite box' in caplog.text and backend.ref == WorkspaceRef(provider='boxlite', id=name)
    fake_boxlite.start_error = fake_boxlite.remove_error = None
    await backend.get_sandbox()
    assert len(fake_boxlite.boxes) == 1 and box.status == 'running'


@pytest.mark.parametrize('opened', [True, False])
async def test_a_command_started_after_its_caller_was_cancelled_is_stopped(
    fake_boxlite: FakeBoxLite, working_dir: str, opened: bool
) -> None:
    # BoxLite starts a command after its caller is cancelled, so the backend waits for it, then stops it.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.exec_started.clear()
    fake_boxlite.exec_gate = gate = asyncio.Event()
    if not opened:
        fake_boxlite.exec_error = boxlite.NetworkError('vsock reset')
    caller = asyncio.create_task(backend.run(['sleep', '30']))
    await fake_boxlite.exec_started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    started = len(fake_boxlite.spawned)
    gate.set()
    with anyio.fail_after(5), pytest.raises(asyncio.CancelledError):
        await caller
    if opened:
        # Without a timeout, nothing but the backend stopping the command ends it.
        (command,) = fake_boxlite.spawned[started:]
        assert await asyncio.to_thread(command.wait) != 0
    else:
        assert len(fake_boxlite.spawned) == started


async def test_a_timeout_while_the_command_starts_still_stops_it(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The deadline covers starting the command; one that starts after it is waited for and stopped.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    fake_boxlite.exec_started.clear()
    fake_boxlite.exec_gate = gate = asyncio.Event()
    started = len(fake_boxlite.spawned)
    deadline = anyio.current_time() + 0.2
    # Output read before the stop would show up: the command is stopped as soon as it starts.
    running = asyncio.create_task(backend.run(['sh', '-c', 'echo started; exec sleep 30'], timeout=0.2))
    await fake_boxlite.exec_started.wait()
    await anyio.sleep_until(deadline + 0.05)
    assert not running.done()
    gate.set()
    with anyio.fail_after(5), pytest.raises(WorkspaceTimeoutError) as caught:
        await running
    assert (caught.value.stdout, caught.value.stderr) == ('', '')
    (command,) = fake_boxlite.spawned[started:]
    assert await asyncio.to_thread(command.wait) != 0


async def test_a_failed_setup_does_not_wait_for_an_operation_queued_behind_it(
    fake_boxlite: FakeBoxLite, working_dir: str
) -> None:
    # A deletion queued behind the creation waits for it, so the creation's own failure handling mustn't wait for the deletion.
    fake_boxlite.start_gate = gate = asyncio.Event()
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    creating = asyncio.create_task(backend.get_sandbox())
    await fake_boxlite.start_started.wait()
    assert backend.ref is not None
    deleting = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    await asyncio.sleep(0)
    fake_boxlite.exec_error = boxlite.NetworkError('vsock reset')
    gate.set()
    with anyio.fail_after(5):
        with pytest.raises(boxlite.NetworkError):
            await creating
        await deleting
    assert fake_boxlite.boxes == {}


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


async def test_a_box_stopped_again_before_the_retried_command_is_unavailable(
    fake_boxlite: FakeBoxLite, working_dir: str
) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    box = await backend.get_sandbox()
    fake_boxlite.stop(box.name)

    def hold_next_exec() -> asyncio.Event:
        # Each exec waits on the gate that is set when it arrives.
        fake_boxlite.exec_started.clear()
        fake_boxlite.exec_gate = gate = asyncio.Event()
        return gate

    first = hold_next_exec()
    running = asyncio.create_task(backend.run(['true']))
    await fake_boxlite.exec_started.wait()  # The command, on the handle the stop invalidated.
    setup = hold_next_exec()
    first.set()
    await fake_boxlite.exec_started.wait()  # The `mkdir` on a new handle.
    retried = hold_next_exec()
    setup.set()
    await fake_boxlite.exec_started.wait()  # The command again, on the new handle.
    fake_boxlite.stop(box.name)
    retried.set()
    with pytest.raises(WorkspaceUnavailableError, match='was stopped while starting'):
        await running


async def test_failed_creation_can_retry(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    fake_boxlite.create_error = boxlite.InvalidArgumentError('invalid argument: cpus')
    with pytest.raises(boxlite.InvalidArgumentError):
        await backend.get_sandbox()
    assert backend.ref is None
    fake_boxlite.create_error = None
    await backend.get_sandbox()
    assert backend.ref is not None


async def test_a_host_that_cannot_run_boxes_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    fake_boxlite.start_error = boxlite.ConfigError(
        'configuration error: The sandbox could not be initialized on this host.'
    )
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceUnavailableError, match=r'This host cannot run a BoxLite box: .* on this host'):
        await backend.run(['true'])
    # The box it recorded is deleted again.
    assert fake_boxlite.boxes == {} and backend.ref is None


async def test_a_runtime_that_cannot_open_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # Another process holding the BoxLite home directory makes the runtime panic.
    fake_boxlite.runtime_error = PanicException('Failed to acquire runtime lock at /home/.boxlite')
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    with pytest.raises(WorkspaceUnavailableError, match='Could not open the BoxLite runtime: Failed to acquire'):
        await backend.run(['true'])
    # No box was created, so once the other process lets go, a retry creates one.
    assert backend.ref is None
    fake_boxlite.runtime_error = None
    await backend.run(['true'])
    assert backend.ref is not None and len(fake_boxlite.boxes) == 1
    # Attaching keeps the ref: it names a box that exists.
    attaching = BoxLiteSandboxBackend(ref=backend.ref, working_dir=working_dir)
    fake_boxlite.runtime_error = PanicException('Failed to acquire runtime lock at /home/.boxlite')
    with pytest.raises(WorkspaceUnavailableError, match='Could not open the BoxLite runtime'):
        await attaching.run(['true'])
    assert attaching.ref == backend.ref
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


async def test_the_backend_attaches_to_a_box_of_any_name(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    await fake_boxlite.module.Boxlite.default().create(None, name='dev')
    backend = BoxLiteSandboxBackend(ref=WorkspaceRef(provider='boxlite', id='dev'), working_dir=working_dir)
    assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'
    assert backend.ref == WorkspaceRef(provider='boxlite', id='dev')


@pytest.mark.parametrize('name', ['dev', None])
async def test_a_box_passed_in_is_used_and_kept(fake_boxlite: FakeBoxLite, working_dir: str, name: str | None) -> None:
    box = await fake_boxlite.module.Boxlite.default().create(None, name=name)
    backend = BoxLiteSandboxBackend(sandbox=box, working_dir=working_dir)
    # Named by its name, or its id if it has none, so the runtime finds it again.
    assert backend.ref == WorkspaceRef(provider='boxlite', id=name or box.id)
    assert await backend.get_sandbox() is box
    assert (await backend.run(['pwd'])).stdout == f'{working_dir}\n'
    fake_boxlite.stop(box.id)
    # A stop invalidates the handle passed in; the backend finds the box again by its ref.
    assert (await backend.run(['echo', 'again'])).stdout == 'again\n'
    assert len(fake_boxlite.boxes) == 1


async def test_a_box_of_another_boxlite_home_is_refused(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    # The default home has its own `dev`; the backend must not reach it through the passed box's name.
    await fake_boxlite.module.Boxlite.default().create(None, name='dev')
    elsewhere = FakeHandle(fake_boxlite, FakeBox(id='other-home', name='dev', options=None))
    backend = BoxLiteSandboxBackend(sandbox=elsewhere, working_dir=working_dir)
    with pytest.raises(UserError, match="'dev' passed as `sandbox` is not in this process's default BoxLite runtime"):
        await backend.run(['true'])
    assert fake_boxlite.commands == []


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


async def test_a_command_that_kills_its_script_reports_the_signal(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    result = await backend.run(['sh', '-c', 'echo before; kill -KILL $PPID; sleep 1'])
    assert (result.exit_code, result.stdout) == (128 + signal.SIGKILL, 'before\n')


async def test_output_past_the_limit_stops_the_command(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    # `yes` never ends by itself: the cap is checked as output arrives, and the command is stopped.
    with anyio.fail_after(30), pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as caught:
        await backend.run(['yes', 'x'])
    assert caught.value.limit == 10 * 1024 * 1024
    assert caught.value.stdout == 'x\n' * 32 * 1024 and caught.value.stderr == ''
    (box,) = fake_boxlite.boxes.values()
    assert box.processes == set()


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
    await _command_started(fake_boxlite)
    script = fake_boxlite.spawned[-1]
    fake_boxlite.remove_gate = gate = asyncio.Event()
    deletion = asyncio.create_task(BoxLiteSandbox[None]().destroy(backend.ref))
    with anyio.fail_after(5):
        await asyncio.to_thread(script.wait)
        # Ended by the deletion, the command waits for it rather than checking a box being removed, which
        # would fail as `stopped during the command`. It may already be waiting at the first check.
        while '_unavailable' not in _awaiting(command):
            await asyncio.sleep(0.01)  # pragma: lax no cover
    gate.set()
    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await command
    await deletion


async def test_a_box_stopped_during_a_command_is_unavailable(fake_boxlite: FakeBoxLite, working_dir: str) -> None:
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    ((name, box),) = fake_boxlite.boxes.items()
    command = asyncio.create_task(backend.run(['sleep', '3']))
    await _command_started(fake_boxlite)
    fake_boxlite.stop(name)
    with pytest.raises(WorkspaceUnavailableError, match=f"The BoxLite box '{name}' is stopped"):
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
    await _command_started(fake_boxlite)
    fake_boxlite.exec_error = boxlite.UpstreamUnavailableError('gRPC/tonic error: status: Unavailable')
    fake_boxlite.spawned[-1].send_signal(signal.SIGTERM)
    with pytest.raises(WorkspaceUnavailableError, match='stopped during the command: gRPC'):
        await command


@pytest.mark.parametrize('lost', ['stream', 'wait'])
async def test_a_failed_output_stream_or_wait_stops_the_command(
    fake_boxlite: FakeBoxLite, working_dir: str, lost: str
) -> None:
    # In a running box, the SDK's error propagates as itself, and the command doesn't run on unread.
    backend = BoxLiteSandboxBackend(working_dir=working_dir)
    await backend.get_sandbox()
    error = boxlite.UpstreamUnavailableError('gRPC/tonic error: status: Unavailable')
    if lost == 'stream':
        fake_boxlite.stream_error = error
    else:
        fake_boxlite.wait_error = error
    with anyio.fail_after(10), pytest.raises(boxlite.UpstreamUnavailableError):
        await backend.run(['sleep', '30'])
    (box,) = fake_boxlite.boxes.values()
    assert box.processes == set()


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
    (name,) = fake_boxlite.boxes
    assert first.workspace.ref == second.workspace.ref == WorkspaceRef(provider='boxlite', id=name)


async def test_history_does_not_reattach_a_box_the_application_passed_in(
    fake_boxlite: FakeBoxLite, working_dir: str
) -> None:
    box = await fake_boxlite.module.Boxlite.default().create(None, name='dev')
    agent = Agent(TestModel(), capabilities=[BoxLiteSandbox(working_dir=working_dir)])
    first = await agent.run('go', workspace=BoxLiteSandboxBackend(sandbox=box, working_dir=working_dir))
    assert first.workspace.ref == WorkspaceRef(provider='boxlite', id='dev')
    # The history names `dev`, but only the application that chose it can open it again.
    with pytest.raises(
        ValueError, match=re.escape("got 'dev'. To use another box, pass `workspace=BoxLiteSandboxBackend")
    ):
        await agent.run('go', message_history=first.all_messages())


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
