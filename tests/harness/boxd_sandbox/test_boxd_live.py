"""Integration tests that require a real boxd machine.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution over boxd's exec stream, a client-owned deadline over a real process, and
real lifecycle state in boxd's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real boxd behavior.

Gating:
  * `boxd_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_BOXD_LIVE=1` opts in explicitly.
  * also requires a non-empty `BOXD_API_KEY`; without one it skips, or fails when
    `BOXD_REQUIRE_LIVE=1`. The gate is `pytest_runtest_setup` in `conftest.py`.
  * the async `client` fixture opens a boxd client per test and closes it when the test ends.

Run locally:
`PYDANTIC_AI_HARNESS_BOXD_LIVE=1 uv run pytest -m boxd_live tests/harness/boxd_sandbox/`
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import boxd
import pytest
from pytest_examples import CodeExample

from pydantic_ai.workspaces import WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.coder import Coder

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_AUTO_SUSPEND_TIMEOUT

pytestmark = pytest.mark.boxd_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


# What a machine sees of boxd's own services: the HTTP status of a token request to the metadata
# endpoint (`000` when the request never reaches it), and whether boxd installed its env vars and
# integrations (`/etc/profile.d/boxd-env.sh`).
_BOXD_SERVICES = (
    "gw=$(ip route | awk '/default/ {print $3}'); "
    'curl -s -m 3 -o /dev/null -w "%{http_code}\\n" "http://$gw:9002/v1/auth/token"; '
    'test -e /etc/profile.d/boxd-env.sh && echo env-installed || echo no-env'
)


@pytest.fixture
async def client() -> AsyncIterator[boxd.AsyncBoxd]:
    """A caller-owned API client for one test's backends, closed when the test ends."""
    client = boxd.AsyncBoxd()
    try:
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def _owned(client: boxd.AsyncBoxd, working_dir: str | None = None) -> AsyncGenerator[BoxdSandboxBackend]:
    """Create a machine and delete it on the way out, even when the test deleted it already."""
    backend = BoxdSandboxBackend(client=client, auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT, working_dir=working_dir)
    try:
        await backend.get_sandbox()
        yield backend
    finally:
        if backend.ref is not None:
            await BoxdSandbox(client=client).destroy(backend.ref)


async def _wait_for_status(client: boxd.AsyncBoxd, machine_id: str, status: str) -> None:
    with anyio.fail_after(120):
        while (await client.machines.get(machine_id)).status != status:
            await anyio.sleep(0.5)


async def test_creates_an_isolated_machine_and_runs_a_command(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that the exec stream reports real output and exit code on
    separate streams, and that the machine boxd creates is isolated: its metadata endpoint refuses token
    requests, and boxd installs no CLI, env vars, or integrations in it, so code the agent runs holds no
    boxd credentials. `test_a_machine_that_is_not_isolated_is_neither_used_nor_deleted` runs the same
    probe on a machine that is not isolated, where it succeeds."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)
        assert (result.stdout, result.stderr, result.exit_code) == ('out\n', 'err\n', 3)

        machine = await backend.get_sandbox()
        assert (machine.networking.isolated, machine.networking.networks) == (True, [])
        probe = await backend.run(['sh', '-c', f'command -v boxd || echo no-cli; {_BOXD_SERVICES}'], timeout=60)
        # The endpoint refuses an isolated machine, and boxd may drop the request before it gets there.
        assert probe.stdout.splitlines() in (['no-cli', '403', 'no-env'], ['no-cli', '000', 'no-env'])


async def test_large_output_is_exact(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that the exec stream delivers a command's output whole and in order."""
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '200000'], timeout=120)
        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 200001))


async def test_reattach_by_ref_reaches_the_same_machine(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that `machines.get` plus a command reaches the same machine by id."""
    path = f'/home/boxd/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        await owner.run(['sh', '-c', 'printf shared > "$1"', 'sh', path], timeout=60)
        assert owner.ref is not None

        attached = BoxdSandboxBackend(client=client, ref=owner.ref)
        assert (await attached.run(['cat', path], timeout=60)).stdout == 'shared'
        assert attached.ref == owner.ref


# A process that outlives the command starting it, and a check that it still runs.
_TICKER = 'setsid sh -c "while :; do date +%s > tick; sleep 1; done" </dev/null >/dev/null 2>&1 & echo $! > ticker.pid'
_TICKING = 'kill -0 "$(cat ticker.pid)" && echo ticking'


async def test_fork_copies_files_and_running_processes_and_the_copies_diverge(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a fork is a new isolated machine on no network, and shows
    what the fake can't: the copy starts with the machine's files and running processes, and the two
    change independently after."""
    sandbox = BoxdSandbox(client=client)
    async with _owned(client) as source:
        await source.run(['sh', '-c', f'echo before > state.txt; {_TICKER}'], timeout=60)
        assert source.ref is not None
        ref = await sandbox.fork(source.ref)
        try:
            assert ref != source.ref
            copy = BoxdSandboxBackend(client=client, ref=ref)
            machine = await copy.get_sandbox()
            assert (machine.networking.isolated, machine.networking.networks) == (True, [])
            assert (
                await copy.run(['sh', '-c', f'cat state.txt; {_TICKING}'], timeout=60)
            ).stdout == 'before\nticking\n'
            await copy.run(['sh', '-c', 'echo after > state.txt'], timeout=60)
            assert (await source.run(['cat', 'state.txt'], timeout=60)).stdout == 'before\n'
        finally:
            await sandbox.destroy(ref)


async def test_fork_wakes_a_suspended_machine_so_the_copy_keeps_its_memory(client: boxd.AsyncBoxd) -> None:
    """boxd forks a suspended machine from its disk alone, without its running processes or the writes it
    had not flushed to disk; `fork` wakes the machine first, so the copy has both."""
    sandbox = BoxdSandbox(client=client)
    async with _owned(client) as source:
        await source.run(['sh', '-c', f'echo unflushed > state.txt; {_TICKER}'], timeout=60)
        assert source.ref is not None
        await client.machines.pause(source.ref.id)
        with anyio.fail_after(60):
            while (await client.machines.get(source.ref.id)).status != 'suspended':
                await anyio.sleep(0.5)
        ref = await sandbox.fork(source.ref)
        try:
            copy = BoxdSandboxBackend(client=client, ref=ref)
            assert (
                await copy.run(['sh', '-c', f'cat state.txt; {_TICKING}'], timeout=60)
            ).stdout == 'unflushed\nticking\n'
        finally:
            await sandbox.destroy(ref)


async def test_timeout_raises_with_the_partial_output_and_stops_the_command(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that output printed before a client-side deadline reaches the
    `WorkspaceTimeoutError`, and that cancelling the exec call kills the command's process group."""
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; (sleep 5; touch {marker}) & sleep 20', shell=True, timeout=3)
        assert exc_info.value.stdout == 'DIAGNOSTIC\n'
        # Outlasts the background `sleep 5`, so a child left running would have written the marker.
        assert (await backend.run(['sleep', '6'], timeout=30)).exit_code == 0
        assert (await backend.run(['test', '-e', marker], timeout=30)).exit_code == 1


async def test_a_cancelled_run_stops_the_command(client: boxd.AsyncBoxd) -> None:
    """Validates the protocol's cancellation contract against real boxd: the command's pid is gone soon after."""
    pid_file = f'/tmp/{_unique("cancelled")}.pid'
    async with _owned(client) as backend:
        with anyio.move_on_after(5):
            await backend.run(f'echo $$ > {pid_file}; exec sleep 60', shell=True, timeout=60)
        pid = (await backend.run(['cat', pid_file], timeout=30)).stdout.strip()
        assert pid.isdigit()
        with anyio.fail_after(30):
            while (await backend.run(['kill', '-0', pid], timeout=30)).exit_code == 0:
                await anyio.sleep(0.2)


async def test_a_new_machine_gets_its_working_dir(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that `mkdir -p` through a unary exec makes `working_dir` on a
    new machine as the non-root `boxd` user, so the first command can start there."""
    directory = f'/home/boxd/{_unique("project")}/nested'
    async with _owned(client, working_dir=directory) as backend:
        result = await backend.run(['sh', '-c', 'pwd -P; stat -c %U .'], timeout=60)
        assert result.stdout.splitlines() == [directory, 'boxd']


async def test_a_detached_child_does_not_hold_the_result(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that boxd reports the exit status when the command exits,
    while a background child still holds its output open."""
    async with _owned(client) as backend:
        with anyio.fail_after(10):
            result = await backend.run('sleep 30 & echo ready', shell=True, timeout=60)
        assert (result.exit_code, result.stdout) == (0, 'ready\n')


async def test_a_deleted_machine_is_unavailable(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions about deletion: a command on a deleted machine fails, a lookup
    then finds nothing, and attaching again is reported as gone rather than retried or replaced."""
    async with _owned(client) as owner:
        assert owner.ref is not None
        await client.machines.delete(owner.ref.id)

        with pytest.raises(WorkspaceUnavailableError):
            await BoxdSandboxBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_a_stopped_machine_is_started_again_with_its_disk(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that a command on a stopped machine is refused with
    `ConflictError`, and that attaching starts it again and waits until it accepts commands."""
    path = f'/home/boxd/{_unique("kept")}.txt'
    async with _owned(client) as owner:
        await owner.run(['sh', '-c', 'printf kept > "$1"', 'sh', path], timeout=60)
        assert owner.ref is not None
        await client.machines.stop(owner.ref.id)
        await _wait_for_status(client, owner.ref.id, 'stopped')

        with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
            await owner.run(['true'], timeout=30)
        attached = BoxdSandboxBackend(client=client, ref=owner.ref)
        assert (await attached.run(['cat', path], timeout=60)).stdout == 'kept'


@pytest.mark.parametrize('sleep', ['pause', 'hibernate'])
async def test_a_sleeping_machine_is_woken_by_a_command(client: boxd.AsyncBoxd, sleep: str) -> None:
    """Validates the fake-encoded assumption that a command wakes a suspended or hibernated machine,
    with its running processes, so neither needs an explicit resume."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        machine_id = backend.ref.id
        await backend.run(['sh', '-c', 'nohup sleep 600 >/dev/null 2>&1 & echo $! > /tmp/sleeper'], timeout=60)
        if sleep == 'pause':
            await client.machines.pause(machine_id)
            await _wait_for_status(client, machine_id, 'suspended')
        else:
            await client.machines.hibernate(machine_id)
            await _wait_for_status(client, machine_id, 'hibernated')

        result = await BoxdSandboxBackend(client=client, ref=backend.ref).run(
            ['sh', '-c', 'kill -0 "$(cat /tmp/sleeper)" && echo alive'], timeout=60
        )
        assert result.stdout == 'alive\n'


async def test_a_machine_that_is_not_isolated_is_neither_used_nor_deleted(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a machine record reports `networking.isolated`, so a
    ref from tampered message history cannot point the agent at, or delete, one of the org's other machines.

    The machine's own view is the positive control for the isolation probe above: here the metadata
    endpoint hands out tokens and boxd's env vars and integrations are installed."""
    machine = await client.machines.create(
        _unique('pydantic-ai-e2e-open')[:40], isolated=False, auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT
    )
    try:
        await client.machines.wait_until_ready(machine.id)
        probe = await client.machines.exec(machine.id, _BOXD_SERVICES)
        assert str(probe.stdout).splitlines() == ['200', 'env-installed']
        ref = WorkspaceRef(provider='boxd', id=machine.id)
        with pytest.raises(WorkspaceUnavailableError, match='is not isolated'):
            await BoxdSandboxBackend(client=client, ref=ref).run(['true'], timeout=30)
        with pytest.raises(ValueError, match='is not isolated'):
            await BoxdSandbox(client=client).destroy(ref)
        with pytest.raises(WorkspaceUnavailableError, match='is not isolated'):
            await BoxdSandbox(client=client).fork(ref)
        assert (await client.machines.get(machine.id)).status != 'destroyed'
    finally:
        await client.machines.delete(machine.id)


async def test_a_snapshot_of_a_machine_on_a_network_is_refused(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a machine created from a snapshot keeps the named networks
    of the machine the snapshot was saved from, even when created isolated, and that the backend refuses it
    before running anything there."""
    snapshot = _unique('pydantic-ai-e2e-net')[:40]
    async with _owned(client) as source:
        assert source.ref is not None
        await client.machines.set_networks(source.ref.id, [snapshot])
        await client.snapshots.create(source.ref.id, snapshot)
        try:
            with anyio.fail_after(120):
                while (await client.snapshots.get(snapshot)).status != 'ready':
                    await anyio.sleep(0.5)
            restored = BoxdSandboxBackend(client=client, snapshot=snapshot)
            try:
                with pytest.raises(WorkspaceUnavailableError, match=f'is on the boxd networks {snapshot}'):
                    await restored.run(['true'], timeout=30)
            finally:
                assert restored.ref is not None
                await BoxdSandbox(client=client).destroy(restored.ref)
        finally:
            await client.snapshots.delete(snapshot)


async def test_a_machine_stopped_during_a_command_is_unavailable(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that stopping a machine ends a running command with
    `APIConnectionError`, after which a lookup reports the machine `stopped`."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        machine_id = backend.ref.id

        async def stop_soon() -> None:
            await anyio.sleep(3)
            await client.machines.stop(machine_id)

        async with anyio.create_task_group() as group:
            group.start_soon(stop_soon)
            with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
                await backend.run(['sleep', '60'], timeout=90)


async def test_coder_tools_run_in_the_machine(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a relative path the shell-derived file operations write
    is where a command started in the working directory finds it, so `Coder`'s tools agree."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from boxd')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )
        assert 'hi from boxd' in results[1]


async def test_commands_get_a_usable_environment(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a command sees the image's `PATH` and `HOME`, and what the
    docs say the default image has: the non-root `boxd` user in `/home/boxd`, git, Python, and ripgrep. A
    per-call `env` adds to the environment rather than replacing it."""
    async with _owned(client) as backend:
        result = await backend.run(
            'echo "$PATH"; echo "$HOME"; whoami; pwd -P; git --version; python3 --version; command -v rg',
            shell=True,
            timeout=60,
        )
        path, home, user, cwd, git, python, rg = result.stdout.splitlines()
        assert (home, user, cwd) == ('/home/boxd', 'boxd', '/home/boxd')
        assert rg.endswith('/rg')
        assert path and git.startswith('git version') and python.startswith('Python 3')
        assert await backend.working_dir() == '/home/boxd'

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/boxd-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real machine.

    The fake stands in for boxd, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's machine.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_machine'))
    assert all(run.used_sandbox for run in runs), runs
    # Follow-up runs work in the first run's machine, or in a copy of it the block forked.
    assert len({run.ref for run in runs}) <= 1 + example.source.count('.fork('), runs
