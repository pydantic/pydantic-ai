"""Integration tests that require a real Daytona sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution, a client-owned deadline over a real process, one filesystem shared by
Daytona's file API and its process sessions, and real lifecycle state in Daytona's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Daytona behavior.

Gating:
  * `daytona_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1` opts in explicitly.
  * also requires a non-empty `DAYTONA_API_KEY`; without one it skips, or fails when
    `DAYTONA_REQUIRE_LIVE=1` (CI's live job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * the module-scoped async `client` fixture keeps the shared Daytona client on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_DAYTONA_LIVE=1 uv run pytest -m daytona_live tests/harness/daytona_sandbox/test_daytona_live.py`
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import daytona
import pytest

from pydantic_ai.workspaces import WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .._tool_calls import call_tools
from .conftest import LIVE_AUTO_STOP_INTERVAL

pytestmark = pytest.mark.daytona_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[daytona.AsyncDaytona]:
    """One caller-owned API client for every backend in this module, closed at the end."""
    async with daytona.AsyncDaytona() as client:
        yield client
    # aiohttp closes SSL transports on the next loop ticks after ClientSession.close().
    # Let them finish before pytest tears down this module's event loop.
    await asyncio.sleep(0.25)


@asynccontextmanager
async def _owned(client: daytona.AsyncDaytona) -> AsyncGenerator[DaytonaSandboxBackend]:
    """Create a sandbox and delete it on the way out, even when the test deleted it already."""
    backend = DaytonaSandboxBackend(client=client, auto_stop_interval=LIVE_AUTO_STOP_INTERVAL)
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        try:
            await native.refresh_data()
            if native.state not in (daytona.SandboxState.DESTROYING, daytona.SandboxState.DESTROYED):
                await native.delete()
        except daytona.DaytonaNotFoundError:
            pass
        # Daytona accepts deletion asynchronously; confirm this test's ID has disappeared.
        for _ in range(60):
            try:
                await client.get(native.id)
            except daytona.DaytonaNotFoundError:
                break
            await asyncio.sleep(1)
        else:
            raise AssertionError(f'Daytona sandbox {native.id} was not deleted')


async def test_destroy_by_ref_does_not_start_sandbox(client: daytona.AsyncDaytona) -> None:
    """Validate that SDK get + delete does not resume a stopped sandbox."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        await DaytonaSandbox(client=client).destroy(backend.ref)
        assert (await client.get(backend.ref.id)).state in (
            daytona.SandboxState.DESTROYING,
            daytona.SandboxState.DESTROYED,
        )


async def test_creates_a_fresh_sandbox_and_runs_a_command(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a session command reports real output and exit code."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)

        assert backend.ref is not None
        assert (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ('out', 'err', 3)


async def test_large_output_is_exact(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a finished command's stored logs are its exact output.

    The SDK's log stream corrupts output above a few KB (it misreads a stream prefix split across
    websocket frames), so the result must not be built from it.
    """
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '50000'], timeout=120)

        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 50001))


async def test_reattach_by_ref_reads_a_file_the_first_backend_wrote(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that `client.get` plus `start` reaches the same sandbox."""
    path = f'/tmp/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'shared')
        assert owner.ref is not None

        attached = DaytonaSandboxBackend(client=client, ref=owner.ref)

        assert await attached.read_bytes(path) == b'shared'
        assert attached.ref == owner.ref


async def test_command_and_file_api_share_one_filesystem(client: daytona.AsyncDaytona) -> None:
    """Validates the protocol's one-environment contract: process sessions and `sandbox.fs` see one tree."""
    root = f'/tmp/{_unique("roundtrip")}'
    async with _owned(client) as backend:
        await backend.write_bytes(f'{root}/nested/in.txt', b'from-file-api\n')
        result = await backend.run(
            ['sh', '-c', 'cat "$1/nested/in.txt" && printf from-shell > "$1/out.txt"', 'sh', root], timeout=60
        )

        assert (result.exit_code, result.stdout) == (0, 'from-file-api\n')
        assert await backend.read_bytes(f'{root}/out.txt') == b'from-shell'


async def test_timeout_raises_with_the_partial_output(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that session logs stream before the command ends.

    The deadline is enforced client-side, so the output printed before it expired must reach
    the `WorkspaceTimeoutError`, and deleting the session must stop the command.
    """
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=5)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        assert '5' in str(exc_info.value)
        assert (await backend.run(['sleep', '20'], timeout=60)).exit_code == 0
        assert await backend.exists(marker) is False


async def test_timeout_stops_only_its_command(client: daytona.AsyncDaytona) -> None:
    """Checks against the real SDK that deleting a session stops its process, not the shared sandbox."""
    async with _owned(client) as backend:
        marker = f'/tmp/{_unique("daytona-stopped")}'
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(f'sleep 5; touch {marker}', shell=True, timeout=1)
        assert (await backend.run(['true'], timeout=30)).exit_code == 0
        assert not await backend.exists(marker)


async def test_detached_child_does_not_hold_command_result(client: daytona.AsyncDaytona) -> None:
    """Check whether a real follow websocket remains open with inherited stdout descriptors."""
    async with _owned(client) as backend:
        try:
            result = await backend.run('sleep 4 & echo ready', shell=True, timeout=3)
        except WorkspaceTimeoutError:
            pytest.xfail('Daytona does not report command completion before inherited output closes')
        assert result.exit_code == 0
        assert result.stdout.strip() == 'ready'


async def test_reattach_to_a_deleted_sandbox_is_unavailable(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption about deletion: `delete()` returns while the sandbox is
    still `destroying`, and a later attach or a call on the held handle is reported as gone.

    If Daytona answers either differently, a deleted sandbox would surface as a retryable error
    or, worse, as a missing file.
    """
    path = f'/tmp/{_unique("deleted")}.txt'
    async with _owned(client) as owner:
        await owner.write_bytes(path, b'before-delete')
        assert owner.ref is not None
        await (await owner.get_sandbox()).delete()

        with pytest.raises(WorkspaceUnavailableError):
            await DaytonaSandboxBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.read_bytes(path)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_coder_tools_run_in_the_sandbox(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a relative path the file API writes is where a
    session command started in the working directory finds it, so `Coder`'s tools agree."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from daytona')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )

        assert 'hi from daytona' in results[1]


async def test_commands_get_a_usable_environment(client: daytona.AsyncDaytona) -> None:
    """Validates the fake-encoded assumption that a command sees the snapshot's `PATH` and `HOME`, and git
    in the default snapshot, and that a per-call `env` adds to them rather than replacing them."""
    async with _owned(client) as backend:
        result = await backend.run('echo "$PATH"; echo "$HOME"; git --version', shell=True, timeout=60)
        path, home, git = result.stdout.splitlines()
        assert path and home and git.startswith('git version')

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]
