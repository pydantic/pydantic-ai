"""Integration tests that require a real, running E2B sandbox.

The fake-backed suites already cover the harness-owned logic: deadline arithmetic, protocol
translation, path resolution math, and exception mapping. This live tier admits only
regressions a correctly written fake could not catch: real process execution, a client-owned
deadline killing a real process, one filesystem shared by E2B's file API and the shell,
create-time environment propagation, and real lifecycle state in E2B's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real E2B behavior.

Gating:
  * `e2b_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_E2B_LIVE=1` opts in explicitly.
  * also requires a non-empty `E2B_API_KEY`; without one it skips, or fails when
    `E2B_REQUIRE_LIVE=1` (CI's live job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * a module-scoped `anyio_backend` fixture keeps the shared E2B handle on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_E2B_LIVE=1 uv run pytest -m e2b_live tests/e2b_sandbox/test_e2b_live.py`
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import e2b
import pytest
from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError, WorkspaceUnavailableError
from pytest_examples import CodeExample

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import (
    E2BSandboxBackend,
)

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools

pytestmark = pytest.mark.e2b_live


def _unique(prefix: str) -> str:
    """Return a collision-resistant path or name segment for a shared live sandbox."""
    return f'{prefix}-{uuid.uuid4().hex}'


@asynccontextmanager
async def _owned(**settings: object) -> AsyncGenerator[E2BSandboxBackend]:
    """Create a workspace and kill its native handle on the way out."""
    backend = E2BSandboxBackend(**settings)  # type: ignore[arg-type]
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        await native.kill()
        assert not await native.is_running()


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def sandbox() -> AsyncIterator[E2BSandboxBackend]:
    """One live owned sandbox shared by command and filesystem tests.

    Each test writes under `_unique(...)` paths, so the shared microVM avoids repeated cold
    starts without coupling test state. Lifecycle tests create their own sandboxes because
    ownership, expiry, attach, and teardown are the behavior under test there.
    """
    async with _owned(sandbox_timeout=600) as live:
        yield live


async def test_destroy_by_ref_without_connecting() -> None:
    async with _owned() as backend:
        assert backend.ref is not None
        from pydantic_ai_harness.e2b_sandbox import E2BSandbox  # noqa: PLC0415 - live SDK

        await E2BSandbox().destroy(backend.ref)
        assert not await (await backend.get_sandbox()).is_running()


class TestRealExecution:
    """Behaviors that only exist because a real process runs in a real microVM."""

    async def test_timeout_kills_the_command_and_keeps_its_output(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that a client-owned deadline kills a real process.

        E2B's own command `timeout` abandons the output stream and leaves the command running,
        so the backend enforces the deadline itself and calls the per-command kill. The marker
        the command would have written after the deadline must never appear.
        """
        marker = f'/tmp/{_unique("after-deadline")}'
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await sandbox.run(f'echo DIAGNOSTIC; sleep 20; touch {marker}', shell=True, timeout=2)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        await anyio.sleep(25)
        assert await sandbox.exists(marker) is False

    @pytest.mark.xfail(reason='envd waits for inherited output pipes to close', strict=True)
    async def test_background_child_does_not_delay_main_process_exit(self, sandbox: E2BSandboxBackend) -> None:
        """The fake does not model a background child holding the SDK output stream open."""
        with anyio.fail_after(4):
            result = await sandbox.run('sleep 7 & echo ready', shell=True, timeout=10)
        assert result.stdout.startswith('ready')

    async def test_timeout_stops_foreground_descendants(self, sandbox: E2BSandboxBackend) -> None:
        """Check whether a timed-out foreground shell leaves a child able to mutate the sandbox."""
        marker = f'/tmp/{_unique("descendant")}'
        with pytest.raises(WorkspaceTimeoutError):
            await sandbox.run(f'(sleep 3; touch {marker}) & sleep 30', shell=True, timeout=1)
        await anyio.sleep(5)
        assert not await sandbox.exists(marker)

    async def test_cancel_before_remote_start_fences_user_command(
        self, sandbox: E2BSandboxBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        native = await sandbox.get_sandbox()
        original = native.commands.run
        entered = anyio.Event()
        release = anyio.Event()
        launches: list[str] = []

        async def delayed(cmd: str, **kwargs: object) -> object:
            if cmd.startswith('setsid sh -c '):
                launches.append(cmd)
                entered.set()
                await release.wait()
            return await original(cmd, **kwargs)  # type: ignore[arg-type]

        marker = f'/tmp/{_unique("late-start")}'
        monkeypatch.setattr(native.commands, 'run', delayed)
        task = asyncio.create_task(sandbox.run(f'touch {marker}', shell=True))
        try:
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            monkeypatch.setattr(native.commands, 'run', original)
            late = await original(launches[0], background=True, timeout=0)
            with pytest.raises(e2b.CommandExitException):
                await late.wait()
            assert not await sandbox.exists(marker)
        finally:
            release.set()
            monkeypatch.setattr(native.commands, 'run', original)

    async def test_a_foreground_group_child_is_stopped(self, sandbox: E2BSandboxBackend) -> None:
        """The deadline stops children that remain in the foreground process group."""
        marker = f'/tmp/{_unique("orphan")}'
        with pytest.raises(WorkspaceTimeoutError):
            await sandbox.run(f'(sleep 5; touch {marker}) & sleep 30', shell=True, timeout=2)

        await anyio.sleep(10)
        assert await sandbox.exists(marker) is False

    async def test_a_cancelled_run_stops_the_command(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the protocol's cancellation contract against real E2B.

        A cancelled `run()` must not knowingly leave the command running; E2B has a
        per-command kill, so the marker written after the cancellation must never appear.
        """
        marker = f'/tmp/{_unique("cancelled")}'
        with anyio.move_on_after(2):
            await sandbox.run(f'sleep 15; touch {marker}', shell=True, timeout=60)

        await anyio.sleep(20)
        assert await sandbox.exists(marker) is False

    async def test_large_stderr_does_not_block_stdout(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B buffers both streams without deadlock."""
        result = await sandbox.run('seq 1 200000 1>&2; echo done', shell=True, timeout=120)

        assert result.exit_code == 0
        assert result.stdout == 'done\n'
        stderr_lines = result.stderr.splitlines()
        assert (stderr_lines[0], stderr_lines[-1], len(stderr_lines)) == ('1', '200000', 200000)

    async def test_concurrent_commands_share_one_sandbox(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that one E2B sandbox multiplexes concurrent commands."""
        results: dict[int, str] = {}

        async def run(n: int) -> None:
            out = await sandbox.run(f'echo job-{n}', shell=True, timeout=60)
            results[n] = out.stdout.strip()

        async with anyio.create_task_group() as tg:
            for n in range(8):
                tg.start_soon(run, n)

        assert results == {n: f'job-{n}' for n in range(8)}

    async def test_signal_exit_is_a_real_exit(self, sandbox: E2BSandboxBackend) -> None:
        """Validates that a signalled death is a plain non-zero exit, not a timeout. E2B reports it as `-1`."""
        result = await sandbox.run('kill -KILL $$', shell=True, timeout=30)

        assert result.exit_code != 0


class TestCreateConfiguration:
    """Create-time configuration reaching the real process, not only E2B create arguments."""

    async def test_env_reaches_commands(self) -> None:
        """Validates the fake-encoded assumption that create-time `env` reaches commands."""
        probe = _unique('live-value')
        async with _owned(sandbox_timeout=120, env={'HARNESS_ENV_PROBE': probe}) as backend:
            result = await backend.run('printf %s "$HARNESS_ENV_PROBE"', shell=True, timeout=30)

        assert result.stdout == probe

    async def test_working_dir_is_discovered_when_not_configured(self) -> None:
        """Validates the fake-encoded assumption that `pwd` answers for a sandbox with no configured working_dir."""
        async with _owned(sandbox_timeout=120) as backend:
            working_dir = await backend.working_dir()
            printed = await backend.run(['pwd'], timeout=30)

        assert working_dir == printed.stdout.strip()

    async def test_a_configured_working_dir_applies_per_command(self) -> None:
        """Validates the fake-encoded assumption that E2B takes a working directory per command.

        E2B has no create-time working directory, so the backend supplies it on every command instead.
        """
        async with _owned(sandbox_timeout=120, working_dir='/tmp') as backend:
            result = await backend.run(['pwd'], timeout=30)

        assert result.stdout.strip() == '/tmp'

    async def test_per_command_cwd_and_env_reach_the_process(self) -> None:
        """Validates the fake-encoded assumption that E2B applies per-command `cwd` and `envs`."""
        probe = _unique('per-command')
        async with _owned(sandbox_timeout=120) as backend:
            result = await backend.run(
                'printf "%s %s" "$(pwd)" "$PROBE"', shell=True, cwd='/etc', env={'PROBE': probe}, timeout=30
            )

        assert result.stdout == f'/etc {probe}'


class TestRealFilesystem:
    """One real filesystem shared by E2B's file API and the shell."""

    async def test_large_filesystem_transfer_near_read_limit(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B's file API handles a near-limit transfer."""
        path = f'/tmp/{_unique("big")}.bin'
        payload = b'A' * (4 * 1024 * 1024)

        await sandbox.write_bytes(path, payload)

        assert (await sandbox.stat(path)).size == len(payload)
        assert await sandbox.read_bytes(path) == payload

    async def test_list_dir_reports_basenames_and_dir_flags(self, sandbox: E2BSandboxBackend) -> None:
        """Validates the fake-encoded assumption that E2B lists entries by basename with a real dir flag."""
        root = f'/tmp/{_unique("ls")}'
        await sandbox.write_bytes(f'{root}/file.txt', b'x')
        await sandbox.write_bytes(f'{root}/sub/nested.txt', b'y')

        entries = await sandbox.list_dir(root)

        assert sorted((entry.name, entry.is_dir, entry.path) for entry in entries) == [
            ('file.txt', False, f'{root}/file.txt'),
            ('sub', True, f'{root}/sub'),
        ]

    async def test_relative_paths_resolve_against_the_working_directory(self) -> None:
        """Validates the fake-encoded assumption that the facade's resolution matches the process cwd."""
        filename = f'{_unique("rel")}.txt'
        async with _owned(sandbox_timeout=120, working_dir='/tmp') as backend:
            facade = Workspace(backend)
            await facade.write_text(filename, 'from-relative-path\n')
            result = await backend.run(['cat', filename], timeout=30)

        assert result.stdout == 'from-relative-path\n'


class TestRealLifecycle:
    """Teardown and attach semantics in E2B's real control plane."""

    async def test_connect_reuses_state_and_leaves_the_sandbox_running(self) -> None:
        """Validates the fake-encoded assumption that connecting reuses state and does not take ownership."""
        marker = f'/tmp/{_unique("persist")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'shared')

            attached = E2BSandboxBackend(ref=owner.ref)
            assert (await attached.get_sandbox()).sandbox_id == (await owner.get_sandbox()).sandbox_id
            assert await attached.read_bytes(marker) == b'shared'

            assert (await owner.run(['cat', marker], timeout=30)).stdout == 'shared'

    async def test_a_killed_sandbox_is_unavailable_at_once(self) -> None:
        """Validates the fake-encoded assumption that a killed sandbox is gone as soon as `kill()` returns.

        The fake 404s a later connect and fails envd calls on a held handle with the SDK's 502
        `TimeoutException`, both right after the kill. If E2B tears the sandbox down eventually
        instead, an operation in that window succeeds and this test fails.
        """
        path = f'/tmp/{_unique("killed")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(path, b'before-kill')
            assert owner.ref is not None
            assert await (await owner.get_sandbox()).kill() is True

            with pytest.raises(WorkspaceUnavailableError):
                await E2BSandboxBackend(ref=owner.ref).working_dir()
            with pytest.raises(WorkspaceUnavailableError):
                await owner.run(['true'], timeout=30)
            with pytest.raises(WorkspaceUnavailableError):
                await owner.read_bytes(path)

    async def test_connect_resumes_a_paused_sandbox(self) -> None:
        """Pins the documented behavior that attaching to a paused sandbox restarts it.

        This is the E2B-specific half of attach mode: a paused sandbox is not gone, and the
        backend's first `get_sandbox()` brings it back rather than failing.
        """
        marker = f'/tmp/{_unique("paused")}.txt'
        async with _owned(sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'before-pause')
            await (await owner.get_sandbox()).beta_pause()

            attached = E2BSandboxBackend(ref=owner.ref)
            assert await attached.read_bytes(marker) == b'before-pause'

    async def test_the_default_lifetime_is_accepted(self) -> None:
        """Validates the fake-encoded assumption that E2B accepts the default `sandbox_timeout`.

        The fake takes any lifetime; E2B refuses one over the plan's limit, and the Hobby plan's is 3600.
        """
        async with _owned() as backend:
            assert (await backend.run(['true'], timeout=30)).exit_code == 0

    async def test_the_documented_cleanup_kills_a_paused_sandbox(self) -> None:
        """Validates that the docs page's `kill_sandbox` kills a paused sandbox, which the fake cannot show.

        Connecting to a paused sandbox would resume it, so the cleanup must kill it by id instead.
        """
        kill_sandbox = documented_cleanup(_DOCS_BLOCKS, 'kill_sandbox')
        async with _owned(sandbox_timeout=120) as owner:
            assert owner.ref is not None
            await (await owner.get_sandbox()).beta_pause()
            await kill_sandbox(owner.ref)

            with pytest.raises(WorkspaceUnavailableError):
                await E2BSandboxBackend(ref=owner.ref).get_sandbox()


class TestCoder:
    """The model's tools running in a real sandbox."""

    async def test_coder_shell_and_file_tools_share_the_sandbox(self) -> None:
        """Validates the fake-encoded assumption that Coder's shell machinery runs in E2B's template.

        The shell tool launches jobs through `sh`, `setsid`, and `base64`, and the file tools
        resolve paths with `readlink -f`; the fake runs them on the host instead.
        """
        async with _owned(sandbox_timeout=120) as backend:
            shell_output, read_output = await call_tools(
                [Coder()],
                [('shell', {'command': 'echo made-in-sandbox > note.txt'}), ('read_file', {'path': 'note.txt'})],
                workspace=backend,
            )

        assert '"exit_code": 0' in shell_output
        assert 'made-in-sandbox' in read_output


class TestEnvironment:
    """What a command sees without being given anything."""

    async def test_commands_get_a_usable_environment(self) -> None:
        """Validates the fake-encoded assumption that a command sees the template's `PATH` and `HOME`, and
        git in the default template, and that a per-call `env` adds to them rather than replacing them.
        """
        async with _owned(sandbox_timeout=120) as backend:
            result = await backend.run('echo "$PATH"; echo "$HOME"; git --version', shell=True, timeout=60)
            path, home, git = result.stdout.splitlines()
            assert path and home and git.startswith('git version')

            result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
            assert result.stdout.splitlines() == ['1', path]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/e2b-sandbox.md')


class TestDocsExamples:
    """The docs page, run as written."""

    @pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
    def test_docs_example(self, example: CodeExample) -> None:
        """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

        The fake stands in for E2B, so only this shows the page's code, its default settings, and its
        cleanup work against the real service. A follow-up run, from the message history or a stored ref,
        works in the first run's sandbox.
        """
        _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'kill_sandbox'))
        assert all(run.used_sandbox for run in runs), runs
        assert len({run.ref for run in runs}) <= 1, runs
