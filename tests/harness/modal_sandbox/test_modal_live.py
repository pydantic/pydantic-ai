"""Integration tests that require a real, running Modal container.

The fake-backed suites already cover the harness-owned logic: timeout quantization,
the output ring buffer, path resolution math, and exception mapping. This live tier
admits only regressions a correctly written fake could not catch: real process
execution, infra-enforced deadlines, one filesystem shared by Modal's file API and
the shell, create-time environment and workdir propagation, and real lifecycle
state in Modal's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption
  it validates against real Modal behavior.

Portability:
  These assert durable sandbox behaviors, not Modal-specific spellings, so if the Modal
  mechanism is later swapped for a different backend the suite retargets rather than gets
  rewritten:
  * TestRealExecution: the exec call and its result fields rename; a timeout is checked via the
    backend's raised timeout error carrying the pre-kill output, not the `-1` sentinel.
  * TestCreateConfiguration: create-time `env` and `workdir` assertions move to the backend's
    sandbox-configuration surface.
  * TestRealFilesystem: byte file operations move to the backend's file API; the relative-path
    case becomes explicit resolution against the working directory.
  * TestRealLifecycle: dead-sandbox checks become the backend's typed not-found error, and
    attach/reuse maps to its attach surface.

Gating:
  * `modal_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` opts in explicitly.
  * also requires `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, or `~/.modal.toml`.
  * a module-scoped `anyio_backend` fixture keeps the shared Modal handle on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_MODAL_LIVE=1 uv run pytest -m modal_live tests/modal_sandbox/test_modal_live.py`
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.modal_sandbox import (
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)


def _has_modal_credentials() -> bool:
    has_env_token = os.getenv('MODAL_TOKEN_ID') is not None and os.getenv('MODAL_TOKEN_SECRET') is not None
    return has_env_token or Path('~/.modal.toml').expanduser().exists()


_live_enabled = os.getenv('PYDANTIC_AI_HARNESS_MODAL_LIVE') == '1'

pytestmark = [
    pytest.mark.modal_live,
    pytest.mark.skipif(
        not _live_enabled or not _has_modal_credentials(),
        reason=(
            'requires PYDANTIC_AI_HARNESS_MODAL_LIVE=1 and either MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or ~/.modal.toml'
        ),
    ),
]

# A small, common image keeps cold starts cheap; these tests need only a POSIX shell and coreutils.
_IMAGE = 'python:3.12-slim'


def _unique(prefix: str) -> str:
    """Return a collision-resistant path or name segment for a shared live sandbox."""
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module')
async def session() -> AsyncIterator[ModalSandboxSession]:
    """One live owned sandbox shared by exec and filesystem tests.

    Each test writes under `_unique(...)` paths, so the shared container avoids repeated cold starts
    without coupling test state. Lifecycle tests create their own sandboxes because ownership,
    expiry, attach, and termination are the behavior under test there.
    """
    async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=600) as live:
        yield live


class TestRealExecution:
    """Behaviors that only exist because a real process runs on real Modal infra."""

    async def test_runs_a_real_process(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that exec stdout, stderr, and exit code match a process."""
        result = await session.exec(['sh', '-c', 'echo out; echo err 1>&2; exit 3'], timeout=30)

        assert result.stdout.strip() == 'out'
        assert result.stderr.strip() == 'err'
        assert result.returncode == 3
        assert result.timed_out is False

    async def test_timeout_preserves_pre_deadline_output(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that output printed before an infra timeout is preserved."""
        result = await session.exec(['sh', '-c', 'echo DIAGNOSTIC; sleep 30'], timeout=2)

        # Durable assertion: a backend that raises a typed timeout error would expose this as its retained stdout.
        assert 'DIAGNOSTIC' in result.stdout
        assert result.timed_out is True

    async def test_timeout_preserves_stderr(self, session: ModalSandboxSession) -> None:
        result = await session.exec(['sh', '-c', 'echo STDERR-DIAGNOSTIC 1>&2; sleep 30'], timeout=2)

        assert 'STDERR-DIAGNOSTIC' in result.stderr
        # Modal reports this deadline kill either as its client-side -1 sentinel or, when
        # the server's SIGKILL wins the race, as a plain 137 exit. The session recognizes
        # both (137 counts once the command consumed its whole deadline window).
        assert result.returncode != 0
        assert result.timed_out is True

    async def test_large_stderr_does_not_block_stdout(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal buffers streams without stderr deadlock."""
        result = await session.exec(['sh', '-c', 'seq 1 300000 1>&2; echo done'], timeout=60)

        assert result.returncode == 0
        assert result.stdout == 'done\n'
        stderr_lines = result.stderr.splitlines()
        assert stderr_lines[0] == '1'
        assert stderr_lines[-1] == '300000'
        assert len(stderr_lines) == 300000

    async def test_concurrent_commands_share_one_container(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that one Modal sandbox can multiplex concurrent execs."""
        results: dict[int, str] = {}

        async def run(n: int) -> None:
            out = await session.exec(['sh', '-c', f'echo job-{n}'], timeout=15)
            results[n] = out.stdout.strip()

        async with anyio.create_task_group() as tg:
            for n in range(8):
                tg.start_soon(run, n)

        assert results == {n: f'job-{n}' for n in range(8)}

    async def test_signal_exit_is_not_timeout(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that signal death is a real exit, not Modal's timeout sentinel."""
        result = await session.exec(['sh', '-c', 'kill -KILL $$'], timeout=15)

        assert result.returncode > 128
        assert result.returncode != -1
        assert result.timed_out is False

    async def test_nonexistent_binary_returns_modal_exit_code(self, session: ModalSandboxSession) -> None:
        """Pins Modal's current return code for an executable lookup failure."""
        binary = _unique('definitely-not-a-real-binary')
        result = await session.exec([binary], timeout=15)
        assert result.returncode == 128


class TestCreateConfiguration:
    """Create-time configuration reaching the real process, not only Modal create kwargs."""

    async def test_env_and_workdir_reach_processes(self) -> None:
        """Validates the fake-encoded assumption that create-time `env` and `workdir` reach commands."""
        probe = _unique('live-value')
        async with ModalSandboxSession(
            image=_IMAGE,
            sandbox_timeout=120,
            workdir='/tmp',
            env={'HARNESS_ENV_PROBE': probe},
        ) as session:
            env_result = await session.exec(['sh', '-c', 'printf %s "$HARNESS_ENV_PROBE"'], timeout=15)
            pwd_result = await session.exec(['pwd'], timeout=15)

        assert env_result.stdout == probe
        assert pwd_result.stdout.strip() == '/tmp'


class TestRealFilesystem:
    """One real filesystem shared by Modal's file API and the shell."""

    async def test_shell_and_file_api_see_the_same_filesystem(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that exec and filesystem APIs share one backing fs."""
        api_path = f'/tmp/{_unique("api")}.txt'
        await session.write_bytes(api_path, b'from-file-api\n')
        via_shell = await session.exec(['cat', api_path], timeout=15)
        assert via_shell.stdout == 'from-file-api\n'

        shell_path = f'/tmp/{_unique("shell")}.txt'
        wrote = await session.exec(['sh', '-c', f'printf from-shell > {shell_path}'], timeout=15)
        assert wrote.returncode == 0
        assert await session.read_bytes(shell_path) == b'from-shell'

    async def test_binary_roundtrip_creating_parent_dirs(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal stores raw bytes and creates real parent dirs."""
        path = f'/tmp/{_unique("io")}/nested/deep/data.bin'
        payload = b'\x00\x01hello \xf0\x9f\x9a\x80 world'

        await session.write_bytes(path, payload)

        assert await session.read_bytes(path) == payload

    async def test_large_filesystem_transfer_near_read_limit(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal's beta fs API handles a near-limit transfer."""
        path = f'/tmp/{_unique("big")}.bin'
        payload = b'A' * (4 * 1024 * 1024)

        await session.write_bytes(path, payload)

        assert await session.file_size(path) == len(payload)
        assert await session.read_bytes(path) == payload

    async def test_missing_file_raises_recoverable_error(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that a missing file is recoverable, not a dead sandbox."""
        with pytest.raises(ModalSandboxError) as exc_info:
            await session.read_bytes(f'/tmp/{_unique("missing")}')

        assert not isinstance(exc_info.value, ModalSandboxTerminalError)

    async def test_list_files_reports_basenames_and_dir_flags(self, session: ModalSandboxSession) -> None:
        """Validates the fake-encoded assumption that Modal lists entries by basename with a real dir flag."""
        root = f'/tmp/{_unique("ls")}'
        await session.write_bytes(f'{root}/file.txt', b'x')
        await session.write_bytes(f'{root}/sub/nested.txt', b'y')

        entries = await session.list_files(root)

        assert sorted(entries) == [('file.txt', False), ('sub', True)]

    async def test_workdir_and_relative_file_resolution_share_one_view(self) -> None:
        """Validates the fake-encoded assumption that relative file API paths share the process cwd view.

        Portability: on a backend whose filesystem seam rejects relative paths, callers resolve
        explicitly against the working directory, and this test then covers that resolution instead.
        """
        filename = f'{_unique("rel")}.txt'
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120, workdir='/tmp') as session:
            await session.write_bytes(filename, b'from-relative-path\n')
            result = await session.exec(['cat', filename], timeout=15)

        assert result.stdout == 'from-relative-path\n'


class TestRealLifecycle:
    """Teardown and attach semantics in Modal's real control plane."""

    async def test_terminate_actually_destroys_the_container(self) -> None:
        """Validates the fake-encoded assumption that exiting an owned session destroys the real sandbox."""
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120) as owner:
            sandbox_id = owner.sandbox_id
            assert sandbox_id is not None

        became_unavailable = False
        attempts = 8
        for attempt in range(attempts):
            try:
                async with ModalSandboxSession(sandbox_id=sandbox_id):
                    pass  # pragma: no cover
            except ModalSandboxUnavailableError:
                became_unavailable = True
                break
            # Modal can lag about 30s before reporting external termination to a fresh attach.
            if attempt < attempts - 1:
                await anyio.sleep(5)

        assert became_unavailable

    async def test_attach_reuses_state_and_leaves_container_running(self) -> None:
        """Validates the fake-encoded assumption that attach reuses state and does not terminate ownership."""
        marker = f'/tmp/{_unique("persist")}.txt'
        async with ModalSandboxSession(image=_IMAGE, sandbox_timeout=120) as owner:
            await owner.write_bytes(marker, b'shared')
            sandbox_id = owner.sandbox_id
            assert sandbox_id is not None

            async with ModalSandboxSession(sandbox_id=sandbox_id) as attached:
                assert attached.sandbox_id == sandbox_id
                assert await attached.read_bytes(marker) == b'shared'

            assert (await owner.exec(['cat', marker], timeout=15)).stdout == 'shared'
