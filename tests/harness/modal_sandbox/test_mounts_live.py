"""`ModalSandbox(mounts=...)` against real Modal sandboxes, with MinIO running inside them.

Opt in with `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` and Modal credentials:
`PYDANTIC_AI_HARNESS_MODAL_LIVE=1 uv run pytest -m modal_live tests/harness/modal_sandbox/test_mounts_live.py`.
The scenarios run in one sandbox on Modal's VM runtime, each at a folder of its own, as the helper's install and
download happen once per sandbox; they are in `tests/harness/mounts/_scenarios_live.py`. A second sandbox, on gVisor,
checks that mounting there is refused with the fix.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

import modal
import pytest

from pydantic_ai.workspaces import CommandResult
from pydantic_ai_harness import MountError
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from ..mounts._scenarios_live import SCENARIOS, LiveSandbox, in_run, s3, start_store
from .conftest import LIVE_IDLE_TIMEOUT, LIVE_SANDBOX_TIMEOUT, skip_or_fail_live_tier

pytestmark = pytest.mark.modal_live


@asynccontextmanager
async def owned_backend(runtime: Literal['gvisor', 'vm']) -> AsyncGenerator[ModalSandboxBackend]:
    backend = ModalSandboxBackend(sandbox_timeout=LIVE_SANDBOX_TIMEOUT, idle_timeout=LIVE_IDLE_TIMEOUT, runtime=runtime)
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        try:
            await native.terminate.aio()
        finally:
            await native.detach.aio()  # pyright: ignore[reportUnknownMemberType]


async def _run_as_root(sandbox: modal.Sandbox, script: str, stdin: bytes) -> CommandResult:
    """Modal runs every command as root."""
    process = await sandbox.exec.aio('sh', '-c', script, workdir='/')
    process.stdin.write(stdin)
    process.stdin.write_eof()
    await process.stdin.drain.aio()
    stdout, stderr = await process.stdout.read.aio(), await process.stderr.read.aio()
    return CommandResult(exit_code=await process.wait.aio(), stdout=stdout, stderr=stderr)


def _live(backend: ModalSandboxBackend, sandbox: modal.Sandbox, working_dir: str) -> LiveSandbox:
    assert backend.ref is not None
    return LiveSandbox(
        ref=backend.ref,
        working_dir=working_dir,
        capability=lambda mounts: ModalSandbox(mounts=mounts, warn_if_no_tools=False),
        run_as_root=lambda script, stdin: _run_as_root(sandbox, script, stdin),
    )


@pytest.fixture(scope='module')
async def live() -> AsyncIterator[LiveSandbox]:
    # Module-scoped, so it is set up before the autouse fixture that skips the live tier.
    skip_or_fail_live_tier()
    async with owned_backend(runtime='vm') as backend:
        live = _live(backend, await backend.get_sandbox(), await backend.working_dir())
        await start_store(live)
        yield live


@pytest.mark.parametrize('scenario', [pytest.param(scenario, id=scenario.__name__) for scenario in SCENARIOS])
async def test_mounts(live: LiveSandbox, scenario: Callable[[LiveSandbox], Awaitable[None]]) -> None:
    await scenario(live)


async def test_a_gvisor_sandbox_is_refused_with_the_runtime_it_needs() -> None:
    """gVisor shows a `/dev/fuse` but refuses to mount it, so the check reads the kernel's name."""
    async with owned_backend(runtime='gvisor') as backend:
        live = _live(backend, await backend.get_sandbox(), await backend.working_dir())

        with pytest.raises(MountError) as raised:
            await in_run(live, [s3('refused')], lambda workspace: workspace.exists('refused'))

        assert str(raised.value) == (
            f'Modal sandbox {live.ref.id} runs on gVisor, which cannot mount the FUSE file systems mounts use. '
            "Create it on Modal's VM runtime: `ModalSandbox(runtime='vm')`."
        )
