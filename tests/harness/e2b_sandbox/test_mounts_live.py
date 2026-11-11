"""`E2BSandbox(mounts=...)` against a real E2B sandbox, with MinIO running inside it.

Opt in with `PYDANTIC_AI_HARNESS_E2B_LIVE=1` and an `E2B_API_KEY`:
`PYDANTIC_AI_HARNESS_E2B_LIVE=1 uv run pytest -m e2b_live tests/harness/e2b_sandbox/test_mounts_live.py`.
The sandbox needs internet access, which the helper uses to install fuse3 and download rclone. One sandbox serves
every scenario, each at a folder of its own, as the helper's install and download happen once per sandbox. The
scenarios are in `tests/harness/mounts/_scenarios_live.py`.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import e2b
import pytest

from pydantic_ai.workspaces import CommandResult
from pydantic_ai_harness import S3Mount
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from ..mounts._scenarios_live import SCENARIOS, SECRET, LiveSandbox, as_root, in_run, s3, start_store

pytestmark = pytest.mark.e2b_live


async def _run_as_root(sandbox: e2b.AsyncSandbox, script: str, stdin: bytes) -> CommandResult:
    handle = await sandbox.commands.run(script, background=True, stdin=True, user='root', cwd='/', timeout=0)
    try:
        # A script that reads nothing may have exited already, and its process with it.
        with contextlib.suppress(e2b.NotFoundException):
            if stdin:
                await sandbox.commands.send_stdin(handle.pid, stdin)
            await sandbox.commands.close_stdin(handle.pid)
        result = await handle.wait()
    except e2b.CommandExitException as exited:
        result = exited
    return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)


@pytest.fixture(scope='module')
async def live() -> AsyncIterator[LiveSandbox]:
    backend = E2BSandboxBackend()
    sandbox = await backend.get_sandbox()
    try:
        assert backend.ref is not None
        live = LiveSandbox(
            ref=backend.ref,
            working_dir=await backend.working_dir(),
            capability=lambda mounts: E2BSandbox(mounts=mounts),
            run_as_root=lambda script, stdin: _run_as_root(sandbox, script, stdin),
        )
        await start_store(live)
        yield live
    finally:
        await sandbox.kill()


@pytest.mark.parametrize('scenario', [pytest.param(scenario, id=scenario.__name__) for scenario in SCENARIOS])
async def test_mounts(live: LiveSandbox, scenario: Callable[[LiveSandbox], Awaitable[None]]) -> None:
    await scenario(live)


async def test_no_key_reaches_the_sandboxs_journal(live: LiveSandbox) -> None:
    """E2B's journal records `sudo`'s arguments; the helper runs through `user='root'` and gets its keys on stdin."""
    mounts: Sequence[S3Mount] = [s3('journal')]
    await in_run(live, mounts, lambda workspace: workspace.exists('journal'))

    journal = await as_root(
        live,
        'journalctl --no-pager -o cat 2> /dev/null'
        " || cat /var/log/journal/*/*.journal /run/log/journal/*/*.journal 2> /dev/null | tr -c '[:print:]' '\\n'",
    )

    assert journal.strip(), 'the sandbox has no journal to search'
    assert SECRET not in journal
