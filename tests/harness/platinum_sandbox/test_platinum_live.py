"""Integration tests that require a real Platinum sandbox.

The fake-backed suites cover the harness-owned logic: request shapes, deadline handling, and error mapping. This
live tier admits only what a correctly written fake could not catch: real process execution and its limits, a
sandbox's lifecycle in Platinum's control plane, the filesystem API's request limits, and the docs page's examples
against the real service.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it validates against real
  Platinum behavior.

Gating:
  * `platinum_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_PLATINUM_LIVE=1` opts in explicitly.
  * also requires a non-empty `PT_TOKEN`; without one it skips, or fails when `PLATINUM_REQUIRE_LIVE=1` (CI's live
    job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * every sandbox a test creates is deleted by the test, and by the session sweep in `conftest.py` if that fails.

Run locally:
`PYDANTIC_AI_HARNESS_PLATINUM_LIVE=1 uv run pytest -m platinum_live tests/harness/platinum_sandbox/`
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import anyio
import platinum
import pytest
from pytest_examples import CodeExample

from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.platinum_sandbox import PlatinumSandbox, PlatinumSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .conftest import LIVE_AUTO_STOP_MINUTES, delete_live_sandbox

pytestmark = [pytest.mark.platinum_live, pytest.mark.anyio(backends=['asyncio'])]

_MODULE = 'pydantic_ai_harness.platinum_sandbox._backend'


@asynccontextmanager
async def _owned(
    *, env: dict[str, str] | None = None, allow_internet_access: bool = True
) -> AsyncGenerator[PlatinumSandboxBackend]:
    """Create a sandbox, and delete it on the way out."""
    backend = PlatinumSandboxBackend(
        auto_stop_minutes=LIVE_AUTO_STOP_MINUTES, env=env, allow_internet_access=allow_internet_access
    )
    try:
        await backend.get_sandbox()
        yield backend
    finally:
        try:
            if backend.ref is not None:
                await delete_live_sandbox(backend.ref)
        finally:
            await backend.aclose()


async def test_a_command_reports_its_output_and_exit_code() -> None:
    """Validates the fake-encoded exec reply: output and exit code in `result`, and the launch script's working dir."""
    async with _owned() as backend:
        result = await backend.run('echo out; echo err >&2; pwd -P; id -un; exit 3', shell=True, timeout=30)
        assert (result.exit_code, result.stdout, result.stderr) == (3, 'out\n/workspace\nroot\n', 'err\n')


async def test_a_long_timeout_runs_as_a_managed_process() -> None:
    """Validates the fake-encoded managed process: a start, an output stream in base64, and an `exit` event."""
    async with _owned(env={'GREETING': 'hello'}) as backend:
        result = await backend.run(['sh', '-c', 'printf "$GREETING $1"; exit 4', 'sh', 'world'], timeout=600)
        assert (result.exit_code, result.stdout) == (4, 'hello world')
        result = await backend.run(['sh', '-c', 'kill -TERM $$'])
        assert result.exit_code == 143


@pytest.mark.parametrize('managed', [False, True])
async def test_a_timeout_stops_the_command_and_keeps_its_output(managed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Validates the fake-encoded timeouts: exec answers `timed_out` with the output so far and kills the command's
    process group; a managed process's `exit` event says `timed_out`."""
    if managed:
        monkeypatch.setattr(f'{_MODULE}._EXEC_MAX_TIMEOUT', 1)
    async with _owned() as backend:
        with pytest.raises(WorkspaceTimeoutError) as raised:
            await backend.run(['sh', '-c', 'echo $$ > /tmp/pid; echo before; exec sleep 600'], timeout=3)
        assert raised.value.stdout == 'before\n'
        alive = await backend.run(['sh', '-c', 'kill -0 "$(cat /tmp/pid)"'], timeout=30)
        assert alive.exit_code != 0


async def test_a_cancelled_command_is_stopped() -> None:
    """Validates the fake-encoded stop: a second exec that kills the process group the first one recorded."""
    async with _owned() as backend:
        # real-wait: a real command, cancelled once it has had time to start in the sandbox.
        with anyio.move_on_after(5):
            await backend.run(['sh', '-c', 'echo $$ > /tmp/pid; exec sleep 600'], timeout=60)
        alive = await backend.run(['sh', '-c', 'kill -0 "$(cat /tmp/pid)"'], timeout=30)
        assert alive.exit_code != 0


async def test_a_background_job_ends_with_a_short_command_unless_detached() -> None:
    """Validates what the Limits section says of exec: a job started with `&` ends with the call, and one detached
    into its own session keeps running."""
    script = (
        'sleep 300 & echo $! > /tmp/plain; '
        'setsid nohup sleep 300 < /dev/null > /dev/null 2>&1 & echo $! > /tmp/detached'
    )
    async with _owned() as backend:
        await backend.run(['sh', '-c', script], timeout=30)
        plain = await backend.run(['sh', '-c', 'kill -0 "$(cat /tmp/plain)"'], timeout=30)
        detached = await backend.run(['sh', '-c', 'kill -0 "$(cat /tmp/detached)"'], timeout=30)
        assert (plain.exit_code != 0, detached.exit_code) == (True, 0)


async def test_a_file_past_one_request_s_limit_round_trips() -> None:
    """Validates the fake-encoded write limit: a `413 payload_too_large` that states `max_bytes`."""
    data = os.urandom(40 * 1024 * 1024)
    async with _owned() as backend:
        workspace = Workspace(backend)
        await workspace.write_bytes('big.bin', data)
        assert await workspace.read_bytes('big.bin') == data


async def test_a_sandbox_stopped_meanwhile_is_started_for_the_next_operation() -> None:
    """Validates the fake-encoded `409 sandbox_not_running` for an operation on a stopped sandbox, and that a start
    keeps the disk."""
    async with _owned() as backend:
        workspace = Workspace(backend)
        await workspace.write_text('kept.txt', 'kept')
        await (await backend.get_sandbox()).stop(wait_for_stopped=True)
        assert await workspace.read_text('kept.txt') == 'kept'
        assert (await backend.run(['cat', 'kept.txt'], timeout=30)).stdout == 'kept'


async def test_attaching_starts_a_stopped_sandbox_and_destroy_does_not() -> None:
    """Validates the fake-encoded lifecycle: attaching to a stopped sandbox starts it, and a delete never starts it."""
    async with _owned() as backend:
        assert backend.ref is not None
        sandbox = await backend.get_sandbox()
        await sandbox.stop(wait_for_stopped=True)
        attached = PlatinumSandbox().backend(backend.ref)
        try:
            assert await attached.working_dir() == '/workspace'
        finally:
            await attached.aclose()
        await sandbox.stop(wait_for_stopped=True)
        await PlatinumSandbox().destroy(backend.ref)
        with pytest.raises(platinum.NotFoundError):
            await sandbox.refresh()
        await PlatinumSandbox().destroy(backend.ref)  # already gone: quiet
        gone = PlatinumSandbox().backend(backend.ref)
        try:
            with pytest.raises(WorkspaceUnavailableError, match="Pass `workspace='new'`"):
                await gone.working_dir()
        finally:
            await gone.aclose()


async def test_without_internet_access_a_new_sandbox_has_no_route_out() -> None:
    """Validates the create field the fake only records: `allow_internet_access=False` blocks outbound connections."""
    probe = 'import socket; socket.create_connection(("1.1.1.1", 443), timeout=5); print("reached")'
    async with _owned(allow_internet_access=False) as blocked:
        assert (await blocked.run(['python3', '-c', probe], timeout=30)).exit_code != 0
    async with _owned() as open_:
        assert (await open_.run(['python3', '-c', probe], timeout=30)).stdout == 'reached\n'


async def test_an_unknown_template_is_unavailable() -> None:
    """Validates the fake-encoded refusal of a create: Platinum answers an unknown template with a 4xx."""
    backend = PlatinumSandboxBackend(template='pydantic-ai-no-such-template')
    try:
        with pytest.raises(WorkspaceUnavailableError, match='template'):
            await backend.get_sandbox()
        assert backend.ref is None
    finally:
        await backend.aclose()


_DOCS_BLOCKS = python_blocks('docs/harness/platinum-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    The fake stands in for Platinum, so only this shows the page's code, its default settings, and its cleanup work
    against the real service. A follow-up run, from the message history or a stored ref, works in the first run's
    sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
