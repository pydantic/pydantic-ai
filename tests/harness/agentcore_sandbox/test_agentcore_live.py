"""Integration tests that require a real AgentCore Code Interpreter session.

The fake-backed suites cover the harness-owned logic. This live tier admits only what a correctly
written fake could not catch: each test names the assumption the fake encodes that it checks
against the real service.

Gating:
  * `agentcore_live` marker separates this tier from the fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1` opts in and boto3 finds AWS credentials;
    the gate is `pytest_runtest_setup` in `conftest.py`.
  * every session a test starts is stopped when it ends; the region is `AWS_REGION` (default `us-east-1`).

Run locally:
`PYDANTIC_AI_HARNESS_AGENTCORE_LIVE=1 uv run pytest -m agentcore_live tests/harness/agentcore_sandbox/`
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable

import anyio
import pytest
from pytest_examples import CodeExample

from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox, AgentCoreSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .conftest import LIVE_REGION

pytestmark = [pytest.mark.agentcore_live, pytest.mark.anyio]


@pytest.fixture
async def new_backend() -> AsyncIterator[Callable[[], AgentCoreSandboxBackend]]:
    """Build backends in `LIVE_REGION`, and stop every session they started when the test ends."""
    created: list[AgentCoreSandboxBackend] = []

    def build() -> AgentCoreSandboxBackend:
        backend = AgentCoreSandboxBackend(region_name=LIVE_REGION, working_dir=f'/tmp/live-{uuid.uuid4().hex}')
        created.append(backend)
        return backend

    yield build
    for backend in created:
        if backend.ref is not None:
            await AgentCoreSandbox[None](region_name=LIVE_REGION).destroy(backend.ref)


async def _alive(backend: AgentCoreSandboxBackend, pid: str) -> bool:
    return (await backend.run(['sh', '-c', 'kill -0 "$1" 2>/dev/null', 'sh', pid])).exit_code == 0


async def test_a_timeout_stops_the_command_child_and_returns_its_output(
    new_backend: Callable[[], AgentCoreSandboxBackend],
) -> None:
    """Fake: `stopTask` kills the child the wrapper `sh -c` runs, and the files hold the output so far."""
    backend = new_backend()
    with pytest.raises(WorkspaceTimeoutError) as caught:
        await backend.run(['sh', '-c', 'echo $$ > pid; printf started; printf warn >&2; exec sleep 3600'], timeout=5)
    assert (caught.value.stdout, caught.value.stderr) == ('started', 'warn')
    pid = (await backend.read_bytes(f'{await backend.working_dir()}/pid')).decode().strip()
    with anyio.fail_after(30):
        while await _alive(backend, pid):
            await anyio.sleep(0.5)


async def test_a_command_cancelled_before_its_start_answers_never_runs_on(
    new_backend: Callable[[], AgentCoreSandboxBackend],
) -> None:
    """Fake: a start that is still in flight at the deadline is fenced, so the command it started is stopped."""
    backend = new_backend()
    root = await backend.working_dir()
    with pytest.raises(WorkspaceTimeoutError):
        # Shorter than one round trip, so the deadline passes while the start request is in flight.
        await backend.run(['sh', '-c', 'sleep 5; touch late'], timeout=0.01)
    await anyio.sleep(10)
    assert not await backend.exists(f'{root}/late')


async def test_stopping_the_session_mid_command_is_unavailable(
    new_backend: Callable[[], AgentCoreSandboxBackend],
) -> None:
    """Fake: AgentCore refuses calls on a stopped session with `ValidationException` saying it is not active."""
    backend = new_backend()
    await backend.working_dir()
    assert backend.ref is not None
    ref = backend.ref

    async def stop_soon() -> None:
        await anyio.sleep(3)
        await AgentCoreSandbox[None](region_name=LIVE_REGION).destroy(ref)

    async with anyio.create_task_group() as group:
        group.start_soon(stop_soon)
        with pytest.raises(WorkspaceUnavailableError, match='is not active in this region'):
            await backend.run(['sleep', '60'], timeout=45)


async def test_attaching_after_destroy_is_unavailable(new_backend: Callable[[], AgentCoreSandboxBackend]) -> None:
    """Fake: an ended session is refused on attach, and destroying it again returns quietly (`ConflictException`)."""
    backend = new_backend()
    await backend.working_dir()
    assert backend.ref is not None
    capability = AgentCoreSandbox[None](region_name=LIVE_REGION)
    await capability.destroy(backend.ref)
    await capability.destroy(backend.ref)
    with pytest.raises(WorkspaceUnavailableError, match='is not active in this region'):
        await capability.backend(backend.ref).exists('/tmp')


async def test_output_over_the_limit_raises_with_previews(new_backend: Callable[[], AgentCoreSandboxBackend]) -> None:
    """Fake: the 4 MiB cap is measured on the redirected files, and the previews are their first 64 KiB."""
    backend = new_backend()
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['sh', '-c', "head -c 5242880 /dev/zero | tr '\\0' a; printf err >&2"])
    assert (caught.value.stdout, caught.value.stderr) == ('a' * 65536, 'err')


async def test_env_values_reach_the_command_as_written(new_backend: Callable[[], AgentCoreSandboxBackend]) -> None:
    """Fake: `env --` passes names and values through, including spaces, `=` and quotes."""
    backend = new_backend()
    value = 'a b = "c" $HOME'
    result = await backend.run(['sh', '-c', 'printf %s "$VALUE"'], env={'VALUE': value})
    assert result.stdout == value


async def test_a_failing_program_is_reported_from_its_traceback(
    new_backend: Callable[[], AgentCoreSandboxBackend],
) -> None:
    """Fake: an exception `executeCode` raises comes back as `isError` with the traceback as text content."""
    backend = new_backend()
    with pytest.raises(WorkspaceError, match=r'(?s)AgentCore executeCode failed: .*ValueError'):
        await backend.read_bytes('/tmp/bad\x00name')


_DOCS_BLOCKS = python_blocks('docs/harness/agentcore-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real session.

    A follow-up run, from the message history or a stored ref, works in the first run's session.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'stop_session'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
