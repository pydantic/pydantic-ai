"""Integration tests that require a real Vercel Sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation, and
exception mapping. This live tier admits only what a correctly written fake could not catch: real
process execution over Vercel's process stream, a client-owned deadline over a real process, and real
lifecycle state in Vercel's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Vercel behavior.

Gating:
  * `vercel_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_VERCEL_LIVE=1` opts in explicitly.
  * also requires `VERCEL_OIDC_TOKEN`, or `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and `VERCEL_PROJECT_ID`;
    without them it skips, or fails when `VERCEL_REQUIRE_LIVE=1`. The gate is `pytest_runtest_setup`
    in `conftest.py`.
  * the async `client` fixture opens a Vercel client per test and closes it when the test ends.

Run locally:
`PYDANTIC_AI_HARNESS_VERCEL_LIVE=1 uv run pytest -m vercel_live tests/harness/vercel_sandbox/`
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from pytest_examples import CodeExample
from vercel.sandbox import SandboxClient

from pydantic_ai.workspaces import WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_EXECUTION_TIME_LIMIT

pytestmark = pytest.mark.vercel_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture
async def client() -> AsyncIterator[SandboxClient]:
    """A caller-owned API client for one test's backends, closed when the test ends."""
    client = SandboxClient.create()
    try:
        yield client
    finally:
        await client.aclose()


@asynccontextmanager
async def _owned(client: SandboxClient, **settings: Any) -> AsyncGenerator[VercelSandboxBackend]:
    """Create a sandbox and destroy it on the way out, even when the test destroyed it already."""
    backend = VercelSandboxBackend(client=client, execution_time_limit=LIVE_EXECUTION_TIME_LIMIT, **settings)
    try:
        await backend.get_sandbox()
        yield backend
    finally:
        if backend.ref is not None:
            await VercelSandbox(client=client).destroy(backend.ref)


async def _stop(client: SandboxClient, name: str) -> None:
    """Stop the sandbox's session and wait until Vercel reports it stopped."""
    sandbox = await client.get_sandbox(name=name)
    await sandbox.stop()
    with anyio.fail_after(120):
        while (await client.get_sandbox(name=name)).status != 'stopped':
            await anyio.sleep(1)


async def test_commands_get_their_streams_exit_code_and_a_usable_environment(client: SandboxClient) -> None:
    """Validates the fake-encoded assumptions that `run_process` reports stdout and stderr apart with the real
    exit code, and that the image has what `Coder` needs: `/bin/sh`, git, Python 3, and ripgrep. A per-call
    `env` adds to the image's environment rather than replacing it, and the docs' claim that the
    sandbox's environment holds no Vercel credentials."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)
        assert (result.stdout, result.stderr, result.exit_code) == ('out\n', 'err\n', 3)

        result = await backend.run(
            'echo "$PATH"; pwd -P; git --version; python3 --version; command -v rg', shell=True, timeout=60
        )
        path, cwd, git, python, rg = result.stdout.splitlines()
        assert cwd == '/vercel/sandbox' == await backend.working_dir()
        assert path and git.startswith('git version') and python.startswith('Python 3') and rg.endswith('/rg')

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]

        result = await backend.run('test -z "$VERCEL_OIDC_TOKEN$VERCEL_TOKEN"', shell=True, timeout=60)
        assert result.exit_code == 0


async def test_large_output_is_exact(client: SandboxClient) -> None:
    """Validates the fake-encoded assumption that the process stream delivers a command's output whole and in order."""
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '200000'], timeout=120)
        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 200001))


async def test_timeout_raises_with_the_partial_output_and_kills_the_command(client: SandboxClient) -> None:
    """Validates the fake-encoded assumptions that output printed before a client-side deadline reaches the
    `WorkspaceTimeoutError`, and that `Process.kill()` ends the command the wrapper `exec`s."""
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sh', '-c', f'echo DIAGNOSTIC; sleep 20; touch {marker}'], timeout=3)
        assert exc_info.value.stdout == 'DIAGNOSTIC\n'
        assert (await backend.run(['sleep', '20'], timeout=60)).exit_code == 0
        assert (await backend.run(['test', '-e', marker], timeout=30)).exit_code == 1


async def test_a_new_sandbox_gets_its_working_dir(client: SandboxClient) -> None:
    """Validates the fake-encoded assumption that `mkdir -p` through `run_process` makes `working_dir` on a
    new sandbox, so the first command can start there."""
    directory = f'/vercel/sandbox/{_unique("project")}/nested'
    async with _owned(client, working_dir=directory) as backend:
        assert (await backend.run(['pwd', '-P'], timeout=60)).stdout == f'{directory}\n'


async def test_a_stopped_persistent_sandbox_resumes_with_its_files(client: SandboxClient) -> None:
    """Validates the SDK behavior the fake skips: a command on a stopped persistent sandbox resumes it from the
    snapshot its session saved, so a backend that attaches later finds the files it left."""
    async with _owned(client) as owner:
        await owner.run(['sh', '-c', 'printf kept > kept.txt'], timeout=60)
        assert owner.ref is not None
        await _stop(client, owner.ref.id)
        attached = VercelSandboxBackend(client=client, ref=owner.ref)
        assert (await attached.run(['cat', 'kept.txt'], timeout=180)).stdout == 'kept'


async def test_a_stopped_sandbox_that_is_not_persistent_is_unavailable(client: SandboxClient) -> None:
    """Validates the fake-encoded assumption that Vercel reports a non-persistent sandbox `stopped` once its
    session ends, and keeps reporting it rather than resuming it."""
    async with _owned(client, persistent=False) as owner:
        assert owner.ref is not None
        await _stop(client, owner.ref.id)
        with pytest.raises(WorkspaceUnavailableError, match='not persistent'):
            await VercelSandboxBackend(client=client, ref=owner.ref).run(['true'], timeout=60)


async def test_a_session_stopped_during_a_command_is_unavailable(client: SandboxClient) -> None:
    """Validates the fake-encoded assumption that stopping the session ends a running command with a stream
    failure, after which a lookup reports the sandbox stopped."""
    marker = f'/tmp/{_unique("started")}'
    async with _owned(client) as backend:
        assert backend.ref is not None
        name = backend.ref.id

        async def stop_once_started() -> None:
            sandbox = await client.get_sandbox(name=name)
            with anyio.fail_after(60):
                while (await sandbox.run_process('test', ['-e', marker])).returncode != 0:
                    await anyio.sleep(0.5)
            await sandbox.stop()

        async with anyio.create_task_group() as group:
            group.start_soon(stop_once_started)
            with pytest.raises(WorkspaceUnavailableError, match='ended during the command'):
                await backend.run(['sh', '-c', f'touch {marker}; exec sleep 60'], timeout=90)


async def test_coder_tools_run_in_the_sandbox(client: SandboxClient) -> None:
    """Validates the fake-encoded assumption that a relative path `Coder` writes through the shell-derived file
    operations is where a command started in the working directory finds it."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from vercel')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )
        assert 'hi from vercel' in results[1]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/vercel-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and an example that runs an agent has its tools do
    their work in a real sandbox.

    The fake stands in for Vercel, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's sandbox. An example that only defines a function is checked to run, not called.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'destroy_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
