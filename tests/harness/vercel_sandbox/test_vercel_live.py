"""Integration tests that require a real Vercel Sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, output collection, and
exception mapping, over real host processes. This live tier admits only what a correctly
written fake could not catch: Vercel's session lifecycle and persistence, its network policy,
what its default image ships, and real lifecycle state in its control plane. The conformance
suite's live class (`test_conformance.py`) covers the workspace contract itself.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Vercel behavior.

Gating:
  * `vercel_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_VERCEL_LIVE=1` opts in explicitly.
  * also requires `VERCEL_OIDC_TOKEN`, or `VERCEL_TOKEN` with `VERCEL_TEAM_ID` and
    `VERCEL_PROJECT_ID`; without them it skips, or fails when `VERCEL_REQUIRE_LIVE=1` (a CI live
    job). The gate is `pytest_runtest_setup` in `conftest.py`.

Run locally:
`PYDANTIC_AI_HARNESS_VERCEL_LIVE=1 uv run pytest -m vercel_live tests/harness/vercel_sandbox/test_vercel_live.py`
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest

from pydantic_ai.workspaces import Workspace, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .._tool_calls import call_tools
from .conftest import LIVE_SANDBOX_TIMEOUT

pytestmark = pytest.mark.vercel_live


@asynccontextmanager
async def _owned(
    *, allow_internet_access: bool = True, working_dir: str | None = None
) -> AsyncGenerator[VercelSandboxBackend]:
    """A backend whose sandbox, and the client it opened, are cleaned up whatever the test does."""
    backend = VercelSandboxBackend(
        sandbox_timeout=LIVE_SANDBOX_TIMEOUT, allow_internet_access=allow_internet_access, working_dir=working_dir
    )
    try:
        yield backend
    finally:
        try:
            if backend.ref is not None:
                await VercelSandbox().destroy(backend.ref)
        finally:
            await backend.aclose()


async def test_a_stopped_session_resumes_with_its_files() -> None:
    """Validates that a sandbox-level `create_process` resumes a stopped persistent sandbox from its
    saved files, which the fake never stops."""
    async with _owned() as backend:
        workspace = Workspace(backend)
        await workspace.write_text('kept.txt', 'survives a stop')
        sandbox = await backend.get_sandbox()
        await sandbox.stop()
        assert await workspace.read_text('kept.txt') == 'survives a stop'


async def test_destroy_does_not_resume_and_a_later_attach_is_unavailable() -> None:
    """Validates that `get_sandbox` and `destroy` work on a stopped sandbox without resuming it, and
    that a deleted sandbox's name then answers 404, which the fake models as one dictionary."""
    backend = VercelSandboxBackend(sandbox_timeout=LIVE_SANDBOX_TIMEOUT)
    try:
        await backend.run(['true'])
        ref = backend.ref
        assert ref is not None
        await (await backend.get_sandbox()).stop()
    finally:
        await backend.aclose()
    await VercelSandbox().destroy(ref)
    await VercelSandbox().destroy(ref)  # Already gone: quiet.
    attached = VercelSandboxBackend(ref=ref)
    try:
        with pytest.raises(WorkspaceUnavailableError, match='was not found'):
            await attached.run(['true'])
    finally:
        await attached.aclose()


async def test_deny_all_network_policy_blocks_the_internet() -> None:
    """Validates that `allow_internet_access=False` maps onto a policy Vercel enforces; the fake
    only records the `NetworkPolicy` it was given."""
    async with _owned(allow_internet_access=False) as backend:
        result = await backend.run(['curl', '-sS', '--max-time', '10', 'https://example.com'], timeout=60)
        assert result.exit_code != 0


async def test_the_default_image_has_what_the_docs_promise() -> None:
    """Validates the docs' claims about `vercel/sandbox/universal`: git and ripgrep for `Coder`, and
    a non-root `ubuntu` user that can create `/vercel/project`; the fake runs on the host."""
    async with _owned(working_dir='/vercel/project') as backend:
        # One name per `command -v`: dash, the image's `sh`, reports only the first of several.
        result = await backend.run('command -v git && command -v rg && id -un && pwd -P', shell=True)
        assert result.exit_code == 0, result.stderr
        *tools, user, cwd = result.stdout.splitlines()
        assert (len(tools), user, cwd) == (2, 'ubuntu', '/vercel/project')


async def test_coder_tools_run_in_the_sandbox() -> None:
    """Validates that a file `Coder` writes through the shell-derived file operations is where a
    command started in the working directory finds it, in a real sandbox."""
    async with _owned() as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from vercel')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )
        assert 'hi from vercel' in results[1]


async def test_attach_by_name_to_a_sandbox_created_elsewhere() -> None:
    """Validates the docs' advice to attach to an SDK-created sandbox by `WorkspaceRef(id=sandbox.name)`."""
    async with _owned() as seed:
        sandbox = await seed.get_sandbox()
        attached = VercelSandboxBackend(ref=WorkspaceRef(provider='vercel', id=sandbox.name))
        try:
            assert (await attached.run(['echo', 'attached'])).stdout == 'attached\n'
        finally:
            await attached.aclose()
