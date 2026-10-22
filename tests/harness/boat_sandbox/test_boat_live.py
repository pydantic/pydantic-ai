"""Integration tests that require a real Boat sandbox.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution, a client-owned deadline over a real detached command, one filesystem
shared by Boat's file API and its commands, and real lifecycle state in Boat's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Boat behavior.

Gating:
  * `boat_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_BOAT_LIVE=1` opts in explicitly.
  * also requires a non-empty `BOAT_API_KEY`; without one it skips, or fails when
    `BOAT_REQUIRE_LIVE=1` (CI's live job). The gate is `pytest_runtest_setup` in `conftest.py`.
  * every sandbox is `small` where a test picks the size, and Boat stops it `LIVE_AUTO_STOP`
    seconds after the backend last extended it, so one a killed job leaves behind does not run for
    Boat's default hour.

Run locally:
`PYDANTIC_AI_HARNESS_BOAT_LIVE=1 uv run pytest -m boat_live tests/harness/boat_sandbox/test_boat_live.py`
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
import pytest
from pytest_examples import CodeExample

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.shell import Shell

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_AUTO_STOP

pytestmark = pytest.mark.boat_live

_OUTPUT_LIMIT = 8 * 1024 * 1024
_INLINE_FILE_LIMIT = 5 * 1024 * 1024


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """A caller-owned API client for one test's backends, configured as the docs say to configure `client=`."""
    async with httpx.AsyncClient(
        base_url='https://boat.dev/api/v1', headers={'Authorization': f'Bearer {os.environ["BOAT_API_KEY"]}'}
    ) as client:
        yield client


@asynccontextmanager
async def _owned(client: httpx.AsyncClient, *, working_dir: str | None = None) -> AsyncGenerator[BoatSandboxBackend]:
    """Create a sandbox and delete it on the way out, even when the test deleted it already."""
    backend = BoatSandboxBackend(client=client, machine_type='small', auto_stop=LIVE_AUTO_STOP, working_dir=working_dir)
    try:
        await backend.working_dir()
        yield backend
    finally:
        if backend.ref is not None:
            await BoatSandbox(client=client).destroy(backend.ref)
            # Boat forgets a deleted sandbox at once.
            assert (await client.get(f'/sandboxes/{backend.ref.id}')).status_code == 404


async def _state(client: httpx.AsyncClient, sandbox_id: str) -> str:
    response = await client.get(f'/sandboxes/{sandbox_id}')
    response.raise_for_status()
    state = response.json()['sandbox']['state']
    assert isinstance(state, str)
    return state


async def _assert_stopped(backend: BoatSandboxBackend, pid_file: str) -> None:
    """Every process whose ID the command wrote to `pid_file` has ended."""
    pids = (await backend.read_bytes(pid_file)).decode().split()
    assert pids
    with anyio.fail_after(30):
        script = 'for pid; do kill -0 "$pid" 2>/dev/null && exit 0; done; exit 1'
        while (await backend.run(['sh', '-c', script, 'sh', *pids], timeout=30)).exit_code == 0:
            await anyio.sleep(0.2)


async def test_an_operation_extends_the_auto_stop(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Validates the fake-encoded assumption that `PATCH /sandboxes/{id}` with `ttlSeconds` moves Boat's
    auto-stop deadline to that many seconds from now, and that the key the live tier runs with may
    send it, so an active run keeps its sandbox up."""
    # Extend on every operation, so the test needs no real wait.
    monkeypatch.setattr('pydantic_ai_harness.boat_sandbox._backend._KEEP_ALIVE_INTERVAL', 0)
    async with _owned(client) as backend:
        assert backend.ref is not None
        sandbox_id = backend.ref.id

        async def deadline() -> str:
            response = await client.get(f'/sandboxes/{sandbox_id}')
            response.raise_for_status()
            archive_after = response.json()['sandbox']['archiveAfter']
            assert isinstance(archive_after, str)
            return archive_after

        before = await deadline()
        await anyio.sleep(2)
        with caplog.at_level('WARNING', logger='pydantic_ai_harness.boat_sandbox._backend'):
            assert (await backend.run(['true'])).exit_code == 0
        assert not caplog.records
        # ISO 8601 timestamps in UTC compare in time order.
        assert await deadline() > before


async def test_commands_get_a_usable_environment(client: httpx.AsyncClient) -> None:
    """Validates what the fake encodes and the docs say about a new sandbox: Ubuntu 24.04 with git,
    Python, Node.js, Docker and ripgrep; commands run as `user` in `/home/user` with passwordless
    `sudo`, Boat's `PATH`, and without the account's GitHub token (`noEnv`). A per-call `env` adds
    to the environment, with its values and names passed through literally."""
    async with _owned(client) as backend:
        result = await backend.run(
            '. /etc/os-release; echo "$VERSION_ID"; echo "$PATH"; echo "$HOME"; whoami; pwd -P; sudo -n true && echo sudo; '
            'git --version; python3 --version; node --version; docker --version; command -v rg; '
            'echo "github-token-${GITHUB_TOKEN+set}"',
            shell=True,
            timeout=60,
        )
        version, path, home, user, cwd, sudo, git, python, node, docker, rg, token = result.stdout.splitlines()
        assert (version, home, user, cwd, sudo, token) == (
            '24.04',
            '/home/user',
            'user',
            '/home/user',
            'sudo',
            'github-token-',
        )
        assert path and rg.endswith('/rg')
        assert git.startswith('git version') and python.startswith('Python 3') and node.startswith('v')
        assert docker.startswith('Docker version')
        assert await backend.working_dir() == '/home/user'

        value = 'it\'s "quoted" $HOME `x`  two spaces \\ end'
        result = await backend.run(
            'printf "%s\\n" "$FOO" "$PATH"', shell=True, env={'FOO': value, 'PAI-DASHED': 'kept'}, timeout=60
        )
        assert result.stdout.splitlines() == [value, path]
        # `dash` drops a name that is not a shell identifier, but a program run without a shell gets it.
        result = await backend.run(['printenv', 'PAI-DASHED'], env={'PAI-DASHED': 'kept'}, timeout=60)
        assert result.stdout == 'kept\n'


async def test_large_output_is_exact(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumption that a finished command's status, read again with a tail as large
    as Boat keeps, is its exact output when the first check's short tail was cut."""
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '50000'], timeout=120)

        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 50001))


async def test_output_over_the_limit_raises_with_a_preview(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumption that Boat keeps the last 8 MiB of each stream and flags a
    longer one as truncated: output at the limit comes back whole, and output past it raises
    `WorkspaceOutputLimitError` with the beginning of each stream read from offset 0."""
    fill = f'head -c {_OUTPUT_LIMIT} /dev/zero | tr "\\0" a'
    async with _owned(client) as backend:
        result = await backend.run(fill, shell=True, timeout=120)
        assert (result.exit_code, len(result.stdout), set(result.stdout)) == (0, _OUTPUT_LIMIT, {'a'})

        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await backend.run(f'printf b; {fill}; printf e >&2', shell=True, timeout=120)
        error = exc_info.value
        assert error.limit == _OUTPUT_LIMIT
        assert error.stdout == 'b' + 'a' * (64 * 1024 - 1)
        assert error.stderr == 'e'


async def test_timeout_raises_with_the_partial_output_and_stops_the_command(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumptions that a running command's status carries the tail of what
    it printed so far, and that `DELETE /commands/{pid}` ends its process tree, background children
    included, while the sandbox keeps running for the next command."""
    pid_file = f'/tmp/{_unique("pids")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(
                f'sleep 600 & echo "$$ $!" > {pid_file}; echo DIAGNOSTIC; echo ERR >&2; wait', shell=True, timeout=5
            )

        assert 'DIAGNOSTIC' in exc_info.value.stdout and 'ERR' in exc_info.value.stderr
        assert '5' in str(exc_info.value)
        await _assert_stopped(backend, pid_file)


async def test_a_cancelled_run_stops_the_command(client: httpx.AsyncClient) -> None:
    """Validates the protocol's cancellation contract against real Boat, and the docs claim that a
    stopped command's whole process tree ends: a cancelled `run()` stops the command, a child that
    left for a session of its own included."""
    pid_file = f'/tmp/{_unique("pids")}'
    async with _owned(client) as backend:
        with anyio.move_on_after(5):
            await backend.run(f'setsid sleep 600 & echo "$$ $!" > {pid_file}; wait', shell=True, timeout=60)

        await _assert_stopped(backend, pid_file)


async def test_a_new_sandbox_gets_its_working_dir(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumption that `mkdir -p` through a synchronous command makes
    `working_dir` on a new sandbox as `user`, so the first command starts there, and the docs claim
    that on an attached sandbox a missing `working_dir` fails with `WorkspaceError`."""
    directory = f'/home/user/{_unique("project")}/nested'
    async with _owned(client, working_dir=directory) as backend:
        result = await backend.run(['pwd', '-P'], timeout=60)
        assert (result.exit_code, result.stdout) == (0, f'{directory}\n')
        assert backend.ref is not None

        attached = BoatSandboxBackend(client=client, ref=backend.ref, working_dir=f'{directory}-missing')
        with pytest.raises(WorkspaceError, match='does not exist'):
            await attached.run(['true'], timeout=60)


async def test_files_over_the_inline_limit_round_trip(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumptions that Boat's file API refuses a file over 5 MiB with
    `File is too large`, and that its signed upload and download URLs, sent without the API key,
    move such a file whole and create its missing parents."""
    data = os.urandom(_INLINE_FILE_LIMIT + 1)
    path = f'/home/user/{_unique("big")}/big.bin'
    async with _owned(client) as backend:
        assert backend.ref is not None
        await backend.write_bytes(path, data)

        refused = await client.get(f'/sandboxes/{backend.ref.id}/files', params={'path': path, 'encoding': 'base64'})
        assert refused.status_code == 400 and refused.json()['message'].startswith('File is too large')
        assert await backend.read_bytes(path) == data
        result = await backend.run(['sha256sum', path], timeout=60)
        assert result.stdout.split()[0] == hashlib.sha256(data).hexdigest()


async def test_paths_outside_the_file_api_roots_round_trip(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumption that Boat's file API refuses a path that resolves outside
    `/home/user` and `/tmp`, by its name (`invalid_path`) or through a symlink (`sandbox_direct_failed`),
    so those reads and writes go through shell commands run as `user` and fail with the built-in errors."""
    root = f'/var/tmp/{_unique("outside")}'
    link = f'/home/user/{_unique("link-out")}'
    data = bytes(range(256))
    async with _owned(client) as backend:
        await backend.write_bytes(f'{root}/nested/all-bytes.bin', data)
        assert await backend.read_bytes(f'{root}/nested/all-bytes.bin') == data
        assert (await backend.stat(f'{root}/nested/all-bytes.bin')).size == 256

        assert (await backend.run(['ln', '-s', root, link], timeout=30)).exit_code == 0
        await backend.write_bytes(f'{link}/through-link.txt', b'through a link')
        assert await backend.read_bytes(f'{root}/through-link.txt') == b'through a link'
        assert await backend.read_bytes(f'{link}/through-link.txt') == b'through a link'

        assert await backend.read_bytes('/etc/hostname')
        with pytest.raises(FileNotFoundError):
            await backend.read_bytes(f'{root}/missing')
        with pytest.raises(PermissionError):
            await backend.write_bytes(f'/etc/{_unique("denied")}', b'x')


async def test_a_sandbox_stopped_under_a_live_backend_is_unavailable_until_attached_again(
    client: httpx.AsyncClient,
) -> None:
    """Validates the fake-encoded assumptions about Boat's stop: once the sandbox is archived, commands
    and file calls answer `409 machine_not_running`, which ends the run as `WorkspaceUnavailableError`,
    and a backend attached by ref resumes the sandbox with its disk."""
    path = f'/home/user/{_unique("kept")}.txt'
    async with _owned(client) as backend:
        await backend.write_bytes(path, b'kept')
        assert backend.ref is not None
        sandbox_id = backend.ref.id
        assert (await client.post(f'/sandboxes/{sandbox_id}/stop')).status_code == 202
        with anyio.fail_after(60):
            while await _state(client, sandbox_id) != 'archived':
                await anyio.sleep(0.5)

        with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
            await backend.run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
            await backend.read_bytes(path)

        attached = BoatSandboxBackend(client=client, ref=backend.ref)
        assert await attached.read_bytes(path) == b'kept'
        assert (await attached.run(['cat', path], timeout=30)).stdout == 'kept'


async def test_a_deleted_sandbox_is_unavailable_and_destroying_it_again_is_quiet(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumptions about deletion: right after `DELETE`, Boat answers
    `404 not_found` on every call for the sandbox, which a held backend reports as
    `WorkspaceUnavailableError` rather than a missing file, an attach reports the same way, and
    destroying it again returns quietly, with a caller's client or one opened from `BOAT_API_KEY`."""
    path = f'/home/user/{_unique("deleted")}.txt'
    async with _owned(client) as backend:
        await backend.write_bytes(path, b'before-delete')
        ref = backend.ref
        assert ref is not None
        await BoatSandbox(client=client).destroy(ref)

        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes(path)
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError, match='was not found'):
            await BoatSandboxBackend(client=client, ref=ref).working_dir()
        await BoatSandbox(client=client).destroy(ref)
        await BoatSandbox().destroy(ref)


async def test_coder_tools_run_in_the_sandbox(client: httpx.AsyncClient) -> None:
    """Validates the fake-encoded assumption that a relative path the file API writes is where a
    command started in the working directory finds it, so `Coder`'s tools agree."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from boat')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )

        assert 'hi from boat' in results[1]


def _shell_model() -> FunctionModel:
    """A model that runs each prompt as one `run_command` call, then answers with what it returned."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        for part in request.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': prompt})])

    return FunctionModel(respond)


async def test_a_later_run_continues_in_the_same_sandbox() -> None:
    """Validates the fake-encoded assumption that the ref a run records, found in the message history
    or passed back from `result.workspace.ref`, attaches to the same Boat sandbox, and that
    `workspace='new'` starts another one even when the history names the first."""
    refs: list[WorkspaceRef] = []
    hooks = Hooks()

    # A run that fails after creating its sandbox returns no result to read the ref from.
    @hooks.on.run_error
    async def keep_ref(ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[Any]:
        if ctx.workspace.ref is not None:
            refs.append(ctx.workspace.ref)
        raise error

    agent = Agent(
        _shell_model(), capabilities=[BoatSandbox(machine_type='small', auto_stop=LIVE_AUTO_STOP), Shell(), hooks]
    )
    try:
        first = await agent.run('echo continued > ~/continued.txt && echo written')
        ref = first.workspace.ref
        assert ref is not None
        refs.append(ref)
        assert 'written' in first.output

        followup = await agent.run('cat ~/continued.txt', message_history=first.all_messages())
        assert 'continued' in followup.output and followup.workspace.ref == ref
        later = await agent.run('cat ~/continued.txt', workspace=ref)
        assert 'continued' in later.output and later.workspace.ref == ref

        fresh = await agent.run(
            'test -e ~/continued.txt && echo present || echo absent',
            message_history=first.all_messages(),
            workspace='new',
        )
        assert fresh.workspace.ref is not None
        refs.append(fresh.workspace.ref)
        assert fresh.workspace.ref != ref and 'absent' in fresh.output
    finally:
        # `destroy` returns quietly for a sandbox already deleted, so a ref recorded twice is harmless.
        for ref in refs:
            await BoatSandbox().destroy(ref)


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/boat-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    The fake stands in for Boat, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's sandbox. Only the auto-stop of a sandbox a block creates is set, to the
    live tier's, so one a killed job leaves behind stops early.
    """
    init = BoatSandboxBackend.__init__

    def stop_early(self: BoatSandboxBackend, *, auto_stop: int | None = None, **settings: Any) -> None:
        init(self, auto_stop=LIVE_AUTO_STOP if auto_stop is None else auto_stop, **settings)

    monkeypatch.setattr(BoatSandboxBackend, '__init__', stop_early)
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
