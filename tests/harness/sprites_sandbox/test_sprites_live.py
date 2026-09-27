"""Integration tests that require a real Fly.io Sprite.

The fake-backed suites cover the harness-owned logic: deadline handling, cancellation, and
exception mapping. This live tier admits only what a correctly written fake could not catch: real
process execution in a Sprite, `env` layered on the Sprite's environment, output reaching the
client before a command ends, a closed exec connection stopping a timed-out command, and deletion
as the Sprites control plane reports it.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real Sprites behavior.

Gating:
  * `sprites_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_SPRITES_LIVE=1` opts in explicitly.
  * also requires a non-empty `SPRITE_TOKEN`; CI sets `SPRITES_REQUIRE_LIVE`, so there a
    missing token fails instead of skipping.
  * a module-scoped `anyio_backend` fixture keeps the shared Sprites client on one asyncio loop.

Run locally:
`PYDANTIC_AI_HARNESS_SPRITES_LIVE=1 uv run pytest -m sprites_live tests/sprites_sandbox/test_sprites_live.py`
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.workspaces import Workspace, WorkspaceTimeoutError, WorkspaceUnavailableError
from pytest_examples import CodeExample
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.exceptions import NotFoundError, SpriteError

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block

pytestmark = [pytest.mark.sprites_live]


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(scope='module', autouse=True)
def wait_out_the_creation_rate_limit(sprites_token: str) -> Iterator[None]:
    """The account creates at most 10 Sprites a minute, and this module creates more than that."""
    create = AsyncSpritesClient.create_sprite

    async def create_sprite(self: AsyncSpritesClient, *args: Any, **kwargs: Any) -> AsyncSprite:
        try:
            return await create(self, *args, **kwargs)
        except SpriteError as error:
            if 'sprite_creation_rate_limited' not in str(error):
                raise
        await anyio.sleep(61)
        return await create(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AsyncSpritesClient, 'create_sprite', create_sprite)
        yield


@pytest.fixture(scope='module')
async def client(sprites_token: str) -> AsyncIterator[AsyncSpritesClient]:
    """One caller-owned API client for every backend in this module, closed at the end."""
    async with AsyncSpritesClient(token=sprites_token) as client:
        yield client


@asynccontextmanager
async def _owned(client: AsyncSpritesClient) -> AsyncGenerator[SpritesSandboxBackend]:
    """Create a Sprite and delete it on the way out, even when the test deleted it already."""
    backend = SpritesSandboxBackend(client=client)
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        try:
            await native.delete()
        except NotFoundError:
            pass


async def test_creates_a_fresh_sprite_and_runs_a_command(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that the exec WebSocket reports real output and exit code."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)

        assert backend.ref is not None
        assert (result.stdout.strip(), result.stderr.strip(), result.exit_code) == ('out', 'err', 3)


async def test_destroy_ref_does_not_attach(client: AsyncSpritesClient) -> None:
    """Only a live Sprite can prove the SDK destroys by id without resuming it."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        await SpritesSandbox(client=client).destroy(backend.ref)
        with pytest.raises(NotFoundError):
            await client.get_sprite(backend.ref.id)


async def test_reattach_by_ref_reads_a_file_the_first_backend_wrote(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that `get_sprite(ref.id)` reaches the same Sprite."""
    path = f'/tmp/{_unique("reattach")}.txt'
    async with _owned(client) as owner:
        assert (await owner.run(['sh', '-c', 'printf shared > "$1"', 'sh', path], timeout=60)).exit_code == 0
        assert owner.ref is not None

        attached = SpritesSandboxBackend(client=client, ref=owner.ref)
        result = await attached.run(['cat', path], timeout=60)

        assert (result.exit_code, result.stdout) == (0, 'shared')
        assert attached.ref == owner.ref


async def test_command_and_file_round_trip(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that binary data survives the Sprite's file API and shell.

    Writes go through the filesystem API and reads through commands; the fake runs both on the host,
    which cannot show what the Sprite's API, shell, and tools do with them.
    """
    root = f'/tmp/{_unique("roundtrip")}'
    async with _owned(client) as backend:
        workspace = Workspace(backend)
        await workspace.write_bytes(f'{root}/nested/in.txt', b'from-file-api\n')
        await workspace.write_bytes(f'{root}/binary.bin', b'\x00\xff\n')
        result = await backend.run(
            ['sh', '-c', 'cat "$1/nested/in.txt" && printf from-shell > "$1/out.txt"', 'sh', root], timeout=60
        )

        assert (result.exit_code, result.stdout) == (0, 'from-file-api\n')
        assert await workspace.read_bytes(f'{root}/out.txt') == b'from-shell'
        assert await workspace.read_bytes(f'{root}/binary.bin') == b'\x00\xff\n'


async def test_large_output_and_files_arrive_whole(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumptions that a command's output is streamed only once the client
    attaches, and that a large write fits no command.

    The live Sprite starts a command before the client's stream attaches and replays only the last
    16 or 64 KiB printed until then, and refuses an exec URL (which carries argv) above about 40 KB.
    """
    path = f'/tmp/{_unique("large")}.bin'
    data = bytes(range(256)) * 4096
    expected = ''.join(f'{i}\n' for i in range(1, 150_001))
    async with _owned(client) as backend:
        workspace = Workspace(backend)
        await workspace.write_bytes(path, data)
        assert await workspace.read_bytes(path) == data
        for _ in range(3):
            assert (await backend.run(['seq', '1', '150000'], timeout=60)).stdout == expected


async def test_a_timed_out_command_is_killed(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumptions that exec output streams before the command ends and
    that closing the exec WebSocket, opened with `max_run_after_disconnect=1s`, stops the command.

    The deadline is enforced client-side, so the output printed before it expired must reach the
    `WorkspaceTimeoutError`. The command must be gone well within the 10 seconds a disconnected
    non-TTY command keeps running by default, so it is that parameter that stopped it.
    """
    pid_file = f'/tmp/{_unique("timed-out")}.pid'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo $$ > {pid_file}; echo DIAGNOSTIC; sleep 60', shell=True, timeout=5)

        assert 'DIAGNOSTIC' in exc_info.value.stdout
        assert str(exc_info.value) == 'Command timed out after 5 seconds'
        check = await backend.run(['sh', '-c', 'sleep 3; kill -0 "$(cat "$1")"', 'sh', pid_file], timeout=60)
        assert check.exit_code != 0


async def test_timeout_and_cancellation_stop_foreground_child(client: AsyncSpritesClient) -> None:
    """The fake kills its subprocess group on close; only a real Sprite can prove child death."""
    backend = SpritesSandboxBackend(client=client)
    native = await backend.get_sandbox()
    try:
        for mode in ('timeout', 'cancel'):
            pid_file = f'/tmp/{_unique("child")}.pid'
            command = f'sleep 30 & echo $! > {pid_file}; wait'
            if mode == 'timeout':
                with pytest.raises(WorkspaceTimeoutError):
                    await backend.run(command, shell=True, timeout=3)
            else:
                with anyio.move_on_after(3) as scope:
                    await backend.run(command, shell=True)
                assert scope.cancelled_caught
            check = await backend.run(['sh', '-c', 'sleep 2; kill -0 "$(cat "$1")"', 'sh', pid_file], timeout=30)
            assert check.exit_code != 0, f'{mode}: foreground child survived'
    finally:
        await native.delete()
        with pytest.raises(NotFoundError):
            await client.get_sprite(native.name)


async def test_timeout_keeps_stderr_and_cleans_capture(client: AsyncSpritesClient) -> None:
    """The fake cannot prove the live exec stream preserves stderr after an interrupted command."""
    backend = SpritesSandboxBackend(client=client)
    native = await backend.get_sandbox()
    try:
        # A listing command creates its own empty capture while it runs; check only nonempty captures.
        check = ['find', '/tmp', '-maxdepth', '1', '-name', 'pydantic-ai-stderr-*', '-size', '+0c']
        previous = await backend.run(check, timeout=30)
        with pytest.raises(WorkspaceTimeoutError) as caught:
            await backend.run('echo ERROR >&2; sleep 30', shell=True, timeout=3)
        assert 'ERROR' in caught.value.stderr
        result = await backend.run(check, timeout=30)
        assert result.stdout == previous.stdout
    finally:
        await native.delete()
        with pytest.raises(NotFoundError):
            await client.get_sprite(native.name)


async def test_reattach_to_a_deleted_sprite_is_unavailable(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption about deletion: once `delete()` returns, `get_sprite`
    raises `NotFoundError` and an exec WebSocket to the Sprite fails its handshake with HTTP 404.

    If Sprites answers either differently, a deleted Sprite would surface as a retryable
    `WorkspaceError` or, worse, as a missing file.
    """
    async with _owned(client) as owner:
        assert (await owner.run(['true'], timeout=60)).exit_code == 0
        assert owner.ref is not None
        await (await owner.get_sandbox()).delete()

        with pytest.raises(WorkspaceUnavailableError):
            await SpritesSandboxBackend(client=client, ref=owner.ref).working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=60)
        with pytest.raises(WorkspaceUnavailableError):
            await Workspace(owner).read_bytes('/tmp/anything')


async def test_coder_shell_and_file_tools_run_in_the_sprite(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that Coder's tools work against a real Sprite.

    The fake runs `shell` on the host, whose `sh`, `git`, and file tools are not the Sprite's.
    """

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            yield {0: DeltaToolCall('write_file', json.dumps({'path': 'notes.txt', 'content': 'hello'}))}
        elif len(returns) == 1:
            yield {0: DeltaToolCall('shell', json.dumps({'command': 'git --version && cat notes.txt'}))}
        else:
            yield str(returns[-1].content)

    async with _owned(client) as backend:
        agent = Agent(FunctionModel(stream_function=model), capabilities=[SpritesSandbox(client=client), Coder()])
        result = await agent.run('go', workspace=backend.ref)

    assert 'git version' in result.output
    assert 'hello' in result.output


async def test_commands_get_a_usable_environment(client: AsyncSpritesClient) -> None:
    """Validates the fake-encoded assumption that a command sees the Sprite's `PATH` and `HOME`, and git,
    and that a per-call `env` adds to them rather than replacing them."""
    async with _owned(client) as backend:
        result = await backend.run('echo "$PATH"; echo "$HOME"; git --version', shell=True, timeout=60)
        path, home, git = result.stdout.splitlines()
        assert path and home and git.startswith('git version')

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/sprites-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample, sprites_token: str) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    The fake stands in for Sprites, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sprite'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
