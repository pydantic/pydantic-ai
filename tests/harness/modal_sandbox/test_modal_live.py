"""Opt-in integration tests for a real Modal workspace.

Skipped unless `PYDANTIC_AI_HARNESS_MODAL_LIVE=1` and Modal credentials are present; CI also
sets `MODAL_REQUIRE_LIVE`, which turns that skip into a failure (see `conftest.py`).

Run locally:
`PYDANTIC_AI_HARNESS_MODAL_LIVE=1 uv run pytest -m modal_live tests/modal_sandbox`
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pytest_examples import CodeExample

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from .conftest import LIVE_IDLE_TIMEOUT, LIVE_SANDBOX_TIMEOUT

pytestmark = [pytest.mark.anyio(backends=['asyncio']), pytest.mark.modal_live]


@asynccontextmanager
async def owned_backend(**settings: Any) -> AsyncGenerator[ModalSandboxBackend, None]:
    defaults: dict[str, Any] = {
        'sandbox_timeout': LIVE_SANDBOX_TIMEOUT,
        'idle_timeout': LIVE_IDLE_TIMEOUT,
    }
    backend = ModalSandboxBackend(**(defaults | settings))
    native = await backend.get_sandbox()
    try:
        yield backend
    finally:
        try:
            await native.terminate.aio()
        finally:
            # Modal 1.5.2 leaves the return type of `detach.aio()` unspecified.
            await native.detach.aio()  # pyright: ignore[reportUnknownMemberType]


async def test_destroy_ref_without_resuming_workspace() -> None:
    async with owned_backend() as backend:
        ref = backend.ref
        assert ref is not None
        await ModalSandbox().destroy(ref)
        # Modal may keep poll() at running while it shuts down; refusal to exec
        # proves the ref can no longer run work without waiting for poll convergence.
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['true'], timeout=10)


async def test_cancel_stops_foreground_descendants_without_destroying_sandbox() -> None:
    marker = uuid.uuid4().hex
    backend = ModalSandboxBackend(sandbox_timeout=LIVE_SANDBOX_TIMEOUT, idle_timeout=LIVE_IDLE_TIMEOUT)
    native = await backend.get_sandbox()
    try:
        task = asyncio.create_task(
            backend.run(
                [
                    'sh',
                    '-c',
                    f'python -c "import os; print(os.getpgrp())" > /tmp/{marker}.pgid; '
                    f'sleep 30 & sleep 30; touch /tmp/{marker}.marker',
                ],
                timeout=None,
            )
        )
        try:
            with anyio.fail_after(30):
                while not await backend.exists(f'/tmp/{marker}.pgid'):
                    await anyio.sleep(0.1)
            assert not task.done(), task.result() if task.done() else None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert backend.ref is not None
            # `kill -0` includes zombies; inspect /proc for live group members instead.
            probe = (
                'import os; pgid=int(open("/tmp/' + marker + '.pgid").read()); '
                'print([p for p in os.listdir("/proc") if p.isdigit() and '
                'os.path.exists("/proc/"+p+"/stat") and '
                'open("/proc/"+p+"/stat").read().split()[2] != "Z" and '
                'int(open("/proc/"+p+"/stat").read().split()[4]) == pgid])'
            )
            group = await backend.run(['python', '-c', probe], timeout=15)
            assert group.stdout.strip() == '[]'
            assert not await backend.exists(f'/tmp/{marker}.marker')
            assert (await backend.run(['true'], timeout=15)).exit_code == 0
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        try:
            await native.terminate.aio()
        finally:
            await native.detach.aio()  # pyright: ignore[reportUnknownMemberType]


async def test_cancel_kills_term_ignoring_child() -> None:
    marker = uuid.uuid4().hex
    async with owned_backend() as backend:
        script = (
            'import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            f'pathlib.Path("/tmp/{marker}.ready").touch(); time.sleep(4); '
            f'pathlib.Path("/tmp/{marker}.late").touch()'
        )
        task = asyncio.create_task(backend.run(['python', '-c', script], timeout=None))
        try:
            with anyio.fail_after(30):
                while not await backend.exists(f'/tmp/{marker}.ready'):
                    await anyio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await anyio.sleep(5)
            assert not await backend.exists(f'/tmp/{marker}.late')
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


async def test_real_command_and_filesystem() -> None:
    async with owned_backend() as backend:
        result = await backend.run(['sh', '-c', 'printf out; printf err >&2; exit 3'], timeout=30)
        assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 3)
        workspace = Workspace(backend)
        await workspace.write_text('/tmp/modal-sandbox.txt', 'content')
        assert await workspace.read_text('/tmp/modal-sandbox.txt') == 'content'


async def test_real_timeout_retains_output() -> None:
    async with owned_backend() as backend:
        await backend.run(['true'], timeout=30)
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sh', '-c', 'echo diagnostic; sleep 30'], timeout=2)
        assert 'diagnostic' in (exc_info.value.stdout or '')


async def test_reattach_sees_state_and_applies_working_dir_and_env() -> None:
    """An attached backend reuses the sandbox's files, and its own `working_dir` and `env` reach commands."""
    marker = f'/tmp/{uuid.uuid4().hex}.txt'
    async with owned_backend() as owner:
        await owner.write_bytes(marker, b'shared')
        attached = ModalSandboxBackend(ref=owner.ref, working_dir='/tmp', env={'PROBE': 'attached'})
        assert await attached.read_bytes(marker) == b'shared'
        result = await attached.run('printf "%s %s" "$(pwd)" "$PROBE"', shell=True, timeout=30)
        assert result.stdout == '/tmp attached'


async def test_a_terminated_sandbox_is_unavailable() -> None:
    async with owned_backend() as owner:
        await (await owner.get_sandbox()).terminate.aio()
        with pytest.raises(WorkspaceUnavailableError):
            await ModalSandboxBackend(ref=owner.ref).run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError):
            await owner.run(['true'], timeout=30)


async def test_an_expired_sandbox_is_unavailable() -> None:
    """A sandbox past its `sandbox_timeout` is gone, both to its owner and to a reattach."""
    async with owned_backend(sandbox_timeout=10) as owner:
        await anyio.sleep(30)
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await owner.run(['true'], timeout=30)
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await ModalSandboxBackend(ref=owner.ref).run(['true'], timeout=30)


async def test_coder_tools_run_in_the_sandbox_modal_sandbox_supplies() -> None:
    """`ModalSandbox` supplies the workspace, and `Coder`'s tools run in it."""
    supplied: list[ModalSandboxBackend] = []

    class RecordingModalSandbox(ModalSandbox[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            assert isinstance(backend, ModalSandboxBackend)
            supplied.append(backend)
            return backend

    try:
        results = await call_tools(
            [
                RecordingModalSandbox(
                    working_dir='/tmp',
                    sandbox_timeout=LIVE_SANDBOX_TIMEOUT,
                    idle_timeout=LIVE_IDLE_TIMEOUT,
                ),
                Coder(),
            ],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hello from modal')\n"}),
                ('shell', {'command': 'python hello.py && git --version'}),
                ('grep', {'pattern': 'hello from modal'}),
            ],
        )
        assert 'hello from modal' in results[1]
        assert 'git version' in results[1]
        assert 'hello.py' in results[2]
        # Core may ask for the workspace again after `for_run` only to compare it; that backend is
        # discarded unused, so exactly one of them created a sandbox.
        assert sum(backend.ref is not None for backend in supplied) == 1
    finally:
        for backend in supplied:
            if backend.ref is not None:
                await (await backend.get_sandbox()).terminate.aio()


async def test_commands_get_a_usable_environment() -> None:
    """With no `image` or `env`, a command sees `PATH` and `HOME` and finds git and ripgrep; `env=` adds to them."""
    async with owned_backend() as backend:
        result = await backend.run('echo "$PATH"; echo "$HOME"; git --version; rg --version', shell=True, timeout=60)
        path, home, git, rg, *_ = result.stdout.splitlines()
        assert path and home and git.startswith('git version') and rg.startswith('ripgrep')

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=30)
        assert result.stdout.splitlines() == ['1', path]


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/modal-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    A follow-up run, from the message history or a stored ref, works in the first run's sandbox.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'terminate_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
