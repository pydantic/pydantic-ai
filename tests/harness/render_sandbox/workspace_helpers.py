"""Reusable fake Render SDK and local-shell harness for workspace tests.

Normal mode preserves a caller-selected local working directory. Isolation mode maps
remote sandbox identities to distinct local directories, while still executing the
real Workspace filesystem fallback in subprocesses. Deliberate sharing flags are
negative controls used to prove that integration assertions detect leaked state.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from render import RenderAsync
from render.experimental.sandbox import (
    Sandbox,
    SandboxExecEvent,
    SandboxExecExit,
    SandboxExecOutput,
    SandboxNotFoundError,
)

from pydantic_ai.workspaces import CommandResult, WorkspaceCommand, WorkspaceRef
from pydantic_ai_harness.render_sandbox import RenderSandboxBackend, RenderSandboxConfig


@dataclass(frozen=True)
class CreateCall:
    timeout_seconds: int
    network_policy: str
    env: dict[str, str]


@dataclass(frozen=True)
class DispatchCall:
    sandbox_id: str
    command: WorkspaceCommand
    working_dir: str
    env: dict[str, str]


@dataclass
class WorkspaceHarness:
    client: RenderAsync
    config: RenderSandboxConfig
    root: Path
    isolated_filesystems: bool = False
    share_filesystem: bool = False
    share_environment: bool = False
    create_calls: dict[str, CreateCall] = field(default_factory=dict[str, CreateCall])
    dispatch_calls: list[DispatchCall] = field(default_factory=list[DispatchCall])
    sandbox_roots: dict[str, Path] = field(default_factory=dict[str, Path])
    sandboxes: dict[str, Sandbox] = field(default_factory=dict[str, Sandbox])
    created: list[str] = field(default_factory=list[str])
    terminated: list[str] = field(default_factory=list[str])
    transfers: list[tuple[str, str, str, str]] = field(default_factory=list[tuple[str, str, str, str]])
    transfer_error: Exception | None = None
    transfer_started: asyncio.Event = field(default_factory=asyncio.Event)
    transfer_gate: asyncio.Event | None = None
    commands: list[WorkspaceCommand] = field(default_factory=list[WorkspaceCommand])
    create_error: Exception | None = None
    command_error: Exception | None = None
    terminate_error: Exception | None = None
    create_gate: asyncio.Event | None = None
    create_started: asyncio.Event = field(default_factory=asyncio.Event)
    command_started: asyncio.Event = field(default_factory=asyncio.Event)
    block_commands: bool = False
    create_status: str = 'running'
    terminate_gate: asyncio.Event | None = None
    terminate_started: asyncio.Event = field(default_factory=asyncio.Event)

    execute: Callable[..., Awaitable[CommandResult]] = field(init=False)

    def backend(self, ref: WorkspaceRef | None = None) -> RenderSandboxBackend:
        return RenderSandboxBackend(self.client, config=self.config, ref=ref)


def make_workspace_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, isolated_filesystems: bool = False
) -> WorkspaceHarness:
    harness = WorkspaceHarness(
        RenderAsync(token='test-only', owner_id='tea-test'),
        RenderSandboxConfig(working_dir=str(tmp_path), cleanup_timeout=2, stop_timeout=1, command_interval=0),
        root=tmp_path,
        isolated_filesystems=isolated_filesystems,
    )

    create, from_id, terminate = lifecycle(harness)
    copy_to, copy_from = transfers(harness)
    harness.execute = commands(harness)
    execute = stream(harness)
    sdk = harness.client.experimental.sandboxes
    monkeypatch.setattr(sdk, 'copy_to', copy_to)
    monkeypatch.setattr(sdk, 'copy_from', copy_from)
    monkeypatch.setattr(sdk, 'create', create)
    monkeypatch.setattr(sdk, 'from_id', from_id)
    monkeypatch.setattr(sdk, 'terminate', terminate)
    monkeypatch.setattr(sdk, 'exec', execute)
    return harness


def lifecycle(
    harness: WorkspaceHarness,
) -> tuple[Callable[..., Awaitable[Sandbox]], Callable[[str], Awaitable[Sandbox]], Callable[[str], Awaitable[None]]]:
    async def create(*, timeout_seconds: int, network_policy: str, env: dict[str, str] | None) -> Sandbox:
        harness.create_started.set()
        if harness.create_gate is not None:
            await harness.create_gate.wait()
        if harness.create_error is not None:
            raise harness.create_error
        identifier = f'sb-{len(harness.created) + 1}'
        sandbox = Sandbox(
            identifier,
            harness.create_status,
            'starter',
            'deny-all',
            'oregon',
            300,
            datetime.now(UTC),
        )
        harness.create_calls[identifier] = CreateCall(timeout_seconds, network_policy, dict(env or {}))
        harness.sandbox_roots[identifier] = harness.root / (
            'shared-sandbox' if harness.share_filesystem else identifier
        )
        harness.sandboxes[identifier] = sandbox
        harness.created.append(identifier)
        return sandbox

    async def from_id(identifier: str) -> Sandbox:
        if identifier not in harness.sandboxes:
            raise SandboxNotFoundError('missing')
        return harness.sandboxes[identifier]

    async def terminate(identifier: str) -> None:
        harness.terminate_started.set()
        if harness.terminate_gate is not None:
            await harness.terminate_gate.wait()
        if harness.terminate_error is not None:
            raise harness.terminate_error
        harness.terminated.append(identifier)
        harness.sandboxes[identifier].status = 'terminated'

    return create, from_id, terminate


def transfers(harness: WorkspaceHarness) -> tuple[Callable[..., Awaitable[None]], Callable[..., Awaitable[str]]]:
    async def transfer(direction: str, identifier: str, source: str, target: str) -> None:
        harness.transfers.append((direction, identifier, source, target))
        harness.transfer_started.set()
        if harness.transfer_gate is not None:
            await harness.transfer_gate.wait()
        if harness.transfer_error is not None:
            raise harness.transfer_error
        if harness.sandboxes[identifier].status != 'running':
            raise SandboxNotFoundError('not running')
        # Atomic replacement matches Render's file endpoint. Following a symlink
        # here would hide an adapter bug, so explicitly replace the leaf itself.
        temporary = Path(target + '.sdk-tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)

    async def copy_to(identifier: str, local: Path, remote: str) -> None:
        await transfer('upload', identifier, str(local), remote)

    async def copy_from(identifier: str, remote: str, local: Path) -> str:
        await transfer('download', identifier, remote, str(local))
        return str(local)

    return copy_to, copy_from


def commands(harness: WorkspaceHarness) -> Callable[..., Awaitable[CommandResult]]:
    async def command(
        client: RenderAsync,
        sandbox_id: str,
        command: WorkspaceCommand,
        *,
        working_dir: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 1_048_576,
        stop_timeout: float = 12.0,
    ) -> CommandResult:
        harness.commands.append(command)
        harness.dispatch_calls.append(DispatchCall(sandbox_id, command, working_dir, dict(env or {})))
        if harness.sandboxes[sandbox_id].status != 'running':
            raise SandboxNotFoundError('not running')
        if working_dir != '/':
            harness.command_started.set()
            if harness.command_error is not None:
                raise harness.command_error
            if harness.block_commands:
                await asyncio.Future[None]()
        if harness.isolated_filesystems and working_dir == '/':
            # Readiness asks the provider for its canonical cwd. Each fake sandbox has
            # its own real local root; later paths come from this public cwd result.
            root = harness.sandbox_roots[sandbox_id]
            root.mkdir(parents=True, exist_ok=True)
            return CommandResult(exit_code=0, stdout=str(root) + '\n', stderr='')
        configured = harness.create_calls[sandbox_id].env
        effective_env = {**configured, **dict(env or {})}
        if harness.share_environment:
            # Fault injection is limited to this test-owned setting. Preserve all
            # other per-command environment fields so unrelated behavior stays valid.
            effective_env['RUN_LABEL'] = next(iter(harness.create_calls.values())).env['RUN_LABEL']
        values = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
        values.update(effective_env)
        argv = ['/bin/sh', '-c', command] if isinstance(command, str) else list(command)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=working_dir,
            env=values,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(timeout):
                stdout, stderr = await process.communicate()
        except BaseException:
            process.kill()
            await process.wait()
            raise
        assert process.returncode is not None
        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    return command


def stream(harness: WorkspaceHarness) -> Callable[[str, str], AsyncGenerator[SandboxExecEvent]]:
    async def execute(identifier: str, launch: str) -> AsyncGenerator[SandboxExecEvent]:
        args = shlex.split(launch)
        config = json.loads(args[4])
        nonce = config['nonce']

        def frame(kind: str, **fields: object) -> SandboxExecOutput:
            return SandboxExecOutput('stdout', json.dumps({'nonce': nonce, 'kind': kind, **fields}) + '\n')

        if args[3] == 'stop':
            yield frame('stopped')
            yield SandboxExecExit(0)
            return
        argv = config['argv']
        invocation = argv[2] if argv[:2] == ['/bin/sh', '-c'] else argv
        result = await harness.execute(
            harness.client,
            identifier,
            invocation,
            working_dir=config['cwd'],
            env=config['env'],
            timeout=config['timeout'],
            max_output_bytes=config['limit'],
        )
        for stream, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
            for start in range(0, len(value), 16384):
                yield frame(
                    'output', stream=stream, data=base64.b64encode(value[start : start + 16384].encode()).decode()
                )
        yield frame('exit', code=result.exit_code)
        yield SandboxExecExit(0)

    return execute
