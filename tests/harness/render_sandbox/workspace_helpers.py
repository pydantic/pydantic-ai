"""A fake Render SDK whose sandboxes are local directories.

User commands run as local subprocesses, or with `supervised` under the shipped supervisor in a child
interpreter, as Render runs them; the supervised transport also offers the knobs the supervisor tests use. The shipped file helper runs in-process, so its coverage needs no child
interpreter; `inline_helpers` runs it as Render does instead, under `python3 -I -c` in the command's
directory. Isolation mode gives each fake sandbox its own root.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import runpy
import shlex
import shutil
import signal
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import anyio
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
from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

_BACKEND_MODULE = 'pydantic_ai_harness.render_sandbox._backend'
FILE_HELPER = files('pydantic_ai_harness.render_sandbox').joinpath('_file_transfer.py')
SUPERVISOR = files('pydantic_ai_harness.render_sandbox').joinpath('_command_runner.py')


def frame(kind: str, **fields: object) -> SandboxExecOutput:
    """One supervisor control frame, as the shipped supervisor prints it."""
    return SandboxExecOutput('stdout', json.dumps({'kind': kind, **fields}) + '\n')


def output(text: str, stream: str = 'stdout') -> SandboxExecOutput:
    """A supervisor output frame carrying `text` on `stream`."""
    return frame('output', stream=stream, data=base64.b64encode(text.encode()).decode())


@dataclass(frozen=True)
class CreateCall:
    timeout_seconds: int
    network_policy: str


@dataclass
class WorkspaceHarness:
    client: RenderAsync
    root: Path
    monkeypatch: pytest.MonkeyPatch
    isolated_filesystems: bool = False
    create_calls: dict[str, CreateCall] = field(default_factory=dict[str, CreateCall])
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
    create_status: str = 'running'
    create_gate: asyncio.Event | None = None
    create_started: asyncio.Event = field(default_factory=asyncio.Event)
    inline_helpers: bool = False
    # Called inside an in-process file helper run with a scoped monkeypatch and the helper's action.
    helper_fault: Callable[[pytest.MonkeyPatch, str], None] | None = None
    opened_clients: list[RenderAsync] = field(default_factory=list[RenderAsync])
    # Run user commands under the shipped supervisor in a child interpreter, as Render does.
    supervised: bool = False
    supervisors: list[tuple[str, dict[str, Any], asyncio.subprocess.Process]] = field(
        default_factory=list[tuple[str, dict[str, Any], asyncio.subprocess.Process]]
    )
    # Supervised transport knobs. `output` receives each user command's output bytes as they stream.
    output: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue[bytes])
    # Variables the sandbox itself has, which the backend's `env` does not supply.
    sandbox_env: dict[str, str] = field(default_factory=dict[str, str])
    # Measure the supervisor child's coverage; off by default to keep the conformance suite cheap.
    measure_supervisor: bool = False
    # Run the launch exactly as Render receives it (`python3 -I -c <source>`) in this directory.
    inline_cwd: Path | None = None
    # Lose the next run's start response before launching it, keeping its launch for a later replay.
    delay_next: bool = False
    delayed: str | None = None
    # Hold a stop until every running command has exited, as a delayed provider stop would.
    drain_before_stop: bool = False

    execute: Callable[..., Awaitable[CommandResult]] = field(init=False)
    install: Callable[[RenderAsync], None] = field(init=False)

    async def wait_supervisors(self) -> None:
        # Every test command either finishes or receives the backend's stop first.
        async with asyncio.timeout(10):
            for _, _, process in self.supervisors:
                await process.wait()

    async def aclose(self) -> None:
        """Reap every supervisor and remove its claim; the tests stop each command they leave running."""
        for _, config, process in self.supervisors:
            if process.returncode is None:  # pragma: lax no cover - a stop may still be finishing
                process.kill()
            await process.communicate()
            shutil.rmtree(config['claim'], ignore_errors=True)

    def backend(self, ref: WorkspaceRef | None = None, **settings: Any) -> RenderSandboxBackend:
        settings.setdefault('working_dir', str(self.root))
        return RenderSandboxBackend(client=self.client, ref=ref, **settings)


def make_workspace_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, isolated_filesystems: bool = False
) -> WorkspaceHarness:
    # Short bounds keep a regressed stop or readiness wait from stalling the test; no test waits them out.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._STOP_TIMEOUT', 1)
    monkeypatch.setattr(f'{_BACKEND_MODULE}._POLL_INTERVAL', 0)
    # A supervised command returns this long after its direct child exits if a background child holds a pipe.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._DRAIN_GRACE', 0.2)
    harness = WorkspaceHarness(
        RenderAsync(token='test-only', owner_id='tea-test'),
        root=tmp_path,
        monkeypatch=monkeypatch,
        isolated_filesystems=isolated_filesystems,
    )

    create, from_id, terminate = lifecycle(harness)
    copy_to, copy_from = transfers(harness)
    harness.execute = commands(harness)
    execute = stream(harness)

    def install(client: RenderAsync) -> None:
        sdk = client.experimental.sandboxes
        monkeypatch.setattr(sdk, 'copy_to', copy_to)
        monkeypatch.setattr(sdk, 'copy_from', copy_from)
        monkeypatch.setattr(sdk, 'create', create)
        monkeypatch.setattr(sdk, 'from_id', from_id)
        monkeypatch.setattr(sdk, 'terminate', terminate)
        monkeypatch.setattr(sdk, 'exec', execute)

    harness.install = install
    install(harness.client)
    return harness


def use_host_credentials(harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let backends without `client=` open their own clients, each wired to this harness's fake SDK."""
    monkeypatch.setenv('RENDER_API_KEY', 'test-only')
    monkeypatch.setenv('RENDER_WORKSPACE_ID', 'tea-test')

    def open_client() -> RenderAsync:
        client = RenderAsync(token='test-only', owner_id='tea-test')
        harness.install(client)
        harness.opened_clients.append(client)
        return client

    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands.RenderAsync', open_client)


def client_is_closed(client: RenderAsync) -> bool:
    return client.experimental.sandboxes.client.get_async_httpx_client().is_closed


def lifecycle(
    harness: WorkspaceHarness,
) -> tuple[Callable[..., Awaitable[Sandbox]], Callable[[str], Awaitable[Sandbox]], Callable[[str], Awaitable[None]]]:
    async def create(*, timeout_seconds: int, network_policy: str) -> Sandbox:
        harness.create_started.set()
        if harness.create_gate is not None:
            await harness.create_gate.wait()
        if harness.create_error is not None:
            raise harness.create_error
        identifier = f'sb-{len(harness.created) + 1}'
        sandbox = Sandbox(identifier, harness.create_status, 'starter', 'deny-all', 'oregon', 300, datetime.now(UTC))
        harness.create_calls[identifier] = CreateCall(timeout_seconds, network_policy)
        harness.sandbox_roots[identifier] = harness.root / identifier
        harness.sandboxes[identifier] = sandbox
        harness.created.append(identifier)
        return sandbox

    async def from_id(identifier: str) -> Sandbox:
        if identifier not in harness.sandboxes:
            raise SandboxNotFoundError('missing')
        return harness.sandboxes[identifier]

    async def terminate(identifier: str) -> None:
        if harness.terminate_error is not None:
            raise harness.terminate_error
        harness.terminated.append(identifier)
        harness.sandboxes[identifier].status = 'terminated'
        for sandbox_id, config, process in harness.supervisors:
            if sandbox_id == identifier and process.returncode is None:
                # Termination ends everything in the sandbox: the command's process group and its supervisor.
                with suppress(FileNotFoundError, KeyError, ProcessLookupError):
                    os.killpg(json.loads((Path(config['claim']) / 'state.json').read_text())['pid'], signal.SIGKILL)
                process.kill()
        if harness.isolated_filesystems:
            # Termination deletes the sandbox's filesystem; only isolated roots belong to one sandbox.
            shutil.rmtree(harness.sandbox_roots[identifier], ignore_errors=True)

    return create, from_id, terminate


def transfers(harness: WorkspaceHarness) -> tuple[Callable[..., Awaitable[None]], Callable[..., Awaitable[str]]]:
    async def transfer(direction: str, identifier: str, source: str, target: str) -> None:
        harness.transfers.append((direction, identifier, source, target))
        harness.transfer_started.set()
        if harness.transfer_gate is not None:
            await harness.transfer_gate.wait()
        if harness.transfer_error is not None:
            raise harness.transfer_error
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


def _run_file_helper(harness: WorkspaceHarness, args: list[str]) -> CommandResult:
    """Run the shipped file helper in this process, as its `__main__`."""
    output = io.StringIO()
    exit_code = 0
    with harness.monkeypatch.context() as child:
        child.setattr(sys, 'argv', [str(FILE_HELPER), *args])
        if harness.helper_fault is not None:
            harness.helper_fault(child, args[0])
        with redirect_stdout(output):
            try:
                runpy.run_path(str(FILE_HELPER), run_name='__main__')
            except SystemExit as error:
                assert isinstance(error.code, int)
                exit_code = error.code
    return CommandResult(exit_code=exit_code, stdout=output.getvalue(), stderr='')


def commands(harness: WorkspaceHarness) -> Callable[..., Awaitable[CommandResult]]:
    async def command(
        sandbox_id: str,
        command: WorkspaceCommand,
        *,
        working_dir: str,
        env: Mapping[str, str],
    ) -> CommandResult:
        harness.commands.append(command)
        if harness.sandboxes[sandbox_id].status != 'running':
            raise SandboxNotFoundError('not running')
        if working_dir == '/':
            if harness.isolated_filesystems:
                # Readiness asks for the canonical cwd; each isolated sandbox has its own local root.
                root = harness.sandbox_roots[sandbox_id]
                root.mkdir(parents=True, exist_ok=True)
                return CommandResult(exit_code=0, stdout=str(root) + '\n', stderr='')
        elif harness.command_error is not None:
            raise harness.command_error
        argv = ['/bin/sh', '-c', command] if isinstance(command, str) else list(command)
        helper = argv[:4] == ['python3', '-I', '-c', FILE_HELPER.read_text()]
        if helper and not harness.inline_helpers:
            return _run_file_helper(harness, argv[4:])
        if helper:
            argv = [sys.executable, *argv[1:]]
        values = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
        values.update(env)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=working_dir,
            env=values,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            process.kill()
            # Draining after the kill closes the pipe transports, so none is left for the garbage collector.
            with anyio.CancelScope(shield=True):
                await process.communicate()
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
        operation, config = args[4], json.loads(args[5])
        argv = config.get('argv', [])
        helper = argv[:4] == ['python3', '-I', '-c', FILE_HELPER.read_text()]
        if harness.supervised and not (operation == 'run' and (config['cwd'] == '/' or helper)):
            async for event in _supervise(harness, identifier, launch):
                yield event
            return
        if operation == 'stop':
            yield frame('stopped')
            yield SandboxExecExit(0)
            return
        invocation = argv[2] if argv[:2] == ['/bin/sh', '-c'] else argv
        result = await harness.execute(identifier, invocation, working_dir=config['cwd'], env=config['env'])
        for name, value in [('stdout', result.stdout), ('stderr', result.stderr)]:
            for start in range(0, len(value), 16384):
                yield output(value[start : start + 16384], name)
        yield frame('exit', code=result.exit_code)
        yield SandboxExecExit(0)

    return execute


async def _supervise(harness: WorkspaceHarness, identifier: str, launch: str) -> AsyncGenerator[SandboxExecEvent]:
    """Run the shipped supervisor in a child interpreter with the sandbox's own environment."""
    args = shlex.split(launch)
    operation, config = args[4], json.loads(args[5])
    if operation == 'run' and harness.delay_next:
        harness.delay_next = False
        harness.delayed = launch
        raise ConnectionError('start response lost before the server launches the command')
    if operation == 'stop' and harness.drain_before_stop:
        await asyncio.gather(*(process.wait() for _, _, process in harness.supervisors))
    env = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
    if harness.measure_supervisor and 'COVERAGE_PROCESS_CONFIG' in os.environ:
        env['COVERAGE_PROCESS_CONFIG'] = os.environ['COVERAGE_PROCESS_CONFIG']
    env.update(harness.sandbox_env)
    # Inline mode runs the received source as Render does; otherwise the package file, so coverage has a path.
    argv = [sys.executable, *args[1:]] if harness.inline_cwd else [sys.executable, str(SUPERVISOR), *args[4:]]
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=harness.inline_cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Render keeps an exec running after the client disconnects; the backend's stop or `aclose` ends it.
    harness.supervisors.append((identifier, config, process))
    assert process.stdout is not None and process.stderr is not None
    while line := await process.stdout.readline():
        frame = json.loads(line)
        if operation == 'run' and frame['kind'] == 'output':
            harness.output.put_nowait(base64.b64decode(frame['data']))
        yield SandboxExecOutput('stdout', line.decode())
    stderr = await process.stderr.read()
    exit_code = await process.wait()
    if harness.sandboxes[identifier].status != 'running':
        # Render ends the exec stream with an error when the sandbox is terminated mid-command.
        raise SandboxNotFoundError('terminated during command')
    assert not stderr, stderr.decode()
    yield SandboxExecExit(exit_code)
