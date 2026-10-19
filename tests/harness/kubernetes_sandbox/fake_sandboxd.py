"""An in-process sandboxd and agent-sandbox SDK fake for `KubernetesSandboxBackend` tests.

`FakeSandboxd` serves sandboxd's `process.v1.ProcessService` `Start` and `WriteStdin` over a real
`grpc.aio` server and runs each command as a host process in its root directory, with sandboxd's
semantics: the command gets its own process group, which is SIGKILLed when the call is cancelled;
stdin stays open until `WriteStdin(eof)`; a program that cannot start fails with `NOT_FOUND`; a
signal death exits 128+n. Messages are built with protobuf from a descriptor matching
`process.proto`, independently of the backend's hand-written encoding, so the two check each other.

`FakeCluster` stands in for the SDK's `AsyncSandboxClient`: each claim gets its own `FakeSandboxd`
and root directory, and deleting a claim stops its server.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import grpc
import grpc.aio
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_GRACE = 0.3
"""How long output pipes may stay open after the command exits (sandboxd waits about 5 seconds)."""


def _messages() -> dict[str, Any]:
    F = descriptor_pb2.FieldDescriptorProto
    file = descriptor_pb2.FileDescriptorProto(
        name='fake_sandboxd/process.proto', package='fake.process.v1', syntax='proto3'
    )

    def message(name: str, *fields: tuple[str, int, int, int, str]) -> descriptor_pb2.DescriptorProto:
        proto = file.message_type.add(name=name)
        for field_name, number, kind, label, type_name in fields:
            added = proto.field.add(name=field_name, number=number, type=kind, label=label)
            if type_name:
                added.type_name = type_name
        return proto

    opt, rep = F.LABEL_OPTIONAL, F.LABEL_REPEATED
    config = message(
        'ProcessConfig',
        ('command', 1, F.TYPE_STRING, rep, ''),
        ('env_vars', 2, F.TYPE_MESSAGE, rep, '.fake.process.v1.ProcessConfig.EnvVarsEntry'),
        ('cwd', 3, F.TYPE_STRING, opt, ''),
    )
    entry = config.nested_type.add(name='EnvVarsEntry')
    entry.options.map_entry = True
    entry.field.add(name='key', number=1, type=F.TYPE_STRING, label=opt)
    entry.field.add(name='value', number=2, type=F.TYPE_STRING, label=opt)
    message('StartRequest', ('config', 1, F.TYPE_MESSAGE, opt, '.fake.process.v1.ProcessConfig'))
    message('InitEvent', ('process_id', 1, F.TYPE_INT32, opt, ''))
    message('ExitEvent', ('exit_code', 1, F.TYPE_INT32, opt, ''))
    response = message(
        'StartResponse',
        ('init', 1, F.TYPE_MESSAGE, opt, '.fake.process.v1.InitEvent'),
        ('stdout', 2, F.TYPE_BYTES, opt, ''),
        ('stderr', 3, F.TYPE_BYTES, opt, ''),
        ('exit', 4, F.TYPE_MESSAGE, opt, '.fake.process.v1.ExitEvent'),
    )
    response.oneof_decl.add(name='event')
    for proto_field in response.field:
        proto_field.oneof_index = 0
    message('Empty')
    stdin = message(
        'WriteStdinRequest',
        ('process_id', 1, F.TYPE_INT32, opt, ''),
        ('input', 2, F.TYPE_BYTES, opt, ''),
        ('eof', 3, F.TYPE_MESSAGE, opt, '.fake.process.v1.Empty'),
    )
    stdin.oneof_decl.add(name='payload')
    stdin.field[1].oneof_index = 0
    stdin.field[2].oneof_index = 0
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)  # pyright: ignore[reportUnknownMemberType]
    names = ('StartRequest', 'StartResponse', 'WriteStdinRequest')
    return {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f'fake.process.v1.{name}'))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        for name in names
    }


MESSAGES = _messages()


def _serialize(message: Any) -> bytes:
    return message.SerializeToString()


def _raw(data: bytes) -> bytes:
    return data


class FakeSandboxd:
    """A sandboxd `ProcessService` serving host processes in `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.server: grpc.aio.Server | None = None
        self.target = ''
        self._processes: dict[int, _Process] = {}
        self._next_id = 0
        self.stdin_error: grpc.StatusCode | None = None
        """When set, `WriteStdin` fails with this code, as sandboxd's `INTERNAL: ... file already closed` race does."""

    async def start(self) -> None:
        self.server = grpc.aio.server()
        handlers: dict[str, grpc.RpcMethodHandler[Any, Any]] = {
            'Start': grpc.unary_stream_rpc_method_handler(self._start, MESSAGES['StartRequest'].FromString, _serialize),
            'WriteStdin': grpc.unary_unary_rpc_method_handler(
                self._write_stdin, MESSAGES['WriteStdinRequest'].FromString, _raw
            ),
        }
        self.server.add_generic_rpc_handlers(
            [grpc.method_handlers_generic_handler('process.v1.ProcessService', handlers)]
        )
        port = self.server.add_insecure_port('127.0.0.1:0')
        self.target = f'127.0.0.1:{port}'
        await self.server.start()

    async def stop(self) -> None:
        for process in list(self._processes.values()):
            process.kill_group()
        assert self.server is not None
        await self.server.stop(grace=0)

    async def _start(self, request: Any, context: grpc.aio.ServicerContext[Any, Any]) -> AsyncIterator[Any]:
        Response = MESSAGES['StartResponse']
        env = {**os.environ, **dict(request.config.env_vars)}
        try:
            process = await _Process.spawn(list(request.config.command), cwd=self.root, env=env)
        except FileNotFoundError as error:
            await context.abort(
                grpc.StatusCode.NOT_FOUND, f'failed to start command: command or path not found: {error}'
            )
        self._next_id += 1
        process_id = self._next_id
        self._processes[process_id] = process
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def pump(stream: asyncio.StreamReader, name: str) -> None:
            try:
                while chunk := await stream.read(4096):
                    await queue.put(Response(**{name: chunk}))
            finally:
                queue.put_nowait(None)  # End of this stream, also when cut off below.

        pumps = [
            asyncio.ensure_future(pump(process.stdout, 'stdout')),
            asyncio.ensure_future(pump(process.stderr, 'stderr')),
        ]
        waiter = asyncio.ensure_future(process.wait())

        async def cut_off_after_exit() -> None:
            # Like sandboxd's wait delay: stop reading pipes a background child keeps open.
            await waiter
            _, pending = await asyncio.wait(pumps, timeout=_GRACE)
            for task in pending:
                task.cancel()

        finisher = asyncio.ensure_future(cut_off_after_exit())
        try:
            yield Response(init={'process_id': process_id})
            ended = 0
            while ended < 2:
                event = await queue.get()
                if event is None:
                    ended += 1
                else:
                    yield event
            code = await waiter
            yield Response(exit={'exit_code': 128 - code if code < 0 else code})
        finally:
            # Reached on cancellation too, which is how sandboxd stops a command whose call ends.
            if not waiter.done():
                process.kill_group()
            for task in (*pumps, finisher):
                task.cancel()
            process.close()
            self._processes.pop(process_id, None)

    async def _write_stdin(self, request: Any, context: grpc.aio.ServicerContext[Any, Any]) -> bytes:
        if self.stdin_error is not None:
            await context.abort(self.stdin_error, 'failed to close stdin: close |1: file already closed')
        process = self._processes.get(request.process_id)
        if process is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f'process {request.process_id} not found')
        assert request.WhichOneof('payload') == 'eof'
        process.close_stdin()
        return b''


class _Process:
    """A command started with `posix_spawn` in its own session, so its process group can be killed.

    Not `asyncio.create_subprocess_exec`: that forks, and gRPC's fork handlers (this process runs a
    gRPC server) write log lines into the child's stderr before it execs.
    """

    def __init__(self, pid: int, stdin: int, stdout: asyncio.StreamReader, stderr: asyncio.StreamReader) -> None:
        self.pid = pid
        self._stdin: int | None = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._transports: list[asyncio.BaseTransport] = []

    @classmethod
    async def spawn(cls, argv: list[str], *, cwd: Path, env: dict[str, str]) -> _Process:
        program = argv[0] if '/' in argv[0] else shutil.which(argv[0], path=env.get('PATH'))
        if program is None or not os.access(program, os.X_OK):
            raise FileNotFoundError(argv[0])
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        # `posix_spawn` has no working directory argument before Python 3.13, so a shell enters it.
        pid = os.posix_spawn(
            '/bin/sh',
            ['sh', '-c', 'cd "$0" && exec "$@"', str(cwd), program, *argv[1:]],
            env,
            file_actions=[
                (os.POSIX_SPAWN_DUP2, stdin_r, 0),
                (os.POSIX_SPAWN_DUP2, stdout_w, 1),
                (os.POSIX_SPAWN_DUP2, stderr_w, 2),
            ],
            setsid=True,
        )
        for fd in (stdin_r, stdout_w, stderr_w):
            os.close(fd)
        loop = asyncio.get_running_loop()
        process = cls(pid, stdin_w, asyncio.StreamReader(), asyncio.StreamReader())
        for fd, reader in ((stdout_r, process.stdout), (stderr_r, process.stderr)):
            transport, _ = await loop.connect_read_pipe(
                lambda reader=reader: asyncio.StreamReaderProtocol(reader), os.fdopen(fd, 'rb', 0)
            )
            process._transports.append(transport)
        return process

    async def wait(self) -> int:
        _, status = await asyncio.to_thread(os.waitpid, self.pid, 0)
        return os.waitstatus_to_exitcode(status)

    def close_stdin(self) -> None:
        if self._stdin is not None:
            os.close(self._stdin)
            self._stdin = None

    def kill_group(self) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)

    def close(self) -> None:
        self.close_stdin()
        for transport in self._transports:
            transport.close()


@dataclass
class FakeSandbox:
    """The SDK's `AsyncSandbox`, as much of it as the backend uses."""

    claim_name: str | None
    namespace: str
    sandboxd: FakeSandboxd
    cluster: FakeCluster
    connector: FakeConnector = field(init=False)

    def __post_init__(self) -> None:
        self.connector = FakeConnector(self)

    async def status(self) -> tuple[str, str]:
        if self.cluster.status_error is not None:
            raise self.cluster.status_error
        if self.cluster.claims.get((self.namespace, self.claim_name or '')) is not self.sandboxd:
            return 'SandboxNotFound', 'Sandbox object not found in Kubernetes.'
        return 'SandboxReady', ''

    async def terminate(self) -> None:
        await self.cluster.delete((self.namespace, self.claim_name or ''))
        self.claim_name = None


class FakeConnector:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.channel: grpc.aio.Channel | None = None

    async def grpc_channel(self) -> grpc.aio.Channel:
        if self.sandbox.cluster.channel_error is not None:
            raise self.sandbox.cluster.channel_error
        if self.channel is None:
            self.channel = grpc.aio.insecure_channel(self.sandbox.sandboxd.target)
        return self.channel


class FakeCluster:
    """Claims, each with its own `FakeSandboxd` in a directory under `base`."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.claims: dict[tuple[str, str], FakeSandboxd] = {}
        self.pools = {'pool'}
        self.clients: list[FakeClient] = []
        self.created: list[tuple[str, str, dict[str, Any]]] = []
        self.status_error: Exception | None = None
        self.channel_error: Exception | None = None
        self.create_error: Exception | None = None
        self.create_gate: asyncio.Event | None = None
        """When set, `create_sandbox` creates the claim, then waits for it (to test cancellation mid-create)."""

    def root(self, namespace: str, claim_name: str) -> Path:
        return (self.base / namespace / claim_name).resolve()

    async def delete(self, key: tuple[str, str]) -> None:
        await self.claims.pop(key).stop()

    async def close(self) -> None:
        for key in list(self.claims):
            await self.delete(key)

    def client(self, connection: str) -> FakeClient:
        client = FakeClient(self, connection)
        self.clients.append(client)
        return client


class FakeClient:
    """The SDK's `AsyncSandboxClient`."""

    def __init__(self, cluster: FakeCluster, connection: str) -> None:
        self.cluster = cluster
        self.connection = connection
        self.closed = False
        self.sandboxes: list[FakeSandbox] = []

    async def create_sandbox(self, warmpool: str, namespace: str = 'default', **kwargs: Any) -> FakeSandbox:
        from k8s_agent_sandbox.exceptions import SandboxWarmPoolNotFoundError

        if self.cluster.create_error is not None:
            raise self.cluster.create_error
        if warmpool not in self.cluster.pools:
            raise SandboxWarmPoolNotFoundError(f"SandboxWarmPool '{warmpool}' not found in namespace '{namespace}'")
        claim_name = kwargs['claim_name']
        root = self.cluster.root(namespace, claim_name)
        root.mkdir(parents=True)
        sandboxd = FakeSandboxd(root)
        await sandboxd.start()
        self.cluster.claims[namespace, claim_name] = sandboxd
        self.cluster.created.append((warmpool, namespace, kwargs))
        if self.cluster.create_gate is not None:
            await self.cluster.create_gate.wait()
        return self._handle(claim_name, namespace, sandboxd)

    async def get_sandbox(self, claim_name: str, namespace: str = 'default') -> FakeSandbox:
        from k8s_agent_sandbox.exceptions import SandboxNotFoundError

        sandboxd = self.cluster.claims.get((namespace, claim_name))
        if sandboxd is None:
            raise SandboxNotFoundError(f"SandboxClaim '{claim_name}' not found in namespace '{namespace}'")
        return self._handle(claim_name, namespace, sandboxd)

    def _handle(self, claim_name: str, namespace: str, sandboxd: FakeSandboxd) -> FakeSandbox:
        sandbox = FakeSandbox(claim_name, namespace, sandboxd, self.cluster)
        self.sandboxes.append(sandbox)
        return sandbox

    async def close(self) -> None:
        self.closed = True
        for sandbox in self.sandboxes:
            if sandbox.connector.channel is not None:
                await sandbox.connector.channel.close()
