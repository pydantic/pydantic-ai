"""An in-process sandboxd and agent-sandbox SDK fake for `KubernetesSandboxBackend` tests.

`FakeSandboxd` serves sandboxd's `process.v1.ProcessService` `Start` over a real `grpc.aio` server
and runs each command as a host process in its root directory, with sandboxd's semantics: the
command gets its own process group, which is SIGKILLed when the call is cancelled; stdin is an
open pipe; a signal death exits 128+n. Messages are built with protobuf from a descriptor matching
`process.proto`, independently of the SDK's generated stubs, so the two check each other.

`FakeCluster` stands in for the SDK's `AsyncSandboxClient`: each claim gets its own `FakeSandboxd`
and root directory. Deleting a claim stops its server as deleting the pod stops sandboxd: commands
get SIGTERM, so a call in flight reports exit 143, and the server goes after them.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import grpc
import grpc.aio
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from k8s_agent_sandbox.commands.async_command_executor import AsyncCommandExecutor

_GRACE = 0.3
"""How long output pipes may stay open after the command exits (sandboxd waits about 5 seconds)."""

_MAX_ARG_STRLEN = 32 * 4096
"""Linux's limit on one `execve` argument or environment string, with its NUL; macOS has none."""


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
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)  # pyright: ignore[reportUnknownMemberType]
    names = ('StartRequest', 'StartResponse')
    return {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f'fake.process.v1.{name}'))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        for name in names
    }


MESSAGES = _messages()


class FakeSandboxd:
    """A sandboxd `ProcessService` serving host processes in `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.server: grpc.aio.Server | None = None
        self.target = ''
        self._processes: dict[int, _Process] = {}
        self._idle = asyncio.Event()
        """Set while no command runs."""
        self._idle.set()
        self._next_id = 0
        self.send_exit = True
        """When false, `Start` ends its stream without the exit event."""
        self.negative_signal_exits = False
        """When set, a signal death exits -1, as sandboxd before v1.0.7 reports it."""
        self.sandbox_id = f'sandbox-{uuid.uuid4().hex[:8]}'
        """The name of the Sandbox object the claim binds."""

    async def start(self) -> None:
        self.server = grpc.aio.server()
        handlers: dict[str, grpc.RpcMethodHandler[Any, Any]] = {
            'Start': grpc.unary_stream_rpc_method_handler(
                self._start, MESSAGES['StartRequest'].FromString, lambda message: message.SerializeToString()
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
            process.kill_group(signal.SIGTERM)
        # Each call reports its command's exit before the server goes. Stopping the server cancels a call
        # whose command ignored SIGTERM, which kills it, as the kubelet does after the grace period.
        with anyio.move_on_after(5):
            await self._idle.wait()
        assert self.server is not None
        await self.server.stop(grace=0)

    async def _start(self, request: Any, context: grpc.aio.ServicerContext[Any, Any]) -> AsyncIterator[Any]:
        Response = MESSAGES['StartResponse']
        command = list(request.config.command)
        call_env = dict(request.config.env_vars)
        strings = [*command, *(f'{key}={value}' for key, value in call_env.items())]
        if any(len(string.encode()) >= _MAX_ARG_STRLEN for string in strings):
            # As on Linux, whatever the host: `execve` fails with E2BIG, which sandboxd reports as INTERNAL.
            await context.abort(grpc.StatusCode.INTERNAL, 'start process: fork/exec /bin/sh: argument list too long')
        process = _Process.spawn(command, cwd=self.root, env={**os.environ, **call_env})
        self._next_id += 1
        process_id = self._next_id
        self._processes[process_id] = process
        self._idle.clear()
        # Reaps the child however the call ends.
        waiter = asyncio.ensure_future(process.wait())
        tasks: list[asyncio.Future[Any]] = []
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def pump(stream: asyncio.StreamReader, name: str) -> None:
            try:
                while chunk := await stream.read(4096):
                    await queue.put(Response(**{name: chunk}))
            finally:
                queue.put_nowait(None)  # End of this stream, also when cut off below.

        async def cut_off_after_exit(pumps: list[asyncio.Task[None]]) -> None:
            # Like sandboxd's wait delay: stop reading pipes a background child keeps open.
            await waiter
            _, pending = await asyncio.wait(pumps, timeout=_GRACE)
            for task in pending:
                task.cancel()

        try:
            # Inside the `try`, so a call cancelled while the pipes connect still kills the command.
            await process.connect_pipes()
            pumps = [
                asyncio.ensure_future(pump(process.stdout, 'stdout')),
                asyncio.ensure_future(pump(process.stderr, 'stderr')),
            ]
            tasks = [*pumps, asyncio.ensure_future(cut_off_after_exit(pumps))]
            yield Response(init={'process_id': process_id})
            ended = 0
            while ended < 2:
                event = await queue.get()
                if event is None:
                    ended += 1
                else:
                    yield event
            code = await waiter
            if self.send_exit:
                yield Response(
                    exit={'exit_code': code if code >= 0 else -1 if self.negative_signal_exits else 128 - code}
                )
        finally:
            # Reached on cancellation too, which is how sandboxd stops a command whose call ends.
            if not waiter.done():
                process.kill_group()
            for task in tasks:
                task.cancel()
            process.close()
            self._processes.pop(process_id, None)
            if not self._processes:
                self._idle.set()


class _Process:
    """A command started with `posix_spawn` in its own process group, as sandboxd starts it, so the group can be killed.

    Not `asyncio.create_subprocess_exec`: that forks, and gRPC's fork handlers (this process runs a
    gRPC server) write log lines into the child's stderr before it execs.
    """

    def __init__(self, pid: int, stdin: int, stdout: int, stderr: int) -> None:
        self.pid = pid
        self._stdin = stdin
        self._pipes = [stdout, stderr]
        """Output pipes not yet handed to the event loop."""
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self._transports: list[asyncio.BaseTransport] = []

    @classmethod
    def spawn(cls, argv: list[str], *, cwd: Path, env: dict[str, str]) -> _Process:
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        # `posix_spawn` has no working directory argument before Python 3.13, so a shell enters it.
        pid = os.posix_spawn(
            '/bin/sh',
            ['sh', '-c', 'cd "$0" && exec "$@"', str(cwd), *argv],
            env,
            file_actions=[
                (os.POSIX_SPAWN_DUP2, stdin_r, 0),
                (os.POSIX_SPAWN_DUP2, stdout_w, 1),
                (os.POSIX_SPAWN_DUP2, stderr_w, 2),
            ],
            # `setpgroup`, not `setsid`: some Linux Python builds lack `posix_spawn` session support.
            setpgroup=0,
        )
        for fd in (stdin_r, stdout_w, stderr_w):
            os.close(fd)
        return cls(pid, stdin_w, stdout_r, stderr_r)

    async def connect_pipes(self) -> None:
        loop = asyncio.get_running_loop()
        for reader in (self.stdout, self.stderr):
            pipe = os.fdopen(self._pipes.pop(0), 'rb', 0)
            transport, _ = await loop.connect_read_pipe(
                lambda reader=reader: asyncio.StreamReaderProtocol(reader), pipe
            )
            self._transports.append(transport)

    async def wait(self) -> int:
        _, status = await asyncio.to_thread(os.waitpid, self.pid, 0)
        return os.waitstatus_to_exitcode(status)

    def kill_group(self, signum: int = signal.SIGKILL) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signum)

    def close(self) -> None:
        for fd in (self._stdin, *self._pipes):
            os.close(fd)
        for transport in self._transports:
            transport.close()


@dataclass
class FakeSandbox:
    """The SDK's `AsyncSandbox`, as much of it as the backend uses.

    `commands` is the SDK's own executor, on a connector to the fake sandboxd, so the backend runs
    commands through the SDK's real `commands.start()`.
    """

    claim_name: str | None
    namespace: str
    sandboxd: FakeSandboxd
    cluster: FakeCluster
    connector: FakeConnector = field(init=False)
    commands: AsyncCommandExecutor = field(init=False)
    sandbox_id: str = field(init=False)
    k8s_helper: FakeK8sHelper = field(init=False)

    def __post_init__(self) -> None:
        self.sandbox_id = self.sandboxd.sandbox_id
        self.k8s_helper = FakeK8sHelper(self.cluster)
        self.connector = FakeConnector(self)
        # The executor only calls the connector methods `FakeConnector` has.
        connector: Any = self.connector
        self.commands = AsyncCommandExecutor(connector, tracer=None, trace_service_name='sandbox-client')


class FakeConnector:
    """The SDK's connector, as much of it as `commands.start()` calls: one gRPC channel to the fake sandboxd."""

    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.channel: grpc.aio.Channel | None = None
        self.channels: list[grpc.aio.Channel] = []

    def is_sandboxd(self) -> bool:
        return True

    async def connect(self) -> None:
        pass

    async def grpc_channel(self) -> grpc.aio.Channel:
        if self.sandbox.cluster.channel_error is not None:
            raise self.sandbox.cluster.channel_error
        if self.channel is None:
            self.channel = grpc.aio.insecure_channel(self.sandbox.sandboxd.target)
            self.channels.append(self.channel)
        return self.channel

    async def invalidate_sandboxd_transport(self, channel: grpc.aio.Channel) -> None:
        pass


class FakeCluster:
    """Claims, each with its own `FakeSandboxd` in a directory under `base`."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.claims: dict[tuple[str, str], FakeSandboxd] = {}
        self.pools = {'pool'}
        self.clients: list[FakeClient] = []
        self.created: list[tuple[str, str, dict[str, Any]]] = []
        self.sandbox_lookup_error: Exception | None = None
        """When set, reading a Sandbox object fails, as when the API server is unreachable."""
        self.sandbox_lookup_gate: anyio.Event | None = None
        """When set, reading a Sandbox object waits for the gate, as an API server that does not answer."""
        self.channel_error: Exception | None = None
        self.create_error: Exception | None = None
        self.create_gate: anyio.Event | None = None
        """When set, `create_sandbox` creates the claim, sets `create_started`, then waits for the gate."""
        self.create_started = anyio.Event()
        self.lookup_error = False
        """When set, `get_sandbox` fails although the claim exists, as the SDK reports any failed lookup."""
        self.deleted_sandboxes: set[str] = set()
        """Sandbox objects deleted while the claim that binds them remains."""
        self.delete_error: Exception | None = None
        self.delete_gate: anyio.Event | None = None
        """When set, deleting a claim waits for the gate, as an API server that does not answer."""

    def root(self, namespace: str, claim_name: str) -> Path:
        return (self.base / namespace / claim_name).resolve()

    async def delete(self, key: tuple[str, str]) -> None:
        await self.claims.pop(key).stop()

    async def close(self) -> None:
        for key in list(self.claims):
            await self.delete(key)

    def client(self, *, connection_config: Any, cleanup: bool) -> FakeClient:
        """Stands in for the `AsyncSandboxClient` class: the backend constructs clients through this."""
        client = FakeClient(self, connection_config, cleanup)
        self.clients.append(client)
        return client


class FakeClient:
    """The SDK's `AsyncSandboxClient`."""

    def __init__(self, cluster: FakeCluster, connection_config: Any, cleanup: bool) -> None:
        self.cluster = cluster
        self.connection_config = connection_config
        self.cleanup = cleanup
        self.closed = False
        self.sandboxes: list[FakeSandbox] = []
        self.k8s_helper = FakeK8sHelper(cluster)

    async def create_sandbox(
        self, warmpool: str, namespace: str = 'default', *, claim_name: str, **kwargs: Any
    ) -> FakeSandbox:
        """Like the SDK given a `claim_name`: a claim that fails or is cancelled is left for the caller to delete."""
        from k8s_agent_sandbox.exceptions import SandboxWarmPoolNotFoundError

        if self.cluster.create_error is not None:
            raise self.cluster.create_error
        if warmpool not in self.cluster.pools:
            raise SandboxWarmPoolNotFoundError(f"SandboxWarmPool '{warmpool}' not found in namespace '{namespace}'")
        root = self.cluster.root(namespace, claim_name)
        root.mkdir(parents=True)
        sandboxd = FakeSandboxd(root)
        await sandboxd.start()
        self.cluster.claims[namespace, claim_name] = sandboxd
        self.cluster.created.append((warmpool, namespace, kwargs))
        if self.cluster.create_gate is not None:
            self.cluster.create_started.set()
            await self.cluster.create_gate.wait()
        return self._handle(claim_name, namespace, sandboxd)

    async def get_sandbox(self, claim_name: str, namespace: str = 'default') -> FakeSandbox:
        from k8s_agent_sandbox.exceptions import SandboxNotFoundError

        sandboxd = self.cluster.claims.get((namespace, claim_name))
        if sandboxd is None or self.cluster.lookup_error:
            raise SandboxNotFoundError(f"SandboxClaim '{claim_name}' not found in namespace '{namespace}'")
        if sandboxd.sandbox_id in self.cluster.deleted_sandboxes:
            raise SandboxNotFoundError(f"Underlying Sandbox '{sandboxd.sandbox_id}' not found.")
        return self._handle(claim_name, namespace, sandboxd)

    def _handle(self, claim_name: str, namespace: str, sandboxd: FakeSandboxd) -> FakeSandbox:
        sandbox = FakeSandbox(claim_name, namespace, sandboxd, self.cluster)
        self.sandboxes.append(sandbox)
        return sandbox

    async def close(self) -> None:
        self.closed = True
        for sandbox in self.sandboxes:
            for channel in sandbox.connector.channels:
                await channel.close()


class FakeK8sHelper:
    """The SDK's `AsyncK8sHelper`: claims read and deleted through the API server."""

    def __init__(self, cluster: FakeCluster) -> None:
        self.cluster = cluster

    async def get_sandbox(self, name: str, namespace: str) -> dict[str, Any] | None:
        if self.cluster.sandbox_lookup_gate is not None:
            await self.cluster.sandbox_lookup_gate.wait()
        if self.cluster.sandbox_lookup_error is not None:
            raise self.cluster.sandbox_lookup_error
        if name in self.cluster.deleted_sandboxes:
            return None
        bound = (sandboxd for (ns, _), sandboxd in self.cluster.claims.items() if ns == namespace)
        return {'metadata': {'name': name}} if any(sandboxd.sandbox_id == name for sandboxd in bound) else None

    async def get_sandbox_claim(self, name: str, namespace: str) -> dict[str, Any] | None:
        return {'metadata': {'name': name}} if (namespace, name) in self.cluster.claims else None

    async def resolve_sandbox_name(self, claim_name: str, namespace: str, timeout: int) -> str:
        """The name of the Sandbox a claim binds, from the claim's status."""
        return self.cluster.claims[namespace, claim_name].sandbox_id

    async def delete_sandbox_claim(self, name: str, namespace: str) -> None:
        if self.cluster.delete_gate is not None:
            await self.cluster.delete_gate.wait()
        if self.cluster.delete_error is not None:
            raise self.cluster.delete_error
        if (namespace, name) in self.cluster.claims:  # A 404 is ignored.
            await self.cluster.delete((namespace, name))
