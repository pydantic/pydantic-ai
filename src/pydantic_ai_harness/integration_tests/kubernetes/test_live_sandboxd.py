"""The workspace conformance suite for `KubernetesSandboxBackend`, against real sandboxd.

The unit suite in `tests/harness/kubernetes_sandbox` runs against a Python stand-in for sandboxd.
This file runs the same rules against the real daemon, so what only the daemon can answer is
checked here: how `Start` streams and ends, that cancelling a call kills the command's process
group, how sandboxd shuts down with a command running, and that the backend's hand-written messages
decode as sandboxd's protobuf.

Only the Kubernetes side is stood in for: each claim starts its own `sandboxd` process with its own
root directory, the way each claimed pod runs one, and deleting the claim stops it with SIGTERM, as
the kubelet stops a deleted pod, and removes the directory. Claims, warm pools and
`kubectl port-forward` need a cluster: `test_live_cluster.py` runs against one.

Run with `make integration-kubernetes` after building sandboxd from a release tag:

    GOBIN="$PWD/.bin" go install sigs.k8s.io/agent-sandbox/packages/sandboxd/cmd/sandboxd@v1.0.7
    SANDBOXD_BIN="$PWD/.bin/sandboxd" make integration-kubernetes

Without the binary the tests skip, unless `SANDBOXD_REQUIRE_LIVE` is set (CI does), where a missing
binary fails instead: a job that tested nothing must not pass as a silent skip.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import grpc.aio
import httpx
import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox, KubernetesSandboxBackend

SANDBOXD = os.environ.get('SANDBOXD_BIN') or shutil.which('sandboxd')

# `grpc.aio` and the SDK run on asyncio only.
pytestmark = [pytest.mark.anyio, pytest.mark.parametrize('anyio_backend', ['asyncio'])]


@pytest.fixture(autouse=True)
def _require_sandboxd() -> None:
    if SANDBOXD is not None:
        return
    reason = 'needs a sandboxd binary on PATH or in SANDBOXD_BIN'
    if os.getenv('SANDBOXD_REQUIRE_LIVE'):
        pytest.fail(f'SANDBOXD_REQUIRE_LIVE is set, but the tests cannot run: they {reason}.')
    pytest.skip(reason)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Daemon:
    """One `sandboxd` process, as one claimed pod runs it."""

    def __init__(self, root: Path) -> None:
        assert SANDBOXD is not None
        self.root = root
        root.mkdir(parents=True)
        self.grpc_port, rest_port = _free_port(), _free_port()
        self.process = subprocess.Popen(
            [
                SANDBOXD,
                f'--root-dir={root}',
                '--listen-host=127.0.0.1',
                f'--grpc-port={self.grpc_port}',
                f'--rest-port={rest_port}',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.health = f'http://127.0.0.1:{rest_port}/v1/health'

    async def ready(self) -> None:
        async with httpx.AsyncClient() as client:
            with anyio.fail_after(30):
                while True:
                    try:
                        if (await client.get(self.health)).status_code == 200:
                            return
                    except httpx.TransportError:
                        pass
                    assert self.process.poll() is None, 'sandboxd exited during startup'
                    await anyio.sleep(0.05)

    def stop(self) -> None:
        # Deleting a pod sends its container SIGTERM, and sandboxd passes it on to the commands it runs.
        # Past the grace period the kubelet kills what is left; sandboxd starts each command in its own
        # process group, which killing the daemon leaves running, so those go too.
        children = subprocess.run(['pgrep', '-P', str(self.process.pid)], capture_output=True, text=True).stdout
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for pid in children.split():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(int(pid), signal.SIGKILL)
        shutil.rmtree(self.root, ignore_errors=True)


class Cluster:
    """The agent-sandbox SDK's view of a cluster: claims, each served by its own `Daemon`."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.claims: dict[tuple[str, str], Daemon] = {}

    def client(self, **config: Any) -> Client:
        """Stands in for the `AsyncSandboxClient` class: the backend constructs clients through this."""
        return Client(self)

    def close(self) -> None:
        for daemon in self.claims.values():
            daemon.stop()


class Client:
    """The SDK's `AsyncSandboxClient`: claims start a `Daemon`, attaching finds one."""

    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster
        self.channels: list[grpc.aio.Channel] = []
        self.k8s_helper = self

    async def create_sandbox(self, warmpool: str, namespace: str = 'default', **kwargs: Any) -> Sandbox:
        claim_name = f'sandbox-claim-{uuid.uuid4().hex[:8]}'  # As the SDK names claims.
        daemon = Daemon(self.cluster.base / namespace / claim_name)
        self.cluster.claims[namespace, claim_name] = daemon
        await daemon.ready()
        return Sandbox(self, namespace, claim_name)

    async def get_sandbox(self, claim_name: str, namespace: str = 'default') -> Sandbox:
        from k8s_agent_sandbox.exceptions import SandboxNotFoundError

        if (namespace, claim_name) not in self.cluster.claims:
            raise SandboxNotFoundError(f"SandboxClaim '{claim_name}' not found")
        return Sandbox(self, namespace, claim_name)

    async def get_sandbox_claim(self, name: str, namespace: str) -> dict[str, Any] | None:
        return {} if (namespace, name) in self.cluster.claims else None

    async def delete_sandbox_claim(self, name: str, namespace: str) -> None:
        daemon = self.cluster.claims.pop((namespace, name), None)
        if daemon is not None:
            # Off the event loop: it waits for the daemon and its commands to exit.
            await anyio.to_thread.run_sync(daemon.stop)

    async def close(self) -> None:
        for channel in self.channels:
            await channel.close()


class Sandbox:
    """The SDK's `AsyncSandbox` and its connector, as much as the backend uses."""

    def __init__(self, client: Client, namespace: str, claim_name: str) -> None:
        self.client = client
        self.namespace = namespace
        self.claim_name = claim_name
        self.connector = self

    async def connect(self) -> None:
        pass

    async def invalidate_sandboxd_transport(self, channel: grpc.aio.Channel) -> None:
        pass

    async def grpc_channel(self) -> grpc.aio.Channel:
        daemon = self.client.cluster.claims[self.namespace, self.claim_name]
        channel = grpc.aio.insecure_channel(f'127.0.0.1:{daemon.grpc_port}')
        self.client.channels.append(channel)
        return channel

    async def status(self) -> tuple[str, str]:
        if (self.namespace, self.claim_name) in self.client.cluster.claims:
            return 'SandboxReady', ''
        return 'SandboxNotFound', 'Sandbox object not found in Kubernetes.'


@pytest.fixture
def cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Cluster]:
    """Replace the SDK's client with one whose claims are served by real sandboxd processes."""
    fake = Cluster(tmp_path.resolve())
    monkeypatch.setattr('k8s_agent_sandbox.async_sandbox_client.AsyncSandboxClient', fake.client)
    yield fake
    fake.close()


class TestLiveSandboxdConformance(WorkspaceBackendSuite):
    """Every conformance rule, against claims served by real sandboxd."""

    @pytest.fixture
    async def created(self, cluster: Cluster) -> AsyncIterator[list[KubernetesSandboxBackend]]:
        backends: list[KubernetesSandboxBackend] = []
        yield backends
        for backend in backends:
            await backend.aclose()

    @pytest.fixture
    def fresh_backend(self, created: list[KubernetesSandboxBackend]) -> Callable[[], WorkspaceBackend]:
        def make() -> WorkspaceBackend:
            backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents')
            created.append(backend)
            return backend

        return make

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self, created: list[KubernetesSandboxBackend]) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        def attach(ref: WorkspaceRef) -> WorkspaceBackend:
            backend = KubernetesSandboxBackend(ref=ref)
            created.append(backend)
            return backend

        return attach

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        async def destroy(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await KubernetesSandbox(warm_pool='pool').destroy(backend.ref)

        return destroy
