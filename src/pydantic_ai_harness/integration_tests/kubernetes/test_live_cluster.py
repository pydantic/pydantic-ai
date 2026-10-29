"""`KubernetesSandbox` against a real cluster: the agent-sandbox controller, its SDK, and `kubectl`.

`test_live_sandboxd.py` checks sandboxd itself, with the cluster stood in for. This file runs the
conformance rules end to end on a kind cluster, and checks what only a cluster can answer: claims
from a warm pool and their expiry, the SDK deleting a claim whose creation was cancelled, a
`kubectl port-forward` that dies, a pod the controller recreates, connection configs, and the
examples on the docs page.

Run with `make integration-kubernetes-cluster`, which needs Docker, kind and kubectl: it creates the
`pydantic-ai-live` cluster with `cluster/up.sh`, writes its credentials to its own kubeconfig, and
runs these tests with `KUBECONFIG` pointing there. They create and delete claims, so they refuse any
cluster whose context is not `kind-pydantic-ai-live`. Without it they skip, unless
`KUBERNETES_CLUSTER_REQUIRE_LIVE` is set (CI does), where they fail instead.

The in-cluster connections need the pod network, which only Linux can route to kind's pods: CI adds
the route and sets `KUBERNETES_POD_NETWORK=1`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import datetime
from typing import Any

import anyio
import grpc
import grpc.aio
import pytest
from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig
from pytest_examples import CodeExample

from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.kubernetes_sandbox import KubernetesSandbox, KubernetesSandboxBackend
from tests.harness._docs_examples import documented_cleanup, python_blocks, run_block
from tests.harness._tool_calls import call_tools

CONTEXT = 'kind-pydantic-ai-live'
NAMESPACE = 'default'
POOL = 'python-pool'
"""The docs page's pool, so its examples run as written."""
LIVE_SANDBOX_TIMEOUT = 900
"""Bounds every claim made here, in case a killed job skips the cleanup."""


def _current_context() -> str | None:
    if not os.environ.get('KUBECONFIG') or shutil.which('kubectl') is None:
        return None
    result = subprocess.run(['kubectl', 'config', 'current-context'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


ON_TEST_CLUSTER = _current_context() == CONTEXT


@pytest.fixture
def anyio_backend() -> str:
    """Run every test on asyncio: the SDK and `grpc.aio` run on nothing else."""
    return 'asyncio'


def _skip_or_fail(reason: str) -> None:
    if os.getenv('KUBERNETES_CLUSTER_REQUIRE_LIVE'):
        pytest.fail(f'KUBERNETES_CLUSTER_REQUIRE_LIVE is set, but the tests cannot run: they {reason}.')
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _require_cluster() -> Iterator[None]:
    if not ON_TEST_CLUSTER:
        _skip_or_fail(f'need the kind cluster from `cluster/up.sh` as the current context of KUBECONFIG ({CONTEXT})')
    before = _claims_now()
    yield
    # A test that failed part way leaves its claims behind; they would hold warm sandboxes from later tests.
    if left := _claims_now() - before:
        subprocess.run(['kubectl', '-n', NAMESPACE, 'delete', 'sandboxclaim', '--wait=false', *left], check=True)


@pytest.fixture
def pod_network() -> None:
    """Skip a test that connects to pod IPs unless this host can route to them."""
    if os.getenv('KUBERNETES_POD_NETWORK') != '1':
        _skip_or_fail('need the pod network routed to this host (KUBERNETES_POD_NETWORK=1)')


def _claims_now() -> set[str]:
    command = ['kubectl', '-n', NAMESPACE, 'get', 'sandboxclaims', '-o', 'jsonpath={.items[*].metadata.name}']
    return set(subprocess.run(command, capture_output=True, text=True, check=True).stdout.split())


async def _kubectl(*args: str) -> str:
    return (await anyio.run_process(['kubectl', '-n', NAMESPACE, *args])).stdout.decode()


async def _claim_names() -> set[str]:
    return await anyio.to_thread.run_sync(_claims_now)


async def _gone(kind: str, name: str) -> bool:
    return not await _kubectl('get', kind, name, '--ignore-not-found', '-o', 'name')


async def _eventually(check: Callable[[], Awaitable[bool]], *, timeout: float = 120) -> None:
    """Wait until `check` passes; `timeout` only bounds a wedged cluster."""
    with anyio.fail_after(timeout):
        while not await check():
            await anyio.sleep(0.5)


def _claim_of(backend: KubernetesSandboxBackend) -> str:
    assert backend.ref is not None
    namespace, _, claim_name = backend.ref.id.partition('/')
    assert namespace == NAMESPACE
    return claim_name


def _new_backend(**settings: Any) -> KubernetesSandboxBackend:
    defaults: dict[str, Any] = {'warm_pool': POOL, 'namespace': NAMESPACE, 'sandbox_timeout': LIVE_SANDBOX_TIMEOUT}
    return KubernetesSandboxBackend(**(defaults | settings))


async def _destroy(backend: KubernetesSandboxBackend) -> None:
    await backend.aclose()
    if backend.ref is not None:
        await KubernetesSandbox(warm_pool=POOL).destroy(backend.ref)


class TestClusterConformance(WorkspaceBackendSuite):
    """Every conformance rule, through the SDK, `kubectl port-forward` and the controller."""

    @pytest.fixture
    async def created(self) -> AsyncIterator[list[KubernetesSandboxBackend]]:
        backends: list[KubernetesSandboxBackend] = []
        yield backends
        for backend in backends:
            await _destroy(backend)

    @pytest.fixture
    def fresh_backend(self, created: list[KubernetesSandboxBackend]) -> Callable[[], WorkspaceBackend]:
        def make() -> WorkspaceBackend:
            backend = _new_backend()
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
        async def destroy_ref(backend: WorkspaceBackend) -> None:
            assert backend.ref is not None
            await KubernetesSandbox(warm_pool=POOL).destroy(backend.ref)

        return destroy_ref


async def test_a_claim_names_its_pool_and_the_sandbox_timeout() -> None:
    """A claim names the warm pool, and the controller deletes it after `sandbox_timeout`, or never with `None`."""
    before = time.time()
    backend = _new_backend(sandbox_timeout=600)
    await backend.run(['true'])
    claim = json.loads(await _kubectl('get', 'sandboxclaim', _claim_of(backend), '-o', 'json'))
    assert claim['spec']['warmPoolRef'] == {'name': POOL}
    assert claim['spec']['lifecycle']['shutdownPolicy'] == 'Delete'
    shutdown = datetime.fromisoformat(claim['spec']['lifecycle']['shutdownTime'].replace('Z', '+00:00')).timestamp()
    # The API server keeps whole seconds.
    assert before + 600 - 1 <= shutdown <= time.time() + 600 + 1
    await _destroy(backend)

    backend = _new_backend(sandbox_timeout=None)
    await backend.run(['true'])
    claim = json.loads(await _kubectl('get', 'sandboxclaim', _claim_of(backend), '-o', 'json'))
    assert 'lifecycle' not in claim['spec']
    await _destroy(backend)


async def test_an_expired_sandbox_is_unavailable() -> None:
    """Once the controller deletes a claim past its `sandbox_timeout`, its owner and a reattach both find it gone."""
    owner = _new_backend(sandbox_timeout=5)
    pod = (await owner.run(['hostname'])).stdout.strip()
    claim_name = _claim_of(owner)
    await _eventually(lambda: _gone('sandboxclaim', claim_name))
    # sandboxd serves until its pod is gone, so a command just after the claim goes still runs.
    await _eventually(lambda: _gone('pod', pod))
    with pytest.raises(WorkspaceUnavailableError):
        await owner.run(['true'])
    await owner.aclose()
    attached = KubernetesSandboxBackend(ref=owner.ref)
    with pytest.raises(WorkspaceUnavailableError):
        await attached.run(['true'])
    await attached.aclose()


async def test_destroy_deletes_the_claim_and_its_pod() -> None:
    """`destroy(ref)` leaves no claim or pod behind, and a second call returns quietly."""
    backend = _new_backend()
    pod = (await backend.run(['hostname'])).stdout.strip()
    claim_name = _claim_of(backend)
    await _destroy(backend)
    await _eventually(lambda: _gone('sandboxclaim', claim_name))
    await _eventually(lambda: _gone('pod', pod))
    assert backend.ref is not None
    await KubernetesSandbox(warm_pool=POOL).destroy(backend.ref)


async def test_a_claim_cancelled_before_it_is_ready_is_deleted() -> None:
    """Cancelling the first operation while its claim is pending leaves no claim behind."""
    before = await _claim_names()
    # The `never-ready` pool's sandboxes never pass their readiness probe, so the claim stays pending.
    backend = _new_backend(warm_pool='never-ready')
    async with anyio.create_task_group() as group:
        group.start_soon(backend.run, ['true'])

        async def claimed() -> bool:
            return bool(await _claim_names() - before)

        await _eventually(claimed)
        group.cancel_scope.cancel()

    async def deleted() -> bool:
        return not await _claim_names() - before

    await _eventually(deleted)
    await backend.aclose()


async def test_an_unknown_warm_pool_is_unavailable_and_leaves_no_claim() -> None:
    """A warm pool that does not exist is reported by name, and nothing is claimed."""
    before = await _claim_names()
    backend = _new_backend(warm_pool='no-such-pool')
    with pytest.raises(WorkspaceUnavailableError, match="warm pool 'no-such-pool'"):
        await backend.run(['true'])
    await backend.aclose()
    assert await _claim_names() <= before


async def test_a_dead_port_forward_is_restarted() -> None:
    """A `kubectl port-forward` that dies is started again for a later command."""
    backend = _new_backend()
    await backend.run(['true'])
    # The SDK runs `kubectl port-forward` as a child of this process.
    children = subprocess.run(['pgrep', '-P', str(os.getpid()), 'kubectl'], capture_output=True, text=True)
    port_forwards = [int(pid) for pid in children.stdout.split()]
    assert port_forwards
    for pid in port_forwards:
        os.kill(pid, signal.SIGKILL)
    # Until asyncio reaps it, the SDK cannot tell the port-forward died, so a command sent at once may
    # fail with a retryable `UNAVAILABLE`, which drops the connection; the next one opens a new one.
    try:
        await backend.run(['true'])
    except grpc.aio.AioRpcError as error:
        assert error.code() == grpc.StatusCode.UNAVAILABLE
    assert (await backend.run(['echo', 'back'])).stdout == 'back\n'
    await _destroy(backend)


async def _recreate_pod(backend: KubernetesSandboxBackend) -> None:
    """Delete the sandbox's pod, as a failed node would, and wait until a command reaches the new one."""
    pod = (await backend.run(['hostname'])).stdout.strip()
    await _kubectl('delete', 'pod', pod, '--wait=true')

    async def back() -> bool:
        try:
            return (await backend.run(['true'])).exit_code == 0
        except WorkspaceUnavailableError:
            raise
        except Exception:
            # A transient error while the controller recreates the pod.
            return False

    await _eventually(back)


async def test_a_recreated_pod_is_not_gone() -> None:
    """A pod the controller recreates for the same claim is reached again, with the same ref."""
    backend = _new_backend()
    await backend.run(['true'])
    ref = backend.ref
    await _recreate_pod(backend)
    assert backend.ref == ref
    await _destroy(backend)


async def test_a_connection_config_reaches_sandboxd_on_other_ports() -> None:
    """An SDK connection config reaches sandboxd on the ports it names."""
    connection = SandboxdPodTunnelConnectionConfig(grpc_port=9191, rest_port=8181)
    backend = _new_backend(warm_pool='sandboxd-other-ports', connection=connection)
    assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'
    await _destroy(backend)


@pytest.mark.usefixtures('pod_network')
@pytest.mark.parametrize(
    ('pool', 'connection'),
    [
        (POOL, 'in-cluster-pod-ip'),
        (
            'sandboxd-other-ports',
            SandboxdInClusterConnectionConfig(mode='in-cluster-pod-ip', grpc_port=9191, rest_port=8181),
        ),
    ],
    ids=['mode', 'config'],
)
async def test_an_in_cluster_connection_follows_a_recreated_pod(pool: str, connection: Any) -> None:
    """A recreated pod has a new IP, which the next command resolves again."""
    backend = _new_backend(warm_pool=pool, connection=connection)
    assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'
    await _recreate_pod(backend)
    await _destroy(backend)


async def test_coder_tools_run_in_the_sandbox_kubernetes_sandbox_supplies() -> None:
    """`KubernetesSandbox` supplies the workspace, and `Coder`'s tools run in it."""
    supplied: list[KubernetesSandboxBackend] = []

    class RecordingKubernetesSandbox(KubernetesSandbox[None]):
        def get_workspace(self, ctx: RunContext[None], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            backend = super().get_workspace(ctx, ref=ref)
            assert isinstance(backend, KubernetesSandboxBackend)
            supplied.append(backend)
            return backend

    try:
        results = await call_tools(
            [
                RecordingKubernetesSandbox(
                    warm_pool=POOL, namespace=NAMESPACE, sandbox_timeout=LIVE_SANDBOX_TIMEOUT, working_dir='/tmp'
                ),
                Coder(),
            ],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hello from kubernetes')\n"}),
                ('shell', {'command': 'python hello.py && hostname'}),
                ('grep', {'pattern': 'hello from kubernetes'}),
            ],
        )
        assert 'hello from kubernetes' in results[1]
        assert 'hello.py' in results[2]
        # Core may ask for the workspace again only to compare it; exactly one backend claimed a sandbox.
        assert sum(backend.ref is not None for backend in supplied) == 1
    finally:
        for backend in supplied:
            await _destroy(backend)


_DOCS_BLOCKS = python_blocks('docs/harness/kubernetes-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real sandbox.

    A follow-up run, from the message history or a stored ref, works in the first run's sandbox.
    """
    # The documented default keeps a sandbox for 24 hours, and a killed job skips the cleanup below.
    init = KubernetesSandboxBackend.__init__

    def bounded_init(self: KubernetesSandboxBackend, **settings: Any) -> None:
        bounded: dict[str, Any] = {**settings, 'sandbox_timeout': LIVE_SANDBOX_TIMEOUT}
        init(self, **bounded)

    monkeypatch.setattr(KubernetesSandboxBackend, '__init__', bounded_init)
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_sandbox'))
    assert all(run.used_sandbox for run in runs), runs
    assert len({run.ref for run in runs}) <= 1, runs
