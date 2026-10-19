"""`KubernetesSandbox` and the backend behaviour the conformance suite does not reach."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any, cast

import anyio
import grpc
import pytest
from k8s_agent_sandbox.exceptions import SandboxClaimFailedError
from k8s_agent_sandbox.models import SandboxdInClusterConnectionConfig, SandboxdPodTunnelConnectionConfig

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness import KubernetesSandbox, KubernetesSandboxBackend, Shell
from pydantic_ai_harness.kubernetes_sandbox import _backend
from pydantic_ai_harness.kubernetes_sandbox._capability import _SuppliedBackend  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.kubernetes_sandbox._process import _close_stdin, start_command  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.kubernetes_sandbox._wire import StartEvent

from .fake_sandboxd import FakeCluster

asyncio_only = pytest.mark.parametrize('anyio_backend', ['asyncio'])


def test_refs_name_a_namespace_and_a_claim() -> None:
    assert _backend.parse_ref(WorkspaceRef(provider='kubernetes', id='agents/claim')) == ('agents', 'claim')
    for bad in ('claim', '/claim', 'agents/', 'a/b/c'):
        with pytest.raises(ValueError, match='<namespace>/<claim name>'):
            _backend.parse_ref(WorkspaceRef(provider='kubernetes', id=bad))
    with pytest.raises(ValueError, match="expected 'kubernetes'"):
        KubernetesSandboxBackend(ref=WorkspaceRef(provider='modal', id='sb-1'))


@pytest.mark.parametrize('module', ['k8s_agent_sandbox', 'grpc'])
def test_missing_kubernetes_extra_has_an_install_hint(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.kubernetes_sandbox')]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[kubernetes\]` to use KubernetesSandbox\.$'):
        importlib.import_module('pydantic_ai_harness.kubernetes_sandbox')


def test_backend_arguments_are_checked_without_io() -> None:
    with pytest.raises(ValueError, match='pass `warm_pool`'):
        KubernetesSandboxBackend()
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        KubernetesSandboxBackend(warm_pool='pool', env={'A': 1})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match='working_dir must be an absolute'):
        KubernetesSandboxBackend(warm_pool='pool', working_dir='relative')


@asyncio_only
async def test_client_matches_the_connection() -> None:
    # Building the SDK client does no I/O: kubeconfig loads on its first request.
    tunnel = _backend._client('port-forward')  # pyright: ignore[reportPrivateUsage]
    in_cluster = _backend._client('in-cluster-pod-ip')  # pyright: ignore[reportPrivateUsage]
    assert isinstance(tunnel.connection_config, SandboxdPodTunnelConnectionConfig)
    assert isinstance(in_cluster.connection_config, SandboxdInClusterConnectionConfig)
    assert in_cluster.connection_config.mode == 'in-cluster-pod-ip'
    await tunnel.close()
    await in_cluster.close()


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused() -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(UserError, match='needs asyncio'):
        await backend.run(['true'])
    with pytest.raises(UserError, match='needs asyncio'):
        await backend.get_sandbox()


@asyncio_only
async def test_a_claim_uses_the_pool_namespace_and_timeout(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents', sandbox_timeout=60, env={'A': 'base'})
    result = await backend.run('printf "$A $B"', shell=True, env={'B': 'call'})
    assert result.stdout == 'base call'
    assert backend.ref is not None and backend.ref.id.startswith('agents/pydantic-ai-')
    [(pool, namespace, kwargs)] = cluster.created
    assert (pool, namespace, kwargs['shutdown_after_seconds']) == ('pool', 'agents', 60)
    sandbox = await backend.get_sandbox()
    assert sandbox.claim_name == backend.ref.id.split('/')[1]
    await backend.aclose()
    await backend.aclose()  # Nothing left to close.


@asyncio_only
async def test_working_dir_is_entered_from_a_shell(cluster: FakeCluster) -> None:
    probe = KubernetesSandboxBackend(warm_pool='pool')
    root = await probe.working_dir()
    sub = f'{root}/sub'
    assert (await probe.run(['mkdir', sub])).exit_code == 0
    backend = KubernetesSandboxBackend(ref=probe.ref, working_dir=sub)
    assert await backend.working_dir() == sub
    assert (await backend.run(['pwd', '-P'])).stdout == f'{sub}\n'
    assert (await backend.run(['missing-program'])).exit_code == 127
    gone = KubernetesSandboxBackend(ref=probe.ref, working_dir=f'{root}/gone')
    assert (await gone.run(['true'])).exit_code == 1
    for each in (probe, backend, gone):
        await each.aclose()


@asyncio_only
async def test_an_unreadable_working_dir_is_unavailable(cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    monkeypatch.setenv('PATH', '/nonexistent')  # `pwd` cannot be found, as in an image without coreutils.
    cluster.claims[_backend.parse_ref(backend.ref)].root.rmdir()  # type: ignore[arg-type]
    with pytest.raises(WorkspaceUnavailableError, match='working directory'):
        await backend.working_dir()
    await backend.aclose()


@asyncio_only
async def test_an_unknown_warm_pool_or_failed_claim_is_unavailable(cluster: FakeCluster) -> None:
    with pytest.raises(WorkspaceUnavailableError, match="warm pool 'missing'"):
        await KubernetesSandboxBackend(warm_pool='missing').run(['true'])
    cluster.create_error = SandboxClaimFailedError('SandboxClaim failed: PodFailed')
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(WorkspaceUnavailableError, match='PodFailed'):
        await backend.run(['true'])
    assert backend.ref is None
    # The next operation claims again, with the same client.
    cluster.create_error = None
    assert (await backend.run(['true'])).exit_code == 0
    assert backend.ref is not None and len(cluster.clients) == 2
    await backend.aclose()


@asyncio_only
async def test_a_cancelled_claim_keeps_its_ref(cluster: FakeCluster) -> None:
    cluster.create_gate = anyio.Event()  # type: ignore[assignment]
    backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents')
    with anyio.move_on_after(0.5):
        await backend.run(['true'])
    # The claim exists although create never returned: the ref lets the application delete it.
    assert backend.ref is not None and _backend.parse_ref(backend.ref) in cluster.claims


@asyncio_only
async def test_a_lost_connection_to_a_live_sandbox_propagates(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    cluster.channel_error = ConnectionError('port-forward dropped')
    # The sandbox is still Ready, so the error is transient and left for durable engines to retry.
    with pytest.raises(ConnectionError):
        await backend.run(['true'])
    # An API server that cannot be asked leaves the original error too.
    cluster.status_error = RuntimeError('API server unreachable')
    with pytest.raises(ConnectionError):
        await backend.run(['true'])
    await backend.aclose()


@asyncio_only
async def test_stdin_close_races_are_tolerated(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    await backend.run(['true'])
    sandboxd = cluster.claims[_backend.parse_ref(backend.ref)]  # type: ignore[arg-type]
    # A process already unregistered: `NOT_FOUND`.
    write_stdin = (await (await backend.get_sandbox()).connector.grpc_channel()).unary_unary(
        '/process.v1.ProcessService/WriteStdin', request_serializer=bytes, response_deserializer=bytes
    )
    await _close_stdin(write_stdin, 999)
    # A process sandboxd is still unregistering: `INTERNAL: ... file already closed`.
    sandboxd.stdin_error = grpc.StatusCode.INTERNAL
    assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'
    sandboxd.stdin_error = grpc.StatusCode.PERMISSION_DENIED
    with pytest.raises(grpc.aio.AioRpcError):
        await backend.run(['echo', 'ok'])
    await backend.aclose()


class _ScriptedChannel:
    """A `grpc.aio.Channel` whose `Start` call yields fixed events."""

    def __init__(self, events: list[StartEvent]) -> None:
        self.events = events

    def unary_stream(self, *args: Any, **kwargs: Any) -> Callable[[bytes], _Call]:
        return lambda request: _Call(self.events)

    def unary_unary(self, *args: Any, **kwargs: Any) -> None:
        return None


class _Call:
    def __init__(self, events: list[StartEvent]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> _Call:
        return self

    async def __anext__(self) -> StartEvent:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    def cancel(self) -> bool:
        return True


def _channel(events: list[StartEvent]) -> grpc.aio.Channel:
    return cast('grpc.aio.Channel', _ScriptedChannel(events))


@asyncio_only
async def test_a_stream_without_an_exit_event_is_an_error() -> None:
    with pytest.raises(RuntimeError, match='without an exit event'):
        await start_command(_channel([]), ['true'], env={}, timeout=None)


@asyncio_only
async def test_events_a_newer_sandboxd_adds_are_ignored() -> None:
    events = [StartEvent('unknown'), StartEvent('stdout', data=b'ok'), StartEvent('exit', number=0)]
    result = await start_command(_channel(events), ['true'], env={}, timeout=None)
    assert (result.exit_code, result.stdout) == (0, 'ok')


@asyncio_only
async def test_a_timeout_keeps_the_output_so_far(cluster: FakeCluster) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool')
    with pytest.raises(WorkspaceTimeoutError) as info:
        await backend.run('echo before; sleep 30', shell=True, timeout=0.5)
    assert info.value.stdout == 'before\n'
    await backend.aclose()


@asyncio_only
async def test_destroying_a_missing_claim_returns_quietly(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool')
    await capability.destroy(WorkspaceRef(provider='kubernetes', id='agents/gone'))
    backend = KubernetesSandboxBackend(warm_pool='pool', namespace='agents')
    await backend.run(['true'])
    assert backend.ref is not None
    await capability.destroy(backend.ref)
    assert not cluster.claims
    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await capability.backend(backend.ref).run(['true'])


def test_capability_arguments_are_checked() -> None:
    with pytest.raises(UserError, match='defer_loading'):
        KubernetesSandbox(warm_pool='pool', defer_loading=True)
    with pytest.raises(UserError, match='sandbox_timeout'):
        KubernetesSandbox(warm_pool='pool', sandbox_timeout=0)
    with pytest.raises(UserError, match='working_dir'):
        KubernetesSandbox(warm_pool='pool', working_dir='relative')
    assert KubernetesSandbox(warm_pool='pool', sandbox_timeout=None).sandbox_timeout is None


def test_get_workspace_only_takes_kubernetes_refs() -> None:
    capability = KubernetesSandbox(warm_pool='pool', namespace='agents', connection='in-cluster-service')
    ctx: Any = type('Ctx', (), {'run_id': 'run-1'})()
    assert capability.get_workspace(ctx, ref=WorkspaceRef(provider='modal', id='sb')) is None
    backend = capability.get_workspace(ctx, ref=None)
    assert isinstance(backend, _SuppliedBackend) and backend.run_id == 'run-1'
    attached = capability.backend(WorkspaceRef(provider='kubernetes', id='agents/claim'))
    assert type(attached) is KubernetesSandboxBackend and attached.ref is not None


def _shell_model(command: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('run_command', {'command': command})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


@asyncio_only
async def test_a_run_keeps_its_connection_open_and_closes_it_at_the_end(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool', namespace='agents')
    agent = Agent(_shell_model('echo hi > note.txt'), capabilities=[capability, Shell()])
    result = await agent.run('write a note')
    ref = result.workspace.ref
    assert ref is not None and ref.provider == 'kubernetes'
    # One client for the whole run, closed when the run ended; the sandbox keeps running.
    [client] = cluster.clients
    assert client.closed and _backend.parse_ref(ref) in cluster.claims
    # A later run attaches by ref and sees the file.
    agent = Agent(_shell_model('cat note.txt'), capabilities=[capability, Shell()])
    result = await agent.run('read it', workspace=ref)
    assert 'hi' in str(result.all_messages()[2])
    await capability.destroy(ref)


@asyncio_only
async def test_a_backend_outside_a_run_closes_after_each_operation(cluster: FakeCluster) -> None:
    ctx: Any = type('Ctx', (), {'run_id': 'run-1'})()
    backend = KubernetesSandbox(warm_pool='pool').get_workspace(ctx, ref=None)
    assert isinstance(backend, _SuppliedBackend)
    await backend.run(['true'])
    await backend.run(['true'])
    # Nothing would call `aclose()` (Temporal builds one per activity), so each operation closed its own client.
    assert [client.closed for client in cluster.clients] == [True, True]


@asyncio_only
async def test_a_parent_runs_workspace_is_not_closed_by_a_subagent(cluster: FakeCluster) -> None:
    capability = KubernetesSandbox(warm_pool='pool')
    ctx: Any = type('Ctx', (), {'run_id': 'parent'})()
    parent = capability.get_workspace(ctx, ref=None)
    assert isinstance(parent, _SuppliedBackend)
    parent.claim(True)
    await parent.run(['true'])
    agent = Agent(_shell_model('true'), capabilities=[capability, Shell()])
    await agent.run('child', workspace=parent)
    assert not cluster.clients[0].closed
    parent.claim(False)
    await parent.aclose()
